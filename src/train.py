import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import argparse
from tqdm import tqdm
import wandb
import random
import numpy as np

from .models.seq2seq import Seq2SeqTransformer, from_token_ids
from .data.dataloader import ProgramDataset
from .lang.parser import parse
from .lang.compiler import JITCompiler
from .lang.grammar import DefaultGrammar


class CyclingSampler(Sampler):
    """Samples sequentially, wrapping around at the end of the dataset."""

    def __init__(self, data_source, num_samples: int, shuffle: bool = True):
        self.data_source = data_source
        self.num_samples = num_samples
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source)
        order = torch.randperm(n).tolist() if self.shuffle else list(range(n))
        for i in range(self.num_samples):
            yield order[i % n]

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    """Split each ``(seq, mask)`` into encoder ``src`` and decoder ``tgt_in``/``tgt_out``.

    The dataset emits ``seq = x + y`` where ``x`` is the I/O context (loss mask 0)
    and ``y = <start> + program + <end>`` (loss mask 1 over the program). We feed
    ``x`` into the encoder, ``<start> + program`` into the decoder, and predict
    ``program + <end>``.
    """
    srcs, tgt_ins, tgt_outs = [], [], []
    for seq, mask in batch:
        len_x = mask.count(0)
        src = seq[:len_x]
        tgt = seq[len_x:]
        srcs.append(torch.tensor(src, dtype=torch.long))
        tgt_ins.append(torch.tensor(tgt[:-1], dtype=torch.long))
        tgt_outs.append(torch.tensor(tgt[1:], dtype=torch.long))
    return from_token_ids(srcs), from_token_ids(tgt_ins), from_token_ids(tgt_outs)


def compute_loss(model, batch, criterion, device):
    src, tgt_in, tgt_out = batch
    src = src.to(device)
    tgt_in = tgt_in.to(device)
    tgt_out = tgt_out.to(device)

    logits = model(src, tgt_in)  # jagged (B, j1, n_tokens)
    flat_logits = logits.values()  # (sum_lens, n_tokens)
    flat_targets = tgt_out.values()  # (sum_lens,)

    loss = criterion(flat_logits, flat_targets)
    return loss, int(flat_targets.numel())


def train_epoch(model, dataloader, optimiser, criterion, device, grad_clip=1.0, use_wandb=False, global_step=0):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimiser.zero_grad()
        batch_loss, n_tokens = compute_loss(model, batch, criterion, device)
        batch_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()

        total_loss += batch_loss.item() * n_tokens
        total_tokens += n_tokens

        pbar.set_postfix({'loss': batch_loss.item()})

        if use_wandb:
            wandb.log({'train/step_loss': batch_loss.item()}, step=global_step)
        global_step += 1

    return total_loss / total_tokens, global_step


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            batch_loss, n_tokens = compute_loss(model, batch, criterion, device)
            total_loss += batch_loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix({'val_loss': batch_loss.item()})

    return total_loss / total_tokens


@torch.no_grad()
def greedy_decode(model, src_tokens, start_token, end_token, max_tokens, device):
    """Greedy-decode a single sequence. ``src_tokens`` is a 1-D LongTensor of ids."""
    src = from_token_ids([src_tokens.to(device)])
    memory = model.encode(src)

    out = [start_token]
    for _ in range(max_tokens):
        tgt = from_token_ids([torch.tensor(out, dtype=torch.long, device=device)])
        logits = model.project(model.decode(tgt, memory))  # jagged (1, len(out), n_tokens)
        next_token = int(logits.values()[-1].argmax())
        out.append(next_token)
        if next_token == end_token:
            break
    return out[1:]  # drop <start>


def compute_validation_accuracy(model, val_dataset, device, max_program_tokens=80, max_examples=None):
    """Functional accuracy: generate a program from the I/O context and check it
    reproduces every shown I/O pair (outputs are reduced mod 100 to match the
    dataset's modular-int convention)."""
    model.eval()
    tokeniser = val_dataset.tokeniser
    compiler = JITCompiler(DefaultGrammar)
    start_tok = val_dataset.start
    end_tok = val_dataset.end

    n_correct = 0
    n_total = 0
    # Evaluate each program once at max n_io_shown (= n_io_per_program).
    max_n = val_dataset.max_n_io_shown
    n_programs = len(val_dataset.programs)
    if max_examples is not None:
        n_programs = min(n_programs, max_examples)

    for prog_idx in tqdm(range(n_programs), desc="Accuracy"):
        idx = prog_idx * max_n + (max_n - 1)
        seq, loss_mask, program = val_dataset.__getitem__(idx, include_program=True)
        len_x = loss_mask.count(0)
        src_tokens = torch.tensor(seq[:len_x], dtype=torch.long)
        io_pairs = program['io_pairs']

        gen_tokens = greedy_decode(model, src_tokens, start_tok, end_tok, max_program_tokens, device)
        if end_tok in gen_tokens:
            gen_tokens = gen_tokens[:gen_tokens.index(end_tok)]

        program_str = tokeniser.detokenise(gen_tokens)
        correct = True
        try:
            fn, _ = compiler.compile(parse(program_str))
            for inp, expected in io_pairs:
                output = fn(list(inp))
                if not isinstance(output, list) or [x % 100 for x in output] != expected:
                    correct = False
                    break
        except Exception:
            correct = False

        if correct:
            n_correct += 1
        n_total += 1

    return n_correct / n_total if n_total > 0 else 0.0


def save_checkpoint(model, optimiser, scheduler, epoch, train_loss, val_loss, args, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args),
    }

    torch.save(checkpoint, checkpoint_dir / 'checkpoint_latest.pt')
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

    best_path = checkpoint_dir / 'checkpoint_best.pt'
    if not best_path.exists():
        torch.save(checkpoint, best_path)
    else:
        best_checkpoint = torch.load(best_path)
        if val_loss < best_checkpoint['val_loss']:
            torch.save(checkpoint, best_path)

    print(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(checkpoint_path, model, optimiser=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimiser is not None and 'optimiser_state_dict' in checkpoint:
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint.get('train_loss'), checkpoint.get('val_loss')


def _parse_corpus_arg(s: str) -> list[Path]:
    return [Path(p.strip()) for p in s.split(',') if p.strip()]


def train():
    parser = argparse.ArgumentParser(description='Train a seq2seq transformer over jagged NestedTensors')

    # Data arguments
    parser.add_argument('--train-corpus', type=str, default='datasets/corpus-a/rl_corpus.json',
                        help='Comma-separated corpus JSON file(s) for training')
    parser.add_argument('--val-corpus', type=str, default=None,
                        help='Comma-separated corpus JSON file(s) for validation')
    parser.add_argument('--n-io-per-program', type=int, default=11,
                        help='Number of I/O pairs sampled per program (also = max_n_io_shown)')
    parser.add_argument('--data-seed', type=int, default=0,
                        help='Seed for the I/O sampler (separate from training seed)')

    # Model arguments
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d-ff', type=int, default=None, help='FFN hidden dim (default: 8/3 * d_model)')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of encoder / decoder layers')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Max sequence length for RoPE cache')
    parser.add_argument('--compile-layers', action='store_true', help='torch.compile each encoder/decoder layer')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=1000,
                        help='Steps per epoch (None = full dataset, otherwise steps_per_epoch * batch_size samples)')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--val-examples', type=int, default=None,
                        help='Max validation programs for accuracy (each evaluated once with all n_io_per_program I/O shown). None = all programs.')

    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save-freq', type=int, default=1, help='Save checkpoint every N epochs')

    # Wandb arguments
    parser.add_argument('--wandb-project', type=str, default='mlmp', help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Wandb entity/team name')
    parser.add_argument('--run-name', type=str, default=None, help='Run name for wandb and checkpoint directory')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        if args.run_name is None:
            args.run_name = wandb.run.name
    elif args.run_name is None:
        args.run_name = f"run_seed{args.seed}"

    # Datasets
    train_files = _parse_corpus_arg(args.train_corpus)
    print(f"Loading training corpus from {train_files}")
    train_dataset = ProgramDataset(
        corpus_files=train_files,
        seed=args.data_seed,
        n_io_per_program=args.n_io_per_program,
    )
    print(f"Training dataset: {len(train_dataset.programs):,} programs -> {len(train_dataset):,} items")

    val_dataset = None
    if args.val_corpus:
        val_files = _parse_corpus_arg(args.val_corpus)
        print(f"Loading validation corpus from {val_files}")
        val_dataset = ProgramDataset(
            corpus_files=val_files,
            seed=args.data_seed,
            n_io_per_program=args.n_io_per_program,
        )
        print(f"Validation dataset: {len(val_dataset.programs):,} programs -> {len(val_dataset):,} items")

    n_tokens = len(train_dataset.tokeniser.vocab)
    print(f"Vocabulary size: {n_tokens}")

    if use_wandb:
        wandb.config.update({
            'vocab_size': n_tokens,
            'train_dataset_size': len(train_dataset),
            'val_dataset_size': len(val_dataset) if val_dataset else 0,
        })

    # Dataloaders
    if args.steps_per_epoch is not None:
        num_samples = args.steps_per_epoch * args.batch_size
        train_sampler = CyclingSampler(train_dataset, num_samples=num_samples, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        print(f"Using {args.steps_per_epoch} steps per epoch ({num_samples} samples)")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    # Model
    print("Creating model...")
    model = Seq2SeqTransformer(
        n_tokens=n_tokens,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        compile_layers=args.compile_layers,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    if use_wandb:
        wandb.config.update({'n_params': n_params})

    device = torch.device(args.device)
    model = model.to(device)

    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr / 10)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.run_name:
        checkpoint_dir = checkpoint_dir / args.run_name

    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, train_loss, val_loss = load_checkpoint(args.resume, model, optimiser, scheduler)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    print("\nStarting training...")
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        current_lr = optimiser.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        train_loss, global_step = train_epoch(
            model, train_loader, optimiser, criterion, device, args.grad_clip,
            use_wandb=use_wandb, global_step=global_step,
        )
        print(f"Train loss: {train_loss:.4f}")

        val_loss = train_loss
        val_accuracy = 0.0
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")

            val_accuracy = compute_validation_accuracy(
                model,
                val_dataset,
                device,
                max_program_tokens=80,
                max_examples=args.val_examples,
            )
            print(f"Validation accuracy: {val_accuracy:.2%}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/accuracy': val_accuracy,
                'learning_rate': current_lr,
                'best_val_loss': best_val_loss,
            })

        scheduler.step()

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimiser, scheduler, epoch, train_loss, val_loss, args, checkpoint_dir)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    train()
