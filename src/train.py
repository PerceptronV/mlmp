import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm
import wandb
import random
import numpy as np

from .models.seq2seq import Seq2SeqTransformer, from_token_ids
from .data.dataloader import ProgramDataset, TRAINING_MODES
from .data.inverse_mlc_dataloader import InverseMLCDataset, INVERSE_MLC_EPISODE_TYPES
from .data.program_io import ProgramIO


DATASETS = ("program", "inverse-mlc")


class CyclingSampler(Sampler):
    """Samples sequentially, wrapping around at the end of the dataset."""

    def __init__(self, data_source, num_samples: int, shuffle: bool = True, track_last: int = 0):
        self.data_source = data_source
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.track_last = track_last
        self.last_indices: list[int] = []

    def __iter__(self):
        n = len(self.data_source)
        order = torch.randperm(n).tolist() if self.shuffle else list(range(n))
        recent: list[int] = []
        for i in range(self.num_samples):
            idx = order[i % n]
            if self.track_last:
                recent.append(idx)
                if len(recent) > self.track_last:
                    recent.pop(0)
            yield idx
        if self.track_last:
            self.last_indices = recent

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
    """Greedy-decode a single sequence. ``src_tokens`` is a 1-D LongTensor of ids.

    Thin wrapper around :meth:`src.data.program_io.ProgramIO.greedy_decode`;
    the canonical implementation lives there. Kept here so external scripts /
    notebooks that already import ``train.greedy_decode`` keep working.
    ``start_token`` / ``end_token`` are accepted for backwards compatibility
    but the underlying ProgramIO already knows them, so they're ignored.
    """
    del start_token, end_token  # supplied by ProgramIO
    return ProgramIO().greedy_decode(model, src_tokens, max_tokens, device)


def _check_program_match(model, dataset, idx, compiler, start_tok, end_tok, device, max_program_tokens, exec_timeout=1.0):
    """Greedy-decode one example and check the prediction is correct.

    Delegates the decode + detokenise + reverse-map + compile + execute
    pipeline to :class:`src.data.program_io.ProgramIO` (held on the dataset as
    ``dataset.io``). This function is now mostly the I/O-pair wiring that
    pulls the right inputs out of the dataset item.

    If ``dataset`` exposes a ``check_prediction(generated_token_ids, info)``
    method (e.g. :class:`InverseMLCDataset`), we delegate to it after
    greedy-decoding. That covers datasets where the target isn't an executable
    program and "correctness" is just exact-token match against the gold tail.
    ``compiler`` / ``start_tok`` / ``end_tok`` are accepted for backwards
    compatibility — the ProgramIO holds equivalents — but no longer used.
    """
    del compiler, start_tok, end_tok  # canonical sources live on dataset.io
    seq, loss_mask, program = dataset.__getitem__(idx, include_program=True)
    len_x = loss_mask.count(0)
    src_tokens = torch.tensor(seq[:len_x], dtype=torch.long)

    # No I/O pairs → vacuous truth: there are zero conditions to violate, so the
    # model can't be wrong on this example. (Also guards against a 0-length src
    # crashing dense-path RoPE inside greedy_decode for in-weight mode; the
    # ``__getitem__`` redirect makes this near-impossible in practice, but the
    # check is cheap and covers the unfilterable edge case where every program
    # in the corpus has empty I/O.)
    use_dataset_checker = hasattr(dataset, 'check_prediction')
    if src_tokens.numel() == 0:
        return True
    if not use_dataset_checker and not program.get('io_pairs'):
        return True

    io = getattr(dataset, 'io', None) or ProgramIO(tokeniser=dataset.tokeniser)
    gen_tokens = io.greedy_decode(model, src_tokens, max_program_tokens, device)

    if use_dataset_checker:
        if io.end in gen_tokens:
            gen_tokens = gen_tokens[: gen_tokens.index(io.end)]
        return dataset.check_prediction(gen_tokens, program)

    # In symbol-shuffling mode the model emits the program with mapped fn
    # names; ``detokenise_program`` reverses the per-episode permutation.
    program_str = io.detokenise_program(gen_tokens, program.get('name_map'))
    return io.check_program(program_str, program['io_pairs'], timeout=exec_timeout)


def compute_accuracy_on_indices(model, dataset, indices, device, max_program_tokens=80, desc="Accuracy"):
    """Functional accuracy over a given list of dataset indices."""
    if not indices:
        return 0.0
    model.eval()
    n_correct = 0
    for idx in tqdm(indices, desc=desc):
        if _check_program_match(model, dataset, idx, None, None, None, device, max_program_tokens):
            n_correct += 1
    return n_correct / len(indices)


def compute_validation_accuracy(model, val_dataset, device, max_program_tokens=80, max_examples=None):
    """Functional accuracy: generate a program from the I/O context and check it
    reproduces every shown I/O pair. Evaluates each program once at
    ``max_n_io_shown``."""
    n_views = val_dataset.n_io_views
    n_programs = len(val_dataset.programs)
    if max_examples is not None:
        n_programs = min(n_programs, max_examples)
    indices = [prog_idx * n_views + (n_views - 1) for prog_idx in range(n_programs)]
    return compute_accuracy_on_indices(
        model, val_dataset, indices, device,
        max_program_tokens=max_program_tokens, desc="Val accuracy",
    )


def save_checkpoint(model, optimiser, scheduler, epoch, train_loss, val_loss, args, checkpoint_dir,
                    global_step=0, best_val_loss=float('inf'), val_accuracy=0.0,
                    best_val_accuracy=0.0, wandb_run_id=None):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        # wandb's *internal* run id (not display name). Stashed so resume can
        # continue the original wandb run even if it was started without
        # --run-name (in which case the id is wandb-generated, not derivable
        # from the run name).
        'wandb_run_id': wandb_run_id,
        'args': vars(args),
    }

    torch.save(checkpoint, checkpoint_dir / 'checkpoint_latest.pt')
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

    # checkpoint_best_loss.pt — replaces the legacy checkpoint_best.pt. Migrate
    # by copying the legacy file across the first time we see a run that has it
    # but not the new file.
    best_loss_path = checkpoint_dir / 'checkpoint_best_loss.pt'
    legacy_best_path = checkpoint_dir / 'checkpoint_best.pt'
    if not best_loss_path.exists() and legacy_best_path.exists():
        shutil.copyfile(legacy_best_path, best_loss_path)

    if not best_loss_path.exists():
        torch.save(checkpoint, best_loss_path)
    else:
        existing = torch.load(best_loss_path, map_location='cpu')
        if val_loss < existing.get('val_loss', float('inf')):
            torch.save(checkpoint, best_loss_path)

    # checkpoint_best_acc.pt — for runs predating accuracy-tracking, seed the
    # file by sweeping per-epoch checkpoints for the highest stored val_accuracy.
    # Old checkpoints without val_accuracy are skipped; if none qualify we fall
    # through and just initialise from the current epoch.
    best_acc_path = checkpoint_dir / 'checkpoint_best_acc.pt'
    if not best_acc_path.exists():
        best_acc = -1.0
        best_acc_src = None
        for ep_path in sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt')):
            try:
                ep_ckpt = torch.load(ep_path, map_location='cpu')
            except Exception:
                continue
            ep_acc = ep_ckpt.get('val_accuracy')
            if ep_acc is None:
                continue
            if ep_acc > best_acc:
                best_acc = ep_acc
                best_acc_src = ep_path
        if best_acc_src is not None:
            shutil.copyfile(best_acc_src, best_acc_path)

    if not best_acc_path.exists():
        torch.save(checkpoint, best_acc_path)
    else:
        existing = torch.load(best_acc_path, map_location='cpu')
        if val_accuracy > existing.get('val_accuracy', -1.0):
            torch.save(checkpoint, best_acc_path)

    print(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(checkpoint_path, model, optimiser=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimiser is not None and 'optimiser_state_dict' in checkpoint:
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        # CosineAnnealingLR's state_dict carries T_max and eta_min, so a plain
        # load_state_dict overwrites the freshly-constructed values from the
        # current --epochs / --lr. Keep the new horizon so resumes that extend
        # --epochs decay over the new total instead of warm-restarting past the
        # old T_max.
        new_T_max = scheduler.T_max
        new_eta_min = scheduler.eta_min
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scheduler.T_max = new_T_max
        scheduler.eta_min = new_eta_min

    return (
        checkpoint['epoch'],
        checkpoint.get('train_loss'),
        checkpoint.get('val_loss'),
        checkpoint.get('global_step'),
        checkpoint.get('best_val_loss'),
        checkpoint.get('best_val_accuracy'),
    )


def _parse_corpus_arg(s: str) -> list[Path]:
    return [Path(p.strip()) for p in s.split(',') if p.strip()]


def _easy_shuffle_k_for_epoch(epoch: int, args, n_total_fns: int) -> int:
    """Linear ramp of the easy-symbol-shuffling table size from
    ``--easy-shuffle-k-start`` to ``--easy-shuffle-k-end`` (default = all
    grammar functions) over ``--easy-shuffle-ramp-epochs`` epochs (default =
    args.epochs). After the ramp window K stays clamped at k_end.

    Schedule is a function of ``epoch`` alone, so resumes pick the right K
    automatically without any extra checkpoint state.
    """
    k_end = args.easy_shuffle_k_end if args.easy_shuffle_k_end is not None else n_total_fns
    ramp = args.easy_shuffle_ramp_epochs if args.easy_shuffle_ramp_epochs is not None else args.epochs
    k_start = args.easy_shuffle_k_start
    if ramp <= 0:
        k = k_end
    else:
        progress = min(1.0, max(0.0, epoch / ramp))
        k = round(k_start + (k_end - k_start) * progress)
    return max(0, min(n_total_fns, k))


def train():
    parser = argparse.ArgumentParser(description='Train a seq2seq transformer over jagged NestedTensors')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='program', choices=list(DATASETS),
                        help='Which dataset to train on: "program" (default; uses '
                             '--train-corpus / --val-corpus + ProgramDataset) or '
                             '"inverse-mlc" (uses inverse-mlc/data_algebraic via '
                             'InverseMLCDataset; --train-corpus / --val-corpus / --mode '
                             '/ --n-io-per-program / --min-n-io-shown / --filter-empty-io '
                             'are ignored).')
    parser.add_argument('--inverse-mlc-episode-type', type=str, default='algebraic',
                        choices=list(INVERSE_MLC_EPISODE_TYPES),
                        help='Episode type when --dataset=inverse-mlc. Mirrors the '
                             'episode types in inverse-mlc/datasets.py:get_dataset.')
    parser.add_argument('--inverse-mlc-data-root', type=str, default=None,
                        help='Path to data_algebraic dir (containing train/ and val/). '
                             'Defaults to the bundled copy under src/data/inverse-mlc/.')
    parser.add_argument('--train-corpus', type=str, default='datasets/corpus-a/rl_corpus.json',
                        help='Comma-separated corpus JSON file(s) for training')
    parser.add_argument('--val-corpus', type=str, default=None,
                        help='Comma-separated corpus JSON file(s) for validation')
    parser.add_argument('--n-io-per-program', type=int, default=11,
                        help='Number of I/O pairs sampled per program (also = max_n_io_shown)')
    parser.add_argument('--min-n-io-shown', type=int, default=1,
                        help='Minimum n_io_shown per training item (each program is seen with '
                             'min_n_io_shown..n_io_per_program I/O pairs visible)')
    parser.add_argument('--data-seed', type=int, default=0,
                        help='Seed for the I/O sampler (separate from training seed)')
    parser.add_argument('--mode', type=str, default='in-weight', choices=list(TRAINING_MODES),
                        help='Training mode: in-weight (standard), symbol-shuffling '
                             '(per-episode random fn-name permutation prepended as a '
                             "<mapped> ≜ <orig> preamble; target program uses mapped names), "
                             "or easy-symbol-shuffling (same but only K of the grammar's "
                             'fn names are permuted per episode; K is ramped up over '
                             'training via --easy-shuffle-k-start / --easy-shuffle-k-end '
                             '/ --easy-shuffle-ramp-epochs)')
    parser.add_argument('--easy-shuffle-k-start', type=int, default=4,
                        help='[easy-symbol-shuffling only] Initial number of functions '
                             'permuted per episode at epoch 0.')
    parser.add_argument('--easy-shuffle-k-end', type=int, default=None,
                        help='[easy-symbol-shuffling only] Final K. None = all grammar functions.')
    parser.add_argument('--easy-shuffle-ramp-epochs', type=int, default=None,
                        help='[easy-symbol-shuffling only] Epochs over which K linearly ramps '
                             'from k_start to k_end. None = args.epochs (ramp over the whole run).')
    parser.add_argument('--max-train-programs', type=int, default=None,
                        help='Cap the training corpus to N randomly-sampled programs. '
                             'Subsampling uses --data-seed for reproducibility. Applies to '
                             'train only, not val.')
    parser.add_argument('--filter-empty-io', dest='filter_empty_io', action='store_true',
                        help='Eagerly pre-sample each program\'s IO pool at dataset init and drop '
                             'programs that return no valid pairs. Off by default (lazy sampling); '
                             'enabling adds a one-time pass over the corpus but prevents '
                             'greedy_decode from crashing on a 0-length src in in-weight mode.')
    parser.set_defaults(filter_empty_io=False)

    # Model arguments
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--d-ff', type=int, default=None, help='FFN hidden dim (default: 8/3 * d_model)')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of encoder / decoder layers')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Max sequence length for RoPE cache')
    parser.add_argument('--compile-layers', action='store_true', help='torch.compile each encoder/decoder layer')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=2500,
                        help='Steps per epoch (None = full dataset, otherwise steps_per_epoch * batch_size samples)')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--constant-lr', action='store_true',
                        help='Disable the cosine LR schedule and train at constant --lr')
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

    # If we're resuming, peek the checkpoint to recover the *actual* wandb id
    # of the prior run. This wins over --run-name as the wandb id, because the
    # name-as-id path only works when the original run was *started* with
    # --run-name. Runs started without it had a wandb-generated id that
    # name-as-id can't reconstruct, which is how you end up with two parallel
    # runs sharing a display name on wandb.
    prior_wandb_run_id = None
    if args.resume:
        try:
            _peek = torch.load(args.resume, map_location='cpu')
            prior_wandb_run_id = _peek.get('wandb_run_id')
            del _peek
        except Exception as e:
            print(f"Warning: couldn't peek wandb_run_id from {args.resume}: {e}")

    use_wandb = not args.no_wandb
    if use_wandb:
        # Resolution priority for the wandb id:
        #   1. id stashed in the resume checkpoint (true continuation),
        #   2. --run-name reused as id (legacy: name == id from the start),
        #   3. None (wandb assigns a fresh id and name).
        wandb_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        if prior_wandb_run_id is not None:
            wandb_kwargs['id'] = prior_wandb_run_id
            wandb_kwargs['resume'] = 'allow'
        elif args.run_name is not None:
            wandb_kwargs['id'] = args.run_name
            wandb_kwargs['resume'] = 'allow'
        wandb.init(**wandb_kwargs)
        if args.run_name is None:
            args.run_name = wandb.run.name
    elif args.run_name is None:
        args.run_name = f"run_seed{args.seed}"

    # Datasets
    if args.dataset == 'inverse-mlc':
        data_root = Path(args.inverse_mlc_data_root) if args.inverse_mlc_data_root else None
        print(f"Loading inverse-mlc dataset (episode_type={args.inverse_mlc_episode_type})")
        train_dataset = InverseMLCDataset(
            mode='train',
            episode_type=args.inverse_mlc_episode_type,
            data_root=data_root,
        )
        print(f"Training dataset: {len(train_dataset.programs):,} episodes -> {len(train_dataset):,} items")
        val_dataset = InverseMLCDataset(
            mode='val',
            episode_type=args.inverse_mlc_episode_type,
            data_root=data_root,
        )
        print(f"Validation dataset: {len(val_dataset.programs):,} episodes -> {len(val_dataset):,} items")
    else:
        train_files = _parse_corpus_arg(args.train_corpus)
        print(f"Loading training corpus from {train_files}")
        train_dataset = ProgramDataset(
            corpus_files=train_files,
            seed=args.data_seed,
            n_io_per_program=args.n_io_per_program,
            min_n_io_shown=args.min_n_io_shown,
            mode=args.mode,
            filter_empty_io=args.filter_empty_io,
            max_programs=args.max_train_programs,
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
                min_n_io_shown=args.min_n_io_shown,
                mode=args.mode,
                filter_empty_io=args.filter_empty_io,
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
    train_sampler = None
    if args.steps_per_epoch is not None:
        num_samples = args.steps_per_epoch * args.batch_size
        train_sampler = CyclingSampler(
            train_dataset, num_samples=num_samples, shuffle=True,
            track_last=args.batch_size,
        )
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
    scheduler = None if args.constant_lr else CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr / 10)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.run_name:
        checkpoint_dir = checkpoint_dir / args.run_name

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, train_loss, val_loss, ckpt_step, ckpt_best, ckpt_best_acc = load_checkpoint(
            args.resume, model, optimiser, scheduler,
        )
        start_epoch += 1
        if ckpt_step is not None:
            global_step = ckpt_step
        else:
            # Older checkpoints predate global_step tracking; fall back to a
            # derived value so wandb's step axis stays monotonic across the
            # resume boundary. This is exact when steps_per_epoch is fixed
            # (the common case); approximate when the dataset/batch_size
            # changed between runs.
            global_step = start_epoch * len(train_loader)
        if ckpt_best is not None:
            best_val_loss = ckpt_best
        else:
            # Older checkpoints didn't track best_val_loss. Prefer the new
            # checkpoint_best_loss.pt sibling, falling back to the legacy
            # checkpoint_best.pt so resumes from pre-migration runs still pick
            # up the tightest lower bound on disk.
            resume_dir = Path(args.resume).parent
            for sibling in (resume_dir / 'checkpoint_best_loss.pt',
                            resume_dir / 'checkpoint_best.pt'):
                if not sibling.exists():
                    continue
                try:
                    disk_best = torch.load(sibling, map_location='cpu')
                    v = disk_best.get('val_loss')
                    if v is not None:
                        best_val_loss = float(v)
                        break
                except Exception:
                    pass
        if ckpt_best_acc is not None:
            best_val_accuracy = ckpt_best_acc
        else:
            # Older checkpoints didn't track best_val_accuracy. Sweep
            # checkpoint_best_acc.pt if it exists (the migration path in
            # save_checkpoint will populate it from per-epoch files).
            sibling_acc = Path(args.resume).parent / 'checkpoint_best_acc.pt'
            if sibling_acc.exists():
                try:
                    disk_best_acc = torch.load(sibling_acc, map_location='cpu')
                    v = disk_best_acc.get('val_accuracy')
                    if v is not None:
                        best_val_accuracy = float(v)
                except Exception:
                    pass
        print(
            f"Resumed from epoch {start_epoch - 1}, train_loss: {train_loss:.4f}, "
            f"val_loss: {val_loss:.4f}, global_step: {global_step}, "
            f"best_val_loss: {best_val_loss:.4f}, best_val_accuracy: {best_val_accuracy:.2%}"
        )

    print("\nStarting training...")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        current_lr = optimiser.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        curriculum_k = None
        if args.mode == 'easy-symbol-shuffling' and args.dataset == 'program':
            # Mutating the dataset before iter() is safe because the default
            # DataLoader spawns fresh workers each epoch, picking up the new K.
            n_fns = len(train_dataset.fn_names)
            curriculum_k = _easy_shuffle_k_for_epoch(epoch, args, n_fns)
            train_dataset.n_permuted = curriculum_k
            if val_dataset is not None:
                val_dataset.n_permuted = curriculum_k
            print(f"easy-symbol-shuffling K: {curriculum_k}/{n_fns}")

        train_loss, global_step = train_epoch(
            model, train_loader, optimiser, criterion, device, args.grad_clip,
            use_wandb=use_wandb, global_step=global_step,
        )
        print(f"Train loss: {train_loss:.4f}")

        train_accuracy = 0.0
        if train_sampler is not None and train_sampler.last_indices:
            train_accuracy = compute_accuracy_on_indices(
                model, train_dataset, train_sampler.last_indices, device,
                max_program_tokens=80, desc="Train accuracy",
            )
            print(f"Train accuracy (last batch): {train_accuracy:.2%}")

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

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"New best validation accuracy: {best_val_accuracy:.2%}")

        if use_wandb:
            log_payload = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_accuracy,
                'val/loss': val_loss,
                'val/accuracy': val_accuracy,
                'learning_rate': current_lr,
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
            }
            if curriculum_k is not None:
                log_payload['curriculum/n_permuted'] = curriculum_k
            wandb.log(log_payload)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model, optimiser, scheduler, epoch, train_loss, val_loss, args, checkpoint_dir,
                global_step=global_step, best_val_loss=best_val_loss,
                val_accuracy=val_accuracy, best_val_accuracy=best_val_accuracy,
                wandb_run_id=wandb.run.id if use_wandb else None,
            )

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    train()
