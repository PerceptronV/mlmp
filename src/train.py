import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import wandb
import random
import numpy as np

from .models.transformer import DecoderOnlyTransformer
from .data.dataloader import ProgramDataset


class CyclingSampler(Sampler):
    """Samples sequentially, wrapping around at the end of the dataset."""
    
    def __init__(self, data_source, num_samples: int, shuffle: bool = True):
        self.data_source = data_source
        self.num_samples = num_samples
        self.shuffle = shuffle
    
    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            # Shuffle the order, then cycle through it
            order = torch.randperm(n).tolist()
        else:
            order = list(range(n))
        
        # Yield indices, cycling through the order
        for i in range(self.num_samples):
            yield order[i % n]
    
    def __len__(self):
        return self.num_samples


def collate_fn(batch, pad_token):
    """Collate function to pad sequences to the same length."""
    max_len = max(len(seq) for seq, _ in batch)
    padded_seqs = []
    padded_masks = []

    for seq, mask in batch:
        missing = max_len - len(seq)
        padded_seqs.append(seq + [pad_token] * missing)
        padded_masks.append(mask + [0] * missing)
    return torch.tensor(padded_seqs, dtype=torch.long), torch.tensor(padded_masks, dtype=torch.float32)


def compute_loss(model, batch, criterion, device):
    """Compute the loss for a batch."""
    model.train()
    seqs, loss_masks = batch
    seqs = seqs.to(device) # (B, L)
    loss_masks = loss_masks.to(device) # (B, L - 1)

    # Prepare input and target
    # Input: all tokens except the last
    # Target: all tokens except the first
    input_seq = seqs[:, :-1] # (B, L-1)
    target_seq = seqs[:, 1:] # (B, L-1)

    # Forward pass with all logits
    logits = model(input_seq, return_all_logits=True)  # (B, L-1, n_tokens)

    # Reshape for loss computation
    logits_vocab_first = logits.transpose(1, 2) # (B, n_tokens, L-1)
    loss = criterion(logits_vocab_first, target_seq) # (B, L-1)

    # Only backpropagate for the positions where the loss mask is 1
    loss = (loss * loss_masks).sum() / loss_masks.sum()

    return loss, loss_masks.sum()


def train_epoch(model, dataloader, optimiser, criterion, device, grad_clip=1.0, use_wandb=False, global_step=0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
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
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            batch_loss, n_tokens = compute_loss(model, batch, criterion, device)
            total_loss += batch_loss.item() * n_tokens
            total_tokens += n_tokens

            pbar.set_postfix({'val_loss': batch_loss.item()})

    return total_loss.item() / total_tokens


def save_checkpoint(model, optimiser, scheduler, epoch, train_loss, val_loss, args, checkpoint_dir):
    """Save a checkpoint."""
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

    # Save latest checkpoint
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)

    # Save epoch checkpoint
    epoch_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, epoch_path)

    # Save best checkpoint if this is the best so far
    best_path = checkpoint_dir / 'checkpoint_best.pt'
    if not best_path.exists():
        torch.save(checkpoint, best_path)
    else:
        best_checkpoint = torch.load(best_path)
        if val_loss < best_checkpoint['val_loss']:
            torch.save(checkpoint, best_path)

    print(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(checkpoint_path, model, optimiser=None, scheduler=None):
    """Load a checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimiser is not None and 'optimiser_state_dict' in checkpoint:
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint.get('train_loss'), checkpoint.get('val_loss')


def train():
    parser = argparse.ArgumentParser(description='Train a decoder-only transformer')

    # Data arguments
    parser.add_argument('--train-dir', type=str, default='datasets/query_first_template_seed42/train',
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default='datasets/query_first_template_seed42/validation',
                        help='Path to validation data directory')

    # Model arguments
    parser.add_argument('--d-embed', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=1000,
                        help='Steps per epoch (if None, use full dataset, otherwise use steps_per_epoch * batch_size samples)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')

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
    parser.add_argument('--num-workers', type=int, default=0, help='Number of dataloader workers (use 0 for now due to pickling issues)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )
        # Update run_name to the wandb generated name if not specified
        if args.run_name is None:
            args.run_name = wandb.run.name
    else:
        # If wandb is disabled and no run_name is provided, use a default
        if args.run_name is None:
            args.run_name = f"run_seed{args.seed}"

    # Load datasets
    print(f"Loading training data from {args.train_dir}")
    train_dataset = ProgramDataset(Path(args.train_dir))
    print(f"Training dataset size: {len(train_dataset)}")

    val_dataset = None
    if args.val_dir:
        print(f"Loading validation data from {args.val_dir}")
        val_dataset = ProgramDataset(Path(args.val_dir))
        print(f"Validation dataset size: {len(val_dataset)}")

    # Get vocabulary size
    n_tokens = len(train_dataset.tokeniser.vocab)
    print(f"Vocabulary size: {n_tokens}")

    # Load or compute max sequence length (with caching)
    dataset_root = Path(args.train_dir).parent
    metadata_path = dataset_root / 'metadata.json'
    
    if metadata_path.exists():
        print(f"Loading cached metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            max_lengths = metadata.get('max_lengths')
            print(f"Max lengths (cached): {max_lengths}")
    else:
        # No metadata file, compute and create it
        print("Computing max sequence length...")
        max_lengths = train_dataset.compute_max_lengths(verbose=True)
        print(f"Max lengths: {max_lengths}")
        
        # Save metadata
        metadata = {
            'max_lengths': max_lengths,
            'vocab_size': n_tokens,
            'train_dataset_size': len(train_dataset),
            'val_dataset_size': len(val_dataset) if val_dataset else 0,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    # Log dataset info to wandb
    if use_wandb:
        wandb.config.update({
            'vocab_size': n_tokens,
            'train_dataset_size': len(train_dataset),
            'val_dataset_size': len(val_dataset) if val_dataset else 0,
            'max_seq_length': max_lengths,
        })

    # Create dataloaders
    if args.steps_per_epoch is not None:
        # Use CyclingSampler to sample sequentially, wrapping at end of dataset
        num_samples = args.steps_per_epoch * args.batch_size
        train_sampler = CyclingSampler(train_dataset, num_samples=num_samples, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, train_dataset.pad),
        )
        print(f"Using {args.steps_per_epoch} steps per epoch ({num_samples} samples)")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, train_dataset.pad),
        )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda batch: collate_fn(batch, val_dataset.pad),
        )

    # Create model
    print("Creating model...")
    model = DecoderOnlyTransformer(
        n_tokens=n_tokens,
        d_embed=args.d_embed,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_len=max_lengths['total'],
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Log model info to wandb
    if use_wandb:
        wandb.config.update({'n_params': n_params})

    device = torch.device(args.device)
    model = model.to(device)

    # Create optimiser and scheduler
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=args.lr / 10)

    # Loss function (ignore padding tokens)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=train_dataset.pad)

    # Set up checkpoint directory with run name
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.run_name:
        checkpoint_dir = checkpoint_dir / args.run_name

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, train_loss, val_loss = load_checkpoint(args.resume, model, optimiser, scheduler)
        start_epoch += 1  # Start from next epoch
        print(f"Resumed from epoch {start_epoch - 1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        current_lr = optimiser.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")

        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimiser, criterion, device, args.grad_clip,
            use_wandb=use_wandb, global_step=global_step
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        val_loss = train_loss  # Default to train loss if no validation set
        if val_loader:
            val_loss = validate(model, val_loader, criterion, device)
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f}")

        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'learning_rate': current_lr,
                'best_val_loss': best_val_loss,
            })

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimiser, scheduler, epoch, train_loss, val_loss, args, checkpoint_dir)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Finish wandb run
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    train()
