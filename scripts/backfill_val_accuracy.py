"""Re-evaluate every per-epoch checkpoint in a run directory against the val
set, stamp val_accuracy/best_val_accuracy into each file, and write
checkpoint_best_acc.pt to the winner.

Intended as a one-shot migration for runs that started before val_accuracy was
recorded inside the checkpoint dict. Args (model dims, val corpus, etc.) are
recovered from the first checkpoint encountered, so a single --checkpoint-dir
flag is all that's needed for an in-place sweep.

    python -m scripts.backfill_val_accuracy --checkpoint-dir <run_dir> [--device cuda]

Use --val-corpus / --val-examples to override what's stored in the checkpoint
args (e.g. if the original val_corpus path moved).
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data.dataloader import ProgramDataset
from src.data.inverse_mlc_dataloader import InverseMLCDataset
from src.models.seq2seq import Seq2SeqTransformer
from src.train import compute_validation_accuracy, _parse_corpus_arg


def _epoch_of(path: Path) -> int:
    # checkpoint_epoch_<N>.pt — sort numerically, not lexically
    return int(path.stem.rsplit('_', 1)[-1])


def _build_val_dataset(saved_args: SimpleNamespace):
    if getattr(saved_args, 'dataset', 'program') == 'inverse-mlc':
        data_root = Path(saved_args.inverse_mlc_data_root) if saved_args.inverse_mlc_data_root else None
        return InverseMLCDataset(
            mode='val',
            episode_type=saved_args.inverse_mlc_episode_type,
            data_root=data_root,
        )

    val_files = _parse_corpus_arg(saved_args.val_corpus)
    return ProgramDataset(
        corpus_files=val_files,
        seed=saved_args.data_seed,
        n_io_per_program=saved_args.n_io_per_program,
        min_n_io_shown=saved_args.min_n_io_shown,
        mode=saved_args.mode,
        filter_empty_io=saved_args.filter_empty_io,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory containing checkpoint_epoch_*.pt files')
    parser.add_argument('--val-corpus', type=str, default=None,
                        help='Override val_corpus path (default: read from checkpoint args)')
    parser.add_argument('--val-examples', type=int, default=None,
                        help='Override val_examples cap (default: read from checkpoint args)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-program-tokens', type=int, default=80)
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and print accuracies without modifying any files')
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    epoch_paths = sorted(ckpt_dir.glob('checkpoint_epoch_*.pt'), key=_epoch_of)
    if not epoch_paths:
        raise SystemExit(f"No checkpoint_epoch_*.pt files in {ckpt_dir}")

    print(f"Found {len(epoch_paths)} per-epoch checkpoints in {ckpt_dir}")

    # Recover training args from the first checkpoint and use them to build the
    # val dataset + model skeleton. All checkpoints in a run share the same
    # args, so loading once is correct.
    first = torch.load(epoch_paths[0], map_location='cpu', weights_only=False)
    saved_args = SimpleNamespace(**first['args'])
    if args.val_corpus is not None:
        saved_args.val_corpus = args.val_corpus
    if args.val_examples is not None:
        saved_args.val_examples = args.val_examples

    print(f"Building val dataset from {saved_args.val_corpus}")
    val_dataset = _build_val_dataset(saved_args)
    print(f"Val dataset: {len(val_dataset.programs):,} programs")

    n_tokens = len(val_dataset.tokeniser.vocab)
    device = torch.device(args.device)
    model = Seq2SeqTransformer(
        n_tokens=n_tokens,
        d_model=saved_args.d_model,
        n_heads=saved_args.n_heads,
        n_layers=saved_args.n_layers,
        d_ff=saved_args.d_ff,
        max_seq_len=saved_args.max_seq_len,
        compile_layers=getattr(saved_args, 'compile_layers', False),
    ).to(device)

    running_best = -1.0
    best_path = None
    results = []

    for path in epoch_paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)

        acc = compute_validation_accuracy(
            model, val_dataset, device,
            max_program_tokens=args.max_program_tokens,
            max_examples=saved_args.val_examples,
        )

        if acc > running_best:
            running_best = acc
            best_path = path

        results.append((path, ckpt.get('epoch'), acc))
        print(f"  epoch {ckpt.get('epoch'):>4}  val_accuracy={acc:.2%}  "
              f"(running best {running_best:.2%} @ {best_path.name if best_path else '?'})")

        if args.dry_run:
            continue

        # Stamp val_accuracy into the per-epoch file along with the
        # running-max best_val_accuracy. Matches what live training would've
        # written so a future resume reads consistent state.
        ckpt['val_accuracy'] = acc
        ckpt['best_val_accuracy'] = running_best
        torch.save(ckpt, path)

    print("\nSummary:")
    print(f"  best val_accuracy: {running_best:.2%} at {best_path.name if best_path else '?'}")

    if args.dry_run:
        print("Dry run — no files modified.")
        return

    # Write checkpoint_best_acc.pt from the winner. Also re-stamp
    # checkpoint_latest.pt's val_accuracy / best_val_accuracy so a resume
    # picks up the correct in-memory floor without sweeping again.
    best_acc_path = ckpt_dir / 'checkpoint_best_acc.pt'
    winner = torch.load(best_path, map_location='cpu', weights_only=False)
    torch.save(winner, best_acc_path)
    print(f"Wrote {best_acc_path}")

    latest_path = ckpt_dir / 'checkpoint_latest.pt'
    if latest_path.exists():
        latest = torch.load(latest_path, map_location='cpu', weights_only=False)
        latest_epoch = latest.get('epoch')
        # Find the matching per-epoch result so val_accuracy is consistent
        # with what we just stamped into the per-epoch file.
        match = next((r for r in results if r[1] == latest_epoch), None)
        if match is not None:
            latest['val_accuracy'] = match[2]
        latest['best_val_accuracy'] = running_best
        torch.save(latest, latest_path)
        print(f"Updated {latest_path} (best_val_accuracy={running_best:.2%})")


if __name__ == '__main__':
    main()
