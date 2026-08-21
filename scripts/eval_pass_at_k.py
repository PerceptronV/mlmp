"""Accuracy@k over the validation set, for one or more trained checkpoints.

For every validation program we sample k programs from the model (at
--temperature, so the samples actually differ) and keep the best one — the one
matching the most of the shown I/O pairs. The reported numbers are then the
same taxonomy the training loop logs, computed over those best-of-k programs:

    accuracy@k          at least one of the k samples reproduced every pair
    failure modes       malformed / runtime_error / total_ / partial_mismatch
                        of the best sample — 'malformed' means all k samples
                        were malformed
    outcomes by n_io    match_0..match_n per n_io_shown bin, best-of-k

Standalone and offline: model dims, val corpus, grammar and training mode all
come from each checkpoint's stored args, so a bare list of checkpoints is
enough. k decode passes per model make this far too expensive for the training
loop, which is why it lives here.

    python -m scripts.eval_pass_at_k <ckpt.pt|run_dir> [...] [-k 8]
        [--temperature 1.0] [--val-examples 256] [--json out.json]

A run directory resolves to checkpoint_<--ckpt-select>.pt (default best_acc).
Symbol-shuffling runs are evaluated at full shuffling, matching how the
training loop scores validation.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.train import (
    FAILURE_MODES,
    build_model,
    build_val_dataset,
    compute_pass_at_k_metrics,
    format_metrics,
)

# Everything build_val_dataset reads. Checkpoints agreeing on all of it share
# one val dataset — loading (and IO-sampling) the same corpus per model is
# pure cost.
_VAL_ARG_KEYS = ('dataset', 'inverse_mlc_episode_type', 'inverse_mlc_data_root',
                 'val_corpus', 'val_split', 'train_corpus', 'data_seed',
                 'n_io_per_program', 'min_n_io_shown', 'mode', 'filter_empty_io',
                 'grammar', 'split_seed')


def _resolve_ckpt(path: Path, select: str) -> Path:
    """A checkpoint file, or a run directory + which checkpoint to take from it
    ('best_acc' | 'best_loss' | 'latest' | 'epoch_<N>')."""
    return path / f"checkpoint_{select}.pt" if path.is_dir() else path


def _val_dataset(saved_args, cache: dict):
    key = tuple(str(getattr(saved_args, k, None)) for k in _VAL_ARG_KEYS)
    if key not in cache:
        dataset = build_val_dataset(saved_args)
        if dataset is None:
            raise SystemExit(
                "Checkpoint was trained without a validation set "
                "(no --val-corpus / --val-split); pass --val-corpus")
        cache[key] = dataset
    return cache[key]


def _load_model(ckpt: dict, saved_args, n_tokens: int, device) -> torch.nn.Module:
    model = build_model(saved_args, n_tokens)
    state = ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state):
        # --compile-layers wraps each layer, prefixing its keys.
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def _jsonable(metrics: dict) -> dict:
    bins = metrics['outcomes_by_n_io']
    return {
        **{k: v for k, v in metrics.items()
           if k not in ('failure_modes', 'outcomes_by_n_io')},
        'failure_modes': dict(metrics['failure_modes']),
        'outcomes_by_n_io': ({str(n): dict(c) for n, c in sorted(bins.items())}
                             if bins else None),
    }


def _summary_table(report: dict, k: int) -> list[str]:
    modes = FAILURE_MODES[1:]  # 'correct' is accuracy@k, already its own column
    lines = [f"{'run':<28}{'acc@1':>8}{f'acc@{k}':>8}"
             + "".join(f"{m:>18}" for m in modes)]
    for name, entry in report.items():
        m = entry['metrics']
        row = f"{name:<28}{m['accuracy_at_k'][0]:>8.1%}{m['accuracy']:>8.1%}"
        total = sum(m['failure_modes'].values())
        for mode in modes:
            cell = f"{m['failure_modes'][mode] / total:.1%}" if total else "n/a"
            row += f"{cell:>18}"
        lines.append(row)
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='+',
                        help='Checkpoint .pt files and/or run directories')
    parser.add_argument('-k', type=int, default=8,
                        help='Programs sampled per validation item (default 8)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default 1.0; 0 = greedy, which '
                             'makes all k samples identical)')
    parser.add_argument('--ckpt-select', type=str, default='best_acc',
                        help="Which checkpoint to take from a run directory: "
                             "best_acc (default) | best_loss | latest | epoch_<N>")
    parser.add_argument('--val-corpus', type=str, default=None,
                        help='Override the val corpus stored in the checkpoint args')
    parser.add_argument('--val-examples', type=int, default=None,
                        help='Cap on validation programs (default: all)')
    parser.add_argument('--max-program-tokens', type=int, default=80)
    parser.add_argument('--decode-batch-size', type=int, default=64)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--constrain', action='store_true',
                        help='Mask each decode step to tokens that can still '
                             'complete a well-formed program, which removes the '
                             'malformed outcome by construction')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds the sampler, so a rerun draws the same k programs')
    parser.add_argument('--json', type=str, default=None,
                        help='Write the full per-model metrics (including every '
                             'outcome counter) to this JSON file')
    args = parser.parse_args()

    if args.k > 1 and args.temperature <= 0:
        print(f"Warning: --temperature {args.temperature} is greedy, so all "
              f"{args.k} samples will be identical")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset_cache: dict = {}
    report: dict = {}

    for raw in args.checkpoints:
        path = _resolve_ckpt(Path(raw).expanduser(), args.ckpt_select)
        if not path.exists():
            raise SystemExit(f"No such checkpoint: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        saved_args = SimpleNamespace(**ckpt['args'])
        if args.val_corpus is not None:
            saved_args.val_corpus = args.val_corpus
        name = getattr(saved_args, 'run_name', None) or path.parent.name
        if name in report:  # several checkpoints of the same run
            name = f"{name}/{path.stem}"

        val_dataset = _val_dataset(saved_args, dataset_cache)
        model = _load_model(ckpt, saved_args, len(val_dataset.tokeniser.vocab), device)

        n_programs = len(val_dataset.programs)
        if args.val_examples is not None:
            n_programs = min(n_programs, args.val_examples)
        greedy = ckpt.get('val_accuracy')
        print(f"\n=== {name} ({path.name}, epoch {ckpt.get('epoch')}, "
              f"mode={saved_args.mode}) ===")
        print(f"  {n_programs:,} val programs, k={args.k}, T={args.temperature}"
              f"{', constrained' if args.constrain else ''}"
              + (f", stored greedy val accuracy: {greedy:.2%}"
                 if isinstance(greedy, (int, float)) else ""))

        metrics = compute_pass_at_k_metrics(
            model, val_dataset, device, k=args.k, temperature=args.temperature,
            max_program_tokens=args.max_program_tokens,
            max_examples=args.val_examples,
            decode_batch_size=args.decode_batch_size,
            constrain=args.constrain,
        )
        for line in format_metrics(metrics):
            print(f"  {line}")
        report[name] = {'checkpoint': str(path), 'epoch': ckpt.get('epoch'),
                        'mode': saved_args.mode, 'n_programs': n_programs,
                        'metrics': metrics}

    print("\n" + "\n".join(_summary_table(report, args.k)))
    print("(failure modes are of each program's best-of-k sample)")

    if args.json:
        out = {'config': {'k': args.k, 'temperature': args.temperature,
                          'seed': args.seed, 'val_examples': args.val_examples,
                          'constrain': args.constrain},
               'runs': {name: {**entry, 'metrics': _jsonable(entry['metrics'])}
                        for name, entry in report.items()}}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.json}")


if __name__ == '__main__':
    main()
