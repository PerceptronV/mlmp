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

import torch

from src.train import FAILURE_MODES, compute_pass_at_k_metrics, format_metrics
from scripts.eval_common import add_common_args, jsonable, load_runs


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
    add_common_args(parser)
    parser.add_argument('-k', type=int, default=8,
                        help='Programs sampled per validation item (default 8)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default 1.0; 0 = greedy, which '
                             'makes all k samples identical)')
    args = parser.parse_args()

    if args.k > 1 and args.temperature <= 0:
        print(f"Warning: --temperature {args.temperature} is greedy, so all "
              f"{args.k} samples will be identical")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    report: dict = {}

    for run in load_runs(args):
        print(f"\n{run.header()}")
        stored = run.stored_accuracy
        print(f"  {run.n_programs:,} val programs, k={args.k}, T={args.temperature}"
              + (', constrained' if args.constrain else '')
              + (f", stored greedy val accuracy: {stored:.2%}" if stored is not None else ""))

        metrics = compute_pass_at_k_metrics(
            run.model, run.dataset, device, k=args.k, temperature=args.temperature,
            max_program_tokens=args.max_program_tokens,
            max_examples=args.val_examples,
            decode_batch_size=args.decode_batch_size,
            constrain=args.constrain,
        )
        for line in format_metrics(metrics):
            print(f"  {line}")
        report[run.name] = {'checkpoint': str(run.path), 'epoch': run.ckpt.get('epoch'),
                            'mode': run.args.mode, 'n_programs': run.n_programs,
                            'metrics': metrics}

    print("\n" + "\n".join(_summary_table(report, args.k)))
    print("(failure modes are of each program's best-of-k sample)")

    if args.json:
        out = {'config': {'k': args.k, 'temperature': args.temperature,
                          'seed': args.seed, 'val_examples': args.val_examples,
                          'constrain': args.constrain},
               'runs': {name: {**entry, 'metrics': jsonable(entry['metrics'])}
                        for name, entry in report.items()}}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.json}")


if __name__ == '__main__':
    main()
