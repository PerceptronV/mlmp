"""Validation accuracy binned by target program length.

Answers the question behind ``--max-program-length``: how far does accuracy
fall as the program the model has to emit gets longer, and does the failure
turn into 'malformed' or into a wrong-but-well-formed program? Each validation
program is decoded once (greedily by default, or best-of-k with ``-k``),
classified with the usual taxonomy, and bucketed by the token length of its
*target* — the thing the decoder would have had to produce.

Standalone and offline, like scripts/eval_pass_at_k.py: model dims, val corpus,
grammar, training mode and any length filter all come from each checkpoint's
stored args.

    python -m scripts.eval_by_length <ckpt.pt|run_dir> [...] [--bin-width 10]
        [-k 1] [--constrain] [--plot outputs/length] [--json out.json]

Note the interaction with ``--max-program-tokens`` (80 by default): a target
longer than that budget can never be emitted, so its bin is capped at 0%
accuracy by construction. Those bins are marked in the printed table.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from src.analysis.plotting import apply_rc, save_fig
from src.train import (
    FAILURE_MODES,
    _failure_mode,
    _val_view_indices,
    decode_best_of_k,
)
from scripts.eval_common import add_common_args, load_runs


def target_lengths(dataset, indices) -> list[int]:
    """Token length of the target program behind each dataset index.

    Read back through ``__getitem__`` rather than off ``dataset.programs``:
    the dataset redirects indices whose program has an empty I/O pool, so the
    program actually scored at an index isn't always the one at that slot.
    """
    lengths = []
    for idx in indices:
        _seq, _mask, info = dataset.__getitem__(idx, include_program=True)
        lengths.append(len(dataset.tokeniser.tokenise_program(info['program'])))
    return lengths


def bin_outcomes(results, lengths, bin_width: int) -> dict[int, dict]:
    """Group per-program outcomes into ``bin_width``-token buckets.

    Keyed by the bucket's lower edge, so bucket ``b`` holds lengths
    ``[b, b + bin_width)``.
    """
    bins: dict[int, dict] = defaultdict(
        lambda: {'n': 0, 'correct': 0, 'modes': Counter(), 'max_length': 0})
    for (correct, status, n_matched, n_shown), length in zip(results, lengths):
        entry = bins[(length // bin_width) * bin_width]
        entry['n'] += 1
        entry['correct'] += int(correct)
        entry['max_length'] = max(entry['max_length'], length)
        if status is not None:
            entry['modes'][_failure_mode(status, n_matched, n_shown)] += 1
    return dict(bins)


def format_bins(bins: dict[int, dict], bin_width: int, budget: int) -> list[str]:
    lines = [f"{'length':>12}{'n':>7}{'accuracy':>10}{'malformed':>11}"
             f"{'mismatch':>10}   (mismatch = total+partial)"]
    for lo in sorted(bins):
        e = bins[lo]
        modes, total = e['modes'], sum(e['modes'].values())
        malformed = modes['malformed'] / total if total else 0.0
        mismatch = ((modes['total_mismatch'] + modes['partial_mismatch']) / total
                    if total else 0.0)
        flag = '  <- past the decode budget' if lo >= budget else ''
        lines.append(f"{f'{lo}-{lo + bin_width - 1}':>12}{e['n']:>7}"
                     f"{e['correct'] / e['n']:>10.1%}{malformed:>11.1%}"
                     f"{mismatch:>10.1%}{flag}")
    return lines


def plot_bins(report: dict, bin_width: int, budget: int, outdir: Path) -> Path:
    """Accuracy, malformed share, and bin population against target length.

    Three stacked panels on a shared x rather than twin y-axes: accuracy and
    counts live on incomparable scales, and overlaying them on one frame is
    the classic dual-axis distortion. The population panel is what keeps the
    long-length bins honest — they are usually thin.
    """
    import matplotlib.pyplot as plt

    apply_rc()
    fig, (ax_acc, ax_mal, ax_n) = plt.subplots(
        3, 1, sharex=True, figsize=(7.0, 7.5),
        gridspec_kw={'height_ratios': [3, 2, 1.5]})

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Colour follows the run, fixed by sorted name so a rerun paints the same
    # run the same way regardless of command-line order.
    colours = {name: cycle[i % len(cycle)]
               for i, name in enumerate(sorted(report))}
    # Runs that agree exactly would hide one another under a single style.
    dashes = ['-', '--', '-.', ':']
    styles = {name: dashes[i % len(dashes)]
              for i, name in enumerate(sorted(report))}

    for name in sorted(report):
        bins = report[name]['bins']
        los = sorted(bins)
        centres = [lo + bin_width / 2 for lo in los]
        acc = [bins[lo]['correct'] / bins[lo]['n'] for lo in los]
        totals = [sum(bins[lo]['modes'].values()) for lo in los]
        mal = [bins[b]['modes']['malformed'] / t if t else 0.0
               for b, t in zip(los, totals)]
        style = dict(color=colours[name], marker='o', markersize=4, label=name,
                     linestyle=styles[name])
        ax_acc.plot(centres, acc, **style)
        ax_mal.plot(centres, mal, **style)
        ax_n.plot(centres, [bins[lo]['n'] for lo in los], drawstyle='steps-mid',
                  color=colours[name], alpha=0.8, linestyle=styles[name])

    for ax in (ax_acc, ax_mal):
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax_n.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax_acc.set_ylabel('accuracy')
    ax_mal.set_ylabel('malformed share')
    ax_n.set_ylabel('programs')
    ax_n.set_xlabel('target program length (decoder tokens)')
    ax_acc.set_title('Validation accuracy by target program length')

    # The decode budget is a hard ceiling: nothing past it can be emitted.
    for ax in (ax_acc, ax_mal, ax_n):
        if budget is not None:
            ax.axvline(budget, color='0.4', linestyle=':', linewidth=1.2)
    ax_acc.text(budget, 0.96, ' decode budget', color='0.4', fontsize=8,
                ha='left', va='top')

    if len(report) > 1:   # a single series is named by the title
        ax_acc.legend(frameon=False)
    fig.align_ylabels()
    return save_fig(fig, outdir, 'accuracy_by_length')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument('--bin-width', type=int, default=10,
                        help='Token width of each length bucket (default 10)')
    parser.add_argument('-k', type=int, default=1,
                        help='Samples per program, best-of-k (default 1 = one '
                             'greedy decode)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: 0 when k=1, else 1.0)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Directory to write accuracy_by_length.{png,pdf} into')
    args = parser.parse_args()

    temperature = args.temperature
    if temperature is None:
        temperature = 0.0 if args.k == 1 else 1.0

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    report: dict = {}

    for run in load_runs(args):
        if hasattr(run.dataset, 'check_prediction'):
            raise SystemExit(
                f"{run.name}: this dataset scores predictions with its own "
                f"checker and has no program targets to measure the length of")
        print(f"\n{run.header()}")
        cap = getattr(run.args, 'max_program_length', None)
        print(f"  {run.n_programs:,} val programs, k={args.k}, T={temperature}"
              + (', constrained' if args.constrain else '')
              + (f", trained with --max-program-length {cap}" if cap else ""))

        indices, _views, _n_views, _varies = _val_view_indices(
            run.dataset, args.val_examples)
        lengths = target_lengths(run.dataset, indices)
        results, _curve = decode_best_of_k(
            run.model, run.dataset, indices, device, k=args.k,
            temperature=temperature, max_program_tokens=args.max_program_tokens,
            decode_batch_size=args.decode_batch_size, constrain=args.constrain)

        bins = bin_outcomes(results, lengths, args.bin_width)
        for line in format_bins(bins, args.bin_width, args.max_program_tokens):
            print(f"  {line}")
        overall = sum(r[0] for r in results) / len(results) if results else 0.0
        n_over = sum(1 for l in lengths if l > args.max_program_tokens)
        print(f"  overall: {overall:.2%}   targets past the "
              f"{args.max_program_tokens}-token decode budget: {n_over}/{len(lengths)} "
              f"({n_over / max(1, len(lengths)):.1%})")
        report[run.name] = {'checkpoint': str(run.path), 'bins': bins,
                            'accuracy': overall, 'n_over_budget': n_over,
                            'max_program_length': cap}

    if args.plot:
        path = plot_bins(report, args.bin_width, args.max_program_tokens,
                         Path(args.plot))
        print(f"\nWrote {path.with_suffix('.png')} (and .pdf)")

    if args.json:
        out = {'config': {'bin_width': args.bin_width, 'k': args.k,
                          'temperature': temperature, 'seed': args.seed,
                          'constrain': args.constrain,
                          'max_program_tokens': args.max_program_tokens},
               'runs': {name: {**entry,
                               'bins': {str(lo): {**b, 'modes': dict(b['modes'])}
                                        for lo, b in entry['bins'].items()}}
                        for name, entry in report.items()}}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"Wrote {args.json}")


if __name__ == '__main__':
    main()
