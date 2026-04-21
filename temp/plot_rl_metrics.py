"""
Visualise RL program telemetry from {output_dir}/rl_metrics.jsonl.

Produces two figures:

  1. stats    — 2x2 grid of min/max/mean/median for
                {generated,novel} x {depth,size} over iterations.
  2. quartiles — 1x2 grid: for each pool, 4 lines showing the mean
                 size/depth ratio within each global size quartile.

Each JSONL line has raw parallel arrays:
    {
      "iteration": int,
      "novel_cumulative": int,
      "generated_cumulative": int,
      "generated": {"depth": [...], "size": [...]},
      "novel":     {"depth": [...], "size": [...]}
    }

Usage:
    python scripts/plot_rl_metrics.py output/corpus-a
    python scripts/plot_rl_metrics.py output/corpus-a --smooth 25
    python scripts/plot_rl_metrics.py output/corpus-a --out plot.png
        # writes plot_stats.png and plot_quartiles.png
"""

import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


POOLS = ['generated', 'novel']

PANELS = [  # for the stats figure
    ('generated', 'depth'),
    ('generated', 'size'),
    ('novel',     'depth'),
    ('novel',     'size'),
]


# ---------- loading ---------------------------------------------------------

def load_metrics(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------- stats figure ----------------------------------------------------

def values_for(row, pool, field):
    return row[pool][field]


def iter_stat(row, pool, field, stat):
    xs = values_for(row, pool, field)
    if not xs:
        return None
    if stat == 'min':    return min(xs)
    if stat == 'max':    return max(xs)
    if stat == 'mean':   return sum(xs) / len(xs)
    if stat == 'median': return median(xs)
    raise ValueError(stat)


def summarise(rows, pool, field):
    all_xs = [x for r in rows for x in values_for(r, pool, field)]
    if not all_xs:
        return None
    return {
        'n':      len(all_xs),
        'min':    min(all_xs),
        'max':    max(all_xs),
        'mean':   sum(all_xs) / len(all_xs),
        'median': median(all_xs),
    }


STAT_COLOURS = {
    'min':    'tab:purple',
    'mean':   'tab:blue',
    'median': 'tab:green',
    'max':    'tab:red',
}


def plot_stat_panel(ax, iters, rows, pool, field, window):
    for stat, colour in STAT_COLOURS.items():
        ys = rolling_mean([iter_stat(r, pool, field, stat) for r in rows], window)
        xs = [i for i, y in zip(iters, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, label=stat, colour=colour, linewidth=1.2)

    s = summarise(rows, pool, field)
    if s is not None:
        title = (f"{pool}_{field}  (n={s['n']:,}  "
                 f"min={s['min']:.0f}  max={s['max']:.0f}  "
                 f"mean={s['mean']:.2f}  med={s['median']:.1f})")
    else:
        title = f"{pool}_{field}"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("iteration")
    ax.set_ylabel(field)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)


# ---------- quartile figure -------------------------------------------------

def size_quartile_bounds(rows, pool):
    all_sizes = sorted(s for r in rows for s in r[pool]['size'])
    if len(all_sizes) < 4:
        return None
    n = len(all_sizes)
    return (all_sizes[n // 4], all_sizes[n // 2], all_sizes[3 * n // 4])


def quartile_of(size, bounds):
    q1, q2, q3 = bounds
    if size <= q1: return 0
    if size <= q2: return 1
    if size <= q3: return 2
    return 3


def iter_mean_by_quartile(row, pool, bounds, metric_fn):
    """Mean of metric_fn(depth, size) per size-quartile for one iteration."""
    buckets = [[], [], [], []]
    for d, s in zip(row[pool]['depth'], row[pool]['size']):
        if d <= 0:
            continue
        buckets[quartile_of(s, bounds)].append(metric_fn(d, s))
    return [(sum(b) / len(b)) if b else None for b in buckets]


QUARTILE_COLOURS = ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:red']


def plot_quartile_panel(ax, iters, rows, pool, window, metric_fn, ylabel, title):
    bounds = size_quartile_bounds(rows, pool)
    if bounds is None:
        ax.set_title(f"{title} (insufficient data)")
        return
    q1, q2, q3 = bounds
    max_size = max(s for r in rows for s in r[pool]['size'])

    labels = [
        f"Q1 (size ≤ {q1:.0f})",
        f"Q2 ({q1:.0f} < size ≤ {q2:.0f})",
        f"Q3 ({q2:.0f} < size ≤ {q3:.0f})",
        f"Q4 ({q3:.0f} < size ≤ {max_size:.0f})",
    ]

    per_iter = [iter_mean_by_quartile(r, pool, bounds, metric_fn) for r in rows]
    for q in range(4):
        ys = rolling_mean([p[q] for p in per_iter], window)
        xs = [i for i, y in zip(iters, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        ax.plot(xs, ys, label=labels[q], colour=QUARTILE_COLOURS[q], linewidth=1.2)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)


# ---------- shared helpers --------------------------------------------------

def rolling_mean(xs, window):
    if window <= 1:
        return list(xs)
    out, buf = [], []
    for x in xs:
        if x is None:
            out.append(None)
            buf = []
            continue
        buf.append(x)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def print_stats_summary(rows):
    print()
    print(f"{'metric':20s}  {'n':>10s}  {'min':>5s}  {'max':>5s}  "
          f"{'mean':>7s}  {'median':>8s}")
    print("-" * 65)
    for pool, field in PANELS:
        s = summarise(rows, pool, field)
        key = f"{pool}_{field}"
        if s is None:
            print(f"{key:20s}  (no data)")
            continue
        print(f"{key:20s}  {s['n']:>10,d}  {s['min']:>5.0f}  {s['max']:>5.0f}  "
              f"{s['mean']:>7.2f}  {s['median']:>8.2f}")


def print_quartile_summary(rows):
    print()
    print(f"{'pool':12s}  {'n':>10s}  {'Q1 size≤':>10s}  {'Q2 size≤':>10s}  "
          f"{'Q3 size≤':>10s}  {'max size':>10s}  {'overall ratio':>14s}")
    print("-" * 90)
    for pool in POOLS:
        sizes = [s for r in rows for s in r[pool]['size']]
        depths = [d for r in rows for d in r[pool]['depth']]
        bounds = size_quartile_bounds(rows, pool)
        if not sizes or bounds is None:
            print(f"{pool:12s}  (insufficient data)")
            continue
        ratios = [s / d for s, d in zip(sizes, depths) if d > 0]
        overall_ratio = sum(ratios) / len(ratios) if ratios else float('nan')
        print(f"{pool:12s}  {len(sizes):>10,d}  {bounds[0]:>10.0f}  "
              f"{bounds[1]:>10.0f}  {bounds[2]:>10.0f}  {max(sizes):>10.0f}  "
              f"{overall_ratio:>14.3f}")


# ---------- figure assembly -------------------------------------------------

def build_stats_figure(rows, iters, window, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    # Row 1: depth (generated | novel)
    plot_stat_panel(axes[0, 0], iters, rows, 'generated', 'depth', window)
    plot_stat_panel(axes[0, 1], iters, rows, 'novel',     'depth', window)
    # Row 2: size (generated | novel)
    plot_stat_panel(axes[1, 0], iters, rows, 'generated', 'size',  window)
    plot_stat_panel(axes[1, 1], iters, rows, 'novel',     'size',  window)
    # Share y within each row
    axes[0, 0].sharey(axes[0, 1])
    axes[1, 0].sharey(axes[1, 1])

    title = f"RL program statistics — {output_dir}"
    if window > 1:
        title += f"  (rolling mean, window={window})"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def build_quartile_figure(rows, iters, window, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    ratio_fn = lambda d, s: s / d
    depth_fn = lambda d, s: d
    # Row 1: mean size/depth ratio
    plot_quartile_panel(axes[0, 0], iters, rows, 'generated', window,
                        ratio_fn, "mean (size/depth)", "generated — size/depth ratio")
    plot_quartile_panel(axes[0, 1], iters, rows, 'novel',     window,
                        ratio_fn, "mean (size/depth)", "novel — size/depth ratio")
    # Row 2: mean depth
    plot_quartile_panel(axes[1, 0], iters, rows, 'generated', window,
                        depth_fn, "mean depth", "generated — depth")
    plot_quartile_panel(axes[1, 1], iters, rows, 'novel',     window,
                        depth_fn, "mean depth", "novel — depth")
    # Share y within each row
    axes[0, 0].sharey(axes[0, 1])
    axes[1, 0].sharey(axes[1, 1])

    title = f"program shape by size-quartile — {output_dir}"
    if window > 1:
        title += f"  (rolling mean, window={window})"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def suffixed(out_path: str, suffix: str) -> str:
    p = Path(out_path)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="Directory containing rl_metrics.jsonl")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Rolling-mean window over iterations (default 1 = no smoothing)")
    ap.add_argument("--out", default=None,
                    help="Base filename; writes <stem>_stats.<ext> and <stem>_quartiles.<ext>")
    args = ap.parse_args()

    path = Path(args.output_dir) / "rl_metrics.jsonl"
    rows = load_metrics(path)
    if not rows:
        print(f"No metrics found in {path}")
        return

    print_stats_summary(rows)
    print_quartile_summary(rows)

    iters = [r['iteration'] for r in rows]
    stats_fig = build_stats_figure(rows, iters, args.smooth, args.output_dir)
    quart_fig = build_quartile_figure(rows, iters, args.smooth, args.output_dir)

    if args.out:
        stats_path = suffixed(args.out, "stats")
        quart_path = suffixed(args.out, "quartiles")
        stats_fig.savefig(stats_path, dpi=150)
        quart_fig.savefig(quart_path, dpi=150)
        print(f"\nSaved {stats_path}")
        print(f"Saved {quart_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
