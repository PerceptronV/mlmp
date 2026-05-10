"""Plot the distribution of valid I/O pair counts returned by RuleIOSampler
for sampled programs from each corpus."""

import argparse
import json
import random
import signal
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data.sampler import RuleIOSampler


@contextmanager
def _alarm(seconds: float):
    def _handler(_s, _f):
        raise TimeoutError()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def collect(corpus_path: Path, n_sample: int, seed: int, n_io_per_program: int) -> list[int]:
    with open(corpus_path) as f:
        entries = json.load(f)
    entries = [e for e in entries if e.get("type") == "list[int]"]
    rng = random.Random(seed)
    rng.shuffle(entries)
    entries = entries[:n_sample]

    sampler = RuleIOSampler(num_io_pairs=n_io_per_program)
    pool_sizes: list[int] = []
    for i, e in enumerate(tqdm(entries, desc=corpus_path.name)):
        prog_rng = random.Random(seed * 1_000_003 + i)
        try:
            with _alarm(30.0):
                pairs = sampler.sample(e["program"], prog_rng)
        except TimeoutError:
            pairs = []
        pool_sizes.append(len(pairs))
    return pool_sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum", default="datasets/corpus-a/enum_corpus_no_rule.json")
    ap.add_argument("--rl", default="datasets/corpus-a/rl_corpus_no_rule.simplified.json")
    ap.add_argument("--n-sample", type=int, default=2000)
    ap.add_argument("--n-io-per-program", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scripts/io_pool_distribution.png")
    args = ap.parse_args()

    enum_pool = collect(Path(args.enum), args.n_sample, args.seed, args.n_io_per_program)
    rl_pool = collect(Path(args.rl), args.n_sample, args.seed + 1, args.n_io_per_program)

    max_n = args.n_io_per_program
    bins = np.arange(0, max_n + 2)  # 0,1,...,max_n -> max_n+1 bins
    enum_hist, _ = np.histogram(enum_pool, bins=bins)
    rl_hist, _ = np.histogram(rl_pool, bins=bins)

    enum_pct = 100 * enum_hist / max(len(enum_pool), 1)
    rl_pct = 100 * rl_hist / max(len(rl_pool), 1)

    x = np.arange(max_n + 1)
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, enum_pct, width, label=f"enum (n={len(enum_pool)})", color="#3b82f6")
    ax.bar(x + width/2, rl_pct, width, label=f"rl (n={len(rl_pool)})", color="#ef4444")
    ax.axvline(4.5, color="grey", linestyle="--", alpha=0.7,
               label="min_n_io_shown=5 floor")
    ax.set_xticks(x)
    ax.set_xlabel(f"# valid I/O pairs returned by RuleIOSampler (cap = {max_n})")
    ax.set_ylabel("% of sampled programs")
    ax.set_title("Distribution of I/O pool size per program")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for name, pool in [("enum", enum_pool), ("rl", rl_pool)]:
        arr = np.array(pool)
        print(f"\n{name}: mean={arr.mean():.2f}  median={np.median(arr):.0f}  "
              f"p25={np.percentile(arr, 25):.0f}  p75={np.percentile(arr, 75):.0f}  "
              f"frac_<5={(arr < 5).mean():.1%}  "
              f"frac_<11={(arr < max_n).mean():.1%}  "
              f"frac_=={max_n}={(arr == max_n).mean():.1%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    main()
