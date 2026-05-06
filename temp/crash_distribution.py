"""Sample programs from each corpus, run 100 random inputs through each, and
plot the distribution of failure counts (exceptions + invalid outputs)."""

import argparse
import json
import random
import signal
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.lang.compiler import JITCompiler
from src.lang.grammar import DefaultGrammar
from src.lang.parser import parse


N_CANDIDATES = 100
MAX_LIST_LEN = 15
MIN_ELEM = 0
MAX_ELEM = 100
PER_CALL_TIMEOUT = 0.25


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


def count_failures(program_str: str, jit: JITCompiler, seed: int) -> int | None:
    """Return number of failing inputs out of N_CANDIDATES, or None if program
    fails to compile."""
    try:
        fn, _ = jit.compile(parse(program_str))
    except Exception:
        return None
    rng = random.Random(seed)
    fails = 0
    for _ in range(N_CANDIDATES):
        L = rng.randint(0, MAX_LIST_LEN)
        inp = [rng.randint(MIN_ELEM, MAX_ELEM) for _ in range(L)]
        try:
            with _alarm(PER_CALL_TIMEOUT):
                out = fn(inp)
            if (not isinstance(out, list)
                    or len(out) > MAX_LIST_LEN
                    or not all(isinstance(x, int) for x in out)):
                fails += 1
        except Exception:
            fails += 1
    return fails


def collect(corpus_path: Path, n_sample: int, seed: int) -> list[int]:
    with open(corpus_path) as f:
        entries = json.load(f)
    entries = [e for e in entries if e.get("type") == "list[int]"]
    rng = random.Random(seed)
    rng.shuffle(entries)
    entries = entries[:n_sample]

    jit = JITCompiler(DefaultGrammar)
    fails: list[int] = []
    compile_errors = 0
    for i, e in enumerate(tqdm(entries, desc=corpus_path.name)):
        f = count_failures(e["program"], jit, seed=seed * 1_000_003 + i)
        if f is None:
            compile_errors += 1
            continue
        fails.append(f)
    print(f"  {corpus_path.name}: {len(fails)} programs, {compile_errors} compile errors")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum", default="datasets/corpus-a/enum_corpus_no_rule.json")
    ap.add_argument("--rl", default="datasets/corpus-a/rl_corpus_no_rule.simplified.json")
    ap.add_argument("--n-sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scripts/crash_distribution.png")
    args = ap.parse_args()

    enum_fails = collect(Path(args.enum), args.n_sample, args.seed)
    rl_fails = collect(Path(args.rl), args.n_sample, args.seed + 1)

    # Bin into deciles 0..100 step 10
    bin_edges = np.arange(0, 101, 10)
    enum_hist, _ = np.histogram(enum_fails, bins=bin_edges)
    rl_hist, _ = np.histogram(rl_fails, bins=bin_edges)

    enum_pct = 100 * enum_hist / max(len(enum_fails), 1)
    rl_pct = 100 * rl_hist / max(len(rl_fails), 1)

    bin_labels = [f"{lo}–{lo+9}" for lo in bin_edges[:-1]]
    x = np.arange(len(bin_labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, enum_pct, width, label=f"enum (n={len(enum_fails)})", color="#3b82f6")
    ax.bar(x + width/2, rl_pct, width, label=f"rl (n={len(rl_fails)})", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(f"# crashes / invalid outputs out of {N_CANDIDATES} random inputs")
    ax.set_ylabel("% of sampled programs")
    ax.set_title("Per-program failure rate on random inputs")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Quick text summary
    for name, fails in [("enum", enum_fails), ("rl", rl_fails)]:
        arr = np.array(fails)
        print(f"\n{name}: mean={arr.mean():.1f}  median={np.median(arr):.0f}  "
              f"p25={np.percentile(arr, 25):.0f}  p75={np.percentile(arr, 75):.0f}  "
              f"frac_with_>=90_failures={(arr >= 90).mean():.1%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    main()
