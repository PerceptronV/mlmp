#!/usr/bin/env python
"""Holistic hyperparameter analysis across all runs in a wandb sweep.

Rather than reporting only the single best run (which can be a lucky seed), this
aggregates performance by hyperparameter VALUE so you can see which settings are
robustly good across the rest of the search.

HYPERBAND CAVEAT: trials are killed at different epochs, so their final
best_val_accuracy reflects different training budgets and is NOT directly
comparable. By default we restrict the aggregation to "full-length" runs (those
that reached >= --min-epoch-frac of the longest run in the sweep) so the
comparison is fair. Pass --all to aggregate over every run regardless of budget
(useful to see which values start strong, but biased against early-killed ones).

Usage:
    python scripts/sweep_analyze.py <sweep_id> [--all] [--min-epoch-frac 0.9] [--top 5]

<sweep_id> may be bare (li7is16c) or a full path (entity/project/li7is16c).
"""
import argparse
import math
import sys

import pandas as pd
import wandb

DEFAULT_ENTITY = "yiding-song-vincent"
DEFAULT_PROJECT = "mlmp"
DISCRETE_KEYS = ["weight_decay", "grad_clip", "batch_size", "constant_lr"]
METRIC = "best_val_accuracy"

pd.set_option("display.width", 100)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def resolve(sweep_id: str) -> str:
    if sweep_id.count("/") == 2:
        return sweep_id
    return f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}/{sweep_id}"


def load_runs(sweep) -> pd.DataFrame:
    rows = []
    for run in sweep.runs:
        cfg, summ = run.config, run.summary
        acc = summ.get(METRIC)
        if acc is None:
            continue  # never logged a val accuracy (e.g. died in epoch 0)
        rows.append({
            "name": run.name,
            "epoch": summ.get("epoch"),
            METRIC: acc,
            "lr": cfg.get("lr"),
            "weight_decay": cfg.get("weight_decay"),
            "grad_clip": cfg.get("grad_clip"),
            "batch_size": cfg.get("batch_size"),
            "constant_lr": cfg.get("constant_lr"),
        })
    return pd.DataFrame(rows)


def marginal(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-value aggregate of the metric, sorted by mean (robustness) desc."""
    g = df.groupby(key)[METRIC].agg(["count", "mean", "median", "max"])
    return g.sort_values("mean", ascending=False)


def lr_bins(df: pd.DataFrame, nbins: int = 4) -> pd.DataFrame:
    """lr is continuous, so bin it in log10 space before aggregating."""
    s = df.dropna(subset=["lr"]).copy()
    if s.empty:
        return pd.DataFrame()
    s["lr_log10_bin"] = pd.cut(s["lr"].apply(math.log10), bins=nbins)
    return s.groupby("lr_log10_bin", observed=True)[METRIC].agg(
        ["count", "mean", "median", "max"])


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_id")
    ap.add_argument("--all", action="store_true",
                    help="aggregate over all runs, not just full-length ones")
    ap.add_argument("--min-epoch-frac", type=float, default=0.9,
                    help="keep runs reaching >= this fraction of the longest run")
    ap.add_argument("--top", type=int, default=5, help="show this many top runs")
    args = ap.parse_args(argv)

    sweep = wandb.Api().sweep(resolve(args.sweep_id))
    df = load_runs(sweep)
    if df.empty:
        print("No runs with a logged best_val_accuracy yet.")
        return 1

    longest = df["epoch"].max()
    if args.all:
        sub = df
        scope = f"all {len(df)} runs (budget-biased)"
    else:
        cutoff = args.min_epoch_frac * longest
        sub = df[df["epoch"] >= cutoff]
        scope = (f"{len(sub)}/{len(df)} full-length runs "
                 f"(epoch >= {cutoff:.0f} of {longest:.0f})")

    print(f"\nSweep {sweep.id}  (mode={sweep.runs[0].config.get('mode', '?')})")
    print(f"Aggregating over: {scope}\n")
    if len(sub) < 4 and not args.all:
        print("  ! Few full-length runs — marginals are thin. Re-run with --all\n"
              "    to see equal-rung trends, or run more trials.\n")

    for key in DISCRETE_KEYS:
        if sub[key].notna().any():
            print(f"--- {key} ---")
            print(marginal(sub, key).to_string())
            print()

    lr_tbl = lr_bins(sub)
    if not lr_tbl.empty:
        print("--- lr (log10 bins) ---")
        print(lr_tbl.to_string())
        print()

    print(f"--- top {args.top} runs by {METRIC} ---")
    cols = ["name", "epoch", METRIC, "lr", "weight_decay",
            "grad_clip", "batch_size", "constant_lr"]
    print(df.sort_values(METRIC, ascending=False).head(args.top)[cols].to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
