"""Re-emit the layer × source robustness plots offline.

The same plots are produced automatically when ``python -m src.analysis`` runs
a probing config whose method names follow the sweep convention. Use this
script only if you want to re-plot from an existing ``auroc.parquet`` without
re-running the analysis.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from src.analysis.metrics.probing import plot_layer_robustness


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to the probing run dir, e.g. outputs/analysis/probe_robustness_v1/probing")
    args = ap.parse_args()

    auroc = pd.read_parquet(args.run_dir / "auroc.parquet")
    wrote = plot_layer_robustness(auroc, args.run_dir)
    if wrote:
        print(f"Wrote robustness_*.pdf and robustness_grid.pdf to {args.run_dir}")
    else:
        print(f"No sweep methods in {args.run_dir / 'auroc.parquet'} — nothing to plot.")


if __name__ == "__main__":
    main()
