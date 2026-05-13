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

from src.analysis.metrics.probing import (
    plot_layer_heatmaps,
    plot_layer_robustness,
    plot_max_per_primitive,
)
from src.analysis.plotting import apply_rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Path to the probing run dir, e.g. outputs/analysis/probe_robustness_v1/probing")
    args = ap.parse_args()

    apply_rc()
    auroc = pd.read_parquet(args.run_dir / "auroc.parquet")
    null_path = args.run_dir / "null.parquet"
    null = pd.read_parquet(null_path) if null_path.exists() else None
    wrote_lines = plot_layer_robustness(auroc, args.run_dir)
    wrote_heat = plot_layer_heatmaps(auroc, args.run_dir)
    wrote_max = plot_max_per_primitive(auroc, args.run_dir, null=null)
    if wrote_lines or wrote_heat or wrote_max:
        print(f"Wrote robustness + heatmap + max-bars PDFs to {args.run_dir}")
    else:
        print(f"No sweep methods in {args.run_dir / 'auroc.parquet'} — nothing to plot.")


if __name__ == "__main__":
    main()
