"""Re-render the study_v1 plots that lack legible legends/labels.

Reads the saved CSVs/parquets in outputs/analysis/study_v1/ and writes
improved figures alongside the originals (suffixed ``_v2``). Standalone — does
not import the analysis package, so it can be run without GPU/checkpoints.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("outputs/analysis/study_v1")
PALETTE = {
    "humans": "#222222",
    "mpl": "#d62728",
    "fleet": "#1f77b4",
    "tx_enum_in_weight": "#2ca02c",
    "tx_enum_easy_shuf": "#ff7f0e",
    "tx_enrl_in_weight": "#9467bd",
    "tx_enrl_easy_shuf": "#8c564b",
    "surface_features": "#7f7f7f",
}


def colour(name: str) -> str:
    return PALETTE.get(name, "#7f7f7f")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.with_suffix('.png')}")


def top_features_per_cluster(cf: pd.DataFrame, method: str, k: int = 3) -> dict[int, list[str]]:
    """Return the top-k positively-enriched, q<0.05 features per cluster id."""
    sub = cf[(cf["method"] == method) & (cf["q"] < 0.05) & (cf["enrichment"] > 0)].copy()
    out: dict[int, list[str]] = {}
    for c, g in sub.groupby("cluster"):
        feats = g.sort_values("enrichment", ascending=False)["feature"].head(k).tolist()
        out[int(c)] = feats
    return out


# ---------------------------------------------------------------------------
# 1) Clustering scatter plots: discrete legend + cluster centroid annotations
# ---------------------------------------------------------------------------

def replot_cluster_scatters() -> None:
    print("[clustering] scatter plots")
    reductions = pd.read_parquet(ROOT / "clustering" / "reductions.parquet")
    clusters = pd.read_parquet(ROOT / "clustering" / "clusters.parquet")
    cf = pd.read_csv(ROOT / "clustering" / "cluster_features.csv")

    methods = clusters["method"].unique().tolist()
    reducers = reductions["reducer"].unique().tolist()
    cmap = plt.get_cmap("tab10")

    for method in methods:
        labels = top_features_per_cluster(cf, method, k=2)
        sub_clusters = clusters[clusters["method"] == method].set_index("function")
        n_clusters = sub_clusters["cluster"].nunique()
        sil = float(sub_clusters["silhouette"].iloc[0])
        for reducer in reducers:
            red = reductions[(reductions["method"] == method) & (reductions["reducer"] == reducer)]
            red = red.set_index("function").loc[sub_clusters.index]
            xs, ys = red["x"].values, red["y"].values
            cluster_ids = sub_clusters["cluster"].values

            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for cid in sorted(set(cluster_ids)):
                mask = cluster_ids == cid
                feats = labels.get(int(cid), [])
                lab = f"c{cid} (n={mask.sum()})"
                if feats:
                    lab += ": " + ", ".join(feats)
                ax.scatter(
                    xs[mask], ys[mask],
                    color=cmap(cid % 10), s=42, alpha=0.85,
                    edgecolors="white", linewidth=0.5, label=lab,
                )
                # Annotate centroid with cluster id only (label set in legend).
                cx, cy = float(xs[mask].mean()), float(ys[mask].mean())
                ax.annotate(
                    f"c{cid}", (cx, cy),
                    fontsize=11, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=cmap(cid % 10), lw=1.5),
                )
            ax.set_xlabel(f"{reducer}-1")
            ax.set_ylabel(f"{reducer}-2")
            ax.legend(loc="best", fontsize=7, frameon=True, framealpha=0.85)
            save(fig, ROOT / "clustering" / f"scatter_{method}_{reducer}_v2")


# ---------------------------------------------------------------------------
# 2) Clustering feature-enrichment heatmap: keep only significant features
# ---------------------------------------------------------------------------

def replot_cluster_feature_heatmaps() -> None:
    print("[clustering] feature heatmaps (significant features only)")
    cf = pd.read_csv(ROOT / "clustering" / "cluster_features.csv")
    for method in cf["method"].unique():
        sub = cf[cf["method"] == method]
        sig_feats = sub[sub["q"] < 0.05]["feature"].unique().tolist()
        if not sig_feats:
            print(f"  {method}: no significant features, skipping")
            continue
        sub_sig = sub[sub["feature"].isin(sig_feats)]
        wide = sub_sig.pivot(index="cluster", columns="feature", values="enrichment")
        sig_mat = sub_sig.pivot(index="cluster", columns="feature", values="q") < 0.05
        # Order columns by max |enrichment| across clusters
        col_order = wide.abs().max(axis=0).sort_values(ascending=False).index.tolist()
        wide = wide[col_order]
        sig_mat = sig_mat[col_order]

        fig_w = max(10, 0.55 * wide.shape[1] + 3.0)
        fig_h = max(3.5, 0.75 * wide.shape[0] + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        vmax = max(0.3, float(wide.abs().to_numpy().max()))
        im = ax.imshow(wide.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(wide.shape[1]))
        ax.set_xticklabels(wide.columns, rotation=55, ha="right", fontsize=10)
        ax.set_yticks(range(wide.shape[0]))
        ax.set_yticklabels([f"c{c}" for c in wide.index], fontsize=11)
        # Annotate enrichment values; star = q<0.05
        for i in range(wide.shape[0]):
            for j in range(wide.shape[1]):
                v = wide.values[i, j]
                if np.isnan(v):
                    continue
                star = "*" if sig_mat.values[i, j] else ""
                ax.text(
                    j, i, f"{v:+.2f}{star}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > vmax * 0.6 else "black",
                )
        cb = fig.colorbar(im, ax=ax, label="P(feature=T | cluster) − P(feature=T)")
        cb.ax.tick_params(labelsize=9)
        save(fig, ROOT / "clustering" / f"cluster_features_{method}_v2")


# ---------------------------------------------------------------------------
# 3) Failure-modes heatmap: add row dendrogram-like ordering + annotate cells
# ---------------------------------------------------------------------------

def replot_failure_mode_heatmap() -> None:
    print("[failure_modes] heatmap (annotated)")
    df = pd.read_parquet(ROOT / "failure_modes" / "result.parquet")
    chi2 = pd.read_csv(ROOT / "failure_modes" / "chi2_fdr.csv")
    wide_t = df[df["value"] == True].pivot(index="method", columns="feature", values="mean")  # noqa: E712
    wide_f = df[df["value"] == False].pivot(index="method", columns="feature", values="mean")  # noqa: E712
    gap = (wide_t - wide_f).fillna(0.0)
    n_t = df[df["value"] == True].pivot(index="method", columns="feature", values="n")  # noqa: E712

    # Order rows: humans/mpl/fleet first, then transformers
    pinned = [m for m in ["humans", "mpl", "fleet"] if m in gap.index]
    rest = [m for m in gap.index if m not in pinned]
    gap = gap.loc[pinned + rest]
    n_t = n_t.loc[pinned + rest]

    sig_lookup = {(r["method"], r["feature"]): r["q"] for _, r in chi2.iterrows()}

    fig, ax = plt.subplots(figsize=(0.6 * gap.shape[1] + 3, 0.45 * gap.shape[0] + 2))
    vmax = 0.5
    im = ax.imshow(gap.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(gap.shape[1]))
    ax.set_xticklabels([f"{c}\n(n={int(n_t.iloc[0][c])})" for c in gap.columns],
                       rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(gap.shape[0]))
    ax.set_yticklabels(gap.index)
    for i, m in enumerate(gap.index):
        for j, f in enumerate(gap.columns):
            v = gap.values[i, j]
            q = sig_lookup.get((m, f), 1.0)
            star = "*" if q < 0.05 else ""
            ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.3 else "black")
    fig.colorbar(im, ax=ax, label="acc(feat=T) − acc(feat=F)  (* = χ² q<0.05)")
    save(fig, ROOT / "failure_modes" / "heatmap_v2")


# ---------------------------------------------------------------------------
# 4) Probing heatmap with significance annotations
# ---------------------------------------------------------------------------

def replot_probing_heatmap() -> None:
    print("[probing] heatmap with permutation p-values")
    auroc = pd.read_parquet(ROOT / "probing" / "auroc.parquet")
    null = pd.read_parquet(ROOT / "probing" / "null.parquet")
    real_mean = auroc.groupby(["method", "primitive"])["auroc"].mean().reset_index()
    null_dist = null.groupby(["method", "primitive"])["auroc"].agg(list).reset_index()
    merged = real_mean.merge(null_dist, on=["method", "primitive"], suffixes=("_real", "_null"))
    merged["p_perm"] = merged.apply(
        lambda r: (sum(np.array(r["auroc_null"]) >= r["auroc_real"]) + 1)
        / (len(r["auroc_null"]) + 1),
        axis=1,
    )
    pmat = merged.pivot(index="method", columns="primitive", values="p_perm")
    wide = real_mean.pivot(index="method", columns="primitive", values="auroc")

    # Pin surface_features at the bottom as the baseline reference.
    other = [m for m in wide.index if m != "surface_features"]
    order = other + (["surface_features"] if "surface_features" in wide.index else [])
    wide = wide.loc[order]
    pmat = pmat.loc[order]

    fig, ax = plt.subplots(figsize=(0.9 * wide.shape[1] + 3, 0.6 * wide.shape[0] + 2))
    im = ax.imshow(wide.values, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(wide.shape[1]))
    ax.set_xticklabels(wide.columns, rotation=30, ha="right")
    ax.set_yticks(range(wide.shape[0]))
    ax.set_yticklabels(wide.index)
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            v = wide.values[i, j]
            if np.isnan(v):
                continue
            p = pmat.values[i, j] if not np.isnan(pmat.values[i, j]) else 1.0
            sig = "*" if p < 0.05 else ""
            ax.text(j, i, f"{v:.2f}{sig}", ha="center", va="center", fontsize=9,
                    color="white" if v < 0.7 else "black")
    fig.colorbar(im, ax=ax, label="AUROC  (* = permutation p<0.05)")
    save(fig, ROOT / "probing" / "heatmap_v2")


if __name__ == "__main__":
    replot_cluster_scatters()
    replot_cluster_feature_heatmaps()
    replot_failure_mode_heatmap()
    replot_probing_heatmap()
    print("\nDone. Files written next to originals with `_v2` suffix.")
