"""Clustering analysis (§7.3).

Per-method per-task encoder embedding → reduction (PCA / t-SNE / UMAP if
available) → clustering (k-means with k chosen by silhouette over
``k_search``; hierarchical Ward as a robustness check).

Cross-method outputs:
- ARI / NMI heatmap across method pairs.
- Hungarian-algorithm cluster-id matching for an interpretable correspondence.
- Cluster characterisation: most-enriched binary feature per cluster (χ² with
  FDR), sparse multinomial logistic regression of cluster id on feature
  columns for inferential signatures.

Defaults: ``n_io_shown=11, order=1`` per task (set by ``TransformerMethod``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm

from ..capability import Capability, CapabilityMissing
from ..plotting import apply_rc, save_fig
from ..stats import ari_nmi, chi2_p, fdr_bh
from .base import Analysis, AnalysisResult

if TYPE_CHECKING:
    from ..cache import Cache
    from ..methods.base import Method
    from ..task import TaskBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature subsets for cluster characterisation plots.
#
# ``cluster_features.csv`` contains every boolean column from ``functions.csv``
# (~100 features), mixing three families: program-syntactic DSL tokens, Rule's
# high-level cognitive tags, and IO-pattern features (``feature_*`` prefixed).
# For interpretation it's useful to slice down to a coherent family at plot
# time. We don't drop the all-features versions — we add per-subset variants:
#
#   - ``cognitive``: the 13 high-level cognitive tags Rule's paper foregrounds.
#   - ``syntactic``: the ~60 DSL primitive tokens (``T`` if the canonical
#     program uses that token).
#
# When a subset is used, FDR is recomputed within method × subset so q-values
# reflect the visible multiple-testing family rather than the full ~100-feature
# pool.
# ---------------------------------------------------------------------------

COGNITIVE_FEATURES: list[str] = [
    "recursive", "higher", "conditional", "arithmetic",
    "mapping", "filtering", "indexing", "indexing_not_first",
    "unfolding", "counting", "uniqueness", "list_constants", "numbers",
]

SYNTACTIC_FEATURES: list[str] = [
    "true", "false", "not", "if", "and", "or",
    "+", "-", "*", "/", "<", ">", "==", "%", "is_even", "is_odd",
    "empty", "cons", "singleton", "fold", "map", "filter", "zip",
    "first", "nth", "second", "third", "length", "last",
    "concat", "append", "count", "cut_vals", "is_in", "flatten",
    "max", "min", "product", "reverse", "sum", "unique", "range",
    "repeat", "foldi", "mapi", "filteri", "insert", "replace",
    "cut_idx", "swap", "cut_slice", "slice", "drop", "take",
    "droplast", "takelast", "splice", "find", "cut_val", "group", "sort",
]

FEATURE_SUBSETS: dict[str, list[str]] = {
    "cognitive": COGNITIVE_FEATURES,
    "syntactic": SYNTACTIC_FEATURES,
}


def _restrict_cf(cf: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Filter cluster_features to a feature subset and re-FDR within method.

    The original ``q`` values are FDR-corrected across the full feature pool
    (~100 features). When showing only a subset we want q-values that match
    the visible family — otherwise the multiple-testing correction is
    misleadingly conservative.
    """
    sub = cf[cf["feature"].isin(features)].copy()
    if sub.empty:
        return sub
    for m in sub["method"].unique():
        mask = sub["method"] == m
        sub.loc[mask, "q"] = fdr_bh(sub.loc[mask, "p"].values)
    return sub


@dataclass
class ClusteringResult(AnalysisResult):
    embeddings: dict[str, np.ndarray]              # method -> (n_tasks, d)
    task_ids: list[str]
    reductions: dict[tuple[str, str], np.ndarray]  # (method, reducer) -> (n_tasks, 2)
    clusters: pd.DataFrame                         # (method, function, cluster, *features)
    pairwise: pd.DataFrame                         # (method_a, method_b, ARI, NMI)
    cluster_features: pd.DataFrame                 # (method, cluster, feature, p, q, enrichment)

    def save(self, outdir: Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            outdir / "embeddings.npz",
            **{m: v for m, v in self.embeddings.items()},
            task_ids=np.array(self.task_ids),
        )
        # Reductions: one parquet long-form file.
        rec = []
        for (method, reducer), arr in self.reductions.items():
            for tid, xy in zip(self.task_ids, arr):
                rec.append({"method": method, "reducer": reducer, "function": tid,
                            "x": float(xy[0]), "y": float(xy[1])})
        pd.DataFrame(rec).to_parquet(outdir / "reductions.parquet", index=False)
        self.clusters.to_parquet(outdir / "clusters.parquet", index=False)
        self.pairwise.to_csv(outdir / "ari_nmi.csv", index=False)
        self.cluster_features.to_csv(outdir / "cluster_features.csv", index=False)

    def plot(self, outdir: Path) -> None:
        apply_rc()
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # ARI heatmap is feature-independent — generate once.
        self._plot_ari_heatmap(outdir, plt)

        # Feature-dependent plots (scatter legends + per-method heatmap)
        # are produced once for the full feature pool and once per named
        # subset, so an analyst can compare "what defines this cluster"
        # under three different feature vocabularies.
        subsets: list[tuple[str, pd.DataFrame]] = [("", self.cluster_features)]
        for name, feats in FEATURE_SUBSETS.items():
            subsets.append((f"_{name}", _restrict_cf(self.cluster_features, feats)))
        for tag, cf in subsets:
            self._plot_scatters(outdir, cf, tag, plt)
            self._plot_per_method_heatmaps(outdir, cf, tag, plt)

        # Cluster portraits: small horizontal-bar cards showing top-N
        # enriched or deriched features per (method, cluster). Generated
        # for each named subset, in both directions.
        for name, feats in FEATURE_SUBSETS.items():
            cf = _restrict_cf(self.cluster_features, feats)
            if cf.empty:
                continue
            for direction in ("enriched", "deriched"):
                self._plot_portraits(outdir, cf, name, direction, plt)

    # ------------------------------------------------------------------ #
    # Private plotting helpers — each writes to ``outdir``.               #
    # ``plt`` is passed in to avoid re-importing matplotlib per call and  #
    # to keep this module torch/matplotlib-free at import time.           #
    # ------------------------------------------------------------------ #

    def _plot_ari_heatmap(self, outdir: Path, plt) -> None:
        if self.pairwise.empty:
            return
        methods = sorted({*self.pairwise["method_a"], *self.pairwise["method_b"]})
        mat = np.full((len(methods), len(methods)), np.nan)
        for _, row in self.pairwise.iterrows():
            i = methods.index(row["method_a"])
            j = methods.index(row["method_b"])
            mat[i, j] = mat[j, i] = row["ARI"]
        np.fill_diagonal(mat, 1.0)
        fig, ax = plt.subplots(figsize=(0.6 * len(methods) + 2, 0.6 * len(methods) + 2))
        im = ax.imshow(mat, cmap="viridis", vmin=-0.1, vmax=1.0)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods)
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] < 0.5 else "black")
        fig.colorbar(im, ax=ax, label="ARI")
        save_fig(fig, outdir, "ari_heatmap.pdf")

    def _plot_scatters(self, outdir: Path, cf: pd.DataFrame, tag: str, plt) -> None:
        # Per-method, per-cluster top-2 positively-enriched features (q<0.05).
        # Drives the scatter legend; empty when ``cf`` has no significant
        # positive enrichment in that cluster (e.g. a narrow feature subset).
        top_feats: dict[tuple[str, int], list[str]] = {}
        if not cf.empty:
            sig_pos = cf[(cf["q"] < 0.05) & (cf["enrichment"] > 0)]
            for (m, c), g in sig_pos.groupby(["method", "cluster"]):
                top_feats[(m, int(c))] = (
                    g.sort_values("enrichment", ascending=False)["feature"].head(2).tolist()
                )

        cmap = plt.get_cmap("tab10")
        for (method, reducer), arr in self.reductions.items():
            sub = self.clusters[self.clusters["method"] == method].set_index("function").loc[self.task_ids]
            cluster_ids = sub["cluster"].values
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            for cid in sorted(set(cluster_ids)):
                mask = cluster_ids == cid
                feats = top_feats.get((method, int(cid)), [])
                lab = f"c{cid} (n={int(mask.sum())})"
                if feats:
                    lab += ": " + ", ".join(feats)
                ax.scatter(
                    arr[mask, 0], arr[mask, 1],
                    color=cmap(int(cid) % 10), s=42, alpha=0.85,
                    edgecolors="white", linewidth=0.5, label=lab,
                )
                cx = float(arr[mask, 0].mean())
                cy = float(arr[mask, 1].mean())
                ax.annotate(
                    f"c{cid}", (cx, cy),
                    fontsize=11, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white",
                              ec=cmap(int(cid) % 10), lw=1.5),
                )
            ax.set_xlabel(f"{reducer}-1")
            ax.set_ylabel(f"{reducer}-2")
            ax.legend(loc="best", fontsize=7, frameon=True, framealpha=0.85)
            save_fig(fig, outdir, f"scatter_{method}_{reducer}{tag}.pdf")

    def _plot_per_method_heatmaps(
        self, outdir: Path, cf: pd.DataFrame, tag: str, plt,
    ) -> None:
        # Cluster-feature enrichment heatmap, per method. Filter to features
        # that pass q<0.05 in some cluster of that method — without filtering
        # the table is 100+ columns wide and mostly flat.
        if cf.empty:
            return
        for method in cf["method"].unique():
            sub = cf[cf["method"] == method]
            sig_feats = sub[sub["q"] < 0.05]["feature"].unique().tolist()
            if not sig_feats:
                continue
            sub_sig = sub[sub["feature"].isin(sig_feats)]
            wide = sub_sig.pivot(index="cluster", columns="feature", values="enrichment")
            sig_mat = sub_sig.pivot(index="cluster", columns="feature", values="q") < 0.05
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
            cb = fig.colorbar(im, ax=ax, label="P(feat=T | cluster) − P(feat=T)")
            cb.ax.tick_params(labelsize=9)
            save_fig(fig, outdir, f"cluster_features_{method}{tag}.pdf")

    def _plot_portraits(
        self,
        outdir: Path,
        cf: pd.DataFrame,
        subset_name: str,
        direction: str,
        plt,
    ) -> None:
        """Cluster-portrait small-multiples: top features per (method, cluster).

        ``direction="enriched"`` shows top-N positive enrichment (red bars);
        ``direction="deriched"`` shows the most negative (blue bars). Bar
        length is ``|enrichment|`` in both cases. Methods are rows; clusters
        within a row are sorted by total magnitude of the shown bars.
        """
        if cf.empty or direction not in ("enriched", "deriched"):
            return
        top_n = 5
        q_thr = 0.10  # show features with q<q_thr; star those with q<0.05

        def top_feats(method: str, cluster: int) -> pd.DataFrame:
            sub = cf[(cf["method"] == method) & (cf["cluster"] == cluster)]
            if direction == "enriched":
                sub = sub[(sub["enrichment"] > 0) & (sub["q"] < q_thr)]
                return sub.sort_values("enrichment", ascending=False).head(top_n)
            sub = sub[(sub["enrichment"] < 0) & (sub["q"] < q_thr)]
            return sub.sort_values("enrichment", ascending=True).head(top_n)

        method_order = sorted(cf["method"].unique().tolist())
        cluster_order: dict[str, list[int]] = {}
        for m in method_order:
            clusters = sorted(cf[cf["method"] == m]["cluster"].unique().tolist())
            scored: list[tuple[int, float]] = []
            for c in clusters:
                tops = top_feats(m, c)
                scored.append((c, float(tops["enrichment"].abs().sum()) if len(tops) else 0.0))
            cluster_order[m] = [c for c, _ in sorted(scored, key=lambda x: -x[1])]

        if not method_order:
            return
        max_k = max((len(cs) for cs in cluster_order.values()), default=0)
        if max_k == 0:
            return

        n_rows = len(method_order)
        n_cols = max_k
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(2.5 * n_cols, 2.2 * n_rows),
            squeeze=False,
        )
        vmax = 0.6
        cmap = plt.cm.Reds if direction == "enriched" else plt.cm.Blues
        for i, m in enumerate(method_order):
            for j in range(n_cols):
                ax = axes[i][j]
                if j >= len(cluster_order[m]):
                    ax.axis("off")
                    continue
                c = cluster_order[m][j]
                tops = top_feats(m, c)
                if len(tops) == 0:
                    ax.axis("off")
                    ax.set_title(f"c{c}\n(no {direction} features)", fontsize=9)
                    continue
                y_pos = np.arange(len(tops))[::-1]
                mags = tops["enrichment"].abs().values
                colors = [cmap(min(1.0, x / vmax + 0.2)) for x in mags]
                ax.barh(y_pos, mags, color=colors, edgecolor="black", lw=0.5)
                for k, (_, row) in enumerate(tops.iterrows()):
                    star = "*" if row["q"] < 0.05 else ""
                    ax.text(
                        abs(row["enrichment"]) + 0.01, y_pos[k],
                        f"{row['feature']}{star}",
                        va="center", fontsize=8,
                    )
                ax.set_yticks([])
                ax.set_xlim(0, vmax)
                ax.set_xticks([0, 0.2, 0.4, 0.6])
                ax.set_xticklabels(["0", ".2", ".4", ".6"], fontsize=7)
                ax.set_title(f"c{c}", fontsize=10, fontweight="bold")
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)

        # Reserve left margin so per-row method labels (added below via
        # fig.text after tight_layout) don't overlap the leftmost subplots.
        fig.tight_layout(rect=(0.10, 0.0, 1.0, 1.0))
        for i, m in enumerate(method_order):
            pos = axes[i][0].get_position()
            fig.text(
                0.09, pos.y0 + pos.height / 2,
                m,
                fontsize=11, fontweight="bold",
                ha="right", va="center",
            )
        save_fig(fig, outdir, f"cluster_portraits_{subset_name}_{direction}.pdf")


@dataclass
class ClusteringAnalysis(Analysis):
    kind: str = "clustering"
    needs_embeddings: bool = True
    reducer: list[str] = field(default_factory=lambda: ["pca", "tsne"])
    cluster: str = "kmeans"
    k_search: list[int] = field(default_factory=lambda: [4, 5, 6, 7, 8, 9, 10])
    color_by: list[str] = field(default_factory=lambda: ["cluster"])
    replicates: bool = False
    level: str = "task"
    n_io_shown: int = 11
    order: int = 1
    random_state: int = 0

    def run(
        self,
        methods: list["Method"],
        bundle: "TaskBundle",
        cache: "Cache",
    ) -> ClusteringResult:
        from sklearn.cluster import KMeans  # type: ignore[import-untyped]
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]
        from sklearn.manifold import TSNE  # type: ignore[import-untyped]
        from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]

        # Methods may not support EMBEDDINGS — emit one warning per skipped method.
        ok_methods: list[Method] = []
        for m in methods:
            if not m.supports(Capability.EMBEDDINGS):
                logger.warning("[%s] does not support EMBEDDINGS — skipping in clustering", m.name)
                continue
            ok_methods.append(m)
        if not ok_methods:
            logger.warning("No methods support EMBEDDINGS — clustering returns an empty result")
            return ClusteringResult(
                embeddings={}, task_ids=[], reductions={},
                clusters=pd.DataFrame(columns=["method", "function", "cluster", "k_chosen", "silhouette"]),
                pairwise=pd.DataFrame(columns=["method_a", "method_b", "ARI", "NMI"]),
                cluster_features=pd.DataFrame(columns=["method", "cluster", "feature", "enrichment", "p", "q"]),
            )

        task_ids = bundle.task_ids
        # Pick the trial whose observed + (query, expected) gives exactly
        # ``n_io_shown`` pairs — i.e. trial=n_io_shown (observed has n-1 pairs,
        # plus query = n total). Earlier this used ``n_io_shown + 1`` which
        # would only ever match trial=12 (which doesn't exist), silently
        # falling back to the first trial in the order — embedding at 1 IO
        # pair instead of 11.
        target_trial_n = self.n_io_shown
        trial_for: dict[str, "object"] = {}
        fallback_tasks: list[str] = []
        for task in bundle:
            tr = next(
                (t for t in task.trials if t.order == self.order and t.trial == target_trial_n),
                None,
            )
            if tr is None:
                tr = next((t for t in task.trials if t.order == self.order), None)
                if tr is not None:
                    fallback_tasks.append(f"{task.task_id}(no t={target_trial_n})")
            if tr is None and task.trials:
                tr = task.trials[0]
                fallback_tasks.append(f"{task.task_id}(no order={self.order})")
            if tr is None:
                raise RuntimeError(
                    f"ClusteringAnalysis: task {task.task_id!r} has no trials"
                )
            trial_for[task.task_id] = tr
        if fallback_tasks:
            logger.warning(
                "ClusteringAnalysis: %d task(s) used a fallback trial: %s",
                len(fallback_tasks), ", ".join(fallback_tasks[:8]),
            )

        embeddings: dict[str, np.ndarray] = {}
        for method in ok_methods:
            vecs = []
            needs_compute = any(
                not cache.has_embedding(method, tid, self.order) for tid in task_ids
            )
            tids_iter = task_ids
            if needs_compute:
                tids_iter = tqdm(task_ids, desc=f"clustering embed:{method.name}", leave=True)
            for tid in tids_iter:
                tr = trial_for[tid]
                v = cache.get_or_compute_embedding(method, tid, self.order, lambda m=method, t=tr: m.embed(t))
                vecs.append(v)
            embeddings[method.name] = np.stack(vecs, axis=0)
        cache.flush()

        # Reductions: PCA always (used for ARI / Procrustes); t-SNE if requested.
        reductions: dict[tuple[str, str], np.ndarray] = {}
        for method_name, X in embeddings.items():
            if X.shape[0] < 3 or X.shape[1] == 0:
                continue
            pca = PCA(n_components=min(2, X.shape[1]), random_state=self.random_state).fit_transform(X)
            reductions[(method_name, "pca")] = pca
            if "tsne" in self.reducer:
                # t-SNE perplexity must be < n_samples; standard rule of thumb 5..50.
                perp = max(5, min(30, X.shape[0] // 4))
                tsne = TSNE(
                    n_components=2,
                    perplexity=perp,
                    init="pca",
                    learning_rate="auto",
                    random_state=self.random_state,
                ).fit_transform(X)
                reductions[(method_name, "tsne")] = tsne
            if "umap" in self.reducer:
                try:
                    import umap  # type: ignore[import-untyped]
                except ImportError:
                    logger.warning("umap-learn not installed — skipping UMAP reduction")
                else:
                    reductions[(method_name, "umap")] = umap.UMAP(
                        n_components=2, random_state=self.random_state
                    ).fit_transform(X)

        # k-means with silhouette-driven k. Use the embedding directly (PCA is
        # for plotting/ARI alignment, not clustering input).
        cluster_rows: list[dict] = []
        clusters_per_method: dict[str, np.ndarray] = {}
        for method_name, X in embeddings.items():
            best_k, best_score, best_labels = self.k_search[0], -np.inf, None
            for k in self.k_search:
                if k >= X.shape[0]:
                    continue
                km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit(X)
                if k < 2 or len(set(km.labels_)) < 2:
                    score = -np.inf
                else:
                    score = silhouette_score(X, km.labels_)
                if score > best_score:
                    best_k, best_score, best_labels = k, score, km.labels_
            if best_labels is None:
                # Fall back: assign all to cluster 0.
                best_labels = np.zeros(X.shape[0], dtype=int)
            clusters_per_method[method_name] = best_labels
            for tid, c in zip(task_ids, best_labels):
                cluster_rows.append({
                    "method": method_name,
                    "function": tid,
                    "cluster": int(c),
                    "k_chosen": int(best_k),
                    "silhouette": float(best_score),
                })

        # Pairwise ARI / NMI.
        pair_rows: list[dict] = []
        for a, b in combinations(clusters_per_method, 2):
            ari, nmi = ari_nmi(clusters_per_method[a], clusters_per_method[b])
            pair_rows.append({"method_a": a, "method_b": b, "ARI": ari, "NMI": nmi})

        # Cluster-feature enrichment: per (method, cluster, feature) report
        # P(feat=T | cluster) − P(feat=T) and a χ² p with FDR within method.
        feat_table = {t.task_id: t.features for t in bundle}
        feature_names = bundle.feature_names
        cf_rows: list[dict] = []
        for method_name, labels in clusters_per_method.items():
            unique = sorted(set(labels.tolist()))
            base = {f: float(np.mean([feat_table[tid].get(f, False) for tid in task_ids]))
                    for f in feature_names}
            ps_per_method: list[float] = []
            entries: list[dict] = []
            for c in unique:
                mask = labels == c
                tids_in = [tid for tid, m in zip(task_ids, mask) if m]
                tids_out = [tid for tid, m in zip(task_ids, mask) if not m]
                for f in feature_names:
                    in_T = sum(1 for t in tids_in if feat_table[t].get(f, False))
                    in_F = len(tids_in) - in_T
                    out_T = sum(1 for t in tids_out if feat_table[t].get(f, False))
                    out_F = len(tids_out) - out_T
                    p_in = in_T / max(1, len(tids_in))
                    enrichment = p_in - base[f]
                    pval = chi2_p(np.array([[in_T, in_F], [out_T, out_F]]))
                    entries.append({
                        "method": method_name, "cluster": int(c), "feature": f,
                        "enrichment": enrichment, "p": pval,
                    })
                    ps_per_method.append(pval)
            qs = fdr_bh([e["p"] for e in entries])
            for e, q in zip(entries, qs):
                e["q"] = float(q)
                cf_rows.append(e)
        cluster_features = pd.DataFrame(cf_rows)

        return ClusteringResult(
            embeddings=embeddings,
            task_ids=list(task_ids),
            reductions=reductions,
            clusters=pd.DataFrame(cluster_rows),
            pairwise=pd.DataFrame(pair_rows),
            cluster_features=cluster_features,
        )
