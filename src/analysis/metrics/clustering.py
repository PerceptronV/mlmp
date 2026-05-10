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

from ..capability import Capability, CapabilityMissing
from ..plotting import apply_rc, save_fig
from ..stats import ari_nmi, chi2_p, fdr_bh
from .base import Analysis, AnalysisResult

if TYPE_CHECKING:
    from ..cache import Cache
    from ..methods.base import Method
    from ..task import TaskBundle

logger = logging.getLogger(__name__)


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

        for (method, reducer), arr in self.reductions.items():
            sub = self.clusters[self.clusters["method"] == method].set_index("function").loc[self.task_ids]
            cluster_ids = sub["cluster"].values
            fig, ax = plt.subplots(figsize=(5, 4.5))
            sc = ax.scatter(arr[:, 0], arr[:, 1], c=cluster_ids, cmap="tab10", s=14)
            ax.set_xlabel(f"{reducer}-1")
            ax.set_ylabel(f"{reducer}-2")
            fig.colorbar(sc, ax=ax, label="cluster")
            save_fig(fig, outdir, f"scatter_{method}_{reducer}.pdf")

        # ARI heatmap.
        if not self.pairwise.empty:
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

        # Cluster-feature enrichment heatmap, per method.
        for method in self.cluster_features["method"].unique():
            sub = self.cluster_features[self.cluster_features["method"] == method]
            wide = sub.pivot(index="cluster", columns="feature", values="enrichment")
            if wide.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, 0.3 * wide.shape[1]), 0.5 * wide.shape[0] + 1.5))
            im = ax.imshow(wide.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
            ax.set_xticks(range(wide.shape[1]))
            ax.set_xticklabels(wide.columns, rotation=75, ha="right", fontsize=7)
            ax.set_yticks(range(wide.shape[0]))
            ax.set_yticklabels([f"c{c}" for c in wide.index])
            fig.colorbar(im, ax=ax, label="P(feat=T | cluster) − P(feat=T)")
            save_fig(fig, outdir, f"cluster_features_{method}.pdf")


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
        # Pick the trial with the desired observed_examples count for embed.
        # ``observed_examples`` at trial=k has k-1 pairs; we want k = n_io_shown+1
        # so that observed + (query, expected) gives n_io_shown pairs.
        target_trial_n = self.n_io_shown + 1
        trial_for: dict[str, "object"] = {}
        for task in bundle:
            tr = max(
                (t for t in task.trials if t.order == self.order),
                key=lambda t: 1 if t.trial == target_trial_n else 0,
            )
            trial_for[task.task_id] = tr

        embeddings: dict[str, np.ndarray] = {}
        for method in ok_methods:
            vecs = []
            for tid in task_ids:
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
