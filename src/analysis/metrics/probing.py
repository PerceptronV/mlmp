"""Probing analysis (§5.2 of ``.claude/plans/probing.md``).

For each transformer (or other ``EmbeddingMethod``) in scope, train a linear
probe on per-task encoder embeddings to predict whether MPL's best solution
uses each meta-primitive in ``METAPRIMITIVE_VOCAB``. Filter to MPL-acquired
tasks (read from ``rule_acquisition/acquired.parquet``); compare AUROC per
method per primitive; report a label-shuffle null and a surface-feature
baseline so the headline numbers are interpretable.

The hypothesis being tested: a meta-learned model's representations should
linearly separate tasks by the meta-primitives MPL uses, *more* than a
non-meta-learned baseline. If e.g. the symbol-shuffling probe AUROCs beat the
in-weight probe's on ``Recurse`` / ``AntiUnify``, that is converging evidence
for functionally-meta-primitive structure in meta-learned representations.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm

from ..capability import Capability
from ..methods.mpl import META_PRIMITIVE_VOCAB
from ..plotting import apply_rc, colour_for, label_for, save_fig
from ..stats import delong_auroc_test, paired_wilcoxon
from .base import Analysis, AnalysisResult

if TYPE_CHECKING:
    from ..cache import Cache
    from ..methods.base import Method
    from ..task import TaskBundle

logger = logging.getLogger(__name__)


_SURFACE_NAME = "surface_features"

# Sweep method names emitted by ``scripts/build_probing_robustness_config.py``,
# e.g. ``tx_enum_iw_enc_L2`` / ``tx_enrl_es_dec_post``. We use this to detect
# whether the AUROC table came from a layer × source sweep and, if so, emit the
# extra robustness plot alongside the headline figures.
_SWEEP_NAME = re.compile(r"^tx_(?P<ckpt>[a-z0-9_]+?)_(?P<src>enc|dec)_(?P<tag>L\d+|post)$")


def _parse_sweep_method(name: str) -> tuple[str, str, int] | None:
    m = _SWEEP_NAME.match(name)
    if not m:
        return None
    tag = m.group("tag")
    layer = -1 if tag == "post" else int(tag[1:])
    return m.group("ckpt"), m.group("src"), layer


def plot_layer_robustness(auroc: pd.DataFrame, outdir: Path) -> bool:
    """Emit per-primitive + grid plots of AUROC vs layer (encoder vs decoder).

    Returns ``True`` if a robustness plot was written, ``False`` if no sweep
    method names were detected (so callers can no-op silently). Used by
    :meth:`ProbingResult.plot` and by ``scripts/plot_probing_robustness.py``.
    """
    import matplotlib.pyplot as plt  # type: ignore[import-untyped]

    rows: list[dict] = []
    for _, r in auroc.iterrows():
        parsed = _parse_sweep_method(r["method"])
        if parsed is None:
            continue
        ckpt, src, layer = parsed
        rows.append({
            "ckpt": ckpt, "src": src, "layer": layer,
            "primitive": r["primitive"], "auroc": float(r["auroc"]),
        })
    if not rows:
        return False

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["ckpt", "src", "layer", "primitive"])["auroc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    n_layers = max((l for l in agg["layer"] if l != -1), default=0)

    def _xpos(layer: int) -> int:
        return n_layers + 1 if layer == -1 else layer

    def _xlabel(layer: int) -> str:
        return "post-norm" if layer == -1 else f"L{layer}"

    agg["xpos"] = agg["layer"].map(_xpos)
    primitives = sorted(agg["primitive"].unique())
    ckpts = sorted(agg["ckpt"].unique())
    x_layers = sorted({(l, _xpos(l)) for l in agg["layer"]}, key=lambda t: t[1])
    x_pos = [p for _, p in x_layers]
    x_labels = [_xlabel(l) for l, _ in x_layers]

    src_colour = {"enc": "#1f77b4", "dec": "#d62728"}
    src_label = {"enc": "encoder", "dec": "decoder"}

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Per-primitive figure: one panel per checkpoint.
    for prim in primitives:
        sub = agg[agg["primitive"] == prim]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, len(ckpts), figsize=(3.4 * len(ckpts), 3.2), sharey=True)
        if len(ckpts) == 1:
            axes = [axes]
        for ax, ck in zip(axes, ckpts):
            for src in ("enc", "dec"):
                s = sub[(sub["ckpt"] == ck) & (sub["src"] == src)].sort_values("xpos")
                if s.empty:
                    continue
                yerr = (
                    s["std"].fillna(0).values
                    / np.sqrt(np.maximum(s["count"].fillna(1).values, 1))
                )
                ax.errorbar(
                    s["xpos"], s["mean"], yerr=yerr, marker="o",
                    color=src_colour[src], label=src_label[src], capsize=2,
                )
            ax.axhline(0.5, color="gray", lw=0.6, alpha=0.6)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=30, ha="right")
            ax.set_title(ck, fontsize=10)
            ax.set_ylim(0.4, 1.02)
        axes[0].set_ylabel(f"AUROC ({prim})")
        axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(outdir / f"robustness_{prim}.pdf")
        plt.close(fig)

    # 2) Grid: rows = primitives, cols = checkpoints.
    fig, axes = plt.subplots(
        len(primitives), len(ckpts),
        figsize=(2.8 * len(ckpts), 1.9 * len(primitives)),
        sharey=True, sharex=True, squeeze=False,
    )
    for i, prim in enumerate(primitives):
        for j, ck in enumerate(ckpts):
            ax = axes[i][j]
            sub = agg[(agg["primitive"] == prim) & (agg["ckpt"] == ck)]
            for src in ("enc", "dec"):
                s = sub[sub["src"] == src].sort_values("xpos")
                if s.empty:
                    continue
                ax.plot(
                    s["xpos"], s["mean"], marker="o", color=src_colour[src],
                    label=src_label[src] if (i == 0 and j == 0) else None,
                )
            ax.axhline(0.5, color="gray", lw=0.5, alpha=0.5)
            ax.set_ylim(0.4, 1.02)
            if i == 0:
                ax.set_title(ck, fontsize=9)
            if j == 0:
                ax.set_ylabel(prim, fontsize=9)
            ax.set_xticks(x_pos)
            if i == len(primitives) - 1:
                ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
            else:
                ax.set_xticklabels([])
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(outdir / "robustness_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


@dataclass
class ProbingResult(AnalysisResult):
    auroc: pd.DataFrame              # (method, primitive, fold, auroc)
    null: pd.DataFrame               # (method, primitive, perm_idx, auroc)
    base: pd.DataFrame               # (primitive, base_rate, n_acquired_tasks)
    cross_method: pd.DataFrame       # (method_a, method_b, primitive, auroc_a, auroc_b, p_delong, p_wilcoxon)
    auroc_conditional: pd.DataFrame = field(default_factory=pd.DataFrame)  # (method, primitive, auroc, n_eligible, n_total)
    config: dict = field(default_factory=dict)

    def save(self, outdir: Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.auroc.to_parquet(outdir / "auroc.parquet", index=False)
        if not self.null.empty:
            self.null.to_parquet(outdir / "null.parquet", index=False)
        self.base.to_parquet(outdir / "base.parquet", index=False)
        if not self.cross_method.empty:
            self.cross_method.to_csv(outdir / "cross_method.csv", index=False)
        if not self.auroc_conditional.empty:
            self.auroc_conditional.to_parquet(outdir / "auroc_conditional.parquet", index=False)
        # Stash the config so re-plotting from disk can reconstruct labels etc.
        import json

        (outdir / "config.json").write_text(json.dumps(self.config, indent=2))

    def plot(self, outdir: Path) -> None:
        apply_rc()
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        if self.auroc.empty:
            logger.warning("ProbingResult: empty auroc table — skipping plots")
            return

        # Aggregate per (method, primitive): mean across folds.
        per_mp = (
            self.auroc.groupby(["method", "primitive"])["auroc"].mean().reset_index()
        )
        wide = per_mp.pivot(index="method", columns="primitive", values="auroc")

        # Pin the surface_features baseline at the bottom so it reads as the
        # comparison reference, not just another method.
        if _SURFACE_NAME in wide.index:
            order = [m for m in wide.index if m != _SURFACE_NAME] + [_SURFACE_NAME]
            wide = wide.loc[order]

        # Per-(method, primitive) one-sided permutation p (real ≥ shuffled-null).
        pmat = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
        if not self.null.empty:
            null_dist = self.null.groupby(["method", "primitive"])["auroc"].agg(list)
            for m in wide.index:
                for p in wide.columns:
                    if (m, p) not in null_dist.index:
                        continue
                    arr = np.asarray(null_dist.loc[(m, p)])
                    real = float(wide.loc[m, p])
                    if arr.size and not np.isnan(real):
                        pmat.loc[m, p] = (np.sum(arr >= real) + 1) / (arr.size + 1)

        # 1) Heatmap: methods × primitives.
        fig, ax = plt.subplots(figsize=(0.9 * wide.shape[1] + 3.0, 0.6 * wide.shape[0] + 2.0))
        im = ax.imshow(wide.values, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(wide.shape[1]))
        ax.set_xticklabels(wide.columns, rotation=30, ha="right")
        ax.set_yticks(range(wide.shape[0]))
        ax.set_yticklabels([label_for(m) for m in wide.index])
        for i in range(wide.shape[0]):
            for j in range(wide.shape[1]):
                v = wide.values[i, j]
                if np.isnan(v):
                    continue
                p = pmat.values[i, j]
                star = "*" if (not np.isnan(p)) and p < 0.05 else ""
                ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=9,
                        color="white" if v < 0.7 else "black")
        fig.colorbar(im, ax=ax, label="AUROC  (* = permutation p<0.05)")
        save_fig(fig, outdir, "heatmap.pdf")

        # 2) Per-primitive grouped bars across methods, with null distribution
        # 95th-percentile dashed line and base-rate annotation.
        primitives = list(wide.columns)
        methods = list(wide.index)
        n_p = len(primitives)
        n_m = len(methods)
        if n_p == 0 or n_m == 0:
            return
        # Mean + std over folds for error bars.
        agg = (
            self.auroc.groupby(["method", "primitive"])["auroc"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * n_p), 4.5))
        x = np.arange(n_p)
        width = 0.8 / max(n_m, 1)
        for i, m in enumerate(methods):
            sub = agg[agg["method"] == m].set_index("primitive").reindex(primitives)
            heights = sub["mean"].values
            yerr = (sub["std"].fillna(0).values / np.sqrt(np.maximum(sub["count"].fillna(1).values, 1)))
            ax.bar(
                x + (i - (n_m - 1) / 2) * width,
                heights,
                width=width,
                label=label_for(m),
                color=colour_for(m),
                yerr=yerr,
                capsize=2,
            )
        # Null distribution 95th percentile per primitive (max across methods so
        # the dashed line is a conservative upper-bound of "chance").
        if not self.null.empty:
            null_q = (
                self.null.groupby("primitive")["auroc"]
                .quantile(0.95)
                .reindex(primitives)
            )
            for j, p in enumerate(primitives):
                v = float(null_q.loc[p]) if p in null_q.index and not np.isnan(null_q.loc[p]) else np.nan
                if np.isnan(v):
                    continue
                ax.hlines(v, x[j] - 0.4, x[j] + 0.4, colors="black", linestyles="dashed", lw=1.0)
        # Base-rate annotation under each x-tick.
        rate_by = {row["primitive"]: row["base_rate"] for _, row in self.base.iterrows()}
        labels = [f"{p}\n(rate={rate_by.get(p, float('nan')):.2f})" for p in primitives]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylim(0.0, 1.02)
        ax.axhline(0.5, color="gray", lw=0.6, alpha=0.6)
        ax.set_ylabel("AUROC (5-fold CV)")
        ax.legend(
            fontsize=8,
            ncols=min(n_m, 3),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            frameon=False,
        )
        save_fig(fig, outdir, "bars_per_primitive.pdf")

        # 3) Conditional bars: AUROC restricted to tasks where MPL AND the
        # method were both correct at label_trial. Mirrors error_similarity's
        # bars_conditional. surface_features is excluded (no model correctness).
        if not self.auroc_conditional.empty:
            cond = self.auroc_conditional.copy()
            cond_primitives = [p for p in primitives if p in set(cond["primitive"])]
            cond_methods = [m for m in methods if m in set(cond["method"]) and m != _SURFACE_NAME]
            n_pc = len(cond_primitives)
            n_mc = len(cond_methods)
            if n_pc and n_mc:
                fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * n_pc), 4.5))
                xc = np.arange(n_pc)
                wc = 0.8 / max(n_mc, 1)
                cond_wide = cond.pivot(index="method", columns="primitive", values="auroc")
                for i, m in enumerate(cond_methods):
                    heights = [
                        float(cond_wide.loc[m, p]) if (m in cond_wide.index and p in cond_wide.columns) else np.nan
                        for p in cond_primitives
                    ]
                    ax.bar(
                        xc + (i - (n_mc - 1) / 2) * wc,
                        heights,
                        width=wc,
                        label=label_for(m),
                        color=colour_for(m),
                    )
                # n_eligible (min across methods per primitive) annotation.
                n_elig = cond.groupby("primitive")["n_eligible"].min().reindex(cond_primitives)
                cond_labels = [f"{p}\n(n={int(n_elig.loc[p])})" for p in cond_primitives]
                ax.set_xticks(xc)
                ax.set_xticklabels(cond_labels, rotation=0, fontsize=8)
                ax.set_ylim(0.0, 1.02)
                ax.axhline(0.5, color="gray", lw=0.6, alpha=0.6)
                ax.set_ylabel("AUROC (MPL-correct ∩ method-correct)")
                ax.legend(
                    fontsize=8,
                    ncols=min(n_mc, 3),
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.12),
                    frameon=False,
                )
                save_fig(fig, outdir, "bars_conditional.pdf")

        # 4) Robustness sweep (additive): if methods follow the sweep naming
        # convention ``tx_<ckpt>_<enc|dec>_<L\d+|post>``, emit per-primitive +
        # grid plots of AUROC vs layer. No-op for non-sweep runs.
        if plot_layer_robustness(self.auroc, outdir):
            logger.info("ProbingResult: robustness plots written (sweep detected)")


@dataclass
class ProbingAnalysis(Analysis):
    kind: str = "probing"
    needs_embeddings: bool = True
    mpl_method: str = "mpl_best"
    acquired_method: str = "mpl"
    # Optional override for where to read ``acquired.parquet`` from. Defaults
    # to ``<output_dir>/<run_name>/rule_acquisition`` (same run). Set this when
    # running probing under a different ``run_name`` than the rule_acquisition
    # output you want to reuse (e.g. probe-robustness sweep reading the
    # headline ``study_v1`` parquet).
    acquired_dir: str | None = None
    primitives: list[str] | None = None
    n_folds: int = 5
    n_perm: int = 200
    embedding_pool: Literal["mean", "last"] = "mean"
    n_io_shown: int = 11
    order: int = 1
    label_aggregation: Literal["majority", "soft"] = "soft"
    majority_threshold: float = 0.5
    surface_baseline: bool = True
    random_state: int = 0
    label_trial: int = 11
    # Additive: when True, also compute per-(method, primitive) AUROC on the
    # subset of tasks where the probe method itself was correct at label_trial.
    # The full task set is already MPL-correct (acquired_on ≤ label_trial), so
    # this conditions on the model also getting the task right. Triggers a
    # prediction pass for each embed method (cached).
    conditional: bool = False

    def run(
        self,
        methods: list["Method"],
        bundle: "TaskBundle",
        cache: "Cache",
    ) -> ProbingResult:
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
        from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
        from sklearn.model_selection import StratifiedKFold  # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

        # 1. Find the MPL method providing labels.
        mpl = next((m for m in methods if m.name == self.mpl_method), None)
        if mpl is None or not hasattr(mpl, "metaprimitives_for"):
            raise ValueError(
                f"ProbingAnalysis: no MPL method named {self.mpl_method!r} with "
                "a ``metaprimitives_for`` helper in the methods list. Add an "
                "``mpl_best`` (or ``mpl``) entry under the analysis ``methods:``."
            )

        # 2. Resolve the acquired-tasks set from rule_acquisition's parquet.
        # Honour an explicit override so a sweep run under a fresh run_name can
        # still point at the headline rule_acquisition output.
        if self.acquired_dir:
            acq_dir = Path(self.acquired_dir).expanduser()
        else:
            acq_dir = Path(cache.root).parent / "rule_acquisition"
        acquired = mpl.acquired_tasks(acq_dir, method_name=self.acquired_method)
        # Restrict to tasks present in the bundle (intersect for safety).
        bundle_ids = set(bundle.task_ids)
        acquired_in_bundle = sorted(acquired & bundle_ids)
        n_acquired = len(acquired_in_bundle)
        logger.info(
            "ProbingAnalysis: %d MPL-acquired tasks (out of %d in bundle, %d in parquet)",
            n_acquired, len(bundle_ids), len(acquired),
        )
        if n_acquired < self.n_folds * 2:
            logger.warning(
                "ProbingAnalysis: only %d acquired tasks — probe will likely be unreliable.",
                n_acquired,
            )

        # 3. Build labels per task: aggregate metaprimitives over (run, order)
        # replicates → soft vector y[task] in [0, 1]^P. Track empty cases.
        primitive_names = self.primitives or list(META_PRIMITIVE_VOCAB.keys())
        vocab = {p: META_PRIMITIVE_VOCAB[p] for p in primitive_names if p in META_PRIMITIVE_VOCAB}
        if len(vocab) != len(primitive_names):
            unknown = [p for p in primitive_names if p not in META_PRIMITIVE_VOCAB]
            raise ValueError(
                f"Unknown primitives: {unknown}. Known: {sorted(META_PRIMITIVE_VOCAB)}"
            )
        primitive_names = list(vocab.keys())

        soft_labels: list[np.ndarray] = []
        kept_tasks: list[str] = []
        for tid in acquired_in_bundle:
            mat = mpl.metaprimitives_for(tid, trial=self.label_trial, vocab=vocab)
            if mat.shape[0] == 0:
                logger.debug("ProbingAnalysis: %s has no MPL replicates at trial %d — dropping",
                             tid, self.label_trial)
                continue
            soft_labels.append(mat.mean(axis=0).astype(np.float32))
            kept_tasks.append(tid)
        if not kept_tasks:
            raise RuntimeError(
                "ProbingAnalysis: no tasks had MPL replicates at the configured "
                "trial — check that mpl_best was loaded with the right filters."
            )
        Y_soft = np.stack(soft_labels, axis=0)                          # (N, P) in [0, 1]
        Y_bin = (Y_soft >= self.majority_threshold).astype(np.int8)     # b = (y ≥ τ)
        N, _P = Y_bin.shape

        # 4. Per-primitive base rates (under ``majority_threshold``). Drop
        # primitives with positive base rate below 10% — too few positives to
        # fit a probe meaningfully. No upper-bound filter: high-rate primitives
        # (eg. MemorizeAll at 0.82) are kept; they have few negatives but the
        # AUROC is still interpretable.
        base_rate = Y_bin.mean(axis=0)
        base_rows = [
            {"primitive": p, "base_rate": float(base_rate[i]), "n_acquired_tasks": N}
            for i, p in enumerate(primitive_names)
        ]
        valid_primitives: list[str] = []
        valid_idx: list[int] = []
        for i, p in enumerate(primitive_names):
            n_pos = int(Y_bin[:, i].sum())
            n_neg = N - n_pos
            rate = float(base_rate[i])
            if rate < 0.10:
                logger.warning(
                    "ProbingAnalysis: primitive %s base rate %.3f < 0.10 "
                    "(under majority_threshold=%.2f) — skipping.",
                    p, rate, self.majority_threshold,
                )
                continue
            if min(n_pos, n_neg) < self.n_folds:
                logger.warning(
                    "ProbingAnalysis: primitive %s has %d/%d pos/neg — < n_folds=%d; "
                    "some folds will be skipped, AUROC will be noisy.",
                    p, n_pos, n_neg, self.n_folds,
                )
            valid_primitives.append(p)
            valid_idx.append(i)

        # 5. Embeddings per (method, task). Skip non-EMBEDDINGS methods (e.g.
        # the MPLBestMethod itself, which doesn't expose embed()).
        embed_methods: list[Method] = [
            m for m in methods if m is not mpl and m.supports(Capability.EMBEDDINGS)
        ]
        if not embed_methods and not self.surface_baseline:
            raise RuntimeError(
                "ProbingAnalysis: no methods with EMBEDDINGS in the methods list "
                "and surface_baseline=False — nothing to probe."
            )

        embeddings: dict[str, np.ndarray] = {}
        if embed_methods:
            # Pick the trial whose observed_examples + (query, expected) gives
            # exactly n_io_shown pairs — i.e. trial = n_io_shown.
            target_trial = self.n_io_shown
            trial_for: dict[str, object] = {}
            for tid in kept_tasks:
                task = bundle.tasks[tid]
                tr = next(
                    (t for t in task.trials if t.order == self.order and t.trial == target_trial),
                    None,
                )
                if tr is None:
                    tr = next((t for t in task.trials if t.order == self.order), None)
                if tr is None and task.trials:
                    tr = task.trials[0]
                if tr is None:
                    raise RuntimeError(f"ProbingAnalysis: no trials for task {tid}")
                trial_for[tid] = tr

            for method in embed_methods:
                vecs = []
                needs_compute = any(
                    not cache.has_embedding(method, tid, self.order) for tid in kept_tasks
                )
                tids_iter = kept_tasks
                if needs_compute:
                    tids_iter = tqdm(kept_tasks, desc=f"probing embed:{method.name}", leave=True)
                for tid in tids_iter:
                    tr = trial_for[tid]
                    v = cache.get_or_compute_embedding(
                        method, tid, self.order, lambda m=method, t=tr: m.embed(t)
                    )
                    vecs.append(v)
                embeddings[method.name] = np.stack(vecs, axis=0)
            cache.flush()

        # 6. Surface-feature baseline: per-task feature vector from
        # functions.csv. Acts as a probe-on-features control.
        if self.surface_baseline:
            feat_names = bundle.feature_names
            X_surface = np.array(
                [
                    [int(bool(bundle.tasks[tid].features.get(f, False))) for f in feat_names]
                    for tid in kept_tasks
                ],
                dtype=np.float32,
            )
            embeddings[_SURFACE_NAME] = X_surface

        # 7. Probe + label-shuffle null per (method, primitive).
        # ``soft`` mode uses the duplicate-with-weights trick: emit two rows
        # ``(x, 1, weight=y)`` and ``(x, 0, weight=1-y)`` per task. This is
        # mathematically identical to BCE against the fractional ``y`` and
        # is the unbiased estimator of P(primitive used | embedding).
        # ``majority`` mode trains on the binary label ``b`` with equal weights.
        # AUROC ground truth is always ``b`` (decoupled from training target).
        rng = np.random.default_rng(self.random_state)
        auroc_rows: list[dict] = []
        null_rows: list[dict] = []

        def _fit_probe(Xs: np.ndarray, b: np.ndarray, ys: np.ndarray, tr_idx: np.ndarray):
            clf = LogisticRegression(
                C=1.0, solver="liblinear", max_iter=1000,
                random_state=self.random_state,
            )
            if self.label_aggregation == "soft":
                X_tr = Xs[tr_idx]
                ys_tr = ys[tr_idx]
                X_dup = np.vstack([X_tr, X_tr])
                y_dup = np.concatenate([np.ones(len(ys_tr)), np.zeros(len(ys_tr))]).astype(np.int8)
                w_dup = np.concatenate([ys_tr, 1.0 - ys_tr]).astype(np.float64)
                mask = w_dup > 0
                if len(set(y_dup[mask].tolist())) < 2:
                    # Degenerate fold (all weight on one class) — fall back to
                    # the binary majority fit so we still produce a probe.
                    clf.fit(Xs[tr_idx], b[tr_idx])
                    return clf
                clf.fit(X_dup[mask], y_dup[mask], sample_weight=w_dup[mask])
                return clf
            clf.fit(Xs[tr_idx], b[tr_idx])
            return clf

        # Total LR fits per (method, primitive): n_folds (CV) + n_perm * n_folds
        # (null). With 48 methods × ~6 valid primitives × (5 + 50 × 5), this is
        # the slow phase after embeddings are cached — surface a pbar with
        # per-pair ETA so the wait time is visible.
        fit_pbar = tqdm(
            total=len(embeddings) * len(valid_primitives),
            desc="probe fits (CV + perm null)",
            leave=True,
            unit="pair",
        )
        for method_name, X in embeddings.items():
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            for i, p in zip(valid_idx, valid_primitives):
                fit_pbar.set_postfix_str(f"{method_name}:{p}")
                b = Y_bin[:, i]
                ys = Y_soft[:, i].astype(np.float64)
                # 5-fold stratified CV. Stratification key is ``b`` regardless
                # of label_aggregation (folds need a binary key); AUROC ground
                # truth is also ``b``.
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(Xs, b)):
                    if len(set(b[tr_idx].tolist())) < 2 or len(set(b[te_idx].tolist())) < 2:
                        continue
                    clf = _fit_probe(Xs, b, ys, tr_idx)
                    scores = clf.predict_proba(Xs[te_idx])[:, 1]
                    try:
                        auc = float(roc_auc_score(b[te_idx], scores))
                    except ValueError:
                        auc = float("nan")
                    auroc_rows.append({
                        "method": method_name,
                        "primitive": p,
                        "fold": int(fold_idx),
                        "auroc": auc,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                    })

                # Label-shuffle null: permute task indices, refit. Permuting
                # indices keeps ``b`` and ``ys`` aligned (a sensible null for
                # both training modes).
                for perm in range(self.n_perm):
                    perm_idx = rng.permutation(N)
                    b_perm = b[perm_idx]
                    ys_perm = ys[perm_idx]
                    fold_aucs: list[float] = []
                    for tr_idx, te_idx in skf.split(Xs, b_perm):
                        if len(set(b_perm[tr_idx].tolist())) < 2 or len(set(b_perm[te_idx].tolist())) < 2:
                            continue
                        clf = _fit_probe(Xs, b_perm, ys_perm, tr_idx)
                        scores = clf.predict_proba(Xs[te_idx])[:, 1]
                        try:
                            fold_aucs.append(float(roc_auc_score(b_perm[te_idx], scores)))
                        except ValueError:
                            continue
                    if fold_aucs:
                        null_rows.append({
                            "method": method_name,
                            "primitive": p,
                            "perm_idx": int(perm),
                            "auroc": float(np.mean(fold_aucs)),
                        })
                fit_pbar.update(1)
        fit_pbar.close()

        auroc_df = pd.DataFrame(auroc_rows)
        null_df = pd.DataFrame(null_rows)
        base_df = pd.DataFrame(base_rows)

        # 8. Cross-method per-primitive comparisons. Wilcoxon over per-fold
        # AUROCs is the cheap default; DeLong needs full-fold scores which we
        # build separately on a held-out CV-prediction pass for fidelity.
        cross_rows: list[dict] = []
        method_names = list(embeddings.keys())
        # Pre-build CV scores per (method, primitive) for DeLong. Uses the
        # same _fit_probe (so soft-mode probes power the DeLong test too).
        cv_scores: dict[tuple[str, str], np.ndarray] = {}
        cv_pbar = tqdm(
            total=len(embeddings) * len(valid_primitives),
            desc="cv scores (DeLong)",
            leave=True,
            unit="pair",
        )
        for method_name, X in embeddings.items():
            Xs = StandardScaler().fit_transform(X)
            for i, p in zip(valid_idx, valid_primitives):
                cv_pbar.set_postfix_str(f"{method_name}:{p}")
                b = Y_bin[:, i]
                ys = Y_soft[:, i].astype(np.float64)
                skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
                scores_full = np.full(N, np.nan, dtype=np.float64)
                for tr_idx, te_idx in skf.split(Xs, b):
                    if len(set(b[tr_idx].tolist())) < 2:
                        continue
                    clf = _fit_probe(Xs, b, ys, tr_idx)
                    scores_full[te_idx] = clf.predict_proba(Xs[te_idx])[:, 1]
                cv_scores[(method_name, p)] = scores_full
                cv_pbar.update(1)
        cv_pbar.close()

        for a, b in combinations(method_names, 2):
            for p in valid_primitives:
                pi = primitive_names.index(p)
                y = Y_bin[:, pi]
                sa = cv_scores.get((a, p))
                sb = cv_scores.get((b, p))
                if sa is None or sb is None:
                    continue
                mask = ~(np.isnan(sa) | np.isnan(sb))
                if mask.sum() < 5 or len(set(y[mask].tolist())) < 2:
                    continue
                auc_a, auc_b, p_dl = delong_auroc_test(y[mask], sa[mask], sb[mask])
                # Wilcoxon over the per-fold AUROCs (more robust to fold
                # noise than DeLong on small samples).
                fa = auroc_df[(auroc_df["method"] == a) & (auroc_df["primitive"] == p)].sort_values("fold")["auroc"].values
                fb = auroc_df[(auroc_df["method"] == b) & (auroc_df["primitive"] == p)].sort_values("fold")["auroc"].values
                if fa.size == fb.size and fa.size >= 3:
                    _, p_wil = paired_wilcoxon(fa, fb)
                else:
                    p_wil = float("nan")
                cross_rows.append({
                    "method_a": a,
                    "method_b": b,
                    "primitive": p,
                    "auroc_a": float(auc_a),
                    "auroc_b": float(auc_b),
                    "p_delong": float(p_dl),
                    "p_wilcoxon": float(p_wil),
                })
        cross_df = pd.DataFrame(cross_rows)

        # 9. Conditional AUROC (additive — does not touch any prior output).
        # Per (probe method, primitive), restrict eval to tasks where the
        # probe method itself was correct at label_trial. The probing set is
        # already MPL-correct (acquired_on ≤ label_trial via acquired.parquet),
        # so this conditions on "MPL-correct ∩ method-correct". surface_features
        # is excluded since it has no model-level correctness.
        auroc_cond_rows: list[dict] = []
        if self.conditional and embed_methods:
            method_correct: dict[str, dict[str, bool]] = {}
            for method in embed_methods:
                per: dict[str, bool] = {}
                for tid in kept_tasks:
                    tr = trial_for[tid]
                    pred = cache.get_or_compute(
                        method, tr, lambda t, m=method: m.predict(t)
                    )
                    per[tid] = bool(pred.correct)
                method_correct[method.name] = per
            cache.flush()

            for method_name, mc in method_correct.items():
                for p in valid_primitives:
                    pi = primitive_names.index(p)
                    y = Y_bin[:, pi]
                    scores = cv_scores.get((method_name, p))
                    if scores is None:
                        continue
                    mask = np.array([
                        (not np.isnan(scores[i])) and mc.get(kept_tasks[i], False)
                        for i in range(N)
                    ])
                    n_eligible = int(mask.sum())
                    n_total = int((~np.isnan(scores)).sum())
                    if n_eligible < 5 or len(set(y[mask].tolist())) < 2:
                        auc = float("nan")
                    else:
                        try:
                            auc = float(roc_auc_score(y[mask], scores[mask]))
                        except ValueError:
                            auc = float("nan")
                    auroc_cond_rows.append({
                        "method": method_name,
                        "primitive": p,
                        "auroc": auc,
                        "n_eligible": n_eligible,
                        "n_total": n_total,
                    })
        auroc_cond_df = pd.DataFrame(auroc_cond_rows)

        return ProbingResult(
            auroc=auroc_df,
            null=null_df,
            base=base_df,
            cross_method=cross_df,
            auroc_conditional=auroc_cond_df,
            config={
                "n_folds": self.n_folds,
                "n_perm": self.n_perm,
                "label_aggregation": self.label_aggregation,
                "majority_threshold": float(self.majority_threshold),
                "embedding_pool": self.embedding_pool,
                "n_io_shown": self.n_io_shown,
                "order": self.order,
                "primitives": primitive_names,
                "valid_primitives": valid_primitives,
                "n_acquired_tasks": int(N),
                "kept_tasks": kept_tasks,
                "mpl_method": self.mpl_method,
                "acquired_method": self.acquired_method,
                "surface_baseline": bool(self.surface_baseline),
                "conditional": bool(self.conditional),
            },
        )
