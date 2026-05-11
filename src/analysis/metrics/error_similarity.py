"""Error-similarity analysis (§7.4).

For each cell ``(function, order, trial)``:
- Build the empirical human response distribution ``P_human(r)`` over the ~20
  subjects who answered (``EMPTY`` / ``NO_RESPONSE`` are valid distinct keys).
- For each non-human method, look up the method's response ``r_model``; the
  per-cell human-likeness is ``P_human(r_model)``.

Per-method aggregate: mean over (5 orders × 11 trials × 100 functions),
collapsed to a length-100 profile by averaging within function (functions are
the bootstrap unit). Cross-method outputs: Spearman / paired Wilcoxon
between profiles, per-trial profiles, and a correct-vs-incorrect conditional
decomposition.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ..capability import Capability
from ..plotting import apply_rc, colour_for, label_for, save_fig
from ..stats import bootstrap_ci, paired_wilcoxon, spearman_with_ci
from .base import Analysis, AnalysisResult

if TYPE_CHECKING:
    from ..cache import Cache
    from ..methods.base import Method
    from ..task import TaskBundle

logger = logging.getLogger(__name__)

_HUMAN_NAMES = {"humans", "human"}


@dataclass
class ErrorSimilarityResult(AnalysisResult):
    cells: pd.DataFrame                      # (method, function, order, trial, p_human, correct, accuracy)
    per_function: pd.DataFrame               # (method, function, mean_p_human)
    per_method: pd.DataFrame                 # (method, mean, lo, hi, conditional_*)
    per_trial: pd.DataFrame                  # (method, trial, mean)
    pairwise: pd.DataFrame                   # (method_a, method_b, spearman, *)

    def save(self, outdir: Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.cells.to_parquet(outdir / "result.parquet", index=False)
        self.per_method.to_csv(outdir / "per_method.csv", index=False)
        self.per_function.to_parquet(outdir / "per_function.parquet", index=False)
        self.per_trial.to_csv(outdir / "per_trial.csv", index=False)
        self.pairwise.to_csv(outdir / "pairwise_spearman.csv", index=False)

    def plot(self, outdir: Path) -> None:
        apply_rc()
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # 1. Bar: mean human-likeness per method, sorted.
        pm = self.per_method.sort_values("mean", ascending=False)
        fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(pm) + 2), 4))
        xs = np.arange(len(pm))
        ax.bar(xs, pm["mean"], yerr=[pm["mean"] - pm["lo"], pm["hi"] - pm["mean"]],
               color=[colour_for(m) for m in pm["method"]])
        ax.set_xticks(xs); ax.set_xticklabels([label_for(m) for m in pm["method"]], rotation=30, ha="right")
        ax.set_ylabel("mean P_human(model response)")
        save_fig(fig, outdir, "bar_human_likeness.pdf")

        # 2. Per-trial human-likeness curves.
        if not self.per_trial.empty:
            fig, ax = plt.subplots()
            for method in sorted(self.per_trial["method"].unique()):
                sub = self.per_trial[self.per_trial["method"] == method].sort_values("trial")
                ax.plot(sub["trial"], sub["mean"], label=label_for(method), color=colour_for(method))
            ax.set_xlabel("trial"); ax.set_ylabel("mean P_human(model response)")
            ax.set_ylim(0, max(0.6, self.per_trial["mean"].max() * 1.1))
            ax.legend()
            save_fig(fig, outdir, "curves_per_trial.pdf")

        # 3. Scatter of human-likeness vs accuracy per method.
        if {"mean", "accuracy"}.issubset(self.per_method.columns):
            fig, ax = plt.subplots()
            for _, row in self.per_method.iterrows():
                ax.scatter(row["accuracy"], row["mean"], color=colour_for(row["method"]), s=60)
                ax.annotate(label_for(row["method"]), (row["accuracy"], row["mean"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set_xlabel("mean accuracy (all trials)")
            ax.set_ylabel("mean human-likeness")
            save_fig(fig, outdir, "scatter_humanness_vs_accuracy.pdf")

        # 4. Conditional bars (correct vs incorrect cells).
        if {"cond_correct", "cond_incorrect"}.issubset(self.per_method.columns):
            fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(self.per_method) + 2), 4))
            xs = np.arange(len(self.per_method))
            w = 0.4
            ax.bar(xs - w / 2, self.per_method["cond_correct"], w, label="model correct")
            ax.bar(xs + w / 2, self.per_method["cond_incorrect"], w, label="model incorrect")
            ax.set_xticks(xs); ax.set_xticklabels([label_for(m) for m in self.per_method["method"]], rotation=30, ha="right")
            ax.set_ylabel("mean P_human(model response)")
            ax.legend()
            save_fig(fig, outdir, "bars_conditional.pdf")


@dataclass
class ErrorSimilarityAnalysis(Analysis):
    kind: str = "error_similarity"
    n_boot: int = 2000

    def run(
        self,
        methods: list["Method"],
        bundle: "TaskBundle",
        cache: "Cache",
    ) -> ErrorSimilarityResult:
        from ..methods.human import HumanMethod

        # Locate the human reference. If absent, the analysis still runs but
        # cannot compute human-likeness — skip with a warning.
        humans = next((m for m in methods if isinstance(m, HumanMethod)), None)
        if humans is None:
            logger.warning("No HumanMethod in methods list — error_similarity returns empty")
            empty = pd.DataFrame()
            return ErrorSimilarityResult(empty, empty, empty, empty, empty)

        non_human = [m for m in methods if m is not humans]
        if not non_human:
            logger.warning("Only humans in methods list — humans excluded from comparison set as the reference")

        # Per-cell (function, order, trial) human distribution. Pre-compute
        # predictions per method as a batch so transformers run through their
        # batched ``predict_many`` path; the inner cell loop is then pure
        # lookup. Tqdm is driven by ``Cache.compute_many`` per method.
        all_trials = list(bundle.iter_trials())
        preds_by_method: dict[str, list] = {}
        for m in non_human:
            show = m.supports(Capability.EMBEDDINGS) and any(
                not cache.has_prediction(m, t) for t in all_trials
            )
            preds_by_method[m.name] = cache.compute_many(
                m, all_trials,
                progress_desc=f"error_sim predict:{m.name}" if show else None,
            )

        cell_rows: list[dict] = []
        for cell_idx, trial in enumerate(all_trials):
            dist = humans.response_distribution(trial)
            total = sum(dist.values()) or 1
            probs = {k: v / total for k, v in dist.items()}
            for method in non_human:
                p = preds_by_method[method.name][cell_idx]
                response = p.response
                from ..methods.csv_method import EMPTY, NO_RESPONSE  # lazy
                if response is None:
                    key = NO_RESPONSE
                elif response == []:
                    key = EMPTY
                else:
                    key = tuple(int(x) for x in response)
                p_human = float(probs.get(key, 0.0))
                acc = float(p.correct)
                if p.effort and "mean_correct" in p.effort:
                    try:
                        acc = float(p.effort["mean_correct"])
                    except (TypeError, ValueError):
                        pass
                cell_rows.append({
                    "method": method.name,
                    "function": trial.task_id,
                    "order": trial.order,
                    "trial": trial.trial,
                    "p_human": p_human,
                    "correct": float(p.correct),  # binarized; used for the cond_correct/cond_incorrect split
                    "accuracy": acc,              # un-thresholded run/subject mean; used as the scatter x-axis
                    "n_humans": int(total),
                })
        cache.flush()
        cells = pd.DataFrame(cell_rows)

        # Per-function profile.
        per_function = (
            cells.groupby(["method", "function"])["p_human"]
            .mean().reset_index().rename(columns={"p_human": "mean_p_human"})
        )

        # Per-method aggregate with bootstrap CI at the function level.
        rng = np.random.default_rng(0)
        pm_rows: list[dict] = []
        for method_name, sub in per_function.groupby("method"):
            ci = bootstrap_ci(sub["mean_p_human"].values, np.mean, n_boot=self.n_boot, rng=rng)
            cell_sub = cells[cells["method"] == method_name]
            cond_c = cell_sub[cell_sub["correct"] >= 0.5]["p_human"].mean() if (cell_sub["correct"] >= 0.5).any() else 0.0
            cond_i = cell_sub[cell_sub["correct"] < 0.5]["p_human"].mean() if (cell_sub["correct"] < 0.5).any() else 0.0
            pm_rows.append({
                "method": method_name,
                "mean": ci.estimate, "lo": ci.lo, "hi": ci.hi,
                # Mean accuracy across all 11 trials and 5 orders, using the
                # un-thresholded run/subject mean (matches Rule's reported
                # per-method accuracy). Drives the scatter_humanness_vs_accuracy
                # plot and is used as a method-summary in downstream analyses.
                "accuracy": float(cell_sub["accuracy"].mean()) if not cell_sub.empty else 0.0,
                "cond_correct": float(cond_c) if not np.isnan(cond_c) else 0.0,
                "cond_incorrect": float(cond_i) if not np.isnan(cond_i) else 0.0,
            })
        per_method = pd.DataFrame(pm_rows)

        # Per-trial profile.
        per_trial = (
            cells.groupby(["method", "trial"])["p_human"]
            .mean().reset_index().rename(columns={"p_human": "mean"})
        )

        # Pairwise Spearman + paired Wilcoxon between length-N_function profiles.
        wide = per_function.pivot(index="function", columns="method", values="mean_p_human")
        pair_rows: list[dict] = []
        for a, b in combinations(wide.columns, 2):
            x, y = wide[a].values, wide[b].values
            sp = spearman_with_ci(x, y, n_boot=self.n_boot, rng=rng)
            wstat, wp = paired_wilcoxon(x, y)
            pair_rows.append({
                "method_a": a, "method_b": b,
                "spearman": sp.estimate, "spearman_lo": sp.lo, "spearman_hi": sp.hi,
                "wilcoxon_stat": wstat, "wilcoxon_p": wp,
            })
        pairwise = pd.DataFrame(pair_rows)

        return ErrorSimilarityResult(
            cells=cells,
            per_function=per_function,
            per_method=per_method,
            per_trial=per_trial,
            pairwise=pairwise,
        )
