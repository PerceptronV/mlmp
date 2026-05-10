"""Rule-acquisition analysis.

Reports two complementary statistics, per the design (§7.1):

1. Strict acquisition trial: rule acquired on trial *n* iff correct on every
   trial ≥ *n* (Rule's criterion). Sentinel = 12 = never acquired in the 11-
   trial window. ``acquired_on[method, function, run, order]``.
2. Per-trial mean accuracy curve, with bootstrap CI at the function level.

Cross-method outputs (computed during ``run``):

- Pearson r between length-100 difficulty profiles (per-function mean accuracy
  averaged across all 11 trials), with paired-bootstrap CIs. This matches the
  metric Rule et al. report in Supp. Fig. 6 (e.g. humans-MPL r ≈ 0.715).
- Paired Wilcoxon signed-rank on the same length-100 vector.
- Log-rank between acquisition-trial KM curves per pair.
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
from ..plotting import apply_rc, colour_for, save_fig
from ..stats import CI, log_rank, paired_wilcoxon, pearson_with_ci
from .base import Analysis, AnalysisResult

if TYPE_CHECKING:
    from ..cache import Cache
    from ..methods.base import Method
    from ..task import TaskBundle

logger = logging.getLogger(__name__)


_NEVER_ACQUIRED = 12  # sentinel for "did not acquire by trial 11"


def _strict_acquisition_trial(correct_by_trial: list[bool]) -> int:
    """Smallest n in 1..len such that correct[n-1:] are all True; else
    ``_NEVER_ACQUIRED`` (== len + 1 by convention).
    """
    n = len(correct_by_trial)
    for i in range(n):
        if all(correct_by_trial[i:]):
            return i + 1
    return _NEVER_ACQUIRED


@dataclass
class RuleAcquisitionResult(AnalysisResult):
    long: pd.DataFrame                          # (method, function, order, trial, correct, accuracy)
    acquired: pd.DataFrame                      # (method, function, order, acquired_on)
    per_trial_mean: pd.DataFrame                # (method, trial, mean, lo, hi)
    pairwise: pd.DataFrame                      # (method_a, method_b, pearson, ...)
    human_per_subject_acquired: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["subject", "function", "order", "acquired_on"])
    )  # populated only when a HumanMethod is in the run

    def save(self, outdir: Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.long.to_parquet(outdir / "result.parquet", index=False)
        self.acquired.to_parquet(outdir / "acquired.parquet", index=False)
        self.per_trial_mean.to_csv(outdir / "per_trial_mean.csv", index=False)
        self.pairwise.to_csv(outdir / "pairwise_stats.csv", index=False)
        if not self.human_per_subject_acquired.empty:
            self.human_per_subject_acquired.to_parquet(
                outdir / "human_per_subject_acquired.parquet", index=False
            )

    def _plot_human_band(self, ax, xs: np.ndarray) -> None:
        """Draw median + 25-75 + min-max bands across subjects on the Fig 3A axes.

        For each subject, ``frac_acquired_by(t) = #(functions acquired by t) /
        #(functions seen)``. The plot uses the across-subject distribution per
        trial: median curve, 25-75 dark band, min-max light band.
        """
        sub_df = self.human_per_subject_acquired
        # Per (subject, t): fraction of their functions acquired by t.
        per_subject: dict[str, np.ndarray] = {}
        for subj, g in sub_df.groupby("subject"):
            ao = g["acquired_on"].values
            n = ao.size
            if n == 0:
                continue
            per_subject[subj] = np.array([(ao <= t).mean() for t in xs])
        if not per_subject:
            return
        mat = np.stack(list(per_subject.values()), axis=0)  # (n_subjects, n_trials)
        median = np.median(mat, axis=0)
        q25, q75 = np.quantile(mat, 0.25, axis=0), np.quantile(mat, 0.75, axis=0)
        lo, hi = mat.min(axis=0), mat.max(axis=0)
        ax.fill_between(xs, lo, hi, color="lightgray", alpha=0.5, step="post",
                        label=f"humans min-max (n={mat.shape[0]})")
        ax.fill_between(xs, q25, q75, color="gray", alpha=0.5, step="post",
                        label="humans 25-75%")
        ax.step(xs, median, where="post", color="black", lw=1.5, label="humans median")

    def plot(self, outdir: Path) -> None:
        apply_rc()
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # 1. Cumulative-acquisition curve per method (Rule Fig 3A reproduction).
        # Humans get a median curve + 25-75 + min-max bands across subjects when
        # per-subject acquired_on is available; everyone else gets a single
        # cohort curve.
        fig, ax = plt.subplots()
        xs = np.arange(1, _NEVER_ACQUIRED + 1)
        for method in sorted(self.acquired["method"].unique()):
            if method == "humans" and not self.human_per_subject_acquired.empty:
                self._plot_human_band(ax, xs)
                continue
            sub = self.acquired[self.acquired["method"] == method]
            cum = np.array([(sub["acquired_on"] <= t).mean() for t in xs])
            ax.step(xs, cum, where="post", label=method, color=colour_for(method))
        ax.set_xlabel("trial")
        ax.set_ylabel("cumulative fraction acquired")
        ax.set_xlim(1, _NEVER_ACQUIRED)
        ax.set_ylim(0, 1)
        ax.legend()
        save_fig(fig, outdir, "curves_acquisition.pdf")

        # 2. Per-trial mean accuracy curve.
        fig, ax = plt.subplots()
        for method in sorted(self.per_trial_mean["method"].unique()):
            sub = self.per_trial_mean[self.per_trial_mean["method"] == method].sort_values("trial")
            ax.plot(sub["trial"], sub["mean"], label=method, color=colour_for(method))
            ax.fill_between(sub["trial"], sub["lo"], sub["hi"], alpha=0.15, color=colour_for(method))
        ax.set_xlabel("trial")
        ax.set_ylabel("mean accuracy")
        ax.set_ylim(0, 1)
        ax.legend()
        save_fig(fig, outdir, "curves_per_trial.pdf")

        # 3. Human-relative score violin (Rule Fig 3B): per-function mean
        # accuracy across all 11 trials, divided by the human mean for that
        # function. Uses ``accuracy`` (un-thresholded) for both numerator and
        # denominator.
        if "humans" in set(self.long["method"]):
            wide = (
                self.long
                .groupby(["method", "function"])["accuracy"].mean()
                .unstack(0)
            )
            if "humans" in wide.columns:
                eps = 1e-3
                normed = wide.divide(wide["humans"].clip(lower=eps), axis=0)
                non_human = [c for c in normed.columns if c != "humans"]
                if non_human:
                    fig, ax = plt.subplots()
                    parts = ax.violinplot(
                        [normed[c].dropna().values for c in non_human],
                        showmeans=True,
                        showextrema=False,
                    )
                    ax.set_xticks(range(1, len(non_human) + 1))
                    ax.set_xticklabels(non_human, rotation=30, ha="right")
                    ax.axhline(1.0, color="black", lw=0.5, alpha=0.5)
                    ax.set_ylabel("mean acc / human mean acc")
                    save_fig(fig, outdir, "violin_human_relative.pdf")


@dataclass
class RuleAcquisitionAnalysis(Analysis):
    kind: str = "rule_acquisition"
    pairs: object = "all"                        # "all" or list[[a, b]] of method-name pairs
    n_boot: int = 2000

    def run(
        self,
        methods: list["Method"],
        bundle: "TaskBundle",
        cache: "Cache",
    ) -> RuleAcquisitionResult:
        # 1. Long table of (method, function, order, trial, correct, accuracy).
        # ``correct`` is the binarized 0/1 used by Rule's strict-acquisition
        # criterion. ``accuracy`` is the un-thresholded run/subject mean from
        # ``Prediction.effort['mean_correct']`` when available (humans average
        # over subjects; MPL/Fleet/etc. average over runs); fallback equals
        # ``float(correct)``. Per-trial means and pairwise Pearson use
        # ``accuracy`` to match Rule et al.'s reported figures.
        long_rows: list[dict] = []
        for method in methods:
            trials = list(bundle.iter_trials())
            show = method.supports(Capability.EMBEDDINGS) and any(
                not cache.has_prediction(method, t) for t in trials
            )
            preds = cache.compute_many(
                method, trials,
                progress_desc=f"rule_acq predict:{method.name}" if show else None,
            )
            for trial, p in zip(trials, preds):
                acc = float(p.correct)
                if p.effort and "mean_correct" in p.effort:
                    try:
                        acc = float(p.effort["mean_correct"])
                    except (TypeError, ValueError):
                        pass
                long_rows.append({
                    "method": method.name,
                    "function": trial.task_id,
                    "order": trial.order,
                    "trial": trial.trial,
                    "correct": float(p.correct),
                    "accuracy": acc,
                })
        long = pd.DataFrame(long_rows)
        cache.flush()

        # 2. Strict acquisition trial: per (method, function, order).
        acq_rows: list[dict] = []
        for (method_name, function, order), grp in long.groupby(["method", "function", "order"]):
            grp = grp.sort_values("trial")
            n = _strict_acquisition_trial([bool(c) for c in grp["correct"].tolist()])
            acq_rows.append({
                "method": method_name,
                "function": function,
                "order": order,
                "acquired_on": n,
            })
        acquired = pd.DataFrame(acq_rows)

        # 3. Per-trial mean with bootstrap CI at the function level. Uses
        # ``accuracy`` so humans are scored as subject-mean (continuous)
        # and MPL/Fleet/etc. as run-mean (continuous), matching Rule's
        # reported per-trial means.
        rng = np.random.default_rng(0)
        per_trial_rows: list[dict] = []
        for method_name in long["method"].unique():
            for trial in sorted(long["trial"].unique()):
                # Average within (function, order) at a given trial first, then
                # bootstrap across functions — respects repeated measures
                # (multiple orders per task).
                within = (
                    long[(long["method"] == method_name) & (long["trial"] == trial)]
                    .groupby("function")["accuracy"].mean().values
                )
                if within.size == 0:
                    continue
                est = float(within.mean())
                n = within.size
                boots = np.empty(self.n_boot)
                for i in range(self.n_boot):
                    idx = rng.integers(0, n, size=n)
                    boots[i] = within[idx].mean()
                lo = float(np.quantile(boots, 0.025))
                hi = float(np.quantile(boots, 0.975))
                per_trial_rows.append({"method": method_name, "trial": int(trial), "mean": est, "lo": lo, "hi": hi})
        per_trial_mean = pd.DataFrame(per_trial_rows)

        # 4. Cross-method stats (Step 4 of the build order — wired now).
        pairwise = self._pairwise_stats(long, acquired, methods)

        # 5. Per-subject human acquired_on for the Fig 3A bands (median +
        # 25-75 + min-max over subjects). Only computed when a HumanMethod is
        # present and exposes ``predict_per_subject``.
        human_per_subject = self._human_per_subject_acquired(methods, bundle)

        return RuleAcquisitionResult(
            long=long,
            acquired=acquired,
            per_trial_mean=per_trial_mean,
            pairwise=pairwise,
            human_per_subject_acquired=human_per_subject,
        )

    def _human_per_subject_acquired(
        self,
        methods: list["Method"],
        bundle: "TaskBundle",
    ) -> pd.DataFrame:
        """Per-subject strict-acquisition trial for the HumanMethod, if present.

        Each subject saw a random subset of (function, order) pairs across 11
        trials; for each such pair we apply Rule's strict criterion. The
        downstream plot uses these per-subject acquired_on values to draw the
        median curve + 25-75 + min-max bands on the Fig 3A reproduction.
        """
        try:
            from ..methods.human import HumanMethod  # lazy
        except Exception:
            return pd.DataFrame(columns=["subject", "function", "order", "acquired_on"])
        humans = next((m for m in methods if isinstance(m, HumanMethod)), None)
        if humans is None or not hasattr(humans, "predict_per_subject"):
            return pd.DataFrame(columns=["subject", "function", "order", "acquired_on"])

        # Build (subject, function, order, trial) -> correct.
        per_cell: dict[tuple, list[tuple[int, bool]]] = {}
        for trial in bundle.iter_trials():
            preds = humans.predict_per_subject(trial)
            for subj, p in preds.items():
                key = (subj, trial.task_id, trial.order)
                per_cell.setdefault(key, []).append((int(trial.trial), bool(p.correct)))

        rows: list[dict] = []
        for (subj, fn, order), entries in per_cell.items():
            entries.sort()
            seq = [c for _, c in entries]
            if len(seq) < 11:
                # Subject saw fewer than 11 trials for this (function, order)
                # — skip; otherwise the strict criterion would be inflated.
                continue
            n = _strict_acquisition_trial(seq)
            rows.append({"subject": subj, "function": fn, "order": order, "acquired_on": n})
        return pd.DataFrame(rows)

    def _pairwise_stats(
        self,
        long: pd.DataFrame,
        acquired: pd.DataFrame,
        methods: list["Method"],
    ) -> pd.DataFrame:
        names = [m.name for m in methods]
        if self.pairs == "all":
            pairs = list(combinations(names, 2))
        else:
            pairs = [tuple(p) for p in self.pairs]  # type: ignore[union-attr]

        # Per-function difficulty profile per method: mean accuracy averaged
        # across all 11 trials. Uses ``accuracy`` (un-thresholded run/subject
        # mean) to match Rule et al. Supp. Fig. 6, which reports Pearson r on
        # this same 100-vector.
        profile = (
            long.groupby(["method", "function"])["accuracy"].mean()
            .unstack(0)
        )

        rows: list[dict] = []
        for a, b in pairs:
            if a not in profile.columns or b not in profile.columns:
                continue
            xa = profile[a].values
            xb = profile[b].values
            mask = ~(np.isnan(xa) | np.isnan(xb))
            xa, xb = xa[mask], xb[mask]
            if xa.size < 3:
                continue
            pr = pearson_with_ci(xa, xb, n_boot=self.n_boot)
            wstat, wp = paired_wilcoxon(xa, xb)
            # Log-rank: 1 = acquired (acquired_on <= 11), 0 = censored at 12.
            acq_a = acquired[acquired["method"] == a]["acquired_on"].astype(float).values
            acq_b = acquired[acquired["method"] == b]["acquired_on"].astype(float).values
            ev_a = (acq_a <= 11).astype(float)
            ev_b = (acq_b <= 11).astype(float)
            t_a = np.where(ev_a > 0, acq_a, 11.0)
            t_b = np.where(ev_b > 0, acq_b, 11.0)
            chi2, lrp = log_rank(t_a, ev_a, t_b, ev_b)
            rows.append({
                "method_a": a,
                "method_b": b,
                "pearson": pr.estimate,
                "pearson_lo": pr.lo,
                "pearson_hi": pr.hi,
                "wilcoxon_stat": wstat,
                "wilcoxon_p": wp,
                "logrank_chi2": chi2,
                "logrank_p": lrp,
                "n_functions": int(xa.size),
            })
        return pd.DataFrame(rows)
