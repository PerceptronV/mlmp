"""Failure-modes analysis (§7.2).

Per-method outputs:
- ``accuracy[method, feature, value ∈ {True, False}]`` averaged over tasks
  where ``task.features[feature] == value``, with bootstrap CIs. Accuracy is
  the un-thresholded run/subject mean (``Prediction.effort["mean_correct"]``)
  averaged across all 11 trials per (function, order). Falling back to
  ``float(Prediction.correct)`` when a method doesn't expose ``mean_correct``.
- FDR-corrected χ² test of feature ↔ correctness per (method, feature). The
  χ² 2×2 table thresholds the per-task mean accuracy at 0.5 to obtain a
  binary "method got the function right" cell.

Cross-method outputs:
- Per-feature difference profile: ranked tasks by ``acc_A − acc_B`` for each
  pair, with the 10 largest gainers/losers per pair.

Plots:
- Method × feature accuracy heatmap (humans/MPL/Fleet pinned as reference rows).
- Difference-profile bars per pair.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ..capability import Capability
from ..plotting import apply_rc, label_for, save_fig
from ..stats import bootstrap_ci, chi2_p, fdr_bh
from .base import Analysis, AnalysisResult

if TYPE_CHECKING:
    from ..cache import Cache
    from ..methods.base import Method
    from ..task import TaskBundle


@dataclass
class FailureModesResult(AnalysisResult):
    accuracy_by_feature: pd.DataFrame      # (method, feature, value, mean, lo, hi, n)
    chi2_table: pd.DataFrame               # (method, feature, chi2_p, q)
    difference_profiles: dict[tuple[str, str], pd.DataFrame]  # per-pair task gap

    def save(self, outdir: Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        self.accuracy_by_feature.to_parquet(outdir / "result.parquet", index=False)
        self.chi2_table.to_csv(outdir / "chi2_fdr.csv", index=False)
        for (a, b), df in self.difference_profiles.items():
            df.to_csv(outdir / f"diff_{a}__vs__{b}.csv", index=False)

    def plot(self, outdir: Path) -> None:
        apply_rc()
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Heatmap: rows = methods, cols = features, cell = (acc_True - acc_False).
        wide = (
            self.accuracy_by_feature[self.accuracy_by_feature["value"] == True]  # noqa: E712
            .pivot(index="method", columns="feature", values="mean")
        )
        wide_f = (
            self.accuracy_by_feature[self.accuracy_by_feature["value"] == False]  # noqa: E712
            .pivot(index="method", columns="feature", values="mean")
        )
        n_t = (
            self.accuracy_by_feature[self.accuracy_by_feature["value"] == True]  # noqa: E712
            .pivot(index="method", columns="feature", values="n")
        )
        gap = (wide - wide_f).fillna(0.0)
        if gap.size:
            # Pin reference rows so they read as the comparison baseline.
            pinned = [m for m in ["humans", "mpl", "fleet"] if m in gap.index]
            rest = [m for m in gap.index if m not in pinned]
            gap = gap.loc[pinned + rest]
            n_t = n_t.loc[pinned + rest]
            sig_lookup: dict[tuple[str, str], float] = {}
            if not self.chi2_table.empty:
                for _, r in self.chi2_table.iterrows():
                    sig_lookup[(r["method"], r["feature"])] = float(r["q"])
            fig, ax = plt.subplots(figsize=(max(7, 0.6 * gap.shape[1] + 3),
                                             0.45 * gap.shape[0] + 2))
            im = ax.imshow(gap.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
            ax.set_xticks(range(gap.shape[1]))
            ns = n_t.iloc[0]  # n is the same across methods (per-feature task count)
            ax.set_xticklabels(
                [f"{c}\n(n={int(ns[c])})" for c in gap.columns],
                rotation=30, ha="right", fontsize=9,
            )
            ax.set_yticks(range(gap.shape[0]))
            ax.set_yticklabels([label_for(m) for m in gap.index])
            for i, m in enumerate(gap.index):
                for j, f in enumerate(gap.columns):
                    v = gap.values[i, j]
                    q = sig_lookup.get((m, f), 1.0)
                    star = "*" if q < 0.05 else ""
                    ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.3 else "black")
            fig.colorbar(im, ax=ax, label="acc(feat=T) − acc(feat=F)  (* = χ² q<0.05)")
            save_fig(fig, outdir, "heatmap.pdf")

        # Difference profile bars per pair.
        for (a, b), df in self.difference_profiles.items():
            top = df.head(10)
            bot = df.tail(10)
            both = pd.concat([top, bot])
            fig, ax = plt.subplots(figsize=(6, 0.35 * len(both) + 1.5))
            ax.barh(both["function"], both["delta"], color=np.where(both["delta"] > 0, "#1f77b4", "#d62728"))
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=0.6)
            ax.set_xlabel(f"acc({label_for(a)}) − acc({label_for(b)})")
            save_fig(fig, outdir, f"difference_profile_{a}__vs__{b}.pdf")


@dataclass
class FailureModesAnalysis(Analysis):
    kind: str = "failure_modes"
    features: list[str] = field(default_factory=list)
    pairs: object = "all"
    n_boot: int = 1000

    def run(
        self,
        methods: list["Method"],
        bundle: "TaskBundle",
        cache: "Cache",
    ) -> FailureModesResult:
        # Per-(method, function, order, trial) accuracy via cache. Use the
        # un-thresholded run/subject mean from ``effort["mean_correct"]`` when
        # available; fall back to binarized ``Prediction.correct``. This matches
        # Rule's per-function mean accuracy framing (humans averaged across
        # subjects, MPL/Fleet averaged across runs) rather than thresholding
        # at 50% before aggregating.
        rows: list[dict] = []
        for method in methods:
            trials = list(bundle.iter_trials())
            show = method.supports(Capability.EMBEDDINGS) and any(
                not cache.has_prediction(method, t) for t in trials
            )
            preds = cache.compute_many(
                method, trials,
                progress_desc=f"failure_modes predict:{method.name}" if show else None,
            )
            for trial, p in zip(trials, preds):
                acc = float(p.correct)
                if p.effort and "mean_correct" in p.effort:
                    try:
                        acc = float(p.effort["mean_correct"])
                    except (TypeError, ValueError):
                        pass
                rows.append({
                    "method": method.name,
                    "function": trial.task_id,
                    "order": trial.order,
                    "trial": trial.trial,
                    "accuracy": acc,
                })
        cache.flush()
        long = pd.DataFrame(rows)
        # Average across all 11 trials and orders to get a per-(method, function)
        # accuracy in [0, 1]. Saved as ``accuracy`` for clarity.
        per_task = long.groupby(["method", "function"])["accuracy"].mean().reset_index()

        # Feature table.
        if not self.features:
            self.features = bundle.feature_names
        feat_table = pd.DataFrame.from_records(
            [{"function": t.task_id, **{f: t.features.get(f, False) for f in self.features}} for t in bundle]
        )
        merged = per_task.merge(feat_table, on="function", how="left")

        # Per-(method, feature, value) bootstrap.
        rng = np.random.default_rng(0)
        acc_rows: list[dict] = []
        for method_name in merged["method"].unique():
            for feat in self.features:
                for val in (True, False):
                    sub = merged[(merged["method"] == method_name) & (merged[feat] == val)]
                    accs = sub["accuracy"].values
                    if accs.size == 0:
                        acc_rows.append({"method": method_name, "feature": feat, "value": val,
                                         "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0})
                        continue
                    ci = bootstrap_ci(accs, np.mean, n_boot=self.n_boot, rng=rng)
                    acc_rows.append({
                        "method": method_name, "feature": feat, "value": val,
                        "mean": ci.estimate, "lo": ci.lo, "hi": ci.hi, "n": int(accs.size),
                    })
        accuracy_by_feature = pd.DataFrame(acc_rows)

        # χ² independence with FDR correction across (method, feature).
        chi2_rows: list[dict] = []
        for method_name in merged["method"].unique():
            for feat in self.features:
                sub = merged[merged["method"] == method_name]
                # 2x2 table: rows = correct/incorrect (per-task accuracy thresholded at 0.5), cols = feat T/F.
                hit = (sub["accuracy"] >= 0.5).astype(int)
                ft = sub[feat].astype(bool)
                tab = np.array([
                    [int(((hit == 1) & ft).sum()), int(((hit == 1) & ~ft).sum())],
                    [int(((hit == 0) & ft).sum()), int(((hit == 0) & ~ft).sum())],
                ])
                p = chi2_p(tab) if tab.sum() else 1.0
                chi2_rows.append({"method": method_name, "feature": feat, "chi2_p": p})
        chi2_df = pd.DataFrame(chi2_rows)
        # FDR correct within each method (independent families).
        chi2_df["q"] = float("nan")
        for method_name in chi2_df["method"].unique():
            mask = chi2_df["method"] == method_name
            chi2_df.loc[mask, "q"] = fdr_bh(chi2_df.loc[mask, "chi2_p"].values)

        # Difference profile per pair.
        names = list(merged["method"].unique())
        if self.pairs == "all":
            pairs = list(combinations(names, 2))
        else:
            pairs = [tuple(p) for p in self.pairs]  # type: ignore[union-attr]
        wide = per_task.pivot(index="function", columns="method", values="accuracy")
        diff_profiles: dict[tuple[str, str], pd.DataFrame] = {}
        gloss = {t.task_id: t.gloss for t in bundle}
        for a, b in pairs:
            if a not in wide.columns or b not in wide.columns:
                continue
            df = pd.DataFrame({
                "function": wide.index,
                "gloss": [gloss.get(f, "") for f in wide.index],
                f"acc_{a}": wide[a].values,
                f"acc_{b}": wide[b].values,
                "delta": wide[a].values - wide[b].values,
            }).sort_values("delta", ascending=False)
            diff_profiles[(a, b)] = df

        return FailureModesResult(
            accuracy_by_feature=accuracy_by_feature,
            chi2_table=chi2_df,
            difference_profiles=diff_profiles,
        )
