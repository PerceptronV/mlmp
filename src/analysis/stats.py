"""Shared estimators used by ``Analysis.run``.

Bootstrap CIs, Spearman / Pearson with bootstrap CI, paired Wilcoxon,
log-rank, ARI / NMI, Benjamini-Hochberg FDR, sparse logistic regression.

All estimators take numpy arrays, return scalars or named-tuples, and never
plot. ``scipy.stats`` is imported lazily so a partial install (no scipy) still
loads the module — analyses that don't need stats can still run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class CI:
    estimate: float
    lo: float
    hi: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.estimate, self.lo, self.hi


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Percentile bootstrap CI on the 1-D array ``data``."""
    rng = rng if rng is not None else np.random.default_rng(0)
    data = np.asarray(data)
    if data.size == 0:
        return CI(float("nan"), float("nan"), float("nan"))
    estimate = float(statistic(data))
    boots = np.empty(n_boot)
    n = data.shape[0]
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = statistic(data[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return CI(estimate, lo, hi)


def spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Spearman ρ on paired vectors with paired-bootstrap CI."""
    from scipy.stats import spearmanr  # type: ignore[import-untyped]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return CI(float("nan"), float("nan"), float("nan"))
    estimate = float(spearmanr(x, y).statistic)
    rng = rng if rng is not None else np.random.default_rng(0)
    boots = np.empty(n_boot)
    n = x.shape[0]
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = spearmanr(x[idx], y[idx]).statistic
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return CI(estimate, lo, hi)


def pearson_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> CI:
    """Pearson r on paired vectors with paired-bootstrap CI.

    Used by ``RuleAcquisitionAnalysis`` to match Rule et al.'s reported
    correlations (Supp. Fig. 6), which are Pearson r on length-100 vectors of
    per-function mean accuracy averaged across all 11 trials.
    """
    from scipy.stats import pearsonr  # type: ignore[import-untyped]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return CI(float("nan"), float("nan"), float("nan"))
    estimate = float(pearsonr(x, y).statistic)
    rng = rng if rng is not None else np.random.default_rng(0)
    boots = np.empty(n_boot)
    n = x.shape[0]
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        # Pearson is undefined when either resample has zero variance; skip
        # those draws (they're rare and dropping them is preferable to
        # injecting NaNs into the percentile CI).
        if xb.std() == 0 or yb.std() == 0:
            boots[i] = np.nan
            continue
        boots[i] = pearsonr(xb, yb).statistic
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return CI(estimate, float("nan"), float("nan"))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return CI(estimate, lo, hi)


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank on paired vectors. Returns (statistic, p)."""
    from scipy.stats import wilcoxon  # type: ignore[import-untyped]

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    diffs = x[mask] - y[mask]
    if diffs.size == 0 or np.all(diffs == 0):
        return (0.0, 1.0)
    res = wilcoxon(diffs, zero_method="wilcox", nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def log_rank(
    times_a: np.ndarray,
    events_a: np.ndarray,
    times_b: np.ndarray,
    events_b: np.ndarray,
) -> tuple[float, float]:
    """Two-sample log-rank χ² and p. Times are nonneg floats; events are 0/1
    (1 = acquired, 0 = censored). Implemented inline so we don't pull in
    ``lifelines``.
    """
    times = np.concatenate([times_a, times_b])
    events = np.concatenate([events_a, events_b])
    groups = np.concatenate([np.zeros_like(times_a), np.ones_like(times_b)])
    order = np.argsort(times)
    times, events, groups = times[order], events[order], groups[order]
    unique_times = np.unique(times[events > 0])
    O1 = E1 = V1 = 0.0
    n1 = float(times_a.size)
    n2 = float(times_b.size)
    for t in unique_times:
        at_risk = times >= t
        n_at = at_risk.sum()
        if n_at == 0 or (n1 == 0 and n2 == 0):
            break
        n1_at = ((groups == 0) & at_risk).sum()
        n2_at = ((groups == 1) & at_risk).sum()
        d_at = ((times == t) & (events > 0)).sum()
        d1_at = ((groups == 0) & (times == t) & (events > 0)).sum()
        if n_at < 2 or d_at == 0:
            continue
        e1_at = d_at * (n1_at / n_at)
        v1_at = d_at * (n1_at * n2_at) * (n_at - d_at) / (n_at * n_at * (n_at - 1))
        O1 += d1_at
        E1 += e1_at
        V1 += v1_at
    if V1 == 0:
        return (0.0, 1.0)
    chi2 = (O1 - E1) ** 2 / V1
    from scipy.stats import chi2 as chi2_dist  # type: ignore[import-untyped]
    p = float(1.0 - chi2_dist.cdf(chi2, df=1))
    return float(chi2), p


def ari_nmi(labels_a: np.ndarray, labels_b: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # type: ignore[import-untyped]

    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    return (
        float(adjusted_rand_score(a, b)),
        float(normalized_mutual_info_score(a, b)),
    )


def fdr_bh(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR-corrected q-values for a 1-D sequence of p-values."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty_like(ranked)
    cur_min = 1.0
    for i in range(n - 1, -1, -1):
        v = ranked[i] * n / (i + 1)
        cur_min = min(cur_min, v)
        q[i] = cur_min
    out = np.empty_like(q)
    out[order] = q
    return out


def chi2_p(table: np.ndarray) -> float:
    """χ² independence p on a 2x2 contingency table."""
    from scipy.stats import chi2_contingency  # type: ignore[import-untyped]

    if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        return 1.0
    res = chi2_contingency(table, correction=False)
    return float(res.pvalue)


def delong_auroc_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float, float]:
    """DeLong's test for the difference between two paired AUROCs on the same
    binary outcome ``y_true`` and two score vectors.

    Returns ``(auroc_a, auroc_b, p_two_sided)``.

    Implementation follows Sun & Xu (2014)'s fast O(n log n) version using
    midrank-based estimates of the AUC variance/covariance. ``y_true`` must be
    in {0, 1}; ``scores_a`` / ``scores_b`` are arbitrary real-valued (higher =
    more confident positive).
    """
    from scipy.stats import norm  # type: ignore[import-untyped]

    y = np.asarray(y_true).astype(int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    pos = y == 1
    neg = ~pos
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        return (float("nan"), float("nan"), float("nan"))

    # Sun & Xu midrank trick: AUC_k = (mean midrank of positives among all
    # samples - (m+1)/2) / n. Variance via per-sample structural components.
    def _structural(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        x_p = scores[pos]
        x_n = scores[neg]
        tx = _midrank(x_p)
        ty = _midrank(x_n)
        tz = _midrank(np.concatenate([x_p, x_n]))
        auc = float((tz[:m].sum() / (m * n)) - (m + 1) / (2 * n))
        v01 = (tz[:m] - tx) / n  # length-m structural components for positives
        v10 = 1.0 - (tz[m:] - ty) / m  # length-n structural components for negatives
        return auc, v01, v10

    auc_a, v01_a, v10_a = _structural(sa)
    auc_b, v01_b, v10_b = _structural(sb)

    s_x = np.cov(np.stack([v01_a, v01_b], axis=0), ddof=1)
    s_y = np.cov(np.stack([v10_a, v10_b], axis=0), ddof=1)
    cov = s_x / m + s_y / n
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var_diff <= 0:
        # Identical scores → p=1; numerically negative → 0 effective variance.
        return (auc_a, auc_b, 1.0)
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * (1 - norm.cdf(abs(z))))
    return (auc_a, auc_b, p)


def _midrank(x: np.ndarray) -> np.ndarray:
    """Average rank with ties broken by midrank. Used by ``delong_auroc_test``."""
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    n = x.shape[0]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1  # 1-indexed midrank
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out
