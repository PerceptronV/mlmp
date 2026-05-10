"""Analysis registry."""
from __future__ import annotations

from typing import Any

from .base import Analysis, AnalysisResult  # noqa: F401  (re-exported)
from .clustering import ClusteringAnalysis
from .error_similarity import ErrorSimilarityAnalysis
from .failure_modes import FailureModesAnalysis
from .probing import ProbingAnalysis
from .rule_acquisition import RuleAcquisitionAnalysis

KINDS: dict[str, type[Analysis]] = {
    "rule_acquisition": RuleAcquisitionAnalysis,
    "failure_modes": FailureModesAnalysis,
    "clustering": ClusteringAnalysis,
    "error_similarity": ErrorSimilarityAnalysis,
    "probing": ProbingAnalysis,
}


def build_analysis(spec: dict[str, Any]) -> Analysis:
    spec = dict(spec)
    kind = spec.pop("kind")
    # The CLI also reads ``methods`` (subset of method names) — strip it before
    # forwarding kwargs; the CLI itself dispatches the subset.
    spec.pop("methods", None)
    if kind not in KINDS:
        raise ValueError(f"Unknown analysis kind {kind!r}; known: {sorted(KINDS)}")
    return KINDS[kind](**spec)
