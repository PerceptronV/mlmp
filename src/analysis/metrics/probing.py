"""Stretch: linear probe predicting MPL meta-primitives from per-task embeddings.

Skeleton only in v1; see ``.claude/plans/probing.md`` for the full delta plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .base import Analysis, AnalysisResult

if TYPE_CHECKING:
    from ..cache import Cache
    from ..methods.base import Method
    from ..task import TaskBundle


@dataclass
class ProbingResult(AnalysisResult):
    def save(self, outdir: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def plot(self, outdir: Path) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class ProbingAnalysis(Analysis):
    kind: str = "probing"
    needs_embeddings: bool = True

    def run(self, methods, bundle, cache):
        raise NotImplementedError(
            "ProbingAnalysis is a v1 stretch — see .claude/plans/probing.md."
        )
