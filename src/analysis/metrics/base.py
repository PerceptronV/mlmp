from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only imports
    from ..cache import Cache
    from ..methods.base import Method
    from ..task import TaskBundle


class AnalysisResult(ABC):
    """Result of running an Analysis. Carries per-method tables and pairwise
    cross-method statistics; ``.save`` and ``.plot`` write artefacts.
    """

    @abstractmethod
    def save(self, outdir: Path) -> None: ...

    @abstractmethod
    def plot(self, outdir: Path) -> None: ...


class Analysis(ABC):
    kind: str = ""
    needs_embeddings: bool = False

    @abstractmethod
    def run(
        self,
        methods: list["Method"],
        bundle: "TaskBundle",
        cache: "Cache",
    ) -> AnalysisResult: ...
