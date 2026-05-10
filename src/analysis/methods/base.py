from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..capability import Capability, CapabilityMissing
from ..task import Trial


@dataclass
class Prediction:
    """Per-trial output from a Method.

    ``response`` is a parsed list of ints (or sentinel ``EMPTY`` / ``NO_RESPONSE``
    from ``methods.csv_method``); ``program`` is the symbolic program if the
    method emits one; ``correct`` is the strict bool match against the trial's
    expected output (computed by the method, since each method has its own
    parsing quirks); ``effort`` is method-specific search-cost metadata.
    """

    response: list[int] | None
    program: str | None
    correct: bool
    effort: dict | None = None


class Method(ABC):
    name: str
    capabilities: ClassVar[Capability] = Capability.PREDICTIONS

    @abstractmethod
    def predict(self, trial: Trial) -> Prediction: ...

    def predict_many(self, trials: list[Trial]) -> list[Prediction]:
        """Default: per-trial fan-out. Methods with a meaningful batched
        forward (``TransformerMethod``) should override.
        """
        return [self.predict(t) for t in trials]

    def embed(self, trial: Trial) -> "np.ndarray":
        raise CapabilityMissing(self.name, Capability.EMBEDDINGS)

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities
