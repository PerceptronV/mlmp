"""MPLMethod / MPLBestMethod adapters for Rule's MPL CSVs."""
from __future__ import annotations

from typing import ClassVar

from .csv_method import CSVMethod


class MPLMethod(CSVMethod):
    csv_filename: ClassVar[str] = "mpl.csv"
    response_col: ClassVar[str] = "response"
    program_col: ClassVar[str] = "program"
    correct_col: ClassVar[str] = "accuracy"
    effort_cols: ClassVar[tuple[str, ...]] = ("time", "count", "lposterior")


class MPLBestMethod(CSVMethod):
    """``mpl_best.csv`` doesn't carry response / accuracy — only the highest-
    posterior program per cell. We treat ``program`` as the prediction, leave
    ``response`` empty, and let consumers compile-and-execute when they need to
    score it. This is enough for clustering / failure modes that only need
    accuracy markers, **provided** the caller supplies a ``response`` column or
    routes through MPLMethod for prediction-level work.
    """

    csv_filename: ClassVar[str] = "mpl_best.csv"
    response_col: ClassVar[str] = "output"  # the gold output, since there is no model response
    program_col: ClassVar[str] = "program"
    correct_col: ClassVar[str | None] = None  # not present in mpl_best
    effort_cols: ClassVar[tuple[str, ...]] = ("time", "count", "lposterior")
