"""RobustFillMethod adapter for Rule's ``robustfill.csv``."""
from __future__ import annotations

from typing import ClassVar

from .csv_method import CSVMethod


class RobustFillMethod(CSVMethod):
    csv_filename: ClassVar[str] = "robustfill.csv"
    effort_cols: ClassVar[tuple[str, ...]] = ("cpu", "count")
