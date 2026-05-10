"""MetagolMethod adapter for Rule's ``metagol.csv`` (no ``program`` column)."""
from __future__ import annotations

from typing import ClassVar

from .csv_method import CSVMethod


class MetagolMethod(CSVMethod):
    csv_filename: ClassVar[str] = "metagol.csv"
    program_col: ClassVar[str | None] = None
    effort_cols: ClassVar[tuple[str, ...]] = ("cpu",)
