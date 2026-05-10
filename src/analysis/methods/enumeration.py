"""EnumerationMethod adapter for Rule's ``enumeration.csv``."""
from __future__ import annotations

from typing import ClassVar

from .csv_method import CSVMethod


class EnumerationMethod(CSVMethod):
    csv_filename: ClassVar[str] = "enumeration.csv"
    effort_cols: ClassVar[tuple[str, ...]] = ("cpu", "count")
