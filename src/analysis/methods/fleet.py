"""FleetMethod / FleetBestMethod adapters for Rule's Fleet CSVs."""
from __future__ import annotations

from typing import ClassVar

from .csv_method import CSVMethod


class FleetMethod(CSVMethod):
    csv_filename: ClassVar[str] = "fleet.csv"
    effort_cols: ClassVar[tuple[str, ...]] = ("cpu", "count")


class FleetBestMethod(CSVMethod):
    csv_filename: ClassVar[str] = "fleet_best.csv"
    effort_cols: ClassVar[tuple[str, ...]] = ("cpu", "count", "lposterior", "lprior", "llikelihood")
