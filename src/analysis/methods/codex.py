"""CodexMethod adapter for Rule's ``codex.csv``.

Codex emits two ``source`` rows per trial (``codex_code_G`` greedy and
``codex_code_P`` pass@50). Pick one via ``filters: {source: codex_code_G}``
or ``codex_code_P``.
"""
from __future__ import annotations

from typing import ClassVar

from .csv_method import CSVMethod


class CodexMethod(CSVMethod):
    csv_filename: ClassVar[str] = "codex.csv"
    effort_cols: ClassVar[tuple[str, ...]] = ("cpu", "count")
