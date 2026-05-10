"""Shared base for the CSV-backed methods.

Per the design (§6) each release CSV has a slightly different schema, so we
keep one ``CSVMethod`` base and let per-source subclasses override column names
via ClassVars. The base does the heavy lifting: load → filter → key-by →
``Prediction`` lookup.
"""
from __future__ import annotations

import ast
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ..capability import Capability
from ..task import Trial
from .base import Method, Prediction

logger = logging.getLogger(__name__)


# Sentinels for parsed responses. ``EMPTY`` means the method explicitly emitted
# the empty list ``"[]"``; ``NO_RESPONSE`` means the cell was null / missing /
# unparseable. Distinguished because Codex emits ~50% nulls and dropping them
# would inflate its apparent human-likeness in error_similarity.
@dataclass(frozen=True)
class _Sentinel:
    label: str

    def __repr__(self) -> str:
        return f"<{self.label}>"


EMPTY = _Sentinel("EMPTY")
NO_RESPONSE = _Sentinel("NO_RESPONSE")

# One-warning-per-(method, reason). Module-level so it survives across
# instances (e.g. cache miss + cold load).
_WARNED: set[tuple[str, str]] = set()


def _warn_once(method_name: str, reason: str, message: str) -> None:
    key = (method_name, reason)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning("[%s] %s", method_name, message)


_NA = {"NA", "N/A", "na", "n/a", "", "nan", "NaN", "None", "null", "NULL"}
_C_WRAPPER = re.compile(r"^\s*[A-Za-z_][\w]*\(\s*(\[.*\])\s*\)\s*$")


def _parse_response(raw: Any, method_name: str = "?") -> tuple | _Sentinel:
    """Parse one ``response`` cell to a tuple of ints, ``EMPTY``, or ``NO_RESPONSE``.

    - ``None`` / ``NA`` / ``""`` / ``nan`` → ``NO_RESPONSE``.
    - ``"[]"`` → ``EMPTY`` (a real, non-missing empty answer).
    - ``"C([1, 2, 3])"`` (MPL TRS-style wrapper) → ``(1, 2, 3)``.
    - Any other string → ``ast.literal_eval`` to a tuple of ints.
    - Bad parses → ``NO_RESPONSE`` + one warning per method.
    """
    if raw is None:
        return NO_RESPONSE
    if isinstance(raw, float):  # pandas nan
        return NO_RESPONSE
    s = str(raw).strip()
    if s in _NA:
        return NO_RESPONSE
    if s == "[]":
        return EMPTY
    m = _C_WRAPPER.match(s)
    if m:
        s = m.group(1)
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        _warn_once(
            method_name,
            "parse_response",
            f"failed to parse response cell {raw!r} (and possibly more); reporting as NO_RESPONSE",
        )
        return NO_RESPONSE
    if isinstance(v, (list, tuple)):
        try:
            return tuple(int(x) for x in v)
        except (TypeError, ValueError):
            _warn_once(method_name, "parse_response", f"non-int element in {raw!r}")
            return NO_RESPONSE
    if isinstance(v, int):
        return (v,)
    return NO_RESPONSE


def _parse_truthy(raw: Any) -> float:
    """Parse a CSV accuracy cell to a float in {0, 1}. Accepts ``True``/``False``
    (codex/fleet/enumeration/metagol/robustfill style) and ``1``/``0`` (mpl,
    predictions style). NA / unparseable → 0."""
    if raw is None:
        return 0.0
    s = str(raw).strip().lower()
    if s in {"true", "t"}:
        return 1.0
    if s in {"false", "f", "", "na", "n/a", "nan", "none"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _value_to_int_list(v) -> list[int] | None:
    if isinstance(v, _Sentinel):
        return None
    if v is None:
        return None
    return [int(x) for x in v]


@dataclass
class CSVMethod(Method):
    """Generic CSV-backed method.

    Subclasses override ``csv_filename`` and the column ClassVars.
    """

    capabilities: ClassVar[Capability] = Capability.PREDICTIONS | Capability.EFFORT

    csv_filename: ClassVar[str] = ""
    join_keys: ClassVar[tuple[str, ...]] = ("id", "order", "trial")
    response_col: ClassVar[str] = "response"
    program_col: ClassVar[str | None] = "program"
    effort_cols: ClassVar[tuple[str, ...]] = ()
    correct_col: ClassVar[str | None] = "accuracy"

    name: str = ""
    root: str | Path = ""
    filters: dict[str, Any] = field(default_factory=dict)
    aggregate: str = "mean"  # how to fold multiple matching rows per trial — "mean" or "first"

    def __post_init__(self) -> None:
        path = Path(self.root) / self.csv_filename
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Apply user-supplied filters: stringly-typed equality on row[col].
        if self.filters:
            rows = [r for r in rows if all(str(r.get(c, "")) == str(v) for c, v in self.filters.items())]
        if not rows:
            raise ValueError(
                f"{self.name}: no rows in {self.csv_filename} match filters {self.filters!r}"
            )
        # Bucket by (id, order, trial). Multiple rows may match (e.g. MPL has
        # 11 search-step rows per cell); we collapse them in ``predict``.
        self._buckets: dict[tuple, list[dict]] = {}
        for r in rows:
            key = self._key(r)
            self._buckets.setdefault(key, []).append(r)
        self._n_runs = len({r.get("run", "1") for r in rows}) or 1

    @classmethod
    def _key(cls, row: dict) -> tuple:
        out: list = []
        for c in cls.join_keys:
            v = row[c]
            if c in ("order", "trial"):
                out.append(int(v))
            else:
                out.append(v)
        return tuple(out)

    def predict(self, trial: Trial) -> Prediction:
        key = (trial.task_id, trial.order, trial.trial)
        rows = self._buckets.get(key)
        if not rows:
            return Prediction(response=None, program=None, correct=False, effort=None)
        # If the CSV has a per-run column, average accuracy across runs and
        # take the modal response.
        if self.correct_col is not None and self.correct_col in rows[0]:
            correct_floats = [_parse_truthy(r.get(self.correct_col)) for r in rows]
            mean_correct = sum(correct_floats) / len(correct_floats)
            correct = mean_correct >= 0.5
        else:
            correct = False
            mean_correct = 0.0

        # Modal response across rows. Sentinel comparison must use identity.
        responses = [_parse_response(r.get(self.response_col, ""), self.name) for r in rows]
        response = _modal(responses)

        program = None
        if self.program_col is not None and self.program_col in rows[0]:
            programs = [r.get(self.program_col, "") for r in rows]
            program = _modal_str(programs)

        effort = None
        if self.effort_cols:
            effort = {}
            for c in self.effort_cols:
                vals = [_parse_float(r.get(c)) for r in rows if r.get(c) not in (None, "")]
                if vals:
                    effort[c] = sum(vals) / len(vals)
            effort["mean_correct"] = mean_correct

        return Prediction(
            response=_value_to_int_list(response),
            program=program,
            correct=correct,
            effort=effort,
        )

    def predict_per_run(self, trial: Trial) -> list[Prediction]:
        """One Prediction per run for this trial. Used by error_similarity for
        run-level bootstrapping."""
        key = (trial.task_id, trial.order, trial.trial)
        rows = self._buckets.get(key, [])
        # Group by run.
        per_run: dict[str, list[dict]] = {}
        for r in rows:
            per_run.setdefault(str(r.get("run", "1")), []).append(r)
        out: list[Prediction] = []
        for run_id, run_rows in per_run.items():
            response = _modal([_parse_response(r.get(self.response_col, ""), self.name) for r in run_rows])
            program = None
            if self.program_col is not None and self.program_col in run_rows[0]:
                program = _modal_str([r.get(self.program_col, "") for r in run_rows])
            correct = False
            if self.correct_col is not None and self.correct_col in run_rows[0]:
                vals = [_parse_truthy(r.get(self.correct_col)) for r in run_rows]
                correct = sum(vals) / len(vals) >= 0.5
            out.append(
                Prediction(
                    response=_value_to_int_list(response),
                    program=program,
                    correct=correct,
                    effort=None,
                )
            )
        return out


def _parse_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _modal(values: list) -> object:
    if not values:
        return NO_RESPONSE
    counts: dict = {}
    keys: list = []
    for v in values:
        # tuples and sentinels both hashable
        if v in counts:
            counts[v] += 1
        else:
            counts[v] = 1
            keys.append(v)
    return max(keys, key=lambda k: counts[k])


def _modal_str(values: list[str]) -> str:
    return _modal([v for v in values if v])  # type: ignore[return-value]
