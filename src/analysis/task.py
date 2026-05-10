"""Rule's 100-list-function benchmark, as the analysis subpackage sees it.

Loads ``functions.csv`` and ``stimuli.csv`` from Rule's OSF release into
``Task`` and ``Trial`` dataclasses. The benchmark uses ``c001..c100``;
``functions.csv`` and ``stimuli.csv`` ship with 250 tasks, but every model CSV
in the release covers only the first 100 — so we filter to that intersection
by default.
"""
from __future__ import annotations

import ast
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# Boolean string parsing. Rule's ``functions.csv`` mixes encodings across
# columns: most use Python-bool ``True``/``False``; ``recursive`` and
# ``counting`` use upper-case ``TRUE``/``FALSE`` plus a handful of cells
# typed as the abbreviations ``"y"`` / ``"n"``. We accept all of them so
# feature columns aren't silently dropped — the OSF data is read-only and
# we don't edit it.
_TRUE_CANON = {"TRUE", "True", "true", "1"}
_FALSE_CANON = {"FALSE", "False", "false", "0"}
_TRUE_EXTRA = {"y", "Y", "yes", "Yes", "YES"}
_FALSE_EXTRA = {"n", "N", "no", "No", "NO"}
_TRUE = _TRUE_CANON | _TRUE_EXTRA
_FALSE = _FALSE_CANON | _FALSE_EXTRA


def _parse_bool(s: str) -> bool:
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return False  # treat NA / unknown as False so feature stratification is still well-defined


def _parse_int_list(s: str) -> list[int]:
    """Parse Rule's bracketed list-of-ints (``"[1, 2, 3]"``)."""
    s = s.strip()
    if not s:
        return []
    # ast.literal_eval handles the standard "[1, 2, 3]" form. Some cells use
    # spaces only (no commas) which literal_eval can't parse — fall back.
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
    except (ValueError, SyntaxError):
        pass
    inner = s.strip("[]").replace(",", " ").split()
    return [int(x) for x in inner]


@dataclass(frozen=True)
class Trial:
    task_id: str
    order: int                                            # 1..5
    trial: int                                            # 1..11
    observed_examples: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    query_input: tuple[int, ...]
    expected_output: tuple[int, ...]

    @property
    def n_io_shown(self) -> int:
        return len(self.observed_examples)


@dataclass(frozen=True)
class Task:
    task_id: str
    program: str
    gloss: str
    features: dict[str, bool]
    trials: tuple[Trial, ...] = field(default_factory=tuple)


class TaskBundle:
    """Bundle of ``Task`` objects, indexed by ``task_id``.

    ``trials`` are ordered by ``(order, trial)`` — 5 orders × 11 trials per task.
    """

    def __init__(self, tasks: dict[str, Task]):
        self.tasks: dict[str, Task] = tasks

    @classmethod
    def load(cls, root: Path, task_ids: list[str] | None = None) -> "TaskBundle":
        root = Path(root)
        # functions.csv → per-task program / gloss / boolean features.
        with open(root / "functions.csv", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            functions = {row["id"]: row for row in reader}
        # Boolean feature columns: those whose every value is in ``_TRUE | _FALSE``.
        # The leading columns (``program_length``, ``depth``, ``apps``,
        # ``variables``, ``lambdas``) carry numerics and are excluded.
        # We warn (once per column) if a column contains any values outside
        # the canonical ``TRUE``/``True``/``true``/``1``/``FALSE``/``False``/
        # ``false``/``0`` set so typos in the OSF source remain visible.
        # Non-canonical values (``y``/``n`` etc.) still parse via the
        # extended ``_TRUE``/``_FALSE`` sets — this is the price of not
        # editing the upstream CSV.
        canonical = _TRUE_CANON | _FALSE_CANON
        bool_strings = _TRUE | _FALSE
        feature_cols: list[str] = []
        for c in fieldnames:
            if c in ("id", "program", "gloss"):
                continue
            vals = {row.get(c, "") for row in functions.values()}
            if not vals.issubset(bool_strings):
                # Either fully numeric or genuinely free-text; not a bool feature.
                continue
            feature_cols.append(c)
            non_canonical = vals - canonical
            if non_canonical:
                logger.warning(
                    "functions.csv: feature column %r contains %d non-canonical "
                    "value(s) %r — accepted via extended bool parser",
                    c, len(non_canonical), sorted(non_canonical),
                )

        # stimuli.csv → per-(id, order, trial) (input, output) pairs.
        stimuli_by_task: dict[str, dict[tuple[int, int], tuple[tuple[int, ...], tuple[int, ...]]]] = {}
        with open(root / "stimuli.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row["id"]
                key = (int(row["order"]), int(row["trial"]))
                inp = tuple(_parse_int_list(row["input"]))
                out = tuple(_parse_int_list(row["output"]))
                stimuli_by_task.setdefault(tid, {})[key] = (inp, out)

        # Default benchmark scope: c001..c100 if any present, else everything.
        if task_ids is None:
            keep = sorted(set(functions) & set(stimuli_by_task))
            preferred = [t for t in keep if t.startswith("c") and t[1:].isdigit() and 1 <= int(t[1:]) <= 100]
            task_ids = preferred or keep

        tasks: dict[str, Task] = {}
        for tid in task_ids:
            row = functions[tid]
            stim = stimuli_by_task[tid]
            features = {c: _parse_bool(row.get(c, "")) for c in feature_cols}
            trials_for_task: list[Trial] = []
            orders = sorted({o for (o, _t) in stim.keys()})
            for order in orders:
                pairs_in_order = sorted([t for (o, t) in stim.keys() if o == order])
                if not pairs_in_order:
                    continue
                ordered_pairs = [stim[(order, t)] for t in pairs_in_order]
                for trial in pairs_in_order:
                    observed = tuple(ordered_pairs[: trial - 1])
                    inp, out = stim[(order, trial)]
                    trials_for_task.append(
                        Trial(
                            task_id=tid,
                            order=order,
                            trial=trial,
                            observed_examples=observed,
                            query_input=inp,
                            expected_output=out,
                        )
                    )
            tasks[tid] = Task(
                task_id=tid,
                program=row.get("program", ""),
                gloss=row.get("gloss", ""),
                features=features,
                trials=tuple(trials_for_task),
            )
        return cls(tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        for tid in sorted(self.tasks):
            yield self.tasks[tid]

    @property
    def task_ids(self) -> list[str]:
        return sorted(self.tasks)

    @property
    def feature_names(self) -> list[str]:
        if not self.tasks:
            return []
        # All tasks share the same feature schema (functions.csv is the single source).
        any_task = next(iter(self.tasks.values()))
        return list(any_task.features.keys())

    def iter_trials(self) -> Iterator[Trial]:
        for task in self:
            yield from task.trials

    def get_trial(self, task_id: str, order: int, trial: int) -> Trial:
        for tr in self.tasks[task_id].trials:
            if tr.order == order and tr.trial == trial:
                return tr
        raise KeyError(f"({task_id}, order={order}, trial={trial}) not in bundle")
