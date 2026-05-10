"""HumanMethod — Rule's ``predictions.csv`` aggregated across subjects.

Capabilities: ``PREDICTIONS`` only. Aggregates over the ``subject`` column when
``predict`` is called; ``subjects()`` is exposed for analyses that want a
subject-level resampling unit (e.g. acquisition curve bootstrap CI band).
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from ..capability import Capability
from ..task import Trial
from .base import Method, Prediction
from .csv_method import _Sentinel, _modal, _parse_response, _parse_truthy


@dataclass
class HumanMethod(Method):
    capabilities: ClassVar[Capability] = Capability.PREDICTIONS
    name: str = "humans"
    root: str | Path = ""

    def __post_init__(self) -> None:
        path = Path(self.root) / "predictions.csv"
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        # Apply Rule et al.'s subject-exclusion criterion (analysis.R:50).
        # A subject is excluded if any of:
        #   - max_same >= 20      (gave the same response on 20+ trials in a row)
        #   - sum(accuracy) <= 10 (got 10 or fewer trials correct out of 110)
        #   - total_time_s < 1200 (under 20 minutes total)
        # Reproduces the paper's headline 392-subject / 0.521 mean-accuracy figure.
        excluded = self._compute_excluded(rows)
        rows = [r for r in rows if r["subject"] not in excluded]
        self._cells: dict[tuple, list[dict]] = defaultdict(list)
        self._subjects: set[str] = set()
        # predictions.csv uses ``block_trial`` (1..11) for the within-block trial
        # and ``total_trial`` (1..110) for the global index; analyses key on the
        # within-block trial.
        trial_col = "block_trial" if rows and "block_trial" in rows[0] else "trial"
        for r in rows:
            key = (r["id"], int(r["order"]), int(r[trial_col]))
            self._cells[key].append(r)
            self._subjects.add(r["subject"])
        self._excluded_subjects: set[str] = excluded

    def _compute_excluded(self, rows: list[dict]) -> set[str]:
        """Replicate ``identify_excluded_subjects`` in analysis.R."""
        # Per-subject totals
        per_subj: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            per_subj[r["subject"]].append(r)
        # participants.csv supplies total_time_s; load if available.
        part_path = Path(self.root) / "participants.csv"
        total_time: dict[str, float] = {}
        if part_path.exists():
            with open(part_path, newline="") as f:
                for p in csv.DictReader(f):
                    try:
                        total_time[p["subject"]] = float(p.get("total_time_s") or 0)
                    except (TypeError, ValueError):
                        total_time[p["subject"]] = 0.0
        excluded: set[str] = set()
        for subj, srows in per_subj.items():
            response_counts = Counter(r.get("response", "") for r in srows)
            max_same = max(response_counts.values()) if response_counts else 0
            acc_sum = sum(_parse_truthy(r.get("accuracy")) for r in srows)
            tt = total_time.get(subj, 0.0)
            if max_same >= 20 or acc_sum <= 10 or tt < 1200:
                excluded.add(subj)
        return excluded

    def subjects(self) -> list[str]:
        return sorted(self._subjects)

    def cache_fingerprint(self) -> str:
        # ``v2`` marks the version that applies Rule's identify_excluded_subjects
        # filter (max_same >= 20 or accuracy <= 10 or total_time_s < 1200).
        # Bump if you change the loading semantics.
        return f"{self.name}::HumanMethod::v2-rule-exclusions"

    def predict(self, trial: Trial) -> Prediction:
        rows = self._cells.get((trial.task_id, trial.order, trial.trial), [])
        if not rows:
            return Prediction(response=None, program=None, correct=False)
        accs = [_parse_truthy(r.get("accuracy")) for r in rows]
        mean_acc = sum(accs) / len(accs)
        modal = _modal([_parse_response(r.get("response", ""), self.name) for r in rows])
        return Prediction(
            response=None if isinstance(modal, _Sentinel) else list(modal),
            program=None,
            correct=mean_acc >= 0.5,
            effort={"mean_correct": mean_acc, "n_subjects": len(rows)},
        )

    def predict_per_subject(self, trial: Trial) -> dict[str, Prediction]:
        rows = self._cells.get((trial.task_id, trial.order, trial.trial), [])
        out: dict[str, Prediction] = {}
        for r in rows:
            response = _parse_response(r.get("response", ""), self.name)
            out[r["subject"]] = Prediction(
                response=None if isinstance(response, _Sentinel) else list(response),
                program=None,
                correct=_parse_truthy(r.get("accuracy")) >= 0.5,
            )
        return out

    def response_distribution(self, trial: Trial) -> Counter:
        """Per-cell empirical distribution of human responses (sentinels kept).

        Keyed by parsed response: a tuple of ints, ``EMPTY``, or ``NO_RESPONSE``.
        """
        rows = self._cells.get((trial.task_id, trial.order, trial.trial), [])
        return Counter(_parse_response(r.get("response", ""), self.name) for r in rows)
