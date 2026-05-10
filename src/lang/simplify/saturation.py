"""Equality-saturation loop.

Two-phase per iteration: (1) collect every ``(rule, subst)`` match
read-only; (2) apply each by constructing the RHS, unioning with the
LHS root, then calling :meth:`EGraph.rebuild` once. This is the design
recommended in §5.1 of ``docs/program-simplification.tex`` — it
prevents new e-nodes added in the same iteration from interfering with
ongoing matching.

Stop conditions: no new equalities (fixpoint), iteration cap, e-node
cap, or wall-clock cap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from .egraph import EClassId, EGraph
from .pattern import ematch
from .rules import Rule


@dataclass
class SaturationConfig:
    max_iterations: int = 30
    max_enodes: int = 10_000
    max_seconds: float = 5.0


StopReason = str  # "saturated" | "iter_cap" | "node_cap" | "time_cap"


@dataclass
class SaturationReport:
    iterations: int
    n_enodes: int
    stopped_reason: StopReason
    matches_per_rule: dict[str, int] = field(default_factory=dict)
    unions_per_rule: dict[str, int] = field(default_factory=dict)


def saturate(
    eg: EGraph,
    rules: Iterable[Rule],
    cfg: SaturationConfig | None = None,
) -> SaturationReport:
    cfg = cfg or SaturationConfig()
    rules = list(rules)
    matches_per_rule: dict[str, int] = {r.name: 0 for r in rules}
    unions_per_rule: dict[str, int] = {r.name: 0 for r in rules}

    start = time.monotonic()
    stopped: StopReason = "iter_cap"
    iteration = 0

    for iteration in range(cfg.max_iterations):
        # Phase 1: collect matches (rule, subst, lhs_class) — read-only.
        applications: list[tuple[Rule, dict[str, EClassId], EClassId]] = []
        for rule in rules:
            n_matches = 0
            for cls in eg.classes():
                for subst in ematch(rule.lhs, cls.id, eg):
                    if rule.side_condition(subst, eg):
                        applications.append((rule, subst, cls.id))
                        n_matches += 1
            matches_per_rule[rule.name] += n_matches

        if not applications:
            stopped = "saturated"
            break

        # Phase 2: apply each match.
        unioned = 0
        for rule, subst, lhs_class in applications:
            try:
                rhs_class = rule.rhs(subst, eg)
            except Exception:
                # Side conditions should make RHS construction safe;
                # any exception here means a bug — skip rather than
                # corrupt the e-graph.
                continue
            lhs_root = eg.find(lhs_class)
            rhs_root = eg.find(rhs_class)
            if lhs_root != rhs_root:
                eg.union(lhs_root, rhs_root)
                unioned += 1
                unions_per_rule[rule.name] += 1

        eg.rebuild()

        if unioned == 0:
            stopped = "saturated"
            break
        if eg.num_enodes() > cfg.max_enodes:
            stopped = "node_cap"
            break
        if time.monotonic() - start > cfg.max_seconds:
            stopped = "time_cap"
            break

    return SaturationReport(
        iterations=iteration + 1,
        n_enodes=eg.num_enodes(),
        stopped_reason=stopped,
        matches_per_rule=matches_per_rule,
        unions_per_rule=unions_per_rule,
    )
