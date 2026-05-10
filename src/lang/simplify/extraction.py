"""Tree-cost extraction from a saturated e-graph.

Implements the Bellman-style fixpoint of §5.3 of
``docs/program-simplification.tex``: initialise every e-class cost to
``inf``, repeatedly relax ``cost[c]`` to ``min_n (w(n) + sum(cost[ch]))``
until no further decrease, then walk the chosen e-nodes to build an
:class:`ASTNode`.

Cyclic e-classes (e.g. introduced by self-referential rewrites) are
handled gracefully: the cyclic e-node has a child whose cost is ``inf``
on the first pass, so any non-cyclic alternative wins.
"""

from __future__ import annotations

import math

from ..ast_nodes import ASTNode
from .cost import CostFn, tree_cost
from .egraph import EClassId, EGraph
from .encode import ENode, decode_term


def extract(
    eg: EGraph,
    root: EClassId,
    cost_fn: CostFn = tree_cost,
) -> tuple[ASTNode, float]:
    """Return the cheapest term for ``root`` and its cost."""
    classes = list(eg.classes())
    costs: dict[EClassId, float] = {c.id: math.inf for c in classes}
    best: dict[EClassId, ENode] = {}

    # Bellman-style relaxation. Iterate to fixpoint over all classes.
    changed = True
    while changed:
        changed = False
        for cls in classes:
            cid = cls.id
            for n in cls.nodes:
                cand = cost_fn(n, costs, eg)
                if cand < costs[cid]:
                    costs[cid] = cand
                    best[cid] = n
                    changed = True

    root_id = eg.find(root)
    if costs[root_id] == math.inf:
        raise RuntimeError(
            "extract: root e-class has infinite cost (every choice is cyclic). "
            "This indicates a bug in the rule set or a missing terminal."
        )
    term = decode_term(eg, root_id, best)
    return term, costs[root_id]
