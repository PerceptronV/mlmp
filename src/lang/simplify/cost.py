"""Cost functions for extraction.

The default :func:`tree_cost` charges 1 per e-node and sums over
children, matching :func:`src.lang.utils.program_size` exactly. Under
this cost, the tree-cost extractor returns the smallest representable
program (in AST size) within the saturated e-graph.
"""

from __future__ import annotations

import math
from typing import Callable

from .egraph import EClassId, EGraph
from .encode import ENode, Op


# A cost function takes an e-node and a mapping from e-class id to its
# currently-best cost; it returns the candidate cost for that e-node.
CostFn = Callable[[ENode, dict[EClassId, float], EGraph], float]


def tree_cost(node: ENode, costs: dict[EClassId, float], eg: EGraph) -> float:
    """Tree-cost matching :func:`src.lang.utils.program_size`.

    Charges 1 per e-node and sums child costs, with one exception that
    mirrors a quirk of ``program_size``: ``ApplicationNode``'s function
    sub-tree is *not* counted toward the AST size, only its arguments.
    For :data:`Op.APP_E` (function-as-expression) we therefore skip
    ``children[0]``.
    """
    # All children must be reachable (finite cost) for the e-node to be
    # selectable. The cost itself excludes the function position of
    # APP_E to match program_size's behaviour.
    for ch in node.children:
        if costs[eg.find(ch)] == math.inf:
            return math.inf

    counted = node.children
    if node.op is Op.APP_E and counted:
        counted = counted[1:]
    return 1.0 + sum(costs[eg.find(ch)] for ch in counted)
