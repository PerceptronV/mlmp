"""Support-set extraction and Stage 0 proxy scoring.

The proxy score of a subset S is the number of quality-passing distinct
behaviors whose *witness* program uses only primitives in S. It is a lower
bound on S's true yield (a behavior may be reachable in S via a program other
than the first-found full-pool witness); Stage 0 only ranks, so a consistent
lower bound is acceptable.
"""

from collections import defaultdict

from ..ast_nodes import ASTNode
from ..enumeration.enumerator import ProgramBank
from ..enumeration.filters import passes_quality_filter


def support_set(ast: ASTNode, pool_names: frozenset[str]) -> frozenset[str]:
    """Grammar functions used by a program (variables filtered out by the
    pool intersection — function_names() includes variable references)."""
    return frozenset(ast.function_names()) & pool_names


def support_buckets(
    bank: ProgramBank,
    pool_names: frozenset[str],
    min_successes: int = 3,
    min_variability: float = 0.3,
    target_type: str | None = None,
) -> dict[frozenset[str], int]:
    """Count quality-passing distinct behaviors per witness support set.

    With ``target_type``, only behaviors of that output type are counted, so
    the proxy ranks subsets by their type-restricted yield — filtering here
    (not just at Stage 1 scoring) is what keeps subsets that excel only at
    the target type from being cut before exact evaluation.
    """
    buckets: dict[frozenset[str], int] = defaultdict(int)
    for prog in bank.all_programs():
        if prog.fingerprint is None:
            continue
        if not passes_quality_filter(prog.fingerprint, min_successes, min_variability):
            continue
        if target_type is not None and str(prog.type) != target_type:
            continue
        buckets[support_set(prog.ast, pool_names)] += 1
    return dict(buckets)


def proxy_score(
    subset: frozenset[str],
    buckets: dict[frozenset[str], int],
) -> int:
    """Lower bound on the subset's distinct-behavior yield."""
    return sum(n for support, n in buckets.items() if support <= subset)
