"""Support-set extraction and Stage 0 proxy scoring.

The proxy score of a subset S is the number of quality-passing distinct
behaviors whose *witness* program uses only primitives in S. It is a lower
bound on S's true yield (a behavior may be reachable in S via a program other
than the first-found full-pool witness); Stage 0 only ranks, so a consistent
lower bound is acceptable.
"""

from collections import defaultdict
from itertools import combinations
from typing import Sequence

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


def proxy_scores(
    buckets: dict[frozenset[str], int],
    pool_names: Sequence[str],
    k: int,
) -> dict[tuple[str, ...], int]:
    """:func:`proxy_score` for every k-subset of the pool at once.

    Scoring subset-by-subset costs ``#subsets x #supports`` — 10^11 set tests
    for the 57-primitive pool at k=4. Expanding each support over the subsets
    that contain it instead costs ``sum_s count(s) * C(n - |s|, k - |s|)``,
    which is ~10^6 for the same sweep, because a support of size s only
    reaches C(n-s, k-s) subsets.

    Returns ``{sorted subset tuple: score}`` for the subsets a support
    actually reached; subsets no support fits inside score 0 and are absent.
    """
    names = tuple(sorted(pool_names))
    scores: dict[tuple[str, ...], int] = defaultdict(int)
    for support, count in buckets.items():
        if len(support) > k:
            continue
        rest = [n for n in names if n not in support]
        base = tuple(support)
        for extra in combinations(rest, k - len(support)):
            scores[tuple(sorted(base + extra))] += count
    return dict(scores)
