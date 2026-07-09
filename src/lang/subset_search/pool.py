"""Curated primitive pool and subset generation for the subset search."""

from itertools import combinations
from typing import Iterator, Optional, Sequence

from ..grammar import DefaultGrammar, Grammar


POOL_NAMES: tuple[str, ...] = (
    '+', '-', '*', '%', '<', '==', 'and', 'not',
    'singleton', 'cons', 'concat', 'take', 'drop', 'length', 'range',
    'map', 'filter', 'fold',
)


def pool_grammar(pool_names: Optional[Sequence[str]] = None) -> Grammar:
    """Grammar restricted to the pool (or an explicit override, for tests)."""
    names = tuple(pool_names) if pool_names is not None else POOL_NAMES
    return DefaultGrammar.subset(set(names))


def iter_subsets(
    sizes: tuple[int, ...] = (5, 6),
    pool_names: Optional[Sequence[str]] = None,
) -> Iterator[frozenset[str]]:
    """Yield every subset of the pool with a cardinality in ``sizes``."""
    names = tuple(pool_names) if pool_names is not None else POOL_NAMES
    for k in sizes:
        for combo in combinations(names, k):
            yield frozenset(combo)


def subset_key(subset: frozenset[str]) -> str:
    """Stable string key for a subset (names contain no spaces)."""
    return ' '.join(sorted(subset))
