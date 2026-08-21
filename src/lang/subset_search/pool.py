"""Primitive pools and subset generation for the subset search.

``POOL_NAMES`` is the curated 18-primitive pool the original size-5/6 search
ran over; ``ALL_NAMES`` is every primitive in ``DefaultGrammar``. Which one a
sweep uses is a parameter (``--pool``), recorded in ``stage0.json``, so runs
over different pools stay distinguishable after the fact.
"""

from itertools import combinations
from typing import Iterator, Optional, Sequence

from ..grammar import DefaultGrammar, Grammar


POOL_NAMES: tuple[str, ...] = (
    '+', '-', '*', '%', '<', '==', 'and', 'not',
    'singleton', 'cons', 'concat', 'take', 'drop', 'length', 'range',
    'map', 'filter', 'fold',
)

ALL_NAMES: tuple[str, ...] = tuple(sorted(DefaultGrammar.functions))


def resolve_pool(spec: Optional[str]) -> tuple[str, ...]:
    """``'curated'`` (the 18 in ``POOL_NAMES``), ``'all'`` (every
    ``DefaultGrammar`` primitive), or a comma-separated list of names."""
    if spec in (None, '', 'curated'):
        return POOL_NAMES
    if spec == 'all':
        return ALL_NAMES
    names = tuple(n.strip() for n in spec.split(',') if n.strip())
    unknown = [n for n in names if n not in DefaultGrammar.functions]
    if unknown:
        raise ValueError(f"unknown primitives: {unknown}")
    return names


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
