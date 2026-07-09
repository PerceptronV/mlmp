"""Tests for the curated primitive pool and subset generation."""

from math import comb

from src.lang.grammar import DefaultGrammar
from src.lang.subset_search.pool import (
    POOL_NAMES,
    pool_grammar,
    iter_subsets,
    subset_key,
)


def test_pool_has_18_primitives_all_in_default_grammar():
    assert len(POOL_NAMES) == 18
    assert len(set(POOL_NAMES)) == 18
    for name in POOL_NAMES:
        assert name in DefaultGrammar.functions, name


def test_pool_grammar_contains_exactly_the_pool():
    g = pool_grammar()
    assert set(g.names) == set(POOL_NAMES)


def test_iter_subsets_counts():
    subsets = list(iter_subsets(sizes=(5, 6)))
    assert len(subsets) == comb(18, 5) + comb(18, 6)  # 27,132
    assert all(isinstance(s, frozenset) for s in subsets[:10])
    assert len(set(subsets)) == len(subsets)


def test_iter_subsets_respects_pool_override():
    small_pool = ('+', '*', 'map', 'fold', 'take', 'concat')
    subsets = list(iter_subsets(sizes=(5,), pool_names=small_pool))
    assert len(subsets) == comb(6, 5)
    for s in subsets:
        assert s <= frozenset(small_pool)


def test_subset_key_is_sorted_and_stable():
    s = frozenset({'map', '+', 'fold'})
    assert subset_key(s) == '+ fold map'
