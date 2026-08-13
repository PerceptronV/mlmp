"""Tests for support-set extraction and Stage 0 proxy scoring."""

from src.lang.ast_nodes import ApplicationNode, LambdaNode, VariableNode
from src.lang.enumeration.enumerator import BottomUpEnumerator
from src.lang.subset_search.pool import pool_grammar
from src.lang.subset_search.support import (
    support_set,
    support_buckets,
    proxy_score,
)


POOL = frozenset({'+', 'map', 'length'})


def test_support_set_intersects_pool_and_ignores_variables():
    # (map (λ (p) (+ p p)) x) — uses map and + from the pool; x and p are vars
    ast = ApplicationNode(
        VariableNode('map'),
        [
            LambdaNode(['p'], ApplicationNode(
                VariableNode('+'), [VariableNode('p'), VariableNode('p')]
            )),
            VariableNode('x'),
        ],
    )
    assert support_set(ast, POOL) == frozenset({'map', '+'})


def test_support_set_of_bare_variable_is_empty():
    assert support_set(VariableNode('x'), POOL) == frozenset()


def test_proxy_score_sums_contained_buckets():
    buckets = {
        frozenset(): 1,                    # e.g. the identity program
        frozenset({'+'}): 4,
        frozenset({'map', '+'}): 7,
        frozenset({'length'}): 2,
        frozenset({'map', 'length'}): 3,
    }
    assert proxy_score(frozenset({'+', 'map'}), buckets) == 1 + 4 + 7
    assert proxy_score(frozenset({'length'}), buckets) == 1 + 2
    assert proxy_score(frozenset(), buckets) == 1


def test_support_buckets_counts_only_quality_fingerprints():
    g = pool_grammar(('+', 'length', 'take'))
    enum = BottomUpEnumerator(grammar=g, max_size=3)
    bank = enum.enumerate()
    pool = frozenset({'+', 'length', 'take'})
    buckets = support_buckets(bank, pool)
    assert all(s <= pool for s in buckets)
    assert all(n > 0 for n in buckets.values())
    total_quality = sum(buckets.values())
    assert 0 < total_quality <= bank.count()


def test_support_buckets_target_type_restricts_counts():
    g = pool_grammar(('+', 'length', 'take'))
    enum = BottomUpEnumerator(grammar=g, max_size=3)
    bank = enum.enumerate()
    pool = frozenset({'+', 'length', 'take'})
    unrestricted = support_buckets(bank, pool)
    restricted = support_buckets(bank, pool, target_type='list[int]')

    # A strict subset of the unrestricted counts: this grammar has quality
    # int-typed behaviors (length, +) that the restriction must drop.
    assert 0 < sum(restricted.values()) < sum(unrestricted.values())
    assert all(restricted[s] <= unrestricted[s] for s in restricted)

    # Restricted totals equal the number of quality list[int] behaviors.
    from src.lang.enumeration.filters import passes_quality_filter
    n_ll = sum(
        1 for p in bank.all_programs()
        if p.fingerprint is not None
        and passes_quality_filter(p.fingerprint)
        and str(p.type) == 'list[int]'
    )
    assert sum(restricted.values()) == n_ll
