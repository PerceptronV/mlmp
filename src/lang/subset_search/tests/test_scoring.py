"""Tests for the Stage 1 score vector."""

import pytest

from src.lang.enumeration.enumerator import BottomUpEnumerator
from src.lang.subset_search.pool import pool_grammar
from src.lang.subset_search.scoring import score_vector, PARETO_DIMS


@pytest.fixture(scope="module")
def scored_bank():
    g = pool_grammar(('+', 'length', 'take', 'map'))
    enum = BottomUpEnumerator(grammar=g, max_size=4)
    bank = enum.enumerate()
    vec = score_vector(bank, enum.attempts, max_size=4)
    return bank, enum, vec


def test_score_vector_has_all_pareto_dims(scored_bank):
    _, _, vec = scored_bank
    assert set(PARETO_DIMS) == {
        'n_distinct', 'slope', 'density', 'type_coverage', 'spread', 'tail'
    }
    for dim in PARETO_DIMS:
        assert dim in vec


def test_curve_is_cumulative_and_ends_at_n_distinct(scored_bank):
    _, _, vec = scored_bank
    curve = vec['curve']
    assert len(curve) == 4
    assert all(curve[i] <= curve[i + 1] for i in range(len(curve) - 1))
    assert curve[-1] == vec['n_distinct']


def test_scalar_ranges(scored_bank):
    _, enum, vec = scored_bank
    assert vec['n_distinct'] > 0
    assert vec['slope'] >= 1.0          # cumulative counts cannot shrink
    assert 0.0 < vec['density'] <= 1.0  # distinct/attempted
    assert vec['type_coverage'] >= 1
    assert 0.0 <= vec['spread'] <= 1.0  # normalized Hamming
    assert 0.0 <= vec['tail'] <= 1.0


def test_score_vector_is_deterministic(scored_bank):
    bank, enum, vec = scored_bank
    vec2 = score_vector(bank, enum.attempts, max_size=4)
    assert vec == vec2


def test_target_type_restricts_every_dimension(scored_bank):
    bank, enum, vec = scored_bank
    vec_ll = score_vector(bank, enum.attempts, max_size=4,
                          target_type='list[int]')
    # This grammar (+, length, take, map) has quality int-typed behaviors,
    # so the restriction strictly shrinks the count — and density with it,
    # since attempts stays the full enumeration effort.
    assert 0 < vec_ll['n_distinct'] < vec['n_distinct']
    assert vec_ll['density'] < vec['density']
    assert vec_ll['type_coverage'] == 1
    assert vec_ll['curve'][-1] == vec_ll['n_distinct']


def test_empty_bank_gives_zero_vector():
    # Note: any *enumerated* bank contains the input variable x, whose
    # identity fingerprint passes the quality filter — so a truly empty
    # bank must be constructed directly to exercise the zero-guards.
    from src.lang.enumeration.enumerator import ProgramBank
    bank = ProgramBank()
    vec = score_vector(bank, attempts=0, max_size=3)
    assert vec['n_distinct'] == 0
    assert vec['density'] == 0.0
    assert vec['spread'] == 0.0
    assert vec['tail'] == 0.0
    assert vec['curve'] == [0, 0, 0]
