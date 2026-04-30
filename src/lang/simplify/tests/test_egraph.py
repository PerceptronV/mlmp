"""Tests for the e-graph data structure."""

from src.lang.simplify.egraph import EGraph
from src.lang.simplify.encode import ENode, Op


def _num(eg: EGraph, v: int) -> int:
    return eg.add(ENode(Op.NUM, (), (v,)))


def _add(eg: EGraph, a: int, b: int) -> int:
    return eg.add(ENode(Op.APP, (a, b), ("+",)))


def test_hashcons_returns_existing_class() -> None:
    eg = EGraph()
    a = _num(eg, 5)
    b = _num(eg, 5)
    assert eg.find(a) == eg.find(b)


def test_distinct_literals_get_distinct_classes() -> None:
    eg = EGraph()
    a = _num(eg, 5)
    b = _num(eg, 6)
    assert eg.find(a) != eg.find(b)


def test_union_collapses_classes() -> None:
    eg = EGraph()
    a = _num(eg, 5)
    b = _num(eg, 6)
    eg.union(a, b)
    eg.rebuild()
    assert eg.find(a) == eg.find(b)


def test_congruence_propagates_through_parents() -> None:
    """Unioning leaf classes should collapse parent classes that share
    the same operator and now-equal child lists."""
    eg = EGraph()
    a = _num(eg, 5)
    b = _num(eg, 6)
    one = _num(eg, 1)
    p_a = _add(eg, a, one)        # (+ 5 1)
    p_b = _add(eg, b, one)        # (+ 6 1)
    assert eg.find(p_a) != eg.find(p_b)

    eg.union(a, b)
    eg.rebuild()

    assert eg.find(p_a) == eg.find(p_b), \
        "parents (+ a 1) and (+ b 1) should collapse after a == b"


def test_rebuild_handles_chains() -> None:
    eg = EGraph()
    a = _num(eg, 1)
    b = _num(eg, 2)
    c = _num(eg, 3)
    eg.union(a, b)
    eg.union(b, c)
    eg.rebuild()
    assert eg.find(a) == eg.find(b) == eg.find(c)


def test_no_eager_congruence_before_rebuild() -> None:
    """Union should not eagerly resolve congruence; rebuild does."""
    eg = EGraph()
    a = _num(eg, 5)
    b = _num(eg, 6)
    one = _num(eg, 1)
    p_a = _add(eg, a, one)
    p_b = _add(eg, b, one)
    eg.union(a, b)
    # Without rebuild, parents may still appear distinct.
    eg.rebuild()
    assert eg.find(p_a) == eg.find(p_b)
