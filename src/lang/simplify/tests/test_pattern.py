"""Tests for e-matching."""

from src.lang.parser import parse
from src.lang.simplify.egraph import EGraph
from src.lang.simplify.encode import encode_ast
from src.lang.simplify.pattern import (
    PVar,
    ematch,
    papp,
    pbool,
    pif,
    pnum,
)


def _enc(s: str) -> tuple[EGraph, int]:
    eg = EGraph()
    return eg, encode_ast(eg, parse(s))


def test_ematch_finds_simple_pattern() -> None:
    eg, root = _enc("(reverse (reverse [1 2 3]))")
    pattern = papp("reverse", papp("reverse", PVar("xs")))
    matches = list(ematch(pattern, root, eg))
    assert len(matches) == 1
    assert "xs" in matches[0]


def test_ematch_repeated_var_consistency_required() -> None:
    eg, root = _enc("(if true 5 5)")
    p_match = pif(PVar("c"), PVar("a"), PVar("a"))
    matches = list(ematch(p_match, root, eg))
    assert len(matches) == 1, "(if true 5 5) should match (if c a a)"

    eg2, root2 = _enc("(if true 5 6)")
    matches2 = list(ematch(p_match, root2, eg2))
    assert not matches2, "(if true 5 6) should NOT match (if c a a)"


def test_ematch_literal_payload() -> None:
    eg, root = _enc("(+ 5 0)")
    p = papp("+", PVar("x"), pnum(0))
    matches = list(ematch(p, root, eg))
    assert len(matches) == 1


def test_ematch_no_match_on_different_op() -> None:
    eg, root = _enc("(+ 1 2)")
    p = papp("*", PVar("x"), PVar("y"))
    assert not list(ematch(p, root, eg))


def test_ematch_bool_literal() -> None:
    eg, root = _enc("(if true 1 2)")
    p = pif(pbool(True), PVar("a"), PVar("b"))
    matches = list(ematch(p, root, eg))
    assert len(matches) == 1
