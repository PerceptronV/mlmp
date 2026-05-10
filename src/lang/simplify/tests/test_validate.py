"""Tests for the rule validator.

Sanity-check that:
  * The default rule set passes validation, leaving a non-empty set kept.
  * An injected unsound rule is rejected.
"""

from src.lang.simplify.encode import ENode, Op
from src.lang.simplify.pattern import PVar, papp, pnum
from src.lang.simplify.rules import DEFAULT_RULES, Rule
from src.lang.simplify.validate import get_validated_default_rules, validate_rules


def test_default_rules_validate() -> None:
    kept, _ = get_validated_default_rules()
    assert len(kept) > 0
    # The clearly-safe core must survive validation.
    survivors = {r.name for r in kept}
    for must_keep in {
        "rev-rev",
        "not-not",
        "concat-empty-r",
        "concat-empty-l",
        "add-zero-r",
        "add-zero-l",
        "mul-one-r",
        "mul-one-l",
        "if-true",
        "if-false",
        "map-id",
        "filter-true",
        "reverse-of-repeat",
    }:
        assert must_keep in survivors, f"validator dropped sound rule: {must_keep}"


def test_unsound_rule_rejected() -> None:
    # `(+ x y) -> x` is plainly unsound: ignores y.
    bad = Rule(
        name="bogus-add",
        lhs=papp("+", PVar("x"), PVar("y")),
        rhs=lambda s, eg: s["x"],
    )
    report = validate_rules([bad])
    assert not report.kept, "unsound rule should be rejected"
    assert report.rejected and report.rejected[0][0].name == "bogus-add"


def test_strictness_unsoundness_caught() -> None:
    """The rule (* 0 x) -> 0 is unsound under strict-with-failures
    semantics. The validator must reject it via a partial probe."""
    bad = Rule(
        name="bogus-mul-zero",
        lhs=papp("*", pnum(0), PVar("x")),
        rhs=lambda s, eg: eg.add(ENode(Op.NUM, (), (0,))),
    )
    report = validate_rules([bad])
    assert not report.kept
