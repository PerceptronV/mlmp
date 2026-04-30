"""Rewrite rules for the MLMP DSL.

The :data:`DEFAULT_RULES` table encodes the §6.2 rule set from
``docs/program-simplification.tex``, adjusted for the actual evaluator
semantics:

* No implicit ``mod 100`` reduction in the constant-folding RHSs — the
  runtime evaluator (``src/lang/evaluator.py``, ``src/lang/grammar.py``)
  does not apply it; that is a harness-level concern, and the validator
  in :mod:`src.lang.simplify.validate` would reject any RHS that
  diverges from the runtime.

Lambda-synthesising fusion rules (e.g. ``map (map g) → map (∘ f g)``)
are deferred: they require a fresh-name generator and ``Op.APP_E``
support for partially-applied functions. The simpler rules below cover
the redundancies flagged in §A.3 of ``docs/enumeration-rl.tex``
(constant sub-trees, double-reverse, identity-map, length distribution,
etc.) which are by far the most common source of corpus bloat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .egraph import EClassId, EGraph
from .encode import ENode, Op
from .pattern import (
    Pattern,
    PVar,
    get_literal_bool,
    get_literal_int,
    is_empty_list,
    papp,
    pbool,
    pempty_list,
    pif,
    plam,
    pnum,
    pvar,
)


Subst = dict[str, EClassId]
SideCondition = Callable[[Subst, EGraph], bool]
RHSBuilder = Callable[[Subst, EGraph], EClassId]


def _always_true(subst: Subst, eg: EGraph) -> bool:
    return True


@dataclass(frozen=True)
class Rule:
    name: str
    lhs: Pattern
    rhs: RHSBuilder
    side_condition: SideCondition = _always_true


# ---------------------------------------------------------------------------
# RHS / side-condition helpers
# ---------------------------------------------------------------------------

def _bind(name: str) -> RHSBuilder:
    """RHS that returns the e-class bound to ``name`` in the substitution."""
    def build(subst: Subst, eg: EGraph) -> EClassId:
        return subst[name]
    return build


def _const_int(value: int) -> RHSBuilder:
    def build(subst: Subst, eg: EGraph) -> EClassId:
        return eg.add(ENode(Op.NUM, (), (int(value),)))
    return build


def _empty_list_rhs(subst: Subst, eg: EGraph) -> EClassId:
    return eg.add(ENode(Op.LIST, (), ()))


def _is_identity_lambda(name: str) -> SideCondition:
    """``?name`` matches an e-class containing ``(λ (p) p)`` for some p."""
    def cond(subst: Subst, eg: EGraph) -> bool:
        c = eg.find(subst[name])
        for n in eg.nodes_in(c):
            if n.op is not Op.LAM:
                continue
            param_tuple = n.payload[0]
            if len(param_tuple) != 1:
                continue
            param = param_tuple[0]
            body_class = n.children[0]
            for body_n in eg.nodes_in(eg.find(body_class)):
                if body_n.op is Op.VAR and body_n.payload == (param,):
                    return True
        return False
    return cond


def _is_constant_lambda(name: str, op: Op, value) -> SideCondition:
    """``?name`` matches a lambda whose body is the literal ``op(value)``."""
    def cond(subst: Subst, eg: EGraph) -> bool:
        c = eg.find(subst[name])
        for n in eg.nodes_in(c):
            if n.op is not Op.LAM:
                continue
            body_class = n.children[0]
            for body_n in eg.nodes_in(eg.find(body_class)):
                if body_n.op is op and body_n.payload == (value,):
                    return True
        return False
    return cond


def _both_int_literals(*names: str) -> SideCondition:
    def cond(subst: Subst, eg: EGraph) -> bool:
        return all(get_literal_int(subst[n], eg) is not None for n in names)
    return cond


def _int_literal_at_least(name: str, lo: int) -> SideCondition:
    def cond(subst: Subst, eg: EGraph) -> bool:
        v = get_literal_int(subst[name], eg)
        return v is not None and v >= lo
    return cond


# Match the runtime's MAX_LIST_SIZE so constant-folding rules that drop a
# `repeat` only fire when the list it would have built is small enough
# not to trigger ListSizeExceeded — keeping the rules sound under the
# evaluator's strict semantics.
_MAX_LIST_SIZE = 1000


def _int_literal_in_range(name: str, lo: int, hi: int) -> SideCondition:
    def cond(subst: Subst, eg: EGraph) -> bool:
        v = get_literal_int(subst[name], eg)
        return v is not None and lo <= v <= hi
    return cond


def _is_lambda(name: str) -> SideCondition:
    """``?name`` matches an e-class containing a (syntactic) lambda.

    Lambdas evaluate to closures without failure, so a rule with this
    side condition can drop the function position safely.
    """
    def cond(subst: Subst, eg: EGraph) -> bool:
        c = eg.find(subst[name])
        for n in eg.nodes_in(c):
            if n.op is Op.LAM:
                return True
        return False
    return cond


# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[Rule] = []


def _add(rule: Rule) -> None:
    DEFAULT_RULES.append(rule)


# ---- identity laws --------------------------------------------------------

_add(Rule(
    name="map-id",
    lhs=papp("map", PVar("f"), PVar("xs")),
    rhs=_bind("xs"),
    side_condition=_is_identity_lambda("f"),
))

_add(Rule(
    name="filter-true",
    lhs=papp("filter", PVar("p"), PVar("xs")),
    rhs=_bind("xs"),
    side_condition=_is_constant_lambda("p", Op.BOOL, True),
))

_add(Rule(
    name="concat-empty-r",
    lhs=papp("concat", PVar("xs"), pempty_list()),
    rhs=_bind("xs"),
    side_condition=_always_true,
))

_add(Rule(
    name="concat-empty-l",
    lhs=papp("concat", pempty_list(), PVar("xs")),
    rhs=_bind("xs"),
    side_condition=_always_true,
))

_add(Rule(
    name="add-zero-r",
    lhs=papp("+", PVar("x"), pnum(0)),
    rhs=_bind("x"),
    side_condition=_always_true,
))

_add(Rule(
    name="add-zero-l",
    lhs=papp("+", pnum(0), PVar("x")),
    rhs=_bind("x"),
    side_condition=_always_true,
))

_add(Rule(
    name="mul-one-r",
    lhs=papp("*", PVar("x"), pnum(1)),
    rhs=_bind("x"),
    side_condition=_always_true,
))

_add(Rule(
    name="mul-one-l",
    lhs=papp("*", pnum(1), PVar("x")),
    rhs=_bind("x"),
    side_condition=_always_true,
))

# ---- annihilators ---------------------------------------------------------

_add(Rule(
    name="mul-zero-r",
    lhs=papp("*", PVar("x"), pnum(0)),
    rhs=_const_int(0),
    side_condition=_always_true,
))

_add(Rule(
    name="mul-zero-l",
    lhs=papp("*", pnum(0), PVar("x")),
    rhs=_const_int(0),
    side_condition=_always_true,
))

_add(Rule(
    name="filter-false",
    lhs=papp("filter", PVar("p"), PVar("xs")),
    rhs=_empty_list_rhs,
    side_condition=_is_constant_lambda("p", Op.BOOL, False),
))

_add(Rule(
    name="map-empty",
    lhs=papp("map", PVar("f"), pempty_list()),
    rhs=_empty_list_rhs,
    side_condition=_always_true,
))

# ---- involutions ----------------------------------------------------------

_add(Rule(
    name="rev-rev",
    lhs=papp("reverse", papp("reverse", PVar("xs"))),
    rhs=_bind("xs"),
    side_condition=_always_true,
))

_add(Rule(
    name="not-not",
    lhs=papp("not", papp("not", PVar("b"))),
    rhs=_bind("b"),
    side_condition=_always_true,
))

# ---- conditional collapse -------------------------------------------------

_add(Rule(
    name="if-true",
    lhs=pif(pbool(True), PVar("a"), PVar("b")),
    rhs=_bind("a"),
    side_condition=_always_true,
))

_add(Rule(
    name="if-false",
    lhs=pif(pbool(False), PVar("a"), PVar("b")),
    rhs=_bind("b"),
    side_condition=_always_true,
))

_add(Rule(
    name="if-same",
    lhs=pif(PVar("cond"), PVar("a"), PVar("a")),  # repeated var: both branches identical
    rhs=_bind("a"),
    side_condition=_always_true,
))

# ---- constant folding -----------------------------------------------------

_add(Rule(
    name="length-of-repeat",
    lhs=papp("length", papp("repeat", PVar("x"), PVar("n"))),
    rhs=_bind("n"),
    # Both args must be literals: ``x`` so we don't drop a partial
    # expression, ``n`` so the original ``repeat`` can't trip
    # ListSizeExceeded.
    side_condition=lambda s, eg: (
        get_literal_int(s["x"], eg) is not None
        and _int_literal_in_range("n", 0, _MAX_LIST_SIZE)(s, eg)
    ),
))


def _product_repeat_rhs(subst: Subst, eg: EGraph) -> EClassId:
    c = get_literal_int(subst["c"], eg)
    n = get_literal_int(subst["n"], eg)
    # Side condition guarantees both are non-None and n >= 0.
    return eg.add(ENode(Op.NUM, (), (int(c) ** int(n),)))


_add(Rule(
    name="product-of-repeat",
    lhs=papp("product", papp("repeat", PVar("c"), PVar("n"))),
    rhs=_product_repeat_rhs,
    side_condition=lambda s, eg: (
        _both_int_literals("c", "n")(s, eg)
        and _int_literal_in_range("n", 0, _MAX_LIST_SIZE)(s, eg)
    ),
))


def _sum_repeat_rhs(subst: Subst, eg: EGraph) -> EClassId:
    c = get_literal_int(subst["c"], eg)
    n = get_literal_int(subst["n"], eg)
    return eg.add(ENode(Op.NUM, (), (int(c) * int(n),)))


_add(Rule(
    name="sum-of-repeat",
    lhs=papp("sum", papp("repeat", PVar("c"), PVar("n"))),
    rhs=_sum_repeat_rhs,
    side_condition=lambda s, eg: (
        _both_int_literals("c", "n")(s, eg)
        and _int_literal_in_range("n", 0, _MAX_LIST_SIZE)(s, eg)
    ),
))


def _reverse_repeat_rhs(subst: Subst, eg: EGraph) -> EClassId:
    return eg.add(ENode(Op.APP, (subst["x"], subst["n"]), ("repeat",)))


_add(Rule(
    name="reverse-of-repeat",
    lhs=papp("reverse", papp("repeat", PVar("x"), PVar("n"))),
    rhs=_reverse_repeat_rhs,
    side_condition=_always_true,
))

_add(Rule(
    name="first-of-repeat",
    lhs=papp("first", papp("repeat", PVar("x"), PVar("n"))),
    rhs=_bind("x"),
    # ``n`` must be in [1, MAX_LIST_SIZE]; ``x`` survives in the RHS so
    # no partial-drop concern.
    side_condition=_int_literal_in_range("n", 1, _MAX_LIST_SIZE),
))

# ---- length distribution --------------------------------------------------


def _length_concat_rhs(subst: Subst, eg: EGraph) -> EClassId:
    len_xs = eg.add(ENode(Op.APP, (subst["xs"],), ("length",)))
    len_ys = eg.add(ENode(Op.APP, (subst["ys"],), ("length",)))
    return eg.add(ENode(Op.APP, (len_xs, len_ys), ("+",)))


_add(Rule(
    name="length-of-concat",
    lhs=papp("length", papp("concat", PVar("xs"), PVar("ys"))),
    rhs=_length_concat_rhs,
    side_condition=_always_true,
))


def _length_only_rhs(subst: Subst, eg: EGraph) -> EClassId:
    return eg.add(ENode(Op.APP, (subst["xs"],), ("length",)))


_add(Rule(
    name="length-of-map",
    lhs=papp("length", papp("map", PVar("f"), PVar("xs"))),
    rhs=_length_only_rhs,
    # ``f`` must be a syntactic lambda so its evaluation is total
    # (closure construction can't fail). Without this, a partial ``f``
    # like ``(if c lam1 lam2)`` with failing ``c`` makes the rule
    # unsound.
    side_condition=_is_lambda("f"),
))

_add(Rule(
    name="length-of-reverse",
    lhs=papp("length", papp("reverse", PVar("xs"))),
    rhs=_length_only_rhs,
    side_condition=_always_true,
))

_add(Rule(
    name="length-of-singleton",
    lhs=papp("length", papp("singleton", PVar("x"))),
    rhs=_const_int(1),
    side_condition=_always_true,
))

_add(Rule(
    name="length-of-cons",
    lhs=papp("length", papp("cons", PVar("x"), PVar("xs"))),
    rhs=lambda s, eg: eg.add(ENode(
        Op.APP,
        (eg.add(ENode(Op.APP, (s["xs"],), ("length",))),
         eg.add(ENode(Op.NUM, (), (1,)))),
        ("+",),
    )),
    side_condition=_always_true,
))


# ---- predicate / arithmetic identities ------------------------------------

_add(Rule(
    name="and-true-r",
    lhs=papp("and", PVar("b"), pbool(True)),
    rhs=_bind("b"),
    side_condition=_always_true,
))

_add(Rule(
    name="and-true-l",
    lhs=papp("and", pbool(True), PVar("b")),
    rhs=_bind("b"),
    side_condition=_always_true,
))

_add(Rule(
    name="and-false-r",
    lhs=papp("and", PVar("b"), pbool(False)),
    rhs=lambda s, eg: eg.add(ENode(Op.BOOL, (), (False,))),
    side_condition=_always_true,
))

_add(Rule(
    name="and-false-l",
    lhs=papp("and", pbool(False), PVar("b")),
    rhs=lambda s, eg: eg.add(ENode(Op.BOOL, (), (False,))),
    side_condition=_always_true,
))

_add(Rule(
    name="or-false-r",
    lhs=papp("or", PVar("b"), pbool(False)),
    rhs=_bind("b"),
    side_condition=_always_true,
))

_add(Rule(
    name="or-false-l",
    lhs=papp("or", pbool(False), PVar("b")),
    rhs=_bind("b"),
    side_condition=_always_true,
))

_add(Rule(
    name="or-true-r",
    lhs=papp("or", PVar("b"), pbool(True)),
    rhs=lambda s, eg: eg.add(ENode(Op.BOOL, (), (True,))),
    side_condition=_always_true,
))

_add(Rule(
    name="or-true-l",
    lhs=papp("or", pbool(True), PVar("b")),
    rhs=lambda s, eg: eg.add(ENode(Op.BOOL, (), (True,))),
    side_condition=_always_true,
))

_add(Rule(
    name="sub-zero",
    lhs=papp("-", PVar("x"), pnum(0)),
    rhs=_bind("x"),
    side_condition=_always_true,
))

# ---- list reduction over empty / singleton --------------------------------

_add(Rule(
    name="reverse-empty",
    lhs=papp("reverse", pempty_list()),
    rhs=_empty_list_rhs,
    side_condition=_always_true,
))

_add(Rule(
    name="length-empty",
    lhs=papp("length", pempty_list()),
    rhs=_const_int(0),
    side_condition=_always_true,
))

_add(Rule(
    name="sum-empty",
    lhs=papp("sum", pempty_list()),
    rhs=_const_int(0),
    side_condition=_always_true,
))

_add(Rule(
    name="product-empty",
    lhs=papp("product", pempty_list()),
    rhs=_const_int(1),
    side_condition=_always_true,
))

_add(Rule(
    name="filter-empty",
    lhs=papp("filter", PVar("p"), pempty_list()),
    rhs=_empty_list_rhs,
    side_condition=_always_true,
))


# A frozen tuple of the curated default rules. Use this where mutability
# would be a footgun.
def _frozen_default_rules() -> tuple[Rule, ...]:
    return tuple(DEFAULT_RULES)
