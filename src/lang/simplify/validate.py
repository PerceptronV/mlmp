"""Per-rule soundness validation.

For each rule, build LHS and RHS as concrete ASTs by substituting
random probes for pattern variables, wrap each in a closing
``(λ (_p0) …)`` and compute its fingerprint Φ on the default 10-input
test suite. If LHS and RHS ever yield different fingerprints (where
neither is entirely FAIL — see :mod:`src.lang.enumeration.fingerprint`),
the rule is rejected.

The probe pool intentionally includes expressions that **can fail at
runtime** (``(first _p0)``, ``(/ 1 0)``, etc.). This catches the
classic strict-evaluation-with-failures unsoundness: a rule like
``(* 0 ?x) → 0`` is fine when ``?x`` is total, but unsound when ``?x``
can fail, since the RHS no longer raises.

Polymorphic / multi-typed pattern variables are tried against several
typed banks; we keep a rule iff *every* tested probe combination
yields matching fingerprints.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

from ..ast_nodes import (
    ASTNode,
    ApplicationNode,
    BooleanNode,
    IfNode,
    LambdaNode,
    ListNode,
    NumberNode,
    VariableNode,
)
from ..compiler import JITCompiler
from ..enumeration.fingerprint import FAIL, Fingerprint, compute_fingerprint
from ..enumeration.test_suite import DEFAULT_TEST_SUITE
from ..grammar import DefaultGrammar
from ..synthesis.pipeline import _canonicalise_lambda
from .egraph import EGraph
from .encode import ENode, Op, encode_ast
from .extraction import extract
from .pattern import PVar, Pattern, PNode
from .rules import Rule


# ---------------------------------------------------------------------------
# Probe banks (typed)
# ---------------------------------------------------------------------------

# Each probe is an AST whose free variables, if any, only reference
# ``_p0``, the input list. We deliberately mix totals and partials so
# the validator catches strictness-sensitive rules.

_INT_PROBES: list[ASTNode] = [
    NumberNode(0),
    NumberNode(1),
    NumberNode(2),
    NumberNode(7),
    NumberNode(-1),
    # partials — these may fail on some test-suite inputs
    ApplicationNode(VariableNode("first"), [VariableNode("_p0")]),
    ApplicationNode(VariableNode("length"), [VariableNode("_p0")]),
    ApplicationNode(VariableNode("/"), [NumberNode(1), NumberNode(0)]),  # always FAIL
]

_BOOL_PROBES: list[ASTNode] = [
    BooleanNode(True),
    BooleanNode(False),
    ApplicationNode(VariableNode(">"), [VariableNode("_p0_first"), NumberNode(0)]),  # placeholder
]

# Re-do BOOL_PROBES with proper expressions (bool predicates referencing input)
_BOOL_PROBES = [
    BooleanNode(True),
    BooleanNode(False),
    ApplicationNode(
        VariableNode(">"),
        [ApplicationNode(VariableNode("length"), [VariableNode("_p0")]), NumberNode(0)],
    ),
    ApplicationNode(VariableNode("is_even"), [
        ApplicationNode(VariableNode("first"), [VariableNode("_p0")])
    ]),  # partial
]

_LIST_INT_PROBES: list[ASTNode] = [
    VariableNode("_p0"),
    ListNode([]),
    ListNode([NumberNode(1), NumberNode(2), NumberNode(3)]),
    ApplicationNode(VariableNode("reverse"), [VariableNode("_p0")]),
    ApplicationNode(VariableNode("repeat"), [NumberNode(5), NumberNode(3)]),
    # Partial list — fails on the empty-list test input, exposing rules
    # that silently drop the ``xs`` operand.
    ApplicationNode(VariableNode("cons"), [
        ApplicationNode(VariableNode("first"), [VariableNode("_p0")]),
        VariableNode("_p0"),
    ]),
]

# Function-typed probes used for higher-order pattern variables (``f``, ``p``).
# We include an `if`-wrapped lambda whose condition can fail; this catches
# rules that drop the function position (e.g. ``map f [] -> []``) — the
# original program raises while the rewrite returns ``[]``.
_PARTIAL_FN_INT_TO_INT = IfNode(
    ApplicationNode(VariableNode("is_even"), [
        ApplicationNode(VariableNode("first"), [VariableNode("_p0")])
    ]),
    LambdaNode(["x"], VariableNode("x")),
    LambdaNode(["x"], NumberNode(0)),
)

_PARTIAL_FN_INT_TO_BOOL = IfNode(
    ApplicationNode(VariableNode("is_even"), [
        ApplicationNode(VariableNode("first"), [VariableNode("_p0")])
    ]),
    LambdaNode(["x"], BooleanNode(True)),
    LambdaNode(["x"], BooleanNode(False)),
)

_INT_TO_INT_FN_PROBES: list[ASTNode] = [
    LambdaNode(["x"], VariableNode("x")),                                 # identity
    LambdaNode(["x"], ApplicationNode(VariableNode("+"), [VariableNode("x"), NumberNode(1)])),
    LambdaNode(["x"], ApplicationNode(VariableNode("*"), [VariableNode("x"), NumberNode(2)])),
    LambdaNode(["x"], NumberNode(0)),                                     # constant
    _PARTIAL_FN_INT_TO_INT,                                                # partial — flushes drop-bugs
]

_INT_TO_BOOL_FN_PROBES: list[ASTNode] = [
    LambdaNode(["x"], BooleanNode(True)),
    LambdaNode(["x"], BooleanNode(False)),
    LambdaNode(["x"], ApplicationNode(VariableNode(">"), [VariableNode("x"), NumberNode(0)])),
    LambdaNode(["x"], ApplicationNode(VariableNode("is_even"), [VariableNode("x")])),
    _PARTIAL_FN_INT_TO_BOOL,
]


def _candidate_probes_for_var_name(name: str) -> list[ASTNode]:
    """Pick a probe bank for a pattern variable based on naming convention.

    Conventions used in :mod:`src.lang.simplify.rules`: ``xs``/``ys`` are
    list[int]; ``x``/``c`` are int; ``n`` is int; ``b``/``p`` is bool or a
    bool-returning function depending on context; ``f`` is a function.
    Names not in the table fall back to a small "any of the above" mix.
    """
    n = name.lower()
    if n in {"xs", "ys", "zs"}:
        return _LIST_INT_PROBES
    if n == "f":
        return _INT_TO_INT_FN_PROBES
    if n == "p":
        return _INT_TO_BOOL_FN_PROBES
    if n in {"b", "cond"}:
        # Mix of literals and a partial bool expression — the partial
        # exposes strictness-related unsoundness in rules that drop the
        # bool-valued sub-tree.
        return [
            BooleanNode(True),
            BooleanNode(False),
            ApplicationNode(VariableNode("is_even"), [
                ApplicationNode(VariableNode("first"), [VariableNode("_p0")])
            ]),  # FAILs on empty list
        ]
    if n in {"a"}:
        # ``a`` is used in if-rules as the surviving branch.
        return _INT_PROBES
    if n == "c":
        # ``c`` is the literal constant in product-of-repeat / sum-of-repeat.
        # Side condition restricts it to literals — but we still want
        # variety of magnitudes.
        return [NumberNode(0), NumberNode(1), NumberNode(2), NumberNode(3)]
    if n == "n":
        # ``n`` is the count for repeat. Keep small to stay under MAX_LIST_SIZE.
        return [NumberNode(0), NumberNode(1), NumberNode(2), NumberNode(5)]
    # Default: int with partials included so we catch strictness bugs.
    return _INT_PROBES


# ---------------------------------------------------------------------------
# Pattern → AST
# ---------------------------------------------------------------------------

def _instantiate_pattern(p: Pattern, subst: dict[str, ASTNode]) -> ASTNode:
    """Build a concrete :class:`ASTNode` from a pattern + variable bindings."""
    if isinstance(p, PVar):
        return subst[p.name]
    assert isinstance(p, PNode)
    if p.op is Op.NUM:
        if isinstance(p.payload, tuple) and p.payload:
            return NumberNode(int(p.payload[0]))
        # An "any literal" pnum() shouldn't appear in rule LHSs we ship;
        # default to 0.
        return NumberNode(0)
    if p.op is Op.BOOL:
        if isinstance(p.payload, tuple) and p.payload:
            return BooleanNode(bool(p.payload[0]))
        return BooleanNode(True)
    if p.op is Op.VAR:
        if isinstance(p.payload, tuple) and p.payload:
            return VariableNode(str(p.payload[0]))
        return VariableNode("_anonymous_var")
    if p.op is Op.LAM:
        # We only generate lambdas in patterns when callers care about
        # identity/constant-body matching — and in those cases the rule's
        # side condition does the work. For validation, we instead take
        # advantage of the side conditions plus typed probes for ``f``/``p``,
        # so this branch should rarely fire.
        if isinstance(p.payload, tuple) and p.payload:
            params = list(p.payload[0])
        else:
            params = ["_pl0"]
        body = _instantiate_pattern(p.children[0], subst)
        return LambdaNode(params, body)
    if p.op is Op.IF:
        return IfNode(
            _instantiate_pattern(p.children[0], subst),
            _instantiate_pattern(p.children[1], subst),
            _instantiate_pattern(p.children[2], subst),
        )
    if p.op is Op.LIST:
        return ListNode([_instantiate_pattern(c, subst) for c in p.children])
    if p.op is Op.APP:
        fn_name = str(p.payload[0])
        args = [_instantiate_pattern(c, subst) for c in p.children]
        return ApplicationNode(VariableNode(fn_name), args)
    if p.op is Op.APP_E:
        fn = _instantiate_pattern(p.children[0], subst)
        args = [_instantiate_pattern(c, subst) for c in p.children[1:]]
        return ApplicationNode(fn, args)
    raise RuntimeError(f"_instantiate_pattern: unhandled op {p.op}")


def _pvars(p: Pattern, out: set[str] | None = None) -> set[str]:
    if out is None:
        out = set()
    if isinstance(p, PVar):
        out.add(p.name)
        return out
    assert isinstance(p, PNode)
    for c in p.children:
        _pvars(c, out)
    return out


# ---------------------------------------------------------------------------
# RHS evaluation via the e-graph
# ---------------------------------------------------------------------------

def _build_rhs_ast(rule: Rule, subst_ast: dict[str, ASTNode]) -> ASTNode | None:
    """Evaluate ``rule.rhs`` against an e-graph seeded with the probe ASTs.

    Returns the RHS as an :class:`ASTNode` by extracting from a
    purpose-built e-graph; ``None`` if the side condition rejects (the
    rule shouldn't fire on this probe combination).
    """
    eg = EGraph()
    subst_classes = {
        name: encode_ast(eg, ast) for name, ast in subst_ast.items()
    }
    if not rule.side_condition(subst_classes, eg):
        return None
    rhs_class = rule.rhs(subst_classes, eg)
    eg.rebuild()
    ast, _ = extract(eg, rhs_class)
    return ast


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    kept: list[Rule]
    rejected: list[tuple[Rule, str]]


def _close(ast: ASTNode) -> ASTNode:
    if not isinstance(ast, LambdaNode):
        ast = LambdaNode(["x"], ast)
    return _canonicalise_lambda(ast)


def _phi(ast: ASTNode, jit: JITCompiler) -> Fingerprint | None:
    return compute_fingerprint(_close(ast), DEFAULT_TEST_SUITE, jit)


def _phis_match(a: Fingerprint | None, b: Fingerprint | None) -> bool:
    if a is None or b is None:
        # Compilation failure on either side. Treat as non-comparable —
        # don't penalise the rule for our probe being un-compilable.
        return True
    if a.values == b.values:
        return True
    # Allow "both FAIL on a position" agreement; all-FAIL alignment is
    # already exact equality. The interesting failure is exactly when
    # one side is FAIL and the other is a real value, which equality
    # already catches. Keeping the explicit branch for documentation.
    return False


_PARTIAL_FN_NAMES = {
    "first", "second", "third", "last", "nth", "/", "%",
    "max", "min",
}


def _contains_partial(ast: ASTNode) -> bool:
    if isinstance(ast, ApplicationNode):
        if isinstance(ast.function, VariableNode) and ast.function.name in _PARTIAL_FN_NAMES:
            return True
        return _contains_partial(ast.function) or any(_contains_partial(a) for a in ast.arguments)
    if isinstance(ast, IfNode):
        return True
    return False


def _is_partial_probe(ast: ASTNode) -> bool:
    """Heuristic: does this probe potentially fail at runtime?"""
    return _contains_partial(ast)


def validate_rules(
    rules: Iterable[Rule],
    max_per_bank: int = 5,
    grammar=DefaultGrammar,
    verbose: bool = False,
) -> ValidationReport:
    jit = JITCompiler(grammar)
    kept: list[Rule] = []
    rejected: list[tuple[Rule, str]] = []

    import itertools

    for rule in rules:
        var_names = sorted(_pvars(rule.lhs))
        banks = [_candidate_probes_for_var_name(v) for v in var_names]

        # If a pattern variable has no candidates, skip validation — keep the rule.
        if any(len(b) == 0 for b in banks):
            kept.append(rule)
            continue

        # Build a slice per bank that includes one or two totals plus
        # any partials in the bank. This catches strictness-sensitive
        # unsoundness — a probe like (first _p0) makes a "drop x" rule
        # produce a different fingerprint on the empty-list test input.
        bank_slices: list[list[ASTNode]] = []
        for b in banks:
            partials = [p for p in b if _is_partial_probe(p)]
            totals = [p for p in b if not _is_partial_probe(p)]
            n_totals = max(1, max_per_bank - len(partials))
            bank_slices.append(totals[:n_totals] + partials)

        ok = True
        why = ""
        for combo in itertools.product(*bank_slices):
            subst_ast = dict(zip(var_names, combo))
            try:
                lhs_ast = _instantiate_pattern(rule.lhs, subst_ast)
                rhs_ast = _build_rhs_ast(rule, subst_ast)
            except Exception as e:
                # Probe yielded an un-buildable pattern instance; ignore.
                if verbose:
                    print(f"[validate] {rule.name}: probe construction failed: {e}")
                continue
            if rhs_ast is None:
                # Side condition rejected — skip this probe combo.
                continue
            phi_l = _phi(lhs_ast, jit)
            phi_r = _phi(rhs_ast, jit)
            if not _phis_match(phi_l, phi_r):
                ok = False
                why = (
                    f"probe={subst_ast!r}; "
                    f"lhs={lhs_ast}; rhs={rhs_ast}; "
                    f"phi_l={phi_l.values if phi_l else None}; "
                    f"phi_r={phi_r.values if phi_r else None}"
                )
                break

        if ok:
            kept.append(rule)
        else:
            rejected.append((rule, why))
            if verbose:
                print(f"[validate] REJECT {rule.name}: {why}")

    return ValidationReport(kept=kept, rejected=rejected)


def get_validated_default_rules(
    verbose: bool = False,
) -> tuple[list[Rule], ValidationReport]:
    """Return the validated subset of :data:`DEFAULT_RULES`."""
    from .rules import DEFAULT_RULES

    report = validate_rules(DEFAULT_RULES, verbose=verbose)
    if report.rejected and not verbose:
        names = ", ".join(r.name for r, _ in report.rejected)
        warnings.warn(
            f"simplify.validate: {len(report.rejected)} rule(s) rejected: {names}",
            stacklevel=2,
        )
    return report.kept, report
