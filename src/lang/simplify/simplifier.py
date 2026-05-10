"""High-level :func:`simplify` entry point.

Pipeline for a single concrete program ``p``:

1. :func:`assert_no_holes` rejects programs with :class:`IntHoleNode`.
2. Build an :class:`EGraph` and encode ``p``.
3. :func:`saturate` applies the rule set to fixpoint or budget.
4. :func:`extract` returns the cheapest-tree-cost representative.
5. If ``verify_phi`` is set, recompute Φ on input and output and
   compare; mismatch raises :class:`SimplifyError`.
6. The output is alpha-renamed via the existing
   :func:`_canonicalise_lambda` helper to preserve the corpus
   convention of depth-indexed binder names ``_p0, _p1, …``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..ast_nodes import ASTNode, LambdaNode
from ..compiler import JITCompiler
from ..enumeration.fingerprint import Fingerprint, compute_fingerprint
from ..enumeration.test_suite import DEFAULT_TEST_SUITE
from ..grammar import DefaultGrammar
from ..synthesis.pipeline import _canonicalise_lambda
from ..utils import program_size as _program_size
from .cost import CostFn, tree_cost
from .egraph import EGraph
from .encode import HoleEncountered, assert_no_holes, encode_ast
from .extraction import extract
from .rules import DEFAULT_RULES, Rule
from .saturation import SaturationConfig, SaturationReport, saturate
from .validate import get_validated_default_rules


# Validate rules once at import time. Unsound rules (caught via Φ
# disagreement on the test suite with at least one partial probe) are
# dropped with a warning. The validated set is what ``simplify`` uses
# by default.
_VALIDATED_DEFAULT_RULES, _VALIDATION_REPORT = get_validated_default_rules()


class SimplifyError(Exception):
    """Raised when simplification produces a Φ-mismatch or fails internally."""


@dataclass
class SimplifyConfig:
    rules: Iterable[Rule] | None = None
    cost_fn: CostFn = tree_cost
    saturation: SaturationConfig = field(default_factory=SaturationConfig)
    verify_phi: bool = True
    test_suite: list = field(default_factory=lambda: list(DEFAULT_TEST_SUITE))
    return_report: bool = False  # if True, simplify returns (ast, report)


_DEFAULT_JIT: JITCompiler | None = None


def _get_jit() -> JITCompiler:
    """Lazy module-level JIT to avoid the cold-start cost on each call."""
    global _DEFAULT_JIT
    if _DEFAULT_JIT is None:
        _DEFAULT_JIT = JITCompiler(DefaultGrammar)
    return _DEFAULT_JIT


def _close(ast: ASTNode) -> ASTNode:
    """Wrap and re-canonicalise the same way ``synthesis.pipeline`` does."""
    if not isinstance(ast, LambdaNode):
        ast = LambdaNode(["x"], ast)
    return _canonicalise_lambda(ast)


def _phi(ast: ASTNode, test_suite: list, jit: JITCompiler) -> Fingerprint | None:
    return compute_fingerprint(_close(ast), test_suite, jit)


def simplify(
    ast: ASTNode,
    cfg: SimplifyConfig | None = None,
) -> ASTNode | tuple[ASTNode, SaturationReport]:
    """Simplify a closed concrete program.

    Args:
      ast: A closed program (typically a top-level ``LambdaNode``) with
        no :class:`IntHoleNode` anywhere in the tree.
      cfg: Optional :class:`SimplifyConfig`. The default uses
        :data:`DEFAULT_RULES`, tree cost, a 30-iteration / 10K-node /
        5-second budget, and a Φ-preservation safety check.

    Returns:
      The simplified :class:`ASTNode`, with binder names alpha-renamed
      to the corpus convention. If ``cfg.return_report`` is set, a
      ``(ast, SaturationReport)`` tuple is returned instead.

    Raises:
      :class:`HoleEncountered` if ``ast`` contains an
        :class:`IntHoleNode`.
      :class:`SimplifyError` if Φ is not preserved (with
        ``verify_phi=True``).
    """
    cfg = cfg or SimplifyConfig()
    assert_no_holes(ast)

    rules = list(cfg.rules) if cfg.rules is not None else list(_VALIDATED_DEFAULT_RULES)

    eg = EGraph()
    root = encode_ast(eg, ast)
    report = saturate(eg, rules, cfg.saturation)
    out, cost = extract(eg, root, cfg.cost_fn)

    # Defensive: if extraction produced something larger than the input
    # under the AST size metric, prefer the input. Should never happen
    # with cost = AST size.
    if _program_size(out) > _program_size(ast):
        out = ast

    out = _canonicalise_lambda(out)

    if cfg.verify_phi:
        jit = _get_jit()
        fp_in = _phi(ast, cfg.test_suite, jit)
        fp_out = _phi(out, cfg.test_suite, jit)
        if fp_in is None or fp_out is None:
            raise SimplifyError(
                f"simplify: failed to fingerprint "
                f"(input fp = {fp_in}, output fp = {fp_out})"
            )
        if fp_in != fp_out:
            raise SimplifyError(
                "simplify: Phi-mismatch — a rule is unsound on this input. "
                f"Input: {ast}; output: {out}; "
                f"in_fp={fp_in.values}; out_fp={fp_out.values}"
            )

    if cfg.return_report:
        return out, report
    return out
