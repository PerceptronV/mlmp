"""Equality-saturation-based simplifier for the MLMP DSL.

Public API:

    from src.lang.simplify import simplify, SimplifyConfig, SimplifyError

    out = simplify(ast)         # uses default validated rules + Phi check

The simplifier rewrites a closed concrete program to a behaviourally
equivalent, smaller form by building an e-graph, saturating it with the
rule set in :mod:`src.lang.simplify.rules`, and extracting the cheapest
representative under tree cost (= AST size).

Programs must be concrete (no :class:`IntHoleNode`); see
:func:`src.lang.simplify.encode.assert_no_holes`.
"""

from .simplifier import simplify, SimplifyConfig, SimplifyError
from .rules import Rule, DEFAULT_RULES
from .saturation import SaturationConfig, SaturationReport

__all__ = [
    "simplify",
    "SimplifyConfig",
    "SimplifyError",
    "Rule",
    "DEFAULT_RULES",
    "SaturationConfig",
    "SaturationReport",
]
