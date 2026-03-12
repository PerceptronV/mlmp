"""Test inputs and evaluation for fingerprinting."""

from typing import Any

from ..lang.compiler import JITCompiler
from ..lang.ast_nodes import ASTNode, LambdaNode


DEFAULT_TEST_SUITE: list[list[int]] = [
    [],                         # i0: empty list
    [0],                        # i1: singleton zero
    [3, 1, 2],                  # i2: small unsorted
    [1, 1, 1, 1],              # i3: all duplicates
    [5, 4, 3, 2, 1],           # i4: reverse-sorted
    [1, 2, 3, 4, 5, 6, 7, 8], # i5: longer sorted
    [10, -3, 7, 7, 0],         # i6: negatives, duplicates, zero
    [2, 8, 3, 8, 2, 3],       # i7: multiple repeated values
    [0, 1, 0, 1, 0],          # i8: binary pattern
    [42],                       # i9: singleton nonzero
]


def evaluate_program(
    ast: ASTNode,
    test_suite: list[list[int]],
    jit: JITCompiler,
) -> list[Any | None]:
    """
    Evaluate a program on each test input.

    The AST should be a LambdaNode wrapping the open term.
    Returns a list of results (values or None on failure).
    """
    try:
        compiled = jit.compile(ast)
    except Exception:
        return [None] * len(test_suite)

    results = []
    for inp in test_suite:
        try:
            result = compiled(inp)
            results.append(result)
        except Exception:
            results.append(None)
    return results
