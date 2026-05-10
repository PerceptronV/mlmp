"""Observational equivalence via fingerprinting."""

from typing import Any

from ..compiler import JITCompiler
from ..ast_nodes import ASTNode, LambdaNode


FAIL = object()  # Sentinel for failed evaluations


def make_hashable(value: Any) -> Any:
    """Convert a value to a hashable form. Lists become tuples recursively."""
    if value is None or value is FAIL:
        return FAIL
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    if callable(value):
        return FAIL
    return value


class Fingerprint:
    """Immutable fingerprint for a program's behaviour on the test suite."""

    __slots__ = ('values', '_hash')

    def __init__(self, values: tuple):
        self.values = values
        self._hash = hash(values)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Fingerprint) and self.values == other.values

    def __repr__(self):
        return f"Fingerprint({self.values})"


class FingerprintTable:
    """Hash table mapping fingerprints to canonical program representatives."""

    def __init__(self):
        self.table: dict[Fingerprint, ASTNode] = {}

    def contains(self, fp: Fingerprint) -> bool:
        return fp in self.table

    def insert(self, fp: Fingerprint, program: ASTNode) -> bool:
        """Insert if novel. Returns True if inserted, False if duplicate."""
        if fp in self.table:
            return False
        self.table[fp] = program
        return True

    def __len__(self):
        return len(self.table)

    def programs(self) -> list[ASTNode]:
        return list(self.table.values())

    def items(self) -> list[tuple[Fingerprint, ASTNode]]:
        return list(self.table.items())


def compute_fingerprint(
    closed_ast: ASTNode,
    test_suite: list[list[int]],
    jit: JITCompiler,
) -> 'Fingerprint | None':
    """
    Compute fingerprint by evaluating a closed (lambda-wrapped) AST on the test suite.

    Args:
        closed_ast: A LambdaNode wrapping the program
        test_suite: List of test inputs
        jit: JIT compiler instance

    Returns:
        Fingerprint or None if compilation fails
    """
    try:
        compiled, _ = jit.compile(closed_ast)
    except Exception:
        return None

    values = []
    for inp in test_suite:
        try:
            result = compiled(inp)
            values.append(make_hashable(result))
        except Exception:
            values.append(FAIL)

    return Fingerprint(tuple(values))
