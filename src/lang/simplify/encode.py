"""Adapter between :mod:`src.lang.ast_nodes` and the e-graph's flat e-nodes.

Each AST shape projects onto an :class:`Op` discriminator with non-child
data carried in :attr:`ENode.payload`. Costs in :mod:`extraction` rely on
this encoding being one-to-one with :func:`src.lang.utils.program_size`,
so e-graph tree cost equals AST size.

Encoding table:

==================================================  ======  =============================  ==================
AST shape                                           ``Op``  payload                         children
==================================================  ======  =============================  ==================
``NumberNode(v)``                                   NUM     ``(v,)``                       ``()``
``BooleanNode(v)``                                  BOOL    ``(v,)``                       ``()``
``VariableNode(name)``                              VAR     ``(name,)``                    ``()``
``LambdaNode([p…], body)``                          LAM     ``(tuple(p…),)``               ``(body,)``
``IfNode(c, t, e)``                                 IF      ``()``                         ``(c, t, e)``
``ListNode([e…])``                                  LIST    ``()``                         ``(e0, …)``
``ApplicationNode(VariableNode(name), [a…])``       APP     ``(name,)``                    ``(a0, …)``
``ApplicationNode(other_expr, [a…])``               APP_E   ``()``                         ``(fn, a0, …)``
==================================================  ======  =============================  ==================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from ..ast_nodes import (
    ASTNode,
    ApplicationNode,
    BooleanNode,
    IfNode,
    IntHoleNode,
    LambdaNode,
    ListNode,
    NumberNode,
    VariableNode,
)

if TYPE_CHECKING:
    from .egraph import EClassId, EGraph


class Op(Enum):
    NUM = auto()
    BOOL = auto()
    VAR = auto()
    LAM = auto()
    IF = auto()
    LIST = auto()
    APP = auto()    # function is a primitive name (in payload)
    APP_E = auto()  # function is itself an expression (children[0])


@dataclass(frozen=True)
class ENode:
    """Flat e-node: ``(op, children, payload)``.

    ``children`` is a tuple of e-class ids; ``payload`` is an immutable
    tuple of any extra data (literal value, variable name, lambda
    parameter list).
    """

    op: Op
    children: tuple = ()
    payload: tuple = ()


class HoleEncountered(ValueError):
    """Raised when an :class:`IntHoleNode` is encountered during encoding."""


def assert_no_holes(ast: ASTNode) -> None:
    """Raise :class:`HoleEncountered` if any sub-tree is an :class:`IntHoleNode`."""
    if isinstance(ast, IntHoleNode):
        raise HoleEncountered(
            "simplify() requires concrete programs; encountered IntHoleNode"
        )
    if isinstance(ast, NumberNode) or isinstance(ast, BooleanNode) or isinstance(ast, VariableNode):
        return
    if isinstance(ast, LambdaNode):
        assert_no_holes(ast.body)
        return
    if isinstance(ast, ApplicationNode):
        assert_no_holes(ast.function)
        for a in ast.arguments:
            assert_no_holes(a)
        return
    if isinstance(ast, ListNode):
        for e in ast.elements:
            assert_no_holes(e)
        return
    if isinstance(ast, IfNode):
        assert_no_holes(ast.condition)
        assert_no_holes(ast.then_expr)
        assert_no_holes(ast.else_expr)
        return
    raise TypeError(f"Unknown AST node type: {type(ast).__name__}")


def encode_ast(eg: "EGraph", ast: ASTNode) -> "EClassId":
    """Insert ``ast`` into ``eg`` and return its e-class id."""
    if isinstance(ast, NumberNode):
        return eg.add(ENode(Op.NUM, (), (ast.value,)))
    if isinstance(ast, BooleanNode):
        return eg.add(ENode(Op.BOOL, (), (ast.value,)))
    if isinstance(ast, VariableNode):
        return eg.add(ENode(Op.VAR, (), (ast.name,)))
    if isinstance(ast, LambdaNode):
        body_id = encode_ast(eg, ast.body)
        return eg.add(ENode(Op.LAM, (body_id,), (tuple(ast.param),)))
    if isinstance(ast, IfNode):
        c = encode_ast(eg, ast.condition)
        t = encode_ast(eg, ast.then_expr)
        e = encode_ast(eg, ast.else_expr)
        return eg.add(ENode(Op.IF, (c, t, e), ()))
    if isinstance(ast, ListNode):
        children = tuple(encode_ast(eg, e) for e in ast.elements)
        return eg.add(ENode(Op.LIST, children, ()))
    if isinstance(ast, ApplicationNode):
        if isinstance(ast.function, VariableNode):
            args = tuple(encode_ast(eg, a) for a in ast.arguments)
            return eg.add(ENode(Op.APP, args, (ast.function.name,)))
        # Non-name function (rare): place fn as the first child.
        fn = encode_ast(eg, ast.function)
        args = tuple(encode_ast(eg, a) for a in ast.arguments)
        return eg.add(ENode(Op.APP_E, (fn,) + args, ()))
    if isinstance(ast, IntHoleNode):
        raise HoleEncountered("encode_ast: IntHoleNode is not supported")
    raise TypeError(f"Unknown AST node type: {type(ast).__name__}")


def decode_term(
    eg: "EGraph",
    root: "EClassId",
    choice: dict["EClassId", ENode],
) -> ASTNode:
    """Reconstruct an :class:`ASTNode` by following ``choice[c]`` recursively.

    ``choice`` must contain an entry for every reachable e-class. The
    extractor produces a complete, acyclic ``choice`` map.
    """
    visited: set[int] = set()

    def go(c: "EClassId") -> ASTNode:
        c = eg.find(c)
        if c in visited:
            raise RuntimeError(f"decode_term: cycle through e-class {c}")
        visited.add(c)
        try:
            n = choice[c]
        except KeyError as exc:
            raise RuntimeError(f"decode_term: no choice for e-class {c}") from exc
        try:
            if n.op is Op.NUM:
                return NumberNode(int(n.payload[0]))
            if n.op is Op.BOOL:
                return BooleanNode(bool(n.payload[0]))
            if n.op is Op.VAR:
                return VariableNode(str(n.payload[0]))
            if n.op is Op.LAM:
                params = list(n.payload[0])
                body = go(n.children[0])
                return LambdaNode(params, body)
            if n.op is Op.IF:
                return IfNode(go(n.children[0]), go(n.children[1]), go(n.children[2]))
            if n.op is Op.LIST:
                return ListNode([go(ch) for ch in n.children])
            if n.op is Op.APP:
                fn = VariableNode(str(n.payload[0]))
                args = [go(ch) for ch in n.children]
                return ApplicationNode(fn, args)
            if n.op is Op.APP_E:
                fn = go(n.children[0])
                args = [go(ch) for ch in n.children[1:]]
                return ApplicationNode(fn, args)
        finally:
            visited.discard(c)
        raise RuntimeError(f"decode_term: unknown op {n.op}")

    return go(root)
