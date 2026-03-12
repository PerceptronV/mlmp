"""Shared AST utilities for enumeration and RL."""

from .lang.ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode,
)
from .lang.type_utils import SubstitutionTable, substitute_type_vars, TypeType
from .lang.grammar import T1, T2


def program_size(node: ASTNode) -> int:
    """
    Compute the size of an AST node.

    For ApplicationNode, the function name is subsumed into the node cost of 1,
    so (length x) = 1 + 1 = 2, (+ x 1) = 1 + 1 + 1 = 3.
    """
    if isinstance(node, (NumberNode, BooleanNode, VariableNode)):
        return 1
    elif isinstance(node, ListNode):
        if not node.elements:
            return 1
        return 1 + sum(program_size(e) for e in node.elements)
    elif isinstance(node, LambdaNode):
        return 1 + program_size(node.body)
    elif isinstance(node, IfNode):
        return 1 + program_size(node.condition) + program_size(node.then_expr) + program_size(node.else_expr)
    elif isinstance(node, ApplicationNode):
        # function name node is subsumed into the 1 cost
        return 1 + sum(program_size(arg) for arg in node.arguments)
    else:
        raise ValueError(f"Unknown node type: {type(node)}")


def free_variables(node: ASTNode, bound: set[str] | None = None) -> set[str]:
    """Return the set of free variable names in node given already-bound names."""
    if bound is None:
        bound = set()

    if isinstance(node, (NumberNode, BooleanNode)):
        return set()
    elif isinstance(node, VariableNode):
        return set() if node.name in bound else {node.name}
    elif isinstance(node, ListNode):
        result = set()
        for elem in node.elements:
            result |= free_variables(elem, bound)
        return result
    elif isinstance(node, LambdaNode):
        return free_variables(node.body, bound | set(node.param))
    elif isinstance(node, IfNode):
        return (
            free_variables(node.condition, bound)
            | free_variables(node.then_expr, bound)
            | free_variables(node.else_expr, bound)
        )
    elif isinstance(node, ApplicationNode):
        result = free_variables(node.function, bound)
        for arg in node.arguments:
            result |= free_variables(arg, bound)
        return result
    return set()


def uses_variable(node: ASTNode, var_name: str) -> bool:
    """Check if var_name appears free in the AST."""
    return var_name in free_variables(node)


def resolve_type(type_: TypeType, T1_val=int, T2_val=int) -> TypeType | None:
    """Resolve type variables using a fresh SubstitutionTable."""
    subs = SubstitutionTable()
    subs[T1] = T1_val
    subs[T2] = T2_val
    try:
        return substitute_type_vars(type_, subs)
    except Exception:
        return None
