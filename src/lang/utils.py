"""Shared AST utilities for enumeration and RL."""

import itertools
from typing import TypeVar

from .ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode, IntHoleNode,
)
from .type_utils import (
    SubstitutionTable, substitute_type_vars, TypeType,
    CallableOrig, get_origin, get_free_types,
)
from .grammar import T1, T2

# ---------------------------------------------------------------------------
# Type universe constants
# ---------------------------------------------------------------------------

TYPE_UNIVERSE = frozenset({int, bool, list[int], list[bool], list[list[int]]})

GROUND_TYPES = [int, bool, list[int]]

PROBE_VALUES = {
    int: [0, 1, 3],
    bool: [True, False],
    list[int]: [[], [1], [2, 1, 3]],
    list[bool]: [[], [True], [True, False]],
}

RANDINT_PROBE_SEQUENCE: list[int] = [5, 3, 7, 2, 9, 0, 8, 1, 6, 4]

Substitution = list[int]  # hole values indexed by pre-order traversal position

# ---------------------------------------------------------------------------
# Instantiation helpers
# ---------------------------------------------------------------------------

def freeze_instantiation(inst: dict) -> tuple:
    """Convert instantiation dict to a hashable tuple for use in Action."""
    return tuple(sorted(inst.items(), key=lambda kv: str(kv[0])))


def thaw_instantiation(frozen: tuple) -> dict:
    """Convert frozen instantiation back to a dict."""
    return dict(frozen)


def compute_valid_instantiations(grammar) -> dict[str, list[dict[TypeVar, TypeType]]]:
    """
    For each grammar function, find all type variable assignments such that
    all argument types and the return type resolve to types within TYPE_UNIVERSE
    (or valid Callable types).
    """
    result = {}

    for func_name in grammar.names:
        func_info = grammar[func_name]

        # Collect free type variables in this function's signature
        free_tvs: set = set()
        subs = SubstitutionTable()
        for t in func_info['arg_types']:
            free_tvs |= get_free_types(t, subs)
        free_tvs |= get_free_types(func_info['ret_type'], subs)

        free_tvs_sorted = sorted(free_tvs, key=str)

        if not free_tvs_sorted:
            result[func_name] = [{}]
            continue

        valid = []
        for assignment in itertools.product(GROUND_TYPES, repeat=len(free_tvs_sorted)):
            inst = dict(zip(free_tvs_sorted, assignment))

            all_ok = True
            for t in func_info['arg_types']:
                resolved = resolve_type(t, instantiation=inst)
                if resolved is None:
                    all_ok = False
                    break
            if all_ok:
                resolved_ret = resolve_type(func_info['ret_type'], instantiation=inst)
                if resolved_ret is None:
                    all_ok = False
            if all_ok:
                valid.append(inst)

        result[func_name] = valid

    return result

# ---------------------------------------------------------------------------
# AST utilities
# ---------------------------------------------------------------------------

def program_size(node: ASTNode) -> int:
    """
    Compute the size of an AST node.

    For ApplicationNode, the function name is subsumed into the node cost of 1,
    so (length x) = 1 + 1 = 2, (+ x 1) = 1 + 1 + 1 = 3.
    """
    if isinstance(node, (NumberNode, BooleanNode, VariableNode, IntHoleNode)):
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


def program_depth(node: ASTNode) -> int:
    """
    Compute the MDP depth of an AST — matches `SynthesisState.depth_budget`
    consumption.

    LAMBDA / APPLY / IF each cost 1 on the longest root-to-leaf path.
    Literals, variables, empty lists, and int holes are terminals (depth 0).
    """
    if isinstance(node, (NumberNode, BooleanNode, VariableNode, IntHoleNode)):
        return 0
    if isinstance(node, ListNode):
        if not node.elements:
            return 0
        return 1 + max(program_depth(e) for e in node.elements)
    if isinstance(node, LambdaNode):
        return 1 + program_depth(node.body)
    if isinstance(node, IfNode):
        return 1 + max(
            program_depth(node.condition),
            program_depth(node.then_expr),
            program_depth(node.else_expr),
        )
    if isinstance(node, ApplicationNode):
        children = [program_depth(node.function)] + [program_depth(a) for a in node.arguments]
        return 1 + max(children)
    raise ValueError(f"Unknown node type: {type(node)}")


def free_variables(node: ASTNode, bound: set[str] | None = None) -> set[str]:
    """Return the set of free variable names in node given already-bound names."""
    if bound is None:
        bound = set()

    if isinstance(node, (NumberNode, BooleanNode, IntHoleNode)):
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


def resolve_type(type_: TypeType, T1_val=int, T2_val=int, instantiation: dict | None = None) -> TypeType | None:
    """
    Resolve type variables using a fresh SubstitutionTable.

    If instantiation is provided, use it instead of T1_val/T2_val defaults.
    Non-callable resolved types are validated against TYPE_UNIVERSE.
    """
    subs = SubstitutionTable()
    if instantiation is not None:
        for tv, concrete in instantiation.items():
            subs[tv] = concrete
    else:
        subs[T1] = T1_val
        subs[T2] = T2_val
    try:
        resolved = substitute_type_vars(type_, subs)
    except Exception:
        return None

    # When using explicit instantiation, validate non-callable types
    if instantiation is not None:
        if get_origin(resolved) != CallableOrig and resolved not in TYPE_UNIVERSE:
            return None

    return resolved
