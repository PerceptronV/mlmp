"""Extract trajectories from ASTs for behavioural cloning."""

from typing import Callable

from .mdp import SynthesisState, Action, ActionType
from ..grammar import Grammar
from ..ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode, IntHoleNode,
)
from ..type_utils import get_args, TypeType
from ..utils import resolve_type, freeze_instantiation


def _infer_ast_type(
    node: ASTNode,
    context: dict,
    grammar: Grammar,
    valid_instantiations: dict,
) -> TypeType | None:
    """
    Recursively infer the concrete type of an AST node.

    Returns None when the type cannot be determined unambiguously
    (e.g. empty ListNode, or complex higher-order terms).
    """
    if isinstance(node, (NumberNode, IntHoleNode)):
        return int
    elif isinstance(node, BooleanNode):
        return bool
    elif isinstance(node, VariableNode):
        return context.get(node.name)
    elif isinstance(node, ListNode):
        if not node.elements:
            return None  # ambiguous: could be list[int], list[bool], …
        elem_t = _infer_ast_type(node.elements[0], context, grammar, valid_instantiations)
        if elem_t == int:
            return list[int]
        if elem_t == bool:
            return list[bool]
        if elem_t == list[int]:
            return list[list[int]]
        return None
    elif isinstance(node, ApplicationNode) and isinstance(node.function, VariableNode):
        func_name = node.function.name
        func_info = grammar[func_name]
        for inst in valid_instantiations.get(func_name, []):
            arg_types = [resolve_type(t, instantiation=inst) for t in func_info['arg_types']]
            consistent = True
            for arg_n, arg_t in zip(node.arguments, arg_types):
                inferred = _infer_ast_type(arg_n, context, grammar, valid_instantiations)
                if inferred is not None and inferred != arg_t:
                    consistent = False
                    break
            if consistent:
                return resolve_type(func_info['ret_type'], instantiation=inst)
    return None


def _infer_instantiation(
    func_name: str,
    target_ret_type: TypeType,
    grammar: Grammar,
    valid_instantiations: dict,
    arg_nodes: list[ASTNode] | None = None,
    context: dict | None = None,
) -> dict | None:
    """
    Given a function and the desired return type, find the instantiation
    that produces that return type.

    When multiple instantiations share the same return type (e.g. ``==``
    works on both int and bool), ``arg_nodes`` and ``context`` are used
    to disambiguate via recursive type inference of the actual argument
    subtrees.
    """
    func_info = grammar[func_name]
    candidates = [
        inst for inst in valid_instantiations[func_name]
        if resolve_type(func_info['ret_type'], instantiation=inst) == target_ret_type
    ]

    if not candidates:
        return None
    if len(candidates) == 1 or arg_nodes is None or context is None:
        return candidates[0]

    # Multiple candidates: use actual argument types to disambiguate.
    arg_types_actual = [
        _infer_ast_type(n, context, grammar, valid_instantiations)
        for n in arg_nodes
    ]

    for inst in candidates:
        arg_types_expected = [resolve_type(t, instantiation=inst) for t in func_info['arg_types']]
        if all(
            actual is None or actual == expected
            for actual, expected in zip(arg_types_actual, arg_types_expected)
        ):
            return inst

    return candidates[0]  # fallback if nothing matches cleanly


def extract_trajectory(
    program: ASTNode,
    target_type: TypeType,
    grammar: Grammar,
    initial_context: dict[str, TypeType] | None = None,
    initial_depth: int = 8,
    valid_instantiations: dict | None = None,
) -> list[tuple[SynthesisState, Action]]:
    """
    Given a complete program AST, extract the trajectory of (state, action)
    pairs that would produce it under the top-down MDP.

    Programs should be the open term (not lambda-wrapped). The target_type
    should be the full Callable type, e.g. Callable[[list[int]], list[int]].

    When valid_instantiations is provided, APPLY actions carry frozen
    instantiation tuples inferred from the target return type.
    """
    if initial_context is None:
        initial_context = {}

    trajectory: list[tuple[SynthesisState, Action]] = []

    def _walk(node: ASTNode, state: SynthesisState):
        if isinstance(node, IntHoleNode):
            trajectory.append((state, Action(ActionType.INT_HOLE, None)))

        elif isinstance(node, NumberNode):
            trajectory.append((state, Action(ActionType.LITERAL_INT, node.value)))

        elif isinstance(node, BooleanNode):
            trajectory.append((state, Action(ActionType.LITERAL_BOOL, node.value)))

        elif isinstance(node, ListNode) and len(node.elements) == 0:
            trajectory.append((state, Action(ActionType.LITERAL_EMPTY_LIST, None)))

        elif isinstance(node, VariableNode):
            trajectory.append((state, Action(ActionType.VARIABLE, node.name)))

        elif isinstance(node, ApplicationNode):
            if isinstance(node.function, VariableNode):
                func_name = node.function.name
                func_info = grammar[func_name]

                if valid_instantiations is not None:
                    inst = _infer_instantiation(
                        func_name, state.target_type, grammar, valid_instantiations,
                        arg_nodes=node.arguments, context=state.context,
                    )
                    if inst is None:
                        raise ValueError(
                            f"No instantiation of '{func_name}' produces "
                            f"return type {state.target_type}"
                        )
                    frozen = freeze_instantiation(inst)
                    trajectory.append((state, Action(ActionType.APPLY, func_name, frozen)))
                    arg_types = [resolve_type(t, instantiation=inst) for t in func_info['arg_types']]
                else:
                    trajectory.append((state, Action(ActionType.APPLY, func_name)))
                    arg_types = [resolve_type(t) for t in func_info['arg_types']]

                generated_siblings = []
                for i, (arg_node, arg_type) in enumerate(zip(node.arguments, arg_types)):
                    child_state = SynthesisState(
                        target_type=arg_type,
                        context=state.context,
                        parent_func=func_name,
                        arg_index=i,
                        siblings=list(generated_siblings),
                        depth_budget=state.depth_budget - 1,
                        nesting_depth=state.nesting_depth,
                    )
                    _walk(arg_node, child_state)
                    generated_siblings.append((arg_node, None))

        elif isinstance(node, LambdaNode):
            trajectory.append((state, Action(ActionType.LAMBDA, None)))

            args = get_args(state.target_type)
            param_types = args[0]
            body_type = args[1]

            new_context = state.context.copy()
            for pname, ptype in zip(node.param, param_types):
                new_context[pname] = ptype

            body_state = SynthesisState(
                target_type=body_type,
                context=new_context,
                parent_func=None,
                arg_index=None,
                siblings=[],
                depth_budget=state.depth_budget - 1,
                nesting_depth=state.nesting_depth + 1,
            )
            _walk(node.body, body_state)

        elif isinstance(node, IfNode):
            trajectory.append((state, Action(ActionType.IF, None)))

            cond_state = SynthesisState(
                target_type=bool,
                context=state.context,
                parent_func=None,
                arg_index=None,
                siblings=[],
                depth_budget=state.depth_budget - 1,
                nesting_depth=state.nesting_depth,
            )
            _walk(node.condition, cond_state)

            then_state = SynthesisState(
                target_type=state.target_type,
                context=state.context,
                parent_func=None,
                arg_index=None,
                siblings=[],
                depth_budget=state.depth_budget - 1,
                nesting_depth=state.nesting_depth,
            )
            _walk(node.then_expr, then_state)

            else_state = SynthesisState(
                target_type=state.target_type,
                context=state.context,
                parent_func=None,
                arg_index=None,
                siblings=[],
                depth_budget=state.depth_budget - 1,
                nesting_depth=state.nesting_depth,
            )
            _walk(node.else_expr, else_state)

    initial_state = SynthesisState(
        target_type=target_type,
        context=initial_context,
        parent_func=None,
        arg_index=None,
        siblings=[],
        depth_budget=initial_depth,
    )
    _walk(program, initial_state)
    return trajectory
