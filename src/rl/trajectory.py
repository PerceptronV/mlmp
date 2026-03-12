"""Extract trajectories from ASTs for behavioural cloning."""

from typing import Callable

from .mdp import SynthesisState, Action, ActionType
from ..lang.grammar import Grammar
from ..lang.ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode,
)
from ..lang.type_utils import get_args, TypeType
from ..utils import resolve_type


def extract_trajectory(
    program: ASTNode,
    target_type: TypeType,
    grammar: Grammar,
    initial_context: dict[str, TypeType] | None = None,
    initial_depth: int = 8,
) -> list[tuple[SynthesisState, Action]]:
    """
    Given a complete program AST, extract the trajectory of (state, action)
    pairs that would produce it under the top-down MDP.

    Programs should be the open term (not lambda-wrapped). The target_type
    should be the full Callable type, e.g. Callable[[list[int]], list[int]].
    """
    if initial_context is None:
        initial_context = {}

    trajectory: list[tuple[SynthesisState, Action]] = []

    def _walk(node: ASTNode, state: SynthesisState):
        if isinstance(node, NumberNode):
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
                trajectory.append((state, Action(ActionType.APPLY, func_name)))

                func_info = grammar[func_name]
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
            )
            _walk(node.condition, cond_state)

            then_state = SynthesisState(
                target_type=state.target_type,
                context=state.context,
                parent_func=None,
                arg_index=None,
                siblings=[],
                depth_budget=state.depth_budget - 1,
            )
            _walk(node.then_expr, then_state)

            else_state = SynthesisState(
                target_type=state.target_type,
                context=state.context,
                parent_func=None,
                arg_index=None,
                siblings=[],
                depth_budget=state.depth_budget - 1,
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
