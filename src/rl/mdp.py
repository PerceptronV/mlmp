"""MDP state/action definitions for program synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from ..lang.grammar import Grammar
from ..lang.ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode,
)
from ..lang.type_utils import CallableOrig, get_args, get_origin, TypeType
from ..utils import resolve_type


class ActionType(Enum):
    LITERAL_INT = auto()
    LITERAL_BOOL = auto()
    LITERAL_EMPTY_LIST = auto()
    VARIABLE = auto()
    APPLY = auto()
    LAMBDA = auto()
    IF = auto()


@dataclass(frozen=True)
class Action:
    """An action in the synthesis MDP. Must be hashable for vocab dict."""
    action_type: ActionType
    payload: Any = None

    def __hash__(self):
        return hash((self.action_type, self.payload))

    def __eq__(self, other):
        return (
            isinstance(other, Action)
            and self.action_type == other.action_type
            and self.payload == other.payload
        )


@dataclass
class SynthesisState:
    """
    State in the program synthesis MDP.

    Attributes:
        target_type: The type that must be generated at this AST node
        context: Dict mapping bound variable names to their types
        parent_func: Name of the function this term is an argument to (None if top-level)
        arg_index: Index of the argument position within parent_func (None if top-level)
        siblings: List of (ASTNode, Fingerprint|None) for already-generated sibling args
        depth_budget: Remaining depth budget
    """
    target_type: TypeType
    context: dict[str, TypeType] = field(default_factory=dict)
    parent_func: str | None = None
    arg_index: int | None = None
    siblings: list[tuple[ASTNode, Any]] = field(default_factory=list)
    depth_budget: int = 8


def valid_actions(
    state: SynthesisState,
    grammar: Grammar,
    seed_constants: list[int],
) -> list[Action]:
    """
    Enumerate all valid actions from the current state.

    An action is valid if it can produce a term of state.target_type
    given the current context and depth budget.
    """
    actions = []
    t = state.target_type

    # Literals
    if t == int:
        for c in seed_constants:
            actions.append(Action(ActionType.LITERAL_INT, c))
    if t == bool:
        actions.append(Action(ActionType.LITERAL_BOOL, True))
        actions.append(Action(ActionType.LITERAL_BOOL, False))
    if get_origin(t) == list or t == list:
        actions.append(Action(ActionType.LITERAL_EMPTY_LIST, None))

    # Variables in context with matching type
    for var_name, var_type in state.context.items():
        if var_type == t:
            actions.append(Action(ActionType.VARIABLE, var_name))

    # Function applications (only if depth budget > 0)
    if state.depth_budget > 0:
        for func_name in grammar.names:
            func_info = grammar[func_name]
            resolved_ret = resolve_type(func_info['ret_type'])
            if resolved_ret == t:
                actions.append(Action(ActionType.APPLY, func_name))

        # If-expression
        actions.append(Action(ActionType.IF, None))

    # Lambda (only if target type is Callable)
    if get_origin(t) == CallableOrig:
        actions.append(Action(ActionType.LAMBDA, None))

    return actions


class Episode:
    """
    Runs a single episode of the synthesis MDP.

    Uses a policy to make decisions at each state, building an AST
    top-down. Records the full trajectory for training.
    """

    def __init__(self, policy, grammar, test_suite, seed_constants, max_depth=6):
        self.policy = policy
        self.grammar = grammar
        self.test_suite = test_suite
        self.seed_constants = seed_constants
        self.max_depth = max_depth
        self.trajectory: list[tuple[SynthesisState, Action]] = []

    def run(self) -> tuple[ASTNode | None, list[tuple[SynthesisState, Action]]]:
        """
        Run one episode.

        Returns:
            (completed_ast, trajectory) or (None, trajectory) if generation fails.
        """
        initial_state = SynthesisState(
            target_type=Callable[[list[int]], list[int]],
            context={},
            parent_func=None,
            arg_index=None,
            siblings=[],
            depth_budget=self.max_depth,
        )

        ast = self._generate(initial_state)
        return ast, self.trajectory

    def _generate(self, state: SynthesisState) -> ASTNode | None:
        """Recursively generate an AST node by querying the policy."""
        actions = valid_actions(state, self.grammar, self.seed_constants)
        if not actions:
            return None

        action = self.policy.select_action(state, actions)
        self.trajectory.append((state, action))

        if action.action_type == ActionType.LITERAL_INT:
            return NumberNode(action.payload)
        elif action.action_type == ActionType.LITERAL_BOOL:
            return BooleanNode(action.payload)
        elif action.action_type == ActionType.LITERAL_EMPTY_LIST:
            return ListNode([])
        elif action.action_type == ActionType.VARIABLE:
            return VariableNode(action.payload)
        elif action.action_type == ActionType.APPLY:
            return self._generate_application(state, action)
        elif action.action_type == ActionType.LAMBDA:
            return self._generate_lambda(state)
        elif action.action_type == ActionType.IF:
            return self._generate_if(state)
        return None

    def _generate_application(self, state, action):
        func_name = action.payload
        func_info = self.grammar[func_name]
        arg_types = [resolve_type(t) for t in func_info['arg_types']]

        arg_nodes = []
        for i, arg_type in enumerate(arg_types):
            child_state = SynthesisState(
                target_type=arg_type,
                context=state.context,
                parent_func=func_name,
                arg_index=i,
                siblings=[(n, None) for n in arg_nodes],
                depth_budget=state.depth_budget - 1,
            )
            arg_node = self._generate(child_state)
            if arg_node is None:
                return None
            arg_nodes.append(arg_node)

        return ApplicationNode(VariableNode(func_name), arg_nodes)

    def _generate_lambda(self, state):
        args = get_args(state.target_type)
        param_types = args[0]
        body_type = args[1]
        param_names = [f"_p{i}" for i in range(len(param_types))]

        new_context = state.context.copy()
        for pname, ptype in zip(param_names, param_types):
            new_context[pname] = ptype

        body_state = SynthesisState(
            target_type=body_type,
            context=new_context,
            parent_func=None,
            arg_index=None,
            siblings=[],
            depth_budget=state.depth_budget - 1,
        )
        body_node = self._generate(body_state)
        if body_node is None:
            return None
        return LambdaNode(param_names, body_node)

    def _generate_if(self, state):
        cond_state = SynthesisState(
            target_type=bool,
            context=state.context,
            parent_func=None,
            arg_index=None,
            siblings=[],
            depth_budget=state.depth_budget - 1,
        )
        cond = self._generate(cond_state)
        if cond is None:
            return None

        then_state = SynthesisState(
            target_type=state.target_type,
            context=state.context,
            parent_func=None,
            arg_index=None,
            siblings=[],
            depth_budget=state.depth_budget - 1,
        )
        then_node = self._generate(then_state)
        if then_node is None:
            return None

        else_state = SynthesisState(
            target_type=state.target_type,
            context=state.context,
            parent_func=None,
            arg_index=None,
            siblings=[],
            depth_budget=state.depth_budget - 1,
        )
        else_node = self._generate(else_state)
        if else_node is None:
            return None

        return IfNode(cond, then_node, else_node)
