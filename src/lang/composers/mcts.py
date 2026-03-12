"""
MCTS Composer

This module generates programs using Monte Carlo Tree Search (MCTS) to learn
which generation strategies produce programs that vary meaningfully with input.

The key insight is to model program generation as a Markov Decision Process:
- State: (target_type, context_types, parent_function, arg_index)
- Actions: generation strategies (literal, variable, lambda, if, application)
- Reward: variability of program output across different inputs

This addresses the problem of generating degenerate programs like
(if false <expr> <expr>) by learning which choices lead to interesting programs.
"""

from typing import Optional, Any
from dataclasses import dataclass, field
import math
import random

from .base import Composer
from ..grammar import Grammar
from ..ast_nodes import (
    ASTNode, VariableNode, ListNode,
    LambdaNode, ApplicationNode, IfNode
)
from ..type_utils import (
    get_args,
    get_base_type,
    CallableOrig,
    TypeType,
    SubstitutionTable,
    substitute_type_vars,
    matchable
)
from ..compiler import JITCompiler, JITCompilationError


def min_depth_for_type(type_: TypeType) -> int:
    """
    Compute the minimum depth required to generate a value of this type.
    """
    base = get_base_type(type_)

    if type_ == int or type_ == bool:
        return 0
    if base == list:
        return 0
    if base == CallableOrig:
        return 1
    return 1


def type_to_hashable(type_: TypeType) -> str:
    """
    Convert a type to a hashable string representation.

    This handles complex types like Callable[[list[int]], list[int]] that
    aren't directly hashable.
    """
    if type_ is int:
        return "int"
    elif type_ is bool:
        return "bool"

    base = get_base_type(type_)
    if base == list:
        args = get_args(type_)
        if args:
            return f"list[{type_to_hashable(args[0])}]"
        return "list"
    elif base == CallableOrig:
        args = get_args(type_)
        if args and len(args) == 2:
            param_types, ret_type = args
            if isinstance(param_types, list):
                params_str = ", ".join(type_to_hashable(p) for p in param_types)
                return f"Callable[[{params_str}], {type_to_hashable(ret_type)}]"
        return "Callable"

    return str(type_)


@dataclass(frozen=True)
class MCTSState:
    """
    Represents a state in the MCTS search tree.

    The state captures the context of a generation decision:
    - target_type: What type we're trying to generate (as hashable string)
    - context_types: What types are available in context (frozen set)
    - parent_function: The function for which we're generating an argument (or None)
    - arg_index: Which argument position we're filling (or -1)
    - depth: Current remaining depth (discretized into buckets)
    """
    target_type: str
    context_types: frozenset
    parent_function: Optional[str]
    arg_index: int
    depth_bucket: int  # 0=shallow(0-1), 1=medium(2-3), 2=deep(4+)

    @classmethod
    def from_context(
        cls,
        target_type: TypeType,
        context: dict[str, TypeType],
        parent_function: Optional[str],
        arg_index: int,
        depth: int,
        substitutions: SubstitutionTable
    ) -> 'MCTSState':
        """Create a state from generation context."""
        # Convert target type to hashable string
        resolved_target = substitute_type_vars(target_type, substitutions)
        target_str = type_to_hashable(resolved_target)

        # Extract types from context (not names, just types)
        context_types = frozenset(
            type_to_hashable(substitute_type_vars(t, substitutions))
            for t in context.values()
        )

        # Discretize depth into buckets for better generalization
        if depth <= 1:
            depth_bucket = 0
        elif depth <= 3:
            depth_bucket = 1
        else:
            depth_bucket = 2

        return cls(
            target_type=target_str,
            context_types=context_types,
            parent_function=parent_function,
            arg_index=arg_index,
            depth_bucket=depth_bucket
        )


@dataclass
class ActionStats:
    """Statistics for a single action at a state."""
    visits: int = 0
    total_reward: float = 0.0

    @property
    def q_value(self) -> float:
        """Average reward for this action."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


@dataclass
class MCTSNode:
    """
    A node in the MCTS tree representing a state.

    Tracks statistics for each possible action from this state.
    """
    state: MCTSState
    action_stats: dict[str, ActionStats] = field(default_factory=dict)
    total_visits: int = 0

    def get_ucb1_score(self, action: str, exploration_weight: float = 1.414) -> float:
        """
        Compute UCB1 score for an action.

        UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))

        Args:
            action: The action to score
            exploration_weight: The exploration constant (default sqrt(2))

        Returns:
            UCB1 score (higher is better)
        """
        stats = self.action_stats.get(action)

        if stats is None or stats.visits == 0:
            return float('inf')  # Unexplored actions have highest priority

        exploitation = stats.q_value
        exploration = exploration_weight * math.sqrt(
            math.log(self.total_visits) / stats.visits
        )

        return exploitation + exploration

    def select_action(
        self,
        available_actions: list[str],
        exploration_weight: float = 1.414
    ) -> str:
        """Select an action using UCB1."""
        best_action = None
        best_score = float('-inf')

        for action in available_actions:
            score = self.get_ucb1_score(action, exploration_weight)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def update(self, action: str, reward: float):
        """Update statistics after observing a reward."""
        if action not in self.action_stats:
            self.action_stats[action] = ActionStats()

        self.action_stats[action].visits += 1
        self.action_stats[action].total_reward += reward
        self.total_visits += 1


class InputSampler:
    """
    Samples random inputs of arbitrary types for program evaluation.

    Used to compute variability rewards by executing programs on
    diverse inputs.
    """

    def __init__(
        self,
        rng: random.Random,
        min_int: int = 0,
        max_int: int = 99,
        min_list_len: int = 0,
        max_list_len: int = 10
    ):
        self.rng = rng
        self.min_int = min_int
        self.max_int = max_int
        self.min_list_len = min_list_len
        self.max_list_len = max_list_len

    def sample(self, type_: TypeType, substitutions: SubstitutionTable) -> Any:
        """
        Sample a random value of the given type.

        Args:
            type_: The type to sample
            substitutions: Current type substitutions

        Returns:
            A random value of that type
        """
        actual_type = substitute_type_vars(type_, substitutions)
        base = get_base_type(actual_type)

        if actual_type == int:
            return self.rng.randint(self.min_int, self.max_int)

        elif actual_type == bool:
            return self.rng.choice([True, False])

        elif base == list:
            args = get_args(actual_type)
            elem_type = args[0] if args else int
            length = self.rng.randint(self.min_list_len, self.max_list_len)
            return [self.sample(elem_type, substitutions) for _ in range(length)]

        elif base == CallableOrig:
            # For callable types, we need to create a function that takes
            # the expected arguments and returns a value of the return type.
            # This is tricky - for now, return a simple function.
            args = get_args(actual_type)
            if args and len(args) == 2:
                _param_types, ret_type = args
                # Create a function that ignores args and returns a sampled value
                ret_val = self.sample(ret_type, substitutions)
                return lambda *args: ret_val
            return lambda x: x  # Identity function as fallback

        else:
            raise ValueError(f"Cannot sample type: {actual_type}")

    def sample_many(
        self,
        type_: TypeType,
        n: int,
        substitutions: SubstitutionTable
    ) -> list[Any]:
        """Sample n random values of the given type."""
        return [self.sample(type_, substitutions) for _ in range(n)]


class VariabilityScorer:
    """
    Computes variability rewards for programs and subexpressions.

    Supports two modes:
    1. Full program evaluation: Tests complete program on various inputs
    2. Subexpression evaluation: Tests a subexpression with sampled context values

    The subexpression mode enables immediate feedback during generation,
    so MCTS learns which subexpressions lead to variable outputs.
    """

    def __init__(
        self,
        grammar: Grammar,
        rng: random.Random,
        num_samples: int = 12,
        identity_penalty: float = 0.7
    ):
        self.grammar = grammar
        self.input_sampler = InputSampler(rng)
        self.jit_compiler = JITCompiler(grammar)
        self.num_samples = num_samples
        self.rng = rng
        self.identity_penalty = identity_penalty

        # Dynamically identify function categories from the grammar
        self._categorize_functions()

    def _categorize_functions(self):
        """
        Categorize functions in the grammar based on their type signatures.

        Categories:
        - Higher-order: functions that take Callable arguments
        - List transformations: functions that take list(s) and return list
        - List queries: functions that take list(s) but return non-list
        """
        self._higher_order_funcs: set[str] = set()
        self._list_transform_funcs: set[str] = set()
        self._list_query_funcs: set[str] = set()

        for name in self.grammar.names:
            func_info = self.grammar[name]
            arg_types = func_info['arg_types']
            ret_type = func_info['ret_type']

            # Check if it's a higher-order function
            is_higher_order = any(
                get_base_type(arg_type) == CallableOrig
                for arg_type in arg_types
            )
            if is_higher_order:
                self._higher_order_funcs.add(name)
                continue  # Higher-order takes precedence

            # Check if it's a list transformation function
            has_list_arg = any(
                get_base_type(arg_type) == list
                for arg_type in arg_types
            )
            returns_list = get_base_type(ret_type) == list

            if has_list_arg and returns_list:
                self._list_transform_funcs.add(name)
            elif has_list_arg:
                self._list_query_funcs.add(name)

    def compute_subexpression_reward(
        self,
        node: ASTNode,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> float:
        """
        Compute reward for a subexpression given its context.

        Evaluates the subexpression by sampling values for context variables
        and checking if the result varies with input.

        Args:
            node: The subexpression AST node
            context: Variables in scope (name -> type)
            substitutions: Current type substitutions

        Returns:
            Reward in [0, 1] based on variability and structure
        """
        var_names = list(context.keys())
        var_types = [context[name] for name in var_names]

        if not var_names:
            # No context variables - score by structure only
            return self._score_by_structure(node)

        # Create wrapper: (λ (v1 v2 ...) node)
        try:
            wrapper = LambdaNode(var_names, node)
            compiled_fn = self.jit_compiler.compile(wrapper)
        except (JITCompilationError, Exception):
            # Give credit for structure even if it doesn't compile
            return self._score_by_structure(node) * 0.5

        # Sample context values and evaluate
        outputs = []
        inputs_for_identity = []
        error_count = 0

        for _ in range(self.num_samples):
            try:
                args = [self.input_sampler.sample(vt, substitutions) for vt in var_types]
                result = compiled_fn(*args)
                outputs.append(self._to_hashable(result))
                inputs_for_identity.append(self._to_hashable(args[0]) if len(args) == 1 else None)
            except Exception:
                error_count += 1

        valid_outputs = [o for o in outputs if o is not None]

        if not valid_outputs:
            return self._score_by_structure(node) * 0.2

        success_rate = len(valid_outputs) / self.num_samples

        # Variability
        unique_outputs = len(set(valid_outputs))
        if len(valid_outputs) == 1:
            variability = 0.0
        else:
            variability = (unique_outputs - 1) / (len(valid_outputs) - 1)

        # Identity penalty for single-variable context
        if len(var_names) == 1 and inputs_for_identity[0] is not None:
            identity_count = sum(1 for o, i in zip(outputs, inputs_for_identity)
                               if o is not None and i is not None and o == i)
            if len(valid_outputs) > 0:
                identity_ratio = identity_count / len(valid_outputs)
                variability *= (1.0 - self.identity_penalty * identity_ratio)

        # Structure bonus
        structure = self._score_by_structure(node)

        # Combined score - favor structure heavily to encourage trying functions
        return (variability * 0.3 + structure * 0.5 + success_rate * 0.2)

    def _score_by_structure(self, node: ASTNode) -> float:
        """
        Score a node based on its syntactic structure.

        Uses grammar-derived information to identify higher-order functions
        rather than hardcoded function names.
        """
        func_names = node.function_names()
        # Filter to only include actual grammar functions (not lambda parameters)
        grammar_funcs = func_names & set(self.grammar.names)

        if grammar_funcs & self._higher_order_funcs:
            return 0.9  # Higher-order functions - strongly preferred
        elif grammar_funcs & self._list_transform_funcs:
            return 0.8  # List transformation functions
        elif grammar_funcs & self._list_query_funcs:
            return 0.7  # List query functions
        elif grammar_funcs:
            return 0.5  # Some grammar function usage
        elif type(node) == VariableNode:
            return 0.15  # Variable reference - slightly penalized
        else:
            return 0.05  # Literal

    def compute_variability(
        self,
        program: ASTNode,
        input_type: TypeType,
        substitutions: SubstitutionTable
    ) -> float:
        """
        Compute the variability score for a complete program.

        Args:
            program: The program to evaluate (should be a lambda)
            input_type: The type of input the program expects
            substitutions: Type substitutions

        Returns:
            Variability score in [0, 1]
        """
        inputs = self.input_sampler.sample_many(input_type, self.num_samples, substitutions)

        try:
            compiled_fn = self.jit_compiler.compile(program)
        except JITCompilationError:
            return self._score_by_structure(program) * 0.3

        outputs = []
        identity_count = 0
        valid_count = 0

        for inp in inputs:
            try:
                result = compiled_fn(inp)
                valid_count += 1
                if self._equals(result, inp):
                    identity_count += 1
                outputs.append(self._to_hashable(result))
            except Exception:
                outputs.append(None)

        valid_outputs = [o for o in outputs if o is not None]
        if not valid_outputs:
            return self._score_by_structure(program) * 0.2

        unique_outputs = len(set(valid_outputs))
        if len(valid_outputs) == 1:
            variability = 0.0
        else:
            variability = (unique_outputs - 1) / (len(valid_outputs) - 1)

        if valid_count > 0:
            identity_ratio = identity_count / valid_count
            variability *= (1.0 - self.identity_penalty * identity_ratio)

        structure = self._score_by_structure(program)
        success_rate = valid_count / self.num_samples

        # Favor structure to encourage using interesting functions
        return (variability * 0.3 + structure * 0.5 + success_rate * 0.2)

    def _equals(self, a: Any, b: Any) -> bool:
        """Check if two values are equal (handling lists)."""
        if type(a) != type(b):
            return False
        if isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(self._equals(x, y) for x, y in zip(a, b))
        return a == b

    def _to_hashable(self, value: Any) -> Any:
        """Convert a value to a hashable representation."""
        if isinstance(value, list):
            return tuple(self._to_hashable(x) for x in value)
        elif isinstance(value, dict):
            return tuple(sorted((k, self._to_hashable(v)) for k, v in value.items()))
        elif callable(value):
            return id(value)
        else:
            return value


class MCTSComposer(Composer):
    """
    Generates well-typed programs using Monte Carlo Tree Search.

    Uses MCTS to learn which generation strategies (literal, variable,
    lambda, if, application) produce programs that vary meaningfully
    with their inputs.

    The composer operates in two modes:
    - Training mode: Uses UCB1 exploration to build the MCTS tree
    - Inference mode: Uses learned Q-values to generate programs

    Key differences from random generation:
    1. Tracks state (type, context, parent function, arg index)
    2. Learns which choices lead to variable outputs
    3. Avoids degenerate patterns like (if false ...)
    """

    def __init__(
        self,
        seed: int,
        grammar: Grammar,
        exploration_weight: float = 1.414,
        num_variability_samples: int = 20,
        training_mode: bool = True,
        inference_temperature: float = 0.5,
        diversity_window: int = 20,
        diversity_penalty: float = 0.3,
        max_tree_size: int = 100000
    ):
        super().__init__(seed, grammar)

        # MCTS tree: maps states to nodes
        self.tree: dict[MCTSState, MCTSNode] = {}
        self.max_tree_size = max_tree_size
        self._access_counter: int = 0  # Global counter for LRU tracking
        self._state_access_time: dict[MCTSState, int] = {}  # Track last access time

        # Configuration
        self.exploration_weight = exploration_weight
        self.training_mode = training_mode
        self.inference_temperature = inference_temperature

        # Diversity tracking: penalize repetitive patterns
        self.diversity_window = diversity_window
        self.diversity_penalty = diversity_penalty
        self._recent_patterns: list[str] = []

        # Immediate feedback configuration
        self.immediate_reward_weight = 0.4
        self.final_reward_weight = 0.6
        self.min_depth_for_immediate = 1
        self.constant_expression_penalty = -0.5  # Penalty for expressions that don't use context
        self._immediate_feedback_given: set[tuple[MCTSState, str]] = set()

        # Variability scorer for computing rewards
        self.variability_scorer = VariabilityScorer(
            grammar, self.rng, num_variability_samples
        )

        # Track the path through the tree during generation
        self._current_path: list[tuple[MCTSState, str]] = []

        # Track the root program type for reward computation
        self._root_input_type: Optional[TypeType] = None

    @classmethod
    def get_name(cls) -> str:
        return "mcts"

    def set_training_mode(self, training: bool):
        """Set whether to use exploration (training) or exploitation (inference)."""
        self.training_mode = training

    def clear_diversity_history(self):
        """Clear the recent patterns history for diversity tracking."""
        self._recent_patterns = []

    def _sample_smart_literal(
        self,
        type_: TypeType,
        substitutions: SubstitutionTable,
        parent_function: Optional[str],
        arg_index: int
    ) -> ASTNode:
        """
        Sample a literal value with context-aware constraints.

        For index/position arguments, generates smaller numbers that are more
        likely to be valid. This improves the success rate of generated programs.
        """
        from ..ast_nodes import NumberNode, BooleanNode, ListNode

        actual_type = substitute_type_vars(type_, substitutions)
        base_type = get_base_type(actual_type)

        if actual_type == int:
            return NumberNode(self.rng.randint(0, 10))

        elif actual_type == bool:
            # For boolean conditions, prefer variable expressions
            # But if we must use a literal, at least randomize it
            return BooleanNode(self.rng.choice([True, False]))

        elif base_type == list:
            return ListNode([])

        else:
            # Fall back to base class
            return self._sample_literal(type_, substitutions)

    def _get_or_create_node(self, state: MCTSState) -> MCTSNode:
        """Get existing node or create a new one for the state, with LRU eviction."""
        # Update access time
        self._access_counter += 1
        self._state_access_time[state] = self._access_counter

        # Return existing node if available
        if state in self.tree:
            return self.tree[state]

        # Check if we need to evict old entries
        if len(self.tree) >= self.max_tree_size:
            # Evict 10% of oldest entries to avoid constant eviction
            num_to_evict = max(1, self.max_tree_size // 10)

            # Find oldest states by access time
            states_by_access = sorted(
                self._state_access_time.items(),
                key=lambda x: x[1]
            )

            for old_state, _ in states_by_access[:num_to_evict]:
                if old_state in self.tree:
                    del self.tree[old_state]
                if old_state in self._state_access_time:
                    del self._state_access_time[old_state]

        # Create new node
        self.tree[state] = MCTSNode(state=state)
        return self.tree[state]

    def _select_action(
        self,
        state: MCTSState,
        candidates: list[tuple[str, Any]],
        weights: list[float]
    ) -> tuple[int, str]:
        """
        Select an action using MCTS (training) or temperature-softmax (inference).

        Args:
            state: Current state
            candidates: List of (action_type, data) tuples
            weights: Prior weights for each candidate

        Returns:
            (index, action_type) of selected action
        """
        action_types = [c[0] for c in candidates]

        if self.training_mode:
            # Use UCB1 for exploration
            node = self._get_or_create_node(state)
            selected_action = node.select_action(action_types, self.exploration_weight)
            idx = action_types.index(selected_action)
        else:
            # Use temperature-based softmax for diverse inference
            if state in self.tree:
                node = self.tree[state]
                scores = []

                for i, action in enumerate(action_types):
                    stats = node.action_stats.get(action)
                    if stats and stats.visits > 0:
                        # Q-value + small prior bonus
                        score = stats.q_value + 0.01 * weights[i]
                    else:
                        # Unexplored: use prior weight (scaled to Q-value range)
                        score = weights[i] * 0.5

                    # Apply diversity penalty for recently used patterns
                    if action.startswith('apply:'):
                        func_name = action[6:]  # Remove 'apply:' prefix
                        recent_count = self._recent_patterns.count(func_name)
                        if recent_count > 0:
                            score -= self.diversity_penalty * min(recent_count, 3)

                    scores.append(score)

                # Temperature-based softmax sampling
                if self.inference_temperature > 0:
                    # Softmax with temperature
                    max_score = max(scores)
                    exp_scores = [
                        math.exp((s - max_score) / self.inference_temperature)
                        for s in scores
                    ]
                    total = sum(exp_scores)
                    probs = [e / total for e in exp_scores]
                    idx = self.rng.choices(range(len(candidates)), weights=probs, k=1)[0]
                else:
                    # Temperature 0: argmax
                    idx = scores.index(max(scores))

                selected_action = action_types[idx]
            else:
                # No data for this state, fall back to weighted random
                idx = self.rng.choices(range(len(candidates)), weights=weights, k=1)[0]
                selected_action = action_types[idx]

        return idx, selected_action

    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None,
        parent_function: Optional[str] = None,
        arg_index: int = -1,
        is_root: bool = True
    ) -> ASTNode:
        """
        Generate a random well-typed program using MCTS.

        Args:
            target_type: The desired output type
            depth: Maximum remaining depth (0 = only literals/variables)
            context: Variable bindings in scope (name -> type)
            substitutions: Current type variable substitutions
            parent_function: Function for which we're generating an argument
            arg_index: Which argument position we're filling
            is_root: Whether this is the root call (for reward computation)

        Returns:
            An AST node of the target type
        """
        if context is None:
            context = {}
        if substitutions is None:
            substitutions = SubstitutionTable()

        # Track root for reward computation
        if is_root:
            self._current_path = []
            self._immediate_feedback_given = set()
            self._root_input_type = self._extract_input_type(target_type, substitutions)

        # Resolve target type through substitutions
        target = substitute_type_vars(target_type, substitutions)
        base_type = get_base_type(target)

        # Create state for this decision point
        state = MCTSState.from_context(
            target_type=target,
            context=context,
            parent_function=parent_function,
            arg_index=arg_index,
            depth=depth,
            substitutions=substitutions
        )

        # Build list of all possible expressions with their weights
        candidates = []
        weights = []

        # Candidate 1: Literal (for atomic types and empty list for list types)
        if target == int or target == bool or base_type == list:
            candidates.append(('literal', None))
            weights.append(0.1)

        # Candidate 2: Each compatible variable from context
        for var_name, var_type in context.items():
            subs_copy = substitutions.copy()
            if matchable(var_type, target, subs_copy):
                candidates.append(('variable', (var_name, subs_copy)))
                weights.append(0.1)

        # Candidate 3: Lambda (for Callable types, only if depth > 0)
        # For Callable types, lambda is the only structural option (no if wrapping)
        # This ensures the lambda parameter is in context for the body
        if base_type == CallableOrig and depth > 0:
            candidates.append(('lambda', None))
            weights.append(0.2)

        # Candidate 4: If expression (only for non-Callable types, depth > 0)
        # Excluding Callable types prevents (if cond (λ x ...) (λ y ...)) patterns
        # where the condition doesn't have access to lambda parameters
        if depth > 0 and base_type != CallableOrig:
            candidates.append(('if', None))
            weights.append(0.2)

        # Candidate 5: Each possible function application from grammar
        # Use function-specific action names so MCTS can learn per-function
        if depth > 0:
            matches = self.grammar.find_matching_functions(
                ret_type=target,
                substitutions=substitutions
            )
            for func_name, func_subs in matches:
                # Check if all argument types can be generated at depth-1
                func_info = self.grammar[func_name]
                can_generate = True
                for arg_type in func_info['arg_types']:
                    resolved_arg = substitute_type_vars(arg_type, func_subs)
                    if min_depth_for_type(resolved_arg) > depth - 1:
                        can_generate = False
                        break
                if can_generate:
                    # Use function-specific action type
                    candidates.append((f'apply:{func_name}', (func_name, func_subs)))
                    weights.append(0.2)

        # If no candidates, raise an error
        if not candidates:
            raise ValueError(f"Cannot generate expression of type {target_type} with depth {depth}")

        # Select action using MCTS
        chosen_idx, expr_type = self._select_action(state, candidates, weights)
        expr_data = candidates[chosen_idx][1]

        # Record the decision in the current path
        self._current_path.append((state, expr_type))

        # Generate the chosen expression
        if expr_type == 'literal':
            result = self._sample_smart_literal(target, substitutions, parent_function, arg_index)

        elif expr_type == 'variable':
            var_name, new_subs = expr_data
            substitutions.update(new_subs)
            result = VariableNode(var_name)

        elif expr_type == 'lambda':
            result = self._generate_lambda(target, depth, context, substitutions)

        elif expr_type == 'if':
            result = self._generate_if(target, depth, context, substitutions)

        elif expr_type.startswith('apply:'):
            func_name, func_subs = expr_data
            result = self._generate_application_for(
                func_name, func_subs, depth, context, substitutions
            )

        else:
            raise ValueError(f"Unknown expression type: {expr_type}")

        # Immediate subexpression feedback
        if (self.training_mode and
            self._should_compute_immediate_reward(result, depth, context)):

            # Fast check: does this subexpression use any context variables?
            # If not, it's a constant expression like (+ 7 7) or (cut_idx 4 [])
            uses_context = self._uses_context_variable(result, set(context.keys()))

            if not uses_context:
                # Strong penalty for constant expressions - they have zero variability
                immediate_reward = self.constant_expression_penalty
            else:
                # Compute actual variability reward
                immediate_reward = self.variability_scorer.compute_subexpression_reward(
                    result, context, substitutions
                )

            imm_node = self._get_or_create_node(state)
            imm_node.update(expr_type, immediate_reward * self.immediate_reward_weight)
            self._immediate_feedback_given.add((state, expr_type))

        # Final reward backpropagation at root
        if is_root and self.training_mode:
            reward = self._compute_reward(result)
            self._backpropagate_final(reward)

        # Track patterns for diversity (in both training and inference)
        if is_root:
            pattern = self._extract_pattern(result)
            if pattern:
                self._recent_patterns.append(pattern)
                # Keep only recent patterns within the window
                if len(self._recent_patterns) > self.diversity_window:
                    self._recent_patterns.pop(0)

        return result

    def _extract_pattern(self, node: ASTNode) -> Optional[str]:
        """
        Extract a pattern signature from an AST for diversity tracking.

        Returns the top-level function application pattern, e.g., 'unique' or 'map'.
        """
        # Unwrap lambda to get the body
        while node.ast_type == 'Lambda':
            node = node.body

        # Get the top-level function if it's an application
        if node.ast_type == 'Application':
            if node.function.ast_type == "Variable":
                return node.function.name

        return None

    def _should_compute_immediate_reward(
        self,
        node: ASTNode,
        depth: int,
        context: dict[str, TypeType]
    ) -> bool:
        """
        Decide whether to compute immediate reward for a subexpression.

        Skip trivial cases where immediate feedback isn't useful:
        - Literals (no variability possible)
        - Simple variable references (variability depends on usage)
        - Leaf nodes at max depth (will get final reward anyway)
        """
        # Skip if no context (can't evaluate variability without inputs)
        if not context:
            return False

        # Skip shallow generations (final reward is close anyway)
        if depth < self.min_depth_for_immediate:
            return False

        # Skip literals - they have zero variability by definition
        node_type = node.ast_type
        if node_type in ('Number', 'Boolean', 'List'):
            return False

        # Skip simple variable references - variability comes from context
        if node_type == 'Variable':
            return False

        # Compute for applications, if-expressions, and lambdas with bodies
        return True

    def _uses_context_variable(self, node: ASTNode, var_names: set[str]) -> bool:
        """
        Check if an AST node uses any of the specified context variables.

        Args:
            node: The AST node to check
            var_names: Set of variable names to look for

        Returns:
            True if any variable from var_names is referenced in the expression
        """
        node_type = type(node)
        if node_type == VariableNode:
            return node.name in var_names
        elif node_type == LambdaNode:
            # Lambda parameters shadow context variables
            remaining = var_names - set(node.param)
            if not remaining:
                return False
            return self._uses_context_variable(node.body, remaining)
        elif node_type == ApplicationNode:
            if self._uses_context_variable(node.function, var_names):
                return True
            return any(self._uses_context_variable(arg, var_names) for arg in node.arguments)
        elif node_type == IfNode:
            return (self._uses_context_variable(node.condition, var_names) or
                    self._uses_context_variable(node.then_expr, var_names) or
                    self._uses_context_variable(node.else_expr, var_names))
        elif node_type == ListNode:
            return any(self._uses_context_variable(elem, var_names) for elem in node.elements)
        # NumberNode, BooleanNode - don't use variables
        return False

    def _extract_input_type(
        self,
        target_type: TypeType,
        substitutions: SubstitutionTable
    ) -> Optional[TypeType]:
        """Extract the input type from a Callable type."""
        resolved = substitute_type_vars(target_type, substitutions)
        base = get_base_type(resolved)

        if base == CallableOrig:
            args = get_args(resolved)
            if args and len(args) == 2:
                param_types = args[0]
                if isinstance(param_types, list) and len(param_types) == 1:
                    return param_types[0]

        return None

    def _compute_reward(self, program: ASTNode) -> float:
        """Compute the variability reward for a completed program."""
        if self._root_input_type is None:
            # Not a function type, can't compute variability
            return 0.5  # Neutral reward

        # Check if program uses its lambda parameter
        if isinstance(program, LambdaNode):
            params = set(program.param)
            if not self._uses_context_variable(program.body, params):
                # Program is constant - strong negative reward
                return self.constant_expression_penalty

        return self.variability_scorer.compute_variability(
            program,
            self._root_input_type,
            SubstitutionTable()
        )

    def _backpropagate_final(self, reward: float):
        """Backpropagate final reward with immediate feedback adjustment."""
        for state, action in self._current_path:
            node = self._get_or_create_node(state)

            if (state, action) in self._immediate_feedback_given:
                # Already got immediate feedback - use reduced final weight
                weight = self.final_reward_weight
            else:
                # No immediate feedback - use full weight
                weight = self.immediate_reward_weight + self.final_reward_weight

            node.update(action, reward * weight)

    def _generate_lambda(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate a lambda expression.

        For Callable[[A1, A2, ...], R], generates a multi-argument lambda:
        (λ (x1 x2 ...) body)
        """
        target = substitute_type_vars(target_type, substitutions)
        type_args = get_args(target)

        if len(type_args) != 2:
            raise ValueError(f"Callable type must have exactly 2 elements: {target}")

        param_types_list, ret_type = type_args

        if not isinstance(param_types_list, list):
            raise ValueError(f"Expected parameter list in Callable, got {param_types_list}")

        # Generate parameter names and add to context
        new_context = context.copy()
        param_names = []

        for param_type in param_types_list:
            param_name = self._fresh_var_name()
            param_names.append(param_name)
            new_context[param_name] = param_type

        # Generate body with all parameters in scope
        # Pass parent_function=None since we're inside a lambda body
        body = self.generate(
            ret_type, depth - 1, new_context, substitutions,
            parent_function=None, arg_index=-1, is_root=False
        )

        # Build a single multi-argument lambda
        return LambdaNode(param_names, body)

    def _generate_if(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate an if statement.

        Format: (if condition then_expr else_expr)
        Both branches must have the target type.
        """
        # Generate boolean condition
        # Track that we're generating the condition of an if
        condition = self.generate(
            bool, depth - 1, context, substitutions,
            parent_function='if', arg_index=0, is_root=False
        )

        # Generate then branch with target type
        then_expr = self.generate(
            target_type, depth - 1, context, substitutions,
            parent_function='if', arg_index=1, is_root=False
        )

        # Generate else branch with target type
        else_expr = self.generate(
            target_type, depth - 1, context, substitutions,
            parent_function='if', arg_index=2, is_root=False
        )

        return IfNode(condition, then_expr, else_expr)

    def _generate_application_for(
        self,
        func_name: str,
        func_subs: SubstitutionTable,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate a function application for a specific function.
        """
        # Instantiate any remaining free type variables
        func_info = self.grammar[func_name]
        for arg_type in func_info['arg_types']:
            func_subs = self._instantiate_free_types(arg_type, func_subs)
        func_subs = self._instantiate_free_types(func_info['ret_type'], func_subs)

        # Update our substitutions
        substitutions.update(func_subs)

        # Generate arguments
        args = []
        for i, arg_type in enumerate(func_info['arg_types']):
            arg = self.generate(
                arg_type, depth - 1, context, substitutions,
                parent_function=func_name, arg_index=i, is_root=False
            )
            args.append(arg)

        return ApplicationNode(VariableNode(func_name), args)

    def get_tree_stats(self) -> dict:
        """Get statistics about the MCTS tree."""
        if not self.tree:
            return {'num_nodes': 0, 'total_visits': 0}

        total_visits = sum(node.total_visits for node in self.tree.values())

        # Find most visited states
        sorted_nodes = sorted(
            self.tree.items(),
            key=lambda x: x[1].total_visits,
            reverse=True
        )

        top_states = []
        for state, node in sorted_nodes[:10]:
            best_action = max(
                node.action_stats.items(),
                key=lambda x: x[1].q_value,
                default=(None, None)
            )
            top_states.append({
                'state': {
                    'target_type': state.target_type,
                    'parent_function': state.parent_function,
                    'arg_index': state.arg_index,
                    'depth_bucket': state.depth_bucket
                },
                'visits': node.total_visits,
                'best_action': best_action[0],
                'best_q': best_action[1].q_value if best_action[1] else None
            })

        return {
            'num_nodes': len(self.tree),
            'total_visits': total_visits,
            'top_states': top_states
        }

    def save_tree(self, filepath: str):
        """Save the MCTS tree to a file."""
        import pickle

        # Convert tree to serializable format
        data = {
            'tree': {
                state: {
                    'total_visits': node.total_visits,
                    'action_stats': {
                        action: {'visits': stats.visits, 'total_reward': stats.total_reward}
                        for action, stats in node.action_stats.items()
                    }
                }
                for state, node in self.tree.items()
            },
            'exploration_weight': self.exploration_weight
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_tree(self, filepath: str):
        """Load the MCTS tree from a file."""
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.tree = {}
        for state, node_data in data['tree'].items():
            node = MCTSNode(state=state)
            node.total_visits = node_data['total_visits']
            for action, stats_data in node_data['action_stats'].items():
                stats = ActionStats(
                    visits=stats_data['visits'],
                    total_reward=stats_data['total_reward']
                )
                node.action_stats[action] = stats
            self.tree[state] = node

        self.exploration_weight = data.get('exploration_weight', 1.414)


def train_mcts_composer(
    grammar: Grammar,
    target_type: TypeType,
    num_episodes: int = 1000,
    depth: int = 4,
    seed: int = 42,
    verbose: bool = True,
    max_tree_size: int = 100000
) -> MCTSComposer:
    """
    Train an MCTS composer by generating many programs and learning from rewards.

    Args:
        grammar: The grammar to use
        target_type: The type of programs to generate
        num_episodes: Number of training episodes
        depth: Maximum program depth
        seed: Random seed
        verbose: Whether to print progress
        max_tree_size: Maximum number of states to keep in the MCTS tree

    Returns:
        A trained MCTSComposer
    """
    composer = MCTSComposer(
        seed=seed,
        grammar=grammar,
        training_mode=True,
        max_tree_size=max_tree_size
    )

    rewards = []
    for i in range(num_episodes):
        try:
            composer.reset_var_counter()
            # Generate program - reward computed and backpropagated inside generate()
            composer.generate(
                target_type=target_type,
                depth=depth,
                context={},
                substitutions=SubstitutionTable()
            )
            if composer._current_path:
                last_state = composer._current_path[0][0]
                if last_state in composer.tree:
                    node = composer.tree[last_state]
                    if node.action_stats:
                        # Get the most recent reward
                        recent_action = composer._current_path[0][1]
                        if recent_action in node.action_stats:
                            rewards.append(node.action_stats[recent_action].total_reward / max(1, node.action_stats[recent_action].visits))
        except Exception as e:
            if verbose and i % 100 == 0:
                print(f"Episode {i}: Error - {e}")

        if verbose and (i + 1) % 100 == 0:
            avg_reward = sum(rewards[-100:]) / max(1, len(rewards[-100:]))
            print(f"Episode {i + 1}/{num_episodes}, Avg reward (last 100): {avg_reward:.3f}, Tree size: {len(composer.tree)}")

    # Switch to inference mode
    composer.set_training_mode(False)

    return composer
