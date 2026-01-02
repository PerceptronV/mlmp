"""
Empirical Program Composer

This module generates programs using empirical distributions learned from
example programs. The distributions are computed by parsing and analysing
programs from a text file (one program per line).

See docs/empirical-algorithm.md for a detailed description of the algorithm.

Key insight: We learn generation strategies conditioned on both the return type
AND the context signature (ordered list of types in scope). This allows us to
learn patterns like "use the 2nd int variable" rather than just "use some variable".
"""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from .base import Composer
from .strategies import (
    Strategy,
    LiteralStrategy,
    VariableStrategy,
    LambdaStrategy,
    IfStrategy,
    ApplicationStrategy,
)
from ..grammar import Grammar
from ..ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, IfNode, ListNode
)
from ..parser import parse as parse_program
from ..type_checker import TypeChecker
from ..type_utils import (
    get_args,
    get_base_type,
    CallableOrig,
    TypeType,
    SubstitutionTable,
    substitute_type_vars,
    matchable,
    isvariable
)


# ============================================================================
# Type Key - Canonical representation of types for distribution keys
# ============================================================================

def type_to_key(type_: TypeType) -> str:
    """
    Convert a type to a canonical string key for use in distributions.

    Type variables are normalised to 'T' so that list[T1] and list[T2]
    produce the same key.
    """
    if type_ is None:
        return "None"

    if isvariable(type_):
        return "T"

    base = get_base_type(type_)
    args = get_args(type_)

    if not args:
        if base == int:
            return "int"
        elif base == bool:
            return "bool"
        else:
            return str(base.__name__) if hasattr(base, '__name__') else str(base)

    if base == list:
        if args:
            return f"list[{type_to_key(args[0])}]"
        return "list[T]"

    if base == CallableOrig:
        if len(args) == 2:
            param_list, ret = args
            if isinstance(param_list, list):
                params_str = ", ".join(type_to_key(p) for p in param_list)
                return f"Callable[[{params_str}], {type_to_key(ret)}]"
        return "Callable"

    args_str = ", ".join(type_to_key(a) for a in args)
    base_name = base.__name__ if hasattr(base, '__name__') else str(base)
    return f"{base_name}[{args_str}]"


def context_to_key(context_types: list[TypeType]) -> tuple[str, ...]:
    """Convert a context (list of types) to a hashable key."""
    return tuple(type_to_key(t) for t in context_types)


ContextKey = (
    tuple[str, ...]  # Just context types
    | tuple[Optional[str], tuple[str, ...]]  # (function_name, context_types)
    | tuple[Optional[str], Optional[int], tuple[str, ...]]  # (function_name, arg_position, context_types)
)


# ============================================================================
# Empirical Distributions
# ============================================================================

class EmpiricalDistributions:
    """
    Holds empirical distributions learned from example programs.

    Stores:
    - P(strategy | return_type, context_signature[, function_name][, arg_position])
    - P(context_signature[, function_name][, arg_position] | return_type)
    - P(int_value | int_literal)
    - P(bool_value | bool_literal)
    """

    def __init__(
        self,
        include_function_name_in_context: bool = False,
        include_arg_position_in_context: bool = False
    ):
        self.include_function_name_in_context = include_function_name_in_context
        self.include_arg_position_in_context = include_arg_position_in_context

        # Main distribution: counts of strategies for each (return_type, context) pair
        # Key: (return_type_key, context_key)
        # Value: Counter of strategies
        self.strategy_counts: dict[tuple[str, ContextKey], Counter[Strategy]] = defaultdict(Counter)

        # Context counts for each return type (for marginalisation)
        # Key: return_type_key
        # Value: Counter of context_keys
        self.context_counts: dict[str, Counter[ContextKey]] = defaultdict(Counter)

        # Constant distributions
        self.int_constants: Counter[int] = Counter()
        self.bool_constants: Counter[bool] = Counter()

        # Normalised probabilities (computed after loading)
        self._normalised = False

    def _context_key(
        self,
        context_types: list[TypeType],
        function_name: Optional[str],
        arg_position: Optional[int] = None
    ) -> ContextKey:
        types_key = context_to_key(context_types)
        if self.include_arg_position_in_context:
            # Include both function name and arg position
            return (function_name, arg_position, types_key)
        elif self.include_function_name_in_context:
            return (function_name, types_key)
        return types_key

    def _split_context_key(
        self,
        ctx_key: ContextKey
    ) -> tuple[Optional[str], Optional[int], tuple[str, ...]]:
        """
        Split a context key into its components.

        Returns:
            (function_name, arg_position, types_key)
        """
        if self.include_arg_position_in_context:
            function_name, arg_position, types_key = ctx_key
            return function_name, arg_position, types_key
        elif self.include_function_name_in_context:
            function_name, types_key = ctx_key
            return function_name, None, types_key
        return None, None, ctx_key

    def record(
        self,
        return_type: TypeType,
        context_types: list[TypeType],
        strategy: Strategy,
        function_name: Optional[str] = None,
        arg_position: Optional[int] = None
    ):
        """Record an observation of a strategy."""
        type_key = type_to_key(return_type)
        ctx_key = self._context_key(context_types, function_name, arg_position)

        self.strategy_counts[(type_key, ctx_key)][strategy] += 1
        self.context_counts[type_key][ctx_key] += 1
        self._normalised = False

    def record_int_constant(self, value: int):
        """Record an integer constant value."""
        self.int_constants[value] += 1
        self._normalised = False

    def record_bool_constant(self, value: bool):
        """Record a boolean constant value."""
        self.bool_constants[value] += 1
        self._normalised = False

    def normalize(self):
        """Compute normalised probability distributions."""
        if self._normalised:
            return

        # Normalise strategy distributions P(strategy | R, C)
        self._strategy_probs: dict[tuple[str, ContextKey], dict[Strategy, float]] = {}
        for key, counts in self.strategy_counts.items():
            total = sum(counts.values())
            if total > 0:
                self._strategy_probs[key] = {s: c / total for s, c in counts.items()}

        # Normalise context distributions P(C | R)
        self._context_probs: dict[str, dict[ContextKey, float]] = {}
        for type_key, counts in self.context_counts.items():
            total = sum(counts.values())
            if total > 0:
                self._context_probs[type_key] = {c: n / total for c, n in counts.items()}

        # Normalise constant distributions
        int_total = sum(self.int_constants.values())
        self._int_probs = {v: c / int_total for v, c in self.int_constants.items()} if int_total > 0 else {}

        bool_total = sum(self.bool_constants.values())
        self._bool_probs = {v: c / bool_total for v, c in self.bool_constants.items()} if bool_total > 0 else {}

        self._normalised = True

    def get_strategy_weights(
        self,
        return_type: TypeType,
        context_types: list[TypeType],
        available_strategies: list[Strategy],
        noise: float = 0.0,
        function_name: Optional[str] = None,
        arg_position: Optional[int] = None
    ) -> list[float]:
        """
        Get weights for available strategies given return type and context.

        If the exact (return_type, context) pair was observed, use those weights.
        Otherwise, compute marginal by summing over observed contexts.

        Args:
            return_type: The target return type
            context_types: Ordered list of types in current context
            available_strategies: List of valid strategies for this situation
            noise: Floor weight for compatible strategies with zero probability
            function_name: Optional function name to include in context
            arg_position: Optional argument position to include in context

        Returns:
            List of weights (one per available strategy)
        """
        self.normalize()

        type_key = type_to_key(return_type)
        ctx_key = self._context_key(context_types, function_name, arg_position)

        weights = [0.0] * len(available_strategies)

        # Check for exact match
        if (type_key, ctx_key) in self._strategy_probs:
            probs = self._strategy_probs[(type_key, ctx_key)]
            for i, strategy in enumerate(available_strategies):
                weights[i] = probs.get(strategy, 0.0)
        else:
            # Compute marginal over observed contexts for this return type
            if type_key in self._context_probs:
                for obs_ctx_key, p_ctx in self._context_probs[type_key].items():
                    if (type_key, obs_ctx_key) not in self._strategy_probs:
                        continue
                    probs = self._strategy_probs[(type_key, obs_ctx_key)]

                    for i, strategy in enumerate(available_strategies):
                        # Map strategy between contexts
                        mapped = self._map_strategy(strategy, ctx_key, obs_ctx_key)
                        if mapped is not None and mapped in probs:
                            weights[i] += p_ctx * probs[mapped]

        # Apply noise floor
        for i in range(len(weights)):
            if weights[i] == 0.0:
                weights[i] = noise

        # Renormalise
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            # Fallback to uniform
            weights = [1.0 / len(available_strategies)] * len(available_strategies)

        return weights

    def _map_strategy(
        self,
        strategy: Strategy,
        from_ctx: ContextKey,
        to_ctx: ContextKey
    ) -> Optional[Strategy]:
        """
        Map a strategy from one context to another.

        Most strategies map directly. Variable strategies need position mapping
        based on type compatibility.
        """
        from_fn, from_arg_pos, from_ctx_types = self._split_context_key(from_ctx)
        to_fn, to_arg_pos, to_ctx_types = self._split_context_key(to_ctx)

        # Must match on function name if enabled
        if self.include_function_name_in_context and from_fn != to_fn:
            return None

        # Must match on arg position if enabled
        if self.include_arg_position_in_context and from_arg_pos != to_arg_pos:
            return None

        if isinstance(strategy, VariableStrategy):
            pos = strategy.position
            if pos >= len(from_ctx_types):
                return None
            var_type = from_ctx_types[pos]

            # Find position in target context with same type
            for i, t in enumerate(to_ctx_types):
                if t == var_type:
                    return VariableStrategy(i)
            return None

        # All other strategies map directly
        return strategy

    def sample_int_constant(self, rng) -> int:
        """Sample an integer constant from the empirical distribution."""
        self.normalize()
        if self._int_probs:
            constants = list(self._int_probs.keys())
            weights = [self._int_probs[c] for c in constants]
            return rng.choices(constants, weights=weights, k=1)[0]
        return rng.randint(0, 99)

    def sample_bool_constant(self, rng) -> bool:
        """Sample a boolean constant from the empirical distribution."""
        self.normalize()
        if self._bool_probs:
            constants = list(self._bool_probs.keys())
            weights = [self._bool_probs[c] for c in constants]
            return rng.choices(constants, weights=weights, k=1)[0]
        return rng.choice([True, False])

    def __repr__(self) -> str:
        n_pairs = len(self.strategy_counts)
        n_types = len(self.context_counts)
        total_obs = sum(sum(c.values()) for c in self.strategy_counts.values())
        return f"EmpiricalDistributions({n_types} types, {n_pairs} (type,ctx) pairs, {total_obs} observations)"


# ============================================================================
# AST Analyser
# ============================================================================

class ASTAnalyser:
    """
    Analyses AST nodes to extract strategies and infer types.

    Uses the TypeChecker for proper Hindley-Milner type inference, then
    walks the AST to record observations with fully resolved types.
    """

    def __init__(self, grammar: Grammar, dists: EmpiricalDistributions):
        self.grammar = grammar
        self.dists = dists
        self.type_checker = TypeChecker(grammar)
        # Cache for node types after inference
        self._node_types: dict[int, TypeType] = {}

    def analyse(
        self,
        node: ASTNode,
        expected_type: TypeType
    ) -> Optional[TypeType]:
        """
        Analyse an AST: infer types using TypeChecker, then record observations.

        Args:
            node: The AST node to analyse
            expected_type: The expected type of the program (e.g., Callable[[list[int]], list[int]])
                          This is used to resolve type variables that can't be inferred.

        Returns:
            The inferred return type, or None if inference fails
        """
        # Build initial context with grammar functions
        context: dict[str, TypeType] = {}
        for name, info in self.grammar.functions.items():
            arg_types = list(info['arg_types'])
            ret_type = info['ret_type']
            if len(arg_types) > 0:
                context[name] = CallableOrig[arg_types, ret_type]
            else:
                context[name] = ret_type

        try:
            # Use TypeChecker to infer types with full unification
            # We'll collect node types during this single pass
            self._node_types.clear()
            inferred_type, substitutions = self._infer_and_cache(node, context, SubstitutionTable())

            # Unify with expected type to resolve remaining type variables
            matchable(inferred_type, expected_type, substitutions, update=True, strict=False)

            # Now record observations with the resolved substitutions
            self._record_observations(node, [], substitutions)

            return substitute_type_vars(inferred_type, substitutions)

        except Exception:
            # Type inference failed, skip this program
            return None

    def _infer_and_cache(
        self,
        node: ASTNode,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> tuple[TypeType, SubstitutionTable]:
        """
        Infer types and cache them in a single pass.

        This is similar to TypeChecker.infer but also caches the type
        of each node we visit. This ensures we use the same type variables
        throughout and can resolve them all with the final substitution table.
        """
        # Numbers have type int
        if isinstance(node, NumberNode):
            self._node_types[id(node)] = int
            return int, substitutions

        # Booleans have type bool
        elif isinstance(node, BooleanNode):
            self._node_types[id(node)] = bool
            return bool, substitutions

        # Variables: look up in context
        elif isinstance(node, VariableNode):
            if node.name not in context:
                raise TypeError(f"Undefined variable: {node.name}")
            var_type = context[node.name]
            # Instantiate only for polymorphic grammar functions
            if node.name in self.grammar.functions:
                var_type = self.type_checker._instantiate_type(var_type)
            self._node_types[id(node)] = var_type
            return var_type, substitutions

        # Lists: infer element type and construct list[T]
        elif isinstance(node, ListNode):
            if not node.elements:
                elem_type = self.type_checker._fresh_type_var()
                list_type = list[elem_type]
                self._node_types[id(node)] = list_type
                return list_type, substitutions

            elem_type, subs = self._infer_and_cache(node.elements[0], context, substitutions)
            for elem in node.elements[1:]:
                elem_t, subs = self._infer_and_cache(elem, context, subs)
                if not matchable(elem_type, elem_t, subs, strict=False):
                    raise TypeError(f"List elements have inconsistent types")

            list_type = list[substitute_type_vars(elem_type, subs)]
            self._node_types[id(node)] = list_type
            return list_type, subs

        # Lambda: introduce type variable(s) for parameter(s), infer body type
        elif isinstance(node, LambdaNode):
            # node.param is always a list (even for single-parameter lambdas)
            param_types = [self.type_checker._fresh_type_var() for _ in node.param]
            new_context = context.copy()
            for param_name, param_type in zip(node.param, param_types):
                new_context[param_name] = param_type

            body_type, subs = self._infer_and_cache(node.body, new_context, substitutions)

            resolved_params = [substitute_type_vars(pt, subs) for pt in param_types]
            resolved_body = substitute_type_vars(body_type, subs)

            func_type = CallableOrig[resolved_params, resolved_body]
            self._node_types[id(node)] = func_type
            return func_type, subs

        # If: check condition is bool, branches have same type
        elif isinstance(node, IfNode):
            cond_type, subs = self._infer_and_cache(node.condition, context, substitutions)

            if not matchable(cond_type, bool, subs, strict=False):
                raise TypeError(f"If condition must be boolean")

            then_type, subs = self._infer_and_cache(node.then_expr, context, subs)
            else_type, subs = self._infer_and_cache(node.else_expr, context, subs)

            if not matchable(then_type, else_type, subs, strict=False):
                raise TypeError(f"If branches have different types")

            result_type = substitute_type_vars(then_type, subs)
            self._node_types[id(node)] = result_type
            return result_type, subs

        # Application: infer function type, check argument types
        elif isinstance(node, ApplicationNode):
            func_type, subs = self._infer_and_cache(node.function, context, substitutions)

            arg_types = []
            for arg in node.arguments:
                arg_t, subs = self._infer_and_cache(arg, context, subs)
                arg_types.append(arg_t)

            result_type = func_type
            for arg_type in arg_types:
                result_type, subs = self.type_checker._apply_function_type(result_type, arg_type, subs)

            self._node_types[id(node)] = result_type
            return result_type, subs

        else:
            raise TypeError(f"Unknown node type: {type(node).__name__}")

    def _get_node_type(self, node: ASTNode, substitutions: SubstitutionTable) -> Optional[TypeType]:
        """Get the cached type for a node, resolved through substitutions."""
        if id(node) in self._node_types:
            return substitute_type_vars(self._node_types[id(node)], substitutions)
        return None

    def _record_observations(
        self,
        node: ASTNode,
        context: list[tuple[str, TypeType]],  # [(name, type), ...] ordered by depth
        substitutions: SubstitutionTable,
        function_name: Optional[str] = None,
        arg_position: Optional[int] = None
    ):
        """
        Walk the AST and record observations with fully resolved types.

        Args:
            node: Current AST node
            context: Ordered list of (name, type) pairs for lambda-bound variables
            substitutions: Fully populated substitution table from type inference
            function_name: Optional function name to include in context
            arg_position: Optional argument position (0-indexed) to include in context
        """
        context_names = {name: i for i, (name, _) in enumerate(context)}

        # Resolve ALL context types through substitutions
        resolved_context = [substitute_type_vars(t, substitutions) for _, t in context]

        if isinstance(node, NumberNode):
            self.dists.record(int, resolved_context, LiteralStrategy('int'), function_name, arg_position)
            self.dists.record_int_constant(node.value)

        elif isinstance(node, BooleanNode):
            self.dists.record(bool, resolved_context, LiteralStrategy('bool'), function_name, arg_position)
            self.dists.record_bool_constant(node.value)

        elif isinstance(node, ListNode):
            if not node.elements:
                # Empty list - use cached type if available, otherwise generic list
                node_type = self._get_node_type(node, substitutions)
                if node_type is not None:
                    self.dists.record(node_type, resolved_context, LiteralStrategy('list'), function_name, arg_position)
                else:
                    self.dists.record(list, resolved_context, LiteralStrategy('list'), function_name, arg_position)
            else:
                # Record elements
                for elem in node.elements:
                    self._record_observations(elem, context, substitutions, function_name, arg_position)

        elif isinstance(node, VariableNode):
            var_name = node.name

            if var_name in context_names:
                # Lambda-bound variable
                position = context_names[var_name]
                var_type = context[position][1]
                resolved_type = substitute_type_vars(var_type, substitutions)
                self.dists.record(resolved_type, resolved_context, VariableStrategy(position), function_name, arg_position)

        elif isinstance(node, LambdaNode):
            # Get param types from the node's inferred type
            # node.param is always a list (even for single-parameter lambdas)
            node_type = self._get_node_type(node, substitutions)
            if node_type is not None:
                base = get_base_type(node_type)
                if base == CallableOrig:
                    type_args = get_args(node_type)
                    if len(type_args) == 2 and isinstance(type_args[0], list):
                        param_types = type_args[0]
                        if len(param_types) == len(node.param):
                            # Add all parameters to context
                            new_context = context
                            for param_name, param_type in zip(node.param, param_types):
                                new_context = new_context + [(param_name, param_type)]

                            # Record body observations (body is not a direct argument, so no arg_position)
                            self._record_observations(node.body, new_context, substitutions, "lambda", None)

                            # Record this lambda
                            self.dists.record(node_type, resolved_context, LambdaStrategy(), function_name, arg_position)
                            return

            # Fallback: use fresh type vars for all parameters
            from typing import TypeVar
            new_context = context
            for param_name in node.param:
                param_type = TypeVar(f"_{param_name}")
                new_context = new_context + [(param_name, param_type)]
            self._record_observations(node.body, new_context, substitutions, "lambda", None)

        elif isinstance(node, IfNode):
            # Record children (if condition/branches are not numbered arguments)
            self._record_observations(node.condition, context, substitutions, "if", None)
            self._record_observations(node.then_expr, context, substitutions, "if", None)
            self._record_observations(node.else_expr, context, substitutions, "if", None)

            # Record this if
            node_type = self._get_node_type(node, substitutions)
            if node_type is not None:
                self.dists.record(node_type, resolved_context, IfStrategy(), function_name, arg_position)

        elif isinstance(node, ApplicationNode):
            func = node.function
            args = node.arguments

            call_function_name = None
            if isinstance(func, VariableNode):
                func_name = func.name
                if func_name in self.grammar.names:
                    call_function_name = func_name
                elif func_name in context_names:
                    position = context_names[func_name]
                    call_function_name = f"@{position}"

            arg_function_name = call_function_name if call_function_name is not None else function_name

            # Record argument observations with their position
            for i, arg in enumerate(args):
                self._record_observations(arg, context, substitutions, arg_function_name, i)

            if isinstance(func, VariableNode):
                func_name = func.name

                if func_name in self.grammar.names:
                    # Grammar function application
                    node_type = self._get_node_type(node, substitutions)
                    if node_type is not None:
                        self.dists.record(node_type, resolved_context, ApplicationStrategy(func_name), function_name, arg_position)

                elif func_name in context_names:
                    # Higher-order function application
                    position = context_names[func_name]
                    node_type = self._get_node_type(node, substitutions)
                    if node_type is not None:
                        self.dists.record(node_type, resolved_context, ApplicationStrategy(f"@{position}"), function_name, arg_position)
            else:
                # Complex function expression
                self._record_observations(func, context, substitutions, function_name, arg_position)


def load(
    txt_path: Path,
    grammar: Grammar,
    expected_type: TypeType = CallableOrig[[list[int]], list[int]],
    include_function_name_in_context: bool = False,
    include_arg_position_in_context: bool = False
) -> EmpiricalDistributions:
    """
    Load and analyse programs from a text file.

    Args:
        txt_path: Path to text file with one program per line
        grammar: Grammar to use for type inference
        expected_type: The expected type of all programs in the file.
                      Defaults to Callable[[list[int]], list[int]] for functions.txt.
        include_function_name_in_context: Whether to include function name in
            the empirical context signature.
        include_arg_position_in_context: Whether to include argument position in
            the empirical context signature.

    Returns:
        EmpiricalDistributions with learned distributions
    """
    dists = EmpiricalDistributions(
        include_function_name_in_context=include_function_name_in_context,
        include_arg_position_in_context=include_arg_position_in_context
    )
    analyser = ASTAnalyser(grammar, dists)

    with open(txt_path, 'r') as f:
        for line in f:
            program_str = line.strip()
            if not program_str:
                continue

            try:
                ast = parse_program(program_str)
                analyser.analyse(ast, expected_type)
            except Exception:
                # Skip unparseable programs
                continue

    dists.normalize()
    return dists


# ============================================================================
# Empirical Composer
# ============================================================================

def min_depth_for_type(type_: TypeType) -> int:
    """
    Compute the minimum depth required to generate a value of this type.

    - Atomic types (int, bool) and list (empty list) need depth 0
    - Callable types need depth 1 (for Lambda)
    - Types that can only be produced by functions need depth based on args
    """
    base = get_base_type(type_)

    # Atomic types can be generated as literals at depth 0
    if type_ == int or type_ == bool:
        return 0

    # Lists can be generated as empty list at depth 0
    if base == list:
        return 0

    # Callable types need Lambda, which needs depth > 0
    if base == CallableOrig:
        return 1

    # For other types, conservative estimate
    return 1


class EmpiricalComposer(Composer):
    """
    Generates programs using empirical distributions learned from examples.

    Uses learned P(strategy | return_type, context) to weight generation choices.
    Falls back to marginal distribution when exact context hasn't been seen.
    """

    _cached_distributions: dict[tuple[str, int, bool, bool], EmpiricalDistributions] = {}

    @classmethod
    def get_name(cls) -> str:
        return "empirical"

    def __init__(
        self,
        seed: int,
        grammar: Grammar,
        functions_path: Optional[Path] = None,
        noise: float = 0.0,
        include_function_name_in_context: bool = True,
        include_arg_position_in_context: bool = True
    ):
        """
        Initialise the empirical composer.

        Args:
            seed: Random seed for deterministic generation
            grammar: Grammar containing available functions
            functions_path: Path to training programs (one per line)
            noise: Floor weight for compatible strategies (0 = strict, >0 = exploration)
            include_function_name_in_context: Whether to include function name in
                the empirical context signature.
            include_arg_position_in_context: Whether to include argument position in
                the empirical context signature.
        """
        super().__init__(seed, grammar)
        self.noise = noise

        if functions_path is None:
            functions_path = Path(__file__).parent / 'data' / 'functions.txt'

        cache_key = (
            str(functions_path.resolve()),
            id(grammar),
            include_function_name_in_context,
            include_arg_position_in_context
        )
        if cache_key not in EmpiricalComposer._cached_distributions:
            if functions_path.exists():
                EmpiricalComposer._cached_distributions[cache_key] = \
                    load(
                        functions_path,
                        grammar,
                        include_function_name_in_context=include_function_name_in_context,
                        include_arg_position_in_context=include_arg_position_in_context
                    )
            else:
                fallback = EmpiricalDistributions(
                    include_function_name_in_context=include_function_name_in_context,
                    include_arg_position_in_context=include_arg_position_in_context
                )
                fallback.normalize()
                EmpiricalComposer._cached_distributions[cache_key] = fallback

        self.dists = EmpiricalComposer._cached_distributions[cache_key]

    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None
    ) -> ASTNode:
        """
        Generate a program using empirical weights.

        Args:
            target_type: The desired output type
            depth: Maximum remaining depth
            context: Variable bindings in scope (name -> type)
            substitutions: Current type variable substitutions

        Returns:
            An AST node of the target type
        """
        if context is None:
            context = {}
        if substitutions is None:
            substitutions = SubstitutionTable()

        # Convert context dict to ordered list for internal use
        # We use insertion order (Python 3.7+ dicts maintain order)
        context_list = [(name, typ) for name, typ in context.items()]

        return self._generate_internal(target_type, depth, context_list, substitutions)

    def _generate_internal(
        self,
        target_type: TypeType,
        depth: int,
        context: list[tuple[str, TypeType]],
        substitutions: SubstitutionTable,
        function_name: Optional[str] = None,
        arg_position: Optional[int] = None
    ) -> ASTNode:
        """Internal generation with ordered context list."""
        target = substitute_type_vars(target_type, substitutions)
        base_type = get_base_type(target)
        context_types = [t for _, t in context]

        # Build available strategies
        strategies: list[Strategy] = []
        strategy_data: list[any] = []  # Associated data for each strategy

        # Literal
        if target == int or target == bool or base_type == list:
            lit_type = 'int' if target == int else ('bool' if target == bool else 'list')
            strategies.append(LiteralStrategy(lit_type))
            strategy_data.append(None)

        # Variables
        for i, (var_name, var_type) in enumerate(context):
            subs_copy = substitutions.copy()
            if matchable(var_type, target, subs_copy):
                strategies.append(VariableStrategy(i))
                strategy_data.append((var_name, subs_copy))

        # Lambda
        if base_type == CallableOrig and depth > 0:
            strategies.append(LambdaStrategy())
            strategy_data.append(None)

        # If (need depth > 1 for Callable targets since branches need Lambda)
        if depth > 0:
            # For Callable types, If needs depth > 1 so branches can use Lambda
            if base_type != CallableOrig or depth > 1:
                strategies.append(IfStrategy())
                strategy_data.append(None)

        # Function applications
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
                    strategies.append(ApplicationStrategy(func_name))
                    strategy_data.append((func_name, func_subs))

        if not strategies:
            raise ValueError(f"Cannot generate type {target_type} at depth {depth}")

        # Get empirical weights
        weights = self.dists.get_strategy_weights(
            target,
            context_types,
            strategies,
            self.noise,
            function_name=function_name,
            arg_position=arg_position
        )

        # Sample
        idx = self.rng.choices(range(len(strategies)), weights=weights, k=1)[0]
        strategy = strategies[idx]
        data = strategy_data[idx]

        # Execute strategy
        if isinstance(strategy, LiteralStrategy):
            return self._sample_literal_empirical(target, substitutions)

        elif isinstance(strategy, VariableStrategy):
            var_name, new_subs = data
            substitutions.update(new_subs)
            return VariableNode(var_name)

        elif isinstance(strategy, LambdaStrategy):
            return self._generate_lambda(target, depth, context, substitutions)

        elif isinstance(strategy, IfStrategy):
            return self._generate_if(target, depth, context, substitutions, function_name)

        elif isinstance(strategy, ApplicationStrategy):
            func_name, func_subs = data
            return self._generate_application(
                func_name,
                func_subs,
                depth,
                context,
                substitutions
            )

        raise ValueError(f"Unknown strategy: {strategy}")

    def _sample_literal_empirical(
        self,
        type_: TypeType,
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Sample a literal using empirical distributions."""
        actual_type = substitute_type_vars(type_, substitutions)
        base_type = get_base_type(actual_type)

        if actual_type == int:
            value = self.dists.sample_int_constant(self.rng)
            return NumberNode(value)
        elif actual_type == bool:
            value = self.dists.sample_bool_constant(self.rng)
            return BooleanNode(value)
        elif base_type == list:
            return ListNode([])
        else:
            raise ValueError(f"Cannot sample literal for type: {actual_type}")

    def _generate_lambda(
        self,
        target_type: TypeType,
        depth: int,
        context: list[tuple[str, TypeType]],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a lambda expression."""
        target = substitute_type_vars(target_type, substitutions)
        type_args = get_args(target)

        if len(type_args) != 2:
            raise ValueError(f"Callable must have 2 type args: {target}")

        param_types_list, ret_type = type_args
        if not isinstance(param_types_list, list):
            raise ValueError(f"Expected param list: {param_types_list}")

        # Build nested lambdas
        new_context = list(context)
        param_names = []

        for param_type in param_types_list:
            param_name = self._fresh_var_name()
            param_names.append(param_name)
            new_context.append((param_name, param_type))

        # Generate body
        body = self._generate_internal(
            ret_type,
            depth - 1,
            new_context,
            substitutions,
            function_name="lambda"
        )

        # Wrap in lambdas
        for param_name in reversed(param_names):
            body = LambdaNode(param_name, body)

        return body

    def _generate_if(
        self,
        target_type: TypeType,
        depth: int,
        context: list[tuple[str, TypeType]],
        substitutions: SubstitutionTable,
        function_name: Optional[str]
    ) -> ASTNode:
        """Generate an if expression."""
        condition = self._generate_internal(
            bool,
            depth - 1,
            context,
            substitutions,
            function_name="if"
        )
        then_expr = self._generate_internal(
            target_type,
            depth - 1,
            context,
            substitutions,
            function_name="if"
        )
        else_expr = self._generate_internal(
            target_type,
            depth - 1,
            context,
            substitutions,
            function_name="if"
        )
        return IfNode(condition, then_expr, else_expr)

    def _generate_application(
        self,
        func_name: str,
        func_subs: SubstitutionTable,
        depth: int,
        context: list[tuple[str, TypeType]],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a function application."""
        func_info = self.grammar[func_name]

        # Instantiate free type variables
        for arg_type in func_info['arg_types']:
            func_subs = self._instantiate_free_types(arg_type, func_subs)
        func_subs = self._instantiate_free_types(func_info['ret_type'], func_subs)

        substitutions.update(func_subs)

        # Generate arguments with their positions
        args = []
        for i, arg_type in enumerate(func_info['arg_types']):
            arg = self._generate_internal(
                arg_type,
                depth - 1,
                context,
                substitutions,
                function_name=func_name,
                arg_position=i
            )
            args.append(arg)

        return ApplicationNode(VariableNode(func_name), args)
