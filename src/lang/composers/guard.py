"""
Strategy Guard for Program Composers

This module provides guards that prevent trivial generation strategies during
program composition. Guards modify strategy weights to ensure generated programs
are semantically meaningful.

Usage:
    guard = StrategyGuard()
    weights = guard.apply(fn_name, arg_pos, strategies_with_weights, other_arg_strategies)

The guard is designed to be:
1. Easy to extend with new rules
2. Efficient with O(1) function name lookup
3. Composable with any composer that uses strategy weights

Guard Rules:
- If nodes: condition cannot be boolean literal
- Binary operators: at least one non-literal argument
- Predicate functions: argument cannot be literal
- Higher-order functions: lambda body cannot be identity
- Redundant compositions: prevent (first (singleton x)), (reverse (reverse x)), etc.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from collections import defaultdict

from .strategies import (
    Strategy,
    LiteralStrategy,
    VariableStrategy,
    LambdaStrategy,
    IfStrategy,
    ApplicationStrategy,
    PredicatePattern,
    TransformPattern,
)


# ============================================================================
# Application Context - passed through generation for guard decisions
# ============================================================================

@dataclass
class ApplicationContext:
    """
    Context for guard decisions when generating function arguments.

    This is passed through recursive generation calls to enable guards
    to make decisions based on the enclosing function application.

    Attributes:
        fn_name: Name of the function we're generating an argument for
        arg_pos: Position of this argument (0-indexed)
        num_args: Total number of arguments for the function
        other_strategies: Strategy types used for other arguments (None if not yet generated)
    """
    fn_name: str
    arg_pos: int
    num_args: int
    other_strategies: list[Optional[str]] = field(default_factory=list)

    def with_strategy(self, pos: int, strategy_type: str) -> 'ApplicationContext':
        """Return a new context with the given strategy recorded."""
        new_strategies = self.other_strategies.copy()
        if pos < len(new_strategies):
            new_strategies[pos] = strategy_type
        return ApplicationContext(
            fn_name=self.fn_name,
            arg_pos=self.arg_pos,
            num_args=self.num_args,
            other_strategies=new_strategies
        )

    @classmethod
    def for_function(cls, fn_name: str, num_args: int, current_pos: int) -> 'ApplicationContext':
        """Create a context for generating the current_pos argument of fn_name."""
        return cls(
            fn_name=fn_name,
            arg_pos=current_pos,
            num_args=num_args,
            other_strategies=[None] * num_args
        )


# ============================================================================
# Guard Rule Types
# ============================================================================

@dataclass
class GuardRule:
    """
    A single guard rule that can block certain strategies.

    Attributes:
        name: Human-readable name for debugging
        fn_names: Set of function names this rule applies to (None = all)
        arg_positions: Set of argument positions this rule applies to (None = all)
        condition: Function that returns True if strategy should be BLOCKED
                   Signature: (fn_name, arg_pos, strategy, other_strategies) -> bool
    """
    name: str
    fn_names: Optional[set[str]]
    arg_positions: Optional[set[int]]
    condition: Callable[[str, int, str, list[Optional[str]]], bool]

    def applies_to(self, fn_name: str, arg_pos: int) -> bool:
        """Check if this rule applies to the given function and position."""
        if self.fn_names is not None and fn_name not in self.fn_names:
            return False
        if self.arg_positions is not None and arg_pos not in self.arg_positions:
            return False
        return True

    def should_block(
        self,
        fn_name: str,
        arg_pos: int,
        strategy_type: str,
        other_strategies: list[Optional[str]]
    ) -> bool:
        """Check if this rule blocks the given strategy."""
        return self.condition(fn_name, arg_pos, strategy_type, other_strategies)


# ============================================================================
# Strategy Guard Class
# ============================================================================

class StrategyGuard:
    """
    Guards against trivial generation strategies.

    The guard maintains a registry of rules indexed by function name for
    efficient lookup. Rules can be added dynamically.

    Strategy types used:
    - 'literal': LiteralStrategy (int, bool, or list literal)
    - 'variable': VariableStrategy
    - 'lambda': LambdaStrategy
    - 'if': IfStrategy
    - 'apply': ApplicationStrategy
    - 'identity': Special marker for identity lambda bodies
    """

    # Strategy type constants
    LITERAL = 'literal'
    VARIABLE = 'variable'
    LAMBDA = 'lambda'
    IF = 'if'
    APPLY = 'apply'
    IDENTITY = 'identity'  # Special marker for lambda x x

    def __init__(self):
        """Initialize with default guard rules."""
        # Index: fn_name -> list of rules
        # None key holds rules that apply to all functions
        self._rules_by_fn: dict[Optional[str], list[GuardRule]] = defaultdict(list)
        self._all_rules: list[GuardRule] = []

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self):
        """Register the default set of guard rules."""

        # -----------------------------------------------------------------
        # Rule: If condition cannot be boolean literal
        # -----------------------------------------------------------------
        self.add_rule(GuardRule(
            name="if_no_literal_condition",
            fn_names={'if'},
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == self.LITERAL
        ))

        # -----------------------------------------------------------------
        # Rule: Binary operators need at least one non-literal
        # -----------------------------------------------------------------
        binary_ops = {'+', '-', '*', '/', '%', '<', '>', '==', '!=', '<=', '>=', 'and', 'or'}

        def binary_needs_nonliteral(fn: str, pos: int, strat: str, others: list[Optional[str]]) -> bool:
            if strat != self.LITERAL:
                return False
            # Block literal if ALL other args are also literals
            for i, other in enumerate(others):
                if i != pos and other is not None and other != self.LITERAL:
                    return False  # There's a non-literal, so this literal is OK
            # All others are literals (or not yet generated), block this literal
            # But only if we're the last arg being generated
            if pos == len(others) - 1 or all(o is not None for o in others):
                return all(o == self.LITERAL for o in others if o is not None)
            return False

        self.add_rule(GuardRule(
            name="binary_ops_need_nonliteral",
            fn_names=binary_ops,
            arg_positions=None,  # All positions
            condition=binary_needs_nonliteral
        ))

        # -----------------------------------------------------------------
        # Rule: Unary predicates cannot take literals
        # -----------------------------------------------------------------
        unary_predicates = {'is_even', 'is_odd', 'not'}

        self.add_rule(GuardRule(
            name="unary_predicate_no_literal",
            fn_names=unary_predicates,
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == self.LITERAL
        ))

        # -----------------------------------------------------------------
        # Rule: Higher-order functions cannot have identity lambdas
        # -----------------------------------------------------------------
        ho_functions = {
            'map', 'mapi', 'filter', 'filteri', 'fold', 'foldi',
            'sort', 'group', 'count', 'find'
        }

        self.add_rule(GuardRule(
            name="ho_no_identity_lambda",
            fn_names=ho_functions,
            arg_positions={0},  # Lambda is always first arg
            condition=lambda fn, pos, strat, others: strat == self.IDENTITY
        ))

        # -----------------------------------------------------------------
        # Rule: (first (singleton x)) is trivial - singleton shouldn't follow first
        # -----------------------------------------------------------------
        self.add_rule(GuardRule(
            name="first_blocks_singleton_strategy",
            fn_names={'first'},
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == 'apply:singleton'
        ))

        # -----------------------------------------------------------------
        # Rule: (length []) is trivial - empty list shouldn't be used with length
        # -----------------------------------------------------------------
        self.add_rule(GuardRule(
            name="length_no_empty_list",
            fn_names={'length'},
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == self.LITERAL
        ))

        # -----------------------------------------------------------------
        # Rule: (reverse (reverse x)) is trivial
        # -----------------------------------------------------------------
        self.add_rule(GuardRule(
            name="reverse_no_reverse",
            fn_names={'reverse'},
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == 'apply:reverse'
        ))

        # -----------------------------------------------------------------
        # Rule: (unique (unique x)) is trivial
        # -----------------------------------------------------------------
        self.add_rule(GuardRule(
            name="unique_no_unique",
            fn_names={'unique'},
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == 'apply:unique'
        ))

        # -----------------------------------------------------------------
        # Rule: (flatten (singleton x)) simplifies to [x]
        # -----------------------------------------------------------------
        self.add_rule(GuardRule(
            name="flatten_no_singleton",
            fn_names={'flatten'},
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == 'apply:singleton'
        ))

        # -----------------------------------------------------------------
        # Rule: (last (singleton x)) is trivial
        # -----------------------------------------------------------------
        self.add_rule(GuardRule(
            name="last_blocks_singleton_strategy",
            fn_names={'last'},
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == 'apply:singleton'
        ))

        # -----------------------------------------------------------------
        # Rule: (sum []) / (product []) / (max []) / (min []) are trivial or error
        # -----------------------------------------------------------------
        list_reducers = {'sum', 'product', 'max', 'min'}
        self.add_rule(GuardRule(
            name="list_reducer_no_empty_literal",
            fn_names=list_reducers,
            arg_positions={0},
            condition=lambda fn, pos, strat, others: strat == self.LITERAL
        ))

    def add_rule(self, rule: GuardRule):
        """
        Add a guard rule to the registry.

        Args:
            rule: The GuardRule to add
        """
        self._all_rules.append(rule)

        if rule.fn_names is None:
            # Rule applies to all functions
            self._rules_by_fn[None].append(rule)
        else:
            # Index by each function name
            for fn_name in rule.fn_names:
                self._rules_by_fn[fn_name].append(rule)

    def get_applicable_rules(self, fn_name: str, arg_pos: int) -> list[GuardRule]:
        """
        Get all rules that apply to a function and argument position.

        Args:
            fn_name: Name of the function
            arg_pos: Position of the argument (0-indexed)

        Returns:
            List of applicable GuardRules
        """
        rules = []

        # Add function-specific rules
        if fn_name in self._rules_by_fn:
            for rule in self._rules_by_fn[fn_name]:
                if rule.applies_to(fn_name, arg_pos):
                    rules.append(rule)

        # Add global rules
        for rule in self._rules_by_fn[None]:
            if rule.applies_to(fn_name, arg_pos):
                rules.append(rule)

        return rules

    def should_block_strategy(
        self,
        fn_name: str,
        arg_pos: int,
        strategy_type: str,
        other_strategies: list[Optional[str]]
    ) -> bool:
        """
        Check if a strategy should be blocked.

        Args:
            fn_name: Name of the function being called
            arg_pos: Position of argument being generated
            strategy_type: Type of strategy ('literal', 'variable', 'apply:func', etc.)
            other_strategies: Strategies for other arguments (None if not yet generated)

        Returns:
            True if the strategy should be blocked (weight set to 0)
        """
        rules = self.get_applicable_rules(fn_name, arg_pos)

        for rule in rules:
            if rule.should_block(fn_name, arg_pos, strategy_type, other_strategies):
                return True

        return False

    def apply(
        self,
        fn_name: str,
        arg_pos: int,
        weights: dict[str, float],
        other_strategies: list[Optional[str]]
    ) -> dict[str, float]:
        """
        Apply guards to strategy weights.

        This is the main entry point for composers. It takes current weights
        and returns modified weights with blocked strategies set to 0.

        Args:
            fn_name: Name of the function being called
            arg_pos: Position of argument being generated
            weights: Current strategy weights (strategy_type -> weight)
            other_strategies: Strategies for other arguments (None if not yet generated)

        Returns:
            Modified weights with blocked strategies zeroed out
        """
        # Fast path: if no rules apply, return unchanged
        rules = self.get_applicable_rules(fn_name, arg_pos)
        if not rules:
            return weights

        # Apply guards
        result = {}
        for strategy_type, weight in weights.items():
            if self.should_block_strategy(fn_name, arg_pos, strategy_type, other_strategies):
                result[strategy_type] = 0.0
            else:
                result[strategy_type] = weight

        return result

    def apply_with_context(
        self,
        ctx: Optional['ApplicationContext'],
        weights: dict[str, float]
    ) -> dict[str, float]:
        """
        Apply guards using an ApplicationContext.

        Convenience method that extracts fn_name, arg_pos, other_strategies from context.

        Args:
            ctx: Application context (if None, returns weights unchanged)
            weights: Current strategy weights

        Returns:
            Modified weights with blocked strategies zeroed out
        """
        if ctx is None:
            return weights
        return self.apply(ctx.fn_name, ctx.arg_pos, weights, ctx.other_strategies)

    def should_block_with_context(
        self,
        ctx: Optional['ApplicationContext'],
        strategy_type: str
    ) -> bool:
        """
        Check if a strategy should be blocked using ApplicationContext.

        Args:
            ctx: Application context (if None, returns False)
            strategy_type: Strategy type to check

        Returns:
            True if strategy should be blocked
        """
        if ctx is None:
            return False
        return self.should_block_strategy(
            ctx.fn_name, ctx.arg_pos, strategy_type, ctx.other_strategies
        )

    def apply_to_strategies(
        self,
        fn_name: str,
        arg_pos: int,
        strategies: list[tuple[Strategy, float]],
        other_strategies: list[Optional[Strategy]]
    ) -> list[tuple[Strategy, float]]:
        """
        Apply guards to a list of (Strategy, weight) pairs.

        This variant works directly with Strategy objects.

        Args:
            fn_name: Name of the function being called
            arg_pos: Position of argument being generated
            strategies: List of (Strategy, weight) pairs
            other_strategies: Strategies for other arguments (None if not yet generated)

        Returns:
            Modified list with blocked strategies zeroed out
        """
        # Convert other strategies to type strings
        other_types = [
            self._strategy_to_type(s) if s is not None else None
            for s in other_strategies
        ]

        result = []
        for strategy, weight in strategies:
            strategy_type = self._strategy_to_type(strategy)
            if self.should_block_strategy(fn_name, arg_pos, strategy_type, other_types):
                result.append((strategy, 0.0))
            else:
                result.append((strategy, weight))

        return result

    def _strategy_to_type(self, strategy: Strategy) -> str:
        """Convert a Strategy object to a type string for rule matching."""
        if isinstance(strategy, LiteralStrategy):
            return self.LITERAL
        elif isinstance(strategy, VariableStrategy):
            return self.VARIABLE
        elif isinstance(strategy, LambdaStrategy):
            return self.LAMBDA
        elif isinstance(strategy, IfStrategy):
            return self.IF
        elif isinstance(strategy, ApplicationStrategy):
            # Include function name for more specific matching
            return f"apply:{strategy.func_name}"
        else:
            return 'unknown'

    def guard_lambda_body(
        self,
        body_strategy: str,
        param_names: list[str],
        body_references_only_params: bool
    ) -> bool:
        """
        Check if a lambda body strategy should be blocked.

        This specifically handles the "lambda x x" (identity) case.

        Args:
            body_strategy: Strategy type for the lambda body
            param_names: Names of lambda parameters
            body_references_only_params: True if body only references parameters

        Returns:
            True if the body strategy should be blocked
        """
        # Block identity lambdas (body is just a parameter reference)
        if body_strategy == self.VARIABLE and body_references_only_params:
            return True
        return False

    def guard_predicate_pattern(
        self,
        pattern: PredicatePattern,
        fn_name: str
    ) -> bool:
        """
        Check if a predicate pattern should be blocked.

        Args:
            pattern: The predicate pattern
            fn_name: The higher-order function using this predicate

        Returns:
            True if the pattern should be blocked
        """
        # Block trivial predicates (literal true/false)
        return pattern.is_trivial()

    def guard_transform_pattern(
        self,
        pattern: TransformPattern,
        fn_name: str,
        must_transform: bool = True
    ) -> bool:
        """
        Check if a transform pattern should be blocked.

        Args:
            pattern: The transform pattern
            fn_name: The higher-order function using this transform
            must_transform: If True, block identity transforms

        Returns:
            True if the pattern should be blocked
        """
        if must_transform and pattern.is_trivial():
            return True
        return False


# ============================================================================
# Global Guard Instance
# ============================================================================

# Singleton guard instance for convenience
_default_guard: Optional[StrategyGuard] = None


def get_default_guard() -> StrategyGuard:
    """Get the default StrategyGuard instance."""
    global _default_guard
    if _default_guard is None:
        _default_guard = StrategyGuard()
    return _default_guard


def apply_guard(
    fn_name: str,
    arg_pos: int,
    weights: dict[str, float],
    other_strategies: list[Optional[str]]
) -> dict[str, float]:
    """
    Convenience function to apply the default guard.

    Args:
        fn_name: Name of the function being called
        arg_pos: Position of argument being generated
        weights: Current strategy weights
        other_strategies: Strategies for other arguments

    Returns:
        Modified weights with blocked strategies zeroed out
    """
    return get_default_guard().apply(fn_name, arg_pos, weights, other_strategies)


# ============================================================================
# Rule Builder Helpers
# ============================================================================

def block_literal_for(fn_names: set[str], positions: Optional[set[int]] = None) -> GuardRule:
    """
    Create a rule that blocks literal strategies for specific functions.

    Args:
        fn_names: Set of function names
        positions: Set of argument positions (None = all)

    Returns:
        A GuardRule that blocks literals
    """
    return GuardRule(
        name=f"block_literal_for_{','.join(sorted(fn_names))}",
        fn_names=fn_names,
        arg_positions=positions,
        condition=lambda fn, pos, strat, others: strat == StrategyGuard.LITERAL
    )


def block_identity_for(fn_names: set[str]) -> GuardRule:
    """
    Create a rule that blocks identity strategies for specific functions.

    Args:
        fn_names: Set of function names (typically HO functions)

    Returns:
        A GuardRule that blocks identity lambdas
    """
    return GuardRule(
        name=f"block_identity_for_{','.join(sorted(fn_names))}",
        fn_names=fn_names,
        arg_positions={0},
        condition=lambda fn, pos, strat, others: strat == StrategyGuard.IDENTITY
    )


def block_nested_application(outer_fn: str, inner_fn: str) -> GuardRule:
    """
    Create a rule that blocks nested applications like (f (g x)).

    Args:
        outer_fn: The outer function name
        inner_fn: The inner function name to block

    Returns:
        A GuardRule that blocks the nesting
    """
    return GuardRule(
        name=f"block_{outer_fn}_of_{inner_fn}",
        fn_names={outer_fn},
        arg_positions={0},
        condition=lambda fn, pos, strat, others: strat == f"apply:{inner_fn}"
    )


def require_nonliteral_for_binary(fn_names: set[str]) -> GuardRule:
    """
    Create a rule that requires at least one non-literal for binary ops.

    Args:
        fn_names: Set of binary operator names

    Returns:
        A GuardRule enforcing the constraint
    """
    def condition(fn: str, pos: int, strat: str, others: list[Optional[str]]) -> bool:
        if strat != StrategyGuard.LITERAL:
            return False
        # Only block if this would make all args literals
        if pos == 1:  # Second arg
            return others[0] == StrategyGuard.LITERAL
        return False

    return GuardRule(
        name=f"require_nonliteral_for_{','.join(sorted(fn_names))}",
        fn_names=fn_names,
        arg_positions=None,
        condition=condition
    )


# ============================================================================
# AST-Level Guard Utilities
# ============================================================================

# Import AST nodes for checking (lazy to avoid circular imports)
_ast_nodes_imported = False
_NumberNode = None
_BooleanNode = None
_ListNode = None
_VariableNode = None
_LambdaNode = None
_ApplicationNode = None


def _ensure_ast_imports():
    """Lazy import of AST node types."""
    global _ast_nodes_imported, _NumberNode, _BooleanNode, _ListNode
    global _VariableNode, _LambdaNode, _ApplicationNode

    if not _ast_nodes_imported:
        from ..ast_nodes import (
            NumberNode, BooleanNode, ListNode,
            VariableNode, LambdaNode, ApplicationNode
        )
        _NumberNode = NumberNode
        _BooleanNode = BooleanNode
        _ListNode = ListNode
        _VariableNode = VariableNode
        _LambdaNode = LambdaNode
        _ApplicationNode = ApplicationNode
        _ast_nodes_imported = True


def is_identity_lambda(node) -> bool:
    """
    Check if a lambda is an identity function (λx.x).

    Args:
        node: The AST node to check

    Returns:
        True if the node is a lambda that just returns its parameter
    """
    _ensure_ast_imports()

    if not isinstance(node, _LambdaNode):
        return False

    # Get parameter name(s)
    params = node.param if isinstance(node.param, list) else [node.param]

    # Check if body is just a variable reference to the (only) parameter
    if len(params) == 1 and isinstance(node.body, _VariableNode):
        return node.body.name == params[0]

    return False


def is_literal_node(node) -> bool:
    """
    Check if a node is a literal value.

    Args:
        node: The AST node to check

    Returns:
        True if the node is a number, boolean, or empty list literal
    """
    _ensure_ast_imports()

    if isinstance(node, (_NumberNode, _BooleanNode)):
        return True
    if isinstance(node, _ListNode) and len(node.elements) == 0:
        return True
    return False


def is_trivial_application(node) -> bool:
    """
    Check if an application is trivial (e.g., all literal args).

    Args:
        node: The AST node to check

    Returns:
        True if the application is trivial
    """
    _ensure_ast_imports()

    if not isinstance(node, _ApplicationNode):
        return False

    # Check if all arguments are literals
    return all(is_literal_node(arg) for arg in node.arguments)


def guard_predicate_weights(
    weights: dict[str, float],
    must_be_meaningful: bool = True
) -> dict[str, float]:
    """
    Apply guards to predicate pattern weights.

    Args:
        weights: Pattern name -> weight mapping
        must_be_meaningful: If True, block trivial patterns

    Returns:
        Modified weights with trivial patterns zeroed
    """
    if not must_be_meaningful:
        return weights

    result = weights.copy()
    # Block literal true/false predicates
    trivial_patterns = {'literal', 'trivial', 'variable'}
    for pattern in trivial_patterns:
        if pattern in result:
            result[pattern] = 0.0
    return result


def guard_transform_weights(
    weights: dict[str, float],
    allow_identity: bool = False
) -> dict[str, float]:
    """
    Apply guards to transform pattern weights.

    Args:
        weights: Pattern name -> weight mapping
        allow_identity: If False, block identity transforms

    Returns:
        Modified weights with blocked patterns zeroed
    """
    if allow_identity:
        return weights

    result = weights.copy()
    if 'identity' in result:
        result['identity'] = 0.0
    return result


def guard_key_weights(
    weights: dict[str, float],
    allow_identity: bool = True
) -> dict[str, float]:
    """
    Apply guards to key function pattern weights.

    Args:
        weights: Pattern name -> weight mapping
        allow_identity: If True, allow identity key functions (for sort ascending)

    Returns:
        Modified weights
    """
    # For key functions, identity is often meaningful (sort ascending)
    # so we allow it by default
    if allow_identity:
        return weights

    result = weights.copy()
    if 'identity' in result:
        result['identity'] = 0.0
    return result
