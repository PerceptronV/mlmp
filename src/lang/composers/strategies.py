"""
Unified Strategy Definitions for Program Composers

This module defines the strategy types used by all composers for generating
program expressions. Strategies represent HOW to generate an expression at
a particular point in the AST.

Strategy Categories:
- Literal: Generate a literal value (int, bool, empty list)
- Variable: Use a variable from the current context
- Lambda: Generate a lambda expression
- If: Generate a conditional expression
- Application: Apply a function from the grammar
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Base Strategy Class
# ============================================================================

@dataclass(frozen=True)
class Strategy:
    """Base class for generation strategies."""
    pass


# ============================================================================
# Core Strategy Types
# ============================================================================

@dataclass(frozen=True)
class LiteralStrategy(Strategy):
    """
    Generate a literal value (number, boolean, or empty list).

    Attributes:
        literal_type: One of 'int', 'bool', or 'list'
    """
    literal_type: str  # 'int', 'bool', or 'list'

    def __repr__(self) -> str:
        return f"Literal({self.literal_type})"

    def is_int(self) -> bool:
        return self.literal_type == 'int'

    def is_bool(self) -> bool:
        return self.literal_type == 'bool'

    def is_list(self) -> bool:
        return self.literal_type == 'list'


@dataclass(frozen=True)
class VariableStrategy(Strategy):
    """
    Use the variable at a specific position in context.

    Attributes:
        position: Index of the variable in the context (0-indexed)
    """
    position: int

    def __repr__(self) -> str:
        return f"Variable({self.position})"


@dataclass(frozen=True)
class LambdaStrategy(Strategy):
    """
    Generate a lambda expression.

    The lambda body will be generated recursively based on the
    expected return type.
    """

    def __repr__(self) -> str:
        return "Lambda"


@dataclass(frozen=True)
class IfStrategy(Strategy):
    """
    Generate an if (conditional) expression.

    Generates: (if condition then_expr else_expr)
    """

    def __repr__(self) -> str:
        return "If"


@dataclass(frozen=True)
class ApplicationStrategy(Strategy):
    """
    Apply a specific function from the grammar.

    Attributes:
        func_name: Name of the function to apply, or "@N" for applying
                   the Nth variable in context (for higher-order usage)
    """
    func_name: str

    def __repr__(self) -> str:
        return f"Apply({self.func_name})"

    def is_variable_application(self) -> bool:
        """Check if this applies a context variable (higher-order)."""
        return self.func_name.startswith('@')

    def get_variable_position(self) -> Optional[int]:
        """Get variable position if this is a variable application."""
        if self.is_variable_application():
            return int(self.func_name[1:])
        return None


# ============================================================================
# Predicate Patterns (for filter, count, find)
# ============================================================================

@dataclass(frozen=True)
class PredicatePattern:
    """
    Pattern for predicate expressions used in higher-order functions.

    Common patterns:
    - 'is_even_odd': Apply is_even or is_odd
    - 'compare_const': Compare with constant (< x 50)
    - 'modulo_check': Check modulo remainder (== (% x 2) 0)
    - 'compound': Combine predicates with and/or
    - 'negation': Negate a predicate with not
    - 'membership': Check membership with is_in
    - 'literal': Boolean literal (trivial)
    - 'variable': Return a variable directly
    """
    pattern: str

    def __repr__(self) -> str:
        return f"Pred({self.pattern})"

    def is_trivial(self) -> bool:
        """Check if this pattern produces trivial predicates."""
        return self.pattern in ('literal', 'variable')


# ============================================================================
# Transform Patterns (for map)
# ============================================================================

@dataclass(frozen=True)
class TransformPattern:
    """
    Pattern for transform expressions used in map.

    Common patterns:
    - 'identity': Return input unchanged (x)
    - 'arithmetic': Apply arithmetic operation (+ x 1)
    - 'modulo': Apply modulo (% x 10)
    - 'conditional': Conditional transform (if pred then else)
    - 'singleton': Wrap in singleton list
    - 'constant': Return a constant (ignores input)
    """
    pattern: str

    def __repr__(self) -> str:
        return f"Trans({self.pattern})"

    def is_trivial(self) -> bool:
        """Check if this pattern produces trivial transforms."""
        return self.pattern == 'identity'


# ============================================================================
# Key Function Patterns (for sort, group)
# ============================================================================

@dataclass(frozen=True)
class KeyPattern:
    """
    Pattern for key function expressions used in sort/group.

    Common patterns:
    - 'identity': Use element as-is
    - 'negate': Negate for reverse sort (- 0 x)
    - 'modulo': Group by remainder (% x 10)
    - 'arithmetic': Apply arithmetic (+ x 1)
    """
    pattern: str

    def __repr__(self) -> str:
        return f"Key({self.pattern})"

    def is_trivial(self) -> bool:
        """Check if this pattern produces trivial key functions."""
        # identity is actually meaningful for sort (ascending order)
        return False


# ============================================================================
# Strategy Identification Utilities
# ============================================================================

def is_literal_strategy(strategy: Strategy) -> bool:
    """Check if a strategy produces a literal value."""
    return isinstance(strategy, LiteralStrategy)


def is_variable_strategy(strategy: Strategy) -> bool:
    """Check if a strategy uses a context variable."""
    return isinstance(strategy, VariableStrategy)


def is_application_strategy(strategy: Strategy) -> bool:
    """Check if a strategy applies a function."""
    return isinstance(strategy, ApplicationStrategy)


def get_strategy_type(strategy: Strategy) -> str:
    """Get a string identifier for the strategy type."""
    if isinstance(strategy, LiteralStrategy):
        return 'literal'
    elif isinstance(strategy, VariableStrategy):
        return 'variable'
    elif isinstance(strategy, LambdaStrategy):
        return 'lambda'
    elif isinstance(strategy, IfStrategy):
        return 'if'
    elif isinstance(strategy, ApplicationStrategy):
        return 'apply'
    else:
        return 'unknown'
