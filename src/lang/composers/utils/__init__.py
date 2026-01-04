"""
Composer Utilities

This module provides utility classes and functions used by program composers:
- strategies: Strategy definitions for program generation
- guard: Guards against trivial generation patterns
"""

from .strategies import (
    Strategy,
    LiteralStrategy,
    VariableStrategy,
    LambdaStrategy,
    IfStrategy,
    ApplicationStrategy,
    PredicatePattern,
    TransformPattern,
    KeyPattern,
    is_literal_strategy,
    is_variable_strategy,
    is_application_strategy,
    get_strategy_type,
)
from .guard import (
    StrategyGuard,
    ApplicationContext,
    GuardRule,
    apply_guard,
    get_default_guard,
    guard_predicate_weights,
    guard_transform_weights,
    guard_key_weights,
)

__all__ = [
    # Strategies
    'Strategy',
    'LiteralStrategy',
    'VariableStrategy',
    'LambdaStrategy',
    'IfStrategy',
    'ApplicationStrategy',
    'PredicatePattern',
    'TransformPattern',
    'KeyPattern',
    'is_literal_strategy',
    'is_variable_strategy',
    'is_application_strategy',
    'get_strategy_type',
    # Guard
    'StrategyGuard',
    'ApplicationContext',
    'GuardRule',
    'apply_guard',
    'get_default_guard',
    'is_identity_lambda',
    'is_literal_node',
    'is_trivial_application',
    'guard_predicate_weights',
    'guard_transform_weights',
    'guard_key_weights',
    'block_literal_for',
    'block_identity_for',
    'block_nested_application',
    'require_nonliteral_for_binary',
]
