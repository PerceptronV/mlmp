"""
Composer Module

This module provides different strategies for generating well-typed programs.
Each composer implements a different generation strategy.

Available composers:
- RandomComposer: Uniform random sampling (original strategy)
- TemplateComposer: Template-based generation with hand-tuned weights
- EmpiricalComposer: Generation using distributions learned from Rule-MPS benchmark
- HybridComposer: Templates weighted by empirical distributions (best of both)
- HierarchicalComposer: Hierarchical empirical generation (learns structure at multiple levels)
"""

from typing import Type, Callable, Optional
from pathlib import Path

from .base import Composer
from .random import RandomComposer
from .template import TemplateComposer
from .empirical import EmpiricalComposer
from .hybrid import HybridComposer
from .hierarchical import HierarchicalComposer
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
)
from .guard import (
    StrategyGuard,
    ApplicationContext,
    apply_guard,
    get_default_guard,
    is_identity_lambda,
    is_literal_node,
    is_trivial_application,
    guard_predicate_weights,
    guard_transform_weights,
    guard_key_weights,
)
from ..grammar import Grammar

# Registry of available composers
COMPOSERS: dict[str, Type[Composer]] = {
    'random': RandomComposer,
    'template': TemplateComposer,
    'empirical': EmpiricalComposer,
    'hybrid': HybridComposer,
    'hierarchical': HierarchicalComposer,
}


def get_composer(
    name: str,
    seed: int,
    grammar: Grammar,
    functions_path: Optional[Path] = None,
    noise: Optional[float] = None,
    include_function_name_in_context: Optional[bool] = None,
    skeleton_depth: Optional[int] = None,
    ho_bias: Optional[float] = None
) -> Composer:
    """
    Get a composer instance by name.

    Args:
        name: Name of the composer ('random', 'template', 'empirical', 'hybrid', or 'hierarchical')
        seed: Random seed for reproducibility
        grammar: Grammar to use for generation
        functions_path: Path to functions file (used by empirical, hybrid, and hierarchical composers).
                       Defaults to data/functions.txt in composer directory.
        noise: Noise parameter for composers that support it (template, empirical, hybrid, hierarchical).
              If None, uses the composer's default.
        include_function_name_in_context: Whether to include function name in the
              empirical context signature. If None, uses the composer's default.
        skeleton_depth: Depth of skeleton extraction for hierarchical composer (1-3 recommended).
              If None, uses the composer's default (2).
        ho_bias: Higher-order function bias for hierarchical composer (0-1).
              Higher values produce more map/filter/sort programs. If None, uses default (0.5).

    Returns:
        A composer instance

    Raises:
        ValueError: If the composer name is not recognised
    """
    if name not in COMPOSERS:
        available = ', '.join(COMPOSERS.keys())
        raise ValueError(f"Unknown composer: {name}. Available: {available}")

    # Pass functions_path and noise to EmpiricalComposer and HybridComposer
    if name == 'empirical':
        include_fn = (
            include_function_name_in_context
            if include_function_name_in_context is not None
            else False
        )
        if noise is not None:
            return COMPOSERS[name](
                seed,
                grammar,
                functions_path=functions_path,
                noise=noise,
                include_function_name_in_context=include_fn
            )
        return COMPOSERS[name](
            seed,
            grammar,
            functions_path=functions_path,
            include_function_name_in_context=include_fn
        )
    if name == 'hybrid':
        if noise is not None:
            return COMPOSERS[name](seed, grammar, functions_path, noise)
        return COMPOSERS[name](seed, grammar, functions_path)
    if name == 'hierarchical':
        kwargs = {'functions_path': functions_path}
        if noise is not None:
            kwargs['noise'] = noise
        if skeleton_depth is not None:
            kwargs['skeleton_depth'] = skeleton_depth
        if ho_bias is not None:
            kwargs['ho_bias'] = ho_bias
        return COMPOSERS[name](seed, grammar, **kwargs)
    if name == 'template':
        if noise is not None:
            return COMPOSERS[name](seed, grammar, noise=noise)
        return COMPOSERS[name](seed, grammar)
    return COMPOSERS[name](seed, grammar)


def list_composers() -> list[str]:
    """Return a list of available composer names."""
    return list(COMPOSERS.keys())


__all__ = [
    # Composers
    'Composer',
    'RandomComposer',
    'TemplateComposer',
    'EmpiricalComposer',
    'HybridComposer',
    'HierarchicalComposer',
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
    # Guard
    'StrategyGuard',
    'ApplicationContext',
    'apply_guard',
    'get_default_guard',
    # Utilities
    'get_composer',
    'list_composers',
    'COMPOSERS',
]
