"""
Sampler Module

This module provides different strategies for sampling batches of programs
using Composers. Samplers handle uniqueness, depth variation, and seed
management to produce diverse program batches.

Available samplers:
- DefaultSampler: Changes seeds, varies depth, ensures uniqueness
- RuleSampler: Generates list-to-list functions with I/O pairs following
               Josh Rule's meta-program learner methodology
"""

from typing import Type

from .base import Sampler
from .default import DefaultSampler
from .rule import RuleSampler, SampledProgram, UniquenessMode, create_list_to_list_type

# Registry of available samplers
SAMPLERS: dict[str, Type[Sampler]] = {
    'default': DefaultSampler,
    'rule': RuleSampler,
}


def get_sampler(name: str) -> Type[Sampler]:
    """
    Get a sampler class by name.

    Args:
        name: Name of the sampler ('default', etc.)

    Returns:
        A sampler class

    Raises:
        ValueError: If the sampler name is not recognised
    """
    if name not in SAMPLERS:
        available = ', '.join(SAMPLERS.keys())
        raise ValueError(f"Unknown sampler: {name}. Available: {available}")

    return SAMPLERS[name]


def list_samplers() -> list[str]:
    """Return a list of available sampler names."""
    return list(SAMPLERS.keys())


__all__ = [
    'Sampler',
    'DefaultSampler',
    'RuleSampler',
    'SampledProgram',
    'UniquenessMode',
    'create_list_to_list_type',
    'get_sampler',
    'list_samplers',
    'SAMPLERS',
]
