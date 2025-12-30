"""
Abstract Sampler Base Class

This module defines the abstract base class for all program samplers.
Samplers use Composers to generate batches of programs with different
strategies for ensuring diversity and uniqueness.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..composers.base import Composer
from ..ast_nodes import ASTNode
from ..type_utils import TypeType


class Sampler(ABC):
    """
    Abstract base class for program samplers.

    A sampler uses a Composer to generate batches of programs,
    handling aspects like uniqueness, depth variation, and seed management.
    """

    def __init__(self, composer: Composer):
        """
        Initialize the sampler.

        Args:
            composer: The composer to use for program generation
        """
        self.composer = composer

    @abstractmethod
    def sample(
        self,
        target_type: TypeType,
        n: int,
        depth: int,
        context: Optional[dict[str, TypeType]] = None
    ) -> list[ASTNode]:
        """
        Sample a batch of n programs.

        Args:
            target_type: The desired output type for all programs
            n: Number of programs to generate
            depth: Base depth for program generation
            context: Variable bindings in scope (name -> type)

        Returns:
            List of n AST nodes, each of the target type
        """
        pass
