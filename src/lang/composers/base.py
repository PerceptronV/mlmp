"""
Abstract Composer Base Class

This module defines the abstract base class for all program composers.
Composers generate well-typed programs according to a grammar.
"""

import random
from abc import ABC, abstractmethod
from typing import Optional

from ..grammar import Grammar
from ..ast_nodes import ASTNode, NumberNode, BooleanNode, ListNode, VariableNode, LambdaNode
from ..type_utils import (
    get_args,
    get_base_type,
    get_free_types,
    CallableOrig,
    TypeType,
    SubstitutionTable,
    substitute_type_vars,
    isvariable
)


class Composer(ABC):
    """
    Abstract base class for program composers.

    A composer generates well-typed programs according to a grammar.
    Subclasses implement different generation strategies.
    """

    def __init__(self, seed: int, grammar: Grammar):
        """
        Initialise the composer.

        Args:
            seed: Random seed for deterministic generation
            grammar: Grammar containing available functions
        """
        self.rng = random.Random(seed)
        self.var_counter = 0
        self.grammar = grammar

    def reset_var_counter(self):
        """Reset the variable counter for fresh variable names."""
        self.var_counter = 0

    def _fresh_var_name(self) -> str:
        """Generate a fresh variable name."""
        name = chr(ord('a') + (23 + self.var_counter) % 26)
        if self.var_counter >= 26:
            name += str(self.var_counter // 26)
        self.var_counter += 1
        return name

    def _sample_concrete_type(self) -> TypeType:
        """Sample a concrete type from the grammar's atomic return types."""
        atomic_types = list(self.grammar.atomic_return_types)
        if not atomic_types:
            return self.rng.choice([int, bool])
        return self.rng.choice(atomic_types)

    def _instantiate_free_types(
        self,
        type_: TypeType,
        substitutions: SubstitutionTable
    ) -> SubstitutionTable:
        """
        Instantiate any remaining free type variables by sampling from atomic types.

        Args:
            type_: The type to check for free variables
            substitutions: Current substitution table

        Returns:
            Updated substitution table with free variables instantiated
        """
        subs = substitutions.copy()
        free_types = get_free_types(type_, subs)

        for tvar in free_types:
            if isvariable(subs[tvar]):
                concrete = self._sample_concrete_type()
                subs[tvar] = concrete

        return subs

    def _sample_literal(self, type_: TypeType, substitutions: SubstitutionTable) -> ASTNode:
        """
        Sample a literal value of the given type.

        Args:
            type_: The desired type
            substitutions: Current type substitutions

        Returns:
            An AST node representing a literal value
        """
        actual_type = substitute_type_vars(type_, substitutions)
        base_type = get_base_type(actual_type)

        if actual_type == int:
            return NumberNode(self.rng.randint(0, 99))
        elif actual_type == bool:
            return BooleanNode(self.rng.choice([True, False]))
        elif base_type == list:
            # Empty list is a valid literal for any list type
            return ListNode([])
        else:
            raise ValueError(f"Cannot sample literal for type: {actual_type}")

    @abstractmethod
    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None
    ) -> ASTNode:
        """
        Generate a well-typed program.

        Args:
            target_type: The desired output type
            depth: Maximum depth of the expression tree
            context: Variable bindings in scope (name -> type)
            substitutions: Current type variable substitutions

        Returns:
            An AST node of the target type
        """
        pass

    def _has_function(self, name: str) -> bool:
        """
        Check if a function exists in the grammar.

        Args:
            name: Function name to check

        Returns:
            True if the function exists in the grammar
        """
        return name in self.grammar.names

    def _get_available_functions(self, names: set[str]) -> set[str]:
        """
        Get the subset of function names that exist in the grammar.

        Args:
            names: Set of function names to check

        Returns:
            Subset of names that exist in the grammar
        """
        return names & set(self.grammar.names)

    def _filter_weights_by_availability(
        self,
        weights: dict[str, float],
        requirements: dict[str, set[str]]
    ) -> dict[str, float]:
        """
        Filter a weight dictionary based on function availability.

        Args:
            weights: Dictionary mapping strategy names to weights
            requirements: Dictionary mapping strategy names to required functions.
                          A strategy is available if ALL its required functions exist.
                          Use empty set for strategies with no requirements.

        Returns:
            Filtered weights dictionary with unavailable strategies removed
        """
        available = {}
        for strategy, weight in weights.items():
            required = requirements.get(strategy, set())
            # Strategy is available if all required functions exist
            if all(self._has_function(fn) for fn in required):
                available[strategy] = weight
        return available

    def _renormalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """
        Renormalize weights to sum to 1.0.

        Args:
            weights: Dictionary of weights

        Returns:
            Normalized weights, or uniform if all weights are zero
        """
        if not weights:
            return {}
        total = sum(weights.values())
        if total <= 0:
            # All weights zero - return uniform
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}

    @classmethod
    def get_name(cls) -> str:
        """Return a short name for this composer."""
        return cls.__name__
