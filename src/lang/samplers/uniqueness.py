"""
Program Uniqueness Checking Utilities

This module provides utilities for checking program uniqueness using either:
1. String-based uniqueness: Programs are unique if their string representations differ
2. Behavioural uniqueness: Programs are unique if they produce different outputs on test inputs

These utilities are used for ensuring diversity across dataset splits (e.g., ensuring
training programs don't overlap with evaluation programs).
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set, Callable

from ..ast_nodes import ASTNode, pretty_print
from ..evaluator import Evaluator, EvaluationError
from ..environment import Closure
from ..grammar import Grammar


class UniquenessMode(Enum):
    """
    Mode for checking program uniqueness.

    This enum is the canonical definition used by both:
    - RuleSampler: for within-batch uniqueness during sampling
    - UniquenessChecker: for cross-split uniqueness checking
    """
    STRING = "string"           # Fast: program string uniqueness
    BEHAVIOURAL = "behavioural"   # Thorough: unique behaviour on held-out inputs


# Type alias for behaviour signatures
BehaviourSignature = tuple[tuple[int, ...], ...]


@dataclass
class UniquenessChecker:
    """
    Checks program uniqueness using string or behavioural comparison.

    This class maintains sets of seen programs and can check whether new
    programs are unique. It supports both string-based (fast) and behavioural
    (thorough) uniqueness checking.

    Attributes:
        mode: Whether to use STRING or behavioural uniqueness
        grammar: Grammar for creating evaluator (needed for behavioural mode)
        num_test_inputs: Number of test inputs for behavioural checking
        min_list_length: Minimum list length for test inputs
        max_list_length: Maximum list length for test inputs
        min_element_value: Minimum element value in test inputs
        max_element_value: Maximum element value in test inputs
    """
    mode: UniquenessMode
    grammar: Grammar
    num_test_inputs: int = 100
    min_list_length: int = 0
    max_list_length: int = 15
    min_element_value: int = 0
    max_element_value: int = 100

    # Internal state
    _seen_strings: Set[str] = field(default_factory=set)
    _seen_behaviours: Set[BehaviourSignature] = field(default_factory=set)
    _test_inputs: list[list[int]] = field(default_factory=list)
    _evaluator: Optional[Evaluator] = field(default=None)

    def __post_init__(self):
        """Initialise test inputs and evaluator."""
        self._generate_test_inputs()
        if self.mode == UniquenessMode.behavioural:
            self._evaluator = Evaluator(self.grammar)

    def _generate_test_inputs(self, seed: int = 42) -> None:
        """Generate fixed test inputs for behavioural uniqueness checking."""
        rng = random.Random(seed)
        self._test_inputs = []

        for _ in range(self.num_test_inputs):
            length = rng.randint(self.min_list_length, self.max_list_length)
            input_list = [
                rng.randint(self.min_element_value, self.max_element_value)
                for _ in range(length)
            ]
            self._test_inputs.append(input_list)

    def _execute_program(
        self,
        program: ASTNode,
        input_list: list[int]
    ) -> Optional[list[int]]:
        """
        Execute a program on an input list.

        Args:
            program: The program AST (should be a lambda)
            input_list: Input list

        Returns:
            Output list, or None if execution fails
        """
        if self._evaluator is None:
            self._evaluator = Evaluator(self.grammar)

        try:
            closure = self._evaluator.eval(program)

            if not isinstance(closure, Closure):
                return None

            env = closure.env.extend(closure.param[0], input_list)
            result = self._evaluator.eval(closure.body, env)

            if not isinstance(result, list):
                return None
            if not all(isinstance(x, int) for x in result):
                return None
            if len(result) > self.max_list_length:
                return None

            return result

        except (EvaluationError, NameError, TypeError, ZeroDivisionError):
            return None

    def compute_behaviour_signature(
        self,
        program: ASTNode
    ) -> Optional[BehaviourSignature]:
        """
        Compute a behavioural signature for a program.

        The signature is a tuple of outputs on the test inputs.
        Programs with the same signature behave identically.

        Args:
            program: The program AST

        Returns:
            Tuple of output tuples, or None if execution fails on all inputs
        """
        outputs: list[tuple[int, ...]] = []

        for input_list in self._test_inputs:
            output = self._execute_program(program, input_list)
            if output is None:
                # Use empty tuple as sentinel for failed executions
                outputs.append(tuple())
            else:
                outputs.append(tuple(output))

        return tuple(outputs)

    def is_unique(self, program: ASTNode, program_str: Optional[str] = None) -> bool:
        """
        Check if a program is unique (not seen before).

        Args:
            program: The program AST
            program_str: Optional pre-computed program string

        Returns:
            True if the program is unique, False if it's been seen before
        """
        if program_str is None:
            program_str = pretty_print(program, inline=True)

        if self.mode == UniquenessMode.STRING:
            return program_str not in self._seen_strings
        else:
            # behavioural mode
            signature = self.compute_behaviour_signature(program)
            if signature is None:
                return False  # Can't execute = not unique (or skip)
            return signature not in self._seen_behaviours

    def mark_seen(self, program: ASTNode, program_str: Optional[str] = None) -> None:
        """
        Mark a program as seen.

        Args:
            program: The program AST
            program_str: Optional pre-computed program string
        """
        if program_str is None:
            program_str = pretty_print(program, inline=True)

        self._seen_strings.add(program_str)

        if self.mode == UniquenessMode.behavioural:
            signature = self.compute_behaviour_signature(program)
            if signature is not None:
                self._seen_behaviours.add(signature)

    def mark_seen_string(self, program_str: str) -> None:
        """
        Mark a program string as seen (for string mode only).

        Args:
            program_str: The program string
        """
        self._seen_strings.add(program_str)

    def mark_seen_signature(self, signature: BehaviourSignature) -> None:
        """
        Mark a behavioural signature as seen.

        Args:
            signature: The behavioural signature
        """
        self._seen_behaviours.add(signature)

    def is_string_unique(self, program_str: str) -> bool:
        """
        Check if a program string is unique (fast check, always available).

        Args:
            program_str: The program string

        Returns:
            True if the string hasn't been seen before
        """
        return program_str not in self._seen_strings

    def clear(self) -> None:
        """Clear all seen programs."""
        self._seen_strings.clear()
        self._seen_behaviours.clear()

    @property
    def num_seen(self) -> int:
        """Return the number of unique programs seen."""
        if self.mode == UniquenessMode.STRING:
            return len(self._seen_strings)
        else:
            return len(self._seen_behaviours)

    @property
    def seen_strings(self) -> Set[str]:
        """Return the set of seen program strings."""
        return self._seen_strings.copy()

    @property
    def seen_behaviours(self) -> Set[BehaviourSignature]:
        """Return the set of seen behavioural signatures."""
        return self._seen_behaviours.copy()


def create_uniqueness_checker(
    mode: UniquenessMode,
    grammar: Grammar,
    num_test_inputs: int = 100
) -> UniquenessChecker:
    """
    Create a uniqueness checker with the specified mode.

    Args:
        mode: STRING for fast string-based checking, behavioural for thorough checking
        grammar: Grammar for creating evaluator
        num_test_inputs: Number of test inputs for behavioural checking

    Returns:
        Configured UniquenessChecker instance
    """
    return UniquenessChecker(
        mode=mode,
        grammar=grammar,
        num_test_inputs=num_test_inputs
    )
