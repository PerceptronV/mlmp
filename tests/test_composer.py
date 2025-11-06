"""
Unit tests for the random typed program sampler in data.composer.
"""

import sys
import os
import unittest

# Make the project src importable as a package root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import sample_program  # type: ignore
from lang.type_checker import TypeChecker  # type: ignore
from lang.type_system import INT, BOOL, list_of, func, Type, TypeVar, ListType, FunctionType  # type: ignore
from lang.ast_nodes import NumberNode, BooleanNode, ListNode, LambdaNode, pretty_print  # type: ignore
from lang.evaluator import Evaluator  # type: ignore


def is_type_compatible(inferred: Type, expected: Type) -> bool:
    """
    Check if inferred type is compatible with (can be instantiated as) expected type.

    In Hindley-Milner, a more general type (with type variables) can be instantiated
    to a more specific type. For example:
    - t0 → Int is compatible with Int → Int (t0 can be instantiated to Int)
    - [t0] is compatible with [Int] (t0 can be instantiated to Int)
    - t0 → t0 is compatible with Int → Int (t0 can be instantiated to Int)
    """
    # If types are equal, they're compatible
    if inferred == expected:
        return True

    # TypeVar in inferred can be instantiated to match expected
    if isinstance(inferred, TypeVar):
        return True  # Type variable can be inst antiated to any type

    # Lists: check element types
    if isinstance(inferred, ListType) and isinstance(expected, ListType):
        return is_type_compatible(inferred.elem_type, expected.elem_type)

    # Functions: check parameter and return types
    if isinstance(inferred, FunctionType) and isinstance(expected, FunctionType):
        # Parameters are contravariant, returns are covariant in subtyping
        # But for type variable instantiation, we just check compatibility
        return (is_type_compatible(inferred.param_type, expected.param_type) and
                is_type_compatible(inferred.return_type, expected.return_type))

    return False


class TestComposerSampler(unittest.TestCase):
    """Tests for the typed random program generator."""

    def test_generates_well_typed_programs(self):
        checker = TypeChecker()
        # Multiple seeds and depths
        for seed in range(5):
            for depth in range(4):
                ast = sample_program(seed=seed, max_depth=depth)
                # Should type-check without raising
                t = checker.check_program(ast)
                self.assertIsNotNone(t)

    def test_respects_target_types(self):
        checker = TypeChecker()
        targets = [
            INT,
            BOOL,
            list_of(INT),
            func(INT, INT),
        ]
        for seed in range(3):
            for target in targets:
                ast = sample_program(seed=seed, max_depth=4, target_type=target)
                t = checker.check_program(ast)
                # Allow more general types (e.g., t0 → Int is valid for Int → Int)
                self.assertTrue(
                    is_type_compatible(t, target),
                    f"Incompatible type for seed={seed}, target={target}:\n"
                    f"  Inferred: {t}\n"
                    f"  Expected: {target}\n"
                    f"  AST: {pretty_print(ast)}"
                )

    def test_determinism_with_seed(self):
        # Same seed, depth, and target should yield identical AST repr
        seed = 123
        depth = 4
        ast1 = sample_program(seed=seed, max_depth=depth)
        ast2 = sample_program(seed=seed, max_depth=depth)
        self.assertEqual(repr(ast1), repr(ast2))

        # With explicit target
        from lang.type_system import list_of
        target = list_of(INT)
        ast3 = sample_program(seed=seed, max_depth=depth, target_type=target)
        ast4 = sample_program(seed=seed, max_depth=depth, target_type=target)
        self.assertEqual(repr(ast3), repr(ast4))

    def test_evaluates_without_runtime_errors(self):
        evaluator = Evaluator()
        # Evaluate a variety of generated programs. For functions, evaluation returns a closure.
        seeds = [0, 1, 2]
        depths = [1, 2, 3]
        targets = [None, INT, BOOL, list_of(INT), func(INT, INT)]
        for seed in seeds:
            for depth in depths:
                for target in targets:
                    ast = sample_program(seed=seed, max_depth=depth, target_type=target)
                    try:
                        _ = evaluator.eval(ast)
                    except Exception as e:
                        self.fail(f"Evaluation failed for seed={seed}, depth={depth}, target={target}: {e}\nAST:\n{pretty_print(ast)}")

    def test_depth_zero_base_cases(self):
        # Depth 0 should produce simple literals or minimal structures
        ast_int = sample_program(seed=42, max_depth=0, target_type=INT)
        self.assertIsInstance(ast_int, NumberNode)

        ast_bool = sample_program(seed=42, max_depth=0, target_type=BOOL)
        self.assertIsInstance(ast_bool, BooleanNode)

        ast_list = sample_program(seed=42, max_depth=0, target_type=list_of(INT))
        self.assertIsInstance(ast_list, ListNode)
        self.assertEqual(len(ast_list.elements), 0)

        ast_fn = sample_program(seed=42, max_depth=0, target_type=func(BOOL, INT))
        self.assertIsInstance(ast_fn, LambdaNode)


if __name__ == "__main__":
    unittest.main()


