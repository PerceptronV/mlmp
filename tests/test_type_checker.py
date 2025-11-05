"""
Unit tests for the type checker.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'lang'))

import unittest
from type_checker import TypeChecker, type_check
from type_system import (
    Type, TypeVar, INT, BOOL, list_of, func, TypeError
)
from parser import parse


class TestBasicTypes(unittest.TestCase):
    """Test type checking of basic values."""
    
    def test_number(self):
        """Test number has type Int."""
        type_ = type_check("42")
        self.assertEqual(type_, INT)
    
    def test_boolean_true(self):
        """Test true has type Bool."""
        type_ = type_check("true")
        self.assertEqual(type_, BOOL)
    
    def test_boolean_false(self):
        """Test false has type Bool."""
        type_ = type_check("false")
        self.assertEqual(type_, BOOL)
    
    def test_empty_list(self):
        """Test empty list has polymorphic type."""
        type_ = type_check("[]")
        self.assertIsInstance(type_, type(list_of(TypeVar("t0"))))


class TestListTypes(unittest.TestCase):
    """Test type checking of lists."""
    
    def test_int_list(self):
        """Test list of integers."""
        type_ = type_check("[1 2 3]")
        self.assertEqual(type_, list_of(INT))
    
    def test_bool_list(self):
        """Test list of booleans."""
        type_ = type_check("[true false true]")
        self.assertEqual(type_, list_of(BOOL))
    
    def test_nested_list(self):
        """Test nested lists."""
        type_ = type_check("[[1 2] [3 4]]")
        self.assertEqual(type_, list_of(list_of(INT)))
    
    def test_heterogeneous_list_error(self):
        """Test that mixed-type lists are rejected."""
        with self.assertRaises(TypeError) as ctx:
            type_check("[1 true 3]")
        self.assertIn("incompatible types", str(ctx.exception))


class TestLambdaTypes(unittest.TestCase):
    """Test type checking of lambda expressions."""
    
    def test_identity(self):
        """Test identity function has polymorphic type."""
        type_ = type_check("(λ x x)")
        self.assertIsInstance(type_, type(func(TypeVar("t0"), TypeVar("t1"))))
    
    def test_constant(self):
        """Test constant function."""
        type_ = type_check("(λ x 42)")
        # Should be t0 → Int
        self.assertIsInstance(type_, type(func(TypeVar("t0"), INT)))
    
    def test_increment(self):
        """Test increment function."""
        type_ = type_check("(λ x (+ x 1))")
        self.assertEqual(type_, func(INT, INT))
    
    def test_nested_lambda(self):
        """Test nested lambda (curried function)."""
        type_ = type_check("(λ x (λ y (+ x y)))")
        self.assertEqual(type_, func(INT, func(INT, INT)))


class TestApplicationTypes(unittest.TestCase):
    """Test type checking of function applications."""
    
    def test_simple_application(self):
        """Test simple function application."""
        type_ = type_check("((λ x x) 42)")
        self.assertEqual(type_, INT)
    
    def test_arithmetic(self):
        """Test arithmetic operations."""
        self.assertEqual(type_check("(+ 1 2)"), INT)
        self.assertEqual(type_check("(- 5 3)"), INT)
        self.assertEqual(type_check("(* 4 5)"), INT)
    
    def test_comparison(self):
        """Test comparison operations."""
        self.assertEqual(type_check("(< 1 2)"), BOOL)
        self.assertEqual(type_check("(> 5 3)"), BOOL)
        self.assertEqual(type_check("(== 1 1)"), BOOL)
    
    def test_type_error_in_application(self):
        """Test type error in function application."""
        with self.assertRaises(TypeError) as ctx:
            type_check("(+ true 2)")
        self.assertIn("Type mismatch", str(ctx.exception))


class TestConditionalTypes(unittest.TestCase):
    """Test type checking of conditional expressions."""
    
    def test_if_with_int(self):
        """Test if expression returning Int."""
        type_ = type_check("(if true 1 2)")
        self.assertEqual(type_, INT)
    
    def test_if_with_bool(self):
        """Test if expression returning Bool."""
        type_ = type_check("(if true true false)")
        self.assertEqual(type_, BOOL)
    
    def test_if_condition_type_error(self):
        """Test that non-boolean condition is rejected."""
        with self.assertRaises(TypeError) as ctx:
            type_check("(if 1 2 3)")
        self.assertIn("condition must have type Bool", str(ctx.exception))
    
    def test_if_branch_type_error(self):
        """Test that branches with different types are rejected."""
        with self.assertRaises(TypeError) as ctx:
            type_check("(if true 1 false)")
        self.assertIn("branches must have the same type", str(ctx.exception))


class TestListOperations(unittest.TestCase):
    """Test type checking of list operations."""
    
    def test_first(self):
        """Test first returns element type."""
        type_ = type_check("(first [1 2 3])")
        self.assertEqual(type_, INT)
    
    def test_cons(self):
        """Test cons returns list."""
        type_ = type_check("(cons 1 [2 3])")
        self.assertEqual(type_, list_of(INT))
    
    def test_append(self):
        """Test append returns list."""
        type_ = type_check("(append [1 2] 3)")
        self.assertEqual(type_, list_of(INT))
    
    def test_concat(self):
        """Test concat returns list."""
        type_ = type_check("(concat [1 2] [3 4])")
        self.assertEqual(type_, list_of(INT))
    
    def test_reverse(self):
        """Test reverse returns list of same type."""
        type_ = type_check("(reverse [1 2 3])")
        self.assertEqual(type_, list_of(INT))
    
    def test_take(self):
        """Test take returns list of same type."""
        type_ = type_check("(take 2 [1 2 3 4])")
        self.assertEqual(type_, list_of(INT))


class TestHigherOrderFunctions(unittest.TestCase):
    """Test type checking of higher-order functions."""
    
    def test_map(self):
        """Test map with lambda."""
        type_ = type_check("(map (λ x (* x 2)) [1 2 3])")
        self.assertEqual(type_, list_of(INT))
    
    def test_map_type_change(self):
        """Test map can change element type."""
        type_ = type_check("(map (λ x (> x 5)) [1 2 3])")
        self.assertEqual(type_, list_of(BOOL))
    
    def test_filter(self):
        """Test filter preserves element type."""
        type_ = type_check("(filter (λ x (> x 2)) [1 2 3 4])")
        self.assertEqual(type_, list_of(INT))
    
    def test_fold(self):
        """Test fold returns accumulator type."""
        type_ = type_check("(fold (λ a (λ x (+ a x))) 0 [1 2 3])")
        self.assertEqual(type_, INT)
    
    def test_map_type_error(self):
        """Test map with wrong function type."""
        with self.assertRaises(TypeError):
            type_check("(map (λ x x) [1 2 3])")  # This should actually work
            # Better example:
            type_check("(map 42 [1 2 3])")  # Not a function


class TestPolymorphicFunctions(unittest.TestCase):
    """Test polymorphic function types."""
    
    def test_identity_with_int(self):
        """Test identity applied to int."""
        type_ = type_check("((λ x x) 42)")
        self.assertEqual(type_, INT)
    
    def test_identity_with_bool(self):
        """Test identity applied to bool."""
        type_ = type_check("((λ x x) true)")
        self.assertEqual(type_, BOOL)
    
    def test_identity_with_list(self):
        """Test identity applied to list."""
        type_ = type_check("((λ x x) [1 2 3])")
        self.assertEqual(type_, list_of(INT))
    
    def test_const_polymorphic(self):
        """Test constant function is polymorphic in input."""
        # (λ x 42) can take any input
        type_ = type_check("((λ x 42) true)")
        self.assertEqual(type_, INT)


class TestComplexTypes(unittest.TestCase):
    """Test complex type checking scenarios."""
    
    def test_nested_application(self):
        """Test nested function application."""
        type_ = type_check("(((λ x (λ y (+ x y))) 3) 4)")
        self.assertEqual(type_, INT)
    
    def test_composition(self):
        """Test function composition."""
        code = "(λ f (λ g (λ x (f (g x)))))"
        type_ = type_check(code)
        # Should be (t1 → t2) → (t3 → t1) → t3 → t2
        self.assertIsInstance(type_, type(func(TypeVar("t0"), TypeVar("t1"))))
    
    def test_map_with_nested_lambda(self):
        """Test map with nested lambda."""
        type_ = type_check("(map (λ x (λ y (+ x y))) [1 2 3])")
        # Should be [Int → Int]
        self.assertEqual(type_, list_of(func(INT, INT)))
    
    def test_filter_and_map(self):
        """Test composing filter and map."""
        code = "(map (λ x (* x 2)) (filter (λ x (> x 2)) [1 2 3 4]))"
        type_ = type_check(code)
        self.assertEqual(type_, list_of(INT))


class TestErrorMessages(unittest.TestCase):
    """Test that error messages are meaningful."""
    
    def test_undefined_variable_error(self):
        """Test undefined variable error message."""
        with self.assertRaises(TypeError) as ctx:
            type_check("x")
        self.assertIn("Undefined variable", str(ctx.exception))
        self.assertIn("'x'", str(ctx.exception))
    
    def test_type_mismatch_error(self):
        """Test type mismatch error message."""
        with self.assertRaises(TypeError) as ctx:
            type_check("(+ 1 true)")
        error_msg = str(ctx.exception)
        self.assertIn("Type mismatch", error_msg)
    
    def test_list_type_error(self):
        """Test list with mixed types error."""
        with self.assertRaises(TypeError) as ctx:
            type_check("[1 true 3]")
        self.assertIn("incompatible types", str(ctx.exception))
    
    def test_if_condition_error(self):
        """Test if with non-boolean condition."""
        with self.assertRaises(TypeError) as ctx:
            type_check("(if 42 1 2)")
        self.assertIn("Bool", str(ctx.exception))
    
    def test_application_error(self):
        """Test applying non-function."""
        with self.assertRaises(TypeError) as ctx:
            type_check("(42 1)")
        self.assertIn("non-function", str(ctx.exception))


class TestBuiltInFunctions(unittest.TestCase):
    """Test built-in function types."""
    
    def test_arithmetic_operators(self):
        """Test arithmetic operator types."""
        for op in ["+", "-", "*", "/", "%"]:
            type_ = type_check(f"({op} 1 2)")
            self.assertEqual(type_, INT)
    
    def test_comparison_operators(self):
        """Test comparison operator types."""
        for op in ["<", ">", "=="]:
            type_ = type_check(f"({op} 1 2)")
            self.assertEqual(type_, BOOL)
    
    def test_boolean_operators(self):
        """Test boolean operator types."""
        self.assertEqual(type_check("(and true false)"), BOOL)
        self.assertEqual(type_check("(or true false)"), BOOL)
        self.assertEqual(type_check("(not true)"), BOOL)
    
    def test_list_functions(self):
        """Test list function types."""
        self.assertEqual(type_check("(length [1 2 3])"), INT)
        self.assertEqual(type_check("(sum [1 2 3])"), INT)
        self.assertEqual(type_check("(product [2 3 4])"), INT)
        self.assertEqual(type_check("(max [1 5 3])"), INT)
        self.assertEqual(type_check("(min [1 5 3])"), INT)


class TestExamplesFromSpec(unittest.TestCase):
    """Test examples from the language specification."""
    
    def test_remove_all_but_element_1(self):
        """Test: (λ x (take 1 x))"""
        type_ = type_check("(λ x (take 1 x))")
        # Should be [t] → [t]
        self.assertIsInstance(type_, type(func(list_of(TypeVar("t0")), list_of(TypeVar("t1")))))
    
    def test_remove_last_element(self):
        """Test: (λ x (droplast 1 x))"""
        type_ = type_check("(λ x (droplast 1 x))")
        # Should be [t] → [t]
        self.assertIsInstance(type_, type(func(list_of(TypeVar("t0")), list_of(TypeVar("t1")))))
    
    def test_singleton_first(self):
        """Test: (λ x (singleton (first x)))"""
        type_ = type_check("(λ x (singleton (first x)))")
        # Should be [t] → [t]
        self.assertIsInstance(type_, type(func(list_of(TypeVar("t0")), list_of(TypeVar("t1")))))


if __name__ == "__main__":
    unittest.main()

