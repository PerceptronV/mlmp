"""
Unit tests for the evaluator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from evaluator import Evaluator, evaluate, EvaluationError
from parser import parse


class TestBasicValues(unittest.TestCase):
    """Test evaluation of basic values."""
    
    def test_number(self):
        """Test number evaluation."""
        self.assertEqual(evaluate("42"), 42)
        self.assertEqual(evaluate("0"), 0)
        self.assertEqual(evaluate("99"), 99)
    
    def test_boolean(self):
        """Test boolean evaluation."""
        self.assertEqual(evaluate("true"), True)
        self.assertEqual(evaluate("false"), False)
    
    def test_list(self):
        """Test list evaluation."""
        self.assertEqual(evaluate("[]"), [])
        self.assertEqual(evaluate("[1 2 3]"), [1, 2, 3])
        self.assertEqual(evaluate("[true false]"), [True, False])


class TestArithmetic(unittest.TestCase):
    """Test arithmetic operations."""
    
    def test_addition(self):
        """Test addition."""
        self.assertEqual(evaluate("(+ 1 2)"), 3)
        self.assertEqual(evaluate("(+ 0 5)"), 5)
        self.assertEqual(evaluate("(+ 10 20)"), 30)
    
    def test_subtraction(self):
        """Test subtraction."""
        self.assertEqual(evaluate("(- 5 3)"), 2)
        self.assertEqual(evaluate("(- 10 10)"), 0)
    
    def test_multiplication(self):
        """Test multiplication."""
        self.assertEqual(evaluate("(* 2 3)"), 6)
        self.assertEqual(evaluate("(* 0 5)"), 0)
        self.assertEqual(evaluate("(* 7 8)"), 56)
    
    def test_division(self):
        """Test integer division."""
        self.assertEqual(evaluate("(/ 10 2)"), 5)
        self.assertEqual(evaluate("(/ 7 3)"), 2)
        self.assertEqual(evaluate("(/ 10 3)"), 3)
    
    def test_division_by_zero(self):
        """Test division by zero raises error."""
        with self.assertRaises(EvaluationError):
            evaluate("(/ 5 0)")
    
    def test_modulo(self):
        """Test modulo operation."""
        self.assertEqual(evaluate("(% 10 3)"), 1)
        self.assertEqual(evaluate("(% 7 2)"), 1)
        self.assertEqual(evaluate("(% 10 5)"), 0)
    
    def test_nested_arithmetic(self):
        """Test nested arithmetic."""
        self.assertEqual(evaluate("(+ (* 2 3) (/ 10 2))"), 11)
        self.assertEqual(evaluate("(* (+ 1 2) (- 5 2))"), 9)


class TestComparison(unittest.TestCase):
    """Test comparison operations."""
    
    def test_less_than(self):
        """Test less than."""
        self.assertEqual(evaluate("(< 1 2)"), True)
        self.assertEqual(evaluate("(< 2 1)"), False)
        self.assertEqual(evaluate("(< 5 5)"), False)
    
    def test_greater_than(self):
        """Test greater than."""
        self.assertEqual(evaluate("(> 2 1)"), True)
        self.assertEqual(evaluate("(> 1 2)"), False)
        self.assertEqual(evaluate("(> 5 5)"), False)
    
    def test_equality(self):
        """Test equality."""
        self.assertEqual(evaluate("(== 1 1)"), True)
        self.assertEqual(evaluate("(== 1 2)"), False)
        self.assertEqual(evaluate("(== true true)"), True)
        self.assertEqual(evaluate("(== [1 2] [1 2])"), True)
        self.assertEqual(evaluate("(== [1 2] [2 1])"), False)


class TestBoolean(unittest.TestCase):
    """Test boolean operations."""
    
    def test_and(self):
        """Test boolean AND."""
        self.assertEqual(evaluate("(and true true)"), True)
        self.assertEqual(evaluate("(and true false)"), False)
        self.assertEqual(evaluate("(and false true)"), False)
        self.assertEqual(evaluate("(and false false)"), False)
    
    def test_or(self):
        """Test boolean OR."""
        self.assertEqual(evaluate("(or true true)"), True)
        self.assertEqual(evaluate("(or true false)"), True)
        self.assertEqual(evaluate("(or false true)"), True)
        self.assertEqual(evaluate("(or false false)"), False)
    
    def test_not(self):
        """Test boolean NOT."""
        self.assertEqual(evaluate("(not true)"), False)
        self.assertEqual(evaluate("(not false)"), True)


class TestConditional(unittest.TestCase):
    """Test conditional expressions."""
    
    def test_if_true(self):
        """Test if with true condition."""
        self.assertEqual(evaluate("(if true 1 2)"), 1)
        self.assertEqual(evaluate("(if (< 1 2) 10 20)"), 10)
    
    def test_if_false(self):
        """Test if with false condition."""
        self.assertEqual(evaluate("(if false 1 2)"), 2)
        self.assertEqual(evaluate("(if (> 1 2) 10 20)"), 20)
    
    def test_nested_if(self):
        """Test nested if expressions."""
        result = evaluate("(if true (if false 1 2) 3)")
        self.assertEqual(result, 2)


class TestLambda(unittest.TestCase):
    """Test lambda expressions."""
    
    def test_identity(self):
        """Test identity function."""
        self.assertEqual(evaluate("((λ x x) 42)"), 42)
        self.assertEqual(evaluate("((λ x x) true)"), True)
    
    def test_constant(self):
        """Test constant function."""
        self.assertEqual(evaluate("((λ x 42) 100)"), 42)
    
    def test_increment(self):
        """Test increment function."""
        self.assertEqual(evaluate("((λ x (+ x 1)) 10)"), 11)
    
    def test_nested_lambda(self):
        """Test nested lambdas (currying)."""
        result = evaluate("(((λ x (λ y (+ x y))) 3) 4)")
        self.assertEqual(result, 7)
    
    def test_closure(self):
        """Test closure captures environment."""
        code = "(((λ x (λ y (+ x y))) 10) 5)"
        self.assertEqual(evaluate(code), 15)


class TestListOperations(unittest.TestCase):
    """Test list operations."""
    
    def test_first(self):
        """Test first element."""
        self.assertEqual(evaluate("(first [1 2 3])"), 1)
    
    def test_second(self):
        """Test second element."""
        self.assertEqual(evaluate("(second [1 2 3])"), 2)
    
    def test_third(self):
        """Test third element."""
        self.assertEqual(evaluate("(third [1 2 3])"), 3)
    
    def test_last(self):
        """Test last element."""
        self.assertEqual(evaluate("(last [1 2 3])"), 3)
        self.assertEqual(evaluate("(last [5])"), 5)
    
    def test_nth(self):
        """Test nth element."""
        self.assertEqual(evaluate("(nth 0 [1 2 3])"), 1)
        self.assertEqual(evaluate("(nth 2 [1 2 3])"), 3)
    
    def test_cons(self):
        """Test cons (prepend)."""
        self.assertEqual(evaluate("(cons 1 [2 3])"), [1, 2, 3])
        self.assertEqual(evaluate("(cons 0 [])"), [0])
    
    def test_append(self):
        """Test append."""
        self.assertEqual(evaluate("(append [1 2] 3)"), [1, 2, 3])
    
    def test_concat(self):
        """Test concatenate."""
        self.assertEqual(evaluate("(concat [1 2] [3 4])"), [1, 2, 3, 4])
        self.assertEqual(evaluate("(concat [] [1 2])"), [1, 2])
    
    def test_length(self):
        """Test length."""
        self.assertEqual(evaluate("(length [])"), 0)
        self.assertEqual(evaluate("(length [1 2 3])"), 3)
    
    def test_reverse(self):
        """Test reverse."""
        self.assertEqual(evaluate("(reverse [1 2 3])"), [3, 2, 1])
        self.assertEqual(evaluate("(reverse [])"), [])
    
    def test_take(self):
        """Test take."""
        self.assertEqual(evaluate("(take 2 [1 2 3 4])"), [1, 2])
        self.assertEqual(evaluate("(take 0 [1 2 3])"), [])
    
    def test_drop(self):
        """Test drop."""
        self.assertEqual(evaluate("(drop 2 [1 2 3 4])"), [3, 4])
        self.assertEqual(evaluate("(drop 0 [1 2 3])"), [1, 2, 3])
    
    def test_droplast(self):
        """Test droplast."""
        self.assertEqual(evaluate("(droplast 1 [1 2 3])"), [1, 2])
        self.assertEqual(evaluate("(droplast 2 [1 2 3])"), [1])


class TestListConstruction(unittest.TestCase):
    """Test list construction functions."""
    
    def test_singleton(self):
        """Test singleton."""
        self.assertEqual(evaluate("(singleton 5)"), [5])
        self.assertEqual(evaluate("(singleton true)"), [True])
    
    def test_repeat(self):
        """Test repeat."""
        self.assertEqual(evaluate("(repeat 3 4)"), [3, 3, 3, 3])
        self.assertEqual(evaluate("(repeat 1 0)"), [])
    
    def test_range(self):
        """Test range."""
        self.assertEqual(evaluate("(range 1 5 1)"), [1, 2, 3, 4, 5])
        self.assertEqual(evaluate("(range 0 10 2)"), [0, 2, 4, 6, 8, 10])
        self.assertEqual(evaluate("(range 5 5 1)"), [5])


class TestListModification(unittest.TestCase):
    """Test list modification functions."""
    
    def test_insert(self):
        """Test insert."""
        self.assertEqual(evaluate("(insert 5 1 [1 2 3])"), [1, 5, 2, 3])
        self.assertEqual(evaluate("(insert 0 0 [1 2])"), [0, 1, 2])
    
    def test_replace(self):
        """Test replace."""
        self.assertEqual(evaluate("(replace 1 5 [1 2 3])"), [1, 5, 3])
    
    def test_swap(self):
        """Test swap."""
        self.assertEqual(evaluate("(swap 0 2 [1 2 3])"), [3, 2, 1])
    
    def test_cut_idx(self):
        """Test cut_idx."""
        self.assertEqual(evaluate("(cut_idx 1 [1 2 3])"), [1, 3])
    
    def test_cut_val(self):
        """Test cut_val (remove first occurrence)."""
        self.assertEqual(evaluate("(cut_val 2 [1 2 3 2])"), [1, 3, 2])
    
    def test_cut_vals(self):
        """Test cut_vals (remove all occurrences)."""
        self.assertEqual(evaluate("(cut_vals 2 [1 2 3 2])"), [1, 3])


class TestListSlicing(unittest.TestCase):
    """Test list slicing functions."""
    
    def test_slice(self):
        """Test slice."""
        self.assertEqual(evaluate("(slice 1 3 [1 2 3 4])"), [2, 3])
    
    def test_takelast(self):
        """Test takelast."""
        self.assertEqual(evaluate("(takelast 2 [1 2 3 4])"), [3, 4])
    
    def test_cut_slice(self):
        """Test cut_slice."""
        self.assertEqual(evaluate("(cut_slice 1 3 [1 2 3 4])"), [1, 4])
    
    def test_splice(self):
        """Test splice."""
        self.assertEqual(evaluate("(splice [5 6] 1 [1 2 3])"), [1, 5, 6, 2, 3])


class TestHigherOrderFunctions(unittest.TestCase):
    """Test higher-order functions."""
    
    def test_map(self):
        """Test map."""
        result = evaluate("(map (λ x (* x 2)) [1 2 3])")
        self.assertEqual(result, [2, 4, 6])
    
    def test_map_empty(self):
        """Test map on empty list."""
        result = evaluate("(map (λ x (* x 2)) [])")
        self.assertEqual(result, [])
    
    def test_filter(self):
        """Test filter."""
        result = evaluate("(filter (λ x (> x 2)) [1 2 3 4])")
        self.assertEqual(result, [3, 4])
    
    def test_filter_all_pass(self):
        """Test filter where all pass."""
        result = evaluate("(filter (λ x (> x 0)) [1 2 3])")
        self.assertEqual(result, [1, 2, 3])
    
    def test_filter_none_pass(self):
        """Test filter where none pass."""
        result = evaluate("(filter (λ x (< x 0)) [1 2 3])")
        self.assertEqual(result, [])
    
    def test_fold(self):
        """Test fold (reduce)."""
        result = evaluate("(fold (λ a (λ x (+ a x))) 0 [1 2 3])")
        self.assertEqual(result, 6)
    
    def test_fold_product(self):
        """Test fold for product."""
        result = evaluate("(fold (λ a (λ x (* a x))) 1 [2 3 4])")
        self.assertEqual(result, 24)
    
    def test_mapi(self):
        """Test mapi (map with index)."""
        # mapi with lambda that takes element then index
        result = evaluate("(mapi (λ x (λ i (+ x i))) [10 20 30])")
        self.assertEqual(result, [10, 21, 32])
    
    def test_filteri(self):
        """Test filteri (filter with index)."""
        # Keep elements at even indices
        result = evaluate("(filteri (λ i (λ x (is_even i))) [10 20 30 40])")
        self.assertEqual(result, [10, 30])
    
    def test_foldi(self):
        """Test foldi (fold with index)."""
        # Sum of indices
        result = evaluate("(foldi (λ a (λ x (λ i (+ a i)))) 0 [10 20 30])")
        self.assertEqual(result, 3)  # 0 + 0 + 1 + 2


class TestListAggregation(unittest.TestCase):
    """Test list aggregation functions."""
    
    def test_sum(self):
        """Test sum."""
        self.assertEqual(evaluate("(sum [1 2 3])"), 6)
        self.assertEqual(evaluate("(sum [])"), 0)
    
    def test_product(self):
        """Test product."""
        self.assertEqual(evaluate("(product [2 3 4])"), 24)
        self.assertEqual(evaluate("(product [])"), 1)
    
    def test_max(self):
        """Test max."""
        self.assertEqual(evaluate("(max [1 5 3])"), 5)
    
    def test_min(self):
        """Test min."""
        self.assertEqual(evaluate("(min [5 1 3])"), 1)


class TestListTransformation(unittest.TestCase):
    """Test list transformation functions."""
    
    def test_unique(self):
        """Test unique."""
        self.assertEqual(evaluate("(unique [1 2 1 3 2])"), [1, 2, 3])
    
    def test_sort(self):
        """Test sort."""
        result = evaluate("(sort (λ x x) [3 1 2])")
        self.assertEqual(result, [1, 2, 3])
    
    def test_flatten(self):
        """Test flatten."""
        self.assertEqual(evaluate("(flatten [[1 2] [3 4]])"), [1, 2, 3, 4])
        self.assertEqual(evaluate("(flatten [[1] [] [2 3]])"), [1, 2, 3])
    
    def test_zip(self):
        """Test zip."""
        self.assertEqual(evaluate("(zip [1 2] [3 4])"), [[1, 3], [2, 4]])
    
    def test_group(self):
        """Test group."""
        # Group by modulo 2
        result = evaluate("(group (λ x (% x 2)) [1 2 3 4])")
        # Should have 2 groups: evens and odds
        self.assertEqual(len(result), 2)


class TestListPredicates(unittest.TestCase):
    """Test list query functions."""
    
    def test_is_in(self):
        """Test is_in."""
        self.assertEqual(evaluate("(is_in [1 2 3] 2)"), True)
        self.assertEqual(evaluate("(is_in [1 2 3] 5)"), False)
    
    def test_count(self):
        """Test count."""
        result = evaluate("(count (λ x (> x 2)) [1 2 3 4])")
        self.assertEqual(result, 2)
    
    def test_find(self):
        """Test find."""
        result = evaluate("(find (λ x (> x 2)) [1 2 3 4])")
        self.assertEqual(result, [2, 3])


class TestNumberPredicates(unittest.TestCase):
    """Test number predicate functions."""
    
    def test_is_even(self):
        """Test is_even."""
        self.assertEqual(evaluate("(is_even 2)"), True)
        self.assertEqual(evaluate("(is_even 3)"), False)
        self.assertEqual(evaluate("(is_even 0)"), True)
    
    def test_is_odd(self):
        """Test is_odd."""
        self.assertEqual(evaluate("(is_odd 3)"), True)
        self.assertEqual(evaluate("(is_odd 2)"), False)
        self.assertEqual(evaluate("(is_odd 0)"), False)


class TestExamplesFromSpec(unittest.TestCase):
    """Test examples from the language specification."""
    
    def test_remove_all_but_element_1(self):
        """Test: (λ x (take 1 x))"""
        result = evaluate("((λ x (take 1 x)) [6 4 7 9])")
        self.assertEqual(result, [6])
    
    def test_remove_all_but_first(self):
        """Test: (λ x (singleton (first x)))"""
        result = evaluate("((λ x (singleton (first x))) [74 1 93 44 5])")
        self.assertEqual(result, [74])
    
    def test_remove_last(self):
        """Test: (λ x (droplast 1 x))"""
        result = evaluate("((λ x (droplast 1 x)) [74 12 59 87 7])")
        self.assertEqual(result, [74, 12, 59, 87])


class TestComplexPrograms(unittest.TestCase):
    """Test complex program examples."""
    
    def test_sum_of_squares(self):
        """Test sum of squares using map and fold."""
        code = "(fold (λ a (λ x (+ a x))) 0 (map (λ x (* x x)) [1 2 3]))"
        result = evaluate(code)
        self.assertEqual(result, 14)  # 1 + 4 + 9
    
    def test_filter_and_map(self):
        """Test combining filter and map."""
        code = "(map (λ x (* x 2)) (filter (λ x (> x 2)) [1 2 3 4]))"
        result = evaluate(code)
        self.assertEqual(result, [6, 8])
    
    def test_nested_list_operations(self):
        """Test nested list operations."""
        code = "(flatten (map (λ x (singleton x)) [1 2 3]))"
        result = evaluate(code)
        self.assertEqual(result, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

