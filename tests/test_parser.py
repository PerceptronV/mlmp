"""
Unit tests for the parser and AST generator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
from parser import Parser, parse, ParseError
from ast_nodes import (
    NumberNode, BooleanNode, VariableNode, LambdaNode,
    ApplicationNode, ListNode, IfNode
)


class TestBasicExpressions(unittest.TestCase):
    """Test parsing of basic expressions."""
    
    def test_number(self):
        """Test parsing number literals."""
        ast = parse("42")
        self.assertIsInstance(ast, NumberNode)
        self.assertEqual(ast.value, 42)
    
    def test_zero(self):
        """Test parsing zero."""
        ast = parse("0")
        self.assertIsInstance(ast, NumberNode)
        self.assertEqual(ast.value, 0)
    
    def test_boolean_true(self):
        """Test parsing true."""
        ast = parse("true")
        self.assertIsInstance(ast, BooleanNode)
        self.assertEqual(ast.value, True)
    
    def test_boolean_false(self):
        """Test parsing false."""
        ast = parse("false")
        self.assertIsInstance(ast, BooleanNode)
        self.assertEqual(ast.value, False)
    
    def test_variable(self):
        """Test parsing variable names."""
        ast = parse("x")
        self.assertIsInstance(ast, VariableNode)
        self.assertEqual(ast.name, "x")
    
    def test_variable_with_underscore(self):
        """Test parsing variables with underscores."""
        ast = parse("foo_bar")
        self.assertIsInstance(ast, VariableNode)
        self.assertEqual(ast.name, "foo_bar")
    
    def test_operator_as_variable(self):
        """Test parsing operators as variables."""
        ast = parse("+")
        self.assertIsInstance(ast, VariableNode)
        self.assertEqual(ast.name, "+")


class TestLists(unittest.TestCase):
    """Test parsing of list literals."""
    
    def test_empty_list(self):
        """Test parsing empty list."""
        ast = parse("[]")
        self.assertIsInstance(ast, ListNode)
        self.assertEqual(len(ast.elements), 0)
    
    def test_single_element_list(self):
        """Test parsing single-element list."""
        ast = parse("[1]")
        self.assertIsInstance(ast, ListNode)
        self.assertEqual(len(ast.elements), 1)
        self.assertIsInstance(ast.elements[0], NumberNode)
        self.assertEqual(ast.elements[0].value, 1)
    
    def test_multiple_element_list(self):
        """Test parsing multi-element list."""
        ast = parse("[1 2 3]")
        self.assertIsInstance(ast, ListNode)
        self.assertEqual(len(ast.elements), 3)
        for i, elem in enumerate(ast.elements):
            self.assertIsInstance(elem, NumberNode)
            self.assertEqual(elem.value, i + 1)
    
    def test_nested_list(self):
        """Test parsing nested lists."""
        ast = parse("[[1 2] [3 4]]")
        self.assertIsInstance(ast, ListNode)
        self.assertEqual(len(ast.elements), 2)
        for elem in ast.elements:
            self.assertIsInstance(elem, ListNode)
            self.assertEqual(len(elem.elements), 2)
    
    def test_mixed_list(self):
        """Test parsing list with mixed types."""
        ast = parse("[1 true x]")
        self.assertIsInstance(ast, ListNode)
        self.assertEqual(len(ast.elements), 3)
        self.assertIsInstance(ast.elements[0], NumberNode)
        self.assertIsInstance(ast.elements[1], BooleanNode)
        self.assertIsInstance(ast.elements[2], VariableNode)


class TestLambdas(unittest.TestCase):
    """Test parsing of lambda expressions."""
    
    def test_identity_function(self):
        """Test parsing identity function: (λ x x)"""
        ast = parse("(λ x x)")
        self.assertIsInstance(ast, LambdaNode)
        self.assertEqual(ast.param, "x")
        self.assertIsInstance(ast.body, VariableNode)
        self.assertEqual(ast.body.name, "x")
    
    def test_constant_function(self):
        """Test parsing constant function: (λ x 42)"""
        ast = parse("(λ x 42)")
        self.assertIsInstance(ast, LambdaNode)
        self.assertEqual(ast.param, "x")
        self.assertIsInstance(ast.body, NumberNode)
        self.assertEqual(ast.body.value, 42)
    
    def test_nested_lambda(self):
        """Test parsing nested lambdas: (λ x (λ y x))"""
        ast = parse("(λ x (λ y x))")
        self.assertIsInstance(ast, LambdaNode)
        self.assertEqual(ast.param, "x")
        self.assertIsInstance(ast.body, LambdaNode)
        self.assertEqual(ast.body.param, "y")
        self.assertIsInstance(ast.body.body, VariableNode)
        self.assertEqual(ast.body.body.name, "x")
    
    def test_lambda_with_application(self):
        """Test parsing lambda with application: (λ x (+ x 1))"""
        ast = parse("(λ x (+ x 1))")
        self.assertIsInstance(ast, LambdaNode)
        self.assertEqual(ast.param, "x")
        self.assertIsInstance(ast.body, ApplicationNode)


class TestApplications(unittest.TestCase):
    """Test parsing of function applications."""
    
    def test_simple_application(self):
        """Test parsing simple application: (f x)"""
        ast = parse("(f x)")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertIsInstance(ast.function, VariableNode)
        self.assertEqual(ast.function.name, "f")
        self.assertEqual(len(ast.arguments), 1)
        self.assertIsInstance(ast.arguments[0], VariableNode)
        self.assertEqual(ast.arguments[0].name, "x")
    
    def test_binary_operator(self):
        """Test parsing binary operator: (+ 1 2)"""
        ast = parse("(+ 1 2)")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertIsInstance(ast.function, VariableNode)
        self.assertEqual(ast.function.name, "+")
        self.assertEqual(len(ast.arguments), 2)
        self.assertIsInstance(ast.arguments[0], NumberNode)
        self.assertEqual(ast.arguments[0].value, 1)
        self.assertIsInstance(ast.arguments[1], NumberNode)
        self.assertEqual(ast.arguments[1].value, 2)
    
    def test_nested_application(self):
        """Test parsing nested application: (f (g x))"""
        ast = parse("(f (g x))")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertIsInstance(ast.function, VariableNode)
        self.assertEqual(ast.function.name, "f")
        self.assertEqual(len(ast.arguments), 1)
        self.assertIsInstance(ast.arguments[0], ApplicationNode)
    
    def test_multiple_arguments(self):
        """Test parsing multiple arguments: (f a b c)"""
        ast = parse("(f a b c)")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(len(ast.arguments), 3)
    
    def test_application_with_lambda(self):
        """Test parsing application with lambda: ((λ x x) 5)"""
        ast = parse("((λ x x) 5)")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertIsInstance(ast.function, LambdaNode)
        self.assertEqual(len(ast.arguments), 1)
        self.assertIsInstance(ast.arguments[0], NumberNode)


class TestIfExpressions(unittest.TestCase):
    """Test parsing of if expressions."""
    
    def test_simple_if(self):
        """Test parsing simple if: (if true 1 2)"""
        ast = parse("(if true 1 2)")
        self.assertIsInstance(ast, IfNode)
        self.assertIsInstance(ast.condition, BooleanNode)
        self.assertIsInstance(ast.then_expr, NumberNode)
        self.assertIsInstance(ast.else_expr, NumberNode)
    
    def test_if_with_condition(self):
        """Test parsing if with comparison: (if (< x 5) true false)"""
        ast = parse("(if (< x 5) true false)")
        self.assertIsInstance(ast, IfNode)
        self.assertIsInstance(ast.condition, ApplicationNode)
        self.assertIsInstance(ast.then_expr, BooleanNode)
        self.assertIsInstance(ast.else_expr, BooleanNode)
    
    def test_nested_if(self):
        """Test parsing nested if expressions."""
        ast = parse("(if true (if false 1 2) 3)")
        self.assertIsInstance(ast, IfNode)
        self.assertIsInstance(ast.then_expr, IfNode)


class TestListOperations(unittest.TestCase):
    """Test parsing list operations from the spec."""
    
    def test_cons(self):
        """Test parsing cons: (cons 1 [2 3])"""
        ast = parse("(cons 1 [2 3])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "cons")
        self.assertEqual(len(ast.arguments), 2)
        self.assertIsInstance(ast.arguments[0], NumberNode)
        self.assertIsInstance(ast.arguments[1], ListNode)
    
    def test_first(self):
        """Test parsing first: (first [1 2 3])"""
        ast = parse("(first [1 2 3])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "first")
    
    def test_last(self):
        """Test parsing last: (last [1 2 3])"""
        ast = parse("(last [1 2 3])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "last")
    
    def test_take(self):
        """Test parsing take: (take 2 [1 2 3])"""
        ast = parse("(take 2 [1 2 3])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "take")
        self.assertEqual(len(ast.arguments), 2)
    
    def test_drop(self):
        """Test parsing drop: (drop 2 [1 2 3])"""
        ast = parse("(drop 2 [1 2 3])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "drop")


class TestHigherOrderFunctions(unittest.TestCase):
    """Test parsing higher-order functions."""
    
    def test_map(self):
        """Test parsing map: (map (λ x (* x 2)) [1 2 3])"""
        ast = parse("(map (λ x (* x 2)) [1 2 3])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "map")
        self.assertEqual(len(ast.arguments), 2)
        self.assertIsInstance(ast.arguments[0], LambdaNode)
        self.assertIsInstance(ast.arguments[1], ListNode)
    
    def test_filter(self):
        """Test parsing filter: (filter (λ x (> x 5)) [3 7 2 9])"""
        ast = parse("(filter (λ x (> x 5)) [3 7 2 9])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "filter")
        self.assertEqual(len(ast.arguments), 2)
        self.assertIsInstance(ast.arguments[0], LambdaNode)
        self.assertIsInstance(ast.arguments[1], ListNode)
    
    def test_fold(self):
        """Test parsing fold: (fold (λ acc (λ x (+ acc x))) 0 [1 2 3])"""
        ast = parse("(fold (λ acc (λ x (+ acc x))) 0 [1 2 3])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "fold")
        self.assertEqual(len(ast.arguments), 3)
        self.assertIsInstance(ast.arguments[0], LambdaNode)
        self.assertIsInstance(ast.arguments[1], NumberNode)
        self.assertIsInstance(ast.arguments[2], ListNode)
    
    def test_foldi(self):
        """Test parsing foldi with three parameters."""
        ast = parse("(foldi (λ acc (λ x (λ i (+ acc i)))) 0 [1 2 3])")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "foldi")


class TestExamplesFromSpec(unittest.TestCase):
    """Test parsing examples from the language specification."""
    
    def test_remove_all_but_element_1(self):
        """Test: (λ x (take 1 x))"""
        ast = parse("(λ x (take 1 x))")
        self.assertIsInstance(ast, LambdaNode)
        self.assertEqual(ast.param, "x")
        self.assertIsInstance(ast.body, ApplicationNode)
        self.assertEqual(ast.body.function.name, "take")
    
    def test_cons_first_empty(self):
        """Test: (λ x (cons (first x) []))"""
        ast = parse("(λ x (cons (first x) []))")
        self.assertIsInstance(ast, LambdaNode)
        self.assertIsInstance(ast.body, ApplicationNode)
        self.assertEqual(ast.body.function.name, "cons")
        self.assertEqual(len(ast.body.arguments), 2)
        self.assertIsInstance(ast.body.arguments[1], ListNode)
        self.assertEqual(len(ast.body.arguments[1].elements), 0)
    
    def test_flatten_mapi(self):
        """Test: (λ x (flatten (mapi (λ y (λ z (cons z (singleton y)))) x)))"""
        ast = parse("(λ x (flatten (mapi (λ y (λ z (cons z (singleton y)))) x)))")
        self.assertIsInstance(ast, LambdaNode)
        self.assertEqual(ast.param, "x")
        # Body should be application of flatten
        self.assertIsInstance(ast.body, ApplicationNode)
        self.assertEqual(ast.body.function.name, "flatten")
    
    def test_singleton_first(self):
        """Test: (λ x (singleton (first x)))"""
        ast = parse("(λ x (singleton (first x)))")
        self.assertIsInstance(ast, LambdaNode)
        self.assertIsInstance(ast.body, ApplicationNode)
        self.assertEqual(ast.body.function.name, "singleton")
    
    def test_droplast(self):
        """Test: (λ x (droplast 1 x))"""
        ast = parse("(λ x (droplast 1 x))")
        self.assertIsInstance(ast, LambdaNode)
        self.assertIsInstance(ast.body, ApplicationNode)
        self.assertEqual(ast.body.function.name, "droplast")


class TestComplexExpressions(unittest.TestCase):
    """Test parsing complex nested expressions."""
    
    def test_curried_application(self):
        """Test parsing curried function application."""
        ast = parse("(((f x) y) z)")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertIsInstance(ast.function, ApplicationNode)
        self.assertIsInstance(ast.function.function, ApplicationNode)
    
    def test_deeply_nested_lambda(self):
        """Test parsing deeply nested lambdas."""
        ast = parse("(λ a (λ b (λ c (λ d (+ a b)))))")
        self.assertIsInstance(ast, LambdaNode)
        self.assertEqual(ast.param, "a")
        self.assertIsInstance(ast.body, LambdaNode)
        self.assertEqual(ast.body.param, "b")
        self.assertIsInstance(ast.body.body, LambdaNode)
    
    def test_complex_list_operations(self):
        """Test parsing complex list operations."""
        code = "(concat (map (λ x (+ x 1)) [1 2]) (filter (λ y (> y 0)) [3 4]))"
        ast = parse(code)
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "concat")
        self.assertEqual(len(ast.arguments), 2)


class TestOperators(unittest.TestCase):
    """Test parsing various operators."""
    
    def test_arithmetic_operators(self):
        """Test parsing arithmetic operators."""
        operators = ["+", "-", "*", "/", "%"]
        for op in operators:
            ast = parse(f"({op} 1 2)")
            self.assertIsInstance(ast, ApplicationNode)
            self.assertEqual(ast.function.name, op)
    
    def test_comparison_operators(self):
        """Test parsing comparison operators."""
        operators = ["<", ">", "=="]
        for op in operators:
            ast = parse(f"({op} 1 2)")
            self.assertIsInstance(ast, ApplicationNode)
            self.assertEqual(ast.function.name, op)
    
    def test_boolean_operators(self):
        """Test parsing boolean operators."""
        ast = parse("(and true false)")
        self.assertIsInstance(ast, ApplicationNode)
        self.assertEqual(ast.function.name, "and")
        
        ast = parse("(or true false)")
        self.assertEqual(ast.function.name, "or")
        
        ast = parse("(not true)")
        self.assertEqual(ast.function.name, "not")


class TestErrorHandling(unittest.TestCase):
    """Test error handling in the parser."""
    
    def test_empty_input(self):
        """Test that empty input raises an error."""
        with self.assertRaises(ParseError):
            parse("")
    
    def test_unclosed_paren(self):
        """Test that unclosed parenthesis raises an error."""
        with self.assertRaises(ParseError):
            parse("(+ 1 2")
    
    def test_unclosed_bracket(self):
        """Test that unclosed bracket raises an error."""
        with self.assertRaises(ParseError):
            parse("[1 2 3")
    
    def test_extra_closing_paren(self):
        """Test that extra closing paren raises an error."""
        with self.assertRaises(ParseError):
            parse("(+ 1 2))")
    
    def test_lambda_without_param(self):
        """Test that lambda without parameter raises an error."""
        with self.assertRaises(ParseError):
            parse("(λ)")
    
    def test_lambda_without_body(self):
        """Test that lambda without body raises an error."""
        with self.assertRaises(ParseError):
            parse("(λ x)")
    
    def test_empty_s_expression(self):
        """Test that empty S-expression raises an error."""
        with self.assertRaises(ParseError):
            parse("()")
    
    def test_if_without_else(self):
        """Test that if without else raises an error."""
        with self.assertRaises(ParseError):
            parse("(if true 1)")


class TestParserState(unittest.TestCase):
    """Test parser state management."""
    
    def test_peek(self):
        """Test the peek function."""
        parser = Parser("(+ 1 2)")
        self.assertEqual(parser.current_token.type.name, "LPAREN")
        next_token = parser.peek()
        self.assertEqual(next_token.type.name, "IDENT")
        # peek shouldn't advance
        self.assertEqual(parser.current_token.type.name, "LPAREN")
    
    def test_advance(self):
        """Test the advance function."""
        parser = Parser("(+ 1 2)")
        parser.advance()
        self.assertEqual(parser.current_token.type.name, "IDENT")
    
    def test_multiple_parses(self):
        """Test that we can parse multiple times."""
        ast1 = parse("(+ 1 2)")
        ast2 = parse("(* 3 4)")
        self.assertIsInstance(ast1, ApplicationNode)
        self.assertIsInstance(ast2, ApplicationNode)
        self.assertEqual(ast1.function.name, "+")
        self.assertEqual(ast2.function.name, "*")


if __name__ == "__main__":
    unittest.main()

