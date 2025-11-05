"""
Unit tests for the functional programming language lexer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'lang'))

import unittest
from lexer import Lexer, Token, TokenType, LexerError, tokenize


class TestBasicTokens(unittest.TestCase):
    """Test cases for basic token types."""
    
    def test_lambda_symbol(self):
        """Test lexing lambda symbol."""
        tokens = tokenize("λ")
        self.assertEqual(tokens[0].type, TokenType.LAMBDA)
        self.assertEqual(tokens[0].value, "λ")
    
    def test_parentheses(self):
        """Test lexing parentheses."""
        tokens = tokenize("()")
        self.assertEqual(tokens[0].type, TokenType.LPAREN)
        self.assertEqual(tokens[1].type, TokenType.RPAREN)
    
    def test_brackets(self):
        """Test lexing square brackets."""
        tokens = tokenize("[]")
        self.assertEqual(tokens[0].type, TokenType.LBRACKET)
        self.assertEqual(tokens[1].type, TokenType.RBRACKET)
    
    def test_numbers(self):
        """Test lexing numbers."""
        tokens = tokenize("0 1 42 99")
        numbers = [t for t in tokens if t.type == TokenType.NUMBER]
        self.assertEqual(len(numbers), 4)
        self.assertEqual(numbers[0].value, "0")
        self.assertEqual(numbers[1].value, "1")
        self.assertEqual(numbers[2].value, "42")
        self.assertEqual(numbers[3].value, "99")
    
    def test_booleans(self):
        """Test lexing boolean values."""
        tokens = tokenize("true false")
        booleans = [t for t in tokens if t.type == TokenType.BOOLEAN]
        self.assertEqual(len(booleans), 2)
        self.assertEqual(booleans[0].value, "true")
        self.assertEqual(booleans[1].value, "false")
    
    def test_identifiers(self):
        """Test lexing identifiers."""
        tokens = tokenize("x foo bar_baz")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(len(idents), 3)
        self.assertEqual(idents[0].value, "x")
        self.assertEqual(idents[1].value, "foo")
        self.assertEqual(idents[2].value, "bar_baz")


class TestLambdaExpressions(unittest.TestCase):
    """Test cases for lambda expressions."""
    
    def test_identity_function(self):
        """Test lexing identity function: (λ x x)"""
        tokens = tokenize("(λ x x)")
        expected = [
            TokenType.LPAREN,
            TokenType.LAMBDA,
            TokenType.IDENT,
            TokenType.IDENT,
            TokenType.RPAREN,
            TokenType.EOF,
        ]
        types = [t.type for t in tokens]
        self.assertEqual(types, expected)
    
    def test_lambda_with_body(self):
        """Test lexing lambda with expression body: (λ x (+ x 1))"""
        tokens = tokenize("(λ x (+ x 1))")
        types = [t.type for t in tokens]
        expected = [
            TokenType.LPAREN,
            TokenType.LAMBDA,
            TokenType.IDENT,      # x
            TokenType.LPAREN,
            TokenType.IDENT,      # +
            TokenType.IDENT,      # x
            TokenType.NUMBER,     # 1
            TokenType.RPAREN,
            TokenType.RPAREN,
            TokenType.EOF,
        ]
        self.assertEqual(types, expected)
    
    def test_nested_lambda(self):
        """Test lexing nested lambdas."""
        tokens = tokenize("(λ x (λ y (+ x y)))")
        lambda_count = len([t for t in tokens if t.type == TokenType.LAMBDA])
        self.assertEqual(lambda_count, 2)


class TestArithmeticOperators(unittest.TestCase):
    """Test cases for arithmetic operators."""
    
    def test_addition(self):
        """Test lexing addition: (+ 1 2)"""
        tokens = tokenize("(+ 1 2)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(len(idents), 1)
        self.assertEqual(idents[0].value, "+")
    
    def test_subtraction(self):
        """Test lexing subtraction: (- 5 3)"""
        tokens = tokenize("(- 5 3)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "-")
    
    def test_multiplication(self):
        """Test lexing multiplication: (* 2 3)"""
        tokens = tokenize("(* 2 3)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "*")
    
    def test_division(self):
        """Test lexing division: (/ 10 2)"""
        tokens = tokenize("(/ 10 2)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "/")
    
    def test_modulo(self):
        """Test lexing modulo: (% 10 3)"""
        tokens = tokenize("(% 10 3)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "%")


class TestComparisonOperators(unittest.TestCase):
    """Test cases for comparison operators."""
    
    def test_less_than(self):
        """Test lexing less than: (< 1 2)"""
        tokens = tokenize("(< 1 2)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "<")
    
    def test_greater_than(self):
        """Test lexing greater than: (> 2 1)"""
        tokens = tokenize("(> 2 1)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, ">")
    
    def test_equality(self):
        """Test lexing equality: (== 1 1)"""
        tokens = tokenize("(== 1 1)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "==")


class TestListOperations(unittest.TestCase):
    """Test cases for list operations."""
    
    def test_empty_list(self):
        """Test lexing empty list: []"""
        tokens = tokenize("[]")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertEqual(types, [TokenType.LBRACKET, TokenType.RBRACKET])
    
    def test_list_literal(self):
        """Test lexing list literal: [1 2 3]"""
        tokens = tokenize("[1 2 3]")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        expected = [
            TokenType.LBRACKET,
            TokenType.NUMBER,
            TokenType.NUMBER,
            TokenType.NUMBER,
            TokenType.RBRACKET,
        ]
        self.assertEqual(types, expected)
    
    def test_cons(self):
        """Test lexing cons: (cons 1 [2 3])"""
        tokens = tokenize("(cons 1 [2 3])")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "cons")
    
    def test_first(self):
        """Test lexing first: (first [1 2 3])"""
        tokens = tokenize("(first [1 2 3])")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "first")
    
    def test_last(self):
        """Test lexing last: (last [1 2 3])"""
        tokens = tokenize("(last [1 2 3])")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "last")


class TestHigherOrderFunctions(unittest.TestCase):
    """Test cases for higher-order functions."""
    
    def test_map(self):
        """Test lexing map: (map (λ x (* x 2)) [1 2 3])"""
        tokens = tokenize("(map (λ x (* x 2)) [1 2 3])")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "map")
    
    def test_filter(self):
        """Test lexing filter: (filter (λ x (> x 5)) [3 7 2 9])"""
        tokens = tokenize("(filter (λ x (> x 5)) [3 7 2 9])")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "filter")
    
    def test_fold(self):
        """Test lexing fold: (fold (λ acc x (+ acc x)) 0 [1 2 3])"""
        tokens = tokenize("(fold (λ acc x (+ acc x)) 0 [1 2 3])")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "fold")
    
    def test_foldi(self):
        """Test lexing foldi with index."""
        tokens = tokenize("(foldi (λ acc x i (+ acc i)) 0 [1 2 3])")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "foldi")


class TestConditionals(unittest.TestCase):
    """Test cases for conditional expressions."""
    
    def test_if_expression(self):
        """Test lexing if expression: (if (< x 5) true false)"""
        tokens = tokenize("(if (< x 5) true false)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "if")
        bools = [t for t in tokens if t.type == TokenType.BOOLEAN]
        self.assertEqual(len(bools), 2)


class TestBooleanOperations(unittest.TestCase):
    """Test cases for boolean operations."""
    
    def test_and(self):
        """Test lexing and: (and true false)"""
        tokens = tokenize("(and true false)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "and")
    
    def test_or(self):
        """Test lexing or: (or true false)"""
        tokens = tokenize("(or true false)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "or")
    
    def test_not(self):
        """Test lexing not: (not true)"""
        tokens = tokenize("(not true)")
        idents = [t for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(idents[0].value, "not")


class TestComplexExpressions(unittest.TestCase):
    """Test cases for complex nested expressions."""
    
    def test_nested_arithmetic(self):
        """Test lexing nested arithmetic: (+ (* 2 3) (/ 10 5))"""
        tokens = tokenize("(+ (* 2 3) (/ 10 5))")
        ops = [t.value for t in tokens if t.type == TokenType.IDENT]
        self.assertEqual(ops, ["+", "*", "/"])
    
    def test_map_with_lambda(self):
        """Test lexing map with lambda function."""
        code = "(map (λ x (* x 2)) [1 2 3])"
        tokens = tokenize(code)
        self.assertEqual(len([t for t in tokens if t.type == TokenType.LAMBDA]), 1)
        self.assertEqual(len([t for t in tokens if t.type == TokenType.LBRACKET]), 1)
    
    def test_filter_with_condition(self):
        """Test lexing filter with conditional."""
        code = "(filter (λ x (> x 5)) [3 7 2 9])"
        tokens = tokenize(code)
        idents = [t.value for t in tokens if t.type == TokenType.IDENT]
        self.assertIn("filter", idents)
        self.assertIn(">", idents)


class TestWhitespaceAndComments(unittest.TestCase):
    """Test cases for whitespace and comment handling."""
    
    def test_whitespace_handling(self):
        """Test that whitespace is properly skipped."""
        tokens1 = tokenize("(+ 1 2)")
        tokens2 = tokenize("(  +   1   2  )")
        tokens3 = tokenize("(\n+\n1\n2\n)")
        
        types1 = [t.type for t in tokens1]
        types2 = [t.type for t in tokens2]
        types3 = [t.type for t in tokens3]
        
        self.assertEqual(types1, types2)
        self.assertEqual(types2, types3)
    
    def test_comments(self):
        """Test that comments are properly skipped."""
        tokens = tokenize("(+ 1 2) # add two numbers")
        types = [t.type for t in tokens]
        expected = [
            TokenType.LPAREN,
            TokenType.IDENT,
            TokenType.NUMBER,
            TokenType.NUMBER,
            TokenType.RPAREN,
            TokenType.EOF,
        ]
        self.assertEqual(types, expected)


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error handling."""
    
    def test_empty_input(self):
        """Test lexing empty input."""
        tokens = tokenize("")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.EOF)
    
    def test_only_whitespace(self):
        """Test lexing input with only whitespace."""
        tokens = tokenize("   \n  \t  ")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, TokenType.EOF)
    
    def test_token_positions(self):
        """Test that token positions are correctly tracked."""
        tokens = tokenize("(+ 1 2)")
        self.assertEqual(tokens[0].position, 0)  # (
        self.assertEqual(tokens[1].position, 1)  # +
        self.assertEqual(tokens[2].position, 3)  # 1
        self.assertEqual(tokens[3].position, 5)  # 2
    
    def test_lexer_iterator(self):
        """Test that the lexer can be used as an iterator."""
        lexer = Lexer("(+ 1 2)")
        tokens = list(lexer)
        self.assertEqual(len(tokens), 6)  # (, +, 1, 2, ), EOF
        self.assertEqual(tokens[-1].type, TokenType.EOF)
    
    def test_token_string_representation(self):
        """Test token string representations."""
        token1 = Token(TokenType.LAMBDA, "λ", 0)
        self.assertEqual(str(token1), "LAMBDA(λ)")
        
        token2 = Token(TokenType.NUMBER, "42", 1)
        self.assertEqual(str(token2), "NUMBER(42)")


class TestBuiltInFunctions(unittest.TestCase):
    """Test cases for built-in function names."""
    
    def test_list_functions(self):
        """Test lexing various list function names."""
        functions = [
            "singleton", "repeat", "range", "cons", "append", "insert",
            "concat", "splice", "first", "second", "third", "last", "nth",
            "replace", "swap", "cut_idx", "cut_val", "cut_vals", "drop",
            "droplast", "cut_slice", "take", "takelast", "slice"
        ]
        for func in functions:
            tokens = tokenize(f"({func})")
            idents = [t for t in tokens if t.type == TokenType.IDENT]
            self.assertEqual(idents[0].value, func, f"Failed to tokenize {func}")
    
    def test_higher_order_functions(self):
        """Test lexing higher-order function names."""
        functions = ["fold", "foldi", "filter", "filteri", "map", "mapi"]
        for func in functions:
            tokens = tokenize(f"({func})")
            idents = [t for t in tokens if t.type == TokenType.IDENT]
            self.assertEqual(idents[0].value, func)
    
    def test_utility_functions(self):
        """Test lexing utility function names."""
        functions = [
            "is_even", "is_odd", "is_in", "count", "find", "group",
            "length", "max", "min", "product", "sum", "unique",
            "sort", "reverse", "flatten", "zip"
        ]
        for func in functions:
            tokens = tokenize(f"({func})")
            idents = [t for t in tokens if t.type == TokenType.IDENT]
            self.assertEqual(idents[0].value, func)


class TestExamplesFromSpec(unittest.TestCase):
    """Test cases based on examples from the specification."""
    
    def test_take_example(self):
        """Test: (λ (x) (take 1 x))"""
        tokens = tokenize("(λ x (take 1 x))")
        self.assertIsNotNone(tokens)
        self.assertEqual(tokens[0].type, TokenType.LPAREN)
        self.assertEqual(tokens[1].type, TokenType.LAMBDA)
    
    def test_cons_head_empty(self):
        """Test: (λ (x) (cons (head x) empty))"""
        tokens = tokenize("(λ x (cons (first x) []))")
        lambda_count = len([t for t in tokens if t.type == TokenType.LAMBDA])
        self.assertEqual(lambda_count, 1)
    
    def test_flatten_mapi(self):
        """Test: (λ (x) (flatten (mapi (λ (y z) (cons z (singleton y))) x)))"""
        tokens = tokenize("(λ x (flatten (mapi (λ y z (cons z (singleton y))) x)))")
        lambda_count = len([t for t in tokens if t.type == TokenType.LAMBDA])
        self.assertEqual(lambda_count, 2)
    
    def test_singleton_first(self):
        """Test: (λ (x) (singleton (first x)))"""
        tokens = tokenize("(λ x (singleton (first x)))")
        idents = [t.value for t in tokens if t.type == TokenType.IDENT]
        self.assertIn("singleton", idents)
        self.assertIn("first", idents)
    
    def test_droplast(self):
        """Test: (λ (x) (droplast 1 x))"""
        tokens = tokenize("(λ x (droplast 1 x))")
        idents = [t.value for t in tokens if t.type == TokenType.IDENT]
        self.assertIn("droplast", idents)


if __name__ == "__main__":
    unittest.main()
