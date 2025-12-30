"""
MLMP - Metalearning Metaprogram Search

Functional Programming Language Implementation

This package provides a complete implementation of a Lisp-style functional
programming language with:
- Lexer for tokenization
- Parser for AST generation
- Type system with Hindley-Milner inference
- Evaluator for program execution
- 50+ built-in functions

Quick start:
    >>> from mlmp import evaluate, type_check
    >>> evaluate("(+ 1 2)")
    3
    >>> evaluate("(map (λ x (* x 2)) [1 2 3])")
    [2, 4, 6]
    >>> type_check("(λ x (+ x 1))")
    Int → Int
"""

from .lexer import Lexer, Token, TokenType, LexerError, tokenize
from .parser import Parser, parse, ParseError
from .ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode,
    Expression, pretty_print
)
from .evaluator import Evaluator, evaluate, EvaluationError
from .environment import Environment, Closure
from .type_checker import TypeChecker, type_check, format_type, TypeCheckError
from .type_utils import (
    analyse_function_types,
    get_origin, CallableOrig, matchable,
    SubstitutionTable,
)
from .grammar import Grammar, DefaultGrammar

__all__ = [
    # Lexer
    'Lexer', 'Token', 'TokenType', 'LexerError', 'tokenize',
    # Parser
    'Parser', 'parse', 'ParseError',
    # AST Nodes
    'ASTNode', 'NumberNode', 'BooleanNode', 'VariableNode',
    'LambdaNode', 'ApplicationNode', 'ListNode', 'IfNode',
    'Expression', 'pretty_print',
    # Evaluator
    'Evaluator', 'evaluate', 'EvaluationError',
    # Environment
    'Environment', 'Closure',
    # Type Checker
    'TypeChecker', 'type_check', 'format_type', 'TypeCheckError',
    # Type Utils
    'analyse_function_types', 'get_origin', 'CallableOrig', 'matchable',
    'SubstitutionTable',
    # Grammar
    'Grammar', 'DefaultGrammar',
]
