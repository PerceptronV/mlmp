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
from .type_system import (
    Type, TypeVar, IntType, BoolType, ListType, FunctionType,
    TypeScheme, INT, BOOL, list_of, func
)
from .type_checker import TypeChecker, type_check

# Re-export TypeError from type_system to avoid confusion
from .type_system import TypeError as TypeSystemError

__version__ = "2.0.0"
__all__ = [
    # Lexer
    "Lexer", "Token", "TokenType", "LexerError", "tokenize",
    # Parser
    "Parser", "parse", "ParseError",
    # AST Nodes
    "ASTNode", "NumberNode", "BooleanNode", "VariableNode",
    "LambdaNode", "ApplicationNode", "ListNode", "IfNode",
    "Expression", "pretty_print",
    # Evaluator
    "Evaluator", "evaluate", "EvaluationError",
    # Environment
    "Environment", "Closure",
    # Type System
    "Type", "TypeVar", "IntType", "BoolType", "ListType", "FunctionType",
    "TypeScheme", "INT", "BOOL", "list_of", "func",
    # Type Checker
    "TypeChecker", "type_check", "TypeSystemError"
]
