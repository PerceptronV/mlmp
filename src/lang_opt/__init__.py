"""
lang_opt - High-Performance Language Implementation

This module provides a hybrid Python/C++ implementation where:
- Lexer, Parser, and Evaluator are implemented in C++ for performance
- AST nodes remain pure Python for easy manipulation
- Seamless integration with the existing Python codebase

Usage:
    from src.lang_opt import tokenize, parse, evaluate
    
    # Tokenize (C++ implementation)
    tokens = tokenize("(λ x (+ x 1))")
    
    # Parse (C++ implementation, returns Python AST)
    ast = parse("(λ x (+ x 1))")
    
    # Evaluate (C++ implementation, works with Python AST)
    result = evaluate("((λ x (+ x 1)) 5)")  # Returns 6
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import the native C++ module (lexer and parser only)
    from . import lang_opt_native
    
    # Re-export C++ functions
    tokenize = lang_opt_native.tokenize
    parse = lang_opt_native.parse
    
    # Re-export C++ classes
    Lexer = lang_opt_native.Lexer
    Parser = lang_opt_native.Parser
    
    # Re-export token types
    TokenType = lang_opt_native.TokenType
    Token = lang_opt_native.Token
    
    # Re-export C++ exceptions
    LexerError = lang_opt_native.LexerError
    ParseError = lang_opt_native.ParseError
    
    # Import Python evaluator (not in C++)
    from src.lang.evaluator import evaluate, Evaluator, EvaluationError
    
    __version__ = lang_opt_native.__version__
    __all__ = [
        'tokenize', 'parse', 'evaluate',
        'Lexer', 'Parser', 'Evaluator',
        'TokenType', 'Token',
        'LexerError', 'ParseError', 'EvaluationError'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"lang_opt_native module not available. Please build the C++ extension. "
        f"Error: {e}\n"
        f"To build: cd src/lang_opt && python setup.py build_ext --inplace"
    )
    
    # Fallback to pure Python implementation
    from src.lang.lexer import tokenize, Lexer, LexerError
    from src.lang.parser import parse, Parser, ParseError
    from src.lang.evaluator import evaluate, Evaluator, EvaluationError
    from src.lang.lexer import Token, TokenType
    
    __version__ = "0.1.0-py"
    __all__ = [
        'tokenize', 'parse', 'evaluate',
        'Lexer', 'Parser', 'Evaluator',
        'TokenType', 'Token',
        'LexerError', 'ParseError', 'EvaluationError'
    ]

