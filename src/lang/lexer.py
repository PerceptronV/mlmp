r"""
Lexer for Functional Programming Language with S-expressions

This module provides a lexer (tokenizer) for parsing a Lisp/Scheme-style
functional programming language with lambda expressions, lists, and built-in operations.

Supported syntax:
    - Lambda abstraction: (λ x body)
    - Numbers: 0, 1, 2, ...99
    - Booleans: true, false
    - Lists: [elem1 elem2 ...]
    - Empty list: []
    - Function application: (func arg1 arg2 ...)
    - Built-in operators: +, -, *, /, %, <, >, ==
    - Built-in functions: cons, map, filter, fold, etc.

Example:
    >>> lexer = Lexer("(λ x (+ x 1))")
    >>> tokens = lexer.tokenise()
    >>> [str(t) for t in tokens]
    ['LPAREN', 'LAMBDA', 'IDENT(x)', 'LPAREN', 'IDENT(+)', 'IDENT(x)', 'NUMBER(1)', 'RPAREN', 'RPAREN', 'EOF']
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


IDENT_CHARS = {'+', '-', '*', '/', '%', '<', '>', '=', '_', '?', '!', '$'}
SPECIAL_CHARS = {'λ', '(', ')', '[', ']', ' ', '#'}
KEYWORDS = {'true', 'false', 'if'}


class TokenType(Enum):
    """Token types for the functional programming language."""
    LAMBDA = auto()      # λ
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    NUMBER = auto()      # Integer literals: 0-99
    BOOLEAN = auto()     # true, false
    IDENT = auto()       # Variable names and function names
    EOF = auto()         # End of input
    

@dataclass
class Token:
    """Represents a single token in the input."""
    type: TokenType
    value: Optional[str] = None
    position: int = 0
    
    def __str__(self) -> str:
        if self.value:
            return f"{self.type.name}({self.value})"
        return self.type.name
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, pos={self.position})"


class LexerError(Exception):
    """Exception raised for lexical analysis errors."""
    def __init__(self, message: str, position: int):
        self.message = message
        self.position = position
        super().__init__(f"Lexer error at position {position}: {message}")


class Lexer:
    """
    Lexical analyser for the functional programming language.
    
    Usage:
        lexer = Lexer("(λ x (+ x 1))")
        tokens = lexer.tokenise()
    """
    
    def __init__(self, input_text: str):
        """
        Initialise the lexer with input text.
        
        Args:
            input_text: The expression to tokenise
        """
        self.input = input_text
        self.position = 0
        self.current_char = self.input[0] if input_text else None
    
    def error(self, message: str) -> None:
        """Raise a lexer error at the current position."""
        raise LexerError(message, self.position)
    
    def advance(self) -> None:
        """Move to the next character in the input."""
        self.position += 1
        if self.position < len(self.input):
            self.current_char = self.input[self.position]
        else:
            self.current_char = None
    
    def peek(self, offset: int = 1) -> Optional[str]:
        """
        Look ahead at the character at current position + offset.
        
        Args:
            offset: Number of characters to look ahead
            
        Returns:
            The character at the specified position, or None if past end
        """
        peek_pos = self.position + offset
        if peek_pos < len(self.input):
            return self.input[peek_pos]
        return None
    
    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self) -> None:
        """Skip single-line comments starting with #."""
        if self.current_char == '#':
            while self.current_char is not None and self.current_char != '\n':
                self.advance()
            if self.current_char == '\n':
                self.advance()
    
    def read_number(self) -> str:
        """
        Read a number (0-99).

        Returns:
            The number as a string
        """
        chars = []
        while self.current_char is not None and self.current_char.isdigit():
            chars.append(self.current_char)
            self.advance()
        return ''.join(chars)
    
    def read_identifier(self) -> str:
        """
        Read an identifier (variable name, function name, or keyword).
        Identifiers can contain letters, digits, underscores, and special characters like +, -, *, /, %, <, >, =, _, ?

        Returns:
            The identifier string
        """
        chars = []

        # Handle multi-character operators and identifiers
        while self.current_char is not None:
            if self.current_char.isalnum() or self.current_char in IDENT_CHARS:
                chars.append(self.current_char)
                self.advance()
            else:
                break

        return ''.join(chars)
    
    def get_next_token(self) -> Token:
        """
        Get the next token from the input.
        
        Returns:
            The next Token object
            
        Raises:
            LexerError: If an invalid character is encountered
        """
        while self.current_char is not None:
            # Skip whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            # Skip comments
            if self.current_char == '#':
                self.skip_comment()
                continue
            
            # Token position for error reporting
            token_pos = self.position
            
            # Lambda symbol (λ or lambda)
            if self.current_char == 'λ':
                self.advance()
                return Token(TokenType.LAMBDA, 'λ', token_pos)

            # Check for 'lambda' keyword
            if self.current_char == 'l':
                # Look ahead for 'lambda'
                saved_pos = self.position
                word = ''
                while self.current_char and self.current_char.isalpha():
                    word += self.current_char
                    self.advance()

                if word == 'lambda':
                    return Token(TokenType.LAMBDA, 'lambda', token_pos)
                else:
                    # Not lambda, reset and parse as identifier
                    self.position = saved_pos
                    self.current_char = self.input[self.position] if self.position < len(self.input) else None
            
            # Left parenthesis
            if self.current_char == '(':
                self.advance()
                return Token(TokenType.LPAREN, '(', token_pos)
            
            # Right parenthesis
            if self.current_char == ')':
                self.advance()
                return Token(TokenType.RPAREN, ')', token_pos)
            
            # Left bracket
            if self.current_char == '[':
                self.advance()
                return Token(TokenType.LBRACKET, '[', token_pos)
            
            # Right bracket
            if self.current_char == ']':
                self.advance()
                return Token(TokenType.RBRACKET, ']', token_pos)
            
            # Numbers
            if self.current_char.isdigit():
                number = self.read_number()
                return Token(TokenType.NUMBER, number, token_pos)
            
            # Identifiers, keywords, and operators
            if (self.current_char.isalpha() or self.current_char in IDENT_CHARS):
                ident = self.read_identifier()
                
                # Check for boolean keywords
                if ident == 'true' or ident == 'false':
                    return Token(TokenType.BOOLEAN, ident, token_pos)
                
                return Token(TokenType.IDENT, ident, token_pos)
            
            # Unknown character
            self.error(f"Unexpected character: '{self.current_char}'")
        
        # End of input
        return Token(TokenType.EOF, None, self.position)
    
    def tokenise(self) -> List[Token]:
        """
        Tokenize the entire input and return a list of tokens.
        
        Returns:
            List of Token objects, ending with EOF token
        """
        tokens = []
        
        while True:
            token = self.get_next_token()
            tokens.append(token)
            
            if token.type == TokenType.EOF:
                break
        
        return tokens
    
    def __iter__(self):
        """Make the lexer iterable, yielding tokens one at a time."""
        while True:
            token = self.get_next_token()
            yield token
            if token.type == TokenType.EOF:
                break


def tokenise(input_text: str) -> List[Token]:
    """
    Convenience function to tokenise an expression.
    
    Args:
        input_text: The expression to tokenise
        
    Returns:
        List of tokens
        
    Example:
        >>> tokens = tokenise("(λ x (+ x 1))")
        >>> [str(t) for t in tokens]
        ['LPAREN', 'LAMBDA', 'IDENT(x)', 'LPAREN', 'IDENT(+)', 'IDENT(x)', 'NUMBER(1)', 'RPAREN', 'RPAREN', 'EOF']
    """
    lexer = Lexer(input_text)
    return lexer.tokenise()


if __name__ == "__main__":
    # Example usage
    examples = [
        "(λ x x)",                              # Identity function
        "(λ x (+ x 1))",                        # Increment function
        "(+ 1 2)",                              # Addition
        "[1 2 3]",                              # List literal
        "[]",                                   # Empty list
        "(cons 1 [2 3])",                       # Cons operation
        "(map (λ x (* x 2)) [1 2 3])",          # Map with lambda
        "(filter (λ x (> x 5)) [3 7 2 9])",     # Filter
        "(fold (λ acc x (+ acc x)) 0 [1 2 3])", # Fold
        "true",                                 # Boolean
        "(if (< x 5) true false)",              # Conditional
        "(== 1 1)",                             # Equality
    ]
    
    print("Functional Language Lexer Examples:")
    print("=" * 80)
    
    for example in examples:
        print(f"\nInput: {example}")
        try:
            tokens = tokenise(example)
            print(f"Tokens: {' '.join(str(t) for t in tokens)}")
        except LexerError as e:
            print(f"Error: {e}")
