"""
Parser for Functional Programming Language

This module implements a recursive descent parser that converts tokens
from the lexer into an Abstract Syntax Tree (AST).

The parser handles:
- Lambda expressions: (λ param body)
- Function application: (func arg1 arg2 ...)
- Numbers, booleans, variables
- Lists: [elem1 elem2 ...]
- Special forms: if

Grammar (informal):
    expression := number | boolean | variable | lambda | application | list | if
    lambda := '(' 'λ' IDENT expression ')'
    application := '(' expression expression* ')'
    list := '[' expression* ']'
    if := '(' 'if' expression expression expression ')'
"""

from typing import List, Optional
from .lexer import Token, TokenType, Lexer
from .ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode
)


class ParseError(Exception):
    """Exception raised for parsing errors."""
    def __init__(self, message: str, token: Optional[Token] = None):
        self.message = message
        self.token = token
        if token:
            super().__init__(f"Parse error at position {token.position}: {message}")
        else:
            super().__init__(f"Parse error: {message}")


class Parser:
    """
    Recursive descent parser for the functional programming language.
    
    Usage:
        parser = Parser("(λ x (+ x 1))")
        ast = parser.parse()
    """
    
    def __init__(self, input_text: str):
        """
        Initialize the parser with input text.
        
        Args:
            input_text: The program source code to parse
        """
        self.lexer = Lexer(input_text)
        self.tokens = self.lexer.tokenize()
        self.position = 0
        self.current_token = self.tokens[0] if self.tokens else None
    
    def error(self, message: str) -> None:
        """Raise a parse error at the current token."""
        raise ParseError(message, self.current_token)
    
    def advance(self) -> None:
        """Move to the next token."""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None
    
    def peek(self, offset: int = 1) -> Optional[Token]:
        """
        Look ahead at a future token.
        
        Args:
            offset: Number of tokens to look ahead
            
        Returns:
            The token at current position + offset, or None
        """
        peek_pos = self.position + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None
    
    def expect(self, token_type: TokenType) -> Token:
        """
        Consume a token of the expected type.
        
        Args:
            token_type: The expected token type
            
        Returns:
            The consumed token
            
        Raises:
            ParseError: If the current token is not of the expected type
        """
        if not self.current_token or self.current_token.type != token_type:
            expected = token_type.name
            actual = self.current_token.type.name if self.current_token else "EOF"
            self.error(f"Expected {expected}, got {actual}")
        
        token = self.current_token
        self.advance()
        return token
    
    def parse(self) -> ASTNode:
        """
        Parse the input and return the AST root.
        
        Returns:
            The root AST node
            
        Raises:
            ParseError: If the input is invalid
        """
        if not self.current_token or self.current_token.type == TokenType.EOF:
            self.error("Empty input")
        
        ast = self.parse_expression()
        
        # Ensure we've consumed all input (except EOF)
        if self.current_token and self.current_token.type != TokenType.EOF:
            self.error(f"Unexpected token after expression: {self.current_token}")
        
        return ast
    
    def parse_expression(self) -> ASTNode:
        """
        Parse an expression.
        
        Returns:
            An AST node representing the expression
        """
        if not self.current_token:
            self.error("Unexpected end of input")
        
        token = self.current_token
        
        # Number literal
        if token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(int(token.value))
        
        # Boolean literal
        elif token.type == TokenType.BOOLEAN:
            self.advance()
            return BooleanNode(token.value == "true")
        
        # Variable/identifier
        elif token.type == TokenType.IDENT:
            self.advance()
            return VariableNode(token.value)
        
        # List literal
        elif token.type == TokenType.LBRACKET:
            return self.parse_list()
        
        # S-expression (lambda, application, or special form)
        elif token.type == TokenType.LPAREN:
            return self.parse_s_expression()
        
        else:
            self.error(f"Unexpected token: {token}")
    
    def parse_list(self) -> ListNode:
        """
        Parse a list literal: [elem1 elem2 ...]
        
        Returns:
            A ListNode containing the list elements
        """
        self.expect(TokenType.LBRACKET)
        
        elements = []
        while self.current_token and self.current_token.type != TokenType.RBRACKET:
            elements.append(self.parse_expression())
        
        self.expect(TokenType.RBRACKET)
        return ListNode(elements)
    
    def parse_s_expression(self) -> ASTNode:
        """
        Parse an S-expression: (...)
        
        This could be:
        - Lambda: (λ param body)
        - If: (if condition then else)
        - Application: (func arg1 arg2 ...)
        
        Returns:
            An AST node for the S-expression
        """
        self.expect(TokenType.LPAREN)
        
        if not self.current_token or self.current_token.type == TokenType.RPAREN:
            self.error("Empty S-expression")
        
        # Check for lambda
        if self.current_token.type == TokenType.LAMBDA:
            return self.parse_lambda()
        
        # Check for special forms
        if self.current_token.type == TokenType.IDENT:
            if self.current_token.value == "if":
                return self.parse_if()
        
        # Otherwise, it's a function application
        return self.parse_application()
    
    def parse_lambda(self) -> LambdaNode:
        """
        Parse a lambda expression: (λ param body)
        
        Note: The opening paren has already been consumed.
        Lambda expressions take a single parameter. For multiple parameters,
        use nested lambdas: (λ x (λ y body))
        
        Returns:
            A LambdaNode
        """
        self.expect(TokenType.LAMBDA)
        
        # Get parameter name
        if not self.current_token or self.current_token.type != TokenType.IDENT:
            self.error("Lambda requires a parameter name")
        param = self.current_token.value
        self.advance()
        
        # Parse body - exactly one expression
        if not self.current_token or self.current_token.type == TokenType.RPAREN:
            self.error("Lambda requires a body expression")
        
        body = self.parse_expression()
        
        self.expect(TokenType.RPAREN)
        
        return LambdaNode(param, body)
    
    def parse_if(self) -> IfNode:
        """
        Parse an if expression: (if condition then_expr else_expr)
        
        Note: The opening paren has already been consumed.
        
        Returns:
            An IfNode
        """
        self.expect(TokenType.IDENT)  # consume 'if'
        
        # Parse condition
        if not self.current_token or self.current_token.type == TokenType.RPAREN:
            self.error("If requires a condition")
        condition = self.parse_expression()
        
        # Parse then branch
        if not self.current_token or self.current_token.type == TokenType.RPAREN:
            self.error("If requires a then expression")
        then_expr = self.parse_expression()
        
        # Parse else branch
        if not self.current_token or self.current_token.type == TokenType.RPAREN:
            self.error("If requires an else expression")
        else_expr = self.parse_expression()
        
        self.expect(TokenType.RPAREN)
        
        return IfNode(condition, then_expr, else_expr)
    
    def parse_application(self) -> ApplicationNode:
        """
        Parse a function application: (func arg1 arg2 ...)
        
        Note: The opening paren has already been consumed.
        
        Returns:
            An ApplicationNode
        """
        # Parse the function expression
        function = self.parse_expression()
        
        # Parse arguments
        arguments = []
        while self.current_token and self.current_token.type != TokenType.RPAREN:
            arguments.append(self.parse_expression())
        
        self.expect(TokenType.RPAREN)
        
        # Handle zero arguments (unusual but technically valid)
        if not arguments:
            # Just return the function itself, or we could error
            # For now, let's allow it
            pass
        
        return ApplicationNode(function, arguments)


def parse(input_text: str) -> ASTNode:
    """
    Convenience function to parse a program.
    
    Args:
        input_text: The program source code
        
    Returns:
        The AST root node
        
    Example:
        >>> ast = parse("(λ x (+ x 1))")
        >>> print(ast)
        Lambda(x, App(Var(+), [Var(x), Number(1)]))
    """
    parser = Parser(input_text)
    return parser.parse()


if __name__ == "__main__":
    from .ast_nodes import pretty_print
    
    # Example programs
    examples = [
        ("Identity", "(λ x x)"),
        ("Increment", "(λ x (+ x 1))"),
        ("Addition", "(+ 1 2)"),
        ("List", "[1 2 3]"),
        ("Empty list", "[]"),
        ("Cons", "(cons 1 [2 3])"),
        ("Map", "(map (λ x (* x 2)) [1 2 3])"),
        ("Filter", "(filter (λ x (> x 5)) [3 7 2 9])"),
        ("If", "(if (< x 5) true false)"),
        ("Nested lambda", "(λ x (λ y (+ x y)))"),
        ("Take", "(λ x (take 1 x))"),
        ("Droplast", "(λ x (droplast 1 x))"),
    ]
    
    print("Parser Examples:")
    print("=" * 80)
    
    for name, code in examples:
        print(f"\n{name}: {code}")
        try:
            ast = parse(code)
            print(f"AST: {ast}")
            print("Pretty:")
            print(pretty_print(ast))
        except ParseError as e:
            print(f"Error: {e}")

