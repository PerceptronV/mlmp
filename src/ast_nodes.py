"""
Abstract Syntax Tree (AST) Node Definitions

This module defines the AST node classes that represent the structure of
parsed programs in the functional programming language.
"""

from dataclasses import dataclass
from typing import List, Union
from abc import ABC, abstractmethod


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the AST node."""
        pass


@dataclass
class NumberNode(ASTNode):
    """Represents a number literal."""
    value: int
    
    def __repr__(self) -> str:
        return f"Number({self.value})"


@dataclass
class BooleanNode(ASTNode):
    """Represents a boolean literal (true/false)."""
    value: bool
    
    def __repr__(self) -> str:
        return f"Boolean({self.value})"


@dataclass
class VariableNode(ASTNode):
    """Represents a variable reference."""
    name: str
    
    def __repr__(self) -> str:
        return f"Var({self.name})"


@dataclass
class LambdaNode(ASTNode):
    """
    Represents a lambda abstraction: (λ param body)
    
    Note: This language uses single-parameter lambdas.
    Multi-parameter functions are represented via nested lambdas (currying).
    """
    param: str
    body: ASTNode
    
    def __repr__(self) -> str:
        return f"Lambda({self.param}, {self.body})"


@dataclass
class ApplicationNode(ASTNode):
    """
    Represents function application: (func arg1 arg2 ...)
    
    For multiple arguments, this represents curried application.
    For example, (+ 1 2) is parsed as Application(Application(+, 1), 2)
    """
    function: ASTNode
    arguments: List[ASTNode]
    
    def __repr__(self) -> str:
        args_str = ", ".join(repr(arg) for arg in self.arguments)
        return f"App({self.function}, [{args_str}])"


@dataclass
class ListNode(ASTNode):
    """Represents a list literal: [elem1 elem2 ...]"""
    elements: List[ASTNode]
    
    def __repr__(self) -> str:
        if not self.elements:
            return "List([])"
        elems_str = ", ".join(repr(elem) for elem in self.elements)
        return f"List([{elems_str}])"


@dataclass
class IfNode(ASTNode):
    """
    Represents a conditional expression: (if condition then_expr else_expr)
    
    Note: While 'if' could be treated as a regular function, we give it
    special treatment for potential optimizations and clarity.
    """
    condition: ASTNode
    then_expr: ASTNode
    else_expr: ASTNode
    
    def __repr__(self) -> str:
        return f"If({self.condition}, {self.then_expr}, {self.else_expr})"


# Type alias for any AST expression
Expression = Union[NumberNode, BooleanNode, VariableNode, LambdaNode, 
                   ApplicationNode, ListNode, IfNode]


def pretty_print(node: ASTNode, indent: int = 0) -> str:
    """
    Pretty-print an AST with indentation for better readability.
    
    Args:
        node: The AST node to print
        indent: Current indentation level
        
    Returns:
        Formatted string representation of the AST
    """
    prefix = "  " * indent
    
    if isinstance(node, NumberNode):
        return f"{prefix}{node.value}"
    
    elif isinstance(node, BooleanNode):
        return f"{prefix}{str(node.value).lower()}"
    
    elif isinstance(node, VariableNode):
        return f"{prefix}{node.name}"
    
    elif isinstance(node, ListNode):
        if not node.elements:
            return f"{prefix}[]"
        result = f"{prefix}[\n"
        for elem in node.elements:
            result += pretty_print(elem, indent + 1) + "\n"
        result += f"{prefix}]"
        return result
    
    elif isinstance(node, LambdaNode):
        result = f"{prefix}(λ {node.param}\n"
        result += pretty_print(node.body, indent + 1) + "\n"
        result += f"{prefix})"
        return result
    
    elif isinstance(node, ApplicationNode):
        result = f"{prefix}(\n"
        result += pretty_print(node.function, indent + 1) + "\n"
        for arg in node.arguments:
            result += pretty_print(arg, indent + 1) + "\n"
        result += f"{prefix})"
        return result
    
    elif isinstance(node, IfNode):
        result = f"{prefix}(if\n"
        result += pretty_print(node.condition, indent + 1) + "\n"
        result += pretty_print(node.then_expr, indent + 1) + "\n"
        result += pretty_print(node.else_expr, indent + 1) + "\n"
        result += f"{prefix})"
        return result
    
    else:
        return f"{prefix}{repr(node)}"


if __name__ == "__main__":
    # Example AST construction
    print("Example AST Nodes:")
    print("=" * 60)
    
    # Number
    num = NumberNode(42)
    print(f"Number: {num}")
    
    # Boolean
    bool_node = BooleanNode(True)
    print(f"Boolean: {bool_node}")
    
    # Variable
    var = VariableNode("x")
    print(f"Variable: {var}")
    
    # List
    lst = ListNode([NumberNode(1), NumberNode(2), NumberNode(3)])
    print(f"List: {lst}")
    
    # Lambda: (λ x x)
    identity = LambdaNode("x", VariableNode("x"))
    print(f"\nIdentity function: {identity}")
    
    # Application: (+ 1 2)
    add = ApplicationNode(
        VariableNode("+"),
        [NumberNode(1), NumberNode(2)]
    )
    print(f"\nAddition: {add}")
    
    # Lambda with application: (λ x (+ x 1))
    increment = LambdaNode(
        "x",
        ApplicationNode(
            VariableNode("+"),
            [VariableNode("x"), NumberNode(1)]
        )
    )
    print(f"\nIncrement function: {increment}")
    print("\nPretty-printed:")
    print(pretty_print(increment))

