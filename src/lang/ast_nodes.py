"""
Abstract Syntax Tree (AST) Node Definitions

This module defines the AST node classes that represent the structure of
parsed programs in the functional programming language.
"""

from dataclasses import dataclass
from typing import List, Union, Set, Optional
from abc import ABC, abstractmethod


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the AST node."""
        pass

    @abstractmethod
    def pretty_print(self, indent: int = 0, inline: bool = True) -> str:
        """Return a pretty-printed string representation of the AST node."""
        return "(*ASTNode*) {self.__class__.__name__}"

    @abstractmethod
    def function_names(self) -> Set[str]:
        """Return a set of function names used in the AST node or its children."""
        pass
    
    def __str__(self) -> str:
        return self.pretty_print(0, True)


@dataclass
class NumberNode(ASTNode):
    """Represents a number literal."""
    value: int
    
    def __repr__(self) -> str:
        return f"Number({self.value})"

    def pretty_print(self, indent: int = 0, inline: bool = True) -> str:
        if inline:
            return str(self.value)
        else:
            return f"  " * indent + str(self.value)
    
    def function_names(self) -> Set[str]:
        return set()


@dataclass
class BooleanNode(ASTNode):
    """Represents a boolean literal (true/false)."""
    value: bool
    
    def __repr__(self) -> str:
        return f"Boolean({self.value})"

    def pretty_print(self, indent: int = 0, inline: bool = True) -> str:
        if inline:
            return str(self.value).lower()
        else:
            return f"  " * indent + str(self.value).lower()

    def function_names(self) -> Set[str]:
        return set()

@dataclass
class VariableNode(ASTNode):
    """Represents a variable reference."""
    name: str
    
    def __repr__(self) -> str:
        return f"Var({self.name})"

    def pretty_print(self, indent: int = 0, inline: bool = True) -> str:
        if inline:
            return self.name
        else:
            return f"  " * indent + self.name

    def function_names(self) -> Set[str]:
        return {self.name}

@dataclass
class LambdaNode(ASTNode):
    """
    Represents a lambda abstraction: (λ param body) or (λ param1 param2 ... body)

    Parameters are always stored as a list of strings, even for single-parameter lambdas.
    This provides a uniform representation for both curried and uncurried functions.
    """
    param: List[str]  # List of parameter names (length 1 for single param)
    body: ASTNode

    def __repr__(self) -> str:
        params_str = " ".join(self.param)
        return f"Lambda({params_str}, {self.body})"
    
    def pretty_print(self, indent: int = 0, inline: bool = True) -> str:
        params_str = " ".join(self.param)
        if inline:
            body_str = self.body.pretty_print(0, True)
            return f"(λ ({params_str}) {body_str})"
        else:
            result = f"  " * indent + f"(λ ({params_str})\n"
            result += self.body.pretty_print(indent + 1, False) + "\n"
            result += f"  " * indent + ")"
            return result
    
    def function_names(self) -> Set[str]:
        return self.body.function_names()

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
    
    def pretty_print(self, indent: int = 0, inline: bool = True) -> str:
        if inline:
            func_str = self.function.pretty_print(0, True)
            args_str = " ".join(arg.pretty_print(0, True) for arg in self.arguments)
            return f"({func_str} {args_str})"
        else:
            result = f"  " * indent + f"(\n"
            result += self.function.pretty_print(indent + 1, False) + "\n"
            for arg in self.arguments:
                result += arg.pretty_print(indent + 1, False) + "\n"
            result += f"  " * indent + ")"
            return result
    
    def function_names(self) -> Set[str]:
        fn = self.function.function_names()
        for arg in self.arguments:
            fn.update(arg.function_names())
        return fn


@dataclass
class ListNode(ASTNode):
    """Represents a list literal: [elem1 elem2 ...]"""
    elements: List[ASTNode]
    
    def __repr__(self) -> str:
        if not self.elements:
            return "List([])"
        elems_str = ", ".join(repr(elem) for elem in self.elements)
        return f"List([{elems_str}])"
    
    def pretty_print(self, indent: int = 0, inline: bool = True) -> str:
        if inline:
            if not self.elements:
                return "[]"
            elems_str = " ".join(elem.pretty_print(0, True) for elem in self.elements)
            return f"[{elems_str}]"
        else:
            if not self.elements:
                return f"  " * indent + "[]"
            result = f"  " * indent + f"[\n"
            for elem in self.elements:
                result += elem.pretty_print(indent + 1, False) + "\n"
            result += f"  " * indent + "]"
            return result
    
    def function_names(self) -> Set[str]:
        fn = set()
        for elem in self.elements:
            fn.update(elem.function_names())
        return fn

@dataclass
class IfNode(ASTNode):
    """
    Represents a conditional expression: (if condition then_expr else_expr)
    
    Note: While 'if' could be treated as a regular function, we give it
    special treatment for potential optimisations and clarity.
    """
    condition: ASTNode
    then_expr: ASTNode
    else_expr: ASTNode
    
    def __repr__(self) -> str:
        return f"If({self.condition}, {self.then_expr}, {self.else_expr})"
    
    def pretty_print(self, indent: int = 0, inline: bool = True) -> str:
        if inline:
            cond_str = self.condition.pretty_print(0, True)
            then_str = self.then_expr.pretty_print(0, True)
            else_str = self.else_expr.pretty_print(0, True)
            return f"(if {cond_str} {then_str} {else_str})"
        else:
            result = f"  " * indent + f"(if\n"
            result += self.condition.pretty_print(indent + 1, False) + "\n"
            result += self.then_expr.pretty_print(indent + 1, False) + "\n"
            result += self.else_expr.pretty_print(indent + 1, False) + "\n"
            result += f"  " * indent + ")"
            return result
    
    def function_names(self) -> Set[str]:
        fn = set()
        fn.update(self.condition.function_names())
        fn.update(self.then_expr.function_names())
        fn.update(self.else_expr.function_names())
        return fn


# Type alias for any AST expression
Expression = Union[NumberNode, BooleanNode, VariableNode, LambdaNode, 
                   ApplicationNode, ListNode, IfNode]


def pretty_print(node: ASTNode, indent: int = 0, inline: bool = True) -> str:
    """
    Pretty-print an AST with optional indentation for better readability.

    Args:
        node: The AST node to print
        indent: Current indentation level (used when inline=False)
        inline: If True, format on a single line; if False, use multi-line with indentation

    Returns:
        Formatted string representation of the AST
    """
    return node.pretty_print(indent, inline)


def extract_function_names(node: ASTNode, grammar_names: Optional[Set[str]] = None) -> Set[str]:
    """
    Extract all function names used in an AST.
    
    Args:
        node: The AST node to analyse
        grammar_names: Set of valid grammar function names
    
    Returns:
        Set of function names from the grammar that appear in the AST
    """
    fn = node.function_names()
    if grammar_names is None:
        return fn
    return fn.intersection(grammar_names)


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

