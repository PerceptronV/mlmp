"""
Evaluator for the Functional Programming Language

This module implements a high-performance interpreter that executes
parsed AST programs. It includes all 50+ built-in functions and
proper closure semantics.
"""

from typing import Any, List, Callable
from .ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode
)
from .environment import Environment, Closure
from .grammar import Grammar, DefaultGrammar


class EvaluationError(Exception):
    """Exception raised during evaluation."""
    pass


class Evaluator:
    """
    Evaluator for the functional programming language.
    
    Executes AST programs with full support for:
    - Lambda functions with closures
    - All built-in functions
    - Lexical scoping
    - Lists and higher-order operations
    """
    
    def __init__(self, grammar: Grammar = DefaultGrammar):
        """Initialise the evaluator with built-in functions."""
        self.global_env = self._create_global_environment(grammar)
        self.grammar = grammar
    
    def _create_global_environment(self, grammar: Grammar) -> Environment:
        """Create the global environment with all built-in functions."""
        env = Environment()
        
        for name, fn in grammar:
            env.define(name, fn)
        
        return env
    
    def eval(self, node: ASTNode, env: Environment = None) -> Any:
        """
        Evaluate an AST node.

        Args:
            node: AST node to evaluate
            env: Environment for variable lookups (uses global if None)

        Returns:
            The result of evaluation
        """
        if env is None:
            env = self.global_env

        # Use type() instead of isinstance() for faster type checking
        node_type = type(node)

        # Numbers evaluate to themselves
        if node_type is NumberNode:
            return node.value

        # Booleans evaluate to themselves
        elif node_type is BooleanNode:
            return node.value

        # Variables: look up in environment
        elif node_type is VariableNode:
            return env.get(node.name)

        # Lists: evaluate all elements
        elif node_type is ListNode:
            return [self.eval(elem, env) for elem in node.elements]

        # Lambda: create closure
        elif node_type is LambdaNode:
            return Closure(node.param, node.body, env)

        # If: evaluate condition, then appropriate branch
        elif node_type is IfNode:
            condition = self.eval(node.condition, env)
            if type(condition) is not bool:
                raise EvaluationError(f"If condition must be boolean, got {type(condition).__name__}")
            if condition:
                return self.eval(node.then_expr, env)
            else:
                return self.eval(node.else_expr, env)

        # Application: apply function to arguments
        elif node_type is ApplicationNode:
            func = self.eval(node.function, env)
            args = [self.eval(arg, env) for arg in node.arguments]
            return self._apply(func, args, env)

        else:
            raise EvaluationError(f"Unknown node type: {node_type.__name__}")
    
    def _apply(self, func: Any, args: List[Any], env: Environment) -> Any:
        """
        Apply a function to arguments.

        Args:
            func: Function (Closure or built-in)
            args: List of evaluated arguments
            env: Current environment

        Returns:
            Result of application
        """
        # User-defined function (closure) - check this first as it's more common
        if type(func) is Closure:
            # Multi-parameter closures: bind all parameters at once
            # Single-parameter closures with multiple args: apply via currying
            if len(func.param) == len(args):
                # Exact match: bind all parameters at once
                new_env = func.env
                for param, arg in zip(func.param, args):
                    new_env = new_env.extend(param, arg)
                return self.eval(func.body, new_env)
            elif len(func.param) == 1 and len(args) > 1:
                # Single-parameter closure with multiple args: curry
                result = func
                for arg in args:
                    if type(result) is not Closure:
                        raise EvaluationError(f"Too many arguments to function")
                    # Bind the single parameter
                    new_env = result.env.extend(result.param[0], arg)
                    result = self.eval(result.body, new_env)
                return result
            else:
                raise EvaluationError(
                    f"Function expects {len(func.param)} arguments, got {len(args)}"
                )

        # Built-in function
        elif callable(func):
            return func(self, *args)

        else:
            raise EvaluationError(f"Cannot apply non-function: {type(func).__name__}")


def evaluate(code: str, grammar: Grammar = DefaultGrammar) -> Any:
    """
    Convenience function to parse and evaluate code.
    
    Args:
        code: Source code string
        grammar: Grammar to use for evaluation
        
    Returns:
        Evaluation result
    """
    from .parser import parse
    ast = parse(code)
    evaluator = Evaluator(grammar)
    return evaluator.eval(ast)


if __name__ == "__main__":
    # Example usage
    print("Evaluator Examples:")
    print("=" * 80)
    
    examples = [
        ("Number", "42"),
        ("Boolean", "true"),
        ("Addition", "(+ 1 2)"),
        ("Subtraction", "(- 5 3)"),
        ("Multiplication", "(* 4 5)"),
        ("List", "[1 2 3]"),
        ("First", "(first [1 2 3])"),
        ("Take", "(take 2 [1 2 3 4])"),
        ("Reverse", "(reverse [1 2 3])"),
        ("Identity", "((λ x x) 42)"),
        ("Increment", "((λ x (+ x 1)) 10)"),
        ("Map", "(map (λ x (* x 2)) [1 2 3])"),
        ("Filter", "(filter (λ x (> x 2)) [1 2 3 4])"),
        ("Fold", "(fold (λ a (λ x (+ a x))) 0 [1 2 3])"),
        ("If true", "(if true 1 2)"),
        ("If false", "(if false 1 2)"),
    ]
    
    for name, code in examples:
        print(f"\n{name}: {code}")
        result = evaluate(code)
        print(f"Result: {result}")
