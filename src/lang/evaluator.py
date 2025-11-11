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
        """Initialize the evaluator with built-in functions."""
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
        
        # Numbers evaluate to themselves
        if isinstance(node, NumberNode):
            return node.value
        
        # Booleans evaluate to themselves
        elif isinstance(node, BooleanNode):
            return node.value
        
        # Variables: look up in environment
        elif isinstance(node, VariableNode):
            return env.get(node.name)
        
        # Lists: evaluate all elements
        elif isinstance(node, ListNode):
            return [self.eval(elem, env) for elem in node.elements]
        
        # Lambda: create closure
        elif isinstance(node, LambdaNode):
            return Closure(node.param, node.body, env)
        
        # If: evaluate condition, then appropriate branch
        elif isinstance(node, IfNode):
            condition = self.eval(node.condition, env)
            if not isinstance(condition, bool):
                raise EvaluationError(f"If condition must be boolean, got {type(condition).__name__}")
            if condition:
                return self.eval(node.then_expr, env)
            else:
                return self.eval(node.else_expr, env)
        
        # Application: apply function to arguments
        elif isinstance(node, ApplicationNode):
            func = self.eval(node.function, env)
            args = [self.eval(arg, env) for arg in node.arguments]
            return self._apply(func, args, env)
        
        else:
            raise EvaluationError(f"Unknown node type: {type(node).__name__}")
    
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
        # Built-in function
        if callable(func) and not isinstance(func, Closure):
            return func(self, *args)
        
        # User-defined function (closure)
        elif isinstance(func, Closure):
            # Apply arguments one at a time (currying)
            result = func
            for arg in args:
                if not isinstance(result, Closure):
                    raise EvaluationError(f"Too many arguments to function")
                
                # Create new environment with parameter binding
                new_env = result.env.extend(result.param, arg)
                # Evaluate body in new environment
                result = self.eval(result.body, new_env)
            
            return result
        
        else:
            raise EvaluationError(f"Cannot apply non-function: {type(func).__name__}")


def evaluate(code: str) -> Any:
    """
    Convenience function to parse and evaluate code.
    
    Args:
        code: Source code string
        
    Returns:
        Evaluation result
    """
    from .parser import parse
    ast = parse(code)
    evaluator = Evaluator()
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
