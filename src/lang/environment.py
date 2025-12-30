"""
Environment for variable scoping and closures.

This module provides the Environment class which manages variable bindings
and lexical scoping for the evaluator.
"""

from typing import Any, Dict, Optional, List


class Environment:
    """
    Environment for managing variable bindings with lexical scoping.
    
    Supports nested scopes via parent environments, enabling proper
    closure behaviour for lambda functions.
    """
    
    def __init__(self, parent: Optional['Environment'] = None):
        """
        Initialise an environment.
        
        Args:
            parent: Parent environment for nested scopes (None for global scope)
        """
        self.bindings: Dict[str, Any] = {}
        self.parent = parent
    
    def define(self, name: str, value: Any) -> None:
        """
        Define a new variable binding in this environment.
        
        Args:
            name: Variable name
            value: Value to bind
        """
        self.bindings[name] = value
    
    def get(self, name: str) -> Any:
        """
        Look up a variable binding.
        
        Args:
            name: Variable name to look up
            
        Returns:
            The value bound to the variable
            
        Raises:
            NameError: If the variable is not found
        """
        if name in self.bindings:
            return self.bindings[name]
        elif self.parent is not None:
            return self.parent.get(name)
        else:
            raise NameError(f"Undefined variable: {name}")
    
    def set(self, name: str, value: Any) -> None:
        """
        Update an existing variable binding.
        
        Args:
            name: Variable name
            value: New value
            
        Raises:
            NameError: If the variable is not found
        """
        if name in self.bindings:
            self.bindings[name] = value
        elif self.parent is not None:
            self.parent.set(name, value)
        else:
            raise NameError(f"Undefined variable: {name}")
    
    def extend(self, name: str, value: Any) -> 'Environment':
        """
        Create a new environment extending this one with a new binding.
        
        This is useful for function application.
        
        Args:
            name: Variable name to bind
            value: Value to bind
            
        Returns:
            New environment with the binding
        """
        new_env = Environment(parent=self)
        new_env.define(name, value)
        return new_env
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        bindings_str = ", ".join(f"{k}={v}" for k, v in self.bindings.items())
        return f"Environment({{{bindings_str}}})"


class Closure:
    """
    Represents a closure (lambda function with captured environment).

    A closure combines:
    - Parameter names - always a list of strings (even for single-parameter functions)
    - Function body (AST node)
    - Captured environment for free variables
    """

    def __init__(self, param: List[str], body: Any, env: Environment):
        """
        Create a closure.

        Args:
            param: List of parameter names (length 1 for single-parameter functions)
            body: Function body (AST node)
            env: Captured environment
        """
        self.param = param
        self.body = body
        self.env = env

    def __repr__(self) -> str:
        """String representation."""
        params_str = " ".join(self.param)
        return f"<closure λ{params_str}>"

    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()


if __name__ == "__main__":
    # Example usage
    print("Environment Example:")
    print("=" * 60)
    
    # Global environment
    global_env = Environment()
    global_env.define("x", 10)
    global_env.define("y", 20)
    print(f"Global: {global_env}")
    print(f"x = {global_env.get('x')}")
    print(f"y = {global_env.get('y')}")
    
    # Nested environment
    local_env = global_env.extend("z", 30)
    print(f"\nLocal: {local_env}")
    print(f"z = {local_env.get('z')}")
    print(f"x (from parent) = {local_env.get('x')}")
    
    # Undefined variable
    try:
        local_env.get("undefined")
    except NameError as e:
        print(f"\nError: {e}")
    
    # Closure example
    print("\nClosure Example:")
    closure = Closure("x", "body", global_env)
    print(f"Closure: {closure}")

