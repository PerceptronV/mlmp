"""
Type Checker for the Functional Programming Language

This module implements Hindley-Milner style type inference with
meaningful error messages.
"""

from typing import Dict, Set, Optional
from dataclasses import dataclass
from .ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode
)
from .type_system import (
    Type, TypeVar, IntType, BoolType, ListType, FunctionType,
    TypeScheme, TypeError, INT, BOOL, list_of, func
)


class TypeEnvironment:
    """Type environment for variable type bindings."""
    
    def __init__(self, parent: Optional['TypeEnvironment'] = None):
        """
        Initialize type environment.
        
        Args:
            parent: Parent environment for nested scopes
        """
        self.bindings: Dict[str, TypeScheme] = {}
        self.parent = parent
    
    def define(self, name: str, type_scheme: TypeScheme) -> None:
        """Define a variable with its type scheme."""
        self.bindings[name] = type_scheme
    
    def get(self, name: str) -> Optional[TypeScheme]:
        """Look up a variable's type scheme."""
        if name in self.bindings:
            return self.bindings[name]
        elif self.parent:
            return self.parent.get(name)
        return None
    
    def extend(self, name: str, type_scheme: TypeScheme) -> 'TypeEnvironment':
        """Create new environment with additional binding."""
        new_env = TypeEnvironment(parent=self)
        new_env.define(name, type_scheme)
        return new_env


class TypeChecker:
    """
    Type checker with Hindley-Milner type inference.
    
    Provides meaningful error messages for type errors.
    """
    
    def __init__(self):
        """Initialize the type checker."""
        self.type_var_counter = 0
        self.substitutions: Dict[TypeVar, Type] = {}
        self.global_env = self._create_global_type_environment()
    
    def _fresh_type_var(self) -> TypeVar:
        """Generate a fresh type variable."""
        var = TypeVar(f"t{self.type_var_counter}")
        self.type_var_counter += 1
        return var
    
    def _create_global_type_environment(self) -> TypeEnvironment:
        """Create the global type environment with built-in functions."""
        env = TypeEnvironment()
        
        # Type variables for polymorphic functions
        t1 = TypeVar("t1")
        t2 = TypeVar("t2")
        t3 = TypeVar("t3")
        
        # Arithmetic operators
        binary_int_op = TypeScheme(set(), func(INT, func(INT, INT)))
        env.define("+", binary_int_op)
        env.define("-", binary_int_op)
        env.define("*", binary_int_op)
        env.define("/", binary_int_op)
        env.define("%", binary_int_op)
        
        # Comparison operators
        int_comparison = TypeScheme(set(), func(INT, func(INT, BOOL)))
        env.define("<", int_comparison)
        env.define(">", int_comparison)
        
        # Equality (polymorphic)
        env.define("==", TypeScheme({t1}, func(t1, func(t1, BOOL))))
        
        # Boolean operators
        binary_bool_op = TypeScheme(set(), func(BOOL, func(BOOL, BOOL)))
        env.define("and", binary_bool_op)
        env.define("or", binary_bool_op)
        env.define("not", TypeScheme(set(), func(BOOL, BOOL)))
        
        # Number predicates
        env.define("is_even", TypeScheme(set(), func(INT, BOOL)))
        env.define("is_odd", TypeScheme(set(), func(INT, BOOL)))
        
        # List construction
        env.define("singleton", TypeScheme({t1}, func(t1, list_of(t1))))
        env.define("repeat", TypeScheme({t1}, func(t1, func(INT, list_of(t1)))))
        env.define("range", TypeScheme(set(), func(INT, func(INT, func(INT, list_of(INT))))))
        env.define("cons", TypeScheme({t1}, func(t1, func(list_of(t1), list_of(t1)))))
        
        # List combination
        env.define("append", TypeScheme({t1}, func(list_of(t1), func(t1, list_of(t1)))))
        env.define("concat", TypeScheme({t1}, func(list_of(t1), func(list_of(t1), list_of(t1)))))
        env.define("zip", TypeScheme({t1}, func(list_of(t1), func(list_of(t1), list_of(list_of(t1))))))
        
        # List access
        env.define("first", TypeScheme({t1}, func(list_of(t1), t1)))
        env.define("second", TypeScheme({t1}, func(list_of(t1), t1)))
        env.define("third", TypeScheme({t1}, func(list_of(t1), t1)))
        env.define("last", TypeScheme({t1}, func(list_of(t1), t1)))
        env.define("nth", TypeScheme({t1}, func(INT, func(list_of(t1), t1))))
        
        # List modification
        env.define("insert", TypeScheme({t1}, func(t1, func(INT, func(list_of(t1), list_of(t1))))))
        env.define("replace", TypeScheme({t1}, func(INT, func(t1, func(list_of(t1), list_of(t1))))))
        env.define("swap", TypeScheme({t1}, func(INT, func(INT, func(list_of(t1), list_of(t1))))))
        
        # List removal
        env.define("cut_idx", TypeScheme({t1}, func(INT, func(list_of(t1), list_of(t1)))))
        env.define("cut_val", TypeScheme({t1}, func(t1, func(list_of(t1), list_of(t1)))))
        env.define("cut_vals", TypeScheme({t1}, func(t1, func(list_of(t1), list_of(t1)))))
        env.define("drop", TypeScheme({t1}, func(INT, func(list_of(t1), list_of(t1)))))
        env.define("droplast", TypeScheme({t1}, func(INT, func(list_of(t1), list_of(t1)))))
        
        # List slicing
        env.define("take", TypeScheme({t1}, func(INT, func(list_of(t1), list_of(t1)))))
        env.define("takelast", TypeScheme({t1}, func(INT, func(list_of(t1), list_of(t1)))))
        env.define("slice", TypeScheme({t1}, func(INT, func(INT, func(list_of(t1), list_of(t1))))))
        env.define("cut_slice", TypeScheme({t1}, func(INT, func(INT, func(list_of(t1), list_of(t1))))))
        env.define("splice", TypeScheme({t1}, func(list_of(t1), func(INT, func(list_of(t1), list_of(t1))))))
        
        # Higher-order functions
        env.define("map", TypeScheme({t1, t2}, 
            func(func(t1, t2), func(list_of(t1), list_of(t2)))))
        env.define("mapi", TypeScheme({t1, t2},
            func(func(t1, func(INT, t2)), func(list_of(t1), list_of(t2)))))
        env.define("filter", TypeScheme({t1},
            func(func(t1, BOOL), func(list_of(t1), list_of(t1)))))
        env.define("filteri", TypeScheme({t1},
            func(func(INT, func(t1, BOOL)), func(list_of(t1), list_of(t1)))))
        env.define("fold", TypeScheme({t1, t2},
            func(func(t2, func(t1, t2)), func(t2, func(list_of(t1), t2)))))
        env.define("foldi", TypeScheme({t1, t2},
            func(func(t2, func(t1, func(INT, t2))), func(t2, func(list_of(t1), t2)))))
        
        # List queries
        env.define("is_in", TypeScheme({t1}, func(list_of(t1), func(t1, BOOL))))
        env.define("count", TypeScheme({t1}, func(func(t1, BOOL), func(list_of(t1), INT))))
        env.define("find", TypeScheme({t1}, func(func(t1, BOOL), func(list_of(t1), list_of(INT)))))
        
        # List aggregation
        env.define("length", TypeScheme({t1}, func(list_of(t1), INT)))
        env.define("max", TypeScheme(set(), func(list_of(INT), INT)))
        env.define("min", TypeScheme(set(), func(list_of(INT), INT)))
        env.define("product", TypeScheme(set(), func(list_of(INT), INT)))
        env.define("sum", TypeScheme(set(), func(list_of(INT), INT)))
        
        # List transformation
        env.define("unique", TypeScheme({t1}, func(list_of(t1), list_of(t1))))
        env.define("sort", TypeScheme({t1}, func(func(t1, INT), func(list_of(t1), list_of(t1)))))
        env.define("reverse", TypeScheme({t1}, func(list_of(t1), list_of(t1))))
        env.define("flatten", TypeScheme({t1}, func(list_of(list_of(t1)), list_of(t1))))
        env.define("group", TypeScheme({t1, t2}, func(func(t1, t2), func(list_of(t1), list_of(list_of(t1))))))
        
        return env
    
    def _apply_substitution(self, type_: Type) -> Type:
        """Apply current substitutions to a type."""
        return type_.substitute(self.substitutions)
    
    def _unify(self, type1: Type, type2: Type, context: str = "") -> None:
        """
        Unify two types, updating substitutions.
        
        Args:
            type1: First type
            type2: Second type
            context: Context for error messages
            
        Raises:
            TypeError: If types cannot be unified
        """
        # Apply existing substitutions
        type1 = self._apply_substitution(type1)
        type2 = self._apply_substitution(type2)
        
        # Same type
        if type1 == type2:
            return
        
        # Type variable unification
        if isinstance(type1, TypeVar):
            if type1.occurs_in(type2):
                raise TypeError(
                    f"Infinite type: {type1} occurs in {type2}",
                    context
                )
            self.substitutions[type1] = type2
            return
        
        if isinstance(type2, TypeVar):
            if type2.occurs_in(type1):
                raise TypeError(
                    f"Infinite type: {type2} occurs in {type1}",
                    context
                )
            self.substitutions[type2] = type1
            return
        
        # List type unification
        if isinstance(type1, ListType) and isinstance(type2, ListType):
            self._unify(type1.elem_type, type2.elem_type, context)
            return
        
        # Function type unification
        if isinstance(type1, FunctionType) and isinstance(type2, FunctionType):
            self._unify(type1.param_type, type2.param_type, context)
            self._unify(type1.return_type, type2.return_type, context)
            return
        
        # Type mismatch
        raise TypeError(
            f"Cannot unify {type1} with {type2}",
            context
        )
    
    def _instantiate(self, scheme: TypeScheme) -> Type:
        """Instantiate a type scheme with fresh type variables."""
        fresh_vars = {var: self._fresh_type_var() for var in scheme.quantified}
        return scheme.instantiate(fresh_vars)
    
    def _generalize(self, type_: Type, env: TypeEnvironment) -> TypeScheme:
        """
        Generalize a type to a type scheme.
        
        Quantifies over type variables not free in the environment.
        """
        # Get free type variables in environment
        env_free_vars: Set[TypeVar] = set()
        for scheme in env.bindings.values():
            env_free_vars |= scheme.type.free_type_vars() - scheme.quantified
        
        # Quantify over variables free in type but not in environment
        type_ = self._apply_substitution(type_)
        quantified = type_.free_type_vars() - env_free_vars
        
        return TypeScheme(quantified, type_)
    
    def check(self, node: ASTNode, env: TypeEnvironment = None) -> Type:
        """
        Type check an AST node and return its type.
        
        Args:
            node: AST node to type check
            env: Type environment
            
        Returns:
            The type of the node
            
        Raises:
            TypeError: If type checking fails
        """
        if env is None:
            env = self.global_env
        
        try:
            # Numbers have type Int
            if isinstance(node, NumberNode):
                return INT
            
            # Booleans have type Bool
            elif isinstance(node, BooleanNode):
                return BOOL
            
            # Variables: look up in environment
            elif isinstance(node, VariableNode):
                scheme = env.get(node.name)
                if scheme is None:
                    raise TypeError(
                        f"Undefined variable '{node.name}'",
                        f"variable {node.name}"
                    )
                return self._instantiate(scheme)
            
            # Lists: check all elements have same type
            elif isinstance(node, ListNode):
                if not node.elements:
                    # Empty list has polymorphic type [t]
                    return list_of(self._fresh_type_var())
                
                # Check first element
                elem_type = self.check(node.elements[0], env)
                
                # Check remaining elements match
                for i, elem in enumerate(node.elements[1:], 1):
                    elem_i_type = self.check(elem, env)
                    try:
                        self._unify(elem_type, elem_i_type, f"list element {i}")
                    except TypeError as e:
                        raise TypeError(
                            f"List elements have incompatible types: "
                            f"element 0 has type {elem_type}, "
                            f"but element {i} has type {elem_i_type}",
                            f"list literal"
                        )
                
                return list_of(self._apply_substitution(elem_type))
            
            # Lambda: create function type
            elif isinstance(node, LambdaNode):
                # Create fresh type variable for parameter
                param_type = self._fresh_type_var()
                
                # Extend environment with parameter binding
                param_scheme = TypeScheme(set(), param_type)
                new_env = env.extend(node.param, param_scheme)
                
                # Check body
                body_type = self.check(node.body, new_env)
                
                # Return function type
                func_type = func(param_type, body_type)
                return self._apply_substitution(func_type)
            
            # If: check condition is Bool, branches have same type
            elif isinstance(node, IfNode):
                cond_type = self.check(node.condition, env)
                
                # Condition must be Bool
                try:
                    self._unify(cond_type, BOOL, "if condition")
                except TypeError:
                    raise TypeError(
                        f"If condition must have type Bool, but has type {cond_type}",
                        "if expression"
                    )
                
                # Check branches
                then_type = self.check(node.then_expr, env)
                else_type = self.check(node.else_expr, env)
                
                # Branches must have same type
                try:
                    self._unify(then_type, else_type, "if branches")
                except TypeError:
                    raise TypeError(
                        f"If branches must have the same type: "
                        f"then branch has type {then_type}, "
                        f"but else branch has type {else_type}",
                        "if expression"
                    )
                
                return self._apply_substitution(then_type)
            
            # Application: check function and argument types
            elif isinstance(node, ApplicationNode):
                func_type = self.check(node.function, env)
                
                # Apply arguments one by one (currying)
                result_type = func_type
                for i, arg in enumerate(node.arguments):
                    arg_type = self.check(arg, env)
                    
                    # Result type should be a function
                    return_type = self._fresh_type_var()
                    expected_func_type = func(arg_type, return_type)
                    
                    try:
                        self._unify(result_type, expected_func_type, f"application argument {i}")
                    except TypeError:
                        result_type = self._apply_substitution(result_type)
                        arg_type = self._apply_substitution(arg_type)
                        
                        if not isinstance(result_type, FunctionType):
                            raise TypeError(
                                f"Cannot apply non-function type {result_type}",
                                f"application"
                            )
                        
                        raise TypeError(
                            f"Type mismatch in application: "
                            f"function expects {result_type.param_type}, "
                            f"but argument has type {arg_type}",
                            f"application argument {i}"
                        )
                    
                    result_type = return_type
                
                return self._apply_substitution(result_type)
            
            else:
                raise TypeError(f"Unknown node type: {type(node).__name__}")
        
        except TypeError:
            raise
        except Exception as e:
            raise TypeError(f"Internal type checking error: {e}")
    
    def check_program(self, node: ASTNode) -> Type:
        """
        Type check a complete program.
        
        Args:
            node: Program AST
            
        Returns:
            The type of the program
        """
        self.substitutions = {}
        self.type_var_counter = 0
        return self.check(node, self.global_env)


def type_check(code: str) -> Type:
    """
    Convenience function to type check a program.
    
    Args:
        code: Source code
        
    Returns:
        The type of the program
    """
    from .parser import parse
    ast = parse(code)
    checker = TypeChecker()
    return checker.check_program(ast)


if __name__ == "__main__":
    print("Type Checker Examples:")
    print("=" * 80)
    
    examples = [
        ("Number", "42"),
        ("Boolean", "true"),
        ("List", "[1 2 3]"),
        ("Identity", "(λ x x)"),
        ("Increment", "(λ x (+ x 1))"),
        ("Map", "(map (λ x (* x 2)) [1 2 3])"),
        ("If", "(if true 1 2)"),
    ]
    
    for name, code in examples:
        print(f"\n{name}: {code}")
        try:
            type_ = type_check(code)
            print(f"Type: {type_}")
        except TypeError as e:
            print(f"Type Error: {e}")

