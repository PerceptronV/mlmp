"""
Type Checker with Hindley-Milner Type Inference

This module implements type inference for the functional programming language.
It uses Algorithm W to infer types for lambda expressions and check type
consistency throughout the program.
"""

from typing import Optional, TypeVar
from .ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode
)
from .grammar import Grammar, DefaultGrammar
from .parser import parse
from .type_utils import (
    TypeType, Callable, CallableOrig, SubstitutionTable,
    substitute_type_vars, matchable, get_origin, get_args, isvariable
)


class TypeCheckError(Exception):
    """Exception raised during type checking."""
    pass


class TypeChecker:
    """
    Type checker with Hindley-Milner type inference.

    Infers types for expressions including lambda functions using
    Algorithm W with unification.
    """

    def __init__(self, grammar: Grammar = DefaultGrammar):
        """Initialise the type checker with a grammar."""
        self.grammar = grammar
        self.type_var_counter = 0

    def _fresh_type_var(self) -> TypeVar:
        """Generate a fresh type variable."""
        name = f"T{self.type_var_counter}"
        self.type_var_counter += 1
        return TypeVar(name)

    def _instantiate_type(self, type_: TypeType, mapping: dict[TypeVar, TypeVar] = None) -> TypeType:
        """
        Instantiate a type by replacing all type variables with fresh ones.

        This is used to ensure polymorphic functions get fresh type variables
        on each use.

        Args:
            type_: The type to instantiate
            mapping: Mapping from old type variables to new ones

        Returns:
            Type with fresh type variables
        """
        if mapping is None:
            mapping = {}

        # If it's a type variable, replace it with a fresh one
        if isvariable(type_):
            if type_ not in mapping:
                mapping[type_] = self._fresh_type_var()
            return mapping[type_]

        origin = get_origin(type_)

        # If no origin, it's a concrete type (int, bool, etc.)
        if origin is None:
            return type_

        args = get_args(type_)

        # Handle Callable specially
        if origin == CallableOrig and isinstance(args[0], list):
            param_types = [self._instantiate_type(p, mapping) for p in args[0]]
            ret_type = self._instantiate_type(args[1], mapping)
            return Callable[param_types, ret_type]

        # Handle other generic types
        if args:
            new_args = tuple(self._instantiate_type(a, mapping) for a in args)
            # Python 3.10 compatible: use __getitem__ with tuple
            return origin[new_args]

        return type_

    def infer(
        self,
        node: ASTNode,
        context: dict[str, TypeType] = None,
        substitutions: SubstitutionTable = None
    ) -> tuple[TypeType, SubstitutionTable]:
        """
        Infer the type of an AST node.

        Args:
            node: AST node to type check
            context: Type environment mapping variables to types
            substitutions: Current type variable substitutions

        Returns:
            Tuple of (inferred_type, updated_substitutions)
        """
        if context is None:
            context = {}
        if substitutions is None:
            substitutions = SubstitutionTable()

        # Numbers have type int
        if isinstance(node, NumberNode):
            return int, substitutions

        # Booleans have type bool
        elif isinstance(node, BooleanNode):
            return bool, substitutions

        # Variables: look up in context
        elif isinstance(node, VariableNode):
            if node.name not in context:
                raise TypeCheckError(f"Undefined variable: {node.name}")
            var_type = context[node.name]
            # Instantiate only for polymorphic grammar functions (not lambda parameters)
            if node.name in self.grammar.functions:
                var_type = self._instantiate_type(var_type)
            return var_type, substitutions

        # Lists: infer element type and construct list[T]
        elif isinstance(node, ListNode):
            if not node.elements:
                # Empty list has type list[T] where T is a fresh type variable
                elem_type = self._fresh_type_var()
                return list[elem_type], substitutions

            # Infer type of first element
            elem_type, subs = self.infer(node.elements[0], context, substitutions)

            # Check all other elements have the same type
            for elem in node.elements[1:]:
                elem_t, subs = self.infer(elem, context, subs)
                if not matchable(elem_type, elem_t, subs, strict=False):
                    raise TypeCheckError(
                        f"List elements have inconsistent types: {elem_type} vs {elem_t}"
                    )

            return list[substitute_type_vars(elem_type, subs)], subs

        # Lambda: introduce type variable(s) for parameter(s), infer body type
        elif isinstance(node, LambdaNode):
            # Params are always a list (even for single-parameter lambdas)
            param_types = [self._fresh_type_var() for _ in node.param]
            new_context = context.copy()
            for param, param_type in zip(node.param, param_types):
                new_context[param] = param_type

            body_type, subs = self.infer(node.body, new_context, substitutions)

            # Resolve all parameter types and body type through substitutions
            resolved_params = [substitute_type_vars(pt, subs) for pt in param_types]
            resolved_body = substitute_type_vars(body_type, subs)

            # Return Callable[[param1_type, ...], body_type]
            return Callable[resolved_params, resolved_body], subs

        # If: check condition is bool, branches have same type
        elif isinstance(node, IfNode):
            cond_type, subs = self.infer(node.condition, context, substitutions)

            # Condition must be boolean
            if not matchable(cond_type, bool, subs, strict=False):
                raise TypeCheckError(
                    f"If condition must be boolean, got {cond_type}"
                )

            # Infer types of both branches
            then_type, subs = self.infer(node.then_expr, context, subs)
            else_type, subs = self.infer(node.else_expr, context, subs)

            # Branches must have same type
            if not matchable(then_type, else_type, subs, strict=False):
                raise TypeCheckError(
                    f"If branches have different types: {then_type} vs {else_type}"
                )

            return substitute_type_vars(then_type, subs), subs

        # Application: infer function type, check argument types
        elif isinstance(node, ApplicationNode):
            func_type, subs = self.infer(node.function, context, substitutions)

            # Infer argument types
            arg_types = []
            for arg in node.arguments:
                arg_t, subs = self.infer(arg, context, subs)
                arg_types.append(arg_t)

            # Apply function type to arguments
            result_type = func_type
            for arg_type in arg_types:
                result_type, subs = self._apply_function_type(result_type, arg_type, subs)

            return result_type, subs

        else:
            raise TypeCheckError(f"Unknown node type: {type(node).__name__}")

    def _apply_function_type(
        self,
        func_type: TypeType,
        arg_type: TypeType,
        substitutions: SubstitutionTable
    ) -> tuple[TypeType, SubstitutionTable]:
        """
        Apply a function type to an argument type.

        Args:
            func_type: Type of the function (should be Callable)
            arg_type: Type of the argument
            substitutions: Current substitutions

        Returns:
            Tuple of (return_type, updated_substitutions)
        """
        # Resolve function type through substitutions
        func_type = substitute_type_vars(func_type, substitutions)

        # Check if function type is callable
        origin = get_origin(func_type)
        if origin != CallableOrig:
            # Maybe it's a type variable - create a fresh callable type
            if isvariable(func_type):
                ret_type = self._fresh_type_var()
                expected_func_type = Callable[[arg_type], ret_type]
                if not matchable(func_type, expected_func_type, substitutions, strict=False):
                    raise TypeCheckError(
                        f"Cannot apply non-function type: {func_type}"
                    )
                return ret_type, substitutions
            else:
                raise TypeCheckError(
                    f"Cannot apply non-function type: {func_type}"
                )

        # Extract parameter and return types
        args = get_args(func_type)
        if len(args) != 2:
            raise TypeCheckError(f"Invalid function type: {func_type}")

        param_types, ret_type = args
        if not isinstance(param_types, list) or len(param_types) == 0:
            raise TypeCheckError(f"Invalid function type: {func_type}")

        # Match first parameter with argument
        # Use strict=False to allow unifying type variables with parameterised types
        expected_param = param_types[0]
        if not matchable(expected_param, arg_type, substitutions, strict=False):
            raise TypeCheckError(
                f"Type mismatch: expected {expected_param}, got {arg_type}"
            )

        # If more parameters remain, return curried function
        if len(param_types) > 1:
            remaining_params = param_types[1:]
            return Callable[remaining_params, ret_type], substitutions
        else:
            # All parameters consumed, return result type
            return substitute_type_vars(ret_type, substitutions), substitutions

    def check(self, node: ASTNode) -> TypeType:
        """
        Type check an AST node and return its type.

        Args:
            node: AST node to type check

        Returns:
            The inferred type
        """
        # Add built-in functions to context
        context = {}
        for name, info in self.grammar.functions.items():
            arg_types = list(info['arg_types'])
            ret_type = info['ret_type']
            if len(arg_types) > 0:
                context[name] = Callable[arg_types, ret_type]
            else:
                context[name] = ret_type

        inferred_type, subs = self.infer(node, context)
        return substitute_type_vars(inferred_type, subs)


def format_type(type_: TypeType) -> str:
    """
    Format a type for display.

    Args:
        type_: Type to format

    Returns:
        Human-readable type string
    """
    if type_ == int:
        return "Int"
    elif type_ == bool:
        return "Bool"
    elif isvariable(type_):
        return str(type_)

    origin = get_origin(type_)

    # Check if the type itself is list or Callable origin
    if type_ == list or origin == list:
        args = get_args(type_)
        if args:
            elem_type = format_type(args[0])
            return f"[{elem_type}]"
        else:
            return "[?]"

    elif type_ == CallableOrig or origin == CallableOrig:
        args = get_args(type_)
        if len(args) == 2:
            param_types, ret_type = args
            if isinstance(param_types, list):
                param_strs = [format_type(p) for p in param_types]
                ret_str = format_type(ret_type)
                # Add parentheses around callable parameters
                formatted_params = []
                for p_str, p_type in zip(param_strs, param_types):
                    p_origin = get_origin(p_type)
                    if p_type == CallableOrig or p_origin == CallableOrig:
                        formatted_params.append(f"({p_str})")
                    else:
                        formatted_params.append(p_str)
                return " → ".join(formatted_params + [ret_str])
        return "Function"

    else:
        return str(type_)


def type_check(code: str, grammar: Grammar = DefaultGrammar) -> str:
    """
    Type check a program and return its type as a string.

    Args:
        code: Source code string
        grammar: Grammar to use for built-in functions

    Returns:
        Type string (e.g., "Int → Int")

    Example:
        >>> type_check("(λ x (+ x 1))")
        "Int → Int"
    """
    ast = parse(code)
    checker = TypeChecker(grammar)
    inferred_type = checker.check(ast)
    return format_type(inferred_type)


if __name__ == "__main__":
    # Example usage
    print("Type Checker Examples:")
    print("=" * 80)

    examples = [
        ("Number", "42"),
        ("Boolean", "true"),
        ("Identity", "(λ x (x))"),
        ("Increment", "(λ x (+ x 1))"),
        ("Add", "(λ x (λ y (+ x y)))"),
        ("If", "(if true 1 2)"),
        ("List", "[1 2 3]"),
        ("Map", "(λ f (λ xs (map f xs)))"),
    ]

    for name, code in examples:
        print(f"\n{name}: {code}")
        try:
            result = type_check(code)
            print(f"Type: {result}")
        except Exception as e:
            print(f"Error: {e}")
