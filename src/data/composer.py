"""
Random Typed Program Generator (Composer)

This module generates random lambda expressions with type consistency.
Programs are generated according to a specified target depth and can be
constrained to produce a specific type.
"""

import random
from typing import Optional, List, Set
from dataclasses import dataclass

from lang.ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode
)
from lang.type_system import (
    Type, TypeVar, IntType, BoolType, ListType, FunctionType,
    INT, BOOL, list_of, func
)


@dataclass
class TypeContext:
    """Context for tracking available variables and their types during generation."""
    bindings: dict[str, Type]

    def __init__(self):
        self.bindings = {}

    def extend(self, name: str, type_: Type) -> 'TypeContext':
        """Create a new context with an additional binding."""
        new_ctx = TypeContext()
        new_ctx.bindings = self.bindings.copy()
        new_ctx.bindings[name] = type_
        return new_ctx

    def get(self, name: str) -> Optional[Type]:
        """Get the type of a variable."""
        return self.bindings.get(name)

    def get_vars_of_type(self, target_type: Type) -> List[str]:
        """Get all variables that match the target type."""
        return [name for name, t in self.bindings.items() if t == target_type]


class ProgramComposer:
    """
    Generates random well-typed programs.

    Uses a depth-based recursive generation strategy that ensures
    type consistency throughout the generated program.
    """

    def __init__(self, seed: int):
        """
        Initialize the composer with a random seed.

        Args:
            seed: Random seed for deterministic generation
        """
        self.rng = random.Random(seed)
        self.var_counter = 0

        # Built-in functions and their types
        self.builtins = self._get_builtin_types()

    def _get_builtin_types(self) -> dict[str, Type]:
        """Get types for built-in functions."""
        # Type variables for polymorphic functions
        t1 = TypeVar("t1")
        t2 = TypeVar("t2")

        builtins = {}

        # Arithmetic
        binary_int = func(INT, func(INT, INT))
        for op in ["+", "-", "*", "/", "%"]:
            builtins[op] = binary_int

        # Comparison
        int_cmp = func(INT, func(INT, BOOL))
        for op in ["<", ">"]:
            builtins[op] = int_cmp

        # Note: == is polymorphic, we'll handle it specially
        builtins["=="] = func(INT, func(INT, BOOL))  # Simplified to Int for now

        # Boolean
        binary_bool = func(BOOL, func(BOOL, BOOL))
        for op in ["and", "or"]:
            builtins[op] = binary_bool
        builtins["not"] = func(BOOL, BOOL)

        # Number predicates
        builtins["is_even"] = func(INT, BOOL)
        builtins["is_odd"] = func(INT, BOOL)

        # List operations (using concrete types, not polymorphic)
        builtins["singleton"] = func(INT, list_of(INT))
        builtins["cons"] = func(INT, func(list_of(INT), list_of(INT)))
        builtins["append"] = func(list_of(INT), func(INT, list_of(INT)))
        builtins["first"] = func(list_of(INT), INT)
        builtins["last"] = func(list_of(INT), INT)
        builtins["length"] = func(list_of(INT), INT)
        builtins["reverse"] = func(list_of(INT), list_of(INT))
        builtins["sum"] = func(list_of(INT), INT)
        builtins["product"] = func(list_of(INT), INT)
        builtins["max"] = func(list_of(INT), INT)
        builtins["min"] = func(list_of(INT), INT)

        return builtins

    def _fresh_var_name(self) -> str:
        """Generate a fresh variable name."""
        name = chr(ord('x') + (self.var_counter % 26))
        if self.var_counter >= 26:
            name += str(self.var_counter // 26)
        self.var_counter += 1
        return name

    def _sample_int(self) -> int:
        """Sample a random integer in valid range [0, 99]."""
        return self.rng.randint(0, 99)

    def _sample_bool(self) -> bool:
        """Sample a random boolean."""
        return self.rng.choice([True, False])

    def _concrete_type(self, type_: Optional[Type]) -> Type:
        """Convert type variables to concrete types."""
        if type_ is None:
            # Choose a random type
            return self.rng.choice([INT, BOOL, list_of(INT)])

        if isinstance(type_, TypeVar):
            # Replace type variables with concrete types
            return self.rng.choice([INT, BOOL, list_of(INT)])

        if isinstance(type_, ListType):
            return list_of(self._concrete_type(type_.elem_type))

        if isinstance(type_, FunctionType):
            return func(
                self._concrete_type(type_.param_type),
                self._concrete_type(type_.return_type)
            )

        return type_

    def generate(
        self,
        max_depth: int,
        target_type: Optional[Type] = None,
        context: Optional[TypeContext] = None
    ) -> ASTNode:
        """
        Generate a random well-typed program.

        Args:
            max_depth: Maximum depth of the expression tree
            target_type: Desired type of the generated expression
            context: Type context with available variables

        Returns:
            A well-typed AST node
        """
        if context is None:
            context = TypeContext()

        # Concretize the target type
        target_type = self._concrete_type(target_type)

        # Base case: depth 0 or force base case sometimes
        force_base = max_depth == 0 or (max_depth > 0 and self.rng.random() < 0.2)

        if force_base:
            return self._generate_base_case(target_type, context)

        # Recursive case: choose a construction strategy
        return self._generate_recursive(max_depth, target_type, context)

    def _generate_base_case(
        self,
        target_type: Type,
        context: TypeContext
    ) -> ASTNode:
        """Generate a base case (literal or variable)."""
        strategies = []

        # Try to use a variable if available
        if isinstance(target_type, (IntType, BoolType, ListType, FunctionType)):
            matching_vars = context.get_vars_of_type(target_type)
            if matching_vars:
                strategies.append(('var', matching_vars))

        # Literals
        if isinstance(target_type, IntType):
            strategies.append(('int_lit', None))
        elif isinstance(target_type, BoolType):
            strategies.append(('bool_lit', None))
        elif isinstance(target_type, ListType):
            # Empty lists get polymorphic types, but tests expect them at depth 0
            strategies.append(('empty_list', None))
        elif isinstance(target_type, FunctionType):
            strategies.append(('lambda', None))

        if not strategies:
            # Fallback: create appropriate literal
            if isinstance(target_type, IntType):
                return NumberNode(self._sample_int())
            elif isinstance(target_type, BoolType):
                return BooleanNode(self._sample_bool())
            elif isinstance(target_type, ListType):
                return ListNode([])
            else:
                # Must be a function type
                return self._generate_lambda(0, target_type, context)

        # Choose a strategy
        strategy, data = self.rng.choice(strategies)

        if strategy == 'var':
            return VariableNode(self.rng.choice(data))
        elif strategy == 'int_lit':
            return NumberNode(self._sample_int())
        elif strategy == 'bool_lit':
            return BooleanNode(self._sample_bool())
        elif strategy == 'empty_list':
            return ListNode([])
        elif strategy == 'singleton_list':
            # Generate a list with one element to force concrete type
            elem_type = data.elem_type
            elem = self.generate(0, elem_type, context)
            return ListNode([elem])
        elif strategy == 'lambda':
            return self._generate_lambda(0, target_type, context)

        raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_recursive(
        self,
        max_depth: int,
        target_type: Type,
        context: TypeContext
    ) -> ASTNode:
        """Generate a recursive case."""
        strategies = []

        # Type-specific strategies
        if isinstance(target_type, IntType):
            strategies.extend(['int_lit', 'arithmetic', 'list_agg', 'if_int'])
        elif isinstance(target_type, BoolType):
            strategies.extend(['bool_lit', 'comparison', 'bool_op', 'predicate', 'if_bool'])
        elif isinstance(target_type, ListType):
            strategies.extend(['list_lit', 'list_cons', 'list_transform', 'if_list'])
        elif isinstance(target_type, FunctionType):
            strategies.extend(['lambda', 'if_func'])

        # Add variable strategy if variables are available
        matching_vars = context.get_vars_of_type(target_type)
        if matching_vars:
            strategies.append('var')

        if not strategies:
            return self._generate_base_case(target_type, context)

        # Try strategies until one succeeds
        self.rng.shuffle(strategies)
        for strategy in strategies:
            try:
                if strategy == 'var':
                    return VariableNode(self.rng.choice(matching_vars))
                elif strategy == 'int_lit':
                    return NumberNode(self._sample_int())
                elif strategy == 'bool_lit':
                    return BooleanNode(self._sample_bool())
                elif strategy == 'arithmetic':
                    return self._generate_arithmetic(max_depth - 1, context)
                elif strategy == 'comparison':
                    return self._generate_comparison(max_depth - 1, context)
                elif strategy == 'bool_op':
                    return self._generate_bool_op(max_depth - 1, context)
                elif strategy == 'predicate':
                    return self._generate_predicate(max_depth - 1, context)
                elif strategy == 'list_agg':
                    return self._generate_list_aggregation(max_depth - 1, context)
                elif strategy == 'list_lit':
                    return self._generate_list_literal(max_depth - 1, target_type, context)
                elif strategy == 'list_cons':
                    return self._generate_list_construction(max_depth - 1, target_type, context)
                elif strategy == 'list_transform':
                    return self._generate_list_transform(max_depth - 1, target_type, context)
                elif strategy == 'lambda':
                    return self._generate_lambda(max_depth - 1, target_type, context)
                elif strategy == 'if_int':
                    return self._generate_if(max_depth - 1, INT, context)
                elif strategy == 'if_bool':
                    return self._generate_if(max_depth - 1, BOOL, context)
                elif strategy == 'if_list':
                    return self._generate_if(max_depth - 1, target_type, context)
                elif strategy == 'if_func':
                    return self._generate_if(max_depth - 1, target_type, context)
            except Exception:
                # Strategy failed, try next one
                continue

        # All strategies failed, use base case
        return self._generate_base_case(target_type, context)

    def _generate_arithmetic(self, max_depth: int, context: TypeContext) -> ASTNode:
        """Generate an arithmetic expression: (op x y)."""
        op = self.rng.choice(["+", "-", "*", "/", "%"])
        arg1 = self.generate(max_depth, INT, context)
        arg2 = self.generate(max_depth, INT, context)
        return ApplicationNode(VariableNode(op), [arg1, arg2])

    def _generate_comparison(self, max_depth: int, context: TypeContext) -> ASTNode:
        """Generate a comparison: (< x y) or (> x y)."""
        op = self.rng.choice(["<", ">", "=="])
        arg1 = self.generate(max_depth, INT, context)
        arg2 = self.generate(max_depth, INT, context)
        return ApplicationNode(VariableNode(op), [arg1, arg2])

    def _generate_bool_op(self, max_depth: int, context: TypeContext) -> ASTNode:
        """Generate a boolean operation: (and x y), (or x y), or (not x)."""
        if self.rng.random() < 0.3:
            # Unary not
            arg = self.generate(max_depth, BOOL, context)
            return ApplicationNode(VariableNode("not"), [arg])
        else:
            # Binary and/or
            op = self.rng.choice(["and", "or"])
            arg1 = self.generate(max_depth, BOOL, context)
            arg2 = self.generate(max_depth, BOOL, context)
            return ApplicationNode(VariableNode(op), [arg1, arg2])

    def _generate_predicate(self, max_depth: int, context: TypeContext) -> ASTNode:
        """Generate a predicate: (is_even x) or (is_odd x)."""
        pred = self.rng.choice(["is_even", "is_odd"])
        arg = self.generate(max_depth, INT, context)
        return ApplicationNode(VariableNode(pred), [arg])

    def _generate_list_aggregation(self, max_depth: int, context: TypeContext) -> ASTNode:
        """Generate list aggregation: (sum xs), (length xs), etc."""
        # Note: length and sum work on empty lists, but max/min/product need non-empty lists
        # For safety, always generate non-empty list literals for operations that need them
        op = self.rng.choice(["sum", "product", "max", "min", "length"])

        # For operations that need non-empty lists, ensure we create a list literal with elements
        if op in ["max", "min", "product"]:
            # Force a list literal with at least 1 element
            num_elems = self.rng.randint(1, 4)
            elements = [self.generate(max(0, max_depth - 1), INT, context) for _ in range(num_elems)]
            arg = ListNode(elements)
        else:
            arg = self.generate(max_depth, list_of(INT), context)

        return ApplicationNode(VariableNode(op), [arg])

    def _generate_list_literal(
        self,
        max_depth: int,
        target_type: Type,
        context: TypeContext
    ) -> ASTNode:
        """Generate a list literal: [e1 e2 ...]."""
        if not isinstance(target_type, ListType):
            raise ValueError("Target type must be ListType")

        elem_type = target_type.elem_type

        # Generate 1-4 elements (avoid empty lists to force concrete types)
        # Empty lists have polymorphic types
        num_elems = self.rng.randint(1, 4)
        elements = []
        for _ in range(num_elems):
            elem = self.generate(max_depth, elem_type, context)
            elements.append(elem)

        return ListNode(elements)

    def _generate_list_construction(
        self,
        max_depth: int,
        target_type: Type,
        context: TypeContext
    ) -> ASTNode:
        """Generate list construction: (cons x xs), (append xs x), etc."""
        if not isinstance(target_type, ListType):
            raise ValueError("Target type must be ListType")

        elem_type = target_type.elem_type

        # Only support Int lists for built-ins
        if elem_type != INT:
            return self._generate_list_literal(max_depth, target_type, context)

        op = self.rng.choice(["cons", "append", "singleton"])

        if op == "singleton":
            elem = self.generate(max_depth, INT, context)
            return ApplicationNode(VariableNode(op), [elem])
        elif op == "cons":
            elem = self.generate(max_depth, INT, context)
            lst = self.generate(max_depth, list_of(INT), context)
            return ApplicationNode(VariableNode(op), [elem, lst])
        elif op == "append":
            lst = self.generate(max_depth, list_of(INT), context)
            elem = self.generate(max_depth, INT, context)
            return ApplicationNode(VariableNode(op), [lst, elem])

        return self._generate_list_literal(max_depth, target_type, context)

    def _generate_list_transform(
        self,
        max_depth: int,
        target_type: Type,
        context: TypeContext
    ) -> ASTNode:
        """Generate list transformation: (reverse xs), etc."""
        if not isinstance(target_type, ListType):
            raise ValueError("Target type must be ListType")

        elem_type = target_type.elem_type

        # Only support Int lists for built-ins
        if elem_type != INT:
            return self._generate_list_literal(max_depth, target_type, context)

        op = self.rng.choice(["reverse"])
        lst = self.generate(max_depth, list_of(INT), context)
        return ApplicationNode(VariableNode(op), [lst])

    def _generate_lambda(
        self,
        max_depth: int,
        target_type: Type,
        context: TypeContext
    ) -> ASTNode:
        """Generate a lambda: (λ x body)."""
        if not isinstance(target_type, FunctionType):
            # Default to Int → Int
            target_type = func(INT, INT)

        # Target type is already concretized by generate(), so use it directly
        param_name = self._fresh_var_name()
        param_type = target_type.param_type
        return_type = target_type.return_type

        # Extend context with parameter
        new_context = context.extend(param_name, param_type)

        # Generate body - prefer using the parameter to constrain its type
        # This ensures type checker infers the correct parameter type
        if max_depth > 0 and self.rng.random() < 0.9:
            # Try to generate a body that uses the parameter
            # This constrains the parameter type
            body = self._generate_body_using_param(max_depth, param_name, param_type, return_type, new_context)
        else:
            # Generate any body of the right return type
            body = self.generate(max_depth, return_type, new_context)

        return LambdaNode(param_name, body)

    def _generate_body_using_param(
        self,
        max_depth: int,
        param_name: str,
        param_type: Type,
        return_type: Type,
        context: TypeContext
    ) -> ASTNode:
        """Generate a lambda body that uses the parameter."""
        # If param is Int and return is Int, apply arithmetic to constrain type
        if isinstance(param_type, IntType) and isinstance(return_type, IntType):
            if max_depth > 0 and self.rng.random() < 0.7:
                # Apply arithmetic operation to force Int type
                op = self.rng.choice(["+", "-", "*"])
                arg2 = self.generate(max_depth - 1, INT, context)
                return ApplicationNode(VariableNode(op), [VariableNode(param_name), arg2])
            else:
                # Just add 0 to force Int type
                return ApplicationNode(VariableNode("+"), [VariableNode(param_name), NumberNode(0)])

        # If param is Int and return is Bool, apply predicate
        if isinstance(param_type, IntType) and isinstance(return_type, BoolType):
            pred = self.rng.choice(["is_even", "is_odd"])
            return ApplicationNode(VariableNode(pred), [VariableNode(param_name)])

        # If param is List and return is Int, apply aggregation
        if isinstance(param_type, ListType) and isinstance(return_type, IntType):
            if param_type.elem_type == INT:
                op = self.rng.choice(["length", "sum"])
                return ApplicationNode(VariableNode(op), [VariableNode(param_name)])

        # If param is List and return is List, apply transformation
        if isinstance(param_type, ListType) and isinstance(return_type, ListType):
            if param_type == return_type and param_type.elem_type == INT:
                return ApplicationNode(VariableNode("reverse"), [VariableNode(param_name)])

        # If param is Bool and return is Bool, apply not twice (identity with type constraint)
        if isinstance(param_type, BoolType) and isinstance(return_type, BoolType):
            return ApplicationNode(VariableNode("not"), [
                ApplicationNode(VariableNode("not"), [VariableNode(param_name)])
            ])

        # Same types but not handled above - just return param (will get polymorphic type)
        if param_type == return_type:
            return VariableNode(param_name)

        # Fallback: generate arbitrary body (parameter won't be used)
        return self.generate(max_depth, return_type, context)

    def _generate_if(
        self,
        max_depth: int,
        target_type: Type,
        context: TypeContext
    ) -> ASTNode:
        """Generate an if expression: (if cond then else)."""
        condition = self.generate(max_depth, BOOL, context)
        then_expr = self.generate(max_depth, target_type, context)
        else_expr = self.generate(max_depth, target_type, context)
        return IfNode(condition, then_expr, else_expr)


def sample_program(
    seed: int,
    max_depth: int,
    target_type: Optional[Type] = None
) -> ASTNode:
    """
    Sample a random well-typed program.

    Args:
        seed: Random seed for reproducibility
        max_depth: Maximum depth of the expression tree
        target_type: Optional target type for the program

    Returns:
        A well-typed AST node

    Example:
        >>> from lang.type_system import INT, list_of
        >>> ast = sample_program(seed=42, max_depth=3, target_type=INT)
        >>> # ast is a well-typed program of type Int
    """
    composer = ProgramComposer(seed)
    return composer.generate(max_depth, target_type)