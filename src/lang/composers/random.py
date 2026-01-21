"""
Random Typed Program Generator (RandomComposer)

This module generates random lambda expressions with type consistency.
Programs are generated according to a specified target depth and can be
constrained to produce a specific type.

This is the original random generation strategy that samples uniformly
from all type-valid candidates.
"""

from typing import Optional

from .base import Composer
from ..grammar import Grammar
from ..ast_nodes import (
    ASTNode, VariableNode,
    LambdaNode, ApplicationNode, IfNode
)
from ..type_utils import (
    get_args,
    get_base_type,
    CallableOrig,
    TypeType,
    SubstitutionTable,
    substitute_type_vars,
    matchable
)


def min_depth_for_type(type_: TypeType) -> int:
    """
    Compute the minimum depth required to generate a value of this type.
    """
    base = get_base_type(type_)

    if type_ == int or type_ == bool:
        return 0
    if base == list:
        return 0
    if base == CallableOrig:
        return 1
    return 1


class RandomComposer(Composer):
    """
    Generates random well-typed programs.

    Uses a depth-based recursive generation strategy that ensures
    type consistency throughout the generated program. Handles type
    variables through unification via the matchable function.

    This composer uses uniform random sampling weighted by flat weights,
    which produces syntactically valid but often semantically degenerate
    programs.
    """

    @classmethod
    def get_name(cls) -> str:
        return "random"

    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None
    ) -> ASTNode:
        """
        Generate a random well-typed program.

        Args:
            target_type: The desired output type
            depth: Maximum remaining depth (0 = only literals/variables)
            context: Variable bindings in scope (name -> type)
            substitutions: Current type variable substitutions

        Returns:
            An AST node of the target type
        """
        if context is None:
            context = {}
        if substitutions is None:
            substitutions = SubstitutionTable()

        # Resolve target type through substitutions
        target = substitute_type_vars(target_type, substitutions)
        base_type = get_base_type(target)

        # Build list of all possible expressions with their weights
        candidates = []
        weights = []

        # Candidate 1: Literal (for atomic types and empty list for list types)
        if target == int or target == bool or base_type == list:
            candidates.append(('literal', None))
            weights.append(0.1)

        # Candidate 2: Each compatible variable from context
        for var_name, var_type in context.items():
            subs_copy = substitutions.copy()
            if matchable(var_type, target, subs_copy):
                candidates.append(('variable', (var_name, subs_copy)))
                weights.append(0.1)

        # Candidate 3: Lambda (for Callable types, only if depth > 0)
        if base_type == CallableOrig and depth > 0:
            candidates.append(('lambda', None))
            weights.append(0.2)

        # Candidate 4: If expression (only if depth > 0)
        # For Callable types, need depth > 1 so branches can use Lambda
        if depth > 0 and (base_type != CallableOrig or depth > 1):
            candidates.append(('if', None))
            weights.append(0.2)

        # Candidate 5: Each possible function application from grammar
        if depth > 0:
            matches = self.grammar.find_matching_functions(
                ret_type=target,
                substitutions=substitutions
            )
            for func_name, func_subs in matches:
                # Check if all argument types can be generated at depth-1
                func_info = self.grammar[func_name]
                can_generate = True
                for arg_type in func_info['arg_types']:
                    resolved_arg = substitute_type_vars(arg_type, func_subs)
                    if min_depth_for_type(resolved_arg) > depth - 1:
                        can_generate = False
                        break
                if can_generate:
                    candidates.append(('application', (func_name, func_subs)))
                    weights.append(0.2)

        # If no candidates, raise an error
        if not candidates:
            raise ValueError(f"Cannot generate expression of type {target_type} with depth {depth}")

        # Sample one candidate using weights
        chosen_idx = self.rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        expr_type, expr_data = candidates[chosen_idx]

        # Generate the chosen expression
        if expr_type == 'literal':
            return self._sample_literal(target, substitutions)

        elif expr_type == 'variable':
            var_name, new_subs = expr_data
            substitutions.update(new_subs)
            return VariableNode(var_name)

        elif expr_type == 'lambda':
            return self._generate_lambda(target, depth, context, substitutions)

        elif expr_type == 'if':
            return self._generate_if(target, depth, context, substitutions)

        elif expr_type == 'application':
            func_name, func_subs = expr_data
            return self._generate_application_for(func_name, func_subs, depth, context, substitutions)

        else:
            raise ValueError(f"Unknown expression type: {expr_type}")

    def _generate_lambda(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate a lambda expression.

        For Callable[[A1, A2, ...], R], generates a multi-argument lambda:
        (λ (x1 x2 ...) body)
        """
        target = substitute_type_vars(target_type, substitutions)
        type_args = get_args(target)

        if len(type_args) != 2:
            raise ValueError(f"Callable type must have exactly 2 elements: {target}")

        param_types_list, ret_type = type_args

        if not isinstance(param_types_list, list):
            raise ValueError(f"Expected parameter list in Callable, got {param_types_list}")

        # Generate parameter names and add to context
        new_context = context.copy()
        param_names = []

        for param_type in param_types_list:
            param_name = self._fresh_var_name()
            param_names.append(param_name)
            new_context[param_name] = param_type

        # Generate body with all parameters in scope
        body = self.generate(ret_type, depth - 1, new_context, substitutions)

        # Build a single multi-argument lambda
        return LambdaNode(param_names, body)

    def _generate_if(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate an if statement.

        Format: (if condition then_expr else_expr)
        Both branches must have the target type.
        """
        # Generate boolean condition
        condition = self.generate(bool, depth - 1, context, substitutions)

        # Generate then branch with target type
        then_expr = self.generate(target_type, depth - 1, context, substitutions)

        # Generate else branch with target type
        else_expr = self.generate(target_type, depth - 1, context, substitutions)

        return IfNode(condition, then_expr, else_expr)

    def _generate_application_for(
        self,
        func_name: str,
        func_subs: SubstitutionTable,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate a function application for a specific function.
        """
        # Instantiate any remaining free type variables
        func_info = self.grammar[func_name]
        for arg_type in func_info['arg_types']:
            func_subs = self._instantiate_free_types(arg_type, func_subs)
        func_subs = self._instantiate_free_types(func_info['ret_type'], func_subs)

        # Update our substitutions
        substitutions.update(func_subs)

        # Generate arguments
        args = []
        for arg_type in func_info['arg_types']:
            arg = self.generate(arg_type, depth - 1, context, substitutions)
            args.append(arg)

        return ApplicationNode(VariableNode(func_name), args)
