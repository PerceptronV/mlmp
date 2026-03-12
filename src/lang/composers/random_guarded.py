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
from .utils.guard import StrategyGuard, ApplicationContext
from .utils.strategies import StrategyType


def min_depth_for_type(type_: TypeType, literal_blocked: bool = False) -> int:
    """
    Compute the minimum depth required to generate a value of this type.
    
    Args:
        type_: The type to generate
        literal_blocked: If True, literals are not available (e.g., due to guard rules)
    """
    base = get_base_type(type_)

    if type_ == int or type_ == bool:
        # If literal is blocked, need depth >= 1 for application/variable
        return 1 if literal_blocked else 0
    if base == list:
        return 1 if literal_blocked else 0
    if base == CallableOrig:
        return 1
    return 1


class RandomGuardedComposer(Composer):
    """
    Generates random well-typed programs with guard rules.

    Uses a depth-based recursive generation strategy that ensures
    type consistency throughout the generated program. Handles type
    variables through unification via the matchable function.

    This composer uses uniform random sampling weighted by flat weights,
    which produces syntactically valid but often semantically degenerate
    programs. Guards prevent trivial generation strategies like generating
    literal booleans for if-conditions or literal numbers for function arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard = StrategyGuard()

    @classmethod
    def get_name(cls) -> str:
        return "random_guarded"

    def _should_block(self, strategy_type: str, app_context: Optional[ApplicationContext]) -> bool:
        """Check if a strategy should be blocked by guard rules."""
        if app_context is None:
            return False
        return self.guard.should_block_strategy(
            app_context.fn_name,
            app_context.arg_pos,
            strategy_type,
            app_context.other_strategies
        )

    def _get_strategy_type(self, node: ASTNode) -> str:
        """Determine the strategy type of a generated AST node."""
        from ..ast_nodes import NumberNode, BooleanNode, ListNode

        if isinstance(node, VariableNode):
            return StrategyType.VARIABLE
        elif isinstance(node, LambdaNode):
            # Check if it's an identity lambda (λ x x)
            if len(node.param) == 1 and isinstance(node.body, VariableNode):
                if node.body.name == node.param[0]:
                    return StrategyType.IDENTITY
            return StrategyType.LAMBDA
        elif isinstance(node, IfNode):
            return StrategyType.IF
        elif isinstance(node, ApplicationNode):
            # Extract function name from application
            if isinstance(node.function, VariableNode):
                func_name = node.function.name
                # Return serialized format for application strategies
                return f"apply:{func_name}"
            return StrategyType.APPLY
        elif isinstance(node, (NumberNode, BooleanNode, ListNode)):
            return StrategyType.LITERAL
        else:
            return StrategyType.LITERAL

    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """
        Generate a random well-typed program.

        Args:
            target_type: The desired output type
            depth: Maximum remaining depth (0 = only literals/variables)
            context: Variable bindings in scope (name -> type)
            substitutions: Current type variable substitutions
            app_context: Application context for guard rules (if generating function argument)

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
            strategy_type = StrategyType.LITERAL
            if not self._should_block(strategy_type, app_context):
                candidates.append(('literal', None))
                weights.append(0.1)

        # Candidate 2: Each compatible variable from context
        for var_name, var_type in context.items():
            subs_copy = substitutions.copy()
            if matchable(var_type, target, subs_copy):
                strategy_type = StrategyType.VARIABLE
                if not self._should_block(strategy_type, app_context):
                    candidates.append(('variable', (var_name, subs_copy)))
                    weights.append(0.1)

        # Candidate 3: Lambda (for Callable types, only if depth > 0)
        if base_type == CallableOrig and depth > 0:
            strategy_type = StrategyType.LAMBDA
            if not self._should_block(strategy_type, app_context):
                candidates.append(('lambda', None))
                weights.append(0.2)

        # Candidate 4: If expression (only if depth > 1)
        # Need depth > 1 because condition generation needs depth >= 1
        # (guard blocks literal bool for if-conditions, so need enough depth for non-literal bool)
        # For Callable types, need depth > 2 so branches can use Lambda
        if depth > 1 and (base_type != CallableOrig or depth > 2):
            strategy_type = StrategyType.IF
            if not self._should_block(strategy_type, app_context):
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
                # considering guard rules that might block literals
                func_info = self.grammar[func_name]
                num_args = len(func_info['arg_types'])
                can_generate = True
                for arg_pos, arg_type in enumerate(func_info['arg_types']):
                    resolved_arg = substitute_type_vars(arg_type, func_subs)
                    # Check if literal would be blocked for this argument
                    arg_ctx = ApplicationContext(func_name, arg_pos, num_args, [None] * num_args)
                    literal_blocked = self.guard.should_block_strategy(
                        func_name, arg_pos, StrategyType.LITERAL, arg_ctx.other_strategies
                    )
                    if min_depth_for_type(resolved_arg, literal_blocked) > depth - 1:
                        can_generate = False
                        break
                if can_generate:
                    strategy_type = f"apply:{func_name}"
                    if not self._should_block(strategy_type, app_context):
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

        For Callable[[A1, A2, ...], R], generates a multi-parameter lambda:
        (λ (x1 x2 ...) body)
        """
        target = substitute_type_vars(target_type, substitutions)
        type_args = get_args(target)

        if len(type_args) != 2:
            raise ValueError(f"Callable type must have exactly 2 elements: {target}")

        param_types_list, ret_type = type_args

        if not isinstance(param_types_list, list):
            raise ValueError(f"Expected parameter list in Callable, got {param_types_list}")

        # Generate nested lambdas
        new_context = context.copy()
        param_names = []

        for param_type in param_types_list:
            param_name = self._fresh_var_name()
            param_names.append(param_name)
            new_context[param_name] = param_type

        # Generate body with all parameters in scope
        body = self.generate(ret_type, depth - 1, new_context, substitutions)

        # Build single multi-parameter lambda
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
        # Create application context for 'if' function arguments
        # if has 3 arguments: condition (pos 0), then_expr (pos 1), else_expr (pos 2)

        # Generate boolean condition (argument 0)
        cond_ctx = ApplicationContext.for_function('if', 3, 0)
        condition = self.generate(bool, depth - 1, context, substitutions, cond_ctx)

        # Track the strategy used for condition
        cond_strategy = self._get_strategy_type(condition)
        cond_ctx = cond_ctx.with_strategy(0, cond_strategy)

        # Generate then branch with target type (argument 1)
        then_ctx = ApplicationContext('if', 1, 3, [cond_strategy, None, None])
        then_expr = self.generate(target_type, depth - 1, context, substitutions, then_ctx)

        # Track the strategy used for then branch
        then_strategy = self._get_strategy_type(then_expr)

        # Generate else branch with target type (argument 2)
        else_ctx = ApplicationContext('if', 2, 3, [cond_strategy, then_strategy, None])
        else_expr = self.generate(target_type, depth - 1, context, substitutions, else_ctx)

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

        # Generate arguments with guard context
        args = []
        num_args = len(func_info['arg_types'])
        arg_strategies = [None] * num_args

        for arg_pos, arg_type in enumerate(func_info['arg_types']):
            # Create application context for this argument
            arg_ctx = ApplicationContext(func_name, arg_pos, num_args, arg_strategies.copy())

            # Generate argument with guard rules applied
            arg = self.generate(arg_type, depth - 1, context, substitutions, arg_ctx)
            args.append(arg)

            # Track what strategy was used for this argument
            arg_strategies[arg_pos] = self._get_strategy_type(arg)

        return ApplicationNode(VariableNode(func_name), args)
