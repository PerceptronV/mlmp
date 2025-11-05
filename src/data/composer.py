"""
Random typed program sampler for the lambda language.

Generates well-typed random ASTs defined in `src/lang`, parameterized by
an RNG seed and a maximum program depth. The sampler respects the language's
type system and built-in function signatures.

Primary entrypoint:
    sample_program(seed: int, max_depth: int, target_type: Optional[Type]) -> ASTNode

Notes:
- The generator uses Hindley–Milner-style type representations from `lang.type_system`.
- Built-in function type schemes come from `lang.type_checker.TypeChecker`.
- For polymorphic functions, type variables are instantiated to random concrete
  types (Int, Bool, list types, and occasionally function types) before sampling
  arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import random

# Import from sibling package `lang` (assumes project adds src/ to sys.path)
from lang.ast_nodes import (
    ASTNode,
    NumberNode,
    BooleanNode,
    VariableNode,
    LambdaNode,
    ApplicationNode,
    ListNode,
    IfNode,
)
from lang.type_checker import TypeChecker
from lang.type_system import (
    Type,
    TypeVar,
    IntType,
    BoolType,
    ListType,
    FunctionType,
    TypeScheme,
    INT,
    BOOL,
    list_of,
    func,
)


@dataclass
class GenerationEnv:
    """Tracks in-scope variables and their (monomorphic) types while generating."""
    var_types: Dict[str, Type]

    def with_binding(self, name: str, type_: Type) -> "GenerationEnv":
        new_bindings = dict(self.var_types)
        new_bindings[name] = type_
        return GenerationEnv(new_bindings)


class Sampler:
    """Typed random AST sampler driven by the language's type system."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.type_checker = TypeChecker()
        # Built-in function type schemes available at the top-level
        self.builtins: Dict[str, TypeScheme] = dict(self.type_checker.global_env.bindings)
        self.param_counter = 0

    # --------------------------- Public API ---------------------------
    def generate(self, max_depth: int, target_type: Optional[Type] = None) -> ASTNode:
        if max_depth < 0:
            max_depth = 0
        if target_type is None:
            target_type = self._random_type(max_depth)
        env = GenerationEnv(var_types={})
        return self._gen_expr(target_type, max_depth, env)

    # --------------------------- Type helpers -------------------------
    def _random_type(self, depth: int) -> Type:
        """Sample a random concrete type; restrict complexity by depth."""
        # Bias towards base types at shallow depths
        choices: List[Callable[[], Type]] = [
            lambda: INT,
            lambda: BOOL,
        ]
        if depth > 0:
            # Allow list types of simpler element types
            choices.append(lambda: list_of(self._random_type(depth - 1)))
            # Occasionally allow function types
            if depth > 1:
                choices.append(lambda: func(self._random_type(depth - 2), self._random_type(depth - 2)))
        return self.rng.choice(choices)()

    def _flatten_func(self, t: Type) -> Tuple[List[Type], Type]:
        """Flatten curried function type into a list of param types and the return type."""
        params: List[Type] = []
        while isinstance(t, FunctionType):
            params.append(t.param_type)
            t = t.return_type
        return params, t

    def _fresh_param_name(self) -> str:
        name = f"x{self.param_counter}"
        self.param_counter += 1
        return name

    def _instantiate_scheme_randomly(self, scheme: TypeScheme, depth: int) -> Type:
        """Instantiate a TypeScheme by mapping each quantified TypeVar to a random concrete type."""
        mapping: Dict[TypeVar, Type] = {}
        # Deterministic order for stability
        for var in sorted(scheme.quantified, key=lambda v: v.name):
            # Avoid deeply nested function types too often
            choice = self._random_type(max(0, depth - 1))
            mapping[var] = choice
        return scheme.type.substitute(mapping)

    # ------------------------ Expression generation ------------------
    def _gen_expr(self, goal: Type, depth: int, env: GenerationEnv) -> ASTNode:
        # Base case generation to ensure termination
        if depth <= 0:
            node = self._gen_base(goal, env)
            if node is not None:
                return node
            # Fallback: minimal lambda for function type or literal defaults
            if isinstance(goal, FunctionType):
                return self._gen_lambda_for_type(goal, max(depth - 1, 0), env)
            return self._fallback_literal(goal)

        # Build a menu of strategies that can produce `goal`
        strategies: List[Callable[[], Optional[ASTNode]]] = []

        # 1) Use an in-scope variable of the exact goal type (if any)
        strategies.append(lambda: self._gen_from_variable(goal, depth, env))

        # 2) If expression: applicable for any goal type
        strategies.append(lambda: self._gen_if(goal, depth, env))

        # 3) Literal/list constructors when applicable
        strategies.append(lambda: self._gen_literal(goal))
        strategies.append(lambda: self._gen_list(goal, depth, env))

        # 4) Lambda when goal is a function type
        strategies.append(lambda: self._gen_lambda(goal, depth, env))

        # 5) Application of a builtin (or partially applied) producing `goal`
        strategies.append(lambda: self._gen_application(goal, depth, env))

        # Shuffle strategies to diversify outputs
        self.rng.shuffle(strategies)
        for strat in strategies:
            node = strat()
            if node is not None:
                return node

        # As a robust fallback, try a lambda if function type, else literal
        if isinstance(goal, FunctionType):
            return self._gen_lambda_for_type(goal, depth - 1, env)
        return self._fallback_literal(goal)

    # ------------------- Concrete strategy implementations ------------
    def _gen_base(self, goal: Type, env: GenerationEnv) -> Optional[ASTNode]:
        # Direct variable of exact type
        var_node = self._gen_from_variable(goal, 0, env)
        if var_node is not None:
            return var_node
        # Direct literal
        literal = self._gen_literal(goal)
        if literal is not None:
            return literal
        # Empty list if target is list
        if isinstance(goal, ListType):
            return ListNode([])
        return None

    def _gen_from_variable(self, goal: Type, depth: int, env: GenerationEnv) -> Optional[ASTNode]:
        candidates = [name for name, t in env.var_types.items() if t == goal]
        if not candidates:
            return None
        name = self.rng.choice(candidates)
        return VariableNode(name)

    def _gen_literal(self, goal: Type) -> Optional[ASTNode]:
        if goal == INT:
            return NumberNode(self.rng.randint(0, 99))
        if goal == BOOL:
            return BooleanNode(self.rng.choice([True, False]))
        return None

    def _gen_list(self, goal: Type, depth: int, env: GenerationEnv) -> Optional[ASTNode]:
        if not isinstance(goal, ListType):
            return None
        # Random length with bias to small
        length = 0 if depth <= 1 else self.rng.choice([0, 1, 2, 3])
        elements: List[ASTNode] = []
        for _ in range(length):
            elements.append(self._gen_expr(goal.elem_type, depth - 1, env))
        return ListNode(elements)

    def _gen_if(self, goal: Type, depth: int, env: GenerationEnv) -> Optional[ASTNode]:
        # Always applicable; but guard depth
        if depth <= 0:
            return None
        condition = self._gen_expr(BOOL, depth - 1, env)
        then_expr = self._gen_expr(goal, depth - 1, env)
        else_expr = self._gen_expr(goal, depth - 1, env)
        return IfNode(condition, then_expr, else_expr)

    def _gen_lambda(self, goal: Type, depth: int, env: GenerationEnv) -> Optional[ASTNode]:
        if not isinstance(goal, FunctionType):
            return None
        return self._gen_lambda_for_type(goal, depth - 1, env)

    def _gen_lambda_for_type(self, t: FunctionType, depth: int, env: GenerationEnv) -> LambdaNode:
        param_name = self._fresh_param_name()
        new_env = env.with_binding(param_name, t.param_type)
        body = self._gen_expr(t.return_type, max(depth, 0), new_env)
        return LambdaNode(param_name, body)

    def _gen_application(self, goal: Type, depth: int, env: GenerationEnv) -> Optional[ASTNode]:
        if depth <= 0:
            return None

        # Try several times to find a builtin instantiation that yields the goal type
        builtin_items = list(self.builtins.items())
        self.rng.shuffle(builtin_items)
        for _ in range(64):
            if not builtin_items:
                return None
            name, scheme = self.rng.choice(builtin_items)
            instantiated = self._instantiate_scheme_randomly(scheme, depth)
            params, ret = self._flatten_func(instantiated)
            if not params:
                # Not a function; skip
                continue

            # Decide how many arguments to apply (partial application allowed)
            # We search for n so that resulting type matches goal
            for n in range(1, len(params) + 1):
                result_t: Type = ret
                for i in range(len(params) - 1, n - 1, -1):
                    result_t = func(params[i], result_t)
                if result_t == goal:
                    # Build application node with n arguments
                    func_node: ASTNode = VariableNode(name)
                    # Generate curried application in one ApplicationNode with n args
                    args_nodes: List[ASTNode] = []
                    for i in range(n):
                        args_nodes.append(self._gen_expr(params[i], depth - 1, env))
                    return ApplicationNode(func_node, args_nodes)

        # If builtin application failed, try applying an in-scope function variable
        # (e.g., partially applied '+' from the environment if such exists)
        func_vars: List[Tuple[str, Type]] = [
            (n, t) for n, t in env.var_types.items() if isinstance(t, FunctionType)
        ]
        if not func_vars:
            return None

        self.rng.shuffle(func_vars)
        for name, t in func_vars:
            params, ret = self._flatten_func(t)
            for n in range(1, len(params) + 1):
                result_t: Type = ret
                for i in range(len(params) - 1, n - 1, -1):
                    result_t = func(params[i], result_t)
                if result_t == goal:
                    func_node = VariableNode(name)
                    args_nodes = [self._gen_expr(p, depth - 1, env) for p in params[:n]]
                    return ApplicationNode(func_node, args_nodes)

        return None

    def _fallback_literal(self, goal: Type) -> ASTNode:
        if isinstance(goal, FunctionType):
            return self._gen_lambda_for_type(goal, 0, GenerationEnv(var_types={}))
        if isinstance(goal, ListType):
            return ListNode([])
        if goal == INT:
            return NumberNode(0)
        if goal == BOOL:
            return BooleanNode(False)
        # Default to an int literal if all else fails
        return NumberNode(0)


def sample_program(seed: int, max_depth: int, target_type: Optional[Type] = None) -> ASTNode:
    """
    Generate a random well-typed program AST.

    Args:
        seed: RNG seed for reproducibility
        max_depth: Maximum recursive construction depth (>= 0)
        target_type: Optional desired result type. If None, a random type is chosen.

    Returns:
        An ASTNode representing a well-typed program.
    """
    sampler = Sampler(seed)
    return sampler.generate(max_depth=max(0, int(max_depth)), target_type=target_type)


__all__ = [
    "sample_program",
    "Sampler",
]


