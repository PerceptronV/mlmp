"""
JIT Compiler for the Functional Programming Language

This module implements a true JIT compiler that:
1. Takes an AST expression
2. Generates equivalent Python source code
3. Compiles to Python bytecode using compile() + exec()
4. Returns a native Python callable that runs WITHOUT re-interpreting the AST

This provides significant speedup for expressions that are compiled once
and executed many times.
"""

import time
from typing import Any, Dict, Set, Callable, Optional
from .ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IfNode, IntHoleNode,
)
from .evaluator import Evaluator
from .grammar import Grammar, DefaultGrammar
from .parser import parse
from .type_utils import get_origin, CallableOrig
from .utils import RANDINT_PROBE_SEQUENCE


class JITCompilationError(Exception):
    """Exception raised during JIT compilation."""
    pass


def _make_curry_aware(fn: Callable, arg_types: tuple) -> Callable:
    """
    Wrap a function to handle curried callable arguments.
    
    If a function expects f(a, b) but receives a curried function,
    this wrapper will call f(a)(b) instead.
    """
    # Find indices of callable arguments
    callable_indices = [
        i for i, t in enumerate(arg_types)
        if get_origin(t) == CallableOrig
    ]
    
    if not callable_indices:
        # No callable arguments, return as-is
        return fn
    
    def curry_aware_wrapper(*args):
        args = list(args)
        for idx in callable_indices:
            if idx < len(args) and callable(args[idx]):
                original_fn = args[idx]
                # Create a wrapper that tries multi-arg call first, then curried
                def make_curry_handler(f):
                    def curry_handler(*inner_args):
                        try:
                            return f(*inner_args)
                        except TypeError:
                            # Try curried application
                            result = f
                            for arg in inner_args:
                                result = result(arg)
                            return result
                    return curry_handler
                args[idx] = make_curry_handler(original_fn)
        return fn(*args)
    
    curry_aware_wrapper.__name__ = getattr(fn, '__name__', 'curry_aware')
    return curry_aware_wrapper


class JITCompiler:
    """
    JIT Compiler that generates native Python functions from AST.
    
    Unlike the interpreter or the existing compiler, this generates actual
    Python source code and compiles it to bytecode. The resulting function
    can be called repeatedly without any AST traversal.
    
    Usage:
        jit = JITCompiler()
        fn = jit.compile(ast)  # Returns a native Python function
        result = fn()          # Execute without interpretation
        result = fn()          # Execute again - no recompilation!
    """
    
    def __init__(self, grammar: Grammar = DefaultGrammar):
        """Initialise the JIT compiler with a grammar."""
        self.grammar = grammar
        self._var_counter = 0
        self._builtins = self._extract_raw_builtins(grammar)
    
    def _extract_raw_builtins(self, grammar: Grammar) -> Dict[str, Callable]:
        """
        Extract raw built-in functions from the grammar, wrapped to handle currying.
        
        The grammar's __iter__ yields wrapped functions that expect an evaluator.
        For JIT compilation, we need the raw functions that work directly,
        but wrapped to handle curried callable arguments.
        """
        builtins = {}
        for name in grammar.names:
            fn_info = grammar[name]
            raw_fn = fn_info['fn']
            arg_types = fn_info['arg_types']
            # Wrap with curry-aware handler for higher-order functions
            builtins[name] = _make_curry_aware(raw_fn, arg_types)
        return builtins
    
    def _fresh_var(self, prefix: str = "_v") -> str:
        """Generate a fresh variable name to avoid collisions."""
        self._var_counter += 1
        return f"{prefix}{self._var_counter}"
    
    def compile(self, node: ASTNode, substitution: list[int] | None = None) -> tuple[Callable, ASTNode]:
        """
        Compile an AST node to a native Python callable.

        Args:
            node: AST node to compile
            substitution: Optional list of integer values for IntHoleNodes,
                indexed by pre-order traversal position. If None, uses
                RANDINT_PROBE_SEQUENCE for fingerprinting.

        Returns:
            A tuple (callable, concrete_ast) where callable is a native Python
            function and concrete_ast is the AST with holes filled in.
            If the AST is a lambda, callable is a function.
            Otherwise, callable is a zero-argument callable that returns the value.
        """
        # Reset variable counter and hole state for each compilation
        self._var_counter = 0
        self._hole_counter = 0
        self._substitution = substitution

        # If the node is a lambda, compile it as a function directly
        if isinstance(node, LambdaNode):
            return self._compile_lambda_to_native(node)

        # Otherwise, wrap in a zero-argument function that returns the value
        return self._compile_expression_to_native(node)

    def execute(self, sketch: ASTNode, substitution: list[int], x) -> tuple:
        """Compile sketch with substitution and evaluate on input x."""
        fn, concrete_ast = self.compile(sketch, substitution)
        return fn(x), concrete_ast
    
    def _compile_expression_to_native(self, node: ASTNode) -> tuple[Callable, ASTNode]:
        """Compile a non-lambda expression to a native callable."""
        code, concrete_node, _ = self._generate(node, set())

        # Create the function source
        # Use *args, **kwargs to allow calling the result if it's a callable
        # This handles cases like (if cond lambda1 lambda2) where the result
        # is a function that should be called with arguments
        func_name = self._fresh_var("_compiled_expr")
        source = f"""def {func_name}(*args, **kwargs):
    result = {code}
    if args or kwargs:
        return result(*args, **kwargs)
    return result
"""

        # Compile and execute
        namespace = {"_builtins": self._builtins}
        exec(compile(source, "<jit>", "exec"), namespace)

        return namespace[func_name], concrete_node
    
    def _compile_lambda_to_native(self, node: LambdaNode) -> tuple[Callable, ASTNode]:
        """
        Compile a lambda node to a native Python function.

        Handles:
        - Single and multiple parameters
        - Captured variables (closures)
        - Nested lambdas
        """
        params = node.param
        body_code, concrete_body, _ = self._generate(node.body, set(params))

        # Generate function definition
        func_name = self._fresh_var("_lambda")
        params_str = ", ".join(params)
        source = f"def {func_name}({params_str}):\n    return {body_code}\n"

        # Compile and execute
        namespace = {"_builtins": self._builtins}
        exec(compile(source, "<jit>", "exec"), namespace)

        fn = namespace[func_name]

        # Add metadata for compatibility with code that checks for Closure attributes
        fn.param = params
        fn.body = concrete_body
        fn.env = {}  # Empty dict since captured vars are in closure

        concrete_ast = LambdaNode(params, concrete_body)
        return fn, concrete_ast
    
    def compile_to_source(self, node: ASTNode) -> str:
        """
        Compile an AST to Python source code (for debugging/inspection).

        Args:
            node: AST node to compile

        Returns:
            Generated Python source code as a string
        """
        self._var_counter = 0
        self._hole_counter = 0
        self._substitution = None

        if isinstance(node, LambdaNode):
            params = node.param
            body_code, _, _ = self._generate(node.body, set(params))
            params_str = ", ".join(params)
            return f"lambda {params_str}: {body_code}"
        else:
            code, _, _ = self._generate(node, set())
            return code
    
    def _generate(self, node: ASTNode, bound_vars: Set[str]) -> tuple[str, ASTNode, Set[str]]:
        """
        Generate Python code for an AST node.

        Args:
            node: AST node to generate code for
            bound_vars: Set of currently bound variable names

        Returns:
            Tuple of (generated_code, concrete_ast_node, set_of_free_variables_used)
        """
        node_type = type(node)

        if node_type is NumberNode:
            return str(node.value), NumberNode(node.value), set()

        elif node_type is BooleanNode:
            return str(node.value), BooleanNode(node.value), set()

        elif node_type is VariableNode:
            name = node.name
            if name in bound_vars:
                # Local variable - use directly
                return name, VariableNode(name), set()
            elif name in self._builtins:
                # Built-in function - access from _builtins dict
                return f"_builtins[{repr(name)}]", VariableNode(name), set()
            else:
                # Free variable - this will be an error at runtime
                raise JITCompilationError(f"Undefined variable: {name}")

        elif node_type is IntHoleNode:
            values = self._substitution if self._substitution is not None else RANDINT_PROBE_SEQUENCE
            v = values[self._hole_counter % len(values)]
            self._hole_counter += 1
            return str(v), NumberNode(v), set()

        elif node_type is ListNode:
            if not node.elements:
                return "[]", ListNode([]), set()
            elem_codes = []
            concrete_elems = []
            free_vars = set()
            for elem in node.elements:
                code, concrete_elem, fv = self._generate(elem, bound_vars)
                elem_codes.append(code)
                concrete_elems.append(concrete_elem)
                free_vars |= fv
            return f"[{', '.join(elem_codes)}]", ListNode(concrete_elems), free_vars

        elif node_type is LambdaNode:
            return self._generate_lambda(node, bound_vars)

        elif node_type is IfNode:
            return self._generate_if(node, bound_vars)

        elif node_type is ApplicationNode:
            return self._generate_application(node, bound_vars)

        else:
            raise JITCompilationError(f"Unknown node type: {node_type.__name__}")
    
    def _generate_lambda(self, node: LambdaNode, bound_vars: Set[str]) -> tuple[str, ASTNode, Set[str]]:
        """Generate code for a lambda expression."""
        params = node.param
        new_bound = bound_vars | set(params)
        body_code, concrete_body, free_vars = self._generate(node.body, new_bound)

        params_str = ", ".join(params)
        code = f"(lambda {params_str}: {body_code})"

        # Remove the lambda's own parameters from free vars
        free_vars -= set(params)

        return code, LambdaNode(params, concrete_body), free_vars

    def _generate_if(self, node: IfNode, bound_vars: Set[str]) -> tuple[str, ASTNode, Set[str]]:
        """Generate code for an if expression."""
        cond_code, concrete_cond, cond_fv = self._generate(node.condition, bound_vars)
        then_code, concrete_then, then_fv = self._generate(node.then_expr, bound_vars)
        else_code, concrete_else, else_fv = self._generate(node.else_expr, bound_vars)

        # Python's conditional expression
        code = f"({then_code} if {cond_code} else {else_code})"

        return code, IfNode(concrete_cond, concrete_then, concrete_else), cond_fv | then_fv | else_fv

    def _generate_application(self, node: ApplicationNode, bound_vars: Set[str]) -> tuple[str, ASTNode, Set[str]]:
        """Generate code for a function application."""
        func_node = node.function
        args = node.arguments

        free_vars = set()

        # Generate argument code
        arg_codes = []
        concrete_args = []
        for arg in args:
            code, concrete_arg, fv = self._generate(arg, bound_vars)
            arg_codes.append(code)
            concrete_args.append(concrete_arg)
            free_vars |= fv

        # Check if this is a direct call to a built-in
        if isinstance(func_node, VariableNode) and func_node.name in self._builtins:
            func_name = func_node.name
            args_str = ", ".join(arg_codes)
            concrete_func = VariableNode(func_name)
            return (
                f"_builtins[{repr(func_name)}]({args_str})",
                ApplicationNode(concrete_func, concrete_args),
                free_vars,
            )

        # General case: compile the function and call it
        func_code, concrete_func, func_fv = self._generate(func_node, bound_vars)
        free_vars |= func_fv

        args_str = ", ".join(arg_codes)
        return (
            f"({func_code})({args_str})",
            ApplicationNode(concrete_func, concrete_args),
            free_vars,
        )
    
    def _find_free_variables(self, node: ASTNode, bound_vars: Set[str]) -> Set[str]:
        """Find all free variables in an expression."""
        node_type = type(node)

        if node_type is NumberNode or node_type is BooleanNode or node_type is IntHoleNode:
            return set()

        elif node_type is VariableNode:
            if node.name in bound_vars or node.name in self._builtins:
                return set()
            return {node.name}
        
        elif node_type is ListNode:
            free = set()
            for elem in node.elements:
                free |= self._find_free_variables(elem, bound_vars)
            return free
        
        elif node_type is LambdaNode:
            new_bound = bound_vars | set(node.param)
            return self._find_free_variables(node.body, new_bound)
        
        elif node_type is IfNode:
            return (
                self._find_free_variables(node.condition, bound_vars) |
                self._find_free_variables(node.then_expr, bound_vars) |
                self._find_free_variables(node.else_expr, bound_vars)
            )
        
        elif node_type is ApplicationNode:
            free = self._find_free_variables(node.function, bound_vars)
            for arg in node.arguments:
                free |= self._find_free_variables(arg, bound_vars)
            return free
        
        return set()


class JITCompiledClosure:
    """
    A wrapper that provides Closure-like interface for JIT-compiled functions.
    
    This allows JIT-compiled lambdas to work with code that expects Closure objects,
    while still providing the performance benefits of native Python functions.
    """
    
    def __init__(self, fn: Callable, params: list, body: ASTNode, captured_env: Dict[str, Any]):
        self.fn = fn
        self.param = params
        self.body = body
        self.env = captured_env
    
    def __call__(self, *args):
        return self.fn(*args)
    
    def __repr__(self):
        params_str = " ".join(self.param)
        return f"<jit-closure λ{params_str}>"


def jit_compile(code: str, grammar: Grammar = DefaultGrammar) -> Callable:
    """
    Convenience function to parse and JIT compile code.
    
    Args:
        code: Source code string
        grammar: Grammar to use for compilation
        
    Returns:
        A native Python callable
        
    Example:
        >>> increment = jit_compile("(λ x (+ x 1))")
        >>> increment(5)
        6
        >>> increment(100)
        101
    """
    ast = parse(code)
    jit = JITCompiler(grammar)
    fn, _ = jit.compile(ast)
    return fn


def jit_compile_and_run(code: str, grammar: Grammar = DefaultGrammar) -> Any:
    """
    Convenience function to parse, JIT compile, and execute code.

    Args:
        code: Source code string
        grammar: Grammar to use

    Returns:
        Execution result
    """
    fn = jit_compile(code, grammar)
    return fn()


if __name__ == "__main__":
    print("JIT Compiler Examples:")
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
        ("Map", "(map (λ x (* x 2)) [1 2 3])"),
        ("Filter", "(filter (λ x (> x 2)) [1 2 3 4])"),
        ("Fold - curried", "(fold (λ a (λ x (+ a x))) 0 [1 2 3])"),
        ("If true", "(if true 1 2)"),
        ("If false", "(if false 1 2)"),
        ("Nested if", "(if (> 5 3) (if (< 2 1) 10 20) 30)"),
    ]
    
    jit = JITCompiler()
    
    for name, code in examples:
        print(f"\n{name}: {code}")
        try:
            ast = parse(code)
            fn, _ = jit.compile(ast)
            result = fn()
            source = jit.compile_to_source(ast)
            print(f"Generated: {source}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

    # Demo: compile a lambda and use it as a Python function
    print("\n" + "=" * 80)
    print("Demo: JIT-Compiled Lambda as Python Function")
    print("=" * 80)

    # Compile a lambda expression
    increment_code = "(λ x (+ x 1))"
    increment_ast = parse(increment_code)
    increment_fn, _ = jit.compile(increment_ast)

    print(f"\nCompiled function: {increment_fn}")
    print(f"Callable: {callable(increment_fn)}")
    print(f"Generated source: {jit.compile_to_source(increment_ast)}")

    # Use it like a normal Python function
    print("\nCalling increment_fn(5):")
    print(f"Result: {increment_fn(5)}")

    print("\nCalling increment_fn(100):")
    print(f"Result: {increment_fn(100)}")

    # Multi-parameter lambda
    multiply_code = "(λ (x y) (* x y))"
    multiply_ast = parse(multiply_code)
    multiply_fn, _ = jit.compile(multiply_ast)

    print(f"\nCompiled multiply: {jit.compile_to_source(multiply_ast)}")
    print(f"multiply_fn(3, 7) = {multiply_fn(3, 7)}")

    # Performance comparison
    print("\n" + "=" * 80)
    print("Performance Comparison: Interpreter vs JIT")
    print("=" * 80)

    test_code = "(map (λ x (* x x)) [1 2 3 4 5])"
    test_ast = parse(test_code)

    # Interpreter
    evaluator = Evaluator()

    # Warm up
    evaluator.eval(test_ast)
    jit_fn, _ = jit.compile(test_ast)
    jit_fn()
    
    # Benchmark
    iterations = 10000
    
    start = time.perf_counter()
    for _ in range(iterations):
        evaluator.eval(test_ast)
    interp_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(iterations):
        jit_fn()
    jit_time = time.perf_counter() - start
    
    print(f"\nExpression: {test_code}")
    print(f"Iterations: {iterations}")
    print(f"Interpreter: {interp_time:.4f}s ({iterations/interp_time:.0f} ops/sec)")
    print(f"JIT:         {jit_time:.4f}s ({iterations/jit_time:.0f} ops/sec)")
    print(f"Speedup:     {interp_time/jit_time:.2f}x")
