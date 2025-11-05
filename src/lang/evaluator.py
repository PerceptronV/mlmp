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
    
    def __init__(self):
        """Initialize the evaluator with built-in functions."""
        self.global_env = self._create_global_environment()
    
    def _create_global_environment(self) -> Environment:
        """Create the global environment with all built-in functions."""
        env = Environment()
        
        # Arithmetic operators
        env.define("+", self._builtin_add)
        env.define("-", self._builtin_sub)
        env.define("*", self._builtin_mul)
        env.define("/", self._builtin_div)
        env.define("%", self._builtin_mod)
        
        # Comparison operators
        env.define("<", self._builtin_lt)
        env.define(">", self._builtin_gt)
        env.define("==", self._builtin_eq)
        
        # Boolean operators
        env.define("and", self._builtin_and)
        env.define("or", self._builtin_or)
        env.define("not", self._builtin_not)
        
        # Number predicates
        env.define("is_even", self._builtin_is_even)
        env.define("is_odd", self._builtin_is_odd)
        
        # List construction
        env.define("singleton", self._builtin_singleton)
        env.define("repeat", self._builtin_repeat)
        env.define("range", self._builtin_range)
        env.define("cons", self._builtin_cons)
        
        # List combination
        env.define("append", self._builtin_append)
        env.define("concat", self._builtin_concat)
        env.define("zip", self._builtin_zip)
        
        # List access
        env.define("first", self._builtin_first)
        env.define("second", self._builtin_second)
        env.define("third", self._builtin_third)
        env.define("last", self._builtin_last)
        env.define("nth", self._builtin_nth)
        
        # List modification
        env.define("insert", self._builtin_insert)
        env.define("replace", self._builtin_replace)
        env.define("swap", self._builtin_swap)
        
        # List removal
        env.define("cut_idx", self._builtin_cut_idx)
        env.define("cut_val", self._builtin_cut_val)
        env.define("cut_vals", self._builtin_cut_vals)
        env.define("drop", self._builtin_drop)
        env.define("droplast", self._builtin_droplast)
        
        # List slicing
        env.define("take", self._builtin_take)
        env.define("takelast", self._builtin_takelast)
        env.define("slice", self._builtin_slice)
        env.define("cut_slice", self._builtin_cut_slice)
        env.define("splice", self._builtin_splice)
        
        # Higher-order functions
        env.define("map", self._builtin_map)
        env.define("mapi", self._builtin_mapi)
        env.define("filter", self._builtin_filter)
        env.define("filteri", self._builtin_filteri)
        env.define("fold", self._builtin_fold)
        env.define("foldi", self._builtin_foldi)
        
        # List queries
        env.define("is_in", self._builtin_is_in)
        env.define("count", self._builtin_count)
        env.define("find", self._builtin_find)
        
        # List aggregation
        env.define("length", self._builtin_length)
        env.define("max", self._builtin_max)
        env.define("min", self._builtin_min)
        env.define("product", self._builtin_product)
        env.define("sum", self._builtin_sum)
        
        # List transformation
        env.define("unique", self._builtin_unique)
        env.define("sort", self._builtin_sort)
        env.define("reverse", self._builtin_reverse)
        env.define("flatten", self._builtin_flatten)
        env.define("group", self._builtin_group)
        
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
            return func(*args)
        
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
    
    # ========================================================================
    # Built-in Functions
    # ========================================================================
    
    # Arithmetic operators
    def _builtin_add(self, a: int, b: int) -> int:
        """Addition: (+ x y)"""
        return a + b
    
    def _builtin_sub(self, a: int, b: int) -> int:
        """Subtraction: (- x y)"""
        return a - b
    
    def _builtin_mul(self, a: int, b: int) -> int:
        """Multiplication: (* x y)"""
        return a * b
    
    def _builtin_div(self, a: int, b: int) -> int:
        """Integer division: (/ x y)"""
        if b == 0:
            raise EvaluationError("Division by zero")
        return a // b
    
    def _builtin_mod(self, a: int, b: int) -> int:
        """Modulo: (% x y)"""
        if b == 0:
            raise EvaluationError("Modulo by zero")
        return a % b
    
    # Comparison operators
    def _builtin_lt(self, a: int, b: int) -> bool:
        """Less than: (< x y)"""
        return a < b
    
    def _builtin_gt(self, a: int, b: int) -> bool:
        """Greater than: (> x y)"""
        return a > b
    
    def _builtin_eq(self, a: Any, b: Any) -> bool:
        """Structural equality: (== x y)"""
        return a == b
    
    # Boolean operators
    def _builtin_and(self, a: bool, b: bool) -> bool:
        """Boolean AND: (and x y)"""
        return a and b
    
    def _builtin_or(self, a: bool, b: bool) -> bool:
        """Boolean OR: (or x y)"""
        return a or b
    
    def _builtin_not(self, a: bool) -> bool:
        """Boolean NOT: (not x)"""
        return not a
    
    # Number predicates
    def _builtin_is_even(self, n: int) -> bool:
        """Check if even: (is_even x)"""
        return n % 2 == 0
    
    def _builtin_is_odd(self, n: int) -> bool:
        """Check if odd: (is_odd x)"""
        return n % 2 == 1
    
    # List construction
    def _builtin_singleton(self, x: Any) -> List[Any]:
        """Create single-element list: (singleton x)"""
        return [x]
    
    def _builtin_repeat(self, x: Any, n: int) -> List[Any]:
        """Repeat element n times: (repeat x n)"""
        return [x] * n
    
    def _builtin_range(self, start: int, end: int, step: int) -> List[int]:
        """Range of numbers: (range i j n)"""
        return list(range(start, end + 1, step))
    
    def _builtin_cons(self, x: Any, xs: List[Any]) -> List[Any]:
        """Prepend element: (cons x xs)"""
        return [x] + xs
    
    # List combination
    def _builtin_append(self, xs: List[Any], x: Any) -> List[Any]:
        """Append element: (append xs x)"""
        return xs + [x]
    
    def _builtin_concat(self, xs: List[Any], ys: List[Any]) -> List[Any]:
        """Concatenate lists: (concat xs ys)"""
        return xs + ys
    
    def _builtin_zip(self, xs: List[Any], ys: List[Any]) -> List[List[Any]]:
        """Zip two lists: (zip xs ys)"""
        return [[x, y] for x, y in zip(xs, ys)]
    
    # List access
    def _builtin_first(self, xs: List[Any]) -> Any:
        """First element: (first xs)"""
        if not xs:
            raise EvaluationError("first: empty list")
        return xs[0]
    
    def _builtin_second(self, xs: List[Any]) -> Any:
        """Second element: (second xs)"""
        if len(xs) < 2:
            raise EvaluationError("second: list too short")
        return xs[1]
    
    def _builtin_third(self, xs: List[Any]) -> Any:
        """Third element: (third xs)"""
        if len(xs) < 3:
            raise EvaluationError("third: list too short")
        return xs[2]
    
    def _builtin_last(self, xs: List[Any]) -> Any:
        """Last element: (last xs)"""
        if not xs:
            raise EvaluationError("last: empty list")
        return xs[-1]
    
    def _builtin_nth(self, i: int, xs: List[Any]) -> Any:
        """Nth element: (nth i xs)"""
        if i < 0 or i >= len(xs):
            raise EvaluationError(f"nth: index {i} out of bounds")
        return xs[i]
    
    # List modification
    def _builtin_insert(self, x: Any, i: int, xs: List[Any]) -> List[Any]:
        """Insert at index: (insert x i xs)"""
        result = xs.copy()
        result.insert(i, x)
        return result
    
    def _builtin_replace(self, i: int, x: Any, xs: List[Any]) -> List[Any]:
        """Replace at index: (replace i x xs)"""
        if i < 0 or i >= len(xs):
            raise EvaluationError(f"replace: index {i} out of bounds")
        result = xs.copy()
        result[i] = x
        return result
    
    def _builtin_swap(self, i: int, j: int, xs: List[Any]) -> List[Any]:
        """Swap elements: (swap i j xs)"""
        if i < 0 or i >= len(xs) or j < 0 or j >= len(xs):
            raise EvaluationError("swap: index out of bounds")
        result = xs.copy()
        result[i], result[j] = result[j], result[i]
        return result
    
    # List removal
    def _builtin_cut_idx(self, i: int, xs: List[Any]) -> List[Any]:
        """Remove at index: (cut_idx i xs)"""
        if i < 0 or i >= len(xs):
            raise EvaluationError(f"cut_idx: index {i} out of bounds")
        return xs[:i] + xs[i+1:]
    
    def _builtin_cut_val(self, x: Any, xs: List[Any]) -> List[Any]:
        """Remove first occurrence: (cut_val x xs)"""
        result = xs.copy()
        try:
            result.remove(x)
        except ValueError:
            pass  # Element not found, return original
        return result
    
    def _builtin_cut_vals(self, x: Any, xs: List[Any]) -> List[Any]:
        """Remove all occurrences: (cut_vals x xs)"""
        return [elem for elem in xs if elem != x]
    
    def _builtin_drop(self, n: int, xs: List[Any]) -> List[Any]:
        """Drop first n elements: (drop n xs)"""
        return xs[n:]
    
    def _builtin_droplast(self, n: int, xs: List[Any]) -> List[Any]:
        """Drop last n elements: (droplast n xs)"""
        if n == 0:
            return xs
        return xs[:-n]
    
    # List slicing
    def _builtin_take(self, n: int, xs: List[Any]) -> List[Any]:
        """Take first n elements: (take n xs)"""
        return xs[:n]
    
    def _builtin_takelast(self, n: int, xs: List[Any]) -> List[Any]:
        """Take last n elements: (takelast n xs)"""
        return xs[-n:] if n > 0 else []
    
    def _builtin_slice(self, i: int, j: int, xs: List[Any]) -> List[Any]:
        """Slice from i to j: (slice i j xs)"""
        return xs[i:j]
    
    def _builtin_cut_slice(self, i: int, j: int, xs: List[Any]) -> List[Any]:
        """Remove slice: (cut_slice i j xs)"""
        return xs[:i] + xs[j:]
    
    def _builtin_splice(self, ys: List[Any], i: int, xs: List[Any]) -> List[Any]:
        """Insert list at index: (splice ys i xs)"""
        return xs[:i] + ys + xs[i:]
    
    # Higher-order functions
    def _builtin_map(self, f: Closure, xs: List[Any]) -> List[Any]:
        """Map function over list: (map f xs)"""
        return [self._apply(f, [x], f.env) for x in xs]
    
    def _builtin_mapi(self, f: Closure, xs: List[Any]) -> List[Any]:
        """Map with index: (mapi f xs)"""
        # f should take element and index (curried)
        result = []
        for i, x in enumerate(xs):
            # Apply f to x first, then to i
            val = self._apply(f, [x, i], f.env)
            result.append(val)
        return result
    
    def _builtin_filter(self, p: Closure, xs: List[Any]) -> List[Any]:
        """Filter list by predicate: (filter p xs)"""
        result = []
        for x in xs:
            if self._apply(p, [x], p.env):
                result.append(x)
        return result
    
    def _builtin_filteri(self, p: Closure, xs: List[Any]) -> List[Any]:
        """Filter with index: (filteri p xs)"""
        result = []
        for i, x in enumerate(xs):
            if self._apply(p, [i, x], p.env):
                result.append(x)
        return result
    
    def _builtin_fold(self, f: Closure, acc: Any, xs: List[Any]) -> Any:
        """Fold/reduce list: (fold f acc xs)"""
        for x in xs:
            acc = self._apply(f, [acc, x], f.env)
        return acc
    
    def _builtin_foldi(self, f: Closure, acc: Any, xs: List[Any]) -> Any:
        """Fold with index: (foldi f acc xs)"""
        for i, x in enumerate(xs):
            acc = self._apply(f, [acc, x, i], f.env)
        return acc
    
    # List queries
    def _builtin_is_in(self, xs: List[Any], x: Any) -> bool:
        """Check membership: (is_in xs x)"""
        return x in xs
    
    def _builtin_count(self, p: Closure, xs: List[Any]) -> int:
        """Count matching elements: (count p xs)"""
        count = 0
        for x in xs:
            if self._apply(p, [x], p.env):
                count += 1
        return count
    
    def _builtin_find(self, p: Closure, xs: List[Any]) -> List[int]:
        """Find indices matching predicate: (find p xs)"""
        return [i for i, x in enumerate(xs) if self._apply(p, [x], p.env)]
    
    # List aggregation
    def _builtin_length(self, xs: List[Any]) -> int:
        """Length of list: (length xs)"""
        return len(xs)
    
    def _builtin_max(self, xs: List[int]) -> int:
        """Maximum element: (max xs)"""
        if not xs:
            raise EvaluationError("max: empty list")
        return max(xs)
    
    def _builtin_min(self, xs: List[int]) -> int:
        """Minimum element: (min xs)"""
        if not xs:
            raise EvaluationError("min: empty list")
        return min(xs)
    
    def _builtin_product(self, xs: List[int]) -> int:
        """Product of elements: (product xs)"""
        result = 1
        for x in xs:
            result *= x
        return result
    
    def _builtin_sum(self, xs: List[int]) -> int:
        """Sum of elements: (sum xs)"""
        return sum(xs)
    
    # List transformation
    def _builtin_unique(self, xs: List[Any]) -> List[Any]:
        """Unique elements: (unique xs)"""
        seen = set()
        result = []
        for x in xs:
            # Use tuple for hashability if list
            key = tuple(x) if isinstance(x, list) else x
            if key not in seen:
                seen.add(key)
                result.append(x)
        return result
    
    def _builtin_sort(self, f: Closure, xs: List[Any]) -> List[Any]:
        """Sort by key function: (sort f xs)"""
        return sorted(xs, key=lambda x: self._apply(f, [x], f.env))
    
    def _builtin_reverse(self, xs: List[Any]) -> List[Any]:
        """Reverse list: (reverse xs)"""
        return list(reversed(xs))
    
    def _builtin_flatten(self, xss: List[List[Any]]) -> List[Any]:
        """Flatten list of lists: (flatten xss)"""
        result = []
        for xs in xss:
            result.extend(xs)
        return result
    
    def _builtin_group(self, f: Closure, xs: List[Any]) -> List[List[Any]]:
        """Group by key function: (group f xs)"""
        from collections import defaultdict
        groups = defaultdict(list)
        for x in xs:
            key = self._apply(f, [x], f.env)
            # Use tuple for hashability if list
            key = tuple(key) if isinstance(key, list) else key
            groups[key].append(x)
        return list(groups.values())


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
        try:
            result = evaluate(code)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")

