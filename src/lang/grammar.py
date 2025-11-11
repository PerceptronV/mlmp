from __future__ import annotations
import inspect
import random
from typing import get_type_hints, get_origin, TypeVar, Callable


SPECIAL_CHARS_DefaultGrammar = ('λ', '(', ')', '[', ']', ' ')
T1, T2 = TypeVar('T1'), TypeVar('T2')

# ============================================================================
# Base Grammar Class
# ============================================================================

class Grammar:
    def __init__(self, special_chars=None, functions=None):
        self.special_chars = special_chars or SPECIAL_CHARS_DefaultGrammar
        self.functions = functions or {}
        self.name_map = {}
    
    def __str__(self):
        return f"Grammar[ {len(self.special_chars)} special char(s), {len(self.functions)} function(s) ]"

    def __repr__(self):
        return f"Grammar(special_chars={self.special_chars}, functions={tuple(self.functions.keys())})"
    
    def _analyse(self, name, fn):
        var_types = get_type_hints(fn)
        signature = inspect.signature(fn)
        arg_names = tuple(p.name for p in signature.parameters.values())
        
        try:
            arg_types = tuple(var_types[a] for a in arg_names)
        except KeyError:
            raise TypeError(f"Missing type hint for arguments ({
                ', '.join(a for a in arg_names if a not in var_types)
            }) of {fn.__name__}")
        
        try:
            ret_type = var_types['return']
        except KeyError:
            raise TypeError(f"Missing type hint for return value of {fn.__name__}")
        
        return {
            'fn': fn,
            '__call__': self._make_evaluable(name, fn, arg_types),
            'arg_names': arg_names,
            'arg_types': arg_types,
            'ret_type': ret_type
        }
    
    def _make_evaluable(self, name: str, fn: Callable, arg_types) -> Callable:
        '''Make a function evaluable by the evaluator.'''

        # find the indices of the callable arguments in fn
        callable_orig = get_origin(Callable)
        callable_indices = [
            e for e, t in enumerate(arg_types)
            if get_origin(t) == callable_orig
        ]

        # When fn is called, it will be called with the evaluator and the arguments.
        # We therefore wrap fn in _eval_fn to handle this.
        def _eval_fn(evaluator, *args):

            # The callable arguments of fn will be supplied as Closure objects,
            # which have their own environments that must be respected.
            # We need to create a new function that will apply the closure
            # to the arguments and return the result as the function would
            # have originally in the user definition of fn. So again, we wrap this.
            def _make_normal_function(_f_closure):
                # this lambda function is what _f should do normally
                return lambda *_f_args: evaluator._apply(
                    _f_closure, list(_f_args), _f_closure.env
                )
            
            args = list(args)
            # convert closure functions to normal functions
            for idx in callable_indices:
                args[idx] = _make_normal_function(args[idx])
            
            return fn(*args)
        
        _eval_fn.__name__ = name
        return _eval_fn
    
    def add_function(self, name, fn):
        self.functions[name] = self._analyse(name, fn)
    
    def load_from_module(self, module):
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            self.add_function(name, fn)

    def __call__(self, fn=None, /, *, name=None):
        # called as @<classname>
        if name is None:
            self.add_function(fn.__name__, fn)
            return fn

        # called as @<classname>(name='...')
        def _wrap(_fn):
            self.add_function(name, _fn)
            return _fn

        return _wrap
    
    def shuffle_bindings(self, bindings: list=None, seed=None):
        random.seed(seed)
        function_names = self.functions.keys()
        new_names = bindings or list(function_names)
        random.shuffle(new_names)
        self.name_map = dict(zip(new_names, function_names))
    
    def __getitem__(self, name):
        name = self.name_map.get(name, name)
        return self.functions[name]
    
    def __iter__(self):
        names = self.name_map.keys() or self.functions.keys()
        for n in names:
            yield n, self[n]['__call__']


# ============================================================================
# Expose DefaultGrammar grammar class for easy access
# ============================================================================

DefaultGrammar = Grammar()

# arithmetic operators
@DefaultGrammar(name='+')
def add(x: int, y: int) -> int:
    """Addition: (+ x y)"""
    return x + y

@DefaultGrammar(name='-')
def subtract(x: int, y: int) -> int:
    """Subtraction: (- x y)"""
    return x - y

@DefaultGrammar(name='*')
def multiply(x: int, y: int) -> int:
    """Multiplication: (* x y)"""
    return x * y

@DefaultGrammar(name='/')
def divide(x: int, y: int) -> int:
    """Integer division: (/ x y)"""
    if y == 0:
        raise ValueError("Division by zero")
    return x // y

@DefaultGrammar(name='%')
def modulo(x: int, y: int) -> int:
    """Modulo: (% x y)"""
    if y == 0:
        raise ValueError("Modulo by zero")
    return x % y

# comparison operators
@DefaultGrammar(name='<')
def less_than(x: int, y: int) -> bool:
    """Less than: (< x y)"""
    return x < y

@DefaultGrammar(name='>')
def greater_than(x: int, y: int) -> bool:
    """Greater than: (> x y)"""
    return x > y

@DefaultGrammar(name='==')
def equals(x: object, y: object) -> bool:
    """Structural equality: (== x y)"""
    return x == y

# boolean operators
@DefaultGrammar(name='and')
def boolean_and(x: bool, y: bool) -> bool:
    """Boolean AND: (and x y)"""
    return x and y

@DefaultGrammar(name='or')
def boolean_or(x: bool, y: bool) -> bool:
    """Boolean OR: (or x y)"""
    return x or y

@DefaultGrammar(name='not')
def boolean_not(x: bool) -> bool:
    """Boolean NOT: (not x)"""
    return not x

# number predicates
@DefaultGrammar
def is_even(n: int) -> bool:
    """Check if even: (is_even x)"""
    return n % 2 == 0

@DefaultGrammar
def is_odd(n: int) -> bool:
    """Check if odd: (is_odd x)"""
    return n % 2 == 1

# list construction
@DefaultGrammar
def singleton(x: object) -> list:
    """Create single-element list: (singleton x)"""
    return [x]

@DefaultGrammar
def repeat(x: object, n: int) -> list:
    """Repeat element n times: (repeat x n)"""
    return [x] * n

@DefaultGrammar(name='range')
def range_fn(start: int, end: int, step: int) -> list:
    """Range of numbers: (range i j n)"""
    return list(range(start, end + 1, step))

@DefaultGrammar
def cons(x: object, xs: list) -> list:
    """Prepend element: (cons x xs)"""
    return [x] + xs

# list combination
@DefaultGrammar
def append(xs: list, x: object) -> list:
    """Append element: (append xs x)"""
    return xs + [x]

@DefaultGrammar
def concat(xs: list, ys: list) -> list:
    """Concatenate lists: (concat xs ys)"""
    return xs + ys

@DefaultGrammar(name='zip')
def zip_fn(xs: list, ys: list) -> list:
    """Zip two lists: (zip xs ys)"""
    return [[x, y] for x, y in zip(xs, ys)]

# list access
@DefaultGrammar
def first(xs: list) -> object:
    """First element: (first xs)"""
    if not xs:
        raise ValueError("first: empty list")
    return xs[0]

@DefaultGrammar
def second(xs: list) -> object:
    """Second element: (second xs)"""
    if len(xs) < 2:
        raise ValueError("second: list too short")
    return xs[1]

@DefaultGrammar
def third(xs: list) -> object:
    """Third element: (third xs)"""
    if len(xs) < 3:
        raise ValueError("third: list too short")
    return xs[2]

@DefaultGrammar
def last(xs: list) -> object:
    """Last element: (last xs)"""
    if not xs:
        raise ValueError("last: empty list")
    return xs[-1]

@DefaultGrammar
def nth(i: int, xs: list) -> object:
    """Nth element: (nth i xs)"""
    if i < 0 or i >= len(xs):
        raise ValueError(f"nth: index {i} out of bounds")
    return xs[i]

# list modification
@DefaultGrammar
def insert(x: object, i: int, xs: list) -> list:
    """Insert at index: (insert x i xs)"""
    result = xs.copy()
    result.insert(i, x)
    return result

@DefaultGrammar
def replace(i: int, x: object, xs: list) -> list:
    """Replace at index: (replace i x xs)"""
    if i < 0 or i >= len(xs):
        raise ValueError(f"replace: index {i} out of bounds")
    result = xs.copy()
    result[i] = x
    return result

@DefaultGrammar
def swap(i: int, j: int, xs: list) -> list:
    """Swap elements: (swap i j xs)"""
    if i < 0 or i >= len(xs) or j < 0 or j >= len(xs):
        raise ValueError("swap: index out of bounds")
    result = xs.copy()
    result[i], result[j] = result[j], result[i]
    return result

# list removal
@DefaultGrammar
def cut_idx(i: int, xs: list) -> list:
    """Remove at index: (cut_idx i xs)"""
    if i < 0 or i >= len(xs):
        raise ValueError(f"cut_idx: index {i} out of bounds")
    return xs[:i] + xs[i+1:]

@DefaultGrammar
def cut_val(x: object, xs: list) -> list:
    """Remove first occurrence: (cut_val x xs)"""
    result = xs.copy()
    try:
        result.remove(x)
    except ValueError:
        pass  # Element not found, return original
    return result

@DefaultGrammar
def cut_vals(x: object, xs: list) -> list:
    """Remove all occurrences: (cut_vals x xs)"""
    return [elem for elem in xs if elem != x]

@DefaultGrammar
def drop(n: int, xs: list) -> list:
    """Drop first n elements: (drop n xs)"""
    return xs[n:]

@DefaultGrammar
def droplast(n: int, xs: list) -> list:
    """Drop last n elements: (droplast n xs)"""
    if n == 0:
        return xs
    return xs[:-n]

# list slicing
@DefaultGrammar
def take(n: int, xs: list) -> list:
    """Take first n elements: (take n xs)"""
    return xs[:n]

@DefaultGrammar
def takelast(n: int, xs: list) -> list:
    """Take last n elements: (takelast n xs)"""
    return xs[-n:] if n > 0 else []

@DefaultGrammar(name='slice')
def slice_fn(i: int, j: int, xs: list) -> list:
    """Slice from i to j: (slice i j xs)"""
    return xs[i:j]

@DefaultGrammar
def cut_slice(i: int, j: int, xs: list) -> list:
    """Remove slice: (cut_slice i j xs)"""
    return xs[:i] + xs[j:]

@DefaultGrammar
def splice(ys: list, i: int, xs: list) -> list:
    """Insert list at index: (splice ys i xs)"""
    return xs[:i] + ys + xs[i:]

# list queries
@DefaultGrammar
def is_in(xs: list, x: object) -> bool:
    """Check membership: (is_in xs x)"""
    return x in xs

@DefaultGrammar
def length(xs: list) -> int:
    """Length of list: (length xs)"""
    return len(xs)

@DefaultGrammar(name='max')
def max_fn(xs: list) -> int:
    """Maximum element: (max xs)"""
    if not xs:
        raise ValueError("max: empty list")
    return max(xs)

@DefaultGrammar(name='min')
def min_fn(xs: list) -> int:
    """Minimum element: (min xs)"""
    if not xs:
        raise ValueError("min: empty list")
    return min(xs)

@DefaultGrammar
def product(xs: list) -> int:
    """Product of elements: (product xs)"""
    result = 1
    for x in xs:
        result *= x
    return result

@DefaultGrammar(name='sum')
def sum_fn(xs: list) -> int:
    """Sum of elements: (sum xs)"""
    return sum(xs)

# list transformation
@DefaultGrammar(name='reverse')
def reverse_fn(xs: list) -> list:
    """Reverse list: (reverse xs)"""
    return list(reversed(xs))

@DefaultGrammar
def flatten(xss: list) -> list:
    """Flatten list of lists: (flatten xss)"""
    result = []
    for xs in xss:
        result.extend(xs)
    return result

# higher-order functions
# Note: Higher-order functions like map, filter, fold require closure support
# from the evaluator. These are placeholders that demonstrate the interface.
@DefaultGrammar
def map(f: Callable[[T1], T2], xs: list[T1]) -> list[T2]:
    """Map function over list: (map f xs)"""
    return [f(x) for x in xs]

@DefaultGrammar
def mapi(f: Callable[[T1, int], T2], xs: list[T1]) -> list[T2]:
    """Map with index: (mapi f xs)"""
    result: list[T2] = []
    for i, x in enumerate(xs):
        result.append(f(x, i))
    return result

@DefaultGrammar
def filter(p: Callable[[T1], bool], xs: list[T1]) -> list[T1]:
    """Filter list by predicate: (filter p xs)"""
    result: list[T1] = []
    for x in xs:
        if p(x):
            result.append(x)
    return result

@DefaultGrammar
def filteri(p: Callable[[int, T1], bool], xs: list[T1]) -> list[T1]:
    """Filter with index: (filteri p xs)"""
    result: list[T1] = []
    for i, x in enumerate(xs):
        if p(i, x):
            result.append(x)
    return result

@DefaultGrammar
def fold(f: Callable[[T2, T1], T2], acc: T2, xs: list[T1]) -> T2:
    """Fold/reduce list: (fold f acc xs)"""
    for x in xs:
        acc = f(acc, x)
    return acc

@DefaultGrammar
def foldi(f: Callable[[T2, T1, int], T2], acc: T2, xs: list[T1]) -> T2:
    """Fold with index: (foldi f acc xs)"""
    for i, x in enumerate(xs):
        acc = f(acc, x, i)
    return acc

@DefaultGrammar
def count(p: Callable[[T1], bool], xs: list[T1]) -> int:
    """Count matching elements: (count p xs)"""
    c = 0
    for x in xs:
        if p(x):
            c += 1
    return c

@DefaultGrammar
def find(p: Callable[[T1], bool], xs: list[T1]) -> list[int]:
    """Find indices matching predicate: (find p xs)"""
    return [i for i, x in enumerate(xs) if p(x)]

@DefaultGrammar
def sort(f: Callable[[T1], int], xs: list[T1]) -> list[T1]:
    """Sort by key function: (sort f xs)"""
    return sorted(xs, key=lambda x: f(x))

@DefaultGrammar
def group(f: Callable[[T1], T2], xs: list[T1]) -> list[list[T1]]:
    """Group by key function: (group f xs)"""
    from collections import defaultdict
    groups = defaultdict(list)
    for x in xs:
        key = f(x)
        key = tuple(key) if isinstance(key, list) else key
        groups[key].append(x)
    return list(groups.values())


if __name__ == "__main__":
    print(DefaultGrammar)
    print(DefaultGrammar['map'])
