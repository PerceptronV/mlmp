from __future__ import annotations
import inspect
import random
from typing import TypeVar, Callable, Optional
from .type_utils import (
    CallableOrig,
    TypeType,
    isatomic,
    get_origin,
    analyse_function_types,
    matchable,
    SubstitutionTable,
)
from .lexer import SPECIAL_CHARS, IDENT_CHARS


T1, T2 = TypeVar('T1'), TypeVar('T2')

# ============================================================================
# Base Grammar Class
# ============================================================================

class Grammar:
    def __init__(self, functions=None, special_chars=None, ident_chars=None):
        self.functions = functions or {}
        self.special_chars = special_chars or SPECIAL_CHARS
        self.ident_chars = ident_chars or IDENT_CHARS
        self.name_map = {}
    
    def __str__(self):
        return f"""Grammar[ {len(self.special_chars)} special char(s), 
         {len(self.ident_chars)} ident char(s),
         {len(self.functions)} function(s) ]"""

    def __repr__(self):
        return f"""Grammar(special_chars={self.special_chars},
        ident_chars={self.ident_chars},
        functions={tuple(self.functions.keys())})"""
    
    def _analyse(self, name, fn, arg_names=None, arg_types=None, ret_type=None):
        if arg_names is None or arg_types is None or ret_type is None:
            arg_names, arg_types, ret_type = analyse_function_types(fn)
        else:
            assert len(arg_names) == len(arg_types), f"Number of argument names ({len(arg_names)}) does not match number of argument types ({len(arg_types)})"
        return {
            'fn': fn,
            '__call__': self._make_evaluable(name, fn, arg_types),
            'arg_names': arg_names,
            'arg_types': arg_types,
            'ret_type': ret_type
        }
    
    def _make_evaluable(self, name: str, fn: Callable, arg_types: tuple) -> Callable:
        '''Make a function evaluable by the evaluator.'''

        # find the indices of the callable arguments in fn
        callable_indices = [
            e for e, t in enumerate(arg_types)
            if get_origin(t) == CallableOrig
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
    
    def valid_name(self, name: str) -> bool:
        for char in name:
            if not char.isalnum() and char not in self.ident_chars:
                return False
        return True
    
    def add_function(self, name, fn, arg_names=None, arg_types=None, ret_type=None):
        if not self.valid_name(name):
            raise ValueError(f"Invalid function name: {name}. Chars must be alphanumeric or one of {self.ident_chars}")
        self.functions[name] = self._analyse(name, fn, arg_names, arg_types, ret_type)
    
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
    
    def shuffle_names(self, bindings: list=None):
        function_names = self.functions.keys()
        new_names = bindings or list(function_names)
        random.shuffle(new_names)
        self.name_map = dict(zip(new_names, function_names))
    
    def use_names(self, bindings: dict):
        self.name_map = bindings.copy()
    
    def get_names(self):
        return self.name_map.copy()
    
    def reset_names(self):
        self.name_map = {}

    @property
    def names(self):
        return tuple(self.name_map.keys() or self.functions.keys())
    
    def __getitem__(self, name):
        name = self.name_map.get(name, name)
        return self.functions[name]
    
    def __len__(self):
        return len(self.names)
    
    def __iter__(self):
        for n in self.names:
            yield n, self[n]['__call__']
    
    def find_matching_functions(
        self, /, *,
        arg_types: Optional[list[TypeType]] = None,
        ret_type: Optional[TypeType] = None,
        substitutions: SubstitutionTable,
    ) -> list[tuple[str, SubstitutionTable]]:
        matches = []
        for n in self.names:
            sub = substitutions.copy()
            if (arg_types is None or matchable(self[n]['arg_types'], arg_types, sub)) and \
                (ret_type is None or matchable(self[n]['ret_type'], ret_type, sub)):
                matches.append((n, sub))
        return matches
    
    @property
    def atomic_return_types(self):
        artypes = set()
        for n in self.names:
            ret = self[n]['ret_type']
            if isatomic(ret):
                artypes.add(ret)
        return artypes

    def subset(self, function_names: set[str]) -> 'Grammar':
        """
        Create a new Grammar containing only the specified functions.

        Args:
            function_names: Set of function names to include

        Returns:
            A new Grammar with only the specified functions
        """
        # Filter to functions that exist in this grammar
        available = function_names & set(self.names)

        # Create new grammar with subset of functions
        new_functions = {}
        for name in available:
            # Get the actual function name (handle name mapping)
            actual_name = self.name_map.get(name, name)
            if actual_name in self.functions:
                new_functions[name] = self.functions[actual_name]

        return Grammar(
            functions=new_functions,
            special_chars=self.special_chars,
            ident_chars=self.ident_chars
        )


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
def equals(x: T1, y: T1) -> bool:
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
def singleton(x: T1) -> list[T1]:
    """Create single-element list: (singleton x)"""
    return [x]

@DefaultGrammar
def repeat(x: T1, n: int) -> list[T1]:
    """Repeat element n times: (repeat x n)"""
    return [x] * n

@DefaultGrammar(name='range')
def range_fn(start: int, end: int, step: int) -> list[int]:
    """Range of numbers: (range i j n)"""
    return list(range(start, end + 1, step))

@DefaultGrammar
def cons(x: T1, xs: list[T1]) -> list[T1]:
    """Prepend element: (cons x xs)"""
    return [x] + xs

# list combination
@DefaultGrammar
def append(xs: list[T1], x: T1) -> list[T1]:
    """Append element: (append xs x)"""
    return xs + [x]

@DefaultGrammar
def concat(xs: list[T1], ys: list[T1]) -> list[T1]:
    """Concatenate lists: (concat xs ys)"""
    return xs + ys

@DefaultGrammar(name='zip')
def zip_fn(xs: list[T1], ys: list[T1]) -> list[list[T1]]:
    """Zip two lists: (zip xs ys)"""
    return [[x, y] for x, y in zip(xs, ys)]

# list access
@DefaultGrammar
def first(xs: list[T1]) -> T1:
    """First element: (first xs)"""
    if not xs:
        raise ValueError("first: empty list")
    return xs[0]

@DefaultGrammar
def second(xs: list[T1]) -> T1:
    """Second element: (second xs)"""
    if len(xs) < 2:
        raise ValueError("second: list too short")
    return xs[1]

@DefaultGrammar
def third(xs: list[T1]) -> T1:
    """Third element: (third xs)"""
    if len(xs) < 3:
        raise ValueError("third: list too short")
    return xs[2]

@DefaultGrammar
def last(xs: list[T1]) -> T1:
    """Last element: (last xs)"""
    if not xs:
        raise ValueError("last: empty list")
    return xs[-1]

@DefaultGrammar
def nth(i: int, xs: list[T1]) -> T1:
    """Nth element: (nth i xs)"""
    if i < 0 or i >= len(xs):
        raise ValueError(f"nth: index {i} out of bounds")
    return xs[i]

# list modification
@DefaultGrammar
def insert(x: T1, i: int, xs: list[T1]) -> list[T1]:
    """Insert at index: (insert x i xs)"""
    result = xs.copy()
    result.insert(i, x)
    return result

@DefaultGrammar
def replace(i: int, x: T1, xs: list[T1]) -> list[T1]:
    """Replace at index: (replace i x xs)"""
    if i < 0 or i >= len(xs):
        raise ValueError(f"replace: index {i} out of bounds")
    result = xs.copy()
    result[i] = x
    return result

@DefaultGrammar
def swap(i: int, j: int, xs: list[T1]) -> list[T1]:
    """Swap elements: (swap i j xs)"""
    if i < 0 or i >= len(xs) or j < 0 or j >= len(xs):
        raise ValueError("swap: index out of bounds")
    result = xs.copy()
    result[i], result[j] = result[j], result[i]
    return result

# list removal
@DefaultGrammar
def cut_idx(i: int, xs: list[T1]) -> list[T1]:
    """Remove at index: (cut_idx i xs)"""
    if i < 0 or i >= len(xs):
        raise ValueError(f"cut_idx: index {i} out of bounds")
    return xs[:i] + xs[i+1:]

@DefaultGrammar
def cut_val(x: T1, xs: list[T1]) -> list[T1]:
    """Remove first occurrence: (cut_val x xs)"""
    result = xs.copy()
    try:
        result.remove(x)
    except ValueError:
        pass  # Element not found, return original
    return result

@DefaultGrammar
def cut_vals(x: T1, xs: list[T1]) -> list[T1]:
    """Remove all occurrences: (cut_vals x xs)"""
    return [elem for elem in xs if elem != x]

@DefaultGrammar
def drop(n: int, xs: list[T1]) -> list[T1]:
    """Drop first n elements: (drop n xs)"""
    return xs[n:]

@DefaultGrammar
def droplast(n: int, xs: list[T1]) -> list[T1]:
    """Drop last n elements: (droplast n xs)"""
    if n == 0:
        return xs
    return xs[:-n]

# list slicing
@DefaultGrammar
def take(n: int, xs: list[T1]) -> list[T1]:
    """Take first n elements: (take n xs)"""
    return xs[:n]

@DefaultGrammar
def takelast(n: int, xs: list[T1]) -> list[T1]:
    """Take last n elements: (takelast n xs)"""
    return xs[-n:] if n > 0 else []

@DefaultGrammar(name='slice')
def slice_fn(i: int, j: int, xs: list[T1]) -> list[T1]:
    """Slice from i to j: (slice i j xs)"""
    return xs[i:j]

@DefaultGrammar
def cut_slice(i: int, j: int, xs: list[T1]) -> list[T1]:
    """Remove slice: (cut_slice i j xs)"""
    return xs[:i] + xs[j:]

@DefaultGrammar
def splice(ys: list[T1], i: int, xs: list[T1]) -> list[T1]:
    """Insert list at index: (splice ys i xs)"""
    return xs[:i] + ys + xs[i:]

# list queries
@DefaultGrammar
def is_in(xs: list[T1], x: T1) -> bool:
    """Check membership: (is_in xs x)"""
    return x in xs

@DefaultGrammar
def length(xs: list[T1]) -> int:
    """Length of list: (length xs)"""
    return len(xs)

@DefaultGrammar(name='max')
def max_fn(xs: list[int]) -> int:
    """Maximum element: (max xs)"""
    if not xs:
        raise ValueError("max: empty list")
    return max(xs)

@DefaultGrammar(name='min')
def min_fn(xs: list[int]) -> int:
    """Minimum element: (min xs)"""
    if not xs:
        raise ValueError("min: empty list")
    return min(xs)

@DefaultGrammar
def product(xs: list[int]) -> int:
    """Product of elements: (product xs)"""
    result = 1
    for x in xs:
        result *= x
    return result

@DefaultGrammar(name='sum')
def sum_fn(xs: list[int]) -> int:
    """Sum of elements: (sum xs)"""
    return sum(xs)

# list transformation
@DefaultGrammar(name='reverse')
def reverse_fn(xs: list[T1]) -> list[T1]:
    """Reverse list: (reverse xs)"""
    return list(reversed(xs))

@DefaultGrammar
def flatten(xss: list[list[T1]]) -> list[T1]:
    """Flatten list of lists: (flatten xss)"""
    result = []
    for xs in xss:
        result.extend(xs)
    return result

# higher-order functions
@DefaultGrammar
def map(f: Callable[[T1], T2], xs: list[T1]) -> list[T2]: # type: ignore
    """Map function over list: (map f xs)"""
    return [f(x) for x in xs]

@DefaultGrammar
def mapi(f: Callable[[T1, int], T2], xs: list[T1]) -> list[T2]: # pyright: ignore[reportInvalidTypeForm]
    """Map with index: (mapi f xs)"""
    result: list[T2] = [] # pyright: ignore[reportInvalidTypeForm]
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
def unique(xs: list[T1]) -> list[T1]:
    """Unique elements: (unique xs)"""
    return list(set(xs))

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
    print('# functions:', len(DefaultGrammar))
    print('map function:', DefaultGrammar['map'])

    print('atomic return types:', DefaultGrammar.atomic_return_types, '\n')

    print(DefaultGrammar.find_matching_functions(ret_type=int, substitutions=SubstitutionTable()))
    print(DefaultGrammar.find_matching_functions(ret_type=list[int], substitutions=SubstitutionTable()))
