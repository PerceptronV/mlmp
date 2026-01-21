"""
Semantic Variations for Meta-Learning

This module defines semantic variations for each grammar function.
Each variation maintains the same type signature but changes the behavior,
creating a more challenging meta-learning task where the model must
learn function semantics from support examples.

Key principles:
1. Variations maintain type signatures
2. Variations are "similar enough" to be composable
3. Variations create meaningful, learnable transformations
4. Each function has 3-8 variants (including the canonical one)
"""

from __future__ import annotations
import random
from typing import TypeVar, Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

T1, T2 = TypeVar('T1'), TypeVar('T2')


@dataclass
class FunctionVariant:
    """A semantic variant of a grammar function."""
    name: str  # The canonical function name this is a variant of
    variant_id: str  # Unique identifier for this variant
    description: str  # Human-readable description
    fn: Callable  # The actual implementation
    # Type info inherited from original
    arg_names: Tuple[str, ...]
    arg_types: Tuple[Any, ...]
    ret_type: Any
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'name': self.name,
            'variant_id': self.variant_id,
            'description': self.description
        }


# =============================================================================
# ARITHMETIC VARIATIONS
# =============================================================================

def _make_add_variants() -> List[FunctionVariant]:
    """Variations of + (addition)."""
    base = {
        'name': '+',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }
    
    return [
        FunctionVariant(
            variant_id='add_canonical',
            description='x + y (standard addition)',
            fn=lambda x, y: x + y,
            **base
        ),
        FunctionVariant(
            variant_id='add_plus_one',
            description='x + y + 1',
            fn=lambda x, y: x + y + 1,
            **base
        ),
        FunctionVariant(
            variant_id='add_max',
            description='max(x, y)',
            fn=lambda x, y: max(x, y),
            **base
        ),
        FunctionVariant(
            variant_id='add_double_first',
            description='2*x + y',
            fn=lambda x, y: 2 * x + y,
            **base
        ),
        FunctionVariant(
            variant_id='add_abs_sum',
            description='|x| + |y|',
            fn=lambda x, y: abs(x) + abs(y),
            **base
        ),
        FunctionVariant(
            variant_id='add_mod_100',
            description='(x + y) mod 100',
            fn=lambda x, y: (x + y) % 100,
            **base
        ),
    ]


def _make_sub_variants() -> List[FunctionVariant]:
    """Variations of - (subtraction)."""
    base = {
        'name': '-',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }
    
    return [
        FunctionVariant(
            variant_id='sub_canonical',
            description='x - y (standard subtraction)',
            fn=lambda x, y: x - y,
            **base
        ),
        FunctionVariant(
            variant_id='sub_reversed',
            description='y - x (reversed operands)',
            fn=lambda x, y: y - x,
            **base
        ),
        FunctionVariant(
            variant_id='sub_abs',
            description='|x - y| (absolute difference)',
            fn=lambda x, y: abs(x - y),
            **base
        ),
        FunctionVariant(
            variant_id='sub_min',
            description='min(x, y)',
            fn=lambda x, y: min(x, y),
            **base
        ),
        FunctionVariant(
            variant_id='sub_minus_one',
            description='x - y - 1',
            fn=lambda x, y: x - y - 1,
            **base
        ),
    ]


def _make_mul_variants() -> List[FunctionVariant]:
    """Variations of * (multiplication)."""
    base = {
        'name': '*',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }
    
    return [
        FunctionVariant(
            variant_id='mul_canonical',
            description='x * y (standard multiplication)',
            fn=lambda x, y: x * y,
            **base
        ),
        FunctionVariant(
            variant_id='mul_plus_product',
            description='x * y + x + y',
            fn=lambda x, y: x * y + x + y,
            **base
        ),
        FunctionVariant(
            variant_id='mul_sum_squares',
            description='x*x + y*y',
            fn=lambda x, y: x * x + y * y,
            **base
        ),
        FunctionVariant(
            variant_id='mul_diff_squares',
            description='x*x - y*y',
            fn=lambda x, y: x * x - y * y,
            **base
        ),
        FunctionVariant(
            variant_id='mul_abs',
            description='|x| * |y|',
            fn=lambda x, y: abs(x) * abs(y),
            **base
        ),
    ]


def _make_div_variants() -> List[FunctionVariant]:
    """Variations of / (division)."""
    base = {
        'name': '/',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }
    
    def safe_div(x, y):
        if y == 0:
            raise ValueError("Division by zero")
        return x // y
    
    def safe_div_reversed(x, y):
        if x == 0:
            raise ValueError("Division by zero")
        return y // x
    
    def safe_div_plus_one(x, y):
        if y == 0:
            raise ValueError("Division by zero")
        return x // y + 1
    
    def safe_div_remainder(x, y):
        if y == 0:
            raise ValueError("Division by zero")
        return x // y + (x % y)
    
    return [
        FunctionVariant(
            variant_id='div_canonical',
            description='x // y (integer division)',
            fn=safe_div,
            **base
        ),
        FunctionVariant(
            variant_id='div_reversed',
            description='y // x (reversed operands)',
            fn=safe_div_reversed,
            **base
        ),
        FunctionVariant(
            variant_id='div_plus_one',
            description='x // y + 1',
            fn=safe_div_plus_one,
            **base
        ),
        FunctionVariant(
            variant_id='div_with_remainder',
            description='x // y + (x % y)',
            fn=safe_div_remainder,
            **base
        ),
    ]


def _make_mod_variants() -> List[FunctionVariant]:
    """Variations of % (modulo)."""
    base = {
        'name': '%',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }
    
    def safe_mod(x, y):
        if y == 0:
            raise ValueError("Modulo by zero")
        return x % y
    
    def safe_mod_reversed(x, y):
        if x == 0:
            raise ValueError("Modulo by zero")
        return y % x
    
    def safe_mod_abs(x, y):
        if y == 0:
            raise ValueError("Modulo by zero")
        return abs(x) % abs(y) if y != 0 else abs(x) % 1
    
    def safe_mod_plus_one(x, y):
        if y == 0:
            raise ValueError("Modulo by zero")
        return (x % y) + 1
    
    return [
        FunctionVariant(
            variant_id='mod_canonical',
            description='x % y (standard modulo)',
            fn=safe_mod,
            **base
        ),
        FunctionVariant(
            variant_id='mod_reversed',
            description='y % x (reversed operands)',
            fn=safe_mod_reversed,
            **base
        ),
        FunctionVariant(
            variant_id='mod_abs',
            description='|x| % |y|',
            fn=safe_mod_abs,
            **base
        ),
        FunctionVariant(
            variant_id='mod_plus_one',
            description='(x % y) + 1',
            fn=safe_mod_plus_one,
            **base
        ),
    ]


# =============================================================================
# COMPARISON VARIATIONS
# =============================================================================

def _make_less_than_variants() -> List[FunctionVariant]:
    """Variations of < (less than)."""
    base = {
        'name': '<',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='lt_canonical',
            description='x < y (standard less than)',
            fn=lambda x, y: x < y,
            **base
        ),
        FunctionVariant(
            variant_id='lt_leq',
            description='x <= y (less than or equal)',
            fn=lambda x, y: x <= y,
            **base
        ),
        FunctionVariant(
            variant_id='lt_gt',
            description='x > y (inverted to greater than)',
            fn=lambda x, y: x > y,
            **base
        ),
        FunctionVariant(
            variant_id='lt_neq',
            description='x != y (not equal)',
            fn=lambda x, y: x != y,
            **base
        ),
        FunctionVariant(
            variant_id='lt_abs',
            description='|x| < |y|',
            fn=lambda x, y: abs(x) < abs(y),
            **base
        ),
    ]


def _make_greater_than_variants() -> List[FunctionVariant]:
    """Variations of > (greater than)."""
    base = {
        'name': '>',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='gt_canonical',
            description='x > y (standard greater than)',
            fn=lambda x, y: x > y,
            **base
        ),
        FunctionVariant(
            variant_id='gt_geq',
            description='x >= y (greater than or equal)',
            fn=lambda x, y: x >= y,
            **base
        ),
        FunctionVariant(
            variant_id='gt_lt',
            description='x < y (inverted to less than)',
            fn=lambda x, y: x < y,
            **base
        ),
        FunctionVariant(
            variant_id='gt_abs',
            description='|x| > |y|',
            fn=lambda x, y: abs(x) > abs(y),
            **base
        ),
    ]


def _make_equals_variants() -> List[FunctionVariant]:
    """Variations of == (equality)."""
    base = {
        'name': '==',
        'arg_names': ('x', 'y'),
        'arg_types': (T1, T1),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='eq_canonical',
            description='x == y (standard equality)',
            fn=lambda x, y: x == y,
            **base
        ),
        FunctionVariant(
            variant_id='eq_neq',
            description='x != y (inverted to not equal)',
            fn=lambda x, y: x != y,
            **base
        ),
        FunctionVariant(
            variant_id='eq_both_truthy',
            description='bool(x) == bool(y)',
            fn=lambda x, y: bool(x) == bool(y),
            **base
        ),
    ]


# =============================================================================
# BOOLEAN VARIATIONS
# =============================================================================

def _make_and_variants() -> List[FunctionVariant]:
    """Variations of and (boolean and)."""
    base = {
        'name': 'and',
        'arg_names': ('x', 'y'),
        'arg_types': (bool, bool),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='and_canonical',
            description='x and y (standard and)',
            fn=lambda x, y: x and y,
            **base
        ),
        FunctionVariant(
            variant_id='and_or',
            description='x or y (switched to or)',
            fn=lambda x, y: x or y,
            **base
        ),
        FunctionVariant(
            variant_id='and_nand',
            description='not (x and y) (NAND)',
            fn=lambda x, y: not (x and y),
            **base
        ),
        FunctionVariant(
            variant_id='and_xor',
            description='x xor y (exclusive or)',
            fn=lambda x, y: x != y,
            **base
        ),
        FunctionVariant(
            variant_id='and_implies',
            description='(not x) or y (implication)',
            fn=lambda x, y: (not x) or y,
            **base
        ),
    ]


def _make_or_variants() -> List[FunctionVariant]:
    """Variations of or (boolean or)."""
    base = {
        'name': 'or',
        'arg_names': ('x', 'y'),
        'arg_types': (bool, bool),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='or_canonical',
            description='x or y (standard or)',
            fn=lambda x, y: x or y,
            **base
        ),
        FunctionVariant(
            variant_id='or_and',
            description='x and y (switched to and)',
            fn=lambda x, y: x and y,
            **base
        ),
        FunctionVariant(
            variant_id='or_nor',
            description='not (x or y) (NOR)',
            fn=lambda x, y: not (x or y),
            **base
        ),
        FunctionVariant(
            variant_id='or_xnor',
            description='x == y (XNOR, equivalence)',
            fn=lambda x, y: x == y,
            **base
        ),
    ]


def _make_not_variants() -> List[FunctionVariant]:
    """Variations of not (boolean not)."""
    base = {
        'name': 'not',
        'arg_names': ('x',),
        'arg_types': (bool,),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='not_canonical',
            description='not x (standard not)',
            fn=lambda x: not x,
            **base
        ),
        FunctionVariant(
            variant_id='not_identity',
            description='x (identity, no negation)',
            fn=lambda x: x,
            **base
        ),
        FunctionVariant(
            variant_id='not_always_true',
            description='True (constant true)',
            fn=lambda x: True,
            **base
        ),
        FunctionVariant(
            variant_id='not_always_false',
            description='False (constant false)',
            fn=lambda x: False,
            **base
        ),
    ]


# =============================================================================
# NUMBER PREDICATE VARIATIONS
# =============================================================================

def _make_is_even_variants() -> List[FunctionVariant]:
    """Variations of is_even."""
    base = {
        'name': 'is_even',
        'arg_names': ('n',),
        'arg_types': (int,),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='is_even_canonical',
            description='n % 2 == 0 (standard is_even)',
            fn=lambda n: n % 2 == 0,
            **base
        ),
        FunctionVariant(
            variant_id='is_even_odd',
            description='n % 2 == 1 (is_odd)',
            fn=lambda n: n % 2 == 1,
            **base
        ),
        FunctionVariant(
            variant_id='is_even_div3',
            description='n % 3 == 0 (divisible by 3)',
            fn=lambda n: n % 3 == 0,
            **base
        ),
        FunctionVariant(
            variant_id='is_even_positive',
            description='n > 0 (is positive)',
            fn=lambda n: n > 0,
            **base
        ),
        FunctionVariant(
            variant_id='is_even_negative',
            description='n < 0 (is negative)',
            fn=lambda n: n < 0,
            **base
        ),
    ]


def _make_is_odd_variants() -> List[FunctionVariant]:
    """Variations of is_odd."""
    base = {
        'name': 'is_odd',
        'arg_names': ('n',),
        'arg_types': (int,),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='is_odd_canonical',
            description='n % 2 == 1 (standard is_odd)',
            fn=lambda n: n % 2 == 1,
            **base
        ),
        FunctionVariant(
            variant_id='is_odd_even',
            description='n % 2 == 0 (is_even)',
            fn=lambda n: n % 2 == 0,
            **base
        ),
        FunctionVariant(
            variant_id='is_odd_div3_not',
            description='n % 3 != 0 (not divisible by 3)',
            fn=lambda n: n % 3 != 0,
            **base
        ),
        FunctionVariant(
            variant_id='is_odd_nonzero',
            description='n != 0 (is nonzero)',
            fn=lambda n: n != 0,
            **base
        ),
    ]


# =============================================================================
# LIST CONSTRUCTION VARIATIONS
# =============================================================================

def _make_singleton_variants() -> List[FunctionVariant]:
    """Variations of singleton."""
    base = {
        'name': 'singleton',
        'arg_names': ('x',),
        'arg_types': (T1,),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='singleton_canonical',
            description='[x] (standard singleton)',
            fn=lambda x: [x],
            **base
        ),
        FunctionVariant(
            variant_id='singleton_double',
            description='[x, x] (duplicate element)',
            fn=lambda x: [x, x],
            **base
        ),
        FunctionVariant(
            variant_id='singleton_triple',
            description='[x, x, x] (triple element)',
            fn=lambda x: [x, x, x],
            **base
        ),
    ]


def _make_repeat_variants() -> List[FunctionVariant]:
    """Variations of repeat."""
    base = {
        'name': 'repeat',
        'arg_names': ('x', 'n'),
        'arg_types': (T1, int),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='repeat_canonical',
            description='[x] * n (standard repeat)',
            fn=lambda x, n: [x] * n,
            **base
        ),
        FunctionVariant(
            variant_id='repeat_plus_one',
            description='[x] * (n + 1)',
            fn=lambda x, n: [x] * (n + 1),
            **base
        ),
        FunctionVariant(
            variant_id='repeat_minus_one',
            description='[x] * max(0, n - 1)',
            fn=lambda x, n: [x] * max(0, n - 1),
            **base
        ),
        FunctionVariant(
            variant_id='repeat_double',
            description='[x] * (2 * n)',
            fn=lambda x, n: [x] * (2 * n),
            **base
        ),
    ]


def _make_range_variants() -> List[FunctionVariant]:
    """Variations of range."""
    base = {
        'name': 'range',
        'arg_names': ('start', 'end', 'step'),
        'arg_types': (int, int, int),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='range_canonical',
            description='range(start, end+1, step) (inclusive end)',
            fn=lambda start, end, step: list(range(start, end + 1, step)) if step > 0 else [],
            **base
        ),
        FunctionVariant(
            variant_id='range_exclusive',
            description='range(start, end, step) (exclusive end)',
            fn=lambda start, end, step: list(range(start, end, step)) if step > 0 else [],
            **base
        ),
        FunctionVariant(
            variant_id='range_reversed',
            description='range(end, start-1, -step) (reversed)',
            fn=lambda start, end, step: list(range(end, start - 1, -step)) if step > 0 else [],
            **base
        ),
        FunctionVariant(
            variant_id='range_double_step',
            description='range(start, end+1, 2*step)',
            fn=lambda start, end, step: list(range(start, end + 1, 2 * step)) if step > 0 else [],
            **base
        ),
    ]


def _make_cons_variants() -> List[FunctionVariant]:
    """Variations of cons (prepend)."""
    base = {
        'name': 'cons',
        'arg_names': ('x', 'xs'),
        'arg_types': (T1, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='cons_canonical',
            description='[x] + xs (prepend)',
            fn=lambda x, xs: [x] + xs,
            **base
        ),
        FunctionVariant(
            variant_id='cons_append',
            description='xs + [x] (append instead)',
            fn=lambda x, xs: xs + [x],
            **base
        ),
        FunctionVariant(
            variant_id='cons_double',
            description='[x, x] + xs (prepend twice)',
            fn=lambda x, xs: [x, x] + xs,
            **base
        ),
        FunctionVariant(
            variant_id='cons_at_one',
            description='xs[:1] + [x] + xs[1:] (insert at position 1)',
            fn=lambda x, xs: xs[:1] + [x] + xs[1:] if xs else [x],
            **base
        ),
    ]


# =============================================================================
# LIST COMBINATION VARIATIONS  
# =============================================================================

def _make_append_variants() -> List[FunctionVariant]:
    """Variations of append."""
    base = {
        'name': 'append',
        'arg_names': ('xs', 'x'),
        'arg_types': (list, T1),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='append_canonical',
            description='xs + [x] (standard append)',
            fn=lambda xs, x: xs + [x],
            **base
        ),
        FunctionVariant(
            variant_id='append_prepend',
            description='[x] + xs (prepend instead)',
            fn=lambda xs, x: [x] + xs,
            **base
        ),
        FunctionVariant(
            variant_id='append_double',
            description='xs + [x, x] (append twice)',
            fn=lambda xs, x: xs + [x, x],
            **base
        ),
    ]


def _make_concat_variants() -> List[FunctionVariant]:
    """Variations of concat."""
    base = {
        'name': 'concat',
        'arg_names': ('xs', 'ys'),
        'arg_types': (list, list),
        'ret_type': list
    }
    
    def interleave(xs, ys):
        result = []
        for i in range(max(len(xs), len(ys))):
            if i < len(xs):
                result.append(xs[i])
            if i < len(ys):
                result.append(ys[i])
        return result
    
    return [
        FunctionVariant(
            variant_id='concat_canonical',
            description='xs + ys (standard concat)',
            fn=lambda xs, ys: xs + ys,
            **base
        ),
        FunctionVariant(
            variant_id='concat_reversed',
            description='ys + xs (reversed order)',
            fn=lambda xs, ys: ys + xs,
            **base
        ),
        FunctionVariant(
            variant_id='concat_interleave',
            description='interleave(xs, ys)',
            fn=interleave,
            **base
        ),
        FunctionVariant(
            variant_id='concat_first_only',
            description='xs (ignore second list)',
            fn=lambda xs, ys: xs,
            **base
        ),
    ]


def _make_zip_variants() -> List[FunctionVariant]:
    """Variations of zip."""
    base = {
        'name': 'zip',
        'arg_names': ('xs', 'ys'),
        'arg_types': (list, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='zip_canonical',
            description='[[x,y] for x,y in zip(xs,ys)]',
            fn=lambda xs, ys: [[x, y] for x, y in zip(xs, ys)],
            **base
        ),
        FunctionVariant(
            variant_id='zip_reversed',
            description='[[y,x] for x,y in zip(xs,ys)] (reversed pairs)',
            fn=lambda xs, ys: [[y, x] for x, y in zip(xs, ys)],
            **base
        ),
        FunctionVariant(
            variant_id='zip_sum',
            description='[x+y for x,y in zip(xs,ys)] (sum pairs)',
            fn=lambda xs, ys: [x + y for x, y in zip(xs, ys)],
            **base
        ),
        FunctionVariant(
            variant_id='zip_diff',
            description='[x-y for x,y in zip(xs,ys)] (difference pairs)',
            fn=lambda xs, ys: [x - y for x, y in zip(xs, ys)],
            **base
        ),
    ]


# =============================================================================
# LIST ACCESS VARIATIONS
# =============================================================================

def _make_first_variants() -> List[FunctionVariant]:
    """Variations of first."""
    base = {
        'name': 'first',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': T1
    }
    
    def safe_first(xs):
        if not xs:
            raise ValueError("first: empty list")
        return xs[0]
    
    def safe_last(xs):
        if not xs:
            raise ValueError("first: empty list")
        return xs[-1]
    
    def safe_second_or_first(xs):
        if not xs:
            raise ValueError("first: empty list")
        return xs[1] if len(xs) > 1 else xs[0]
    
    def safe_middle(xs):
        if not xs:
            raise ValueError("first: empty list")
        return xs[len(xs) // 2]
    
    return [
        FunctionVariant(
            variant_id='first_canonical',
            description='xs[0] (standard first)',
            fn=safe_first,
            **base
        ),
        FunctionVariant(
            variant_id='first_last',
            description='xs[-1] (last element)',
            fn=safe_last,
            **base
        ),
        FunctionVariant(
            variant_id='first_second',
            description='xs[1] if len>1 else xs[0]',
            fn=safe_second_or_first,
            **base
        ),
        FunctionVariant(
            variant_id='first_middle',
            description='xs[len//2] (middle element)',
            fn=safe_middle,
            **base
        ),
    ]


def _make_second_variants() -> List[FunctionVariant]:
    """Variations of second."""
    base = {
        'name': 'second',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': T1
    }
    
    def safe_second(xs):
        if len(xs) < 2:
            raise ValueError("second: list too short")
        return xs[1]
    
    def safe_first(xs):
        if len(xs) < 2:
            raise ValueError("second: list too short")
        return xs[0]
    
    def safe_third_or_second(xs):
        if len(xs) < 2:
            raise ValueError("second: list too short")
        return xs[2] if len(xs) > 2 else xs[1]
    
    def safe_second_last(xs):
        if len(xs) < 2:
            raise ValueError("second: list too short")
        return xs[-2]
    
    return [
        FunctionVariant(
            variant_id='second_canonical',
            description='xs[1] (standard second)',
            fn=safe_second,
            **base
        ),
        FunctionVariant(
            variant_id='second_first',
            description='xs[0] (first element)',
            fn=safe_first,
            **base
        ),
        FunctionVariant(
            variant_id='second_third',
            description='xs[2] if len>2 else xs[1]',
            fn=safe_third_or_second,
            **base
        ),
        FunctionVariant(
            variant_id='second_last',
            description='xs[-2] (second to last)',
            fn=safe_second_last,
            **base
        ),
    ]


def _make_third_variants() -> List[FunctionVariant]:
    """Variations of third."""
    base = {
        'name': 'third',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': T1
    }
    
    def safe_third(xs):
        if len(xs) < 3:
            raise ValueError("third: list too short")
        return xs[2]
    
    def safe_fourth_or_third(xs):
        if len(xs) < 3:
            raise ValueError("third: list too short")
        return xs[3] if len(xs) > 3 else xs[2]
    
    def safe_second(xs):
        if len(xs) < 3:
            raise ValueError("third: list too short")
        return xs[1]
    
    def safe_third_last(xs):
        if len(xs) < 3:
            raise ValueError("third: list too short")
        return xs[-3]
    
    return [
        FunctionVariant(
            variant_id='third_canonical',
            description='xs[2] (standard third)',
            fn=safe_third,
            **base
        ),
        FunctionVariant(
            variant_id='third_fourth',
            description='xs[3] if len>3 else xs[2]',
            fn=safe_fourth_or_third,
            **base
        ),
        FunctionVariant(
            variant_id='third_second',
            description='xs[1] (second element)',
            fn=safe_second,
            **base
        ),
        FunctionVariant(
            variant_id='third_last',
            description='xs[-3] (third to last)',
            fn=safe_third_last,
            **base
        ),
    ]


def _make_last_variants() -> List[FunctionVariant]:
    """Variations of last."""
    base = {
        'name': 'last',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': T1
    }
    
    def safe_last(xs):
        if not xs:
            raise ValueError("last: empty list")
        return xs[-1]
    
    def safe_first(xs):
        if not xs:
            raise ValueError("last: empty list")
        return xs[0]
    
    def safe_second_last(xs):
        if not xs:
            raise ValueError("last: empty list")
        return xs[-2] if len(xs) > 1 else xs[-1]
    
    return [
        FunctionVariant(
            variant_id='last_canonical',
            description='xs[-1] (standard last)',
            fn=safe_last,
            **base
        ),
        FunctionVariant(
            variant_id='last_first',
            description='xs[0] (first element)',
            fn=safe_first,
            **base
        ),
        FunctionVariant(
            variant_id='last_second_last',
            description='xs[-2] if len>1 else xs[-1]',
            fn=safe_second_last,
            **base
        ),
    ]


def _make_nth_variants() -> List[FunctionVariant]:
    """Variations of nth."""
    base = {
        'name': 'nth',
        'arg_names': ('i', 'xs'),
        'arg_types': (int, list),
        'ret_type': T1
    }
    
    def safe_nth(i, xs):
        if i < 0 or i >= len(xs):
            raise ValueError(f"nth: index {i} out of bounds")
        return xs[i]
    
    def safe_nth_from_end(i, xs):
        if i < 0 or i >= len(xs):
            raise ValueError(f"nth: index {i} out of bounds")
        return xs[-(i + 1)]
    
    def safe_nth_plus_one(i, xs):
        idx = (i + 1) % len(xs) if xs else 0
        if not xs:
            raise ValueError(f"nth: empty list")
        return xs[idx]
    
    def safe_nth_cyclic(i, xs):
        if not xs:
            raise ValueError(f"nth: empty list")
        return xs[i % len(xs)]
    
    return [
        FunctionVariant(
            variant_id='nth_canonical',
            description='xs[i] (standard nth)',
            fn=safe_nth,
            **base
        ),
        FunctionVariant(
            variant_id='nth_from_end',
            description='xs[-(i+1)] (from end)',
            fn=safe_nth_from_end,
            **base
        ),
        FunctionVariant(
            variant_id='nth_plus_one',
            description='xs[(i+1) % len]',
            fn=safe_nth_plus_one,
            **base
        ),
        FunctionVariant(
            variant_id='nth_cyclic',
            description='xs[i % len] (cyclic)',
            fn=safe_nth_cyclic,
            **base
        ),
    ]


# =============================================================================
# LIST MODIFICATION VARIATIONS
# =============================================================================

def _make_insert_variants() -> List[FunctionVariant]:
    """Variations of insert."""
    base = {
        'name': 'insert',
        'arg_names': ('x', 'i', 'xs'),
        'arg_types': (T1, int, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='insert_canonical',
            description='insert x at position i',
            fn=lambda x, i, xs: xs[:i] + [x] + xs[i:],
            **base
        ),
        FunctionVariant(
            variant_id='insert_from_end',
            description='insert x at position len-i',
            fn=lambda x, i, xs: xs[:len(xs)-i] + [x] + xs[len(xs)-i:] if i <= len(xs) else [x] + xs,
            **base
        ),
        FunctionVariant(
            variant_id='insert_double',
            description='insert [x, x] at position i',
            fn=lambda x, i, xs: xs[:i] + [x, x] + xs[i:],
            **base
        ),
    ]


def _make_replace_variants() -> List[FunctionVariant]:
    """Variations of replace."""
    base = {
        'name': 'replace',
        'arg_names': ('i', 'x', 'xs'),
        'arg_types': (int, T1, list),
        'ret_type': list
    }
    
    def safe_replace(i, x, xs):
        if i < 0 or i >= len(xs):
            raise ValueError(f"replace: index {i} out of bounds")
        result = xs.copy()
        result[i] = x
        return result
    
    def safe_replace_from_end(i, x, xs):
        idx = len(xs) - 1 - i
        if idx < 0 or idx >= len(xs):
            raise ValueError(f"replace: index {i} out of bounds")
        result = xs.copy()
        result[idx] = x
        return result
    
    def safe_replace_and_next(i, x, xs):
        if i < 0 or i >= len(xs):
            raise ValueError(f"replace: index {i} out of bounds")
        result = xs.copy()
        result[i] = x
        if i + 1 < len(xs):
            result[i + 1] = x
        return result
    
    return [
        FunctionVariant(
            variant_id='replace_canonical',
            description='replace xs[i] with x',
            fn=safe_replace,
            **base
        ),
        FunctionVariant(
            variant_id='replace_from_end',
            description='replace xs[-(i+1)] with x',
            fn=safe_replace_from_end,
            **base
        ),
        FunctionVariant(
            variant_id='replace_and_next',
            description='replace xs[i] and xs[i+1] with x',
            fn=safe_replace_and_next,
            **base
        ),
    ]


def _make_swap_variants() -> List[FunctionVariant]:
    """Variations of swap."""
    base = {
        'name': 'swap',
        'arg_names': ('i', 'j', 'xs'),
        'arg_types': (int, int, list),
        'ret_type': list
    }
    
    def safe_swap(i, j, xs):
        if i < 0 or i >= len(xs) or j < 0 or j >= len(xs):
            raise ValueError("swap: index out of bounds")
        result = xs.copy()
        result[i], result[j] = result[j], result[i]
        return result
    
    def safe_swap_neighbors(i, j, xs):
        # Swap i with i+1 and j with j+1 if possible
        if i < 0 or i >= len(xs) or j < 0 or j >= len(xs):
            raise ValueError("swap: index out of bounds")
        result = xs.copy()
        if i + 1 < len(xs):
            result[i], result[i + 1] = result[i + 1], result[i]
        return result
    
    def safe_reverse_between(i, j, xs):
        if i < 0 or i >= len(xs) or j < 0 or j >= len(xs):
            raise ValueError("swap: index out of bounds")
        lo, hi = min(i, j), max(i, j)
        return xs[:lo] + xs[lo:hi+1][::-1] + xs[hi+1:]
    
    return [
        FunctionVariant(
            variant_id='swap_canonical',
            description='swap xs[i] and xs[j]',
            fn=safe_swap,
            **base
        ),
        FunctionVariant(
            variant_id='swap_neighbors',
            description='swap xs[i] with xs[i+1]',
            fn=safe_swap_neighbors,
            **base
        ),
        FunctionVariant(
            variant_id='swap_reverse_between',
            description='reverse xs[i:j+1]',
            fn=safe_reverse_between,
            **base
        ),
    ]


# =============================================================================
# LIST REMOVAL VARIATIONS
# =============================================================================

def _make_cut_idx_variants() -> List[FunctionVariant]:
    """Variations of cut_idx."""
    base = {
        'name': 'cut_idx',
        'arg_names': ('i', 'xs'),
        'arg_types': (int, list),
        'ret_type': list
    }
    
    def safe_cut_idx(i, xs):
        if i < 0 or i >= len(xs):
            raise ValueError(f"cut_idx: index {i} out of bounds")
        return xs[:i] + xs[i+1:]
    
    def safe_cut_from_end(i, xs):
        idx = len(xs) - 1 - i
        if idx < 0 or idx >= len(xs):
            raise ValueError(f"cut_idx: index {i} out of bounds")
        return xs[:idx] + xs[idx+1:]
    
    def safe_cut_two(i, xs):
        if i < 0 or i >= len(xs):
            raise ValueError(f"cut_idx: index {i} out of bounds")
        result = xs[:i] + xs[i+1:]
        if i < len(result):
            result = result[:i] + result[i+1:]
        return result
    
    return [
        FunctionVariant(
            variant_id='cut_idx_canonical',
            description='remove element at index i',
            fn=safe_cut_idx,
            **base
        ),
        FunctionVariant(
            variant_id='cut_idx_from_end',
            description='remove element at index -(i+1)',
            fn=safe_cut_from_end,
            **base
        ),
        FunctionVariant(
            variant_id='cut_idx_two',
            description='remove elements at i and i+1',
            fn=safe_cut_two,
            **base
        ),
    ]


def _make_cut_val_variants() -> List[FunctionVariant]:
    """Variations of cut_val."""
    base = {
        'name': 'cut_val',
        'arg_names': ('x', 'xs'),
        'arg_types': (T1, list),
        'ret_type': list
    }
    
    def cut_first(x, xs):
        result = xs.copy()
        try:
            result.remove(x)
        except ValueError:
            pass
        return result
    
    def cut_last(x, xs):
        result = xs.copy()
        for i in range(len(result) - 1, -1, -1):
            if result[i] == x:
                result.pop(i)
                break
        return result
    
    def cut_all(x, xs):
        return [elem for elem in xs if elem != x]
    
    def keep_only(x, xs):
        return [elem for elem in xs if elem == x]
    
    return [
        FunctionVariant(
            variant_id='cut_val_canonical',
            description='remove first occurrence of x',
            fn=cut_first,
            **base
        ),
        FunctionVariant(
            variant_id='cut_val_last',
            description='remove last occurrence of x',
            fn=cut_last,
            **base
        ),
        FunctionVariant(
            variant_id='cut_val_all',
            description='remove all occurrences of x',
            fn=cut_all,
            **base
        ),
        FunctionVariant(
            variant_id='cut_val_keep',
            description='keep only elements equal to x',
            fn=keep_only,
            **base
        ),
    ]


def _make_cut_vals_variants() -> List[FunctionVariant]:
    """Variations of cut_vals."""
    base = {
        'name': 'cut_vals',
        'arg_names': ('x', 'xs'),
        'arg_types': (T1, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='cut_vals_canonical',
            description='remove all occurrences of x',
            fn=lambda x, xs: [elem for elem in xs if elem != x],
            **base
        ),
        FunctionVariant(
            variant_id='cut_vals_first',
            description='remove first occurrence only',
            fn=lambda x, xs: xs[:xs.index(x)] + xs[xs.index(x)+1:] if x in xs else xs,
            **base
        ),
        FunctionVariant(
            variant_id='cut_vals_keep',
            description='keep only elements equal to x',
            fn=lambda x, xs: [elem for elem in xs if elem == x],
            **base
        ),
    ]


def _make_drop_variants() -> List[FunctionVariant]:
    """Variations of drop."""
    base = {
        'name': 'drop',
        'arg_names': ('n', 'xs'),
        'arg_types': (int, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='drop_canonical',
            description='xs[n:] (drop first n)',
            fn=lambda n, xs: xs[n:],
            **base
        ),
        FunctionVariant(
            variant_id='drop_from_end',
            description='xs[:-n] if n>0 else xs (drop last n)',
            fn=lambda n, xs: xs[:-n] if n > 0 else xs,
            **base
        ),
        FunctionVariant(
            variant_id='drop_plus_one',
            description='xs[n+1:]',
            fn=lambda n, xs: xs[n+1:],
            **base
        ),
        FunctionVariant(
            variant_id='drop_every_nth',
            description='keep only elements at indices not divisible by n+1',
            fn=lambda n, xs: [x for i, x in enumerate(xs) if (i + 1) % (n + 1) != 0] if n >= 0 else xs,
            **base
        ),
    ]


def _make_droplast_variants() -> List[FunctionVariant]:
    """Variations of droplast."""
    base = {
        'name': 'droplast',
        'arg_names': ('n', 'xs'),
        'arg_types': (int, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='droplast_canonical',
            description='xs[:-n] if n>0 else xs',
            fn=lambda n, xs: xs[:-n] if n > 0 else xs,
            **base
        ),
        FunctionVariant(
            variant_id='droplast_from_start',
            description='xs[n:] (drop first n)',
            fn=lambda n, xs: xs[n:],
            **base
        ),
        FunctionVariant(
            variant_id='droplast_plus_one',
            description='xs[:-(n+1)] if n>=0 else xs',
            fn=lambda n, xs: xs[:-(n+1)] if n >= 0 and n + 1 <= len(xs) else [],
            **base
        ),
    ]


# =============================================================================
# LIST SLICING VARIATIONS
# =============================================================================

def _make_take_variants() -> List[FunctionVariant]:
    """Variations of take."""
    base = {
        'name': 'take',
        'arg_names': ('n', 'xs'),
        'arg_types': (int, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='take_canonical',
            description='xs[:n] (take first n)',
            fn=lambda n, xs: xs[:n],
            **base
        ),
        FunctionVariant(
            variant_id='take_from_end',
            description='xs[-n:] if n>0 else [] (take last n)',
            fn=lambda n, xs: xs[-n:] if n > 0 else [],
            **base
        ),
        FunctionVariant(
            variant_id='take_plus_one',
            description='xs[:n+1]',
            fn=lambda n, xs: xs[:n+1],
            **base
        ),
        FunctionVariant(
            variant_id='take_every_nth',
            description='take every nth element (first n elements where i%step==0)',
            fn=lambda n, xs: xs[::max(1, (len(xs) // n) if n > 0 else 1)][:n] if n > 0 else [],
            **base
        ),
    ]


def _make_takelast_variants() -> List[FunctionVariant]:
    """Variations of takelast."""
    base = {
        'name': 'takelast',
        'arg_names': ('n', 'xs'),
        'arg_types': (int, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='takelast_canonical',
            description='xs[-n:] if n>0 else [] (take last n)',
            fn=lambda n, xs: xs[-n:] if n > 0 else [],
            **base
        ),
        FunctionVariant(
            variant_id='takelast_from_start',
            description='xs[:n] (take first n)',
            fn=lambda n, xs: xs[:n],
            **base
        ),
        FunctionVariant(
            variant_id='takelast_plus_one',
            description='xs[-(n+1):] if n>=0 else []',
            fn=lambda n, xs: xs[-(n+1):] if n >= 0 else [],
            **base
        ),
    ]


def _make_slice_variants() -> List[FunctionVariant]:
    """Variations of slice."""
    base = {
        'name': 'slice',
        'arg_names': ('i', 'j', 'xs'),
        'arg_types': (int, int, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='slice_canonical',
            description='xs[i:j] (standard slice)',
            fn=lambda i, j, xs: xs[i:j],
            **base
        ),
        FunctionVariant(
            variant_id='slice_inclusive',
            description='xs[i:j+1] (inclusive end)',
            fn=lambda i, j, xs: xs[i:j+1],
            **base
        ),
        FunctionVariant(
            variant_id='slice_reversed',
            description='xs[i:j][::-1] (reversed slice)',
            fn=lambda i, j, xs: xs[i:j][::-1],
            **base
        ),
        FunctionVariant(
            variant_id='slice_complement',
            description='xs[:i] + xs[j:] (complement)',
            fn=lambda i, j, xs: xs[:i] + xs[j:],
            **base
        ),
    ]


def _make_cut_slice_variants() -> List[FunctionVariant]:
    """Variations of cut_slice."""
    base = {
        'name': 'cut_slice',
        'arg_names': ('i', 'j', 'xs'),
        'arg_types': (int, int, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='cut_slice_canonical',
            description='xs[:i] + xs[j:] (remove slice)',
            fn=lambda i, j, xs: xs[:i] + xs[j:],
            **base
        ),
        FunctionVariant(
            variant_id='cut_slice_keep',
            description='xs[i:j] (keep slice instead)',
            fn=lambda i, j, xs: xs[i:j],
            **base
        ),
        FunctionVariant(
            variant_id='cut_slice_inclusive',
            description='xs[:i] + xs[j+1:] (inclusive end)',
            fn=lambda i, j, xs: xs[:i] + xs[j+1:],
            **base
        ),
    ]


def _make_splice_variants() -> List[FunctionVariant]:
    """Variations of splice."""
    base = {
        'name': 'splice',
        'arg_names': ('ys', 'i', 'xs'),
        'arg_types': (list, int, list),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='splice_canonical',
            description='xs[:i] + ys + xs[i:]',
            fn=lambda ys, i, xs: xs[:i] + ys + xs[i:],
            **base
        ),
        FunctionVariant(
            variant_id='splice_replace',
            description='xs[:i] + ys + xs[i+len(ys):]',
            fn=lambda ys, i, xs: xs[:i] + ys + xs[i+len(ys):],
            **base
        ),
        FunctionVariant(
            variant_id='splice_at_end',
            description='xs + ys (append at end)',
            fn=lambda ys, i, xs: xs + ys,
            **base
        ),
    ]


# =============================================================================
# LIST QUERY VARIATIONS
# =============================================================================

def _make_is_in_variants() -> List[FunctionVariant]:
    """Variations of is_in."""
    base = {
        'name': 'is_in',
        'arg_names': ('xs', 'x'),
        'arg_types': (list, T1),
        'ret_type': bool
    }
    
    return [
        FunctionVariant(
            variant_id='is_in_canonical',
            description='x in xs',
            fn=lambda xs, x: x in xs,
            **base
        ),
        FunctionVariant(
            variant_id='is_in_not',
            description='x not in xs',
            fn=lambda xs, x: x not in xs,
            **base
        ),
        FunctionVariant(
            variant_id='is_in_first',
            description='xs[0] == x if xs else False',
            fn=lambda xs, x: xs[0] == x if xs else False,
            **base
        ),
        FunctionVariant(
            variant_id='is_in_last',
            description='xs[-1] == x if xs else False',
            fn=lambda xs, x: xs[-1] == x if xs else False,
            **base
        ),
    ]


def _make_length_variants() -> List[FunctionVariant]:
    """Variations of length."""
    base = {
        'name': 'length',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }
    
    return [
        FunctionVariant(
            variant_id='length_canonical',
            description='len(xs)',
            fn=lambda xs: len(xs),
            **base
        ),
        FunctionVariant(
            variant_id='length_plus_one',
            description='len(xs) + 1',
            fn=lambda xs: len(xs) + 1,
            **base
        ),
        FunctionVariant(
            variant_id='length_minus_one',
            description='max(0, len(xs) - 1)',
            fn=lambda xs: max(0, len(xs) - 1),
            **base
        ),
        FunctionVariant(
            variant_id='length_double',
            description='2 * len(xs)',
            fn=lambda xs: 2 * len(xs),
            **base
        ),
    ]


def _make_max_variants() -> List[FunctionVariant]:
    """Variations of max."""
    base = {
        'name': 'max',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }
    
    def safe_max(xs):
        if not xs:
            raise ValueError("max: empty list")
        return max(xs)
    
    def safe_min(xs):
        if not xs:
            raise ValueError("max: empty list")
        return min(xs)
    
    def safe_second_max(xs):
        if not xs:
            raise ValueError("max: empty list")
        if len(xs) == 1:
            return xs[0]
        sorted_xs = sorted(set(xs), reverse=True)
        return sorted_xs[1] if len(sorted_xs) > 1 else sorted_xs[0]
    
    def safe_sum(xs):
        if not xs:
            raise ValueError("max: empty list")
        return sum(xs)
    
    return [
        FunctionVariant(
            variant_id='max_canonical',
            description='max(xs)',
            fn=safe_max,
            **base
        ),
        FunctionVariant(
            variant_id='max_min',
            description='min(xs)',
            fn=safe_min,
            **base
        ),
        FunctionVariant(
            variant_id='max_second',
            description='second largest',
            fn=safe_second_max,
            **base
        ),
        FunctionVariant(
            variant_id='max_sum',
            description='sum(xs)',
            fn=safe_sum,
            **base
        ),
    ]


def _make_min_variants() -> List[FunctionVariant]:
    """Variations of min."""
    base = {
        'name': 'min',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }
    
    def safe_min(xs):
        if not xs:
            raise ValueError("min: empty list")
        return min(xs)
    
    def safe_max(xs):
        if not xs:
            raise ValueError("min: empty list")
        return max(xs)
    
    def safe_second_min(xs):
        if not xs:
            raise ValueError("min: empty list")
        if len(xs) == 1:
            return xs[0]
        sorted_xs = sorted(set(xs))
        return sorted_xs[1] if len(sorted_xs) > 1 else sorted_xs[0]
    
    def safe_first(xs):
        if not xs:
            raise ValueError("min: empty list")
        return xs[0]
    
    return [
        FunctionVariant(
            variant_id='min_canonical',
            description='min(xs)',
            fn=safe_min,
            **base
        ),
        FunctionVariant(
            variant_id='min_max',
            description='max(xs)',
            fn=safe_max,
            **base
        ),
        FunctionVariant(
            variant_id='min_second',
            description='second smallest',
            fn=safe_second_min,
            **base
        ),
        FunctionVariant(
            variant_id='min_first',
            description='xs[0]',
            fn=safe_first,
            **base
        ),
    ]


def _make_product_variants() -> List[FunctionVariant]:
    """Variations of product."""
    base = {
        'name': 'product',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }
    
    def product(xs):
        result = 1
        for x in xs:
            result *= x
        return result
    
    def product_plus_one(xs):
        result = 1
        for x in xs:
            result *= x
        return result + 1
    
    def product_abs(xs):
        result = 1
        for x in xs:
            result *= abs(x)
        return result
    
    return [
        FunctionVariant(
            variant_id='product_canonical',
            description='product of all elements',
            fn=product,
            **base
        ),
        FunctionVariant(
            variant_id='product_plus_one',
            description='product + 1',
            fn=product_plus_one,
            **base
        ),
        FunctionVariant(
            variant_id='product_abs',
            description='product of absolute values',
            fn=product_abs,
            **base
        ),
        FunctionVariant(
            variant_id='product_sum',
            description='sum instead of product',
            fn=lambda xs: sum(xs),
            **base
        ),
    ]


def _make_sum_variants() -> List[FunctionVariant]:
    """Variations of sum."""
    base = {
        'name': 'sum',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }
    
    def product(xs):
        result = 1
        for x in xs:
            result *= x
        return result
    
    return [
        FunctionVariant(
            variant_id='sum_canonical',
            description='sum(xs)',
            fn=lambda xs: sum(xs),
            **base
        ),
        FunctionVariant(
            variant_id='sum_plus_len',
            description='sum(xs) + len(xs)',
            fn=lambda xs: sum(xs) + len(xs),
            **base
        ),
        FunctionVariant(
            variant_id='sum_abs',
            description='sum of absolute values',
            fn=lambda xs: sum(abs(x) for x in xs),
            **base
        ),
        FunctionVariant(
            variant_id='sum_product',
            description='product instead of sum',
            fn=product,
            **base
        ),
    ]


# =============================================================================
# LIST TRANSFORMATION VARIATIONS
# =============================================================================

def _make_reverse_variants() -> List[FunctionVariant]:
    """Variations of reverse."""
    base = {
        'name': 'reverse',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': list
    }
    
    return [
        FunctionVariant(
            variant_id='reverse_canonical',
            description='xs[::-1] (standard reverse)',
            fn=lambda xs: list(reversed(xs)),
            **base
        ),
        FunctionVariant(
            variant_id='reverse_identity',
            description='xs (no change)',
            fn=lambda xs: xs.copy(),
            **base
        ),
        FunctionVariant(
            variant_id='reverse_rotate',
            description='xs[1:] + xs[:1] (rotate left)',
            fn=lambda xs: xs[1:] + xs[:1] if xs else [],
            **base
        ),
        FunctionVariant(
            variant_id='reverse_rotate_right',
            description='xs[-1:] + xs[:-1] (rotate right)',
            fn=lambda xs: xs[-1:] + xs[:-1] if xs else [],
            **base
        ),
    ]


def _make_flatten_variants() -> List[FunctionVariant]:
    """Variations of flatten."""
    base = {
        'name': 'flatten',
        'arg_names': ('xss',),
        'arg_types': (list,),
        'ret_type': list
    }
    
    def flatten_standard(xss):
        result = []
        for xs in xss:
            result.extend(xs)
        return result
    
    def flatten_reversed(xss):
        result = []
        for xs in reversed(xss):
            result.extend(xs)
        return result
    
    def flatten_first_only(xss):
        return xss[0] if xss else []
    
    def flatten_concat_reversed(xss):
        result = []
        for xs in xss:
            result.extend(reversed(xs))
        return result
    
    return [
        FunctionVariant(
            variant_id='flatten_canonical',
            description='flatten nested lists',
            fn=flatten_standard,
            **base
        ),
        FunctionVariant(
            variant_id='flatten_reversed',
            description='flatten in reverse order',
            fn=flatten_reversed,
            **base
        ),
        FunctionVariant(
            variant_id='flatten_first',
            description='return first sublist only',
            fn=flatten_first_only,
            **base
        ),
        FunctionVariant(
            variant_id='flatten_each_reversed',
            description='flatten with each sublist reversed',
            fn=flatten_concat_reversed,
            **base
        ),
    ]


# =============================================================================
# HIGHER-ORDER FUNCTION VARIATIONS
# =============================================================================

def _make_map_variants() -> List[FunctionVariant]:
    """Variations of map."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'map',
        'arg_names': ('f', 'xs'),
        'arg_types': (CallableOrig[[T1], T2], list),
        'ret_type': list
    }
    
    def map_standard(f, xs):
        return [f(x) for x in xs]
    
    def map_skip_evens(f, xs):
        """Map but skip even indices."""
        return [f(x) if i % 2 == 1 else x for i, x in enumerate(xs)]
    
    def map_skip_odds(f, xs):
        """Map but skip odd indices."""
        return [f(x) if i % 2 == 0 else x for i, x in enumerate(xs)]
    
    def map_twice(f, xs):
        """Apply f twice to each element."""
        return [f(f(x)) for x in xs]
    
    def map_reversed(f, xs):
        """Map then reverse."""
        return [f(x) for x in xs][::-1]
    
    return [
        FunctionVariant(
            variant_id='map_canonical',
            description='[f(x) for x in xs]',
            fn=map_standard,
            **base
        ),
        FunctionVariant(
            variant_id='map_skip_evens',
            description='apply f only to odd indices',
            fn=map_skip_evens,
            **base
        ),
        FunctionVariant(
            variant_id='map_skip_odds',
            description='apply f only to even indices',
            fn=map_skip_odds,
            **base
        ),
        FunctionVariant(
            variant_id='map_twice',
            description='apply f twice: f(f(x))',
            fn=map_twice,
            **base
        ),
        FunctionVariant(
            variant_id='map_reversed',
            description='map then reverse result',
            fn=map_reversed,
            **base
        ),
    ]


def _make_mapi_variants() -> List[FunctionVariant]:
    """Variations of mapi (map with index)."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'mapi',
        'arg_names': ('f', 'xs'),
        'arg_types': (CallableOrig[[T1, int], T2], list),
        'ret_type': list
    }
    
    def mapi_standard(f, xs):
        return [f(x, i) for i, x in enumerate(xs)]
    
    def mapi_reversed_index(f, xs):
        """Index counts down from len-1."""
        return [f(x, len(xs) - 1 - i) for i, x in enumerate(xs)]
    
    def mapi_double_index(f, xs):
        """Index is doubled."""
        return [f(x, 2 * i) for i, x in enumerate(xs)]
    
    def mapi_skip_evens(f, xs):
        """Skip even indices."""
        return [f(x, i) if i % 2 == 1 else x for i, x in enumerate(xs)]
    
    return [
        FunctionVariant(
            variant_id='mapi_canonical',
            description='[f(x, i) for i, x in enumerate(xs)]',
            fn=mapi_standard,
            **base
        ),
        FunctionVariant(
            variant_id='mapi_reversed_index',
            description='index counts down from len-1',
            fn=mapi_reversed_index,
            **base
        ),
        FunctionVariant(
            variant_id='mapi_double_index',
            description='index is doubled',
            fn=mapi_double_index,
            **base
        ),
        FunctionVariant(
            variant_id='mapi_skip_evens',
            description='skip even indices',
            fn=mapi_skip_evens,
            **base
        ),
    ]


def _make_filter_variants() -> List[FunctionVariant]:
    """Variations of filter."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'filter',
        'arg_names': ('p', 'xs'),
        'arg_types': (CallableOrig[[T1], bool], list),
        'ret_type': list
    }
    
    def filter_standard(p, xs):
        return [x for x in xs if p(x)]
    
    def filter_not(p, xs):
        """Keep elements where predicate is False."""
        return [x for x in xs if not p(x)]
    
    def filter_first_n(p, xs, n=3):
        """Keep first n matching elements."""
        result = []
        for x in xs:
            if p(x):
                result.append(x)
                if len(result) >= n:
                    break
        return result
    
    def filter_skip_first(p, xs):
        """Skip first matching element."""
        found_first = False
        result = []
        for x in xs:
            if p(x):
                if found_first:
                    result.append(x)
                found_first = True
            else:
                result.append(x)
        return result
    
    return [
        FunctionVariant(
            variant_id='filter_canonical',
            description='[x for x in xs if p(x)]',
            fn=filter_standard,
            **base
        ),
        FunctionVariant(
            variant_id='filter_not',
            description='keep elements where p(x) is False',
            fn=filter_not,
            **base
        ),
        FunctionVariant(
            variant_id='filter_first_three',
            description='keep first 3 matching elements',
            fn=filter_first_n,
            **base
        ),
        FunctionVariant(
            variant_id='filter_skip_first',
            description='skip first match, keep rest',
            fn=filter_skip_first,
            **base
        ),
    ]


def _make_filteri_variants() -> List[FunctionVariant]:
    """Variations of filteri (filter with index)."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'filteri',
        'arg_names': ('p', 'xs'),
        'arg_types': (CallableOrig[[int, T1], bool], list),
        'ret_type': list
    }
    
    def filteri_standard(p, xs):
        return [x for i, x in enumerate(xs) if p(i, x)]
    
    def filteri_not(p, xs):
        return [x for i, x in enumerate(xs) if not p(i, x)]
    
    def filteri_reversed_index(p, xs):
        return [x for i, x in enumerate(xs) if p(len(xs) - 1 - i, x)]
    
    return [
        FunctionVariant(
            variant_id='filteri_canonical',
            description='[x for i, x if p(i, x)]',
            fn=filteri_standard,
            **base
        ),
        FunctionVariant(
            variant_id='filteri_not',
            description='keep where p(i, x) is False',
            fn=filteri_not,
            **base
        ),
        FunctionVariant(
            variant_id='filteri_reversed_index',
            description='index counts down',
            fn=filteri_reversed_index,
            **base
        ),
    ]


def _make_fold_variants() -> List[FunctionVariant]:
    """Variations of fold (reduce)."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'fold',
        'arg_names': ('f', 'acc', 'xs'),
        'arg_types': (CallableOrig[[T2, T1], T2], T2, list),
        'ret_type': T2
    }
    
    def fold_standard(f, acc, xs):
        for x in xs:
            acc = f(acc, x)
        return acc
    
    def foldr(f, acc, xs):
        """Fold from right."""
        for x in reversed(xs):
            acc = f(acc, x)
        return acc
    
    def fold_skip_evens(f, acc, xs):
        """Skip even indices."""
        for i, x in enumerate(xs):
            if i % 2 == 1:
                acc = f(acc, x)
        return acc
    
    def fold_skip_odds(f, acc, xs):
        """Skip odd indices."""
        for i, x in enumerate(xs):
            if i % 2 == 0:
                acc = f(acc, x)
        return acc
    
    return [
        FunctionVariant(
            variant_id='fold_canonical',
            description='fold left: f(...f(f(acc, x0), x1)..., xn)',
            fn=fold_standard,
            **base
        ),
        FunctionVariant(
            variant_id='fold_right',
            description='fold right: f(x0, f(x1, ...f(xn, acc)))',
            fn=foldr,
            **base
        ),
        FunctionVariant(
            variant_id='fold_skip_evens',
            description='fold only odd indices',
            fn=fold_skip_evens,
            **base
        ),
        FunctionVariant(
            variant_id='fold_skip_odds',
            description='fold only even indices',
            fn=fold_skip_odds,
            **base
        ),
    ]


def _make_foldi_variants() -> List[FunctionVariant]:
    """Variations of foldi (fold with index)."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'foldi',
        'arg_names': ('f', 'acc', 'xs'),
        'arg_types': (CallableOrig[[T2, T1, int], T2], T2, list),
        'ret_type': T2
    }
    
    def foldi_standard(f, acc, xs):
        for i, x in enumerate(xs):
            acc = f(acc, x, i)
        return acc
    
    def foldi_reversed_index(f, acc, xs):
        for i, x in enumerate(xs):
            acc = f(acc, x, len(xs) - 1 - i)
        return acc
    
    def foldi_double_index(f, acc, xs):
        for i, x in enumerate(xs):
            acc = f(acc, x, 2 * i)
        return acc
    
    return [
        FunctionVariant(
            variant_id='foldi_canonical',
            description='fold with index',
            fn=foldi_standard,
            **base
        ),
        FunctionVariant(
            variant_id='foldi_reversed_index',
            description='index counts down',
            fn=foldi_reversed_index,
            **base
        ),
        FunctionVariant(
            variant_id='foldi_double_index',
            description='index is doubled',
            fn=foldi_double_index,
            **base
        ),
    ]


def _make_count_variants() -> List[FunctionVariant]:
    """Variations of count."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'count',
        'arg_names': ('p', 'xs'),
        'arg_types': (CallableOrig[[T1], bool], list),
        'ret_type': int
    }
    
    def count_standard(p, xs):
        return sum(1 for x in xs if p(x))
    
    def count_not(p, xs):
        return sum(1 for x in xs if not p(x))
    
    def count_plus_one(p, xs):
        return sum(1 for x in xs if p(x)) + 1
    
    return [
        FunctionVariant(
            variant_id='count_canonical',
            description='count elements where p(x) is True',
            fn=count_standard,
            **base
        ),
        FunctionVariant(
            variant_id='count_not',
            description='count elements where p(x) is False',
            fn=count_not,
            **base
        ),
        FunctionVariant(
            variant_id='count_plus_one',
            description='count + 1',
            fn=count_plus_one,
            **base
        ),
    ]


def _make_find_variants() -> List[FunctionVariant]:
    """Variations of find."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'find',
        'arg_names': ('p', 'xs'),
        'arg_types': (CallableOrig[[T1], bool], list),
        'ret_type': list
    }
    
    def find_standard(p, xs):
        return [i for i, x in enumerate(xs) if p(x)]
    
    def find_not(p, xs):
        return [i for i, x in enumerate(xs) if not p(x)]
    
    def find_first_only(p, xs):
        for i, x in enumerate(xs):
            if p(x):
                return [i]
        return []
    
    def find_reversed(p, xs):
        return [len(xs) - 1 - i for i, x in enumerate(xs) if p(x)][::-1]
    
    return [
        FunctionVariant(
            variant_id='find_canonical',
            description='indices where p(x) is True',
            fn=find_standard,
            **base
        ),
        FunctionVariant(
            variant_id='find_not',
            description='indices where p(x) is False',
            fn=find_not,
            **base
        ),
        FunctionVariant(
            variant_id='find_first',
            description='first matching index only',
            fn=find_first_only,
            **base
        ),
        FunctionVariant(
            variant_id='find_reversed',
            description='indices from end',
            fn=find_reversed,
            **base
        ),
    ]


def _make_unique_variants() -> List[FunctionVariant]:
    """Variations of unique."""
    base = {
        'name': 'unique',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': list
    }
    
    def unique_standard(xs):
        return list(set(xs))
    
    def unique_ordered(xs):
        """Preserve order."""
        seen = set()
        result = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result
    
    def unique_reversed(xs):
        """Unique in reverse order."""
        seen = set()
        result = []
        for x in reversed(xs):
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result[::-1]
    
    return [
        FunctionVariant(
            variant_id='unique_canonical',
            description='unique elements (set)',
            fn=unique_standard,
            **base
        ),
        FunctionVariant(
            variant_id='unique_ordered',
            description='unique, preserving first occurrence order',
            fn=unique_ordered,
            **base
        ),
        FunctionVariant(
            variant_id='unique_reversed',
            description='unique, preserving last occurrence order',
            fn=unique_reversed,
            **base
        ),
    ]


def _make_sort_variants() -> List[FunctionVariant]:
    """Variations of sort."""
    from .type_utils import CallableOrig
    
    base = {
        'name': 'sort',
        'arg_names': ('f', 'xs'),
        'arg_types': (CallableOrig[[T1], int], list),
        'ret_type': list
    }
    
    def sort_standard(f, xs):
        return sorted(xs, key=lambda x: f(x))
    
    def sort_reversed(f, xs):
        return sorted(xs, key=lambda x: f(x), reverse=True)
    
    def sort_by_negative(f, xs):
        return sorted(xs, key=lambda x: -f(x))
    
    return [
        FunctionVariant(
            variant_id='sort_canonical',
            description='sort by key function (ascending)',
            fn=sort_standard,
            **base
        ),
        FunctionVariant(
            variant_id='sort_reversed',
            description='sort by key function (descending)',
            fn=sort_reversed,
            **base
        ),
        FunctionVariant(
            variant_id='sort_by_negative',
            description='sort by negated key',
            fn=sort_by_negative,
            **base
        ),
    ]


def _make_group_variants() -> List[FunctionVariant]:
    """Variations of group."""
    from .type_utils import CallableOrig
    from collections import defaultdict
    
    base = {
        'name': 'group',
        'arg_names': ('f', 'xs'),
        'arg_types': (CallableOrig[[T1], T2], list),
        'ret_type': list
    }
    
    def group_standard(f, xs):
        groups = defaultdict(list)
        for x in xs:
            key = f(x)
            key = tuple(key) if isinstance(key, list) else key
            groups[key].append(x)
        return list(groups.values())
    
    def group_reversed(f, xs):
        groups = defaultdict(list)
        for x in xs:
            key = f(x)
            key = tuple(key) if isinstance(key, list) else key
            groups[key].append(x)
        return [list(reversed(g)) for g in groups.values()]
    
    def group_sorted(f, xs):
        groups = defaultdict(list)
        for x in xs:
            key = f(x)
            key = tuple(key) if isinstance(key, list) else key
            groups[key].append(x)
        return [sorted(g) for g in groups.values()]
    
    return [
        FunctionVariant(
            variant_id='group_canonical',
            description='group by key function',
            fn=group_standard,
            **base
        ),
        FunctionVariant(
            variant_id='group_reversed',
            description='group with each group reversed',
            fn=group_reversed,
            **base
        ),
        FunctionVariant(
            variant_id='group_sorted',
            description='group with each group sorted',
            fn=group_sorted,
            **base
        ),
    ]


# =============================================================================
# VARIATION REGISTRY
# =============================================================================

class VariationRegistry:
    """Registry of all function variants."""
    
    _instance = None
    _variants: Dict[str, List[FunctionVariant]] = {}
    
    @classmethod
    def get_instance(cls) -> 'VariationRegistry':
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_all()
        return cls._instance
    
    def _register_all(self):
        """Register all variants."""
        variant_makers = [
            # Arithmetic
            _make_add_variants,
            _make_sub_variants,
            _make_mul_variants,
            _make_div_variants,
            _make_mod_variants,
            # Comparison
            _make_less_than_variants,
            _make_greater_than_variants,
            _make_equals_variants,
            # Boolean
            _make_and_variants,
            _make_or_variants,
            _make_not_variants,
            # Number predicates
            _make_is_even_variants,
            _make_is_odd_variants,
            # List construction
            _make_singleton_variants,
            _make_repeat_variants,
            _make_range_variants,
            _make_cons_variants,
            # List combination
            _make_append_variants,
            _make_concat_variants,
            _make_zip_variants,
            # List access
            _make_first_variants,
            _make_second_variants,
            _make_third_variants,
            _make_last_variants,
            _make_nth_variants,
            # List modification
            _make_insert_variants,
            _make_replace_variants,
            _make_swap_variants,
            # List removal
            _make_cut_idx_variants,
            _make_cut_val_variants,
            _make_cut_vals_variants,
            _make_drop_variants,
            _make_droplast_variants,
            # List slicing
            _make_take_variants,
            _make_takelast_variants,
            _make_slice_variants,
            _make_cut_slice_variants,
            _make_splice_variants,
            # List queries
            _make_is_in_variants,
            _make_length_variants,
            _make_max_variants,
            _make_min_variants,
            _make_product_variants,
            _make_sum_variants,
            # List transformation
            _make_reverse_variants,
            _make_flatten_variants,
            # Higher-order functions
            _make_map_variants,
            _make_mapi_variants,
            _make_filter_variants,
            _make_filteri_variants,
            _make_fold_variants,
            _make_foldi_variants,
            _make_count_variants,
            _make_find_variants,
            _make_unique_variants,
            _make_sort_variants,
            _make_group_variants,
        ]
        
        for maker in variant_makers:
            variants = maker()
            if variants:
                name = variants[0].name
                self._variants[name] = variants
    
    def get_variants(self, name: str) -> List[FunctionVariant]:
        """Get all variants for a function."""
        return self._variants.get(name, [])
    
    def get_canonical_variant(self, name: str) -> Optional[FunctionVariant]:
        """Get the canonical (original) variant for a function."""
        variants = self.get_variants(name)
        for v in variants:
            if v.variant_id.endswith('_canonical'):
                return v
        return variants[0] if variants else None
    
    def get_random_variant(self, name: str, rng: random.Random) -> Optional[FunctionVariant]:
        """Get a random variant for a function."""
        variants = self.get_variants(name)
        return rng.choice(variants) if variants else None
    
    def get_all_function_names(self) -> List[str]:
        """Get all function names with registered variants."""
        return list(self._variants.keys())
    
    def sample_variant_set(
        self,
        rng: random.Random,
        canonical_prob: float = 0.0
    ) -> Dict[str, FunctionVariant]:
        """
        Sample a complete set of variants (one per function).
        
        Args:
            rng: Random generator
            canonical_prob: Probability of selecting canonical variant
                           (0.0 = always random, 1.0 = always canonical)
        
        Returns:
            Dict mapping function names to selected variants
        """
        result = {}
        for name in self._variants:
            if rng.random() < canonical_prob:
                variant = self.get_canonical_variant(name)
            else:
                variant = self.get_random_variant(name, rng)
            if variant:
                result[name] = variant
        return result


# =============================================================================
# SEMANTIC GRAMMAR - A Grammar with Variant Semantics
# =============================================================================

class SemanticGrammar:
    """
    A grammar where each function's semantics comes from a sampled variant.
    
    This creates a new "language" where functions have the same names but
    different (related) behaviors. Used for semantic meta-learning.
    """
    
    def __init__(
        self,
        base_grammar,  # The DefaultGrammar or similar
        variants: Dict[str, FunctionVariant],
        seed: Optional[int] = None
    ):
        """
        Create a semantic grammar from base grammar and variant selection.
        
        Args:
            base_grammar: The base grammar providing function names and types
            variants: Dict mapping function names to selected variants
            seed: Random seed for any internal randomness
        """
        self.base_grammar = base_grammar
        self.variants = variants
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()
        
        # Build the function lookup
        self._functions = {}
        self._variant_info = {}
        
        for name in base_grammar.names:
            if name in variants:
                variant = variants[name]
                self._functions[name] = {
                    'fn': variant.fn,
                    '__call__': self._make_evaluable(name, variant.fn, variant.arg_types),
                    'arg_names': variant.arg_names,
                    'arg_types': variant.arg_types,
                    'ret_type': variant.ret_type
                }
                self._variant_info[name] = {
                    'variant_id': variant.variant_id,
                    'description': variant.description
                }
            else:
                # Fall back to base grammar
                self._functions[name] = base_grammar.functions[name]
                self._variant_info[name] = {
                    'variant_id': 'canonical',
                    'description': 'original function'
                }
    
    def _make_evaluable(self, name: str, fn: Callable, arg_types: tuple) -> Callable:
        """Make a function evaluable by the evaluator (same as Grammar)."""
        from .type_utils import CallableOrig, get_origin
        
        callable_indices = [
            e for e, t in enumerate(arg_types)
            if get_origin(t) == CallableOrig
        ]
        
        def _eval_fn(evaluator, *args):
            def _make_normal_function(_f_closure):
                return lambda *_f_args: evaluator._apply(
                    _f_closure, list(_f_args), _f_closure.env
                )
            
            args = list(args)
            for idx in callable_indices:
                args[idx] = _make_normal_function(args[idx])
            
            return fn(*args)
        
        _eval_fn.__name__ = name
        return _eval_fn
    
    def __getitem__(self, name: str):
        """Get function info by name."""
        return self._functions.get(name, self.base_grammar[name])
    
    def __iter__(self):
        """Iterate over function names and their evaluable functions."""
        for name in self.names:
            yield name, self[name]['__call__']
    
    @property
    def names(self):
        """All function names."""
        return self.base_grammar.names
    
    @property
    def functions(self):
        """All functions dict."""
        return self._functions
    
    def __len__(self):
        return len(self.names)
    
    def get_variant_info(self) -> Dict[str, Dict[str, str]]:
        """Get variant information for all functions."""
        return self._variant_info.copy()
    
    def to_variant_mapping(self) -> Dict[str, str]:
        """
        Get a mapping from function name to variant ID.
        
        Useful for storing in episode metadata.
        """
        return {name: info['variant_id'] for name, info in self._variant_info.items()}
    
    @classmethod
    def sample(
        cls,
        base_grammar,
        rng: random.Random,
        canonical_prob: float = 0.0
    ) -> 'SemanticGrammar':
        """
        Sample a new semantic grammar with random variants.
        
        Args:
            base_grammar: The base grammar
            rng: Random generator
            canonical_prob: Probability of canonical variant (0-1)
        
        Returns:
            A new SemanticGrammar with sampled variants
        """
        registry = VariationRegistry.get_instance()
        variants = registry.sample_variant_set(rng, canonical_prob)
        return cls(base_grammar, variants, seed=rng.randint(0, 2**31))
    
    @classmethod
    def canonical(cls, base_grammar) -> 'SemanticGrammar':
        """Create a semantic grammar with all canonical variants."""
        registry = VariationRegistry.get_instance()
        variants = {}
        for name in registry.get_all_function_names():
            variant = registry.get_canonical_variant(name)
            if variant:
                variants[name] = variant
        return cls(base_grammar, variants)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_variant_registry() -> VariationRegistry:
    """Get the singleton variant registry."""
    return VariationRegistry.get_instance()


def sample_semantic_grammar(
    base_grammar,
    seed: int,
    canonical_prob: float = 0.0
) -> SemanticGrammar:
    """
    Convenience function to sample a semantic grammar.
    
    Args:
        base_grammar: The base grammar (e.g., DefaultGrammar)
        seed: Random seed
        canonical_prob: Probability of canonical variant
    
    Returns:
        A SemanticGrammar with sampled variants
    """
    rng = random.Random(seed)
    return SemanticGrammar.sample(base_grammar, rng, canonical_prob)


if __name__ == "__main__":
    # Test the variation system
    from .grammar import DefaultGrammar
    
    print("Testing Variation System")
    print("=" * 60)
    
    registry = get_variant_registry()
    print(f"\nRegistered functions: {len(registry.get_all_function_names())}")
    
    # Show some variants
    for name in ['+', 'map', 'filter', 'first']:
        variants = registry.get_variants(name)
        print(f"\n{name} variants ({len(variants)}):")
        for v in variants[:3]:  # Show first 3
            print(f"  - {v.variant_id}: {v.description}")
    
    # Test sampling
    print("\n" + "=" * 60)
    print("Sampling semantic grammar...")
    
    sg = sample_semantic_grammar(DefaultGrammar, seed=42, canonical_prob=0.0)
    
    print("\nSampled variants:")
    for name, info in list(sg.get_variant_info().items())[:10]:
        print(f"  {name}: {info['variant_id']}")
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("Testing variant evaluation...")
    
    add_fn = sg['+']["fn"]
    print(f"+ variant: {sg._variant_info['+']['description']}")
    print(f"  2 + 3 = {add_fn(2, 3)}")
    
    map_fn = sg['map']["fn"]
    print(f"map variant: {sg._variant_info['map']['description']}")
    print(f"  map(λx.x+1, [1,2,3,4]) = {map_fn(lambda x: x+1, [1,2,3,4])}")
