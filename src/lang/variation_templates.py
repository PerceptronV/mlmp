"""
Template-Based Semantic Variations for Meta-Learning

This module provides a more flexible variation system using:
1. Parameterized templates - Sample constants, offsets, etc.
2. Compositional templates - Generate variant bodies using the grammar
3. DSL program storage - Save actual program strings in episode data

Example usage:
    registry = TemplateVariationRegistry()
    variant = registry.sample_variant('+', rng)
    # variant.program = '(λ (x y) (+ (+ x y) 3))'
    # variant.description = 'x + y + 3'
"""

from __future__ import annotations
import random
from typing import TypeVar, Callable, List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

T1, T2 = TypeVar('T1'), TypeVar('T2')


# =============================================================================
# PARAMETER SPECIFICATIONS
# =============================================================================

@dataclass
class ParamSpec(ABC):
    """Base class for parameter specifications."""

    @abstractmethod
    def sample(self, rng: random.Random) -> Any:
        """Sample a value for this parameter."""
        pass

    @abstractmethod
    def format(self, value: Any) -> str:
        """Format the value for use in a DSL program."""
        pass


@dataclass
class IntParam(ParamSpec):
    """Integer parameter with range (non-negative only for DSL compatibility)."""
    min_val: int = 0
    max_val: int = 10
    exclude: Tuple[int, ...] = ()  # Values to exclude (e.g., exclude 0 for canonical)

    def sample(self, rng: random.Random) -> int:
        # Only non-negative values for DSL compatibility
        choices = [i for i in range(max(0, self.min_val), self.max_val + 1) if i not in self.exclude]
        return rng.choice(choices) if choices else 1

    def format(self, value: int) -> str:
        # Keep values mod 100 for safety
        return str(value % 100)


@dataclass
class ChoiceParam(ParamSpec):
    """Choose from a list of options."""
    choices: List[Any] = field(default_factory=list)
    weights: Optional[List[float]] = None

    def sample(self, rng: random.Random) -> Any:
        if self.weights:
            return rng.choices(self.choices, weights=self.weights, k=1)[0]
        return rng.choice(self.choices)

    def format(self, value: Any) -> str:
        return str(value)


@dataclass
class BoolParam(ParamSpec):
    """Boolean parameter."""
    true_prob: float = 0.5

    def sample(self, rng: random.Random) -> bool:
        return rng.random() < self.true_prob

    def format(self, value: bool) -> str:
        return 'true' if value else 'false'


@dataclass
class FunctionParam(ParamSpec):
    """Choose a function from the grammar matching a signature."""
    candidates: List[str] = field(default_factory=list)

    def sample(self, rng: random.Random) -> str:
        return rng.choice(self.candidates)

    def format(self, value: str) -> str:
        return value


# =============================================================================
# VARIATION TEMPLATES
# =============================================================================

@dataclass
class VariationTemplate:
    """
    A template for generating function variants.

    Templates can have parameters that are sampled at generation time,
    producing different variants each time.

    Example:
        template = VariationTemplate(
            name='+',
            template_id='add_offset',
            description_template='x + y + {k}',
            program_template='(λ (x y) (+ (+ x y) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=3)}
        )
    """
    name: str  # Canonical function name
    template_id: str  # Unique template identifier
    description_template: str  # Human description with {param} placeholders
    program_template: str  # DSL program with {param} placeholders
    parameters: Dict[str, ParamSpec] = field(default_factory=dict)
    # Type info (inherited from canonical)
    arg_names: Tuple[str, ...] = ()
    arg_types: Tuple[Any, ...] = ()
    ret_type: Any = None

    def sample(self, rng: random.Random) -> 'GeneratedVariant':
        """Sample parameter values and generate a concrete variant."""
        param_values = {
            name: spec.sample(rng)
            for name, spec in self.parameters.items()
        }

        # Format parameters for DSL
        formatted = {
            name: spec.format(param_values[name])
            for name, spec in self.parameters.items()
        }

        # Generate description and program
        description = self.description_template.format(**formatted)
        program = self.program_template.format(**formatted)

        # Generate variant ID
        variant_id = f"{self.template_id}_" + "_".join(
            f"{k}{v}" for k, v in sorted(param_values.items())
        )

        return GeneratedVariant(
            name=self.name,
            template_id=self.template_id,
            variant_id=variant_id,
            description=description,
            program=program,
            param_values=param_values,
            arg_names=self.arg_names,
            arg_types=self.arg_types,
            ret_type=self.ret_type,
        )


@dataclass
class GeneratedVariant:
    """A concrete variant generated from a template."""
    name: str  # Canonical function name
    template_id: str  # Which template generated this
    variant_id: str  # Unique identifier for this specific variant
    description: str  # Human-readable description
    program: str  # DSL program string (e.g., "(λ (x y) (+ (+ x y) 3))")
    param_values: Dict[str, Any]  # The sampled parameter values
    arg_names: Tuple[str, ...]
    arg_types: Tuple[Any, ...]
    ret_type: Any
    _compiled_fn: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for episode data."""
        return {
            'name': self.name,
            'template_id': self.template_id,
            'variant_id': self.variant_id,
            'description': self.description,
            'program': self.program,
            'param_values': self.param_values,
        }

    def compile(self, grammar) -> Callable:
        """Compile the DSL program to a callable function using JIT compilation."""
        if self._compiled_fn is not None:
            return self._compiled_fn

        from .compiler import jit_compile

        # JIT compile the program to native Python bytecode
        # This is faster than using the Evaluator, which interprets the AST at runtime
        self._compiled_fn = jit_compile(self.program, grammar)
        return self._compiled_fn


# =============================================================================
# ARITHMETIC TEMPLATES
# =============================================================================

def _make_add_templates() -> List[VariationTemplate]:
    """Templates for + (addition) variations."""
    base = {
        'name': '+',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }

    return [
        # Canonical
        VariationTemplate(
            template_id='add_canonical',
            description_template='x + y',
            program_template='(λ (x y) (+ x y))',
            parameters={},
            **base
        ),
        # Offset: x + y + k
        VariationTemplate(
            template_id='add_offset',
            description_template='x + y + {k}',
            program_template='(λ (x y) (+ (+ x y) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=5)},
            **base
        ),
        # Scale first: k*x + y
        VariationTemplate(
            template_id='add_scale_first',
            description_template='{k}*x + y',
            program_template='(λ (x y) (+ (* {k} x) y))',
            parameters={'k': IntParam(min_val=0, max_val=3, exclude=(1,))},
            **base
        ),
        # Scale second: x + k*y
        VariationTemplate(
            template_id='add_scale_second',
            description_template='x + {k}*y',
            program_template='(λ (x y) (+ x (* {k} y)))',
            parameters={'k': IntParam(min_val=0, max_val=3, exclude=(1,))},
            **base
        ),
        # Replace with binary op
        VariationTemplate(
            template_id='add_replace_op',
            description_template='{op}(x, y)',
            program_template='(λ (x y) ({op} x y))',
            parameters={'op': ChoiceParam(choices=['-', '*'])},
            **base
        ),
        # Modular addition
        VariationTemplate(
            template_id='add_mod',
            description_template='(x + y) mod {m}',
            program_template='(λ (x y) (% (+ x y) {m}))',
            parameters={'m': IntParam(min_val=2, max_val=10)},
            **base
        ),
    ]


def _make_sub_templates() -> List[VariationTemplate]:
    """Templates for - (subtraction) variations."""
    base = {
        'name': '-',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='sub_canonical',
            description_template='x - y',
            program_template='(λ (x y) (- x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='sub_offset',
            description_template='x - y + {k}',
            program_template='(λ (x y) (+ (- x y) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=5)},
            **base
        ),
        VariationTemplate(
            template_id='sub_reversed',
            description_template='y - x',
            program_template='(λ (x y) (- y x))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='sub_scaled',
            description_template='{k}*x - y',
            program_template='(λ (x y) (- (* {k} x) y))',
            parameters={'k': IntParam(min_val=0, max_val=3, exclude=(1,))},
            **base
        ),
        VariationTemplate(
            template_id='sub_replace_op',
            description_template='{op}(x, y)',
            program_template='(λ (x y) ({op} x y))',
            parameters={'op': ChoiceParam(choices=['+', '*'])},
            **base
        ),
    ]


def _make_mul_templates() -> List[VariationTemplate]:
    """Templates for * (multiplication) variations."""
    base = {
        'name': '*',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='mul_canonical',
            description_template='x * y',
            program_template='(λ (x y) (* x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='mul_offset',
            description_template='x * y + {k}',
            program_template='(λ (x y) (+ (* x y) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=5)},
            **base
        ),
        VariationTemplate(
            template_id='mul_scale',
            description_template='{k} * x * y',
            program_template='(λ (x y) (* {k} (* x y)))',
            parameters={'k': IntParam(min_val=2, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='mul_sum_squares',
            description_template='x*x + y*y',
            program_template='(λ (x y) (+ (* x x) (* y y)))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='mul_diff_squares',
            description_template='x*x - y*y',
            program_template='(λ (x y) (- (* x x) (* y y)))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='mul_replace_op',
            description_template='{op}(x, y)',
            program_template='(λ (x y) ({op} x y))',
            parameters={'op': ChoiceParam(choices=['+', '-'])},
            **base
        ),
    ]


def _make_div_templates() -> List[VariationTemplate]:
    """Templates for / (division) variations."""
    base = {
        'name': '/',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='div_canonical',
            description_template='x / y',
            program_template='(λ (x y) (/ x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='div_offset',
            description_template='x / y + {k}',
            program_template='(λ (x y) (+ (/ x y) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='div_reversed',
            description_template='y / x',
            program_template='(λ (x y) (/ y x))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='div_with_remainder',
            description_template='x / y + x % y',
            program_template='(λ (x y) (+ (/ x y) (% x y)))',
            parameters={},
            **base
        ),
    ]


def _make_mod_templates() -> List[VariationTemplate]:
    """Templates for % (modulo) variations."""
    base = {
        'name': '%',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='mod_canonical',
            description_template='x % y',
            program_template='(λ (x y) (% x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='mod_offset',
            description_template='(x % y) + {k}',
            program_template='(λ (x y) (+ (% x y) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='mod_reversed',
            description_template='y % x',
            program_template='(λ (x y) (% y x))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# COMPARISON TEMPLATES
# =============================================================================

def _make_less_than_templates() -> List[VariationTemplate]:
    """Templates for < variations."""
    base = {
        'name': '<',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': bool
    }

    return [
        VariationTemplate(
            template_id='lt_canonical',
            description_template='x < y',
            program_template='(λ (x y) (< x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='lt_offset',
            description_template='x + {k} < y',
            program_template='(λ (x y) (< (+ x {k}) y))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='lt_replace_op',
            description_template='x {op} y',
            program_template='(λ (x y) ({op} x y))',
            parameters={'op': ChoiceParam(choices=['>', '=='])},
            **base
        ),
    ]


def _make_greater_than_templates() -> List[VariationTemplate]:
    """Templates for > variations."""
    base = {
        'name': '>',
        'arg_names': ('x', 'y'),
        'arg_types': (int, int),
        'ret_type': bool
    }

    return [
        VariationTemplate(
            template_id='gt_canonical',
            description_template='x > y',
            program_template='(λ (x y) (> x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='gt_offset',
            description_template='x + {k} > y',
            program_template='(λ (x y) (> (+ x {k}) y))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='gt_replace_op',
            description_template='x {op} y',
            program_template='(λ (x y) ({op} x y))',
            parameters={'op': ChoiceParam(choices=['<', '=='])},
            **base
        ),
    ]


def _make_equals_templates() -> List[VariationTemplate]:
    """Templates for == variations."""
    base = {
        'name': '==',
        'arg_names': ('x', 'y'),
        'arg_types': (T1, T1),
        'ret_type': bool
    }

    return [
        VariationTemplate(
            template_id='eq_canonical',
            description_template='x == y',
            program_template='(λ (x y) (== x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='eq_negated',
            description_template='not (x == y)',
            program_template='(λ (x y) (not (== x y)))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# BOOLEAN TEMPLATES
# =============================================================================

def _make_and_templates() -> List[VariationTemplate]:
    """Templates for 'and' variations."""
    base = {
        'name': 'and',
        'arg_names': ('x', 'y'),
        'arg_types': (bool, bool),
        'ret_type': bool
    }

    return [
        VariationTemplate(
            template_id='and_canonical',
            description_template='x and y',
            program_template='(λ (x y) (and x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='and_replace_op',
            description_template='x {op} y',
            program_template='(λ (x y) ({op} x y))',
            parameters={'op': ChoiceParam(choices=['or'])},
            **base
        ),
        VariationTemplate(
            template_id='and_nand',
            description_template='not (x and y)',
            program_template='(λ (x y) (not (and x y)))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='and_implies',
            description_template='(not x) or y',
            program_template='(λ (x y) (or (not x) y))',
            parameters={},
            **base
        ),
    ]


def _make_or_templates() -> List[VariationTemplate]:
    """Templates for 'or' variations."""
    base = {
        'name': 'or',
        'arg_names': ('x', 'y'),
        'arg_types': (bool, bool),
        'ret_type': bool
    }

    return [
        VariationTemplate(
            template_id='or_canonical',
            description_template='x or y',
            program_template='(λ (x y) (or x y))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='or_replace_op',
            description_template='x {op} y',
            program_template='(λ (x y) ({op} x y))',
            parameters={'op': ChoiceParam(choices=['and'])},
            **base
        ),
        VariationTemplate(
            template_id='or_nor',
            description_template='not (x or y)',
            program_template='(λ (x y) (not (or x y)))',
            parameters={},
            **base
        ),
    ]


def _make_not_templates() -> List[VariationTemplate]:
    """Templates for 'not' variations."""
    base = {
        'name': 'not',
        'arg_names': ('x',),
        'arg_types': (bool,),
        'ret_type': bool
    }

    return [
        VariationTemplate(
            template_id='not_canonical',
            description_template='not x',
            program_template='(λ (x) (not x))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='not_identity',
            description_template='x (identity)',
            program_template='(λ (x) x)',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='not_double',
            description_template='not (not x)',
            program_template='(λ (x) (not (not x)))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# PREDICATE TEMPLATES
# =============================================================================

def _make_is_even_templates() -> List[VariationTemplate]:
    """Templates for is_even variations."""
    base = {
        'name': 'is_even',
        'arg_names': ('n',),
        'arg_types': (int,),
        'ret_type': bool
    }

    return [
        VariationTemplate(
            template_id='is_even_canonical',
            description_template='n % 2 == 0',
            program_template='(λ (n) (== (% n 2) 0))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='is_even_divisible_by',
            description_template='n % {k} == 0',
            program_template='(λ (n) (== (% n {k}) 0))',
            parameters={'k': IntParam(min_val=2, max_val=5)},
            **base
        ),
        VariationTemplate(
            template_id='is_even_negated',
            description_template='n % 2 != 0 (is_odd)',
            program_template='(λ (n) (not (== (% n 2) 0)))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='is_even_comparison',
            description_template='n {op} 0',
            program_template='(λ (n) ({op} n 0))',
            parameters={'op': ChoiceParam(choices=['>', '<', '=='])},
            **base
        ),
    ]


def _make_is_odd_templates() -> List[VariationTemplate]:
    """Templates for is_odd variations."""
    base = {
        'name': 'is_odd',
        'arg_names': ('n',),
        'arg_types': (int,),
        'ret_type': bool
    }

    return [
        VariationTemplate(
            template_id='is_odd_canonical',
            description_template='n % 2 == 1',
            program_template='(λ (n) (== (% n 2) 1))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='is_odd_not_divisible_by',
            description_template='n % {k} != 0',
            program_template='(λ (n) (not (== (% n {k}) 0)))',
            parameters={'k': IntParam(min_val=2, max_val=5)},
            **base
        ),
        VariationTemplate(
            template_id='is_odd_negated',
            description_template='n % 2 == 0 (is_even)',
            program_template='(λ (n) (== (% n 2) 0))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# LIST CONSTRUCTION TEMPLATES
# =============================================================================

def _make_singleton_templates() -> List[VariationTemplate]:
    """Templates for singleton variations."""
    base = {
        'name': 'singleton',
        'arg_names': ('x',),
        'arg_types': (T1,),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='singleton_canonical',
            description_template='[x]',
            program_template='(λ (x) (singleton x))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='singleton_repeat',
            description_template='repeat x {n} times',
            program_template='(λ (x) (repeat x {n}))',
            parameters={'n': IntParam(min_val=1, max_val=4)},
            **base
        ),
    ]


def _make_repeat_templates() -> List[VariationTemplate]:
    """Templates for repeat variations."""
    base = {
        'name': 'repeat',
        'arg_names': ('x', 'n'),
        'arg_types': (T1, int),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='repeat_canonical',
            description_template='repeat x n times',
            program_template='(λ (x n) (repeat x n))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='repeat_offset',
            description_template='repeat x (n + {k}) times',
            program_template='(λ (x n) (repeat x (+ n {k})))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='repeat_scaled',
            description_template='repeat x ({k} * n) times',
            program_template='(λ (x n) (repeat x (* {k} n)))',
            parameters={'k': IntParam(min_val=0, max_val=3, exclude=(1,))},
            **base
        ),
    ]


def _make_cons_templates() -> List[VariationTemplate]:
    """Templates for cons (prepend) variations."""
    base = {
        'name': 'cons',
        'arg_names': ('x', 'xs'),
        'arg_types': (T1, list),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='cons_canonical',
            description_template='prepend x to xs',
            program_template='(λ (x xs) (cons x xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='cons_append',
            description_template='append x to xs',
            program_template='(λ (x xs) (append xs x))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='cons_double',
            description_template='prepend x twice',
            program_template='(λ (x xs) (cons x (cons x xs)))',
            parameters={},
            **base
        ),
    ]


def _make_append_templates() -> List[VariationTemplate]:
    """Templates for append variations."""
    base = {
        'name': 'append',
        'arg_names': ('xs', 'x'),
        'arg_types': (list, T1),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='append_canonical',
            description_template='append x to xs',
            program_template='(λ (xs x) (append xs x))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='append_prepend',
            description_template='prepend x to xs',
            program_template='(λ (xs x) (cons x xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='append_double',
            description_template='append x twice',
            program_template='(λ (xs x) (append (append xs x) x))',
            parameters={},
            **base
        ),
    ]


def _make_concat_templates() -> List[VariationTemplate]:
    """Templates for concat variations."""
    base = {
        'name': 'concat',
        'arg_names': ('xs', 'ys'),
        'arg_types': (list, list),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='concat_canonical',
            description_template='xs ++ ys',
            program_template='(λ (xs ys) (concat xs ys))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='concat_reversed',
            description_template='ys ++ xs',
            program_template='(λ (xs ys) (concat ys xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='concat_first_only',
            description_template='xs (ignore ys)',
            program_template='(λ (xs ys) xs)',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='concat_second_only',
            description_template='ys (ignore xs)',
            program_template='(λ (xs ys) ys)',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# LIST ACCESS TEMPLATES
# =============================================================================

def _make_first_templates() -> List[VariationTemplate]:
    """Templates for first variations."""
    base = {
        'name': 'first',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': T1
    }

    return [
        VariationTemplate(
            template_id='first_canonical',
            description_template='first element',
            program_template='(λ (xs) (first xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='first_last',
            description_template='last element',
            program_template='(λ (xs) (last xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='first_nth',
            description_template='element at index {n}',
            program_template='(λ (xs) (nth {n} xs))',
            parameters={'n': IntParam(min_val=0, max_val=3)},
            **base
        ),
    ]


def _make_last_templates() -> List[VariationTemplate]:
    """Templates for last variations."""
    base = {
        'name': 'last',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': T1
    }

    return [
        VariationTemplate(
            template_id='last_canonical',
            description_template='last element',
            program_template='(λ (xs) (last xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='last_first',
            description_template='first element',
            program_template='(λ (xs) (first xs))',
            parameters={},
            **base
        ),
    ]


def _make_nth_templates() -> List[VariationTemplate]:
    """Templates for nth variations."""
    base = {
        'name': 'nth',
        'arg_names': ('i', 'xs'),
        'arg_types': (int, list),
        'ret_type': T1
    }

    return [
        VariationTemplate(
            template_id='nth_canonical',
            description_template='element at index i',
            program_template='(λ (i xs) (nth i xs))',
            parameters={},
            **base
        ),
        # Note: Removed nth_offset as it shifts indices and causes out-of-bounds
        # errors for programs that work with canonical semantics
        VariationTemplate(
            template_id='nth_from_end',
            description_template='element at index (length - 1 - i)',
            program_template='(λ (i xs) (nth (- (- (length xs) 1) i) xs))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# LIST SLICING TEMPLATES
# =============================================================================

def _make_take_templates() -> List[VariationTemplate]:
    """Templates for take variations."""
    base = {
        'name': 'take',
        'arg_names': ('n', 'xs'),
        'arg_types': (int, list),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='take_canonical',
            description_template='take first n elements',
            program_template='(λ (n xs) (take n xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='take_offset',
            description_template='take first n + {k} elements',
            program_template='(λ (n xs) (take (+ n {k}) xs))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='take_last',
            description_template='take last n elements',
            program_template='(λ (n xs) (takelast n xs))',
            parameters={},
            **base
        ),
    ]


def _make_drop_templates() -> List[VariationTemplate]:
    """Templates for drop variations."""
    base = {
        'name': 'drop',
        'arg_names': ('n', 'xs'),
        'arg_types': (int, list),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='drop_canonical',
            description_template='drop first n elements',
            program_template='(λ (n xs) (drop n xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='drop_offset',
            description_template='drop first n + {k} elements',
            program_template='(λ (n xs) (drop (+ n {k}) xs))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='drop_last',
            description_template='drop last n elements',
            program_template='(λ (n xs) (droplast n xs))',
            parameters={},
            **base
        ),
    ]


def _make_slice_templates() -> List[VariationTemplate]:
    """Templates for slice variations."""
    base = {
        'name': 'slice',
        'arg_names': ('i', 'j', 'xs'),
        'arg_types': (int, int, list),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='slice_canonical',
            description_template='slice from i to j',
            program_template='(λ (i j xs) (slice i j xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='slice_offset_start',
            description_template='slice from i + {k} to j',
            program_template='(λ (i j xs) (slice (+ i {k}) j xs))',
            parameters={'k': IntParam(min_val=1, max_val=2)},
            **base
        ),
        VariationTemplate(
            template_id='slice_offset_end',
            description_template='slice from i to j + {k}',
            program_template='(λ (i j xs) (slice i (+ j {k}) xs))',
            parameters={'k': IntParam(min_val=1, max_val=2)},
            **base
        ),
        VariationTemplate(
            template_id='slice_reversed',
            description_template='reverse(slice from i to j)',
            program_template='(λ (i j xs) (reverse (slice i j xs)))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# LIST QUERY TEMPLATES
# =============================================================================

def _make_length_templates() -> List[VariationTemplate]:
    """Templates for length variations."""
    base = {
        'name': 'length',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='length_canonical',
            description_template='length of xs',
            program_template='(λ (xs) (length xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='length_offset',
            description_template='length + {k}',
            program_template='(λ (xs) (+ (length xs) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='length_scaled',
            description_template='{k} * length',
            program_template='(λ (xs) (* {k} (length xs)))',
            parameters={'k': IntParam(min_val=0, max_val=3, exclude=(1,))},
            **base
        ),
    ]


def _make_sum_templates() -> List[VariationTemplate]:
    """Templates for sum variations."""
    base = {
        'name': 'sum',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='sum_canonical',
            description_template='sum of xs',
            program_template='(λ (xs) (sum xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='sum_offset',
            description_template='sum + {k}',
            program_template='(λ (xs) (+ (sum xs) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=5)},
            **base
        ),
        VariationTemplate(
            template_id='sum_scaled',
            description_template='{k} * sum',
            program_template='(λ (xs) (* {k} (sum xs)))',
            parameters={'k': IntParam(min_val=2, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='sum_plus_length',
            description_template='sum + length',
            program_template='(λ (xs) (+ (sum xs) (length xs)))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='sum_product',
            description_template='product instead of sum',
            program_template='(λ (xs) (product xs))',
            parameters={},
            **base
        ),
    ]


def _make_product_templates() -> List[VariationTemplate]:
    """Templates for product variations."""
    base = {
        'name': 'product',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='product_canonical',
            description_template='product of xs',
            program_template='(λ (xs) (product xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='product_offset',
            description_template='product + {k}',
            program_template='(λ (xs) (+ (product xs) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=5)},
            **base
        ),
        VariationTemplate(
            template_id='product_sum',
            description_template='sum instead of product',
            program_template='(λ (xs) (sum xs))',
            parameters={},
            **base
        ),
    ]


def _make_max_templates() -> List[VariationTemplate]:
    """Templates for max variations."""
    base = {
        'name': 'max',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='max_canonical',
            description_template='max of xs',
            program_template='(λ (xs) (max xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='max_offset',
            description_template='max + {k}',
            program_template='(λ (xs) (+ (max xs) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='max_min',
            description_template='min instead of max',
            program_template='(λ (xs) (min xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='max_sum',
            description_template='sum instead of max',
            program_template='(λ (xs) (sum xs))',
            parameters={},
            **base
        ),
    ]


def _make_min_templates() -> List[VariationTemplate]:
    """Templates for min variations."""
    base = {
        'name': 'min',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': int
    }

    return [
        VariationTemplate(
            template_id='min_canonical',
            description_template='min of xs',
            program_template='(λ (xs) (min xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='min_offset',
            description_template='min + {k}',
            program_template='(λ (xs) (+ (min xs) {k}))',
            parameters={'k': IntParam(min_val=1, max_val=3)},
            **base
        ),
        VariationTemplate(
            template_id='min_max',
            description_template='max instead of min',
            program_template='(λ (xs) (max xs))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# LIST TRANSFORMATION TEMPLATES
# =============================================================================

def _make_reverse_templates() -> List[VariationTemplate]:
    """Templates for reverse variations."""
    base = {
        'name': 'reverse',
        'arg_names': ('xs',),
        'arg_types': (list,),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='reverse_canonical',
            description_template='reverse xs',
            program_template='(λ (xs) (reverse xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='reverse_identity',
            description_template='xs (no reverse)',
            program_template='(λ (xs) xs)',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='reverse_double',
            description_template='reverse twice',
            program_template='(λ (xs) (reverse (reverse xs)))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# HIGHER-ORDER FUNCTION TEMPLATES
# =============================================================================

def _make_map_templates() -> List[VariationTemplate]:
    """Templates for map variations."""
    from .type_utils import CallableOrig

    base = {
        'name': 'map',
        'arg_names': ('f', 'xs'),
        'arg_types': (CallableOrig[[T1], T2], list),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='map_canonical',
            description_template='map f over xs',
            program_template='(λ (f xs) (map f xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='map_reversed',
            description_template='map f over reversed xs',
            program_template='(λ (f xs) (map f (reverse xs)))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='map_then_reverse',
            description_template='reverse(map f xs)',
            program_template='(λ (f xs) (reverse (map f xs)))',
            parameters={},
            **base
        ),
    ]


def _make_filter_templates() -> List[VariationTemplate]:
    """Templates for filter variations."""
    from .type_utils import CallableOrig

    base = {
        'name': 'filter',
        'arg_names': ('p', 'xs'),
        'arg_types': (CallableOrig[[T1], bool], list),
        'ret_type': list
    }

    return [
        VariationTemplate(
            template_id='filter_canonical',
            description_template='filter xs by p',
            program_template='(λ (p xs) (filter p xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='filter_reversed',
            description_template='filter reversed xs by p',
            program_template='(λ (p xs) (filter p (reverse xs)))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='filter_then_reverse',
            description_template='reverse(filter xs by p)',
            program_template='(λ (p xs) (reverse (filter p xs)))',
            parameters={},
            **base
        ),
    ]


def _make_fold_templates() -> List[VariationTemplate]:
    """Templates for fold variations."""
    from .type_utils import CallableOrig

    base = {
        'name': 'fold',
        'arg_names': ('f', 'acc', 'xs'),
        'arg_types': (CallableOrig[[T2, T1], T2], T2, list),
        'ret_type': T2
    }

    return [
        VariationTemplate(
            template_id='fold_canonical',
            description_template='fold f acc xs',
            program_template='(λ (f acc xs) (fold f acc xs))',
            parameters={},
            **base
        ),
        VariationTemplate(
            template_id='fold_reversed',
            description_template='fold f acc (reverse xs)',
            program_template='(λ (f acc xs) (fold f acc (reverse xs)))',
            parameters={},
            **base
        ),
    ]


# =============================================================================
# TEMPLATE VARIATION REGISTRY
# =============================================================================

class TemplateVariationRegistry:
    """
    Registry of variation templates.

    Provides a more flexible variation system where variants are generated
    on-the-fly by sampling template parameters.
    """

    _instance = None
    _templates: Dict[str, List[VariationTemplate]] = {}

    @classmethod
    def get_instance(cls) -> 'TemplateVariationRegistry':
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_all()
        return cls._instance

    def _register_all(self):
        """Register all templates."""
        template_makers = [
            # Arithmetic
            _make_add_templates,
            _make_sub_templates,
            _make_mul_templates,
            _make_div_templates,
            _make_mod_templates,
            # Comparison
            _make_less_than_templates,
            _make_greater_than_templates,
            _make_equals_templates,
            # Boolean
            _make_and_templates,
            _make_or_templates,
            _make_not_templates,
            # Predicates
            _make_is_even_templates,
            _make_is_odd_templates,
            # List construction
            _make_singleton_templates,
            _make_repeat_templates,
            _make_cons_templates,
            _make_append_templates,
            _make_concat_templates,
            # List access
            _make_first_templates,
            _make_last_templates,
            _make_nth_templates,
            # List slicing
            _make_take_templates,
            _make_drop_templates,
            _make_slice_templates,
            # List queries
            _make_length_templates,
            _make_sum_templates,
            _make_product_templates,
            _make_max_templates,
            _make_min_templates,
            # List transformation
            _make_reverse_templates,
            # Higher-order
            _make_map_templates,
            _make_filter_templates,
            _make_fold_templates,
        ]

        for maker in template_makers:
            templates = maker()
            if templates:
                name = templates[0].name
                self._templates[name] = templates

    def get_templates(self, name: str) -> List[VariationTemplate]:
        """Get all templates for a function."""
        return self._templates.get(name, [])

    def get_canonical_template(self, name: str) -> Optional[VariationTemplate]:
        """Get the canonical template for a function."""
        templates = self.get_templates(name)
        for t in templates:
            if t.template_id.endswith('_canonical'):
                return t
        return templates[0] if templates else None

    def sample_variant(
        self,
        name: str,
        rng: random.Random,
        exclude_canonical: bool = False
    ) -> Optional[GeneratedVariant]:
        """
        Sample a variant for a function.

        Args:
            name: Function name
            rng: Random generator
            exclude_canonical: If True, don't select canonical template

        Returns:
            A generated variant, or None if no templates exist
        """
        templates = self.get_templates(name)
        if not templates:
            return None

        if exclude_canonical:
            templates = [t for t in templates if not t.template_id.endswith('_canonical')]
            if not templates:
                return None

        template = rng.choice(templates)
        return template.sample(rng)

    def sample_canonical_variant(self, name: str, rng: random.Random) -> Optional[GeneratedVariant]:
        """Sample the canonical variant for a function."""
        template = self.get_canonical_template(name)
        if template is None:
            return None
        return template.sample(rng)

    def get_all_function_names(self) -> List[str]:
        """Get all function names with templates."""
        return list(self._templates.keys())

    def sample_variant_set(
        self,
        rng: random.Random,
        canonical_prob: float = 0.0
    ) -> Dict[str, GeneratedVariant]:
        """
        Sample a complete set of variants (one per function).

        Args:
            rng: Random generator
            canonical_prob: Probability of selecting canonical variant

        Returns:
            Dict mapping function names to generated variants
        """
        result = {}
        for name in self._templates:
            if rng.random() < canonical_prob:
                variant = self.sample_canonical_variant(name, rng)
            else:
                variant = self.sample_variant(name, rng)
            if variant:
                result[name] = variant
        return result


# =============================================================================
# TEMPLATE-BASED SEMANTIC GRAMMAR
# =============================================================================

class TemplateSemanticGrammar:
    """
    A grammar with template-generated semantic variations.

    Each function's semantics comes from a sampled variant that is
    represented as an actual DSL program.
    """

    def __init__(
        self,
        base_grammar,
        variants: Dict[str, GeneratedVariant],
        seed: Optional[int] = None
    ):
        self.base_grammar = base_grammar
        self.variants = variants
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()

        self._functions = {}
        self._variant_info = {}

        for name in base_grammar.names:
            if name in variants:
                variant = variants[name]
                # Compile the variant program
                compiled_fn = variant.compile(base_grammar)

                self._functions[name] = {
                    'fn': compiled_fn,
                    '__call__': self._make_evaluable(name, compiled_fn, variant.arg_types),
                    'arg_names': variant.arg_names,
                    'arg_types': variant.arg_types,
                    'ret_type': variant.ret_type
                }
                self._variant_info[name] = variant.to_dict()
            else:
                self._functions[name] = base_grammar.functions[name]
                self._variant_info[name] = {
                    'name': name,
                    'template_id': 'canonical',
                    'variant_id': f'{name}_canonical',
                    'description': 'original function',
                    'program': None,
                    'param_values': {},
                }

    def _make_evaluable(self, name: str, fn: Callable, arg_types: tuple) -> Callable:
        """Make a function evaluable by the evaluator."""
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
        return self._functions.get(name, self.base_grammar[name])

    def __iter__(self):
        for name in self.names:
            yield name, self[name]['__call__']

    @property
    def names(self):
        return self.base_grammar.names

    @property
    def functions(self):
        return self._functions

    def __len__(self):
        return len(self.names)

    def get_variant_info(self) -> Dict[str, Dict[str, Any]]:
        """Get full variant information for all functions."""
        return self._variant_info.copy()

    def to_variant_mapping(self) -> Dict[str, str]:
        """Get a mapping from function name to variant ID."""
        return {name: info['variant_id'] for name, info in self._variant_info.items()}

    def to_program_mapping(self) -> Dict[str, Optional[str]]:
        """Get a mapping from function name to DSL program string."""
        return {name: info.get('program') for name, info in self._variant_info.items()}

    @classmethod
    def sample(
        cls,
        base_grammar,
        rng: random.Random,
        canonical_prob: float = 0.0
    ) -> 'TemplateSemanticGrammar':
        """Sample a new semantic grammar with template-generated variants."""
        registry = TemplateVariationRegistry.get_instance()
        variants = registry.sample_variant_set(rng, canonical_prob)
        return cls(base_grammar, variants, seed=rng.randint(0, 2**31))

    @classmethod
    def canonical(cls, base_grammar) -> 'TemplateSemanticGrammar':
        """Create a semantic grammar with all canonical variants."""
        registry = TemplateVariationRegistry.get_instance()
        variants = {}
        rng = random.Random(0)  # Deterministic for canonical
        for name in registry.get_all_function_names():
            variant = registry.sample_canonical_variant(name, rng)
            if variant:
                variants[name] = variant
        return cls(base_grammar, variants)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_template_registry() -> TemplateVariationRegistry:
    """Get the singleton template registry."""
    return TemplateVariationRegistry.get_instance()


def sample_template_grammar(
    base_grammar,
    seed: int,
    canonical_prob: float = 0.0
) -> TemplateSemanticGrammar:
    """
    Convenience function to sample a template-based semantic grammar.

    Args:
        base_grammar: The base grammar (e.g., DefaultGrammar)
        seed: Random seed
        canonical_prob: Probability of canonical variant

    Returns:
        A TemplateSemanticGrammar with sampled variants
    """
    rng = random.Random(seed)
    return TemplateSemanticGrammar.sample(base_grammar, rng, canonical_prob)


if __name__ == "__main__":
    # Demo
    from .grammar import DefaultGrammar

    print("Template-Based Variation Demo")
    print("=" * 60)

    registry = TemplateVariationRegistry.get_instance()
    rng = random.Random(42)

    # Sample some variants
    print("\nSampled variants:")
    for name in ['+', '-', '*', 'length', 'sum', 'map']:
        variant = registry.sample_variant(name, rng)
        if variant:
            print(f"\n{name}:")
            print(f"  variant_id: {variant.variant_id}")
            print(f"  description: {variant.description}")
            print(f"  program: {variant.program}")
            print(f"  params: {variant.param_values}")

    # Create a full grammar
    print("\n" + "=" * 60)
    print("Creating TemplateSemanticGrammar...")

    grammar = TemplateSemanticGrammar.sample(DefaultGrammar, rng, canonical_prob=0.3)

    print("\nVariant programs for key functions:")
    programs = grammar.to_program_mapping()
    for name in ['+', '-', 'length', 'sum']:
        print(f"  {name}: {programs.get(name)}")
