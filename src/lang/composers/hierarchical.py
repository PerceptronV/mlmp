"""
Hierarchical Empirical Program Composer

This module generates programs using hierarchically-structured empirical
distributions learned from example programs. Unlike the flat empirical composer,
this learns at multiple levels of abstraction:

1. Skeleton level: Overall program structure (e.g., "filter applied to input",
   "map composed with reverse", etc.)
2. Pattern level: Sub-structures within skeletons (predicate patterns,
   transform patterns, key function patterns)
3. Leaf level: Specific values (constants, variable choices)

All distributions are learned purely from example programs - no predefined
templates. This combines the benefits of hierarchical generation (coherent
high-level structure) with purely empirical learning (no manual tuning).

Key insight: By abstracting programs into skeletons first, we can learn
meaningful high-level patterns even when specific (type, context) pairs
are sparse. The skeleton captures "filter with a comparison predicate"
rather than the specific details.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib

from .base import Composer
from ..grammar import Grammar
from ..ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, IfNode, ListNode
)
from ..parser import parse as parse_program
from ..type_checker import TypeChecker
from ..type_utils import (
    get_args,
    get_base_type,
    CallableOrig,
    TypeType,
    SubstitutionTable,
    substitute_type_vars,
    matchable,
    isvariable
)


# ============================================================================
# Skeleton Representation
# ============================================================================

@dataclass(frozen=True)
class Skeleton:
    """
    A skeleton represents the high-level structure of a program with holes.

    Skeletons abstract away:
    - Specific variable names (replaced with $0, $1, etc. for positional refs)
    - Specific constant values (replaced with #int, #bool)
    - Lambda parameter names (normalized)

    Skeletons preserve:
    - Function names (map, filter, +, etc.)
    - Overall AST structure
    - Nesting depth and composition patterns
    """
    # String representation of the skeleton
    pattern: str

    # Depth of the skeleton (how many levels of abstraction)
    depth: int = 1

    def __repr__(self) -> str:
        return f"Skeleton({self.pattern})"

    def __hash__(self) -> int:
        return hash(self.pattern)


@dataclass(frozen=True)
class PredicatePattern:
    """Pattern for predicate expressions (used in filter, count, find)."""
    pattern: str  # e.g., "is_even", "compare_const", "modulo_check", "compound"

    def __repr__(self) -> str:
        return f"Pred({self.pattern})"


@dataclass(frozen=True)
class TransformPattern:
    """Pattern for transform expressions (used in map)."""
    pattern: str  # e.g., "identity", "arithmetic", "modulo", "conditional"

    def __repr__(self) -> str:
        return f"Trans({self.pattern})"


@dataclass(frozen=True)
class KeyPattern:
    """Pattern for key function expressions (used in sort, group)."""
    pattern: str  # e.g., "identity", "negate", "modulo", "arithmetic"

    def __repr__(self) -> str:
        return f"Key({self.pattern})"


# ============================================================================
# Skeleton Extraction
# ============================================================================

class SkeletonExtractor:
    """
    Extracts skeletons from AST nodes at various levels of abstraction.

    Abstraction levels:
    - Level 0: Just the outermost operation (e.g., "filter", "map")
    - Level 1: One level of structure (e.g., "(filter PRED $0)")
    - Level 2: Two levels (e.g., "(filter (λ (is_even $)) $0)")
    - Level 3+: Full structure with holes for constants only
    """

    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.ho_functions = {'map', 'mapi', 'filter', 'filteri', 'fold', 'foldi',
                            'sort', 'group', 'count', 'find'}

    def extract(self, node: ASTNode, depth: int = 1, var_map: Optional[dict] = None) -> Skeleton:
        """
        Extract a skeleton from an AST node.

        Args:
            node: The AST node to extract from
            depth: How many levels of structure to preserve (0 = most abstract)
            var_map: Mapping from variable names to positional indices

        Returns:
            A Skeleton representing the program's structure
        """
        if var_map is None:
            var_map = {}

        pattern = self._extract_pattern(node, depth, var_map, 0)
        return Skeleton(pattern=pattern, depth=depth)

    def _extract_pattern(
        self,
        node: ASTNode,
        max_depth: int,
        var_map: dict[str, int],
        current_depth: int
    ) -> str:
        """Recursively extract pattern string."""

        # At max depth, abstract everything to a hole
        if current_depth >= max_depth:
            if isinstance(node, NumberNode):
                return "#int"
            elif isinstance(node, BooleanNode):
                return "#bool"
            elif isinstance(node, ListNode):
                return "#list"
            elif isinstance(node, VariableNode):
                if node.name in var_map:
                    return f"${var_map[node.name]}"
                elif node.name in self.grammar.names:
                    return node.name
                else:
                    return "$?"
            elif isinstance(node, LambdaNode):
                return "LAMBDA"
            elif isinstance(node, ApplicationNode):
                if isinstance(node.function, VariableNode):
                    return f"({node.function.name} ...)"
                return "(APP ...)"
            elif isinstance(node, IfNode):
                return "(IF ...)"
            return "?"

        # Extract with structure
        if isinstance(node, NumberNode):
            return "#int"

        elif isinstance(node, BooleanNode):
            return "#bool"

        elif isinstance(node, ListNode):
            if not node.elements:
                return "[]"
            elem_patterns = [
                self._extract_pattern(e, max_depth, var_map, current_depth + 1)
                for e in node.elements
            ]
            return f"[{' '.join(elem_patterns)}]"

        elif isinstance(node, VariableNode):
            if node.name in var_map:
                return f"${var_map[node.name]}"
            elif node.name in self.grammar.names:
                return node.name
            else:
                # Unknown variable - might be from outer scope
                return f"${node.name}"

        elif isinstance(node, LambdaNode):
            # Normalize lambda parameters
            new_var_map = var_map.copy()
            params = node.param if isinstance(node.param, list) else [node.param]
            for i, p in enumerate(params):
                new_var_map[p] = len(var_map) + i

            body_pattern = self._extract_pattern(
                node.body, max_depth, new_var_map, current_depth + 1
            )

            if len(params) == 1:
                return f"(λ {body_pattern})"
            else:
                return f"(λ{len(params)} {body_pattern})"

        elif isinstance(node, ApplicationNode):
            func_pattern = self._extract_pattern(
                node.function, max_depth, var_map, current_depth
            )
            arg_patterns = [
                self._extract_pattern(arg, max_depth, var_map, current_depth + 1)
                for arg in node.arguments
            ]
            return f"({func_pattern} {' '.join(arg_patterns)})"

        elif isinstance(node, IfNode):
            cond = self._extract_pattern(node.condition, max_depth, var_map, current_depth + 1)
            then = self._extract_pattern(node.then_expr, max_depth, var_map, current_depth + 1)
            else_ = self._extract_pattern(node.else_expr, max_depth, var_map, current_depth + 1)
            return f"(if {cond} {then} {else_})"

        return "?"

    def extract_predicate_pattern(self, node: ASTNode) -> PredicatePattern:
        """Extract a pattern from a predicate expression."""
        if isinstance(node, ApplicationNode):
            if isinstance(node.function, VariableNode):
                fn = node.function.name

                if fn in {'is_even', 'is_odd'}:
                    return PredicatePattern('is_even_odd')

                if fn in {'<', '>', '==', '!=', '<=', '>='}:
                    # Check for modulo comparison
                    if len(node.arguments) >= 1:
                        first_arg = node.arguments[0]
                        if isinstance(first_arg, ApplicationNode):
                            if isinstance(first_arg.function, VariableNode):
                                if first_arg.function.name == '%':
                                    return PredicatePattern('modulo_check')
                    return PredicatePattern('compare_const')

                if fn in {'and', 'or'}:
                    return PredicatePattern('compound')

                if fn == 'not':
                    return PredicatePattern('negation')

                if fn == 'is_in':
                    return PredicatePattern('membership')

        if isinstance(node, VariableNode):
            return PredicatePattern('variable')

        if isinstance(node, BooleanNode):
            return PredicatePattern('literal')

        return PredicatePattern('other')

    def extract_transform_pattern(self, node: ASTNode, param_name: str) -> TransformPattern:
        """Extract a pattern from a transform expression."""
        if isinstance(node, VariableNode):
            if node.name == param_name:
                return TransformPattern('identity')
            return TransformPattern('other_var')

        if isinstance(node, NumberNode):
            return TransformPattern('constant')

        if isinstance(node, ApplicationNode):
            if isinstance(node.function, VariableNode):
                fn = node.function.name

                if fn in {'+', '-', '*', '/'}:
                    return TransformPattern('arithmetic')

                if fn == '%':
                    return TransformPattern('modulo')

                if fn == 'singleton':
                    return TransformPattern('singleton')

        if isinstance(node, IfNode):
            return TransformPattern('conditional')

        return TransformPattern('other')

    def extract_key_pattern(self, node: ASTNode, param_name: str) -> KeyPattern:
        """Extract a pattern from a key function expression."""
        if isinstance(node, VariableNode):
            if node.name == param_name:
                return KeyPattern('identity')
            return KeyPattern('other_var')

        if isinstance(node, ApplicationNode):
            if isinstance(node.function, VariableNode):
                fn = node.function.name

                if fn == '-':
                    # Check for negation pattern: (- 0 x)
                    if len(node.arguments) >= 2:
                        if isinstance(node.arguments[0], NumberNode):
                            if node.arguments[0].value == 0:
                                return KeyPattern('negate')
                    return KeyPattern('arithmetic')

                if fn in {'+', '*', '/'}:
                    return KeyPattern('arithmetic')

                if fn == '%':
                    return KeyPattern('modulo')

        return KeyPattern('other')


# ============================================================================
# Hierarchical Distributions
# ============================================================================

def type_to_key(type_: TypeType) -> str:
    """Convert a type to a canonical string key."""
    if type_ is None:
        return "None"

    if isvariable(type_):
        return "T"

    base = get_base_type(type_)
    args = get_args(type_)

    if not args:
        if base == int:
            return "int"
        elif base == bool:
            return "bool"
        else:
            return str(base.__name__) if hasattr(base, '__name__') else str(base)

    if base == list:
        if args:
            return f"list[{type_to_key(args[0])}]"
        return "list[T]"

    if base == CallableOrig:
        if len(args) == 2:
            param_list, ret = args
            if isinstance(param_list, list):
                params_str = ", ".join(type_to_key(p) for p in param_list)
                return f"Callable[[{params_str}], {type_to_key(ret)}]"
        return "Callable"

    args_str = ", ".join(type_to_key(a) for a in args)
    base_name = base.__name__ if hasattr(base, '__name__') else str(base)
    return f"{base_name}[{args_str}]"


class HierarchicalDistributions:
    """
    Holds empirical distributions at multiple levels of abstraction.

    Level 1: P(skeleton | function_type)
    Level 2: P(sub_pattern | skeleton_context)
    Level 3: P(constant | type), P(variable_position | context)
    """

    def __init__(self, skeleton_depth: int = 2):
        self.skeleton_depth = skeleton_depth

        # Skeleton distributions: P(skeleton | return_type)
        # Key: return_type_key
        # Value: Counter of skeleton patterns
        self.skeleton_counts: dict[str, Counter[str]] = defaultdict(Counter)

        # Sub-pattern distributions conditioned on parent context
        # Key: (parent_skeleton_prefix, position_in_skeleton)
        # Value: Counter of sub-patterns
        self.predicate_pattern_counts: dict[str, Counter[PredicatePattern]] = defaultdict(Counter)
        self.transform_pattern_counts: dict[str, Counter[TransformPattern]] = defaultdict(Counter)
        self.key_pattern_counts: dict[str, Counter[KeyPattern]] = defaultdict(Counter)

        # Detailed pattern distributions (for filling in sub-patterns)
        # Predicate details
        self.comparison_op_counts: Counter[str] = Counter()
        self.comparison_const_counts: Counter[int] = Counter()
        self.modulo_divisor_counts: Counter[int] = Counter()
        self.modulo_remainder_counts: Counter[int] = Counter()

        # Transform details
        self.arithmetic_op_counts: Counter[str] = Counter()
        self.arithmetic_const_counts: Counter[int] = Counter()

        # Leaf distributions
        self.int_constants: Counter[int] = Counter()
        self.bool_constants: Counter[bool] = Counter()

        # Composition patterns: what operations are composed together
        # Key: outer_operation
        # Value: Counter of inner operations
        self.composition_counts: dict[str, Counter[str]] = defaultdict(Counter)

        # Normalised probabilities (computed after loading)
        self._normalised = False

    def record_skeleton(self, return_type: TypeType, skeleton: Skeleton):
        """Record an observation of a skeleton."""
        type_key = type_to_key(return_type)
        self.skeleton_counts[type_key][skeleton.pattern] += 1
        self._normalised = False

    def record_predicate_pattern(self, context: str, pattern: PredicatePattern):
        """Record a predicate pattern observation."""
        self.predicate_pattern_counts[context][pattern] += 1
        self._normalised = False

    def record_transform_pattern(self, context: str, pattern: TransformPattern):
        """Record a transform pattern observation."""
        self.transform_pattern_counts[context][pattern] += 1
        self._normalised = False

    def record_key_pattern(self, context: str, pattern: KeyPattern):
        """Record a key pattern observation."""
        self.key_pattern_counts[context][pattern] += 1
        self._normalised = False

    def record_comparison(self, op: str, const: int):
        """Record a comparison operation."""
        self.comparison_op_counts[op] += 1
        self.comparison_const_counts[const] += 1
        self._normalised = False

    def record_modulo(self, divisor: int, remainder: int = 0):
        """Record a modulo operation."""
        self.modulo_divisor_counts[divisor] += 1
        self.modulo_remainder_counts[remainder] += 1
        self._normalised = False

    def record_arithmetic(self, op: str, const: int):
        """Record an arithmetic operation."""
        self.arithmetic_op_counts[op] += 1
        self.arithmetic_const_counts[const] += 1
        self._normalised = False

    def record_int_constant(self, value: int):
        """Record an integer constant."""
        self.int_constants[value] += 1
        self._normalised = False

    def record_bool_constant(self, value: bool):
        """Record a boolean constant."""
        self.bool_constants[value] += 1
        self._normalised = False

    def record_composition(self, outer: str, inner: str):
        """Record a composition of operations."""
        self.composition_counts[outer][inner] += 1
        self._normalised = False

    def normalize(self):
        """Compute normalised probability distributions."""
        if self._normalised:
            return

        # Normalize skeleton distributions
        self._skeleton_probs: dict[str, dict[str, float]] = {}
        for type_key, counts in self.skeleton_counts.items():
            total = sum(counts.values())
            if total > 0:
                self._skeleton_probs[type_key] = {s: c / total for s, c in counts.items()}

        # Normalize pattern distributions
        self._predicate_probs = self._normalize_counter_dict(self.predicate_pattern_counts)
        self._transform_probs = self._normalize_counter_dict(self.transform_pattern_counts)
        self._key_probs = self._normalize_counter_dict(self.key_pattern_counts)

        # Normalize detail distributions
        self._comparison_op_probs = self._normalize_counter(self.comparison_op_counts)
        self._comparison_const_probs = self._normalize_counter(self.comparison_const_counts)
        self._modulo_divisor_probs = self._normalize_counter(self.modulo_divisor_counts)
        self._arithmetic_op_probs = self._normalize_counter(self.arithmetic_op_counts)
        self._arithmetic_const_probs = self._normalize_counter(self.arithmetic_const_counts)
        self._int_probs = self._normalize_counter(self.int_constants)
        self._bool_probs = self._normalize_counter(self.bool_constants)

        # Normalize composition distributions
        self._composition_probs: dict[str, dict[str, float]] = {}
        for outer, counts in self.composition_counts.items():
            total = sum(counts.values())
            if total > 0:
                self._composition_probs[outer] = {i: c / total for i, c in counts.items()}

        self._normalised = True

    def _normalize_counter(self, counter: Counter) -> dict:
        """Normalize a counter to probabilities."""
        total = sum(counter.values())
        if total > 0:
            return {k: v / total for k, v in counter.items()}
        return {}

    def _normalize_counter_dict(self, counter_dict: dict) -> dict:
        """Normalize a dict of counters."""
        result = {}
        for key, counter in counter_dict.items():
            total = sum(counter.values())
            if total > 0:
                result[key] = {k: v / total for k, v in counter.items()}
        return result

    def sample_skeleton(self, return_type: TypeType, rng, noise: float = 0.0) -> Optional[str]:
        """Sample a skeleton pattern for the given return type."""
        self.normalize()
        type_key = type_to_key(return_type)

        if type_key not in self._skeleton_probs:
            return None

        probs = self._skeleton_probs[type_key]
        return self._sample_with_noise(probs, rng, noise)

    def sample_predicate_pattern(self, context: str, rng, noise: float = 0.0) -> Optional[PredicatePattern]:
        """Sample a predicate pattern."""
        self.normalize()
        if context not in self._predicate_probs:
            # Fall back to marginal over all contexts
            all_patterns: Counter[PredicatePattern] = Counter()
            for counts in self.predicate_pattern_counts.values():
                all_patterns.update(counts)
            if not all_patterns:
                return None
            probs = self._normalize_counter(all_patterns)
            result = self._sample_with_noise(probs, rng, noise)
            return result if result else None

        probs = self._predicate_probs[context]
        return self._sample_with_noise(probs, rng, noise)

    def sample_transform_pattern(self, context: str, rng, noise: float = 0.0) -> Optional[TransformPattern]:
        """Sample a transform pattern."""
        self.normalize()
        if context not in self._transform_probs:
            all_patterns: Counter[TransformPattern] = Counter()
            for counts in self.transform_pattern_counts.values():
                all_patterns.update(counts)
            if not all_patterns:
                return None
            probs = self._normalize_counter(all_patterns)
            result = self._sample_with_noise(probs, rng, noise)
            return result if result else None

        probs = self._transform_probs[context]
        return self._sample_with_noise(probs, rng, noise)

    def sample_key_pattern(self, context: str, rng, noise: float = 0.0) -> Optional[KeyPattern]:
        """Sample a key function pattern."""
        self.normalize()
        if context not in self._key_probs:
            all_patterns: Counter[KeyPattern] = Counter()
            for counts in self.key_pattern_counts.values():
                all_patterns.update(counts)
            if not all_patterns:
                return None
            probs = self._normalize_counter(all_patterns)
            result = self._sample_with_noise(probs, rng, noise)
            return result if result else None

        probs = self._key_probs[context]
        return self._sample_with_noise(probs, rng, noise)

    def sample_comparison_op(self, rng, noise: float = 0.0) -> str:
        """Sample a comparison operator."""
        self.normalize()
        if self._comparison_op_probs:
            result = self._sample_with_noise(self._comparison_op_probs, rng, noise)
            if result:
                return result
        return rng.choice(['<', '>', '=='])

    def sample_comparison_const(self, rng, noise: float = 0.0) -> int:
        """Sample a comparison constant."""
        self.normalize()
        if self._comparison_const_probs:
            result = self._sample_with_noise(self._comparison_const_probs, rng, noise)
            if result is not None:
                return result
        return rng.randint(0, 99)

    def sample_modulo_divisor(self, rng, noise: float = 0.0) -> int:
        """Sample a modulo divisor."""
        self.normalize()
        if self._modulo_divisor_probs:
            result = self._sample_with_noise(self._modulo_divisor_probs, rng, noise)
            if result is not None:
                return result
        return rng.choice([2, 3, 5, 10])

    def sample_arithmetic_op(self, rng, noise: float = 0.0) -> str:
        """Sample an arithmetic operator."""
        self.normalize()
        if self._arithmetic_op_probs:
            # Filter to safe operators
            safe_ops = {'+', '-', '*'}
            filtered = {k: v for k, v in self._arithmetic_op_probs.items() if k in safe_ops}
            if filtered:
                total = sum(filtered.values())
                filtered = {k: v / total for k, v in filtered.items()}
                result = self._sample_with_noise(filtered, rng, noise)
                if result:
                    return result
        return rng.choice(['+', '-', '*'])

    def sample_arithmetic_const(self, rng, noise: float = 0.0) -> int:
        """Sample an arithmetic constant."""
        self.normalize()
        if self._arithmetic_const_probs:
            result = self._sample_with_noise(self._arithmetic_const_probs, rng, noise)
            if result is not None:
                return result
        return rng.randint(1, 10)

    def sample_int_constant(self, rng, noise: float = 0.0) -> int:
        """Sample an integer constant."""
        self.normalize()
        if self._int_probs:
            result = self._sample_with_noise(self._int_probs, rng, noise)
            if result is not None:
                return result
        return rng.randint(0, 99)

    def sample_bool_constant(self, rng, noise: float = 0.0) -> bool:
        """Sample a boolean constant."""
        self.normalize()
        if self._bool_probs:
            result = self._sample_with_noise(self._bool_probs, rng, noise)
            if result is not None:
                return result
        return rng.choice([True, False])

    def sample_inner_operation(self, outer: str, rng, noise: float = 0.0) -> Optional[str]:
        """Sample an inner operation for composition."""
        self.normalize()
        if outer in self._composition_probs:
            return self._sample_with_noise(self._composition_probs[outer], rng, noise)
        # Fall back to marginal
        all_inner: Counter[str] = Counter()
        for counts in self.composition_counts.values():
            all_inner.update(counts)
        if all_inner:
            probs = self._normalize_counter(all_inner)
            return self._sample_with_noise(probs, rng, noise)
        return None

    def _sample_with_noise(self, probs: dict, rng, noise: float):
        """Sample from a probability distribution with noise."""
        if not probs:
            return None

        keys = list(probs.keys())
        weights = list(probs.values())

        # Apply noise: interpolate toward uniform
        n = len(keys)
        uniform = 1.0 / n
        weights = [(1 - noise) * w + noise * uniform for w in weights]

        # Renormalize
        total = sum(weights)
        weights = [w / total for w in weights]

        return rng.choices(keys, weights=weights, k=1)[0]

    def __repr__(self) -> str:
        n_skeletons = sum(len(c) for c in self.skeleton_counts.values())
        n_types = len(self.skeleton_counts)
        total_obs = sum(sum(c.values()) for c in self.skeleton_counts.values())
        return f"HierarchicalDistributions({n_types} types, {n_skeletons} skeletons, {total_obs} observations)"


# ============================================================================
# AST Analyser for Hierarchical Learning
# ============================================================================

class HierarchicalAnalyser:
    """
    Analyses AST nodes to extract hierarchical patterns.

    Walks the AST to:
    1. Extract overall skeleton
    2. Identify and classify sub-patterns (predicates, transforms, keys)
    3. Record detailed patterns for constants and operators
    """

    def __init__(self, grammar: Grammar, dists: HierarchicalDistributions, skeleton_depth: int = 2):
        self.grammar = grammar
        self.dists = dists
        self.skeleton_depth = skeleton_depth
        self.extractor = SkeletonExtractor(grammar)
        self.type_checker = TypeChecker(grammar)

        # Higher-order function sets for pattern detection
        self.ho_map = {'map', 'mapi'}
        self.ho_filter = {'filter', 'filteri'}
        self.ho_fold = {'fold', 'foldi'}
        self.ho_sort = {'sort', 'group'}
        self.ho_count = {'count', 'find'}

    def analyse(self, node: ASTNode, expected_type: TypeType) -> bool:
        """
        Analyse an AST and record hierarchical observations.

        Returns True if analysis succeeded, False otherwise.
        """
        try:
            # Extract skeleton
            skeleton = self.extractor.extract(node, depth=self.skeleton_depth)
            self.dists.record_skeleton(expected_type, skeleton)

            # Extract sub-patterns and details
            self._analyse_node(node, context="top")

            return True
        except Exception:
            return False

    def _analyse_node(self, node: ASTNode, context: str):
        """Recursively analyse a node for patterns."""

        if isinstance(node, NumberNode):
            self.dists.record_int_constant(node.value)

        elif isinstance(node, BooleanNode):
            self.dists.record_bool_constant(node.value)

        elif isinstance(node, ListNode):
            for elem in node.elements:
                self._analyse_node(elem, context)

        elif isinstance(node, LambdaNode):
            params = node.param if isinstance(node.param, list) else [node.param]
            self._analyse_node(node.body, f"{context}/lambda")

        elif isinstance(node, IfNode):
            self._analyse_node(node.condition, f"{context}/if_cond")
            self._analyse_node(node.then_expr, f"{context}/if_then")
            self._analyse_node(node.else_expr, f"{context}/if_else")

        elif isinstance(node, ApplicationNode):
            func = node.function
            args = node.arguments

            if isinstance(func, VariableNode):
                fn = func.name

                # Detect higher-order function patterns
                if fn in self.ho_filter | self.ho_count:
                    if args and isinstance(args[0], LambdaNode):
                        pred_node = args[0]
                        params = pred_node.param if isinstance(pred_node.param, list) else [pred_node.param]
                        pattern = self.extractor.extract_predicate_pattern(pred_node.body)
                        self.dists.record_predicate_pattern(fn, pattern)
                        self._analyse_predicate_details(pred_node.body, params[0] if params else None)

                elif fn in self.ho_map:
                    if args and isinstance(args[0], LambdaNode):
                        transform_node = args[0]
                        params = transform_node.param if isinstance(transform_node.param, list) else [transform_node.param]
                        param_name = params[0] if params else ""
                        pattern = self.extractor.extract_transform_pattern(transform_node.body, param_name)
                        self.dists.record_transform_pattern(fn, pattern)
                        self._analyse_transform_details(transform_node.body)

                elif fn in self.ho_sort:
                    if args and isinstance(args[0], LambdaNode):
                        key_node = args[0]
                        params = key_node.param if isinstance(key_node.param, list) else [key_node.param]
                        param_name = params[0] if params else ""
                        pattern = self.extractor.extract_key_pattern(key_node.body, param_name)
                        self.dists.record_key_pattern(fn, pattern)
                        self._analyse_key_details(key_node.body)

                # Detect compositions
                for arg in args:
                    if isinstance(arg, ApplicationNode):
                        if isinstance(arg.function, VariableNode):
                            inner_fn = arg.function.name
                            if inner_fn in (self.ho_map | self.ho_filter | self.ho_sort | self.ho_fold):
                                self.dists.record_composition(fn, inner_fn)

            # Recurse into all arguments
            for arg in args:
                self._analyse_node(arg, f"{context}/{func}")

    def _analyse_predicate_details(self, body: ASTNode, param_name: Optional[str]):
        """Extract detailed patterns from a predicate body."""
        if isinstance(body, ApplicationNode):
            if isinstance(body.function, VariableNode):
                fn = body.function.name

                if fn in {'<', '>', '==', '!=', '<=', '>='}:
                    self.dists.record_comparison(fn, 0)  # Record op
                    # Try to find constant
                    for arg in body.arguments:
                        if isinstance(arg, NumberNode):
                            self.dists.record_comparison(fn, arg.value)

                if fn == '%':
                    if len(body.arguments) >= 2:
                        if isinstance(body.arguments[1], NumberNode):
                            self.dists.record_modulo(body.arguments[1].value)

    def _analyse_transform_details(self, body: ASTNode):
        """Extract detailed patterns from a transform body."""
        if isinstance(body, ApplicationNode):
            if isinstance(body.function, VariableNode):
                fn = body.function.name

                if fn in {'+', '-', '*', '/'}:
                    for arg in body.arguments:
                        if isinstance(arg, NumberNode):
                            self.dists.record_arithmetic(fn, arg.value)

                if fn == '%':
                    if len(body.arguments) >= 2:
                        if isinstance(body.arguments[1], NumberNode):
                            self.dists.record_modulo(body.arguments[1].value)

    def _analyse_key_details(self, body: ASTNode):
        """Extract detailed patterns from a key function body."""
        self._analyse_transform_details(body)  # Same patterns apply


# ============================================================================
# Loading Function
# ============================================================================

def load(
    txt_path: Path,
    grammar: Grammar,
    expected_type: TypeType = CallableOrig[[list[int]], list[int]],
    skeleton_depth: int = 2
) -> HierarchicalDistributions:
    """
    Load and analyse programs from a text file.

    Args:
        txt_path: Path to text file with one program per line
        grammar: Grammar to use for analysis
        expected_type: The expected type of all programs
        skeleton_depth: How deep to extract skeletons (1-3 recommended)

    Returns:
        HierarchicalDistributions with learned distributions
    """
    dists = HierarchicalDistributions(skeleton_depth=skeleton_depth)
    analyser = HierarchicalAnalyser(grammar, dists, skeleton_depth=skeleton_depth)

    with open(txt_path, 'r') as f:
        for line in f:
            program_str = line.strip()
            if not program_str:
                continue

            try:
                ast = parse_program(program_str)
                analyser.analyse(ast, expected_type)
            except Exception:
                # Skip unparseable programs
                continue

    dists.normalize()
    return dists


# ============================================================================
# Hierarchical Composer
# ============================================================================

class HierarchicalComposer(Composer):
    """
    Generates programs using hierarchically-structured empirical distributions.

    Generation proceeds top-down:
    1. Sample a skeleton pattern for the target type
    2. Parse the skeleton to identify what sub-patterns are needed
    3. Sample sub-patterns (predicates, transforms, keys) based on context
    4. Fill in leaf values (constants, variable choices) from empirical distributions

    This produces more coherent programs than flat empirical generation because
    the high-level structure is decided first, then details are filled in
    consistently with that structure.
    """

    _cached_distributions: dict[tuple[str, int, int], HierarchicalDistributions] = {}

    @classmethod
    def get_name(cls) -> str:
        return "hierarchical"

    def __init__(
        self,
        seed: int,
        grammar: Grammar,
        functions_path: Optional[Path] = None,
        noise: float = 0.1,
        skeleton_depth: int = 2,
        ho_bias: float = 0.5
    ):
        """
        Initialise the hierarchical composer.

        Args:
            seed: Random seed for deterministic generation
            grammar: Grammar containing available functions
            functions_path: Path to training programs (one per line)
            noise: Noise parameter for exploration (0 = strict, 1 = uniform)
            skeleton_depth: Depth of skeleton extraction (1-3 recommended)
            ho_bias: Bias toward higher-order functions (0 = pure empirical,
                    1 = always use HO functions). When a skeleton maps to a
                    simple strategy, with probability ho_bias it will be
                    "upgraded" to map/filter/sort/composition instead.
        """
        super().__init__(seed, grammar)
        self.noise = noise
        self.skeleton_depth = skeleton_depth
        self.ho_bias = ho_bias

        if functions_path is None:
            functions_path = Path(__file__).parent / 'data' / 'functions.txt'

        cache_key = (str(functions_path.resolve()), id(grammar), skeleton_depth)
        if cache_key not in HierarchicalComposer._cached_distributions:
            if functions_path.exists():
                HierarchicalComposer._cached_distributions[cache_key] = load(
                    functions_path, grammar, skeleton_depth=skeleton_depth
                )
            else:
                fallback = HierarchicalDistributions(skeleton_depth=skeleton_depth)
                fallback.normalize()
                HierarchicalComposer._cached_distributions[cache_key] = fallback

        self.dists = HierarchicalComposer._cached_distributions[cache_key]
        self.extractor = SkeletonExtractor(grammar)

        # Cache for skeleton -> generation strategy mapping
        self._skeleton_strategies: dict[str, str] = {}

        # Cache for available functions (computed lazily)
        self._available_ho_cache: dict[str, bool] | None = None
        self._available_simple_ops_cache: list[str] | None = None
        self._available_comparison_ops_cache: list[str] | None = None
        self._available_arithmetic_ops_cache: list[str] | None = None

        # Simple strategies that can be upgraded to HO functions
        self._simple_strategies = {
            'take_drop', 'list_build', 'list_modify', 'simple_op',
            'reverse', 'unique', 'identity', 'other'
        }
        # HO function strategies to upgrade to
        self._ho_strategies = ['map', 'filter', 'sort', 'composition']
        self._ho_weights = [0.30, 0.30, 0.15, 0.25]

    # ========================================================================
    # Availability Checking Methods
    # ========================================================================

    def _get_available_ho_functions(self) -> dict[str, bool]:
        """Get availability of higher-order functions."""
        if self._available_ho_cache is not None:
            return self._available_ho_cache

        self._available_ho_cache = {
            'map': self._has_function('map'),
            'mapi': self._has_function('mapi'),
            'filter': self._has_function('filter'),
            'filteri': self._has_function('filteri'),
            'sort': self._has_function('sort'),
            'fold': self._has_function('fold'),
            'foldi': self._has_function('foldi'),
        }
        return self._available_ho_cache

    def _get_available_simple_ops(self) -> list[str]:
        """Get list of available simple list operations."""
        if self._available_simple_ops_cache is not None:
            return self._available_simple_ops_cache

        all_ops = ['reverse', 'unique', 'take', 'drop', 'takelast', 'droplast']
        self._available_simple_ops_cache = [op for op in all_ops if self._has_function(op)]
        return self._available_simple_ops_cache

    def _get_available_comparison_ops(self) -> list[str]:
        """Get list of available comparison operators."""
        if self._available_comparison_ops_cache is not None:
            return self._available_comparison_ops_cache

        all_ops = ['<', '>', '==', '!=', '<=', '>=']
        self._available_comparison_ops_cache = [op for op in all_ops if self._has_function(op)]
        return self._available_comparison_ops_cache

    def _get_available_arithmetic_ops(self) -> list[str]:
        """Get list of available arithmetic operators."""
        if self._available_arithmetic_ops_cache is not None:
            return self._available_arithmetic_ops_cache

        all_ops = ['+', '-', '*', '/', '%']
        self._available_arithmetic_ops_cache = [op for op in all_ops if self._has_function(op)]
        return self._available_arithmetic_ops_cache

    def _is_strategy_available(self, strategy: str) -> bool:
        """Check if a strategy can be used based on function availability."""
        ho_funcs = self._get_available_ho_functions()
        simple_ops = self._get_available_simple_ops()

        strategy_requirements = {
            'identity': True,  # Always available
            'filter': ho_funcs.get('filter', False) or ho_funcs.get('filteri', False),
            'map': ho_funcs.get('map', False) or ho_funcs.get('mapi', False),
            'sort': ho_funcs.get('sort', False),
            'reverse': 'reverse' in simple_ops,
            'unique': 'unique' in simple_ops,
            'take_drop': any(op in simple_ops for op in ['take', 'drop', 'takelast', 'droplast']),
            'list_build': any(self._has_function(fn) for fn in ['cons', 'append', 'concat']),
            'list_modify': any(self._has_function(fn) for fn in ['replace', 'swap', 'cut_val']),
            'simple_op': len(simple_ops) > 0,
            'composition': (
                (ho_funcs.get('filter', False) or ho_funcs.get('map', False) or len(simple_ops) > 0)
            ),
            'conditional': (
                self._has_function('length') and
                len(self._get_available_comparison_ops()) > 0
            ),
            'other': True,  # Fallback always available
        }
        return strategy_requirements.get(strategy, False)

    def _get_available_ho_strategies(self) -> tuple[list[str], list[float]]:
        """Get available HO strategies and their weights for upgrade."""
        ho_funcs = self._get_available_ho_functions()
        simple_ops = self._get_available_simple_ops()

        strategies = []
        weights = []

        # map: requires map or mapi
        if ho_funcs.get('map', False) or ho_funcs.get('mapi', False):
            strategies.append('map')
            weights.append(0.30)

        # filter: requires filter or filteri
        if ho_funcs.get('filter', False) or ho_funcs.get('filteri', False):
            strategies.append('filter')
            weights.append(0.30)

        # sort: requires sort
        if ho_funcs.get('sort', False):
            strategies.append('sort')
            weights.append(0.15)

        # composition: requires at least one operation
        if (ho_funcs.get('filter', False) or ho_funcs.get('map', False) or
            'reverse' in simple_ops or 'unique' in simple_ops):
            strategies.append('composition')
            weights.append(0.25)

        return strategies, weights

    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None
    ) -> ASTNode:
        """
        Generate a program using hierarchical empirical distributions.

        Args:
            target_type: The desired output type
            depth: Maximum remaining depth
            context: Variable bindings in scope (name -> type)
            substitutions: Current type variable substitutions

        Returns:
            An AST node of the target type
        """
        if context is None:
            context = {}
        if substitutions is None:
            substitutions = SubstitutionTable()

        target = substitute_type_vars(target_type, substitutions)
        base_type = get_base_type(target)

        if base_type == CallableOrig:
            return self._generate_function(target, depth, context, substitutions)

        if base_type == list:
            return self._generate_list_expression(target, depth, context, substitutions)

        if target == int:
            return self._generate_int_expression(depth, context, substitutions)
        elif target == bool:
            return self._generate_bool_expression(depth, context, substitutions)

        raise ValueError(f"Cannot generate expression of type {target}")

    def _generate_function(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a function using skeleton-guided generation."""
        target = substitute_type_vars(target_type, substitutions)
        type_args = get_args(target)

        if len(type_args) != 2:
            raise ValueError(f"Callable must have 2 type args: {target}")

        param_types_list, ret_type = type_args
        if not isinstance(param_types_list, list):
            raise ValueError(f"Expected param list: {param_types_list}")

        # Create parameters
        new_context = context.copy()
        param_names = []
        for param_type in param_types_list:
            param_name = self._fresh_var_name()
            param_names.append(param_name)
            new_context[param_name] = param_type

        # Try to sample a skeleton and use it for generation
        skeleton_pattern = self.dists.sample_skeleton(target, self.rng, self.noise)

        if skeleton_pattern:
            # Use skeleton-guided generation
            body = self._generate_from_skeleton(
                skeleton_pattern, param_names, param_types_list, ret_type,
                depth - 1, new_context, substitutions
            )
        else:
            # Fall back to template-based generation
            body = self._generate_body_fallback(
                param_names, param_types_list, ret_type,
                depth - 1, new_context, substitutions
            )

        # Wrap in lambdas
        for param_name in reversed(param_names):
            body = LambdaNode(param_name, body)

        return body

    def _generate_from_skeleton(
        self,
        skeleton: str,
        param_names: list[str],
        param_types: list[TypeType],
        ret_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate a program body guided by a skeleton pattern.

        The skeleton tells us the high-level structure; we fill in the details
        using sub-pattern and leaf distributions.
        """
        # Parse the skeleton to determine what to generate
        strategy = self._parse_skeleton_strategy(skeleton)

        # Check if strategy is available, fall back to 'other' if not
        if not self._is_strategy_available(strategy):
            strategy = 'identity' if self._is_strategy_available('identity') else 'other'

        # Apply ho_bias: upgrade simple strategies to HO functions with some probability
        if strategy in self._simple_strategies and self.rng.random() < self.ho_bias:
            available_ho, ho_weights = self._get_available_ho_strategies()
            if available_ho:
                strategy = self.rng.choices(available_ho, weights=ho_weights, k=1)[0]

        input_var = param_names[0] if param_names else None
        input_type = param_types[0] if param_types else None

        if input_type:
            input_args = get_args(input_type)
            elem_type = input_args[0] if input_args else int
        else:
            elem_type = int

        # Generate based on detected strategy
        if strategy == 'identity':
            if input_var:
                return VariableNode(input_var)
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'filter':
            if input_var:
                return self._generate_filter(input_var, elem_type, depth, context, substitutions)
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'map':
            if input_var:
                return self._generate_map(input_var, elem_type, ret_type, depth, context, substitutions)
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'sort':
            if input_var:
                return self._generate_sort(input_var, elem_type, depth, context, substitutions)
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'reverse':
            if input_var:
                return ApplicationNode(VariableNode('reverse'), [VariableNode(input_var)])
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'unique':
            if input_var:
                return ApplicationNode(VariableNode('unique'), [VariableNode(input_var)])
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'composition':
            if input_var and depth >= 2:
                return self._generate_composition(
                    input_var, elem_type, ret_type, depth, context, substitutions
                )
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'take_drop':
            if input_var:
                # Filter to available take/drop operations
                take_drop_ops = [op for op in ['take', 'drop', 'takelast', 'droplast']
                                 if self._has_function(op)]
                if take_drop_ops:
                    op = self.rng.choice(take_drop_ops)
                    n = self.rng.randint(1, 5)
                    return ApplicationNode(VariableNode(op), [NumberNode(n), VariableNode(input_var)])
                # Fall back to identity if no take/drop available
                return VariableNode(input_var)
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'conditional':
            if input_var:
                return self._generate_conditional(
                    input_var, input_type, ret_type, depth, context, substitutions
                )
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'list_build':
            if input_var:
                return self._generate_list_build(input_var, elem_type, depth, context, substitutions)
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'list_modify':
            if input_var:
                return self._generate_list_modify(input_var, elem_type, depth, context, substitutions)
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        elif strategy == 'simple_op':
            if input_var:
                return self._generate_simple_list_op(input_var, input_type, ret_type, substitutions)
            return self._generate_expression_of_type(ret_type, depth, context, substitutions)

        else:
            # Unknown strategy ('other') - fall back to compositional generation
            return self._generate_body_fallback(
                param_names, param_types, ret_type, depth, context, substitutions
            )

    def _parse_skeleton_strategy(self, skeleton: str) -> str:
        """
        Parse a skeleton pattern to determine the generation strategy.

        This maps skeleton patterns to high-level strategies.
        """
        # Cache for efficiency
        if skeleton in self._skeleton_strategies:
            return self._skeleton_strategies[skeleton]

        # Simple pattern matching on skeleton string
        skeleton_lower = skeleton.lower()

        if skeleton == "(λ $0)" or "$0)" == skeleton[-3:] and "(" not in skeleton[3:]:
            strategy = 'identity'
        elif 'filter' in skeleton_lower and 'filteri' not in skeleton_lower:
            strategy = 'filter'
        elif 'filteri' in skeleton_lower:
            strategy = 'filter'
        elif 'map' in skeleton_lower and 'mapi' not in skeleton_lower:
            strategy = 'map'
        elif 'mapi' in skeleton_lower:
            strategy = 'map'
        elif 'sort' in skeleton_lower or 'group' in skeleton_lower:
            strategy = 'sort'
        elif 'reverse' in skeleton_lower:
            if any(x in skeleton_lower for x in ['filter', 'map', 'sort']):
                strategy = 'composition'
            else:
                strategy = 'reverse'
        elif 'unique' in skeleton_lower:
            if any(x in skeleton_lower for x in ['filter', 'map', 'sort']):
                strategy = 'composition'
            else:
                strategy = 'unique'
        elif any(x in skeleton_lower for x in ['take', 'drop', 'takelast', 'droplast']):
            strategy = 'take_drop'
        elif any(x in skeleton_lower for x in ['slice', 'cut_slice', 'cut_idx']):
            strategy = 'take_drop'  # Similar to take/drop
        elif any(x in skeleton_lower for x in ['cons', 'append', 'concat', 'insert', 'splice']):
            strategy = 'list_build'
        elif any(x in skeleton_lower for x in ['replace', 'swap', 'cut_val', 'cut_vals']):
            strategy = 'list_modify'
        elif 'singleton' in skeleton_lower:
            strategy = 'simple_op'
        elif 'flatten' in skeleton_lower:
            strategy = 'composition'  # flatten usually wraps another operation
        elif 'fold' in skeleton_lower:
            strategy = 'composition'  # fold is complex, generate composition instead
        elif 'if' in skeleton_lower:
            strategy = 'conditional'
        elif skeleton.count('(') >= 3:  # Nested structure suggests composition
            strategy = 'composition'
        else:
            strategy = 'other'

        self._skeleton_strategies[skeleton] = strategy
        return strategy

    # ========================================================================
    # Higher-Order Function Generators
    # ========================================================================

    def _generate_filter(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a filter expression using empirical predicate patterns."""
        ho_funcs = self._get_available_ho_functions()

        # Check availability and choose between filter/filteri
        has_filter = ho_funcs.get('filter', False)
        has_filteri = ho_funcs.get('filteri', False)

        if not has_filter and not has_filteri:
            # Fall back to identity
            return VariableNode(input_var)

        # Sample whether to use index (only if filteri available)
        use_index = has_filteri and (not has_filter or self.rng.random() < 0.15)
        fn_name = 'filteri' if use_index else 'filter'

        # Sample predicate pattern
        pred_pattern = self.dists.sample_predicate_pattern(fn_name, self.rng, self.noise)

        elem_var = self._fresh_var_name()
        idx_var = self._fresh_var_name() if use_index else None

        # Generate predicate body based on pattern
        pred_body = self._generate_predicate_from_pattern(
            pred_pattern, elem_var, elem_type, idx_var, context, substitutions
        )

        # Build lambda
        if use_index:
            lambda_node = LambdaNode(idx_var, LambdaNode(elem_var, pred_body))
        else:
            lambda_node = LambdaNode(elem_var, pred_body)

        return ApplicationNode(VariableNode(fn_name), [lambda_node, VariableNode(input_var)])

    def _generate_map(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a map expression using empirical transform patterns."""
        ho_funcs = self._get_available_ho_functions()

        # Check availability and choose between map/mapi
        has_map = ho_funcs.get('map', False)
        has_mapi = ho_funcs.get('mapi', False)

        if not has_map and not has_mapi:
            # Fall back to identity
            return VariableNode(input_var)

        # Sample whether to use index (only if mapi available)
        use_index = has_mapi and (not has_map or self.rng.random() < 0.2)
        fn_name = 'mapi' if use_index else 'map'

        # Sample transform pattern
        trans_pattern = self.dists.sample_transform_pattern(fn_name, self.rng, self.noise)

        output_args = get_args(output_type)
        output_elem_type = output_args[0] if output_args else int

        elem_var = self._fresh_var_name()
        idx_var = self._fresh_var_name() if use_index else None

        # Generate transform body based on pattern
        trans_body = self._generate_transform_from_pattern(
            trans_pattern, elem_var, elem_type, output_elem_type,
            idx_var, context, substitutions
        )

        # Build lambda
        if use_index:
            lambda_node = LambdaNode(elem_var, LambdaNode(idx_var, trans_body))
        else:
            lambda_node = LambdaNode(elem_var, trans_body)

        return ApplicationNode(VariableNode(fn_name), [lambda_node, VariableNode(input_var)])

    def _generate_sort(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a sort expression using empirical key patterns."""
        if not self._has_function('sort'):
            # Fall back to identity
            return VariableNode(input_var)

        # Sample key pattern
        key_pattern = self.dists.sample_key_pattern('sort', self.rng, self.noise)

        elem_var = self._fresh_var_name()

        # Generate key body based on pattern
        key_body = self._generate_key_from_pattern(
            key_pattern, elem_var, elem_type, context, substitutions
        )

        lambda_node = LambdaNode(elem_var, key_body)

        return ApplicationNode(VariableNode('sort'), [lambda_node, VariableNode(input_var)])

    def _generate_composition(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a composition of operations using empirical patterns."""
        ho_funcs = self._get_available_ho_functions()
        simple_ops = self._get_available_simple_ops()

        # Build list of available outer operations
        outer_ops = []
        if ho_funcs.get('filter', False) or ho_funcs.get('filteri', False):
            outer_ops.append('filter')
        if ho_funcs.get('map', False) or ho_funcs.get('mapi', False):
            outer_ops.append('map')
        if 'reverse' in simple_ops:
            outer_ops.append('reverse')
        if 'unique' in simple_ops:
            outer_ops.append('unique')

        if not outer_ops:
            # No operations available, return identity
            return VariableNode(input_var)

        outer_op = self.rng.choice(outer_ops)

        # Build list of available inner operations
        inner_ops = []
        if ho_funcs.get('filter', False) or ho_funcs.get('filteri', False):
            inner_ops.append('filter')
        if ho_funcs.get('map', False) or ho_funcs.get('mapi', False):
            inner_ops.append('map')
        if ho_funcs.get('sort', False):
            inner_ops.append('sort')

        # Try to sample from empirical, filter to available
        inner_op = self.dists.sample_inner_operation(outer_op, self.rng, self.noise)
        if inner_op is None or inner_op not in inner_ops:
            if inner_ops:
                inner_op = self.rng.choice(inner_ops)
            else:
                inner_op = None

        # Generate inner operation
        if inner_op == 'filter':
            inner = self._generate_filter(input_var, elem_type, depth - 1, context, substitutions)
        elif inner_op == 'map':
            inner = self._generate_map(input_var, elem_type, output_type, depth - 1, context, substitutions)
        elif inner_op == 'sort':
            inner = self._generate_sort(input_var, elem_type, depth - 1, context, substitutions)
        else:
            inner = VariableNode(input_var)

        # Apply outer operation
        if outer_op == 'filter':
            temp_var = self._fresh_var_name()
            filter_expr = self._generate_filter(temp_var, elem_type, depth - 1, context, substitutions)
            return self._substitute_var_in_expr(filter_expr, temp_var, inner)

        elif outer_op == 'map':
            temp_var = self._fresh_var_name()
            map_expr = self._generate_map(temp_var, elem_type, output_type, depth - 1, context, substitutions)
            return self._substitute_var_in_expr(map_expr, temp_var, inner)

        elif outer_op == 'reverse':
            return ApplicationNode(VariableNode('reverse'), [inner])

        elif outer_op == 'unique':
            return ApplicationNode(VariableNode('unique'), [inner])

        return inner

    def _generate_conditional(
        self,
        input_var: str,
        input_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a conditional expression."""
        input_node = VariableNode(input_var)
        comparison_ops = self._get_available_comparison_ops()

        # Check if we have the necessary functions
        has_length = self._has_function('length')
        has_eq = '==' in comparison_ops
        has_lt = '<' in comparison_ops

        if not has_length or (not has_eq and not has_lt):
            # Can't generate conditional, fall back to simple op or identity
            simple_ops = self._get_available_simple_ops()
            if simple_ops:
                return self._generate_simple_list_op(input_var, input_type, output_type, substitutions)
            return VariableNode(input_var)

        # Generate condition based on list properties
        cond_types = []
        if has_eq:
            cond_types.append('empty')
        if has_lt:
            cond_types.append('length')

        cond_type = self.rng.choice(cond_types) if cond_types else 'empty'

        if cond_type == 'empty' and has_eq:
            condition = ApplicationNode(VariableNode('=='), [
                ApplicationNode(VariableNode('length'), [input_node]),
                NumberNode(0)
            ])
        else:
            threshold = self.rng.randint(1, 5)
            condition = ApplicationNode(VariableNode('<'), [
                ApplicationNode(VariableNode('length'), [input_node]),
                NumberNode(threshold)
            ])

        # Generate branches
        then_expr = self._generate_simple_list_op(input_var, input_type, output_type, substitutions)
        else_expr = self._generate_simple_list_op(input_var, input_type, output_type, substitutions)

        return IfNode(condition, then_expr, else_expr)

    def _generate_list_build(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a list building operation (cons, append, concat)."""
        input_node = VariableNode(input_var)
        const = self.dists.sample_int_constant(self.rng, self.noise)

        # Filter to available operations
        all_ops = {'cons': 0.4, 'append': 0.4, 'concat': 0.2}
        available_ops = {op: w for op, w in all_ops.items() if self._has_function(op)}

        if not available_ops:
            # Fall back to identity
            return VariableNode(input_var)

        # Renormalize weights
        total = sum(available_ops.values())
        ops = list(available_ops.keys())
        weights = [available_ops[op] / total for op in ops]

        op = self.rng.choices(ops, weights=weights, k=1)[0]

        if op == 'cons':
            return ApplicationNode(VariableNode('cons'), [NumberNode(const), input_node])
        elif op == 'append':
            return ApplicationNode(VariableNode('append'), [input_node, NumberNode(const)])
        else:  # concat
            return ApplicationNode(VariableNode('concat'), [input_node, input_node])

    def _generate_list_modify(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a list modification operation (replace, swap, cut_val)."""
        input_node = VariableNode(input_var)
        const = self.dists.sample_int_constant(self.rng, self.noise)
        idx = self.rng.randint(0, 5)

        # Filter to available operations
        all_ops = {'replace': 0.4, 'swap': 0.3, 'cut_val': 0.3}
        available_ops = {op: w for op, w in all_ops.items() if self._has_function(op)}

        if not available_ops:
            # Fall back to identity
            return VariableNode(input_var)

        # Renormalize weights
        total = sum(available_ops.values())
        ops = list(available_ops.keys())
        weights = [available_ops[op] / total for op in ops]

        op = self.rng.choices(ops, weights=weights, k=1)[0]

        if op == 'replace':
            return ApplicationNode(VariableNode('replace'), [NumberNode(idx), NumberNode(const), input_node])
        elif op == 'swap':
            idx2 = self.rng.randint(0, 5)
            return ApplicationNode(VariableNode('swap'), [NumberNode(idx), NumberNode(idx2), input_node])
        else:  # cut_val
            return ApplicationNode(VariableNode('cut_val'), [NumberNode(const), input_node])

    # ========================================================================
    # Pattern-Based Generators
    # ========================================================================

    def _generate_predicate_from_pattern(
        self,
        pattern: Optional[PredicatePattern],
        elem_var: str,
        elem_type: TypeType,
        idx_var: Optional[str],
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a predicate expression from a pattern."""
        elem_node = VariableNode(elem_var)
        comparison_ops = self._get_available_comparison_ops()
        arithmetic_ops = self._get_available_arithmetic_ops()

        if pattern is None:
            pattern = PredicatePattern('compare_const')

        # Check availability for each pattern type and fall back if needed
        if pattern.pattern == 'is_even_odd':
            available_parity = [fn for fn in ['is_even', 'is_odd'] if self._has_function(fn)]
            if available_parity:
                fn = self.rng.choice(available_parity)
                return ApplicationNode(VariableNode(fn), [elem_node])
            # Fall back to comparison
            pattern = PredicatePattern('compare_const')

        if pattern.pattern == 'compare_const':
            if comparison_ops:
                op = self.rng.choice(comparison_ops)
                const = self.dists.sample_comparison_const(self.rng, self.noise)
                return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
            # Fall back to literal true
            return BooleanNode(True)

        elif pattern.pattern == 'modulo_check':
            has_mod = '%' in arithmetic_ops
            has_eq = '==' in comparison_ops
            if has_mod and has_eq:
                divisor = self.dists.sample_modulo_divisor(self.rng, self.noise)
                remainder = self.rng.randint(0, max(0, divisor - 1))
                mod_expr = ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])
                return ApplicationNode(VariableNode('=='), [mod_expr, NumberNode(remainder)])
            # Fall back to comparison
            if comparison_ops:
                op = self.rng.choice(comparison_ops)
                const = self.dists.sample_comparison_const(self.rng, self.noise)
                return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
            return BooleanNode(True)

        elif pattern.pattern == 'compound':
            available_combiners = [c for c in ['and', 'or'] if self._has_function(c)]
            if available_combiners:
                pred1 = self._generate_simple_predicate(elem_var, elem_type)
                pred2 = self._generate_simple_predicate(elem_var, elem_type)
                combiner = self.rng.choice(available_combiners)
                return ApplicationNode(VariableNode(combiner), [pred1, pred2])
            # Fall back to simple predicate
            return self._generate_simple_predicate(elem_var, elem_type)

        elif pattern.pattern == 'negation':
            if self._has_function('not'):
                inner = self._generate_simple_predicate(elem_var, elem_type)
                return ApplicationNode(VariableNode('not'), [inner])
            # Fall back to simple predicate
            return self._generate_simple_predicate(elem_var, elem_type)

        elif pattern.pattern == 'membership':
            has_is_in = self._has_function('is_in')
            has_singleton = self._has_function('singleton')
            if has_is_in and has_singleton:
                list_expr = ApplicationNode(VariableNode('singleton'), [
                    NumberNode(self.dists.sample_int_constant(self.rng, self.noise))
                ])
                return ApplicationNode(VariableNode('is_in'), [elem_node, list_expr])
            # Fall back to comparison
            if comparison_ops:
                op = self.rng.choice(comparison_ops)
                const = self.dists.sample_comparison_const(self.rng, self.noise)
                return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
            return BooleanNode(True)

        else:
            # Default to comparison
            if comparison_ops:
                op = self.rng.choice(comparison_ops)
                const = self.dists.sample_comparison_const(self.rng, self.noise)
                return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
            return BooleanNode(True)

    def _generate_simple_predicate(self, elem_var: str, elem_type: TypeType) -> ASTNode:
        """Generate a simple predicate (for compound predicates)."""
        elem_node = VariableNode(elem_var)
        comparison_ops = self._get_available_comparison_ops()

        # Check availability of parity functions
        available_parity = [fn for fn in ['is_even', 'is_odd'] if self._has_function(fn)]

        if available_parity and self.rng.random() < 0.4:
            fn = self.rng.choice(available_parity)
            return ApplicationNode(VariableNode(fn), [elem_node])
        elif comparison_ops:
            op = self.rng.choice(comparison_ops)
            const = self.dists.sample_comparison_const(self.rng, self.noise)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
        elif available_parity:
            # Only parity available
            fn = self.rng.choice(available_parity)
            return ApplicationNode(VariableNode(fn), [elem_node])
        else:
            # Nothing available, return literal
            return BooleanNode(True)

    def _generate_transform_from_pattern(
        self,
        pattern: Optional[TransformPattern],
        elem_var: str,
        elem_type: TypeType,
        ret_type: TypeType,
        idx_var: Optional[str],
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a transform expression from a pattern."""
        elem_node = VariableNode(elem_var)
        arithmetic_ops = self._get_available_arithmetic_ops()
        # Filter to safe ops for transforms (exclude %)
        safe_arithmetic = [op for op in arithmetic_ops if op in ['+', '-', '*']]

        if pattern is None:
            pattern = TransformPattern('arithmetic')

        if pattern.pattern == 'identity':
            return elem_node

        elif pattern.pattern == 'arithmetic':
            if safe_arithmetic:
                op = self.rng.choice(safe_arithmetic)
                const = self.dists.sample_arithmetic_const(self.rng, self.noise)
                # Sometimes use index if available
                if idx_var and self.rng.random() < 0.3:
                    return ApplicationNode(VariableNode(op), [elem_node, VariableNode(idx_var)])
                return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
            # Fall back to identity
            return elem_node

        elif pattern.pattern == 'modulo':
            if '%' in arithmetic_ops:
                divisor = self.dists.sample_modulo_divisor(self.rng, self.noise)
                return ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])
            # Fall back to identity
            return elem_node

        elif pattern.pattern == 'conditional':
            pred = self._generate_simple_predicate(elem_var, elem_type)
            then_val = NumberNode(self.rng.randint(0, 10))
            else_val = NumberNode(self.rng.randint(0, 10))
            return IfNode(pred, then_val, else_val)

        elif pattern.pattern == 'singleton':
            if self._has_function('singleton'):
                return ApplicationNode(VariableNode('singleton'), [elem_node])
            # Fall back to identity
            return elem_node

        elif pattern.pattern == 'constant':
            return NumberNode(self.dists.sample_int_constant(self.rng, self.noise))

        else:
            # Default to arithmetic or identity
            if safe_arithmetic:
                op = self.rng.choice(safe_arithmetic)
                const = self.dists.sample_arithmetic_const(self.rng, self.noise)
                return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
            return elem_node

    def _generate_key_from_pattern(
        self,
        pattern: Optional[KeyPattern],
        elem_var: str,
        elem_type: TypeType,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a key function expression from a pattern."""
        elem_node = VariableNode(elem_var)
        arithmetic_ops = self._get_available_arithmetic_ops()

        if pattern is None:
            pattern = KeyPattern('identity')

        if pattern.pattern == 'identity':
            return elem_node

        elif pattern.pattern == 'negate':
            if '-' in arithmetic_ops:
                return ApplicationNode(VariableNode('-'), [NumberNode(0), elem_node])
            # Fall back to identity
            return elem_node

        elif pattern.pattern == 'modulo':
            if '%' in arithmetic_ops:
                divisor = self.dists.sample_modulo_divisor(self.rng, self.noise)
                return ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])
            # Fall back to identity
            return elem_node

        elif pattern.pattern == 'arithmetic':
            # Filter to safe ops
            safe_ops = [op for op in arithmetic_ops if op in ['+', '-', '*']]
            if safe_ops:
                op = self.rng.choice(safe_ops)
                const = self.dists.sample_arithmetic_const(self.rng, self.noise)
                return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
            # Fall back to identity
            return elem_node

        else:
            return elem_node

    # ========================================================================
    # Fallback and Helper Generators
    # ========================================================================

    def _generate_body_fallback(
        self,
        param_names: list[str],
        param_types: list[TypeType],
        ret_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Fallback body generation when skeleton sampling fails."""
        actual_ret = substitute_type_vars(ret_type, substitutions)
        ret_base = get_base_type(actual_ret)

        # For list -> list, sample a simple operation
        if len(param_names) == 1 and ret_base == list:
            param_type = substitute_type_vars(param_types[0], substitutions)
            if get_base_type(param_type) == list:
                input_var = param_names[0]
                input_args = get_args(param_type)
                elem_type = input_args[0] if input_args else int

                ho_funcs = self._get_available_ho_functions()
                simple_ops = self._get_available_simple_ops()

                # Build available operations with weights
                available_ops: dict[str, float] = {'identity': 0.1}

                if ho_funcs.get('filter', False) or ho_funcs.get('filteri', False):
                    available_ops['filter'] = 0.3
                if ho_funcs.get('map', False) or ho_funcs.get('mapi', False):
                    available_ops['map'] = 0.3
                if 'reverse' in simple_ops:
                    available_ops['reverse'] = 0.15
                if 'unique' in simple_ops:
                    available_ops['unique'] = 0.15

                # Renormalize weights
                total = sum(available_ops.values())
                ops = list(available_ops.keys())
                weights = [available_ops[o] / total for o in ops]

                op = self.rng.choices(ops, weights=weights, k=1)[0]

                if op == 'identity':
                    return VariableNode(input_var)
                elif op == 'filter':
                    return self._generate_filter(input_var, elem_type, depth, context, substitutions)
                elif op == 'map':
                    return self._generate_map(input_var, elem_type, actual_ret, depth, context, substitutions)
                elif op == 'reverse':
                    return ApplicationNode(VariableNode('reverse'), [VariableNode(input_var)])
                elif op == 'unique':
                    return ApplicationNode(VariableNode('unique'), [VariableNode(input_var)])

        # Generic fallback
        if param_names:
            param_type = substitute_type_vars(param_types[0], substitutions)
            if matchable(param_type, actual_ret, substitutions.copy(), update=False):
                return VariableNode(param_names[0])

        return self._generate_expression_of_type(actual_ret, depth, context, substitutions)

    def _generate_simple_list_op(
        self,
        input_var: str,
        input_type: TypeType,
        output_type: TypeType,
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a simple list operation."""
        input_node = VariableNode(input_var)
        simple_ops = self._get_available_simple_ops()

        if not simple_ops:
            # No operations available, return identity
            return input_node

        # Build weights for available ops
        all_weights = {
            'reverse': 0.25, 'unique': 0.20,
            'take': 0.15, 'drop': 0.15,
            'takelast': 0.12, 'droplast': 0.13
        }
        available_weights = {op: all_weights.get(op, 0.1) for op in simple_ops}

        # Renormalize
        total = sum(available_weights.values())
        ops = list(available_weights.keys())
        weights = [available_weights[o] / total for o in ops]

        op = self.rng.choices(ops, weights=weights, k=1)[0]

        if op in ['reverse', 'unique']:
            return ApplicationNode(VariableNode(op), [input_node])
        else:
            n = self.rng.randint(1, 5)
            return ApplicationNode(VariableNode(op), [NumberNode(n), input_node])

    def _substitute_var_in_expr(self, expr: ASTNode, var_name: str, replacement: ASTNode) -> ASTNode:
        """Replace a variable with an expression."""
        if isinstance(expr, VariableNode):
            if expr.name == var_name:
                return replacement
            return expr
        elif isinstance(expr, NumberNode) or isinstance(expr, BooleanNode):
            return expr
        elif isinstance(expr, LambdaNode):
            params = expr.param if isinstance(expr.param, list) else [expr.param]
            if var_name in params:
                return expr
            return LambdaNode(expr.param, self._substitute_var_in_expr(expr.body, var_name, replacement))
        elif isinstance(expr, ApplicationNode):
            new_fn = self._substitute_var_in_expr(expr.function, var_name, replacement)
            new_args = [self._substitute_var_in_expr(arg, var_name, replacement) for arg in expr.arguments]
            return ApplicationNode(new_fn, new_args)
        elif isinstance(expr, IfNode):
            return IfNode(
                self._substitute_var_in_expr(expr.condition, var_name, replacement),
                self._substitute_var_in_expr(expr.then_expr, var_name, replacement),
                self._substitute_var_in_expr(expr.else_expr, var_name, replacement)
            )
        elif isinstance(expr, ListNode):
            return ListNode([self._substitute_var_in_expr(e, var_name, replacement) for e in expr.elements])
        return expr

    def _generate_list_expression(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a list expression."""
        for var_name, var_type in context.items():
            if matchable(var_type, target_type, substitutions.copy(), update=False):
                return VariableNode(var_name)

        if self._has_function('singleton'):
            const = self.dists.sample_int_constant(self.rng, self.noise)
            return ApplicationNode(VariableNode('singleton'), [NumberNode(const)])

        # Fall back to empty list
        return ListNode([])

    def _generate_int_expression(
        self,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate an integer expression."""
        arithmetic_ops = self._get_available_arithmetic_ops()
        # Filter to safe ops
        safe_ops = [op for op in arithmetic_ops if op in ['+', '-', '*']]

        if depth <= 0 or not safe_ops or self.rng.random() < 0.5:
            return NumberNode(self.dists.sample_int_constant(self.rng, self.noise))

        op = self.rng.choice(safe_ops)
        left = NumberNode(self.rng.randint(0, 20))
        right = NumberNode(self.rng.randint(1, 10))
        return ApplicationNode(VariableNode(op), [left, right])

    def _generate_bool_expression(
        self,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a boolean expression."""
        comparison_ops = self._get_available_comparison_ops()

        if depth <= 0 or not comparison_ops or self.rng.random() < 0.3:
            return BooleanNode(self.dists.sample_bool_constant(self.rng, self.noise))

        op = self.rng.choice(comparison_ops)
        left = NumberNode(self.dists.sample_int_constant(self.rng, self.noise))
        right = NumberNode(self.dists.sample_int_constant(self.rng, self.noise))
        return ApplicationNode(VariableNode(op), [left, right])

    def _generate_expression_of_type(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate an expression of the given type."""
        base = get_base_type(target_type)

        if target_type == int:
            return self._generate_int_expression(depth, context, substitutions)
        elif target_type == bool:
            return self._generate_bool_expression(depth, context, substitutions)
        elif base == list:
            return self._generate_list_expression(target_type, depth, context, substitutions)
        elif base == CallableOrig:
            return self._generate_function(target_type, depth, context, substitutions)

        return NumberNode(0)
