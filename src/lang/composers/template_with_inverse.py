"""
Coverage-Guided Template Composer

This module extends the TemplateComposer with coverage-guided generation.
The key innovation is maintaining a dynamic distribution over grammar functions
that favours less-used functions, ensuring diverse coverage across programs.

Usage:
    composer = CoverageGuidedComposer(seed=42, grammar=DefaultGrammar)
    
    # Generate programs that collectively cover diverse functions
    composer.reset_coverage()
    for _ in range(n_support):
        program = composer.generate(target_type, depth)
        # Coverage automatically updated
    
    # Or explicitly set required functions for the next generation
    composer.set_required_functions({'map', 'filter', '+'})
    program = composer.generate(target_type, depth)
"""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Optional, Set
import random
import threading

from .template import TemplateComposer
from .utils.strategies import StrategyType
from .utils.guard import (
    guard_transform_weights,
    guard_predicate_weights,
    ApplicationContext,
    get_default_guard,
)
from ..grammar import Grammar
from ..ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, IfNode, ListNode
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
from pathlib import Path


def load_target_distribution(grammar_names: set[str]) -> dict[str, float]:
    """
    Load target function distribution from reference functions.txt.

    Returns a dict mapping function name -> target frequency (0-1).
    Functions not in the reference get a small baseline frequency.

    Args:
        grammar_names: Set of valid grammar function names to filter by

    Returns:
        Dict of function -> target frequency (sums to ~1)
    """
    from ..parser import parse

    # Find functions.txt relative to this file
    this_dir = Path(__file__).parent
    functions_path = this_dir.parent.parent / 'data' / 'rule' / 'functions.txt'

    if not functions_path.exists():
        # Fall back to uniform distribution
        n = len(grammar_names)
        return {fn: 1.0 / n for fn in grammar_names}

    # Count function occurrences in reference
    fn_counts: Counter[str] = Counter()
    with open(functions_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ast = parse(line)
                fns = ast.function_names() & grammar_names
                fn_counts.update(fns)
            except Exception:
                pass

    # Convert to frequencies
    total = sum(fn_counts.values())
    if total == 0:
        n = len(grammar_names)
        return {fn: 1.0 / n for fn in grammar_names}

    # Compute target frequencies with smoothing for missing functions
    # Use Laplace smoothing: (count + alpha) / (total + alpha * n_functions)
    alpha = 0.5  # Smoothing parameter
    n_fns = len(grammar_names)
    smoothed_total = total + alpha * n_fns

    target_dist = {}
    for fn in grammar_names:
        count = fn_counts.get(fn, 0)
        target_dist[fn] = (count + alpha) / smoothed_total

    return target_dist


def build_function_to_templates_mapping() -> dict[str, list[str]]:
    """
    Build a static mapping from function names to templates that can use them.
    
    This is the "inverse" of the template requirements: given a function,
    which templates can produce a program using that function?
    
    Returns:
        Dict mapping function name -> list of template names
    """
    mapping: dict[str, list[str]] = defaultdict(list)
    
    # Direct template functions (function is the main operation of the template)
    template_functions = {
        'map': ['map', 'mapi'],
        'filter': ['filter', 'filteri'],
        'sort': ['sort'],
        'fold': ['fold', 'foldi'],
        'list_access': ['first', 'last', 'second', 'third', 'nth'],
        'list_construct': ['cons', 'append', 'concat', 'repeat', 'singleton', 'flatten'],
        'aggregation': ['max', 'min', 'sum', 'product', 'length'],
        'structural': ['slice', 'swap', 'cut_idx', 'cut_val', 'cut_vals', 
                       'insert', 'replace', 'cut_slice', 'splice'],
        'advanced_hof': ['count', 'zip', 'find', 'group'],
        'simple_op': ['reverse', 'unique', 'take', 'drop', 'takelast', 'droplast'],
    }
    
    for template, funcs in template_functions.items():
        for fn in funcs:
            mapping[fn].append(template)
    
    # Predicate functions (appear inside filter, count, find templates)
    predicate_funcs = ['is_even', 'is_odd', '<', '>', '==', 'and', 'or', 'not', 'is_in']
    predicate_templates = ['filter', 'advanced_hof', 'conditional']
    for fn in predicate_funcs:
        mapping[fn].extend(predicate_templates)
    
    # Transform functions (appear inside map, fold templates)
    transform_funcs = ['+', '-', '*', '/', '%']
    transform_templates = ['map', 'fold', 'aggregation']
    for fn in transform_funcs:
        mapping[fn].extend(transform_templates)
    
    # Range function
    mapping['range'].extend(['aggregation', 'list_construct'])
    
    return mapping


# Static mapping built once
FUNCTION_TO_TEMPLATES = build_function_to_templates_mapping()


class CoverageGuidedComposer(TemplateComposer):
    """
    Template composer with coverage-guided generation.
    
    Maintains a dynamic distribution over grammar functions that biases
    generation toward less-used functions. This ensures that when generating
    multiple programs (e.g., support examples), they collectively cover
    diverse functions from the grammar.
    
    The coverage bias is applied at every choice point:
    - Template selection
    - Predicate form selection
    - Transform form selection
    - Operator selection (comparisons, arithmetic)
    - Accessor selection (first, last, nth, etc.)
    
    Key methods:
    - reset_coverage(): Clear coverage tracking for new batch
    - set_required_functions(fns): Force next generation to include these functions
    - get_coverage(): Get current coverage counts
    - bias_weights_by_coverage(weights, functions): Apply coverage bias to weights
    """
    
    @classmethod
    def get_name(cls) -> str:
        return "template_coverage"
    
    def __init__(
        self,
        seed: int,
        grammar: Grammar,
        noise: float = 0.0,
        guard=None,
        coverage_strength: float = 2.0,
        shared_coverage: Optional[tuple] = None,  # (Counter, Lock) or (Counter, Lock, [cycle])
        target_distribution: Optional[dict[str, float]] = None
    ):
        """
        Initialize the coverage-guided composer.

        Args:
            seed: Random seed for reproducibility
            grammar: Grammar defining available functions
            noise: Noise parameter (0 = default weights, higher = more uniform)
            guard: Strategy guard for blocking trivial patterns
            coverage_strength: How strongly to bias toward target distribution.
                             Higher values = stronger preference for under-represented.
                             0 = no coverage bias (behaves like regular TemplateComposer)
            shared_coverage: Optional tuple of (Counter, Lock) for shared coverage tracking
                            across multiple composer instances (for multithreading)
            target_distribution: Optional dict of function -> target frequency.
                               If provided, bias toward this distribution instead of uniform.
                               Use load_target_distribution() to load from functions.txt.
        """
        super().__init__(seed, grammar, noise, guard)
        self.coverage_strength = coverage_strength

        # Target distribution (if None, use uniform)
        if target_distribution is not None:
            self._target_distribution = target_distribution
        else:
            # Default to uniform distribution
            n = len(grammar.names)
            self._target_distribution = {fn: 1.0 / n for fn in grammar.names}

        # Track function usage across generations (thread-safe)
        # shared_coverage can be:
        #   - (Counter, Lock) - legacy format
        #   - (Counter, Lock, [cycle_number]) - extended format with cycle tracking
        if shared_coverage is not None:
            if len(shared_coverage) == 3:
                self._coverage_counts, self._coverage_lock, self._shared_cycle = shared_coverage
            else:
                # Legacy format - add cycle tracking
                self._coverage_counts, self._coverage_lock = shared_coverage
                self._shared_cycle = [0]
            self._total_programs = 0
            self._owns_coverage = False
        else:
            # Create own coverage tracking
            self._coverage_counts = Counter()
            self._total_programs = 0
            self._coverage_lock = threading.Lock()
            self._shared_cycle = [0]
            self._owns_coverage = True

        # Thread-local cycle number for detecting resets
        self._local_cycle = 0

        # Required functions for next generation (optional)
        self._required_functions: Optional[Set[str]] = None

        # Build function-to-template mapping filtered by grammar
        self._fn_to_templates = self._build_grammar_fn_to_templates()

        # Cache template-to-functions mapping (inverse of above)
        self._template_to_fns = self._build_template_to_functions()
    
    def _build_grammar_fn_to_templates(self) -> dict[str, list[str]]:
        """Build function-to-templates mapping filtered by available grammar functions."""
        result = {}
        grammar_fns = set(self.grammar.names)
        
        for fn, templates in FUNCTION_TO_TEMPLATES.items():
            if fn in grammar_fns:
                # Filter to templates that are actually available
                available_templates = [t for t in templates if self._is_template_available(t)]
                if available_templates:
                    result[fn] = available_templates
        
        return result
    
    def _build_template_to_functions(self) -> dict[str, set[str]]:
        """Build mapping from template -> functions it can use."""
        result: dict[str, set[str]] = defaultdict(set)
        
        for fn, templates in self._fn_to_templates.items():
            for template in templates:
                result[template].add(fn)
        
        return dict(result)
    
    # ========================================================================
    # Coverage Tracking
    # ========================================================================
    
    def reset_coverage(self) -> None:
        """Reset coverage tracking. Call when starting a new batch of programs."""
        self._coverage_counts.clear()
        self._total_programs = 0
        self._required_functions = None
    
    def get_coverage(self) -> dict[str, int]:
        """Get current coverage counts for all functions."""
        return dict(self._coverage_counts)
    
    def get_uncovered_functions(self) -> set[str]:
        """Get functions that haven't been used yet."""
        all_fns = set(self.grammar.names)
        covered = set(self._coverage_counts.keys())
        return all_fns - covered
    
    def get_coverage_ratio(self) -> float:
        """Get fraction of grammar functions that have been covered."""
        all_fns = set(self.grammar.names)
        covered = set(self._coverage_counts.keys())
        return len(covered) / len(all_fns) if all_fns else 1.0
    
    def set_required_functions(self, functions: Optional[Set[str]]) -> None:
        """
        Set functions that MUST appear in the next generated program.
        
        The composer will bias heavily toward templates and patterns
        that include these functions.
        
        Args:
            functions: Set of function names to require, or None to clear
        """
        if functions:
            # Filter to functions in grammar
            self._required_functions = functions & set(self.grammar.names)
        else:
            self._required_functions = None
    
    def _update_coverage(self, program: ASTNode) -> None:
        """
        Update coverage counts from a generated program (thread-safe).

        Also checks for cycle completion - if all functions are now covered,
        resets counts and increments the shared cycle number.
        """
        program_fns = program.function_names() & set(self.grammar.names)
        total_fns = len(self.grammar.names)

        with self._coverage_lock:
            self._coverage_counts.update(program_fns)
            self._total_programs += 1

            # Check if we've achieved full coverage - start new cycle
            covered_fns = sum(1 for c in self._coverage_counts.values() if c > 0)
            if covered_fns >= total_fns:
                # Full coverage achieved! Reset for new cycle
                self._coverage_counts.clear()
                self._shared_cycle[0] += 1
    
    # ========================================================================
    # Coverage-Biased Weight Computation
    # ========================================================================
    
    def _compute_coverage_bias(self, function: str) -> float:
        """
        Compute coverage bias multiplier for a function (thread-safe).

        Uses a cyclical coverage system with gradually increasing pressure:
        - Tracks which functions have been covered (count > 0) in current cycle
        - As coverage progresses, pressure on uncovered functions increases
        - When 100% coverage is achieved, cycle resets (in _update_coverage)

        Formula for uncovered functions (aggressive, non-linear):
            bias = 1 + (coverage_ratio ** 1.5) * max_pressure
        Where coverage_ratio = covered_fns / total_fns

        This non-linear formula builds pressure more aggressively:
        - At 0% coverage: bias = 1 (no extra pressure)
        - At 50% coverage: bias = 1 + 0.35 * max_pressure (starting to build)
        - At 80% coverage: bias = 1 + 0.72 * max_pressure (strong pressure)
        - At 90% coverage: bias = 1 + 0.85 * max_pressure (very strong)
        - At 95% coverage: bias = 1 + 0.93 * max_pressure (extreme pressure)

        For covered functions: slight reduction to favor uncovered ones.
        """
        if self.coverage_strength <= 0:
            return 1.0

        with self._coverage_lock:
            fn_count = self._coverage_counts.get(function, 0)

            # If function is required, give it VERY strong bias
            if self._required_functions and function in self._required_functions:
                if fn_count == 0:
                    return 100.0 * self.coverage_strength
                else:
                    return 5.0 * self.coverage_strength

            # Compute coverage progress in current cycle
            total_fns = len(self.grammar.names)
            covered_fns = sum(1 for c in self._coverage_counts.values() if c > 0)
            coverage_ratio = covered_fns / total_fns if total_fns > 0 else 0

            if fn_count == 0:
                # Uncovered function: very aggressive bias that grows with coverage
                # Using square root provides strong, consistent pressure
                # At 50%: sqrt(0.5) * 500 * 3 = 1061x
                # At 75%: sqrt(0.75) * 500 * 3 = 1299x
                # At 90%: sqrt(0.9) * 500 * 3 = 1423x
                import math
                max_pressure = 500.0 * self.coverage_strength
                bias = 1.0 + math.sqrt(coverage_ratio) * max_pressure
            else:
                # Covered function: slight reduction to favor uncovered
                # As cycle progresses, covered functions get relatively less weight
                bias = 1.0 / (1.0 + coverage_ratio * 0.5)

            return bias
    
    def bias_weights_by_coverage(
        self,
        weights: dict[str, float],
        item_to_functions: dict[str, set[str]]
    ) -> dict[str, float]:
        """
        Apply coverage bias to a weight dictionary.

        Args:
            weights: Dict of item -> weight
            item_to_functions: Dict of item -> set of functions it uses

        Returns:
            New weights dict with coverage bias applied
        """
        if self.coverage_strength <= 0:
            return weights.copy()
        
        biased = {}
        for item, weight in weights.items():
            if weight <= 0:
                biased[item] = 0
                continue
            
            # Get functions this item can use
            functions = item_to_functions.get(item, set())
            
            if not functions:
                biased[item] = weight
                continue
            
            # Special handling for required functions:
            # If this item can use ANY required function, give it a big boost
            if self._required_functions:
                required_in_item = functions & self._required_functions
                if required_in_item:
                    # Strong boost for items that can use required functions
                    # Use max bias among required functions
                    max_bias = max(self._compute_coverage_bias(fn) for fn in required_in_item)
                    biased[item] = weight * max_bias
                    continue
            
            # Use arithmetic mean of function biases (less dilution than geometric)
            # Geometric mean caused excessive dilution: (1000*0.5*0.5)^(1/3) = 7.9x
            # Arithmetic mean gives: (1000+0.5+0.5)/3 = 333x (much better!)
            bias_sum = sum(self._compute_coverage_bias(fn) for fn in functions)
            aggregate_bias = bias_sum / len(functions) if functions else 1.0
            biased[item] = weight * aggregate_bias
        
        return biased
    
    def _bias_function_choices(
        self,
        functions: list[str],
        base_weights: Optional[list[float]] = None
    ) -> list[float]:
        """
        Compute biased weights for choosing among functions.
        
        Args:
            functions: List of function names to choose from
            base_weights: Optional base weights (uniform if not provided)
        
        Returns:
            List of biased weights
        """
        if base_weights is None:
            base_weights = [1.0] * len(functions)
        
        if self.coverage_strength <= 0:
            return base_weights
        
        return [
            w * self._compute_coverage_bias(fn)
            for w, fn in zip(base_weights, functions)
        ]
    
    # ========================================================================
    # Override Template Selection
    # ========================================================================
    
    def _get_available_templates(self) -> dict[str, float]:
        """
        Get available templates with coverage bias applied.
        
        Overrides parent to apply coverage-guided weighting.
        """
        # Get base available templates
        base_templates = super()._get_available_templates()
        
        if self.coverage_strength <= 0:
            return base_templates
        
        # Apply coverage bias
        return self.bias_weights_by_coverage(base_templates, self._template_to_fns)
    
    # ========================================================================
    # Override Predicate Generation
    # ========================================================================
    
    def _get_available_predicate_forms(
        self,
        elem_type: TypeType,
        use_index: bool = False
    ) -> list[tuple[str, float]]:
        """
        Get available predicate forms with coverage bias.
        """
        # Get base forms
        base_forms = super()._get_available_predicate_forms(elem_type, use_index)
        
        if self.coverage_strength <= 0 or not base_forms:
            return base_forms
        
        # Map predicate forms to functions they use
        form_to_functions = {
            'is_even_odd': {'is_even', 'is_odd'},
            'compare_const': {'<', '>', '=='},
            'modulo_check': {'%', '=='},
            'compound': {'and', 'or'},
            'index_based': {'<', '>', '==', 'is_even', 'is_odd'},
            'is_in': {'is_in'},
            'equality': {'=='},
            'trivial': set(),
        }
        
        # Apply coverage bias
        forms_dict = {f[0]: f[1] for f in base_forms}
        biased = self.bias_weights_by_coverage(forms_dict, form_to_functions)
        
        return [(name, weight) for name, weight in biased.items()]
    
    def _get_available_comparison_ops(self) -> list[str]:
        """Get available comparison operators."""
        ops = ['<', '>', '==']
        return [op for op in ops if self._has_function(op)]
    
    def _get_available_arithmetic_ops(self) -> list[str]:
        """Get available arithmetic operators."""
        ops = ['+', '-', '*']
        return [op for op in ops if self._has_function(op)]
    
    def _choose_comparison_op(self) -> str:
        """Choose a comparison operator with coverage bias."""
        ops = self._get_available_comparison_ops()
        if not ops:
            return '=='
        if len(ops) == 1 or self.coverage_strength <= 0:
            return ops[0]
        
        weights = [self._compute_coverage_bias(op) for op in ops]
        return self.rng.choices(ops, weights=weights, k=1)[0]
    
    def _choose_arithmetic_op(self) -> str:
        """Choose an arithmetic operator with coverage bias."""
        ops = self._get_available_arithmetic_ops()
        if not ops:
            return '+'
        if len(ops) == 1 or self.coverage_strength <= 0:
            return ops[0]
        
        weights = [self._compute_coverage_bias(op) for op in ops]
        return self.rng.choices(ops, weights=weights, k=1)[0]
    
    # ========================================================================
    # Override Predicate Component Selection
    # ========================================================================
    
    def _generate_simple_predicate(self, elem_var: str, elem_type: TypeType) -> ASTNode:
        """Generate a simple predicate with coverage-biased operator selection."""
        elem_node = VariableNode(elem_var)

        # Check what's available
        available_parity = [fn for fn in ['is_even', 'is_odd'] if self._has_function(fn)]
        available_compare = self._get_available_comparison_ops()

        options = []
        if available_parity and elem_type == int:
            options.append('is_even_odd')
        if available_compare:
            options.append('compare')

        if not options:
            return BooleanNode(True)

        form = self.rng.choice(options)

        if form == 'is_even_odd':
            if len(available_parity) == 1 or self.coverage_strength <= 0:
                fn_name = available_parity[0]
            else:
                weights = [self._compute_coverage_bias(fn) for fn in available_parity]
                fn_name = self.rng.choices(available_parity, weights=weights, k=1)[0]
            return ApplicationNode(VariableNode(fn_name), [elem_node])
        else:
            op = self._choose_comparison_op()
            const = self.rng.randint(0, 99)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])
    
    # ========================================================================
    # Override Transform Generation  
    # ========================================================================
    
    def _generate_predicate(
        self,
        elem_var: str,
        elem_type: TypeType,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        use_index: bool = False,
        idx_var: str = None
    ) -> ASTNode:
        """Override to use coverage-biased operator selection."""
        actual_type = substitute_type_vars(elem_type, substitutions)
        elem_node = VariableNode(elem_var)

        # Get available predicate forms with coverage bias
        available_forms = self._get_available_predicate_forms(actual_type, use_index and idx_var is not None)

        if not available_forms:
            return BooleanNode(True)

        # Extract forms and weights, apply noise
        forms_dict = {f[0]: f[1] for f in available_forms}
        forms = list(forms_dict.keys())
        weights = self._perturb_weights(list(forms_dict.values()))

        # Apply guards AFTER noise
        perturbed_dict = dict(zip(forms, weights))
        guarded_dict = guard_predicate_weights(perturbed_dict, must_be_meaningful=True)

        if not any(w > 0 for w in guarded_dict.values()):
            if actual_type == int and self._has_function('is_even'):
                guarded_dict['is_even_odd'] = 1.0
            elif actual_type == int and any(self._has_function(op) for op in ['<', '>', '==']):
                guarded_dict['compare_const'] = 1.0
            else:
                return BooleanNode(True)

        forms = list(guarded_dict.keys())
        weights = list(guarded_dict.values())
        form = self.rng.choices(forms, weights=weights, k=1)[0]

        if form == 'is_even_odd':
            available_parity = [fn for fn in ['is_even', 'is_odd'] if self._has_function(fn)]
            if len(available_parity) == 1 or self.coverage_strength <= 0:
                fn_name = available_parity[0]
            else:
                fn_weights = [self._compute_coverage_bias(fn) for fn in available_parity]
                fn_name = self.rng.choices(available_parity, weights=fn_weights, k=1)[0]
            return ApplicationNode(VariableNode(fn_name), [elem_node])

        elif form == 'compare_const':
            op = self._choose_comparison_op()
            const = self.rng.randint(0, 99)
            # Sometimes use modulo in comparison if available
            if self.rng.random() < 0.3 and self._has_function('%'):
                mod_expr = ApplicationNode(VariableNode('%'), [elem_node, NumberNode(self.rng.choice([2, 3, 5, 10]))])
                return ApplicationNode(VariableNode(op), [mod_expr, NumberNode(self.rng.randint(0, 9))])
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

        elif form == 'modulo_check':
            divisor = self.rng.choice([2, 3, 5, 10])
            remainder = self.rng.randint(0, divisor - 1)
            mod_expr = ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])
            return ApplicationNode(VariableNode('=='), [mod_expr, NumberNode(remainder)])

        elif form == 'compound':
            pred1 = self._generate_simple_predicate(elem_var, actual_type)
            pred2 = self._generate_simple_predicate(elem_var, actual_type)
            available_combiners = [c for c in ['and', 'or'] if self._has_function(c)]
            if len(available_combiners) == 1 or self.coverage_strength <= 0:
                combiner = available_combiners[0]
            else:
                combiner_weights = [self._compute_coverage_bias(c) for c in available_combiners]
                combiner = self.rng.choices(available_combiners, weights=combiner_weights, k=1)[0]
            # Occasionally wrap with 'not' if available
            result = ApplicationNode(VariableNode(combiner), [pred1, pred2])
            if self._has_function('not') and self.rng.random() < 0.2:
                result = ApplicationNode(VariableNode('not'), [result])
            return result

        elif form == 'index_based' and idx_var:
            idx_node = VariableNode(idx_var)
            available_idx_ops = []
            for op in ['<', '>', '==']:
                if self._has_function(op):
                    available_idx_ops.append(op)
            for fn in ['is_even', 'is_odd']:
                if self._has_function(fn):
                    available_idx_ops.append(fn)
            
            if len(available_idx_ops) == 1 or self.coverage_strength <= 0:
                op = available_idx_ops[0]
            else:
                op_weights = [self._compute_coverage_bias(op) for op in available_idx_ops]
                op = self.rng.choices(available_idx_ops, weights=op_weights, k=1)[0]
            
            if op in ['is_even', 'is_odd']:
                return ApplicationNode(VariableNode(op), [idx_node])
            else:
                const = self.rng.randint(0, 10)
                return ApplicationNode(VariableNode(op), [idx_node, NumberNode(const)])

        elif form == 'is_in' and self._has_function('is_in'):
            list_vars = [name for name, t in context.items() if get_base_type(t) == list]
            if list_vars:
                list_var = self.rng.choice(list_vars)
                return ApplicationNode(VariableNode('is_in'), [VariableNode(list_var), elem_node])
            return ApplicationNode(VariableNode('=='), [elem_node, NumberNode(self.rng.randint(0, 99))])

        elif form == 'equality' and self._has_function('=='):
            return ApplicationNode(VariableNode('=='), [elem_node, NumberNode(self.rng.randint(0, 99))])

        else:
            return BooleanNode(True)
    
    def _generate_transform(
        self,
        elem_var: str,
        elem_type: TypeType,
        ret_type: TypeType,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        use_index: bool = False,
        idx_var: str = None,
        allow_identity: bool = False
    ) -> ASTNode:
        """Override to use coverage-biased operator selection."""
        actual_elem = substitute_type_vars(elem_type, substitutions)
        actual_ret = substitute_type_vars(ret_type, substitutions)
        elem_node = VariableNode(elem_var)

        # Get available transform forms with coverage bias
        available_forms = self._get_available_transform_forms(
            elem_type, ret_type, use_index and idx_var is not None, substitutions,
            allow_identity=allow_identity
        )

        # Extract forms and weights, apply noise
        forms_dict = {f[0]: f[1] for f in available_forms}
        forms = list(forms_dict.keys())
        weights = self._perturb_weights(list(forms_dict.values()))

        # Apply guards AFTER noise
        perturbed_dict = dict(zip(forms, weights))
        guarded_dict = guard_transform_weights(perturbed_dict, allow_identity=allow_identity)

        # If guard zeroed all forms, add a fallback
        if not any(w > 0 for w in guarded_dict.values()):
            if any(self._has_function(op) for op in ['+', '-', '*']):
                guarded_dict['arithmetic'] = 1.0
            elif self._has_function('%'):
                guarded_dict['modulo'] = 1.0
            else:
                guarded_dict['conditional'] = 1.0

        forms = list(guarded_dict.keys())
        weights = list(guarded_dict.values())
        form = self.rng.choices(forms, weights=weights, k=1)[0]

        if form == 'identity':
            return elem_node

        elif form == 'arithmetic':
            op = self._choose_arithmetic_op()
            const = self.rng.randint(1, 10)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

        elif form == 'modulo':
            divisor = self.rng.choice([2, 3, 5, 10, 100])
            return ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])

        elif form == 'division':
            divisor = self.rng.choice([2, 3, 5, 10, 100])
            return ApplicationNode(VariableNode('/'), [elem_node, NumberNode(divisor)])

        elif form == 'with_index' and idx_var:
            idx_node = VariableNode(idx_var)
            op = self._choose_arithmetic_op()
            return ApplicationNode(VariableNode(op), [elem_node, idx_node])

        elif form == 'conditional':
            if actual_elem == int:
                pred = self._generate_simple_predicate(elem_var, actual_elem)
            else:
                pred = ApplicationNode(VariableNode('=='), [elem_node, elem_node])
            then_val = NumberNode(self.rng.randint(0, 10))
            else_val = NumberNode(self.rng.randint(0, 10))
            return IfNode(pred, then_val, else_val)

        elif form == 'singleton':
            return ApplicationNode(VariableNode('singleton'), [elem_node])

        return elem_node
    
    def _generate_key_function_body(
        self,
        elem_var: str,
        elem_type: TypeType,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Override to use coverage-biased operator selection."""
        actual_type = substitute_type_vars(elem_type, substitutions)
        elem_node = VariableNode(elem_var)

        if actual_type != int:
            return elem_node

        # Get available key forms with coverage bias
        available_forms = self._get_available_key_forms(actual_type)

        # Extract forms and weights, apply noise
        forms_dict = {f[0]: f[1] for f in available_forms}
        forms = list(forms_dict.keys())
        weights = self._perturb_weights(list(forms_dict.values()))

        # Apply guards
        from .utils.guard import guard_key_weights
        perturbed_dict = dict(zip(forms, weights))
        guarded_dict = guard_key_weights(perturbed_dict, allow_identity=True)

        forms = list(guarded_dict.keys())
        weights = list(guarded_dict.values())
        form = self.rng.choices(forms, weights=weights, k=1)[0]

        if form == 'identity':
            return elem_node
        elif form == 'negate':
            return ApplicationNode(VariableNode('-'), [NumberNode(0), elem_node])
        elif form == 'modulo':
            divisor = self.rng.choice([2, 3, 5, 10])
            return ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])
        elif form == 'arithmetic':
            op = self._choose_arithmetic_op()
            const = self.rng.randint(1, 10)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

        return elem_node
    
    def _get_available_transform_forms(
        self,
        elem_type: TypeType,
        ret_type: TypeType,
        use_index: bool = False,
        substitutions: SubstitutionTable | None = None,
        allow_identity: bool = False
    ) -> list[tuple[str, float]]:
        """
        Get available transform forms with coverage bias.
        """
        # Get base forms
        base_forms = super()._get_available_transform_forms(
            elem_type, ret_type, use_index, substitutions, allow_identity
        )
        
        if self.coverage_strength <= 0 or not base_forms:
            return base_forms
        
        # Map transform forms to functions they use
        form_to_functions = {
            'identity': set(),
            'arithmetic': {'+', '-', '*'},
            'division': {'/'},
            'modulo': {'%'},
            'with_index': {'+', '-', '*'},
            'conditional': set(),  # Uses predicates
            'singleton': {'singleton'},
        }
        
        # Apply coverage bias
        forms_dict = {f[0]: f[1] for f in base_forms}
        biased = self.bias_weights_by_coverage(forms_dict, form_to_functions)
        
        return [(name, weight) for name, weight in biased.items()]
    
    # ========================================================================
    # Override Key Function Generation
    # ========================================================================
    
    def _get_available_key_forms(self, elem_type: TypeType) -> list[tuple[str, float]]:
        """Get available key function forms with coverage bias."""
        base_forms = super()._get_available_key_forms(elem_type)
        
        if self.coverage_strength <= 0 or not base_forms:
            return base_forms
        
        # Map key forms to functions
        form_to_functions = {
            'identity': set(),
            'negate': {'-'},
            'modulo': {'%'},
            'arithmetic': {'+', '-', '*'},
        }
        
        forms_dict = {f[0]: f[1] for f in base_forms}
        biased = self.bias_weights_by_coverage(forms_dict, form_to_functions)
        
        return [(name, weight) for name, weight in biased.items()]
    
    # ========================================================================
    # Override Expression Generation
    # ========================================================================
    
    def _generate_int_expression(
        self,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Override to use coverage-biased operator selection."""
        block_literal = self.guard.should_block_with_context(app_context, StrategyType.LITERAL)

        if not block_literal and (depth <= 0 or self.rng.random() < 0.5):
            return NumberNode(self.rng.randint(0, 99))

        # Try to use a context variable if available
        for var_name, var_type in context.items():
            if var_type == int:
                return VariableNode(var_name)

        # Fall back to arithmetic expression with coverage-biased op
        available_ops = self._get_available_arithmetic_ops()
        if available_ops:
            op = self._choose_arithmetic_op()
            left, right = self._generate_binary_op_args(op, depth, context, substitutions)
            return ApplicationNode(VariableNode(op), [left, right])

        return NumberNode(self.rng.randint(0, 99))
    
    def _generate_bool_expression(
        self,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Override to use coverage-biased operator selection."""
        block_literal = self.guard.should_block_with_context(app_context, StrategyType.LITERAL)

        if not block_literal and (depth <= 0 or self.rng.random() < 0.3):
            return BooleanNode(self.rng.choice([True, False]))

        # Generate a comparison expression with coverage-biased op
        available_ops = self._get_available_comparison_ops()
        if available_ops:
            op = self._choose_comparison_op()
            left, right = self._generate_binary_op_args(op, depth, context, substitutions)
            return ApplicationNode(VariableNode(op), [left, right])

        return ApplicationNode(VariableNode('=='), [NumberNode(0), NumberNode(0)])
    
    # ========================================================================
    # Coverage-Aware Selection Helpers
    # ========================================================================
    
    def _choose_with_coverage_bias(
        self,
        options: list[str],
        base_weights: Optional[list[float]] = None
    ) -> str:
        """
        Choose from options with coverage bias.
        
        Args:
            options: List of options to choose from
            base_weights: Optional base weights
        
        Returns:
            Chosen option
        """
        if not options:
            raise ValueError("No options to choose from")
        
        if len(options) == 1:
            return options[0]
        
        weights = self._bias_function_choices(options, base_weights)
        return self.rng.choices(options, weights=weights, k=1)[0]
    
    def _choose_accessor_with_bias(self, accessors: list[str]) -> str:
        """Choose an accessor (first, last, nth, etc.) with coverage bias."""
        return self._choose_with_coverage_bias(accessors)
    
    def _choose_aggregator_with_bias(self, aggregators: list[str]) -> str:
        """Choose an aggregator (max, min, sum, etc.) with coverage bias."""
        return self._choose_with_coverage_bias(aggregators)
    
    def _choose_constructor_with_bias(self, constructors: list[str]) -> str:
        """Choose a constructor (cons, append, etc.) with coverage bias."""
        return self._choose_with_coverage_bias(constructors)
    
    # ========================================================================
    # Override Main Generation to Track Coverage
    # ========================================================================
    
    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None
    ) -> ASTNode:
        """
        Generate a program WITHOUT automatically updating coverage.
        
        Overrides parent but SKIPS automatic coverage tracking.
        Coverage should be explicitly updated by calling update_coverage_from_program()
        after a program is accepted/used.
        
        Note: Does NOT clear required_functions - caller manages this via
        set_required_functions(None) when done.
        """
        program = super().generate(target_type, depth, context, substitutions)
        # Do NOT auto-update coverage here - let caller decide when to update
        return program
    
    def update_coverage_from_program(self, program: ASTNode) -> None:
        """
        Explicitly update coverage tracking from a program.
        
        Call this AFTER a program is accepted/finalized, not during generation attempts.
        """
        self._update_coverage(program)
    
    # ========================================================================
    # Override List Access Template (example of coverage-aware pattern selection)
    # ========================================================================
    
    def _generate_list_access_template(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate list access patterns with coverage-guided accessor selection.
        
        Generates diverse patterns including:
        - (singleton (accessor x))
        - (cons (accessor x) x)
        - (repeat (accessor x) n)
        - (filter (λ (y) (< y (accessor x))) x)
        - (map (λ (y) (+ y (accessor x))) x)
        """
        input_node = VariableNode(input_var)
        
        # Get available accessors with coverage bias
        accessors = []
        accessor_names = ['first', 'last', 'second', 'third', 'nth']
        for acc in accessor_names:
            if self._has_function(acc):
                accessors.append(acc)
        
        if not accessors:
            return input_node
        
        # Choose accessor with coverage bias
        accessor = self._choose_accessor_with_bias(accessors)
        
        # Build the accessor expression
        if accessor == 'nth':
            idx = self.rng.randint(0, 3)
            access_expr = ApplicationNode(
                VariableNode('nth'),
                [NumberNode(idx), VariableNode(input_var)]
            )
        else:
            access_expr = ApplicationNode(
                VariableNode(accessor),
                [VariableNode(input_var)]
            )
        
        # Build list of available patterns for diversity
        patterns = []
        
        # Pattern 1: (singleton (accessor x))
        if self._has_function('singleton'):
            patterns.append('singleton')
        
        # Pattern 2: (cons (accessor x) x) - prepend accessed element
        if self._has_function('cons'):
            patterns.append('cons_prepend')
        
        # Pattern 3: (repeat (accessor x) n) - repeat accessed element (capped)
        if self._has_function('repeat') and self._has_function('min'):
            patterns.append('repeat_accessor')
        
        # Pattern 4: (filter (λ (y) (< y (accessor x))) x) - filter by accessor
        if self._has_function('filter') and self._has_function('<'):
            patterns.append('filter_by_accessor')
        
        # Pattern 5: (map (λ (y) (+ y (accessor x))) x) - transform by accessor
        if self._has_function('map') and self._has_function('+'):
            patterns.append('map_with_accessor')
        
        if not patterns:
            return input_node
        
        # Choose pattern randomly
        pattern = self.rng.choice(patterns)
        
        elem_var = self._fresh_var_name()
        elem_node = VariableNode(elem_var)
        
        if pattern == 'singleton':
            return ApplicationNode(VariableNode('singleton'), [access_expr])
        
        elif pattern == 'cons_prepend':
            # (cons (accessor x) x)
            return ApplicationNode(VariableNode('cons'), [access_expr, input_node])
        
        elif pattern == 'repeat_accessor':
            # (repeat (accessor x) (min 5 (length x))) - repeat up to 5 times
            len_expr = ApplicationNode(VariableNode('length'), [input_node])
            capped = ApplicationNode(VariableNode('min'), [NumberNode(5), len_expr])
            return ApplicationNode(VariableNode('repeat'), [access_expr, capped])
        
        elif pattern == 'filter_by_accessor':
            # (filter (λ (y) (< y (accessor x))) x)
            pred_body = ApplicationNode(VariableNode('<'), [elem_node, access_expr])
            pred = LambdaNode(elem_var, pred_body)
            return ApplicationNode(VariableNode('filter'), [pred, input_node])
        
        elif pattern == 'map_with_accessor':
            # (map (λ (y) (+ y (accessor x))) x)
            ops = ['+', '-', '%']
            available_ops = [op for op in ops if self._has_function(op)]
            op = self._choose_with_coverage_bias(available_ops) if available_ops else '+'
            transform_body = ApplicationNode(VariableNode(op), [elem_node, access_expr])
            transform = LambdaNode(elem_var, transform_body)
            return ApplicationNode(VariableNode('map'), [transform, input_node])
        
        return input_node
    
    def _generate_aggregation_template(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate aggregation patterns with coverage-guided selection.
        
        Generates diverse patterns including:
        - (singleton (agg x))
        - (repeat 1 (agg x))
        - (filter (λ (y) (< y (agg x))) x)
        - (map (λ (y) (+ y (agg x))) x)
        - (take (min (length x) (agg x)) x)
        """
        input_node = VariableNode(input_var)
        
        # Get available aggregators
        aggregators = []
        for agg in ['max', 'min', 'sum', 'product', 'length']:
            if self._has_function(agg):
                aggregators.append(agg)
        
        if not aggregators:
            return input_node
        
        # Choose aggregator with coverage bias
        aggregator = self._choose_aggregator_with_bias(aggregators)
        
        agg_expr = ApplicationNode(VariableNode(aggregator), [input_node])
        
        # Build list of available patterns
        patterns = []
        
        # Pattern 1: (singleton (agg x)) - always available if singleton exists
        if self._has_function('singleton'):
            patterns.append('singleton')
        
        # Pattern 2: (repeat 1 (agg x)) - creates variable length output
        if self._has_function('repeat'):
            patterns.append('repeat_const')
        
        # Pattern 3: (filter (λ (y) (< y (agg x))) x) - filter by agg
        if self._has_function('filter') and self._has_function('<'):
            patterns.append('filter_by_agg')
        
        # Pattern 4: (map (λ (y) (+ y (agg x))) x) - transform by agg
        if self._has_function('map') and self._has_function('+'):
            patterns.append('map_with_agg')
        
        # Pattern 5: (take (min (length x) n) x) where n uses agg
        if self._has_function('take') and self._has_function('min') and self._has_function('length'):
            patterns.append('take_by_agg')
        
        # Pattern 6: (repeat agg1 agg2) - two aggregators
        if self._has_function('repeat') and len(aggregators) >= 2:
            patterns.append('repeat_two_aggs')
        
        if not patterns:
            return input_node
        
        # Choose pattern randomly (with some weighting toward variety)
        pattern = self.rng.choice(patterns)
        
        elem_var = self._fresh_var_name()
        elem_node = VariableNode(elem_var)
        
        if pattern == 'singleton':
            return ApplicationNode(VariableNode('singleton'), [agg_expr])
        
        elif pattern == 'repeat_const':
            # (repeat 1 (min (agg x) 15)) - repeat 1 up to 15 times (capped to avoid memory issues)
            const = NumberNode(1)
            if self._has_function('min'):
                capped_count = ApplicationNode(VariableNode('min'), [agg_expr, NumberNode(15)])
                return ApplicationNode(VariableNode('repeat'), [const, capped_count])
            else:
                # Fallback to singleton if min not available
                return ApplicationNode(VariableNode('singleton'), [agg_expr])
        
        elif pattern == 'filter_by_agg':
            # (filter (λ (y) (< y (agg x))) x)
            pred_body = ApplicationNode(VariableNode('<'), [elem_node, agg_expr])
            pred = LambdaNode(elem_var, pred_body)
            return ApplicationNode(VariableNode('filter'), [pred, input_node])
        
        elif pattern == 'map_with_agg':
            # (map (λ (y) (+ y (agg x))) x) or (map (λ (y) (% y (agg x))) x)
            # Choose operator with coverage bias
            ops = ['+', '-', '%']
            available_ops = [op for op in ops if self._has_function(op)]
            if available_ops:
                op = self._choose_with_coverage_bias(available_ops)
            else:
                op = '+'
            transform_body = ApplicationNode(VariableNode(op), [elem_node, agg_expr])
            transform = LambdaNode(elem_var, transform_body)
            return ApplicationNode(VariableNode('map'), [transform, input_node])
        
        elif pattern == 'take_by_agg':
            # (take (min (length x) (% (agg x) 10)) x)
            # Use modulo to keep take count reasonable
            len_expr = ApplicationNode(VariableNode('length'), [input_node])
            if self._has_function('%'):
                mod_agg = ApplicationNode(VariableNode('%'), [agg_expr, NumberNode(10)])
                min_expr = ApplicationNode(VariableNode('min'), [len_expr, mod_agg])
            else:
                min_expr = ApplicationNode(VariableNode('min'), [len_expr, agg_expr])
            return ApplicationNode(VariableNode('take'), [min_expr, input_node])
        
        elif pattern == 'repeat_two_aggs':
            # (repeat agg1 (min agg2 15)) - cap repeat count to avoid memory issues
            other_aggs = [a for a in aggregators if a != aggregator]
            other = self._choose_aggregator_with_bias(other_aggs)
            other_expr = ApplicationNode(VariableNode(other), [input_node])
            if self._has_function('min'):
                capped_count = ApplicationNode(VariableNode('min'), [other_expr, NumberNode(15)])
                return ApplicationNode(VariableNode('repeat'), [agg_expr, capped_count])
            else:
                # Fallback to singleton
                return ApplicationNode(VariableNode('singleton'), [agg_expr])
        
        return input_node
    
    # ========================================================================
    # Override Advanced HOF Template
    # ========================================================================
    
    def _generate_advanced_hof_template(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate advanced HOF patterns with coverage-guided function selection."""
        input_node = VariableNode(input_var)
        
        # Determine which HOF functions are available
        available_hofs = []
        if self._has_function('count'):
            available_hofs.append('count')
        if self._has_function('zip'):
            available_hofs.append('zip')
        if self._has_function('find'):
            available_hofs.append('find')
        if self._has_function('group'):
            available_hofs.append('group')
        
        if not available_hofs:
            return input_node
        
        # Choose HOF with coverage bias
        hof = self._choose_with_coverage_bias(available_hofs)
        
        if hof == 'count':
            # (singleton (count pred x)) or (map (λ y (count (== y) x)) x)
            if self._has_function('singleton') and self.rng.random() < 0.7:
                # (singleton (count pred x))
                elem_var = self._fresh_var_name()
                pred = self._generate_simple_predicate(elem_var, int)
                lambda_pred = LambdaNode(elem_var, pred)
                count_expr = ApplicationNode(VariableNode('count'), [lambda_pred, input_node])
                return ApplicationNode(VariableNode('singleton'), [count_expr])
            elif self._has_function('map'):
                # (map (λ y (count (== y) x)) x)
                elem_var = self._fresh_var_name()
                inner_var = self._fresh_var_name()
                eq_pred = ApplicationNode(VariableNode('=='), [VariableNode(elem_var), VariableNode(inner_var)])
                inner_lambda = LambdaNode(inner_var, eq_pred)
                count_expr = ApplicationNode(VariableNode('count'), [inner_lambda, input_node])
                outer_lambda = LambdaNode(elem_var, count_expr)
                return ApplicationNode(VariableNode('map'), [outer_lambda, input_node])
            else:
                # Fallback
                elem_var = self._fresh_var_name()
                pred = VariableNode('is_even') if self._has_function('is_even') else BooleanNode(True)
                if isinstance(pred, VariableNode):
                    count_expr = ApplicationNode(VariableNode('count'), [pred, input_node])
                else:
                    lambda_pred = LambdaNode(elem_var, pred)
                    count_expr = ApplicationNode(VariableNode('count'), [lambda_pred, input_node])
                return ApplicationNode(VariableNode('singleton'), [count_expr]) if self._has_function('singleton') else count_expr
        
        elif hof == 'zip':
            # Various zip patterns
            if self._has_function('reverse') and self.rng.random() < 0.5:
                rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
                zip_expr = ApplicationNode(VariableNode('zip'), [input_node, rev_expr])
                if self._has_function('flatten') and self.rng.random() < 0.5:
                    return ApplicationNode(VariableNode('flatten'), [zip_expr])
                elif self._has_function('map') and self._has_function('first'):
                    return ApplicationNode(VariableNode('map'), [VariableNode('first'), zip_expr])
                return zip_expr
            elif self._has_function('droplast') and self._has_function('drop'):
                # (zip (droplast 1 x) (drop 1 x))
                droplast_expr = ApplicationNode(VariableNode('droplast'), [NumberNode(1), input_node])
                drop_expr = ApplicationNode(VariableNode('drop'), [NumberNode(1), input_node])
                return ApplicationNode(VariableNode('zip'), [droplast_expr, drop_expr])
            else:
                # Simple zip with reverse
                if self._has_function('reverse'):
                    rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
                    return ApplicationNode(VariableNode('zip'), [input_node, rev_expr])
                return input_node
        
        elif hof == 'find':
            # (find pred x)
            elem_var = self._fresh_var_name()
            if self._has_function('is_even') and self.rng.random() < 0.5:
                pred = VariableNode('is_even')
            elif self._has_function('is_odd') and self.rng.random() < 0.5:
                pred = VariableNode('is_odd')
            else:
                inner_pred = self._generate_simple_predicate(elem_var, int)
                pred = LambdaNode(elem_var, inner_pred)
            return ApplicationNode(VariableNode('find'), [pred, input_node])
        
        elif hof == 'group':
            # (map first (group key x)) or (map length (group key x))
            elem_var = self._fresh_var_name()
            if self._has_function('%'):
                key_body = ApplicationNode(VariableNode('%'), [VariableNode(elem_var), NumberNode(10)])
            else:
                key_body = VariableNode(elem_var)
            key_lambda = LambdaNode(elem_var, key_body)
            group_expr = ApplicationNode(VariableNode('group'), [key_lambda, input_node])
            
            if self._has_function('map'):
                # Choose between first/length with coverage bias
                available_mappers = []
                if self._has_function('first'):
                    available_mappers.append('first')
                if self._has_function('length'):
                    available_mappers.append('length')
                if available_mappers:
                    mapper = self._choose_with_coverage_bias(available_mappers)
                    return ApplicationNode(VariableNode('map'), [VariableNode(mapper), group_expr])
            return group_expr
        
        return input_node
    
    # ========================================================================
    # Override List Construct Template for range/flatten
    # ========================================================================
    
    def _generate_list_construct_template(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate list construction patterns with coverage-guided selection."""
        input_node = VariableNode(input_var)
        
        patterns = []
        
        # cons, append, concat patterns
        if self._has_function('cons'):
            patterns.extend(['cons'] * 3)
        if self._has_function('append'):
            patterns.extend(['append'] * 2)
        if self._has_function('concat'):
            patterns.extend(['concat'] * 2)
        if self._has_function('repeat'):
            patterns.append('repeat')
        if self._has_function('flatten'):
            patterns.extend(['flatten'] * 2)
        if self._has_function('range'):
            patterns.extend(['range'] * 3)
        
        if not patterns:
            return input_node
        
        pattern = self.rng.choice(patterns)
        
        if pattern == 'cons':
            val = NumberNode(self.rng.randint(0, 99))
            return ApplicationNode(VariableNode('cons'), [val, input_node])
        
        elif pattern == 'append':
            val = NumberNode(self.rng.randint(0, 99))
            return ApplicationNode(VariableNode('append'), [input_node, val])
        
        elif pattern == 'concat':
            if self._has_function('reverse') and self.rng.random() < 0.4:
                rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
                return ApplicationNode(VariableNode('concat'), [rev_expr, input_node])
            else:
                return ApplicationNode(VariableNode('concat'), [input_node, input_node])
        
        elif pattern == 'repeat':
            if self._has_function('first'):
                elem = ApplicationNode(VariableNode('first'), [input_node])
            else:
                elem = NumberNode(self.rng.randint(0, 10))
            count = self.rng.randint(2, 5)
            return ApplicationNode(VariableNode('repeat'), [elem, NumberNode(count)])
        
        elif pattern == 'flatten':
            # Choose among diverse flatten patterns
            flatten_patterns = []
            
            # Pattern 1: (flatten (map (λ y (repeat y 2)) x)) - duplicate each element
            if self._has_function('map') and self._has_function('repeat'):
                flatten_patterns.append('repeat_each')
            
            # Pattern 2: (flatten (zip x (reverse x))) - interleave with reverse
            if self._has_function('zip') and self._has_function('reverse'):
                flatten_patterns.append('zip_reverse')
            
            # Pattern 3: (flatten (map (λ y (cons y (singleton y))) x)) - duplicate pairs
            if self._has_function('map') and self._has_function('cons') and self._has_function('singleton'):
                flatten_patterns.append('cons_singleton')
            
            # Pattern 4: (flatten (zip x x)) - double each element via zip
            if self._has_function('zip'):
                flatten_patterns.append('zip_self')
            
            if not flatten_patterns:
                return input_node
            
            fp = self.rng.choice(flatten_patterns)
            elem_var = self._fresh_var_name()
            
            if fp == 'repeat_each':
                # (flatten (map (λ y (repeat y 2)) x))
                repeat_expr = ApplicationNode(VariableNode('repeat'), 
                    [VariableNode(elem_var), NumberNode(2)])
                lambda_node = LambdaNode(elem_var, repeat_expr)
                map_expr = ApplicationNode(VariableNode('map'), [lambda_node, input_node])
                return ApplicationNode(VariableNode('flatten'), [map_expr])
            
            elif fp == 'zip_reverse':
                # (flatten (zip x (reverse x)))
                rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
                zip_expr = ApplicationNode(VariableNode('zip'), [input_node, rev_expr])
                return ApplicationNode(VariableNode('flatten'), [zip_expr])
            
            elif fp == 'cons_singleton':
                # (flatten (map (λ y (cons y (singleton y))) x))
                singleton_expr = ApplicationNode(VariableNode('singleton'), [VariableNode(elem_var)])
                cons_expr = ApplicationNode(VariableNode('cons'), 
                    [VariableNode(elem_var), singleton_expr])
                lambda_node = LambdaNode(elem_var, cons_expr)
                map_expr = ApplicationNode(VariableNode('map'), [lambda_node, input_node])
                return ApplicationNode(VariableNode('flatten'), [map_expr])
            
            elif fp == 'zip_self':
                # (flatten (zip x x))
                zip_expr = ApplicationNode(VariableNode('zip'), [input_node, input_node])
                return ApplicationNode(VariableNode('flatten'), [zip_expr])
            
            return input_node
        
        elif pattern == 'range':
            # Generate various range patterns
            available_aggs = []
            if self._has_function('min'):
                available_aggs.append('min')
            if self._has_function('max'):
                available_aggs.append('max')
            if self._has_function('first'):
                available_aggs.append('first')
            if self._has_function('last'):
                available_aggs.append('last')
            
            if len(available_aggs) >= 2:
                # (range (min x) (max x) step) - grammar expects (start, end, step)
                start_fn = self._choose_with_coverage_bias(available_aggs)
                end_fns = [fn for fn in available_aggs if fn != start_fn]
                end_fn = self._choose_with_coverage_bias(end_fns) if end_fns else start_fn
                
                start_expr = ApplicationNode(VariableNode(start_fn), [input_node])
                end_expr = ApplicationNode(VariableNode(end_fn), [input_node])
                step = NumberNode(self.rng.choice([1, 2]))
                return ApplicationNode(VariableNode('range'), [start_expr, end_expr, step])
            else:
                # (range 1 (last x) 1) - grammar expects (start, end, step)
                step = NumberNode(1)
                if self._has_function('last'):
                    end_expr = ApplicationNode(VariableNode('last'), [input_node])
                else:
                    end_expr = NumberNode(10)
                return ApplicationNode(VariableNode('range'), [NumberNode(1), end_expr, step])
        
        return input_node
    
    # ========================================================================
    # Override Fold Template for foldi support
    # ========================================================================
    
    def _generate_fold_template(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate fold patterns, including foldi if available."""
        input_node = VariableNode(input_var)
        
        # Decide whether to use fold or foldi
        has_fold = self._has_function('fold')
        has_foldi = self._has_function('foldi')
        
        if not has_fold and not has_foldi:
            return input_node
        
        # Choose with coverage bias
        available_folds = []
        if has_fold:
            available_folds.append('fold')
        if has_foldi:
            available_folds.append('foldi')
        
        fold_fn = self._choose_with_coverage_bias(available_folds)
        use_index = (fold_fn == 'foldi')
        
        acc_var = self._fresh_var_name()
        elem_var = self._fresh_var_name()
        idx_var = self._fresh_var_name() if use_index else None
        
        acc_node = VariableNode(acc_var)
        elem_node = VariableNode(elem_var)
        
        # Choose fold pattern with coverage bias
        # cons pattern (reverse) is more interesting than append (identity)
        patterns = []
        if self._has_function('cons'):
            patterns.extend(['cons'] * 3)  # Prefer cons (produces reverse, more interesting)
        if self._has_function('append'):
            patterns.append('append')  # Append produces identity, less interesting
        
        if not patterns:
            return input_node
        
        pattern = self._choose_with_coverage_bias(patterns) if len(patterns) > 1 else patterns[0]
        
        if pattern == 'cons':
            # Reverse via fold: (fold (λ (y z) (cons z y)) [] x)
            body = ApplicationNode(VariableNode('cons'), [elem_node, acc_node])
        else:
            # Identity via fold: (fold (λ (y z) (append y z)) [] x)
            body = ApplicationNode(VariableNode('append'), [acc_node, elem_node])
        
        init = ListNode([])
        
        # Generate foldi lambda: (λ (acc idx elem) body) as multi-parameter
        if use_index:
            lambda_node = LambdaNode([acc_var, idx_var, elem_var], body)
        else:
            lambda_node = LambdaNode([acc_var, elem_var], body)
        
        return ApplicationNode(VariableNode(fold_fn), [lambda_node, init, input_node])
    
    # ========================================================================
    # Override Simple List Op Generation
    # ========================================================================
    
    def _generate_simple_list_op(
        self,
        input_var: str,
        input_type: TypeType,
        output_type: TypeType,
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a simple list operation with coverage bias."""
        input_node = VariableNode(input_var)
        
        # Get available simple ops
        available_ops = self._get_available_simple_ops()
        
        if not available_ops:
            return input_node
        
        # Choose op with coverage bias
        op = self._choose_with_coverage_bias(available_ops)
        
        # Simple ops that take just the list
        simple_ops = ['reverse', 'unique']
        
        if op in simple_ops:
            return ApplicationNode(VariableNode(op), [input_node])
        else:
            # Operations that take an integer argument
            n = self.rng.randint(1, 5)
            return ApplicationNode(VariableNode(op), [NumberNode(n), input_node])
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_templates_for_function(self, function: str) -> list[str]:
        """Get templates that can use a specific function."""
        return self._fn_to_templates.get(function, [])
    
    def get_functions_for_template(self, template: str) -> set[str]:
        """Get functions that a template can use."""
        return self._template_to_fns.get(template, set())
    
    def get_function_to_templates_mapping(self) -> dict[str, list[str]]:
        """Get the full function -> templates mapping."""
        return dict(self._fn_to_templates)
