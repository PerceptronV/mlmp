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
        shared_coverage: Optional[tuple[Counter[str], threading.Lock]] = None
    ):
        """
        Initialize the coverage-guided composer.
        
        Args:
            seed: Random seed for reproducibility
            grammar: Grammar defining available functions
            noise: Noise parameter (0 = default weights, higher = more uniform)
            guard: Strategy guard for blocking trivial patterns
            coverage_strength: How strongly to bias toward uncovered functions.
                             Higher values = stronger preference for uncovered.
                             0 = no coverage bias (behaves like regular TemplateComposer)
            shared_coverage: Optional tuple of (Counter, Lock) for shared coverage tracking
                            across multiple composer instances (for multithreading)
        """
        super().__init__(seed, grammar, noise, guard)
        self.coverage_strength = coverage_strength
        
        # Track function usage across generations (thread-safe)
        if shared_coverage is not None:
            # Use externally provided shared coverage tracking
            self._coverage_counts, self._coverage_lock = shared_coverage
            self._total_programs = 0  # Local count not used when sharing
            self._owns_coverage = False
        else:
            # Create own coverage tracking
            self._coverage_counts = Counter()
            self._total_programs = 0
            self._coverage_lock = threading.Lock()
            self._owns_coverage = True
        
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
        """Update coverage counts from a generated program (thread-safe)."""
        program_fns = program.function_names() & set(self.grammar.names)
        with self._coverage_lock:
            self._coverage_counts.update(program_fns)
            self._total_programs += 1
    
    # ========================================================================
    # Coverage-Biased Weight Computation
    # ========================================================================
    
    def _compute_coverage_bias(self, function: str) -> float:
        """
        Compute coverage bias multiplier for a function (thread-safe).
        
        Returns a multiplier > 1 for under-covered functions
        and < 1 for over-covered functions.
        
        The formula is: bias = (avg_count + 1) / (fn_count + 1)
        where avg_count is the average count across all grammar functions.
        """
        if self.coverage_strength <= 0:
            return 1.0
        
        with self._coverage_lock:
            fn_count = self._coverage_counts.get(function, 0)
            
            # If function is required, give it VERY strong bias
            if self._required_functions and function in self._required_functions:
                if fn_count == 0:
                    # Extremely strong bias for uncovered required functions
                    return 100.0 * self.coverage_strength
                else:
                    # Still favor required functions even if covered
                    return 5.0 * self.coverage_strength
            
            # Compute average count
            total_fns = len(self.grammar.names)
            total_count = sum(self._coverage_counts.values())
            avg_count = total_count / total_fns if total_fns > 0 else 0
            
            # Bias formula: uncovered functions get higher multiplier
            bias = (avg_count + 1) / (fn_count + 1)
            
            # Apply strength scaling (raise to power)
            return bias ** self.coverage_strength
    
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
            
            # For non-required: use geometric mean of function biases
            bias_product = 1.0
            for fn in functions:
                bias_product *= self._compute_coverage_bias(fn)
            
            # Geometric mean
            aggregate_bias = bias_product ** (1.0 / len(functions))
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
        
        # Build the expression
        has_singleton = self._has_function('singleton')
        has_cons = self._has_function('cons')
        
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
        
        # Wrap in singleton if available
        if has_singleton:
            return ApplicationNode(VariableNode('singleton'), [access_expr])
        elif has_cons:
            return ApplicationNode(
                VariableNode('cons'),
                [access_expr, ListNode([])]
            )
        
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
        
        # Wrap to return list
        if self._has_function('singleton'):
            return ApplicationNode(VariableNode('singleton'), [agg_expr])
        elif self._has_function('repeat') and len(aggregators) >= 2:
            # (repeat agg1 agg2)
            other_aggs = [a for a in aggregators if a != aggregator]
            other = self._choose_aggregator_with_bias(other_aggs)
            other_expr = ApplicationNode(VariableNode(other), [input_node])
            return ApplicationNode(VariableNode('repeat'), [agg_expr, other_expr])
        
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
            if self._has_function('map') and self._has_function('singleton'):
                # (flatten (map (λ y (singleton y)) x))
                elem_var = self._fresh_var_name()
                singleton_expr = ApplicationNode(VariableNode('singleton'), [VariableNode(elem_var)])
                lambda_node = LambdaNode(elem_var, singleton_expr)
                map_expr = ApplicationNode(VariableNode('map'), [lambda_node, input_node])
                return ApplicationNode(VariableNode('flatten'), [map_expr])
            elif self._has_function('zip'):
                # (flatten (zip x (reverse x)))
                if self._has_function('reverse'):
                    rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
                    zip_expr = ApplicationNode(VariableNode('zip'), [input_node, rev_expr])
                else:
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
        
        if use_index:
            lambda_node = LambdaNode(acc_var, LambdaNode(idx_var, LambdaNode(elem_var, body)))
        else:
            lambda_node = LambdaNode(acc_var, LambdaNode(elem_var, body))
        
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
