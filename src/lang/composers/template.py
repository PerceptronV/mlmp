"""
Template Program Composer

This module generates semantically meaningful programs using template-based
generation with context-sensitive weighted sampling. Programs follow
compositional patterns similar to those in the Rule-MPS-DSL benchmark.

Key design:
1. Templates define high-level program structures (map, filter, fold, etc.)
2. Holes in templates are filled with context-appropriate expressions
3. Weights are hand-tuned to produce behaviourally meaningful programs
"""

from typing import Optional

from .base import Composer
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


class TemplateComposer(Composer):
    """
    Generates semantically meaningful well-typed programs using templates.

    Uses template-based generation with context-sensitive hand-tuned weights
    to produce programs that follow compositional patterns (map, filter, fold, etc.)
    rather than random assemblies.

    The composer checks function availability in the grammar and only generates
    code using functions that exist. Templates and patterns that require missing
    functions are filtered out.
    """

    @classmethod
    def get_name(cls) -> str:
        return "template"

    def __init__(self, seed: int, grammar: Grammar, noise: float = 0.0, guard = None):
        """
        Initialize the TemplateComposer.

        Args:
            seed: Random seed for reproducibility
            grammar: Grammar defining available functions
            noise: Noise parameter (0 = default weights, higher = more uniform/random)
        """
        super().__init__(seed, grammar)
        self.noise = noise
        self.guard = guard or get_default_guard()

        # Higher-order functions that form meaningful templates
        self.ho_functions = {'map', 'mapi', 'filter', 'filteri', 'fold', 'foldi',
                             'sort', 'group', 'count', 'find'}

        # Functions useful in predicate bodies (return bool)
        self.predicate_funcs = {'is_even', 'is_odd', '<', '>', '==', 'and', 'or', 'not', 'is_in'}

        # Functions useful in transform bodies (return int from int)
        self.transform_funcs = {'+', '-', '*', '/', '%'}

        # Simple list operations for composition
        self.simple_list_ops = {'reverse', 'unique', 'take', 'drop', 'takelast', 'droplast'}

        # Template weights (relative probabilities)
        # Rebalanced to match canonical distribution:
        # - Reduced map/filter from 0.20 to 0.08/0.06 (canonical ~5%/2%)
        # - Added list_access (18% canonical), list_construct (19% canonical)
        # - Added aggregation (7% canonical), structural (8% canonical), fold (5% for coverage)
        # - Added advanced_hof for count, zip, find, group (~3% canonical)
        self.template_weights = {
            'identity': 0.03,
            'map': 0.07,              # Reduced from 0.20 (canonical ~5%)
            'filter': 0.05,           # Reduced from 0.20 (canonical ~2%)
            'sort': 0.04,
            'fold': 0.05,             # Increased from 0.02 to ensure fold/foldi coverage
            'composition': 0.08,      # Reduced from 0.25
            'simple_op': 0.07,
            'conditional': 0.02,      # Reduced from 0.05
            'list_access': 0.18,      # NEW - canonical ~18%
            'list_construct': 0.18,   # NEW - canonical ~19%
            'aggregation': 0.12,      # Increased - canonical ~7% (need more max/min/sum)
            'structural': 0.10,       # NEW - canonical ~8%
            'advanced_hof': 0.04,     # NEW - for count, zip, find, group (~3% canonical)
        }

        # Define function requirements for each template
        # A template is available only if ALL its required functions exist
        self._template_requirements: dict[str, set[str]] = {
            'identity': set(),  # No functions needed - just return input
            'map': {'map'},     # Needs map (or mapi, handled separately)
            'filter': {'filter'},  # Needs filter (or filteri)
            'sort': {'sort'},
            'fold': {'fold'},      # NEW - needs fold function
            'composition': set(),   # Checked dynamically based on available templates
            'simple_op': set(),     # Checked dynamically
            'conditional': {'length'},  # Needs length for condition
            'list_access': set(),      # NEW - checked dynamically (first, last, nth, etc.)
            'list_construct': set(),   # NEW - checked dynamically (cons, append, concat, etc.)
            'aggregation': set(),      # NEW - checked dynamically (max, min, sum, etc.)
            'structural': set(),       # NEW - checked dynamically (slice, swap, cut_idx, etc.)
            'advanced_hof': set(),     # NEW - checked dynamically (count, zip, find, group)
        }

        # Predicate pattern requirements
        self._predicate_requirements: dict[str, set[str]] = {
            'is_even_odd': {'is_even', 'is_odd'},  # Needs at least one
            'compare_const': {'<', '>', '=='},      # Needs at least one comparison
            'modulo_check': {'%', '=='},
            'compound': {'and', 'or'},              # Needs at least one combiner
            'index_based': {'<', '>', '==', 'is_even', 'is_odd'},  # At least one
            'equality': {'=='},
        }

        # Transform pattern requirements
        self._transform_requirements: dict[str, set[str]] = {
            'identity': set(),
            'arithmetic': {'+', '-', '*'},  # Needs at least one
            'modulo': {'%'},
            'with_index': {'+', '-', '*'},
            'conditional': set(),  # Uses predicates, checked separately
            'singleton': {'singleton'},
        }

        # Key function pattern requirements
        self._key_requirements: dict[str, set[str]] = {
            'identity': set(),
            'negate': {'-'},
            'modulo': {'%'},
            'arithmetic': {'+', '-', '*'},
        }

        # Cache available functions for efficiency
        self._available_templates: dict[str, float] | None = None
        self._available_simple_ops: list[str] | None = None

    # ========================================================================
    # Availability Checking
    # ========================================================================

    def _get_available_templates(self) -> dict[str, float]:
        """
        Get templates that are available based on grammar functions.

        Returns:
            Dictionary of template name -> weight for available templates
        """
        if self._available_templates is not None:
            return self._available_templates

        available = {}
        for template, weight in self.template_weights.items():
            required = self._template_requirements.get(template, set())

            # Special handling for templates with "at least one" requirements
            if template == 'map':
                # Needs map or mapi
                if self._has_function('map') or self._has_function('mapi'):
                    available[template] = weight
            elif template == 'filter':
                # Needs filter or filteri
                if self._has_function('filter') or self._has_function('filteri'):
                    available[template] = weight
            elif template == 'simple_op':
                # Available if at least one simple op exists
                if self._get_available_simple_ops():
                    available[template] = weight
            elif template == 'composition':
                # Available if we have at least 2 other templates to compose
                other_available = sum(
                    1 for t in ['map', 'filter', 'simple_op']
                    if t in available or (t not in available and self._is_template_available(t))
                )
                if other_available >= 1:
                    available[template] = weight
            elif template == 'list_access':
                # Needs at least one list accessor: first, last, second, third, nth
                accessors = ['first', 'last', 'second', 'third', 'nth']
                if any(self._has_function(fn) for fn in accessors):
                    available[template] = weight
            elif template == 'list_construct':
                # Needs at least one constructor: cons, append, concat, repeat
                constructors = ['cons', 'append', 'concat', 'repeat', 'singleton']
                if any(self._has_function(fn) for fn in constructors):
                    available[template] = weight
            elif template == 'aggregation':
                # Needs at least one aggregator: max, min, sum, product, length
                aggregators = ['max', 'min', 'sum', 'product', 'length']
                if any(self._has_function(fn) for fn in aggregators):
                    available[template] = weight
            elif template == 'structural':
                # Needs at least one structural op: slice, swap, cut_idx, insert, replace
                structural_ops = ['slice', 'swap', 'cut_idx', 'insert', 'replace', 'cut_slice', 'cut_val', 'cut_vals', 'splice']
                if any(self._has_function(fn) for fn in structural_ops):
                    available[template] = weight
            elif template == 'fold':
                # Needs fold function
                if self._has_function('fold'):
                    available[template] = weight
            elif template == 'advanced_hof':
                # Needs at least one of: count, zip, find, group
                advanced_hofs = ['count', 'zip', 'find', 'group']
                if any(self._has_function(fn) for fn in advanced_hofs):
                    available[template] = weight
            else:
                # Standard check: all required functions must exist
                if all(self._has_function(fn) for fn in required):
                    available[template] = weight

        # Always include identity as fallback
        if 'identity' not in available:
            available['identity'] = 0.05

        self._available_templates = available
        return available

    def _is_template_available(self, template: str) -> bool:
        """Check if a specific template is available."""
        required = self._template_requirements.get(template, set())

        if template == 'map':
            return self._has_function('map') or self._has_function('mapi')
        elif template == 'filter':
            return self._has_function('filter') or self._has_function('filteri')
        elif template == 'simple_op':
            return bool(self._get_available_simple_ops())
        else:
            return all(self._has_function(fn) for fn in required)

    def _get_available_simple_ops(self) -> list[str]:
        """Get list of available simple list operations."""
        if self._available_simple_ops is not None:
            return self._available_simple_ops

        simple_ops = ['reverse', 'unique']
        int_arg_ops = ['take', 'drop', 'takelast', 'droplast']

        available = []
        for op in simple_ops:
            if self._has_function(op):
                available.append(op)
        for op in int_arg_ops:
            if self._has_function(op):
                available.append(op)

        self._available_simple_ops = available
        return available

    def _perturb_weights(self, weights: list[float]) -> list[float]:
        """
        Perturb weights toward uniform distribution based on noise parameter.

        noise = 0: original weights
        noise = 1: uniform distribution

        Args:
            weights: List of original weights

        Returns:
            List of perturbed weights
        """
        if self.noise <= 0 or not weights:
            return weights

        n = len(weights)
        total = sum(weights)
        if total <= 0:
            return [1.0 / n] * n

        uniform = total / n
        return [(1 - self.noise) * w + self.noise * uniform for w in weights]

    def _get_available_predicate_forms(
        self,
        elem_type: TypeType,
        use_index: bool = False
    ) -> list[tuple[str, float]]:
        """
        Get available predicate forms based on grammar functions.

        Args:
            elem_type: Type of element being tested
            use_index: Whether index is available
            must_be_meaningful: If True, trivial predicates are blocked

        Returns:
            List of (form_name, weight) tuples
        """
        forms = {}

        if elem_type == int:
            # is_even_odd: needs at least one of is_even, is_odd
            # Reduced from 0.25 to 0.15 to match canonical distribution
            if self._has_function('is_even') or self._has_function('is_odd'):
                forms['is_even_odd'] = 0.15

            # compare_const: needs at least one comparison operator
            # Reduced from 0.30 to 0.20 to reduce over-representation of < and >
            if any(self._has_function(op) for op in ['<', '>', '==']):
                forms['compare_const'] = 0.20

            # modulo_check: needs % and ==
            # Reduced from 0.20 to 0.15 to reduce % over-representation
            if self._has_function('%') and self._has_function('=='):
                forms['modulo_check'] = 0.15

            # compound: needs at least one combiner and a base predicate
            if (self._has_function('and') or self._has_function('or')) and forms:
                forms['compound'] = 0.10

            # index_based: needs comparisons or is_even/is_odd
            if use_index:
                if any(self._has_function(op) for op in ['<', '>', '==', 'is_even', 'is_odd']):
                    forms['index_based'] = 0.10

            # is_in: check if element is in list - (is_in x val)
            if self._has_function('is_in'):
                forms['is_in'] = 0.08
        else:
            # For non-int, default to equality if available
            if self._has_function('=='):
                forms['equality'] = 1.0

        # If no forms available, fall back to a trivial predicate
        if not forms:
            forms['trivial'] = 1.0

        # NOTE: Guards NOT applied here - caller applies after noise
        return [(name, weight) for name, weight in forms.items()]

    def _get_available_transform_forms(
        self,
        elem_type: TypeType,
        ret_type: TypeType,
        use_index: bool = False,
        substitutions: SubstitutionTable | None = None,
        allow_identity: bool = False
    ) -> list[tuple[str, float]]:
        """
        Get available transform forms based on grammar functions.

        Returns raw weights WITHOUT guards applied. Guards are applied after noise
        in the calling function.

        Args:
            elem_type: Type of input element
            ret_type: Expected return type
            use_index: Whether index is available
            substitutions: Type substitutions
            allow_identity: If False, identity transforms are blocked (for map)

        Returns:
            List of (form_name, weight) tuples (without guards applied)
        """
        if substitutions is None:
            substitutions = SubstitutionTable()

        actual_elem = substitute_type_vars(elem_type, substitutions)
        actual_ret = substitute_type_vars(ret_type, substitutions)

        forms = {}

        # identity: always available if types match
        if matchable(actual_elem, actual_ret, substitutions.copy(), update=False):
            forms['identity'] = 0.10

        if actual_elem == int and actual_ret == int:
            # arithmetic: needs at least one arithmetic operator
            # Reduced from 0.40 to 0.25 to reduce over-representation
            if any(self._has_function(op) for op in ['+', '-', '*']):
                forms['arithmetic'] = 0.25

            # division: needs / - common in canonical (e.g., (/ y 10) for digit extraction)
            if self._has_function('/'):
                forms['division'] = 0.12

            # modulo: needs %
            # Reduced from 0.15 to 0.10 to reduce % over-representation
            if self._has_function('%'):
                forms['modulo'] = 0.10

            # with_index: needs arithmetic ops
            if use_index and any(self._has_function(op) for op in ['+', '-', '*']):
                forms['with_index'] = 0.15

        if actual_ret == int:
            # conditional: available if we have predicates
            # Increased from 0.15 to 0.20 for variety
            pred_forms = self._get_available_predicate_forms(actual_elem)
            if pred_forms:
                forms['conditional'] = 0.20

        if get_base_type(actual_ret) == list:
            # singleton: needs singleton function
            if self._has_function('singleton'):
                forms['singleton'] = 0.05

        # If no forms available, default to identity
        if not forms:
            forms['identity'] = 1.0

        # NOTE: Guards NOT applied here - caller applies after noise
        return [(name, weight) for name, weight in forms.items()]

    def _get_available_key_forms(self, elem_type: TypeType) -> list[tuple[str, float]]:
        """
        Get available key function forms based on grammar functions.

        Returns:
            List of (form_name, weight) tuples
        """
        forms = [('identity', 0.30)]  # Always available

        if elem_type == int:
            # negate: needs -
            if self._has_function('-'):
                forms.append(('negate', 0.20))

            # modulo: needs %
            if self._has_function('%'):
                forms.append(('modulo', 0.30))

            # arithmetic: needs at least one op
            if any(self._has_function(op) for op in ['+', '-', '*']):
                forms.append(('arithmetic', 0.20))

        return forms

    def _get_available_comparison_ops(self) -> list[str]:
        """Get available comparison operators."""
        ops = ['<', '>', '==']
        return [op for op in ops if self._has_function(op)]

    def _get_available_arithmetic_ops(self) -> list[str]:
        """Get available arithmetic operators (excluding division to avoid errors)."""
        ops = ['+', '-', '*']
        return [op for op in ops if self._has_function(op)]

    # ========================================================================
    # Guard Helpers
    # ========================================================================

    def _generate_safe_list_arg(
        self,
        fn_name: str,
        input_var: str,
        input_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate a list argument for fn_name, ensuring it's not a blocked pattern.

        This is used for functions like first, last, length, reverse, unique, etc.
        where certain argument patterns are trivial.
        """
        # First try: just use the input variable (always safe)
        input_node = VariableNode(input_var)
        if not self.guard.should_block_strategy(fn_name, 0, StrategyType.VARIABLE, []):
            return input_node

        # If even the variable is blocked (shouldn't happen), generate something else
        # Try a simple transformation
        if self._has_function('reverse') and fn_name != 'reverse':
            return ApplicationNode(VariableNode('reverse'), [input_node])
        if self._has_function('unique') and fn_name != 'unique':
            return ApplicationNode(VariableNode('unique'), [input_node])

        # Fallback to input
        return input_node

    def _generate_binary_op_args(
        self,
        op: str,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        left_type: TypeType = int,
        right_type: TypeType = int
    ) -> tuple[ASTNode, ASTNode]:
        """
        Generate arguments for a binary operator, ensuring at least one is non-literal.

        This handles the binary_ops_need_nonliteral guard rule.
        Shuffles generation order to avoid bias.
        """
        # Try to find context variables first
        int_vars = [name for name, t in context.items() if t == int]

        # Shuffle generation order to avoid bias
        # first_pos is which position (0=left, 1=right) we generate first
        first_pos = self.rng.choice([0, 1])
        second_pos = 1 - first_pos

        results: list[Optional[ASTNode]] = [None, None]
        strategies: list[Optional[StrategyType]] = [None, None]

        # Generate first argument
        if int_vars and self.rng.random() < 0.7:
            results[first_pos] = VariableNode(self.rng.choice(int_vars))
            strategies[first_pos] = StrategyType.VARIABLE
        else:
            results[first_pos] = NumberNode(self.rng.randint(0, 99))
            strategies[first_pos] = StrategyType.LITERAL

        # Create context for second arg
        ctx = ApplicationContext.for_function(op, num_args=2, current_pos=second_pos)
        ctx = ctx.with_strategy(first_pos, strategies[first_pos])

        # Check if we should block literal for second arg
        block_literal = self.guard.should_block_with_context(ctx, StrategyType.LITERAL)

        if block_literal:
            # Must use non-literal for second arg
            if int_vars:
                results[second_pos] = VariableNode(self.rng.choice(int_vars))
            else:
                # Generate an arithmetic expression
                if self._get_available_arithmetic_ops():
                    inner_op = self.rng.choice(self._get_available_arithmetic_ops())
                    results[second_pos] = ApplicationNode(
                        VariableNode(inner_op),
                        [NumberNode(self.rng.randint(0, 20)), NumberNode(self.rng.randint(1, 10))]
                    )
                else:
                    # Last resort: use first again if it's a variable
                    if strategies[first_pos] == StrategyType.VARIABLE:
                        results[second_pos] = results[first_pos]
                    else:
                        # Can't avoid all literals, use literal anyway
                        results[second_pos] = NumberNode(self.rng.randint(1, 99))
        else:
            # Can use literal for second arg
            results[second_pos] = NumberNode(self.rng.randint(0, 99))

        return results[0], results[1]

    # ========================================================================
    # Predicate Generation (for filter, count, find)
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
        """
        Generate a predicate expression that returns bool.

        Common patterns:
        - (is_even x), (is_odd x)
        - (< x 50), (> x 10)
        - (== x 0), (== (% x 2) 0)

        Only uses functions that exist in the grammar.
        """
        actual_type = substitute_type_vars(elem_type, substitutions)
        elem_node = VariableNode(elem_var)

        # Get available predicate forms (raw weights, no guards yet)
        available_forms = self._get_available_predicate_forms(actual_type, use_index and idx_var is not None)

        if not available_forms:
            # No predicate functions available - return True literal
            return BooleanNode(True)

        # Extract forms and weights, apply noise first
        forms_dict = {f[0]: f[1] for f in available_forms}
        forms = list(forms_dict.keys())
        weights = self._perturb_weights(list(forms_dict.values()))

        # Apply guards AFTER noise (block trivial predicates)
        perturbed_dict = dict(zip(forms, weights))
        guarded_dict = guard_predicate_weights(perturbed_dict, must_be_meaningful=True)

        # If guard zeroed all forms, fall back to a simple predicate
        if not any(w > 0 for w in guarded_dict.values()):
            if actual_type == int and self._has_function('is_even'):
                guarded_dict['is_even_odd'] = 1.0
            elif actual_type == int and any(self._has_function(op) for op in ['<', '>', '==']):
                guarded_dict['compare_const'] = 1.0
            else:
                # Last resort - return True literal
                return BooleanNode(True)

        # Sample from guarded distribution
        forms = list(guarded_dict.keys())
        weights = list(guarded_dict.values())
        form = self.rng.choices(forms, weights=weights, k=1)[0]

        if form == 'is_even_odd':
            # Choose from available is_even/is_odd
            available_parity = [fn for fn in ['is_even', 'is_odd'] if self._has_function(fn)]
            fn_name = self.rng.choice(available_parity)
            return ApplicationNode(VariableNode(fn_name), [elem_node])

        elif form == 'compare_const':
            available_ops = self._get_available_comparison_ops()
            op = self.rng.choice(available_ops)
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
            combiner = self.rng.choice(available_combiners)
            return ApplicationNode(VariableNode(combiner), [pred1, pred2])

        elif form == 'index_based' and idx_var:
            idx_node = VariableNode(idx_var)
            available_idx_ops = []
            for op in ['<', '>', '==']:
                if self._has_function(op):
                    available_idx_ops.append(op)
            for fn in ['is_even', 'is_odd']:
                if self._has_function(fn):
                    available_idx_ops.append(fn)
            op = self.rng.choice(available_idx_ops)
            if op in ['is_even', 'is_odd']:
                return ApplicationNode(VariableNode(op), [idx_node])
            else:
                const = self.rng.randint(0, 10)
                return ApplicationNode(VariableNode(op), [idx_node, NumberNode(const)])

        elif form == 'is_in' and self._has_function('is_in'):
            # (is_in x val) - check if val is in list x
            # We need to pass the list variable from context
            list_vars = [name for name, t in context.items() if get_base_type(t) == list]
            if list_vars:
                list_var = self.rng.choice(list_vars)
                return ApplicationNode(VariableNode('is_in'), [VariableNode(list_var), elem_node])
            # Fallback: use a constant
            return ApplicationNode(VariableNode('=='), [elem_node, NumberNode(self.rng.randint(0, 99))])

        elif form == 'equality' and self._has_function('=='):
            return ApplicationNode(VariableNode('=='), [elem_node, NumberNode(self.rng.randint(0, 99))])

        else:
            # Trivial fallback - return True
            return BooleanNode(True)

    def _generate_simple_predicate(self, elem_var: str, elem_type: TypeType) -> ASTNode:
        """Generate a simple predicate (used in compound predicates)."""
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
            # Nothing available - return True
            return BooleanNode(True)

        form = self.rng.choice(options)

        if form == 'is_even_odd':
            fn_name = self.rng.choice(available_parity)
            return ApplicationNode(VariableNode(fn_name), [elem_node])
        else:
            op = self.rng.choice(available_compare)
            const = self.rng.randint(0, 99)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

    # ========================================================================
    # Transform Generation (for map)
    # ========================================================================

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
        """
        Generate a transform expression.

        Common patterns:
        - x (identity) - blocked by default for map
        - (+ x 1), (* x 2)
        - (% x 10)

        Only uses functions that exist in the grammar.
        """
        actual_elem = substitute_type_vars(elem_type, substitutions)
        actual_ret = substitute_type_vars(ret_type, substitutions)

        elem_node = VariableNode(elem_var)

        # Get available transform forms (raw weights, no guards yet)
        available_forms = self._get_available_transform_forms(
            elem_type, ret_type, use_index and idx_var is not None, substitutions,
            allow_identity=allow_identity
        )

        # Extract forms and weights, apply noise first
        forms_dict = {f[0]: f[1] for f in available_forms}
        forms = list(forms_dict.keys())
        weights = self._perturb_weights(list(forms_dict.values()))

        # Apply guards AFTER noise
        perturbed_dict = dict(zip(forms, weights))
        guarded_dict = guard_transform_weights(perturbed_dict, allow_identity=allow_identity)

        # If guard zeroed all forms, add a fallback (never re-enable identity)
        if not any(w > 0 for w in guarded_dict.values()):
            if any(self._has_function(op) for op in ['+', '-', '*']):
                guarded_dict['arithmetic'] = 1.0
            elif self._has_function('%'):
                guarded_dict['modulo'] = 1.0
            else:
                guarded_dict['conditional'] = 1.0

        # Sample from guarded distribution
        forms = list(guarded_dict.keys())
        weights = list(guarded_dict.values())
        form = self.rng.choices(forms, weights=weights, k=1)[0]

        if form == 'identity':
            return elem_node

        elif form == 'arithmetic':
            available_ops = self._get_available_arithmetic_ops()
            op = self.rng.choice(available_ops)
            const = self.rng.randint(1, 10)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

        elif form == 'modulo':
            divisor = self.rng.choice([2, 3, 5, 10, 100])
            return ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])

        elif form == 'division':
            # Division patterns like (/ y 10) for digit extraction
            divisor = self.rng.choice([2, 3, 5, 10, 100])
            return ApplicationNode(VariableNode('/'), [elem_node, NumberNode(divisor)])

        elif form == 'with_index' and idx_var:
            idx_node = VariableNode(idx_var)
            available_ops = self._get_available_arithmetic_ops()
            op = self.rng.choice(available_ops)
            return ApplicationNode(VariableNode(op), [elem_node, idx_node])

        elif form == 'conditional':
            # Generate a non-trivial predicate (guard blocks boolean literals for if conditions)
            if actual_elem == int:
                pred = self._generate_simple_predicate(elem_var, actual_elem)
            else:
                # For non-int types, generate a comparison instead of literal True
                pred = ApplicationNode(VariableNode('=='), [elem_node, elem_node])
            then_val = NumberNode(self.rng.randint(0, 10))
            else_val = NumberNode(self.rng.randint(0, 10))
            return IfNode(pred, then_val, else_val)

        elif form == 'singleton':
            return ApplicationNode(VariableNode('singleton'), [elem_node])

        return elem_node

    # ========================================================================
    # Key Function Generation (for sort, group)
    # ========================================================================

    def _generate_key_function_body(
        self,
        elem_var: str,
        elem_type: TypeType,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a key function body for sort/group. Only uses available functions."""
        actual_type = substitute_type_vars(elem_type, substitutions)
        elem_node = VariableNode(elem_var)

        if actual_type != int:
            return elem_node

        # Get available key forms (raw weights, no guards yet)
        available_forms = self._get_available_key_forms(actual_type)

        # Extract forms and weights, apply noise first
        forms_dict = {f[0]: f[1] for f in available_forms}
        forms = list(forms_dict.keys())
        weights = self._perturb_weights(list(forms_dict.values()))

        # Apply guards AFTER noise (for sort, identity is usually allowed)
        from .utils.guard import guard_key_weights
        perturbed_dict = dict(zip(forms, weights))
        guarded_dict = guard_key_weights(perturbed_dict, allow_identity=True)

        # Sample from guarded distribution
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
            available_ops = self._get_available_arithmetic_ops()
            op = self.rng.choice(available_ops)
            const = self.rng.randint(1, 10)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

        return elem_node

    # ========================================================================
    # Main Generation
    # ========================================================================

    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None
    ) -> ASTNode:
        """
        Generate a meaningful program of the given type.

        For list[int] -> list[int], uses template-based generation.
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
        """Generate a function using templates."""
        target = substitute_type_vars(target_type, substitutions)
        type_args = get_args(target)

        if len(type_args) != 2:
            raise ValueError(f"Callable must have 2 type args: {target}")

        param_types_list, ret_type = type_args
        if not isinstance(param_types_list, list):
            raise ValueError(f"Expected param list: {param_types_list}")

        new_context = context.copy()
        param_names = []
        for param_type in param_types_list:
            param_name = self._fresh_var_name()
            param_names.append(param_name)
            new_context[param_name] = param_type

        body = self._generate_body_with_template(
            param_names, param_types_list, ret_type,
            depth - 1, new_context, substitutions
        )

        # Use multi-parameter lambda instead of nested lambdas
        return LambdaNode(param_names, body)

    def _generate_body_with_template(
        self,
        param_names: list[str],
        param_types: list[TypeType],
        ret_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate function body using templates."""

        actual_ret = substitute_type_vars(ret_type, substitutions)
        ret_base = get_base_type(actual_ret)

        if (len(param_names) == 1 and ret_base == list and len(param_types) == 1):
            param_type = substitute_type_vars(param_types[0], substitutions)

            if get_base_type(param_type) == list:
                return self._generate_list_to_list_body(
                    param_names[0], param_type, actual_ret,
                    depth, context, substitutions
                )

        if param_names:
            param_type = substitute_type_vars(param_types[0], substitutions)
            if matchable(param_type, actual_ret, substitutions.copy(), update=False):
                return VariableNode(param_names[0])

        return self._generate_expression_of_type(actual_ret, depth, context, substitutions)

    def _generate_list_to_list_body(
        self,
        input_var: str,
        input_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a list -> list transformation using templates.

        Only uses templates whose required functions exist in the grammar.
        """
        input_node = VariableNode(input_var)

        # Get available templates based on grammar functions
        available_templates = self._get_available_templates()

        if not available_templates:
            # Fallback to identity if nothing available
            return input_node

        # Adjust weights for shallow depth
        if depth <= 1:
            # Reduce composition and conditional weights at low depth
            adjusted = {}
            for t, w in available_templates.items():
                if t in ['composition', 'conditional', 'fold_to_list']:
                    adjusted[t] = 0.0
                elif t == 'simple_op':
                    adjusted[t] = w * 2.0
                else:
                    adjusted[t] = w
            available_templates = adjusted

        # Normalize weights
        available_templates = self._renormalize_weights(available_templates)

        templates = list(available_templates.keys())
        weights = self._perturb_weights(list(available_templates.values()))

        template = self.rng.choices(templates, weights=weights, k=1)[0]

        input_args = get_args(input_type)
        elem_type = input_args[0] if input_args else int

        if template == 'identity':
            return input_node

        elif template == 'map':
            return self._generate_map_template(
                input_var, elem_type, output_type, depth, context, substitutions
            )

        elif template == 'filter':
            return self._generate_filter_template(
                input_var, elem_type, depth, context, substitutions
            )

        elif template == 'sort':
            return self._generate_sort_template(
                input_var, elem_type, depth, context, substitutions
            )

        elif template == 'fold':
            return self._generate_fold_template(
                input_var, elem_type, output_type, depth, context, substitutions
            )

        elif template == 'composition':
            return self._generate_composition(
                input_var, elem_type, output_type, depth, context, substitutions
            )

        elif template == 'simple_op':
            return self._generate_simple_list_op(input_var, input_type, output_type, substitutions)

        elif template == 'conditional':
            return self._generate_conditional_list(
                input_var, input_type, output_type, depth, context, substitutions
            )

        elif template == 'list_access':
            return self._generate_list_access_template(
                input_var, elem_type, depth, context, substitutions
            )

        elif template == 'list_construct':
            return self._generate_list_construct_template(
                input_var, elem_type, output_type, depth, context, substitutions
            )

        elif template == 'aggregation':
            return self._generate_aggregation_template(
                input_var, elem_type, depth, context, substitutions
            )

        elif template == 'structural':
            return self._generate_structural_template(
                input_var, elem_type, depth, context, substitutions
            )

        elif template == 'advanced_hof':
            return self._generate_advanced_hof_template(
                input_var, elem_type, output_type, depth, context, substitutions
            )

        return input_node

    def _generate_map_template(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate (map (λ y transform) x) or (mapi ...). Uses only available functions."""
        input_node = VariableNode(input_var)

        # Determine which map function to use based on availability
        has_map = self._has_function('map')
        has_mapi = self._has_function('mapi')

        if not has_map and not has_mapi:
            # No map function available - fallback to identity
            return input_node

        # Decide whether to use index
        use_index = has_mapi and self.rng.random() < 0.2
        fn_name = 'mapi' if use_index else 'map'

        # Make sure chosen function exists
        if fn_name == 'mapi' and not has_mapi:
            fn_name = 'map'
            use_index = False
        elif fn_name == 'map' and not has_map:
            fn_name = 'mapi'
            use_index = True

        output_args = get_args(output_type)
        output_elem_type = output_args[0] if output_args else int

        elem_var = self._fresh_var_name()
        idx_var = self._fresh_var_name() if use_index else None

        transform_body = self._generate_transform(
            elem_var, elem_type, output_elem_type,
            context, substitutions,
            use_index=use_index, idx_var=idx_var
        )

        if use_index:
            lambda_node = LambdaNode([elem_var, idx_var], transform_body)
        else:
            lambda_node = LambdaNode(elem_var, transform_body)

        return ApplicationNode(
            VariableNode(fn_name),
            [lambda_node, VariableNode(input_var)]
        )

    def _generate_filter_template(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate (filter (λ y predicate) x) or (filteri ...). Uses only available functions."""
        input_node = VariableNode(input_var)

        # Determine which filter function to use based on availability
        has_filter = self._has_function('filter')
        has_filteri = self._has_function('filteri')

        if not has_filter and not has_filteri:
            # No filter function available - fallback to identity
            return input_node

        # Decide whether to use index
        use_index = has_filteri and self.rng.random() < 0.15
        fn_name = 'filteri' if use_index else 'filter'

        # Make sure chosen function exists
        if fn_name == 'filteri' and not has_filteri:
            fn_name = 'filter'
            use_index = False
        elif fn_name == 'filter' and not has_filter:
            fn_name = 'filteri'
            use_index = True

        if use_index:
            idx_var = self._fresh_var_name()
            elem_var = self._fresh_var_name()
        else:
            elem_var = self._fresh_var_name()
            idx_var = None

        pred_body = self._generate_predicate(
            elem_var, elem_type, context, substitutions,
            use_index=use_index, idx_var=idx_var
        )

        if use_index:
            lambda_node = LambdaNode([idx_var, elem_var], pred_body)
        else:
            lambda_node = LambdaNode(elem_var, pred_body)

        return ApplicationNode(
            VariableNode(fn_name),
            [lambda_node, VariableNode(input_var)]
        )

    def _generate_sort_template(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate (sort (λ y key) x). Uses only available functions."""
        input_node = VariableNode(input_var)

        if not self._has_function('sort'):
            # No sort function available - fallback to identity
            return input_node

        elem_var = self._fresh_var_name()

        key_body = self._generate_key_function_body(
            elem_var, elem_type, context, substitutions
        )

        lambda_node = LambdaNode(elem_var, key_body)

        return ApplicationNode(
            VariableNode('sort'),
            [lambda_node, VariableNode(input_var)]
        )

    # ========================================================================
    # NEW Template Generation Methods (to match canonical distribution)
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
        Generate list access patterns like:
        - (singleton (first x))
        - (singleton (last x))
        - (singleton (nth n x))
        - (cons (first x) (singleton (last x)))

        These are common in canonical programs (~18% of usage).
        """
        input_node = VariableNode(input_var)

        # Get available accessors
        accessors = []
        if self._has_function('first'):
            accessors.append('first')
        if self._has_function('last'):
            accessors.append('last')
        if self._has_function('second'):
            accessors.append('second')
        if self._has_function('third'):
            accessors.append('third')
        if self._has_function('nth'):
            accessors.append('nth')

        if not accessors:
            return input_node

        # Choose pattern based on available functions
        patterns = []
        has_singleton = self._has_function('singleton')
        has_cons = self._has_function('cons')

        if has_singleton:
            patterns.extend(['singleton_access'] * 5)  # High weight for simple singleton patterns
        if has_cons and has_singleton and len(accessors) >= 2:
            patterns.extend(['cons_access'] * 2)  # (cons (first x) (singleton (last x)))
        if has_cons:
            patterns.extend(['cons_to_list'] * 2)  # (cons (first x) x)
        if self._has_function('take') and 'first' in accessors:
            patterns.append('take_first')  # (take (first x) (drop 1 x))

        if not patterns:
            # Fallback to simple drop/take
            if self._has_function('take'):
                n = self.rng.randint(1, 3)
                return ApplicationNode(VariableNode('take'), [NumberNode(n), input_node])
            return input_node

        pattern = self.rng.choice(patterns)

        if pattern == 'singleton_access':
            accessor = self.rng.choice(accessors)
            if accessor == 'nth':
                n = self.rng.randint(0, 5)
                access_expr = ApplicationNode(VariableNode('nth'), [NumberNode(n), input_node])
            else:
                access_expr = ApplicationNode(VariableNode(accessor), [input_node])
            return ApplicationNode(VariableNode('singleton'), [access_expr])

        elif pattern == 'cons_access':
            # (cons (first x) (singleton (last x)))
            accessor1 = self.rng.choice([a for a in accessors if a != 'nth'])
            accessor2 = self.rng.choice([a for a in accessors if a != accessor1 and a != 'nth'])
            if not accessor2:
                accessor2 = accessor1
            access1 = ApplicationNode(VariableNode(accessor1), [input_node])
            access2 = ApplicationNode(VariableNode(accessor2), [input_node])
            singleton_expr = ApplicationNode(VariableNode('singleton'), [access2])
            return ApplicationNode(VariableNode('cons'), [access1, singleton_expr])

        elif pattern == 'cons_to_list':
            # (cons (first x) x) or (cons (last x) x)
            accessor = self.rng.choice([a for a in accessors if a != 'nth']) if any(a != 'nth' for a in accessors) else 'first'
            if not self._has_function(accessor):
                accessor = accessors[0]
            if accessor == 'nth':
                access_expr = ApplicationNode(VariableNode('nth'), [NumberNode(0), input_node])
            else:
                access_expr = ApplicationNode(VariableNode(accessor), [input_node])
            return ApplicationNode(VariableNode('cons'), [access_expr, input_node])

        elif pattern == 'take_first':
            # (take (first x) (drop 1 x))
            first_expr = ApplicationNode(VariableNode('first'), [input_node])
            drop_expr = ApplicationNode(VariableNode('drop'), [NumberNode(1), input_node])
            return ApplicationNode(VariableNode('take'), [first_expr, drop_expr])

        return input_node

    def _generate_list_construct_template(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate list construction patterns like:
        - (cons val x)
        - (append x val)
        - (concat x (singleton val))
        - (repeat (first x) n)
        - (flatten (map ... x))

        These are common in canonical programs (~19% of usage).
        """
        input_node = VariableNode(input_var)

        patterns = []

        # Cons patterns: (cons val x)
        if self._has_function('cons'):
            patterns.extend(['cons_val'] * 4)
            if self._has_function('first') or self._has_function('last'):
                patterns.extend(['cons_element'] * 2)  # (cons (first x) x)

        # Append patterns: (append x val)
        if self._has_function('append'):
            patterns.extend(['append_val'] * 3)
            if self._has_function('first') or self._has_function('last'):
                patterns.extend(['append_element'] * 2)  # (append x (first x))

        # Concat patterns: (concat x y)
        if self._has_function('concat'):
            patterns.extend(['concat_self'] * 2)  # (concat x x)
            if self._has_function('reverse'):
                patterns.append('concat_reverse')  # (concat (reverse x) x)
            if self._has_function('drop') or self._has_function('take'):
                patterns.append('concat_parts')  # (concat (drop n x) (take n x))

        # Repeat patterns: (repeat (first x) n)
        if self._has_function('repeat'):
            if self._has_function('first') or self._has_function('last'):
                patterns.extend(['repeat_element'] * 2)
            if self._has_function('max') or self._has_function('min'):
                patterns.append('repeat_agg')  # (repeat (max x) (min x))

        # Flatten patterns: (flatten (map singleton x)) -- simpler flatten usage
        if self._has_function('flatten') and self._has_function('map'):
            patterns.append('flatten_map')

        if not patterns:
            return input_node

        pattern = self.rng.choice(patterns)

        if pattern == 'cons_val':
            val = NumberNode(self.rng.randint(0, 99))
            return ApplicationNode(VariableNode('cons'), [val, input_node])

        elif pattern == 'cons_element':
            accessor = 'first' if self._has_function('first') else 'last'
            access_expr = ApplicationNode(VariableNode(accessor), [input_node])
            return ApplicationNode(VariableNode('cons'), [access_expr, input_node])

        elif pattern == 'append_val':
            val = NumberNode(self.rng.randint(0, 99))
            return ApplicationNode(VariableNode('append'), [input_node, val])

        elif pattern == 'append_element':
            accessor = 'first' if self._has_function('first') else 'last'
            access_expr = ApplicationNode(VariableNode(accessor), [input_node])
            return ApplicationNode(VariableNode('append'), [input_node, access_expr])

        elif pattern == 'concat_self':
            return ApplicationNode(VariableNode('concat'), [input_node, input_node])

        elif pattern == 'concat_reverse':
            rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
            return ApplicationNode(VariableNode('concat'), [rev_expr, input_node])

        elif pattern == 'concat_parts':
            n = self.rng.randint(1, 4)
            if self._has_function('drop') and self._has_function('take'):
                drop_expr = ApplicationNode(VariableNode('drop'), [NumberNode(n), input_node])
                take_expr = ApplicationNode(VariableNode('take'), [NumberNode(n), input_node])
                return ApplicationNode(VariableNode('concat'), [drop_expr, take_expr])
            return ApplicationNode(VariableNode('concat'), [input_node, input_node])

        elif pattern == 'repeat_element':
            accessor = 'first' if self._has_function('first') else 'last'
            access_expr = ApplicationNode(VariableNode(accessor), [input_node])
            n = self.rng.randint(2, 10)
            return ApplicationNode(VariableNode('repeat'), [access_expr, NumberNode(n)])

        elif pattern == 'repeat_agg':
            agg1 = 'max' if self._has_function('max') else 'min'
            agg2 = 'min' if self._has_function('min') else 'max'
            agg1_expr = ApplicationNode(VariableNode(agg1), [input_node])
            agg2_expr = ApplicationNode(VariableNode(agg2), [input_node])
            return ApplicationNode(VariableNode('repeat'), [agg1_expr, agg2_expr])

        elif pattern == 'flatten_map':
            # (flatten (map (λ y (cons y (singleton y))) x))
            elem_var = self._fresh_var_name()
            elem_node = VariableNode(elem_var)
            if self._has_function('singleton') and self._has_function('cons'):
                inner = ApplicationNode(VariableNode('singleton'), [elem_node])
                body = ApplicationNode(VariableNode('cons'), [elem_node, inner])
            elif self._has_function('singleton'):
                body = ApplicationNode(VariableNode('singleton'), [elem_node])
            else:
                body = elem_node
            lambda_node = LambdaNode(elem_var, body)
            map_expr = ApplicationNode(VariableNode('map'), [lambda_node, input_node])
            return ApplicationNode(VariableNode('flatten'), [map_expr])

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
        Generate aggregation patterns like:
        - (singleton (max x))
        - (singleton (min x))
        - (singleton (sum x))
        - (singleton (product x))
        - (repeat (max x) (min x))
        - (range (min x) 1 (max x))

        These are common in canonical programs (~7% of usage).
        """
        input_node = VariableNode(input_var)

        # Get available aggregators
        aggregators = []
        if self._has_function('max'):
            aggregators.append('max')
        if self._has_function('min'):
            aggregators.append('min')
        if self._has_function('sum'):
            aggregators.append('sum')
        if self._has_function('product'):
            aggregators.append('product')
        if self._has_function('length'):
            aggregators.append('length')

        if not aggregators:
            return input_node

        patterns = []
        has_singleton = self._has_function('singleton')
        has_repeat = self._has_function('repeat')
        has_range = self._has_function('range')

        if has_singleton:
            patterns.extend(['singleton_agg'] * 5)  # Most common
        if has_repeat and 'max' in aggregators and 'min' in aggregators:
            patterns.extend(['repeat_agg'] * 2)  # (repeat (max x) (min x))
        if has_range and 'max' in aggregators and 'min' in aggregators:
            patterns.append('range_agg')  # (range (min x) 1 (max x))
        if has_singleton and 'length' in aggregators and self._has_function('unique'):
            patterns.append('unique_length')  # (singleton (length (unique x)))

        if not patterns:
            return input_node

        pattern = self.rng.choice(patterns)

        if pattern == 'singleton_agg':
            agg = self.rng.choice(aggregators)
            agg_expr = ApplicationNode(VariableNode(agg), [input_node])
            return ApplicationNode(VariableNode('singleton'), [agg_expr])

        elif pattern == 'repeat_agg':
            # (repeat (max x) (min x))
            max_expr = ApplicationNode(VariableNode('max'), [input_node])
            min_expr = ApplicationNode(VariableNode('min'), [input_node])
            return ApplicationNode(VariableNode('repeat'), [max_expr, min_expr])

        elif pattern == 'range_agg':
            # (range (min x) (max x) 1) or (range (min x) (max x) 2)
            min_expr = ApplicationNode(VariableNode('min'), [input_node])
            max_expr = ApplicationNode(VariableNode('max'), [input_node])
            step = NumberNode(self.rng.choice([1, 2]))
            return ApplicationNode(VariableNode('range'), [min_expr, max_expr, step])

        elif pattern == 'unique_length':
            # (singleton (length (unique x)))
            unique_expr = ApplicationNode(VariableNode('unique'), [input_node])
            length_expr = ApplicationNode(VariableNode('length'), [unique_expr])
            return ApplicationNode(VariableNode('singleton'), [length_expr])

        return input_node

    def _generate_structural_template(
        self,
        input_var: str,
        elem_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate structural manipulation patterns like:
        - (slice i j x)
        - (cut_idx i x)
        - (swap i j x)
        - (insert val i x)
        - (replace i val x)

        These are common in canonical programs (~8% of usage).
        """
        input_node = VariableNode(input_var)

        patterns = []

        # Slice patterns
        if self._has_function('slice'):
            patterns.extend(['slice_const'] * 2)
            if self._has_function('first') and self._has_function('second'):
                patterns.append('slice_dynamic')  # (slice (first x) (second x) (drop 2 x))

        # Swap patterns
        if self._has_function('swap'):
            patterns.extend(['swap_const'] * 2)

        # Cut patterns
        if self._has_function('cut_idx'):
            patterns.extend(['cut_idx_const'] * 2)
            if self._has_function('first'):
                patterns.append('cut_idx_dynamic')  # (cut_idx (first x) (drop 1 x))

        if self._has_function('cut_slice'):
            patterns.extend(['cut_slice_const'] * 2)
            if self._has_function('first') and self._has_function('second'):
                patterns.append('cut_slice_dynamic')  # (cut_slice (first x) (second x) x)

        if self._has_function('cut_val'):
            patterns.append('cut_val')  # (cut_val n x) or (cut_val (max x) x)

        if self._has_function('cut_vals'):
            patterns.append('cut_vals')  # (cut_vals n x)

        # Insert patterns
        if self._has_function('insert'):
            patterns.extend(['insert_const'] * 2)

        # Replace patterns
        if self._has_function('replace'):
            patterns.extend(['replace_const'] * 2)
            if self._has_function('first') or self._has_function('last'):
                patterns.append('replace_dynamic')  # (replace i (last x) x)

        # Splice patterns
        if self._has_function('splice') and self._has_function('singleton'):
            patterns.append('splice_const')

        if not patterns:
            return input_node

        pattern = self.rng.choice(patterns)

        if pattern == 'slice_const':
            i = self.rng.randint(0, 3)
            j = i + self.rng.randint(1, 4)
            return ApplicationNode(VariableNode('slice'), [NumberNode(i), NumberNode(j), input_node])

        elif pattern == 'slice_dynamic':
            first_expr = ApplicationNode(VariableNode('first'), [input_node])
            second_expr = ApplicationNode(VariableNode('second'), [input_node])
            drop_expr = ApplicationNode(VariableNode('drop'), [NumberNode(2), input_node])
            return ApplicationNode(VariableNode('slice'), [first_expr, second_expr, drop_expr])

        elif pattern == 'swap_const':
            i = self.rng.randint(0, 3)
            j = self.rng.randint(i + 1, 5)
            return ApplicationNode(VariableNode('swap'), [NumberNode(i), NumberNode(j), input_node])

        elif pattern == 'cut_idx_const':
            i = self.rng.randint(1, 5)
            return ApplicationNode(VariableNode('cut_idx'), [NumberNode(i), input_node])

        elif pattern == 'cut_idx_dynamic':
            first_expr = ApplicationNode(VariableNode('first'), [input_node])
            drop_expr = ApplicationNode(VariableNode('drop'), [NumberNode(1), input_node])
            return ApplicationNode(VariableNode('cut_idx'), [first_expr, drop_expr])

        elif pattern == 'cut_slice_const':
            i = self.rng.randint(0, 4)
            j = i + self.rng.randint(1, 4)
            return ApplicationNode(VariableNode('cut_slice'), [NumberNode(i), NumberNode(j), input_node])

        elif pattern == 'cut_slice_dynamic':
            first_expr = ApplicationNode(VariableNode('first'), [input_node])
            second_expr = ApplicationNode(VariableNode('second'), [input_node])
            return ApplicationNode(VariableNode('cut_slice'), [first_expr, second_expr, input_node])

        elif pattern == 'cut_val':
            if self._has_function('max') and self.rng.random() < 0.5:
                val = ApplicationNode(VariableNode('max'), [input_node])
            else:
                val = NumberNode(self.rng.randint(0, 10))
            return ApplicationNode(VariableNode('cut_val'), [val, input_node])

        elif pattern == 'cut_vals':
            if self._has_function('first') and self.rng.random() < 0.5:
                val = ApplicationNode(VariableNode('first'), [input_node])
            else:
                val = NumberNode(self.rng.randint(0, 10))
            return ApplicationNode(VariableNode('cut_vals'), [val, input_node])

        elif pattern == 'insert_const':
            val = NumberNode(self.rng.randint(0, 99))
            i = self.rng.randint(0, 5)
            return ApplicationNode(VariableNode('insert'), [val, NumberNode(i), input_node])

        elif pattern == 'replace_const':
            i = self.rng.randint(0, 5)
            val = NumberNode(self.rng.randint(0, 99))
            return ApplicationNode(VariableNode('replace'), [NumberNode(i), val, input_node])

        elif pattern == 'replace_dynamic':
            i = self.rng.randint(0, 5)
            accessor = 'last' if self._has_function('last') else 'first'
            val = ApplicationNode(VariableNode(accessor), [input_node])
            return ApplicationNode(VariableNode('replace'), [NumberNode(i), val, input_node])

        elif pattern == 'splice_const':
            # (splice (singleton val) i x)
            val = NumberNode(self.rng.randint(0, 99))
            singleton_expr = ApplicationNode(VariableNode('singleton'), [val])
            i = self.rng.randint(0, 5)
            return ApplicationNode(VariableNode('splice'), [singleton_expr, NumberNode(i), input_node])

        return input_node

    def _generate_fold_template(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate fold patterns like:
        - (fold (λ (y z) (append y z)) [] x) - simple fold
        - (fold (λ (y z) (append y (+ (last y) z))) (singleton 0) x) - cumulative sum
        - (fold (λ (y z) (if (> z (last y)) (append y z) y)) (take 1 x) (drop 1 x)) - filter fold
        - (fold (λ (y z) (cons z (reverse y))) [] x) - reverse via fold

        These are present in canonical programs (~2% of usage).
        """
        input_node = VariableNode(input_var)

        if not self._has_function('fold'):
            return input_node

        patterns = []

        # Simple fold patterns
        if self._has_function('append'):
            patterns.append('fold_append')  # (fold (λ (y z) (append y (singleton z))) [] x)

        # Cumulative sum via fold
        if self._has_function('append') and self._has_function('last') and self._has_function('+') and self._has_function('singleton'):
            patterns.append('fold_cumsum')  # (fold (λ (y z) (append y (+ (last y) z))) (singleton 0) x)

        # Filter-like fold
        if self._has_function('append') and self._has_function('last') and self._has_function('>') and self._has_function('take') and self._has_function('drop'):
            patterns.append('fold_filter')  # (fold (λ (y z) (if (> z (last y)) (append y z) y)) (take 1 x) (drop 1 x))

        # Reverse via fold
        if self._has_function('cons'):
            patterns.append('fold_reverse')  # (fold (λ (y z) (cons z y)) [] x)

        if not patterns:
            # Fallback to simple operation
            return self._generate_simple_list_op(input_var, list[elem_type], output_type, substitutions)

        pattern = self.rng.choice(patterns)

        acc_var = self._fresh_var_name()
        elem_var = self._fresh_var_name()
        acc_node = VariableNode(acc_var)
        elem_node = VariableNode(elem_var)

        if pattern == 'fold_append':
            # (fold (λ (y z) (append y (singleton z))) [] x)
            if self._has_function('singleton'):
                singleton_z = ApplicationNode(VariableNode('singleton'), [elem_node])
                body = ApplicationNode(VariableNode('append'), [acc_node, singleton_z])
            else:
                body = ApplicationNode(VariableNode('append'), [acc_node, elem_node])
            init = ListNode([])
            lambda_node = LambdaNode([acc_var, elem_var], body)
            return ApplicationNode(VariableNode('fold'), [lambda_node, init, input_node])

        elif pattern == 'fold_cumsum':
            # (drop 1 (fold (λ (y z) (append y (+ (last y) z))) (singleton 0) x))
            last_y = ApplicationNode(VariableNode('last'), [acc_node])
            sum_expr = ApplicationNode(VariableNode('+'), [last_y, elem_node])
            body = ApplicationNode(VariableNode('append'), [acc_node, sum_expr])
            init = ApplicationNode(VariableNode('singleton'), [NumberNode(0)])
            lambda_node = LambdaNode([acc_var, elem_var], body)
            fold_expr = ApplicationNode(VariableNode('fold'), [lambda_node, init, input_node])
            # Optionally drop the initial 0
            if self._has_function('drop') and self.rng.random() < 0.5:
                return ApplicationNode(VariableNode('drop'), [NumberNode(1), fold_expr])
            return fold_expr

        elif pattern == 'fold_filter':
            # (fold (λ (y z) (if (> z (last y)) (append y z) y)) (take 1 x) (drop 1 x))
            last_y = ApplicationNode(VariableNode('last'), [acc_node])
            cond = ApplicationNode(VariableNode('>'), [elem_node, last_y])
            then_expr = ApplicationNode(VariableNode('append'), [acc_node, elem_node])
            else_expr = acc_node
            body = IfNode(cond, then_expr, else_expr)
            init = ApplicationNode(VariableNode('take'), [NumberNode(1), input_node])
            rest = ApplicationNode(VariableNode('drop'), [NumberNode(1), input_node])
            lambda_node = LambdaNode([acc_var, elem_var], body)
            return ApplicationNode(VariableNode('fold'), [lambda_node, init, rest])

        elif pattern == 'fold_reverse':
            # (fold (λ (y z) (cons z y)) [] x) - equivalent to reverse
            body = ApplicationNode(VariableNode('cons'), [elem_node, acc_node])
            init = ListNode([])
            lambda_node = LambdaNode([acc_var, elem_var], body)
            return ApplicationNode(VariableNode('fold'), [lambda_node, init, input_node])

        return input_node

    def _generate_advanced_hof_template(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """
        Generate advanced higher-order function patterns like:
        - (singleton (count pred x)) - count elements matching predicate
        - (map (λ y (count (== y) x)) x) - count occurrences of each element
        - (zip x (reverse x)) - zip list with its reverse
        - (flatten (zip x y)) - interleave two lists
        - (find pred x) - find first matching element
        - (map first (group key x)) - get first element of each group

        These are present in canonical programs (~3% of usage).
        """
        input_node = VariableNode(input_var)

        patterns = []

        # Count patterns
        if self._has_function('count'):
            if self._has_function('singleton'):
                patterns.extend(['count_singleton'] * 2)  # (singleton (count pred x))
            if self._has_function('map'):
                patterns.append('count_map')  # (map (λ y (count (== y) x)) x)

        # Zip patterns
        if self._has_function('zip'):
            if self._has_function('reverse'):
                patterns.extend(['zip_reverse'] * 2)  # (zip x (reverse x))
            if self._has_function('flatten'):
                patterns.append('zip_flatten')  # (flatten (zip x (reverse x)))
            if self._has_function('map') and self._has_function('first'):
                patterns.append('zip_map')  # (map first (zip x y))
            if self._has_function('droplast') and self._has_function('drop'):
                patterns.append('zip_adjacent')  # (zip (droplast 1 x) (drop 1 x))

        # Find patterns
        if self._has_function('find'):
            patterns.extend(['find_pred'] * 2)  # (find pred x)

        # Group patterns
        if self._has_function('group'):
            if self._has_function('map') and self._has_function('first'):
                patterns.append('group_first')  # (map first (group key x))
            if self._has_function('map') and self._has_function('length'):
                patterns.append('group_length')  # (map length (group key x))

        if not patterns:
            return input_node

        pattern = self.rng.choice(patterns)

        if pattern == 'count_singleton':
            # (singleton (count pred x))
            elem_var = self._fresh_var_name()
            pred = self._generate_simple_predicate(elem_var, int)
            lambda_pred = LambdaNode(elem_var, pred)
            count_expr = ApplicationNode(VariableNode('count'), [lambda_pred, input_node])
            return ApplicationNode(VariableNode('singleton'), [count_expr])

        elif pattern == 'count_map':
            # (map (λ y (count (== y) x)) x) - count occurrences of each element
            elem_var = self._fresh_var_name()
            inner_var = self._fresh_var_name()
            eq_pred = ApplicationNode(VariableNode('=='), [VariableNode(elem_var), VariableNode(inner_var)])
            inner_lambda = LambdaNode(inner_var, eq_pred)
            count_expr = ApplicationNode(VariableNode('count'), [inner_lambda, input_node])
            outer_lambda = LambdaNode(elem_var, count_expr)
            return ApplicationNode(VariableNode('map'), [outer_lambda, input_node])

        elif pattern == 'zip_reverse':
            # (zip x (reverse x))
            rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
            return ApplicationNode(VariableNode('zip'), [input_node, rev_expr])

        elif pattern == 'zip_flatten':
            # (flatten (zip x (reverse x)))
            rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
            zip_expr = ApplicationNode(VariableNode('zip'), [input_node, rev_expr])
            return ApplicationNode(VariableNode('flatten'), [zip_expr])

        elif pattern == 'zip_map':
            # (map first (zip x (reverse x))) or (map sum (zip x (reverse x)))
            rev_expr = ApplicationNode(VariableNode('reverse'), [input_node])
            zip_expr = ApplicationNode(VariableNode('zip'), [input_node, rev_expr])
            fn = 'first' if self._has_function('first') else 'sum' if self._has_function('sum') else 'first'
            return ApplicationNode(VariableNode('map'), [VariableNode(fn), zip_expr])

        elif pattern == 'zip_adjacent':
            # (zip (droplast 1 x) (drop 1 x)) - zip adjacent pairs
            droplast_expr = ApplicationNode(VariableNode('droplast'), [NumberNode(1), input_node])
            drop_expr = ApplicationNode(VariableNode('drop'), [NumberNode(1), input_node])
            return ApplicationNode(VariableNode('zip'), [droplast_expr, drop_expr])

        elif pattern == 'find_pred':
            # (find pred x)
            elem_var = self._fresh_var_name()
            # Use is_even/is_odd if available, else comparison
            if self._has_function('is_even'):
                pred = VariableNode('is_even')
            elif self._has_function('is_odd'):
                pred = VariableNode('is_odd')
            else:
                inner_pred = self._generate_simple_predicate(elem_var, int)
                pred = LambdaNode(elem_var, inner_pred)
            return ApplicationNode(VariableNode('find'), [pred, input_node])

        elif pattern == 'group_first':
            # (map first (group (λ y x) x)) - get first element of each group
            elem_var = self._fresh_var_name()
            # Simple key function (identity or modulo)
            if self._has_function('%'):
                key_body = ApplicationNode(VariableNode('%'), [VariableNode(elem_var), NumberNode(10)])
            else:
                key_body = VariableNode(elem_var)
            key_lambda = LambdaNode(elem_var, key_body)
            group_expr = ApplicationNode(VariableNode('group'), [key_lambda, input_node])
            return ApplicationNode(VariableNode('map'), [VariableNode('first'), group_expr])

        elif pattern == 'group_length':
            # (map length (group (λ y x) x)) - get size of each group
            elem_var = self._fresh_var_name()
            if self._has_function('%'):
                key_body = ApplicationNode(VariableNode('%'), [VariableNode(elem_var), NumberNode(10)])
            else:
                key_body = VariableNode(elem_var)
            key_lambda = LambdaNode(elem_var, key_body)
            group_expr = ApplicationNode(VariableNode('group'), [key_lambda, input_node])
            return ApplicationNode(VariableNode('map'), [VariableNode('length'), group_expr])

        return input_node

    def _generate_simple_list_op(
        self,
        input_var: str,
        input_type: TypeType,
        output_type: TypeType,
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a simple list operation. Uses only available functions."""
        input_node = VariableNode(input_var)

        # Get available simple ops
        available_ops = self._get_available_simple_ops()

        if not available_ops:
            # No simple ops available - fallback to identity
            return input_node

        op = self.rng.choice(available_ops)

        # Simple ops that take just the list
        simple_ops = ['reverse', 'unique']

        if op in simple_ops:
            return ApplicationNode(VariableNode(op), [input_node])
        else:
            # Operations that take an integer argument
            n = self.rng.randint(1, 5)
            return ApplicationNode(VariableNode(op), [NumberNode(n), input_node])

    def _generate_composition(
        self,
        input_var: str,
        elem_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a composition of operations. Uses only available functions."""
        input_node = VariableNode(input_var)

        if depth < 2:
            return input_node

        # Determine which operations are available
        has_filter = self._has_function('filter') or self._has_function('filteri')
        has_map = self._has_function('map') or self._has_function('mapi')
        available_simple = self._get_available_simple_ops()

        # Build list of available composition patterns
        patterns = []
        if has_filter and has_map:
            patterns.extend(['filter_then_map', 'map_then_filter'])
        if has_filter and available_simple:
            patterns.extend(['filter_then_simple', 'simple_then_filter'])
        if has_map and available_simple:
            patterns.append('map_then_simple')

        if not patterns:
            # No compositions available - fallback to single operation or identity
            if has_filter:
                return self._generate_filter_template(input_var, elem_type, depth - 1, context, substitutions)
            elif has_map:
                return self._generate_map_template(input_var, elem_type, output_type, depth - 1, context, substitutions)
            elif available_simple:
                return self._generate_simple_list_op(input_var, list[elem_type], output_type, substitutions)
            return input_node

        pattern = self.rng.choice(patterns)

        if pattern == 'filter_then_map':
            filtered = self._generate_filter_template(
                input_var, elem_type, depth - 1, context, substitutions
            )
            temp_var = self._fresh_var_name()
            context = context.copy()
            context[temp_var] = list[elem_type]

            mapped = self._generate_map_template(
                temp_var, elem_type, output_type, depth - 1, context, substitutions
            )
            return self._substitute_var_in_expr(mapped, temp_var, filtered)

        elif pattern == 'map_then_filter':
            output_args = get_args(output_type)
            output_elem = output_args[0] if output_args else int

            mapped = self._generate_map_template(
                input_var, elem_type, output_type, depth - 1, context, substitutions
            )
            temp_var = self._fresh_var_name()
            context = context.copy()
            context[temp_var] = output_type

            filtered = self._generate_filter_template(
                temp_var, output_elem, depth - 1, context, substitutions
            )
            return self._substitute_var_in_expr(filtered, temp_var, mapped)

        elif pattern == 'filter_then_simple':
            filtered = self._generate_filter_template(
                input_var, elem_type, depth - 1, context, substitutions
            )
            # Get an available simple op that takes just the list
            simple_unary = [op for op in available_simple if op in ['reverse', 'unique']]
            if simple_unary:
                simple_op = self.rng.choice(simple_unary)
                return ApplicationNode(VariableNode(simple_op), [filtered])
            return filtered

        elif pattern == 'simple_then_filter':
            # Get an available simple op that takes just the list
            simple_unary = [op for op in available_simple if op in ['reverse', 'unique']]
            if simple_unary:
                simple_op = self.rng.choice(simple_unary)
                simple_result = ApplicationNode(VariableNode(simple_op), [VariableNode(input_var)])
            else:
                simple_result = VariableNode(input_var)

            temp_var = self._fresh_var_name()
            context = context.copy()
            context[temp_var] = list[elem_type]

            filtered = self._generate_filter_template(
                temp_var, elem_type, depth - 1, context, substitutions
            )
            return self._substitute_var_in_expr(filtered, temp_var, simple_result)

        else:  # map_then_simple
            mapped = self._generate_map_template(
                input_var, elem_type, output_type, depth - 1, context, substitutions
            )
            # Get an available simple op that takes just the list
            simple_unary = [op for op in available_simple if op in ['reverse', 'unique']]
            if simple_unary:
                simple_op = self.rng.choice(simple_unary)
                return ApplicationNode(VariableNode(simple_op), [mapped])
            return mapped

    def _substitute_var_in_expr(self, expr: ASTNode, var_name: str, replacement: ASTNode) -> ASTNode:
        """Replace a variable with an expression."""
        if isinstance(expr, VariableNode):
            if expr.name == var_name:
                return replacement
            return expr
        elif isinstance(expr, NumberNode) or isinstance(expr, BooleanNode):
            return expr
        elif isinstance(expr, LambdaNode):
            if expr.param == var_name:
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
        return expr

    def _generate_conditional_list(
        self,
        input_var: str,
        input_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate (if condition list_expr1 list_expr2)."""

        input_node = VariableNode(input_var)

        conditions = [
            ApplicationNode(VariableNode('=='), [
                ApplicationNode(VariableNode('length'), [input_node]),
                NumberNode(0)
            ]),
            ApplicationNode(VariableNode('<'), [
                ApplicationNode(VariableNode('length'), [input_node]),
                NumberNode(self.rng.randint(1, 5))
            ]),
        ]
        condition = self.rng.choice(conditions)

        then_expr = self._generate_simple_list_op(input_var, input_type, output_type, substitutions)
        else_expr = self._generate_simple_list_op(input_var, input_type, output_type, substitutions)

        return IfNode(condition, then_expr, else_expr)

    # ========================================================================
    # Helper generation methods
    # ========================================================================

    def _generate_list_expression(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Generate a list expression.

        Args:
            target_type: The list type to generate
            depth: Remaining depth
            context: Variable bindings
            substitutions: Type substitutions
            app_context: If provided, used to check for blocked patterns (e.g., singleton for first/last)
        """
        for var_name, var_type in context.items():
            if matchable(var_type, target_type, substitutions.copy(), update=False):
                return VariableNode(var_name)

        # Check if singleton would be blocked (e.g., for first, last, flatten)
        if app_context is not None:
            singleton_blocked = self.guard.should_block_strategy(
                app_context.fn_name, app_context.arg_pos, 'apply:singleton', []
            )
            if singleton_blocked:
                # Try to find an alternative - use a list variable if available
                for var_name, var_type in context.items():
                    if get_base_type(var_type) == list:
                        return VariableNode(var_name)
                # No list variable, generate empty list (also might be blocked, but safer)
                return ListNode([])

        return ApplicationNode(VariableNode('singleton'), [NumberNode(self.rng.randint(0, 99))])

    def _generate_int_expression(
        self,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Generate an integer expression.

        Args:
            depth: Remaining depth
            context: Variable bindings
            substitutions: Type substitutions
            app_context: If provided, used to apply guards (e.g., block literals for binary ops)
        """
        # Check if we should block literals (e.g., for binary ops when other arg is literal)
        block_literal = self.guard.should_block_with_context(app_context, StrategyType.LITERAL)

        if not block_literal and (depth <= 0 or self.rng.random() < 0.5):
            return NumberNode(self.rng.randint(0, 99))

        # Try to use a context variable if available
        for var_name, var_type in context.items():
            if var_type == int:
                return VariableNode(var_name)

        # Fall back to arithmetic expression with guard
        available_ops = self._get_available_arithmetic_ops()
        if available_ops:
            op = self.rng.choice(available_ops)
            left, right = self._generate_binary_op_args(op, depth, context, substitutions)
            return ApplicationNode(VariableNode(op), [left, right])

        # No arithmetic ops available, use literal
        return NumberNode(self.rng.randint(0, 99))

    def _generate_bool_expression(
        self,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Generate a boolean expression.

        Args:
            depth: Remaining depth
            context: Variable bindings
            substitutions: Type substitutions
            app_context: If provided, used to apply guards (e.g., block literals for if conditions)
        """

        # Check if we should block literals (e.g., for if condition)
        block_literal = self.guard.should_block_with_context(app_context, StrategyType.LITERAL)

        if not block_literal and (depth <= 0 or self.rng.random() < 0.3):
            return BooleanNode(self.rng.choice([True, False]))

        # Generate a comparison expression with guard
        available_ops = self._get_available_comparison_ops()
        if available_ops:
            op = self.rng.choice(available_ops)
            left, right = self._generate_binary_op_args(op, depth, context, substitutions)
            return ApplicationNode(VariableNode(op), [left, right])

        # No comparison ops available, use tautology
        return ApplicationNode(VariableNode('=='), [NumberNode(0), NumberNode(0)])

    def _generate_expression_of_type(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Generate an expression of the given type.

        Args:
            target_type: Type to generate
            depth: Remaining depth
            context: Variable bindings
            substitutions: Type substitutions
            app_context: If provided, used to apply guards
        """
        base = get_base_type(target_type)

        if target_type == int:
            return self._generate_int_expression(depth, context, substitutions, app_context)
        elif target_type == bool:
            return self._generate_bool_expression(depth, context, substitutions, app_context)
        elif base == list:
            return self._generate_list_expression(target_type, depth, context, substitutions, app_context)
        elif base == CallableOrig:
            return self._generate_function(target_type, depth, context, substitutions)

        return NumberNode(0)
