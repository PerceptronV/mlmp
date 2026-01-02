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
from .guard import (
    guard_transform_weights,
    guard_predicate_weights,
    ApplicationContext,
    StrategyGuard,
    get_default_guard,
    is_literal_node,
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

    def __init__(self, seed: int, grammar: Grammar, noise: float = 0.0):
        """
        Initialize the TemplateComposer.

        Args:
            seed: Random seed for reproducibility
            grammar: Grammar defining available functions
            noise: Noise parameter (0 = default weights, higher = more uniform/random)
        """
        super().__init__(seed, grammar)
        self.noise = noise

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
        self.template_weights = {
            'identity': 0.05,
            'map': 0.20,
            'filter': 0.20,
            'sort': 0.10,
            'fold_to_list': 0.05,
            'composition': 0.25,
            'simple_op': 0.10,
            'conditional': 0.05,
        }

        # Define function requirements for each template
        # A template is available only if ALL its required functions exist
        self._template_requirements: dict[str, set[str]] = {
            'identity': set(),  # No functions needed - just return input
            'map': {'map'},     # Needs map (or mapi, handled separately)
            'filter': {'filter'},  # Needs filter (or filteri)
            'sort': {'sort'},
            'fold_to_list': set(),  # Falls back to simple_op
            'composition': set(),   # Checked dynamically based on available templates
            'simple_op': set(),     # Checked dynamically
            'conditional': {'length'},  # Needs length for condition
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
            elif template == 'fold_to_list':
                # Falls back to simple_op, so check that
                if self._get_available_simple_ops():
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
        use_index: bool = False,
        must_be_meaningful: bool = True
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
            if self._has_function('is_even') or self._has_function('is_odd'):
                forms['is_even_odd'] = 0.25

            # compare_const: needs at least one comparison operator
            if any(self._has_function(op) for op in ['<', '>', '==']):
                forms['compare_const'] = 0.30

            # modulo_check: needs % and ==
            if self._has_function('%') and self._has_function('=='):
                forms['modulo_check'] = 0.20

            # compound: needs at least one combiner and a base predicate
            if (self._has_function('and') or self._has_function('or')) and forms:
                forms['compound'] = 0.15

            # index_based: needs comparisons or is_even/is_odd
            if use_index:
                if any(self._has_function(op) for op in ['<', '>', '==', 'is_even', 'is_odd']):
                    forms['index_based'] = 0.10
        else:
            # For non-int, default to equality if available
            if self._has_function('=='):
                forms['equality'] = 1.0

        # If no forms available, fall back to a trivial predicate
        if not forms:
            forms['trivial'] = 1.0

        # Apply guard to block trivial predicates
        forms = guard_predicate_weights(forms, must_be_meaningful=must_be_meaningful)

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

        Args:
            elem_type: Type of input element
            ret_type: Expected return type
            use_index: Whether index is available
            substitutions: Type substitutions
            allow_identity: If False, identity transforms are blocked (for map)

        Returns:
            List of (form_name, weight) tuples
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
            if any(self._has_function(op) for op in ['+', '-', '*']):
                forms['arithmetic'] = 0.40

            # modulo: needs %
            if self._has_function('%'):
                forms['modulo'] = 0.15

            # with_index: needs arithmetic ops
            if use_index and any(self._has_function(op) for op in ['+', '-', '*']):
                forms['with_index'] = 0.15

        if actual_ret == int:
            # conditional: available if we have predicates
            pred_forms = self._get_available_predicate_forms(actual_elem)
            if pred_forms:
                forms['conditional'] = 0.15

        if get_base_type(actual_ret) == list:
            # singleton: needs singleton function
            if self._has_function('singleton'):
                forms['singleton'] = 0.05

        # If no forms available, default to identity
        if not forms:
            forms['identity'] = 1.0

        # Apply guard to block identity transforms for map
        forms = guard_transform_weights(forms, allow_identity=allow_identity)

        # If guard zeroed all forms, add a fallback (never re-enable identity)
        if not any(w > 0 for w in forms.values()):
            if any(self._has_function(op) for op in ['+', '-', '*']):
                forms['arithmetic'] = 1.0
            elif self._has_function('%'):
                forms['modulo'] = 1.0
            else:
                forms['conditional'] = 1.0

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

    def _is_blocked_application(self, fn_name: str, arg: ASTNode) -> bool:
        """
        Check if an application (fn_name arg) would be blocked by guard rules.

        This handles rules like:
        - first_blocks_singleton: (first (singleton x)) is trivial
        - reverse_no_reverse: (reverse (reverse x)) is trivial
        - unique_no_unique: (unique (unique x)) is trivial
        - flatten_no_singleton: (flatten (singleton x)) is trivial
        - last_blocks_singleton: (last (singleton x)) is trivial
        - length_no_empty_list: (length []) is trivial
        - list_reducer_no_empty_literal: (sum []), (product []), etc. are trivial
        """
        guard = get_default_guard()

        # Determine strategy type of the argument
        if is_literal_node(arg):
            strategy_type = StrategyGuard.LITERAL
        elif isinstance(arg, ApplicationNode) and isinstance(arg.function, VariableNode):
            strategy_type = f"apply:{arg.function.name}"
        elif isinstance(arg, VariableNode):
            strategy_type = StrategyGuard.VARIABLE
        else:
            strategy_type = 'unknown'

        # Check if this strategy is blocked for this function
        return guard.should_block_strategy(fn_name, 0, strategy_type, [])

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
        if not self._is_blocked_application(fn_name, input_node):
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
        """
        guard = get_default_guard()

        # Try to find context variables first
        int_vars = [name for name, t in context.items() if t == int]

        # Strategy: generate left first, then right with context
        if int_vars and self.rng.random() < 0.7:
            # Use a variable for left
            left = VariableNode(self.rng.choice(int_vars))
            left_strategy = StrategyGuard.VARIABLE
        else:
            # Use a literal for left
            left = NumberNode(self.rng.randint(0, 99))
            left_strategy = StrategyGuard.LITERAL

        # Create context for right arg
        ctx = ApplicationContext.for_function(op, num_args=2, current_pos=1)
        ctx = ctx.with_strategy(0, left_strategy)

        # Check if we should block literal for right
        block_right_literal = guard.should_block_with_context(ctx, StrategyGuard.LITERAL)

        if block_right_literal:
            # Must use non-literal for right
            if int_vars:
                right = VariableNode(self.rng.choice(int_vars))
            else:
                # Generate an arithmetic expression
                if self._get_available_arithmetic_ops():
                    inner_op = self.rng.choice(self._get_available_arithmetic_ops())
                    right = ApplicationNode(
                        VariableNode(inner_op),
                        [NumberNode(self.rng.randint(0, 20)), NumberNode(self.rng.randint(1, 10))]
                    )
                else:
                    # Last resort: use left again if it's a variable
                    if left_strategy == StrategyGuard.VARIABLE:
                        right = left
                    else:
                        # Can't avoid all literals, use literal anyway
                        right = NumberNode(self.rng.randint(1, 99))
        else:
            # Can use literal for right
            right = NumberNode(self.rng.randint(0, 99))

        return left, right

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

        # Get available predicate forms
        available_forms = self._get_available_predicate_forms(actual_type, use_index and idx_var is not None)

        if not available_forms:
            # No predicate functions available - return True literal
            return BooleanNode(True)

        forms = [f[0] for f in available_forms]
        weights = self._perturb_weights([f[1] for f in available_forms])

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

        # Get available transform forms (identity blocked by guard unless allowed)
        available_forms = self._get_available_transform_forms(
            elem_type, ret_type, use_index and idx_var is not None, substitutions,
            allow_identity=allow_identity
        )

        forms = [f[0] for f in available_forms]
        weights = self._perturb_weights([f[1] for f in available_forms])

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

        # Get available key forms
        available_forms = self._get_available_key_forms(actual_type)

        forms = [f[0] for f in available_forms]
        weights = self._perturb_weights([f[1] for f in available_forms])

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

        for param_name in reversed(param_names):
            body = LambdaNode(param_name, body)

        return body

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

        elif template == 'fold_to_list':
            return self._generate_simple_list_op(input_var, input_type, output_type, substitutions)

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
            lambda_node = LambdaNode(elem_var, LambdaNode(idx_var, transform_body))
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
            lambda_node = LambdaNode(idx_var, LambdaNode(elem_var, pred_body))
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
            guard = get_default_guard()
            singleton_blocked = guard.should_block_strategy(
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
        guard = get_default_guard()

        # Check if we should block literals (e.g., for binary ops when other arg is literal)
        block_literal = guard.should_block_with_context(app_context, StrategyGuard.LITERAL)

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
        guard = get_default_guard()

        # Check if we should block literals (e.g., for if condition)
        block_literal = guard.should_block_with_context(app_context, StrategyGuard.LITERAL)

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
