"""
Hybrid Program Composer

Combines template-based generation with empirically learned distributions.
Uses templates to define high-level program structures (map, filter, sort,
composition, etc.) but weights them according to distributions learned from
gold-standard example programs.

Key insight: Templates provide semantic structure, while empirical distributions
ensure the generated programs match the statistical patterns of real programs.
This produces more semantically meaningful programs with controlled variance.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import Composer
from .guard import (
    guard_transform_weights,
    guard_predicate_weights,
    ApplicationContext,
    StrategyGuard,
    get_default_guard,
)
from ..grammar import Grammar
from ..ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, IfNode, ListNode
)
from ..parser import parse as parse_program
from ..type_utils import (
    get_args,
    get_base_type,
    CallableOrig,
    TypeType,
    SubstitutionTable,
    substitute_type_vars,
    matchable
)


# ============================================================================
# Template and Pattern Types
# ============================================================================

@dataclass
class TemplateDistributions:
    """Holds empirical distributions for template-based generation."""

    # High-level template distribution
    template_counts: Counter  # Template name -> count

    # Predicate pattern distribution (for filter, count, find)
    predicate_counts: Counter  # Pattern name -> count

    # Transform pattern distribution (for map)
    transform_counts: Counter  # Pattern name -> count

    # Key function pattern distribution (for sort, group)
    key_counts: Counter  # Pattern name -> count

    # Simple list operation distribution
    simple_op_counts: Counter  # Operation name -> count

    # Composition pattern distribution
    composition_counts: Counter  # Pattern name -> count

    # Numeric constants distribution
    int_constants: Counter  # Value -> count

    # Comparison operators distribution
    comparison_ops: Counter  # Operator -> count

    # Arithmetic operators distribution
    arithmetic_ops: Counter  # Operator -> count

    # Divisor distribution (for modulo operations)
    divisors: Counter  # Divisor value -> count

    def __init__(self):
        self.template_counts = Counter()
        self.predicate_counts = Counter()
        self.transform_counts = Counter()
        self.key_counts = Counter()
        self.simple_op_counts = Counter()
        self.composition_counts = Counter()
        self.int_constants = Counter()
        self.comparison_ops = Counter()
        self.arithmetic_ops = Counter()
        self.divisors = Counter()

    def normalize_with_noise(self, counts: Counter, noise: float) -> dict:
        """
        Convert counts to probabilities with noise floor.

        Args:
            counts: Counter of observations
            noise: Noise floor to add to all categories

        Returns:
            Dictionary of probabilities
        """
        if not counts:
            return {}

        total = sum(counts.values())
        n_categories = len(counts)

        # Add noise proportionally
        probs = {}
        for key, count in counts.items():
            base_prob = count / total if total > 0 else 0
            # Noise perturbs toward uniform: prob = (1-noise)*empirical + noise*uniform
            probs[key] = (1 - noise) * base_prob + noise / n_categories

        # Renormalize
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}

        return probs

    def sample(self, counts: Counter, noise: float, rng, default=None):
        """Sample from a distribution with noise."""
        if not counts:
            return default

        probs = self.normalize_with_noise(counts, noise)
        keys = list(probs.keys())
        weights = [probs[k] for k in keys]
        return rng.choices(keys, weights=weights, k=1)[0]

    @staticmethod
    def perturb_weights(weights: dict, noise: float) -> dict:
        """
        Perturb hand-tuned weights with noise toward uniform distribution.

        Args:
            weights: Dictionary of base weights (should sum to ~1)
            noise: How much to perturb toward uniform (0 = original, 1 = uniform)

        Returns:
            Perturbed weights dictionary
        """
        n = len(weights)
        if n == 0:
            return {}

        uniform = 1.0 / n
        perturbed = {}
        for key, w in weights.items():
            perturbed[key] = (1 - noise) * w + noise * uniform

        # Renormalize
        total = sum(perturbed.values())
        if total > 0:
            perturbed = {k: v / total for k, v in perturbed.items()}

        return perturbed


# ============================================================================
# AST Pattern Analyzer
# ============================================================================

class TemplateAnalyzer:
    """Analyzes AST nodes to extract template and pattern information."""

    def __init__(self, grammar: Grammar):
        self.grammar = grammar

        # Higher-order functions
        self.ho_map = {'map', 'mapi'}
        self.ho_filter = {'filter', 'filteri'}
        self.ho_fold = {'fold', 'foldi'}
        self.ho_sort = {'sort', 'group'}
        self.ho_count = {'count', 'find'}

        # Simple list operations
        self.simple_ops = {
            'reverse', 'unique', 'take', 'drop', 'takelast', 'droplast',
            'singleton', 'cons', 'append', 'concat', 'flatten'
        }

        # Predicate functions
        self.predicate_funcs = {'is_even', 'is_odd', '<', '>', '==', 'and', 'or', 'not', 'is_in'}

        # Arithmetic functions
        self.arithmetic_funcs = {'+', '-', '*', '/', '%'}

    def analyze_programs(self, programs: list[ASTNode]) -> TemplateDistributions:
        """Analyze a list of programs and extract distributions."""
        dists = TemplateDistributions()

        for prog in programs:
            self._analyze_program(prog, dists)

        return dists

    def _analyze_program(self, node: ASTNode, dists: TemplateDistributions):
        """Analyze a single program."""
        # Programs should be lambdas taking a list
        if not isinstance(node, LambdaNode):
            return

        body = node.body
        input_var = node.param[0] if isinstance(node.param, list) else node.param

        # Detect template type from body
        template = self._detect_template(body, input_var)
        dists.template_counts[template] += 1

        # Extract patterns from body
        self._extract_patterns(body, dists)

    def _detect_template(self, body: ASTNode, input_var: str) -> str:
        """Detect which template category this body represents."""
        if isinstance(body, VariableNode):
            if body.name == input_var:
                return 'identity'
            return 'other'

        if isinstance(body, ApplicationNode):
            func = body.function
            if isinstance(func, VariableNode):
                fn = func.name

                # Map template
                if fn in self.ho_map:
                    return 'map'

                # Filter template
                if fn in self.ho_filter:
                    return 'filter'

                # Sort/group template
                if fn in self.ho_sort:
                    return 'sort'

                # Fold template
                if fn in self.ho_fold:
                    return 'fold'

                # Count/find template
                if fn in self.ho_count:
                    return 'count_find'

                # Simple operations
                if fn in self.simple_ops:
                    return 'simple_op'

                # Check for composition (nested HO functions)
                if self._is_composition(body):
                    return 'composition'

        if isinstance(body, IfNode):
            return 'conditional'

        return 'other'

    def _is_composition(self, node: ASTNode) -> bool:
        """Check if this is a composition of operations."""
        if not isinstance(node, ApplicationNode):
            return False

        func = node.function
        if not isinstance(func, VariableNode):
            return False

        fn = func.name
        ho_funcs = self.ho_map | self.ho_filter | self.ho_sort | self.ho_fold

        if fn in ho_funcs:
            # Check if any argument contains another HO function
            for arg in node.arguments:
                if self._contains_ho_function(arg):
                    return True

        if fn in self.simple_ops:
            # Check if argument is a HO function
            for arg in node.arguments:
                if isinstance(arg, ApplicationNode):
                    if isinstance(arg.function, VariableNode):
                        if arg.function.name in ho_funcs:
                            return True

        return False

    def _contains_ho_function(self, node: ASTNode) -> bool:
        """Check if node contains a higher-order function application."""
        ho_funcs = self.ho_map | self.ho_filter | self.ho_sort | self.ho_fold

        if isinstance(node, ApplicationNode):
            if isinstance(node.function, VariableNode):
                if node.function.name in ho_funcs:
                    return True
            for arg in node.arguments:
                if self._contains_ho_function(arg):
                    return True

        return False

    def _extract_patterns(self, node: ASTNode, dists: TemplateDistributions):
        """Extract patterns from an AST node recursively."""
        if isinstance(node, NumberNode):
            dists.int_constants[node.value] += 1

        elif isinstance(node, ApplicationNode):
            func = node.function
            if isinstance(func, VariableNode):
                fn = func.name

                # Record operator usage
                if fn in {'<', '>', '=='}:
                    dists.comparison_ops[fn] += 1
                if fn in {'+', '-', '*', '/', '%'}:
                    dists.arithmetic_ops[fn] += 1

                # Record divisor for modulo
                if fn == '%' and len(node.arguments) >= 2:
                    if isinstance(node.arguments[1], NumberNode):
                        dists.divisors[node.arguments[1].value] += 1

                # Record simple operations
                if fn in self.simple_ops:
                    dists.simple_op_counts[fn] += 1

                # Analyze predicate patterns in filter/count/find
                if fn in self.ho_filter | self.ho_count:
                    if len(node.arguments) >= 1:
                        pred_arg = node.arguments[0]
                        if isinstance(pred_arg, LambdaNode):
                            pattern = self._classify_predicate(pred_arg.body)
                            dists.predicate_counts[pattern] += 1
                        elif isinstance(pred_arg, VariableNode):
                            if pred_arg.name in {'is_even', 'is_odd'}:
                                dists.predicate_counts['is_even_odd'] += 1

                # Analyze transform patterns in map
                if fn in self.ho_map:
                    if len(node.arguments) >= 1:
                        transform_arg = node.arguments[0]
                        if isinstance(transform_arg, LambdaNode):
                            pattern = self._classify_transform(transform_arg.body)
                            dists.transform_counts[pattern] += 1

                # Analyze key function patterns in sort/group
                if fn in self.ho_sort:
                    if len(node.arguments) >= 1:
                        key_arg = node.arguments[0]
                        if isinstance(key_arg, LambdaNode):
                            pattern = self._classify_key_function(key_arg.body)
                            dists.key_counts[pattern] += 1

            # Recurse into arguments
            for arg in node.arguments:
                self._extract_patterns(arg, dists)

        elif isinstance(node, LambdaNode):
            self._extract_patterns(node.body, dists)

        elif isinstance(node, IfNode):
            self._extract_patterns(node.condition, dists)
            self._extract_patterns(node.then_expr, dists)
            self._extract_patterns(node.else_expr, dists)

    def _classify_predicate(self, body: ASTNode) -> str:
        """Classify a predicate body into a pattern category."""
        if isinstance(body, ApplicationNode):
            func = body.function
            if isinstance(func, VariableNode):
                fn = func.name

                if fn in {'is_even', 'is_odd'}:
                    return 'is_even_odd'

                if fn in {'<', '>', '=='}:
                    # Check if it's a comparison with constant or modulo check
                    if len(body.arguments) >= 1:
                        first_arg = body.arguments[0]
                        if isinstance(first_arg, ApplicationNode):
                            if isinstance(first_arg.function, VariableNode):
                                if first_arg.function.name == '%':
                                    return 'modulo_check'
                    return 'compare_const'

                if fn in {'and', 'or'}:
                    return 'compound'

                if fn == 'not':
                    return 'negation'

        if isinstance(body, VariableNode):
            return 'variable'

        return 'other'

    def _classify_transform(self, body: ASTNode) -> str:
        """Classify a transform body into a pattern category."""
        if isinstance(body, VariableNode):
            return 'identity'

        if isinstance(body, ApplicationNode):
            func = body.function
            if isinstance(func, VariableNode):
                fn = func.name

                if fn in {'+', '-', '*', '/'}:
                    return 'arithmetic'

                if fn == '%':
                    return 'modulo'

                if fn == 'singleton':
                    return 'singleton'

        if isinstance(body, IfNode):
            return 'conditional'

        return 'other'

    def _classify_key_function(self, body: ASTNode) -> str:
        """Classify a key function body into a pattern category."""
        if isinstance(body, VariableNode):
            return 'identity'

        if isinstance(body, ApplicationNode):
            func = body.function
            if isinstance(func, VariableNode):
                fn = func.name

                if fn == '-':
                    # Check for negation pattern: (- 0 x)
                    if len(body.arguments) >= 2:
                        if isinstance(body.arguments[0], NumberNode):
                            if body.arguments[0].value == 0:
                                return 'negate'
                    return 'arithmetic'

                if fn in {'+', '*', '/'}:
                    return 'arithmetic'

                if fn == '%':
                    return 'modulo'

        return 'other'


# ============================================================================
# Hybrid Composer
# ============================================================================

def load_distributions(
    txt_path: Path,
    grammar: Grammar
) -> TemplateDistributions:
    """
    Load and analyze programs from a text file to build distributions.

    Args:
        txt_path: Path to text file with one program per line
        grammar: Grammar to use for analysis

    Returns:
        TemplateDistributions with learned distributions
    """
    analyzer = TemplateAnalyzer(grammar)
    programs = []

    with open(txt_path, 'r') as f:
        for line in f:
            program_str = line.strip()
            if not program_str:
                continue

            try:
                ast = parse_program(program_str)
                programs.append(ast)
            except Exception:
                # Skip unparseable programs
                continue

    return analyzer.analyze_programs(programs)


class HybridComposer(Composer):
    """
    Generates programs using templates weighted by empirical distributions.

    Combines the structured approach of TemplateComposer with the
    statistical patterns learned from gold-standard example programs.
    Uses noise to control exploration vs exploitation tradeoff.

    The composer checks function availability in the grammar and only generates
    code using functions that exist. Templates and patterns that require missing
    functions are filtered out.
    """

    _cached_distributions: dict[tuple[str, int], TemplateDistributions] = {}

    @classmethod
    def get_name(cls) -> str:
        return "hybrid"

    def __init__(
        self,
        seed: int,
        grammar: Grammar,
        functions_path: Optional[Path] = None,
        noise: float = 0.1
    ):
        """
        Initialize the hybrid composer.

        Args:
            seed: Random seed for deterministic generation
            grammar: Grammar containing available functions
            functions_path: Path to training programs (one per line)
            noise: Noise parameter to perturb distributions (0 = pure empirical,
                   1 = uniform). Noise is applied proportionally.
        """
        super().__init__(seed, grammar)
        self.noise = noise

        if functions_path is None:
            functions_path = Path(__file__).parent / 'data' / 'functions.txt'

        cache_key = (str(functions_path.resolve()), id(grammar))
        if cache_key not in HybridComposer._cached_distributions:
            if functions_path.exists():
                HybridComposer._cached_distributions[cache_key] = \
                    load_distributions(functions_path, grammar)
            else:
                HybridComposer._cached_distributions[cache_key] = TemplateDistributions()

        self.dists = HybridComposer._cached_distributions[cache_key]

        # Fallback weights if distributions are empty
        self._default_template_weights = {
            'identity': 0.05,
            'map': 0.20,
            'filter': 0.20,
            'sort': 0.10,
            'fold': 0.05,
            'composition': 0.20,
            'simple_op': 0.15,
            'conditional': 0.05,
        }

        # Cache for available functions
        self._available_templates_cache: dict[str, float] | None = None
        self._available_simple_ops_cache: list[str] | None = None

    # ========================================================================
    # Availability Checking
    # ========================================================================

    def _get_available_simple_ops(self) -> list[str]:
        """Get list of available simple list operations."""
        if self._available_simple_ops_cache is not None:
            return self._available_simple_ops_cache

        simple_ops = ['reverse', 'unique']
        int_arg_ops = ['take', 'drop', 'takelast', 'droplast']

        available = []
        for op in simple_ops:
            if self._has_function(op):
                available.append(op)
        for op in int_arg_ops:
            if self._has_function(op):
                available.append(op)

        self._available_simple_ops_cache = available
        return available

    def _get_available_comparison_ops(self) -> list[str]:
        """Get available comparison operators."""
        ops = ['<', '>', '==']
        return [op for op in ops if self._has_function(op)]

    def _get_available_arithmetic_ops(self) -> list[str]:
        """Get available arithmetic operators (excluding division to avoid errors)."""
        ops = ['+', '-', '*']
        return [op for op in ops if self._has_function(op)]

    def _is_template_available(self, template: str) -> bool:
        """Check if a specific template is available based on grammar functions."""
        if template == 'identity':
            return True
        elif template == 'map':
            return self._has_function('map') or self._has_function('mapi')
        elif template == 'filter':
            return self._has_function('filter') or self._has_function('filteri')
        elif template == 'sort':
            return self._has_function('sort')
        elif template == 'fold':
            return self._has_function('fold') or self._has_function('foldi')
        elif template == 'simple_op':
            return bool(self._get_available_simple_ops())
        elif template == 'composition':
            # Need at least one composable operation
            has_ho = (self._has_function('map') or self._has_function('mapi') or
                     self._has_function('filter') or self._has_function('filteri'))
            has_simple = bool(self._get_available_simple_ops())
            return has_ho or has_simple
        elif template == 'conditional':
            return self._has_function('length') and self._has_function('==')
        return True

    def generate(
        self,
        target_type: TypeType,
        depth: int,
        context: Optional[dict[str, TypeType]] = None,
        substitutions: Optional[SubstitutionTable] = None
    ) -> ASTNode:
        """Generate a program using template-based generation with empirical weights."""
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
        """Generate function body using empirically-weighted templates."""
        actual_ret = substitute_type_vars(ret_type, substitutions)
        ret_base = get_base_type(actual_ret)

        # For list[int] -> list[int] functions, use template-based generation
        if (len(param_names) == 1 and ret_base == list and len(param_types) == 1):
            param_type = substitute_type_vars(param_types[0], substitutions)

            if get_base_type(param_type) == list:
                return self._generate_list_to_list_body(
                    param_names[0], param_type, actual_ret,
                    depth, context, substitutions
                )

        # Fallback: return input if types match
        if param_names:
            param_type = substitute_type_vars(param_types[0], substitutions)
            if matchable(param_type, actual_ret, substitutions.copy(), update=False):
                return VariableNode(param_names[0])

        return self._generate_expression_of_type(actual_ret, depth, context, substitutions)

    def _get_template_weights(self, depth: int) -> dict[str, float]:
        """
        Get template weights using hand-tuned base weights perturbed by noise.

        We use hand-tuned weights (not empirical) because empirical distributions
        from functions.txt are heavily skewed toward simple_op (building blocks
        like cons, singleton dominate). Hand-tuned weights ensure variety.

        Only includes templates whose required functions exist in the grammar.
        """
        # Hand-tuned weights that produce good variety (like TemplateComposer)
        # Tuned to match template composer's output distribution:
        # - High emphasis on map, filter, composition for semantic variety
        # - Low emphasis on identity and simple_op to avoid trivial programs
        base_weights = {
            'identity': 0.005,     # Very rare - just return x
            'map': 0.27,           # Common - transform elements
            'filter': 0.27,        # Common - select elements
            'sort': 0.10,          # Moderate - sort by key
            'fold': 0.01,          # Rare - complex accumulation
            'composition': 0.32,   # Common - chain operations (higher for variety)
            'simple_op': 0.02,     # Rare - single simple op
            'conditional': 0.015,  # Rare - if-then-else on lists
        }

        # Filter by available templates
        available_weights = {
            t: w for t, w in base_weights.items()
            if self._is_template_available(t)
        }

        # Always include identity as fallback
        if not available_weights or 'identity' not in available_weights:
            available_weights['identity'] = 0.005

        # Perturb toward uniform based on noise
        return TemplateDistributions.perturb_weights(available_weights, self.noise)

    def _generate_list_to_list_body(
        self,
        input_var: str,
        input_type: TypeType,
        output_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a list -> list transformation using empirically-weighted templates."""
        weights = self._get_template_weights(depth)

        # At low depth, disable complex templates
        if depth <= 1:
            weights['composition'] = 0.0
            weights['fold'] = 0.0
            weights['conditional'] = 0.0

        # Sample template
        templates = list(weights.keys())
        template_weights = [weights[t] for t in templates]

        # Renormalize after zeroing
        total = sum(template_weights)
        if total > 0:
            template_weights = [w / total for w in template_weights]
        else:
            template_weights = [1.0 / len(templates)] * len(templates)

        template = self.rng.choices(templates, weights=template_weights, k=1)[0]
        input_node = VariableNode(input_var)

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
            # Fold is complex; fall back to simple op
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

        # Fallback
        return input_node

    # ========================================================================
    # Predicate Generation
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
        """Generate a predicate using hand-tuned weights perturbed by noise.

        Only uses functions that exist in the grammar.
        """
        actual_type = substitute_type_vars(elem_type, substitutions)
        elem_node = VariableNode(elem_var)

        # Build weights based on available functions
        base_weights = {}

        if actual_type == int:
            # is_even_odd: needs at least one of is_even, is_odd
            if self._has_function('is_even') or self._has_function('is_odd'):
                base_weights['is_even_odd'] = 0.25

            # compare_const: needs at least one comparison operator
            if self._get_available_comparison_ops():
                base_weights['compare_const'] = 0.30

            # modulo_check: needs % and ==
            if self._has_function('%') and self._has_function('=='):
                base_weights['modulo_check'] = 0.20

            # compound: needs at least one combiner and a base predicate
            available_combiners = [c for c in ['and', 'or'] if self._has_function(c)]
            if available_combiners and base_weights:
                base_weights['compound'] = 0.15

            if use_index and idx_var:
                # index_based: needs comparisons or is_even/is_odd
                if (self._get_available_comparison_ops() or
                    self._has_function('is_even') or self._has_function('is_odd')):
                    base_weights['index_based'] = 0.10
        else:
            if self._has_function('=='):
                base_weights['compare_const'] = 1.0

        # Fallback to trivial predicate if nothing available
        if not base_weights:
            return BooleanNode(True)

        # Apply guard to block trivial predicates
        base_weights = guard_predicate_weights(base_weights, must_be_meaningful=True)

        pattern_weights = TemplateDistributions.perturb_weights(base_weights, self.noise)

        patterns = list(pattern_weights.keys())
        weights = [pattern_weights[p] for p in patterns]
        pattern = self.rng.choices(patterns, weights=weights, k=1)[0]

        if pattern == 'is_even_odd':
            available_parity = [fn for fn in ['is_even', 'is_odd'] if self._has_function(fn)]
            fn_name = self.rng.choice(available_parity)
            return ApplicationNode(VariableNode(fn_name), [elem_node])

        elif pattern == 'compare_const':
            available_ops = self._get_available_comparison_ops()
            op = self.rng.choice(available_ops)
            const = self._sample_int_constant()
            if self.rng.random() < 0.3 and self._has_function('%'):
                # Sometimes use modulo in comparison
                divisor = self._sample_divisor()
                mod_expr = ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])
                return ApplicationNode(VariableNode(op), [mod_expr, NumberNode(self.rng.randint(0, divisor - 1))])
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

        elif pattern == 'modulo_check':
            divisor = self._sample_divisor()
            remainder = self.rng.randint(0, divisor - 1)
            mod_expr = ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])
            return ApplicationNode(VariableNode('=='), [mod_expr, NumberNode(remainder)])

        elif pattern == 'compound':
            pred1 = self._generate_simple_predicate(elem_var, actual_type)
            pred2 = self._generate_simple_predicate(elem_var, actual_type)
            available_combiners = [c for c in ['and', 'or'] if self._has_function(c)]
            combiner = self.rng.choice(available_combiners)
            return ApplicationNode(VariableNode(combiner), [pred1, pred2])

        elif pattern == 'index_based' and use_index and idx_var:
            idx_node = VariableNode(idx_var)
            available_idx_ops = self._get_available_comparison_ops()
            available_idx_ops.extend([fn for fn in ['is_even', 'is_odd'] if self._has_function(fn)])
            op = self.rng.choice(available_idx_ops)
            if op in ['is_even', 'is_odd']:
                return ApplicationNode(VariableNode(op), [idx_node])
            else:
                const = self.rng.randint(0, 10)
                return ApplicationNode(VariableNode(op), [idx_node, NumberNode(const)])

        else:
            # Default to trivial predicate
            return BooleanNode(True)

    def _generate_simple_predicate(self, elem_var: str, elem_type: TypeType) -> ASTNode:
        """Generate a simple predicate (used in compound predicates).

        Only uses functions that exist in the grammar.
        """
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
            fn_name = self.rng.choice(available_parity)
            return ApplicationNode(VariableNode(fn_name), [elem_node])
        else:
            op = self.rng.choice(available_compare)
            const = self._sample_int_constant()
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

    # ========================================================================
    # Transform Generation
    # ========================================================================

    def _generate_transform(
        self,
        elem_var: str,
        elem_type: TypeType,
        ret_type: TypeType,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        use_index: bool = False,
        idx_var: str = None
    ) -> ASTNode:
        """Generate a transform using hand-tuned weights perturbed by noise.

        Only uses functions that exist in the grammar.
        """
        actual_elem = substitute_type_vars(elem_type, substitutions)
        actual_ret = substitute_type_vars(ret_type, substitutions)
        elem_node = VariableNode(elem_var)

        # Build base weights based on types and available functions
        base_weights = {}

        if matchable(actual_elem, actual_ret, substitutions.copy(), update=False):
            base_weights['identity'] = 0.10

        if actual_elem == int and actual_ret == int:
            # arithmetic: needs at least one arithmetic operator
            if self._get_available_arithmetic_ops():
                base_weights['arithmetic'] = 0.40

            # modulo: needs %
            if self._has_function('%'):
                base_weights['modulo'] = 0.15

            # with_index: needs arithmetic ops
            if use_index and idx_var and self._get_available_arithmetic_ops():
                base_weights['with_index'] = 0.15

        if actual_ret == int:
            # conditional: can always do this if we have predicates
            base_weights['conditional'] = 0.15

        if get_base_type(actual_ret) == list:
            # singleton: needs singleton function
            if self._has_function('singleton'):
                base_weights['singleton'] = 0.05

        if not base_weights:
            base_weights['identity'] = 1.0

        # Apply guard to block identity transforms for map (lambda x x is trivial)
        base_weights = guard_transform_weights(base_weights, allow_identity=False)

        pattern_weights = TemplateDistributions.perturb_weights(base_weights, self.noise)

        # Handle case where all weights are zeroed after guard
        if not any(w > 0 for w in pattern_weights.values()):
            # Fall back to arithmetic if available (never re-enable identity)
            if self._get_available_arithmetic_ops():
                pattern_weights['arithmetic'] = 1.0
            elif self._has_function('%'):
                pattern_weights['modulo'] = 1.0
            else:
                # As last resort, use conditional which is non-trivial
                pattern_weights['conditional'] = 1.0

        patterns = list(pattern_weights.keys())
        weights = [pattern_weights[p] for p in patterns]
        pattern = self.rng.choices(patterns, weights=weights, k=1)[0]

        if pattern == 'identity':
            return elem_node

        elif pattern == 'arithmetic':
            available_ops = self._get_available_arithmetic_ops()
            op = self.rng.choice(available_ops)
            const = self.rng.randint(1, 10)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

        elif pattern == 'modulo':
            divisor = self._sample_divisor()
            return ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])

        elif pattern == 'with_index' and use_index and idx_var:
            idx_node = VariableNode(idx_var)
            available_ops = self._get_available_arithmetic_ops()
            op = self.rng.choice(available_ops)
            return ApplicationNode(VariableNode(op), [elem_node, idx_node])

        elif pattern == 'conditional':
            # Generate a non-trivial predicate (guard blocks boolean literals for if conditions)
            if actual_elem == int:
                pred = self._generate_simple_predicate(elem_var, actual_elem)
            else:
                # For non-int types, generate a comparison instead of literal True
                pred = ApplicationNode(VariableNode('=='), [elem_node, elem_node])
            then_val = NumberNode(self.rng.randint(0, 10))
            else_val = NumberNode(self.rng.randint(0, 10))
            return IfNode(pred, then_val, else_val)

        elif pattern == 'singleton':
            return ApplicationNode(VariableNode('singleton'), [elem_node])

        return elem_node

    # ========================================================================
    # Key Function Generation
    # ========================================================================

    def _generate_key_function_body(
        self,
        elem_var: str,
        elem_type: TypeType,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable
    ) -> ASTNode:
        """Generate a key function body using hand-tuned weights perturbed by noise.

        Only uses functions that exist in the grammar.
        """
        actual_type = substitute_type_vars(elem_type, substitutions)
        elem_node = VariableNode(elem_var)

        if actual_type != int:
            return elem_node

        # Build weights based on available functions
        base_weights = {'identity': 0.30}  # Always available

        if self._has_function('-'):
            base_weights['negate'] = 0.20

        if self._has_function('%'):
            base_weights['modulo'] = 0.30

        if self._get_available_arithmetic_ops():
            base_weights['arithmetic'] = 0.20

        pattern_weights = TemplateDistributions.perturb_weights(base_weights, self.noise)

        patterns = list(pattern_weights.keys())
        weights = [pattern_weights[p] for p in patterns]
        pattern = self.rng.choices(patterns, weights=weights, k=1)[0]

        if pattern == 'identity':
            return elem_node
        elif pattern == 'negate':
            return ApplicationNode(VariableNode('-'), [NumberNode(0), elem_node])
        elif pattern == 'modulo':
            divisor = self._sample_divisor()
            return ApplicationNode(VariableNode('%'), [elem_node, NumberNode(divisor)])
        elif pattern == 'arithmetic':
            available_ops = self._get_available_arithmetic_ops()
            op = self.rng.choice(available_ops)
            const = self.rng.randint(1, 10)
            return ApplicationNode(VariableNode(op), [elem_node, NumberNode(const)])

        return elem_node

    # ========================================================================
    # Template Generators
    # ========================================================================

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

        has_map = self._has_function('map')
        has_mapi = self._has_function('mapi')

        if not has_map and not has_mapi:
            return input_node

        use_index = has_mapi and self.rng.random() < 0.2
        fn_name = 'mapi' if use_index else 'map'

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

        has_filter = self._has_function('filter')
        has_filteri = self._has_function('filteri')

        if not has_filter and not has_filteri:
            return input_node

        use_index = has_filteri and self.rng.random() < 0.15
        fn_name = 'filteri' if use_index else 'filter'

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
        """Generate a simple list operation using hand-tuned weights.

        Only uses functions that exist in the grammar.
        """
        input_node = VariableNode(input_var)

        # Get available simple ops
        available_ops = self._get_available_simple_ops()

        if not available_ops:
            return input_node

        # Build weights for available ops only
        weight_map = {
            'reverse': 0.25,
            'unique': 0.20,
            'take': 0.15,
            'drop': 0.15,
            'takelast': 0.12,
            'droplast': 0.13,
        }

        base_weights = {op: weight_map.get(op, 0.1) for op in available_ops}

        op_weights = TemplateDistributions.perturb_weights(base_weights, self.noise)
        ops = list(op_weights.keys())
        weights = [op_weights[op] for op in ops]
        op = self.rng.choices(ops, weights=weights, k=1)[0]

        simple_ops = ['reverse', 'unique']
        if op in simple_ops:
            return ApplicationNode(VariableNode(op), [input_node])
        else:
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
        simple_unary = [op for op in available_simple if op in ['reverse', 'unique']]

        # Build list of available composition patterns
        patterns = []
        if has_filter and has_map:
            patterns.extend(['filter_then_map', 'map_then_filter'])
        if has_filter and simple_unary:
            patterns.extend(['filter_then_simple', 'simple_then_filter'])
        if has_map and simple_unary:
            patterns.append('map_then_simple')

        if not patterns:
            # No compositions available - fallback
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
            simple_op = self.rng.choice(simple_unary)
            return ApplicationNode(VariableNode(simple_op), [filtered])

        elif pattern == 'simple_then_filter':
            simple_op = self.rng.choice(simple_unary)
            simple_result = ApplicationNode(VariableNode(simple_op), [VariableNode(input_var)])
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
            simple_op = self.rng.choice(simple_unary)
            return ApplicationNode(VariableNode(simple_op), [mapped])

    def _substitute_var_in_expr(self, expr: ASTNode, var_name: str, replacement: ASTNode) -> ASTNode:
        """Replace a variable with an expression."""
        if isinstance(expr, VariableNode):
            if expr.name == var_name:
                return replacement
            return expr
        elif isinstance(expr, NumberNode) or isinstance(expr, BooleanNode):
            return expr
        elif isinstance(expr, LambdaNode):
            if var_name in expr.param:
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
    # Sampling Helpers
    # ========================================================================

    def _sample_int_constant(self) -> int:
        """Sample an integer constant from empirical distribution."""
        if self.dists.int_constants:
            return self.dists.sample(self.dists.int_constants, self.noise, self.rng, default=0)
        return self.rng.randint(0, 99)

    def _sample_comparison_op(self) -> str:
        """Sample a comparison operator from empirical distribution."""
        if self.dists.comparison_ops:
            return self.dists.sample(self.dists.comparison_ops, self.noise, self.rng, default='>')
        return self.rng.choice(['<', '>', '=='])

    def _sample_arithmetic_op(self) -> str:
        """Sample an arithmetic operator from empirical distribution."""
        if self.dists.arithmetic_ops:
            # Filter to only valid operators for our use case
            valid_ops = {'+', '-', '*'}
            filtered = Counter({k: v for k, v in self.dists.arithmetic_ops.items() if k in valid_ops})
            if filtered:
                return self.dists.sample(filtered, self.noise, self.rng, default='+')
        return self.rng.choice(['+', '-', '*'])

    def _sample_divisor(self) -> int:
        """Sample a divisor from empirical distribution."""
        if self.dists.divisors:
            return self.dists.sample(self.dists.divisors, self.noise, self.rng, default=2)
        return self.rng.choice([2, 3, 5, 10])

    # ========================================================================
    # Helper Expression Generators
    # ========================================================================

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

        return ApplicationNode(VariableNode('singleton'), [NumberNode(self._sample_int_constant())])

    def _generate_int_expression(
        self,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Generate an integer expression."""
        guard = get_default_guard()
        block_literal = guard.should_block_with_context(app_context, StrategyGuard.LITERAL)

        if not block_literal and (depth <= 0 or self.rng.random() < 0.5):
            return NumberNode(self._sample_int_constant())

        # Try to use a context variable if available
        for var_name, var_type in context.items():
            if var_type == int:
                return VariableNode(var_name)

        op = self._sample_arithmetic_op()
        left = NumberNode(self.rng.randint(0, 20))
        right = NumberNode(self.rng.randint(1, 10))
        return ApplicationNode(VariableNode(op), [left, right])

    def _generate_bool_expression(
        self,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Generate a boolean expression."""
        guard = get_default_guard()
        block_literal = guard.should_block_with_context(app_context, StrategyGuard.LITERAL)

        if not block_literal and (depth <= 0 or self.rng.random() < 0.3):
            return BooleanNode(self.rng.choice([True, False]))

        op = self._sample_comparison_op()
        left = NumberNode(self._sample_int_constant())
        right = NumberNode(self._sample_int_constant())
        return ApplicationNode(VariableNode(op), [left, right])

    def _generate_expression_of_type(
        self,
        target_type: TypeType,
        depth: int,
        context: dict[str, TypeType],
        substitutions: SubstitutionTable,
        app_context: Optional[ApplicationContext] = None
    ) -> ASTNode:
        """Generate an expression of the given type."""
        base = get_base_type(target_type)

        if target_type == int:
            return self._generate_int_expression(depth, context, substitutions, app_context)
        elif target_type == bool:
            return self._generate_bool_expression(depth, context, substitutions, app_context)
        elif base == list:
            return self._generate_list_expression(target_type, depth, context, substitutions)
        elif base == CallableOrig:
            return self._generate_function(target_type, depth, context, substitutions)

        return NumberNode(0)
