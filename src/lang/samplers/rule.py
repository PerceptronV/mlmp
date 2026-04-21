"""
Rule-Style Program Sampler

This module implements the RuleSampler which generates programs of type
Callable[[list[int]], list[int]] with associated I/O pairs, following the
methodology from Josh Rule's meta-program learner paper.

Key features:
- Input and output lists restricted to 0-15 elements
- Favours functions with:
  (i) variance in input and output length
  (ii) variance in the elements of the lists
  (iii) a high number of unique outputs
  (iv) a low number of examples where input == output
- Samples 100 input lists and selects the best 11 based on criteria
- Batch constraints: max 1 identity-like function per batch
- Uniqueness via program strings or behavioural equivalence
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional
import random

from .base import Sampler
from .uniqueness import UniquenessMode


def _fast_variance(values: list[int | float]) -> float:
    """
    Compute sample variance using fast float arithmetic.

    Much faster than statistics.variance which uses exact Fraction arithmetic.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / (n - 1)


from ..composers.base import Composer
from ..ast_nodes import ASTNode, LambdaNode, VariableNode, pretty_print
from ..type_utils import TypeType, SubstitutionTable
from ..evaluator import Evaluator, EvaluationError
from ..environment import Closure
from ..compiler import JITCompiler, JITCompilationError


@dataclass
class SampledProgram:
    """A sampled program with its I/O pairs and metadata."""
    program: ASTNode
    program_str: str
    io_pairs: list[tuple[list[int], list[int]]]
    is_identity_like: bool
    behaviour_signature: Optional[tuple[tuple[int, ...], ...]] = None


class RuleSampler(Sampler):
    """
    Sampler following Josh Rule's meta-program learner methodology.

    Generates programs of type Callable[[list[int]], list[int]] with 11
    associated I/O pairs selected to maximise diversity according to
    Rule's criteria.

    Attributes:
        composer: The composer used for program generation
        evaluator: Evaluator for executing programs
        uniqueness_mode: How to check program uniqueness
        num_io_pairs: Number of I/O pairs to select (default 11)
        num_candidate_inputs: Number of candidate inputs to sample (default 100)
        min_list_length: Minimum list length (default 0)
        max_list_length: Maximum list length (default 15)
        min_element_value: Minimum element value in lists
        max_element_value: Maximum element value in lists
        held_out_inputs: Inputs for behavioural uniqueness checking
    """

    def __init__(
        self,
        composer: Composer,
        uniqueness_mode: UniquenessMode = UniquenessMode.STRING,
        num_io_pairs: int = 11,
        num_candidate_inputs: int = 100,
        min_list_length: int = 0,
        max_list_length: int = 15,
        min_element_value: int = 0,
        max_element_value: int = 100,
        num_held_out_inputs: int = 100,
        depth_variation: int = 2,
        max_attempts_multiplier: int = 100,
        min_quality_score: float = 0.7,
        min_quality_score_required: float = 0.3,
        execution_grammar=None
    ):
        """
        Initialise the Rule sampler.

        Args:
            composer: The composer to use for program generation
            uniqueness_mode: STRING for program string uniqueness,
                           BEHAVIOURAL for behavioural uniqueness
            num_io_pairs: Number of I/O pairs to select per program
            num_candidate_inputs: Number of candidate inputs to sample
            min_list_length: Minimum list length (Rule paper uses 0)
            max_list_length: Maximum list length (Rule paper uses 15)
            min_element_value: Minimum element value in lists
            max_element_value: Maximum element value in lists
            num_held_out_inputs: Number of held-out inputs for behavioural mode
            depth_variation: Maximum depth variation from base depth
            max_attempts_multiplier: Max attempts = n * multiplier
            min_quality_score: Minimum I/O quality score to accept a function.
                             Functions with lower scores are rejected as "boring"
                             (e.g., functions that always return empty list).
                             Default 0.7 filters out degenerate functions.
            min_quality_score_required: Lower quality threshold for programs
                             containing required functions. Some required functions
                             (like aggregators) produce constant-length outputs
                             which score lower. Default 0.3.
            execution_grammar: Optional grammar to use for program execution.
                             If None, uses the composer's grammar.
                             This allows using variant semantics for I/O generation
                             while using canonical grammar for program composition.
        """
        super().__init__(composer)
        # Use execution_grammar for evaluation, or default to composer's grammar
        self._execution_grammar = execution_grammar or composer.grammar
        self.evaluator = Evaluator(self._execution_grammar)
        self.jit_compiler = JITCompiler(self._execution_grammar)
        self.uniqueness_mode = uniqueness_mode
        self.num_io_pairs = num_io_pairs
        self.num_candidate_inputs = num_candidate_inputs
        self.min_list_length = min_list_length
        self.max_list_length = max_list_length
        self.min_element_value = min_element_value
        self.max_element_value = max_element_value
        self.num_held_out_inputs = num_held_out_inputs
        self.depth_variation = depth_variation
        self.max_attempts_multiplier = max_attempts_multiplier
        self.min_quality_score = min_quality_score
        self.min_quality_score_required = min_quality_score_required

        # Execution cache: maps (program_str, input_tuple) -> output or None
        self._execution_cache: dict[tuple[str, tuple[int, ...]], Optional[list[int]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # JIT compilation cache: maps program_str -> compiled function
        self._jit_cache: dict[str, Callable] = {}
        self._jit_cache_hits = 0
        self._jit_cache_misses = 0

        # Generate held-out inputs for behavioural uniqueness checking
        self._generate_held_out_inputs()

        # Precompute normalisation constants for scoring
        self._max_len_var = (self.max_list_length - self.min_list_length) ** 2 / 4
        self._max_elem_var = (self.max_element_value - self.min_element_value) ** 2 / 4

    def _generate_held_out_inputs(self) -> None:
        """Generate held-out inputs for behavioural uniqueness checking."""
        # Use a fixed seed for reproducible held-out inputs
        held_out_rng = random.Random(42)
        self.held_out_inputs: list[list[int]] = []

        for _ in range(self.num_held_out_inputs):
            length = held_out_rng.randint(self.min_list_length, self.max_list_length)
            input_list = [
                held_out_rng.randint(self.min_element_value, self.max_element_value)
                for _ in range(length)
            ]
            self.held_out_inputs.append(input_list)

    def generate_io_pairs_for_program(
        self,
        program: ASTNode,
        rng: Optional[random.Random] = None
    ) -> Optional[list[tuple[list[int], list[int]]]]:
        """
        Generate I/O pairs for a given program using Rule's methodology.

        This is a public method that can be used to generate I/O pairs for
        programs not created by this sampler (e.g., from a file).

        Args:
            program: The program AST (should be a lambda)
            rng: Random number generator (uses internal RNG if None)

        Returns:
            List of (input, output) tuples, or None if generation fails
        """
        if rng is None:
            rng = self.composer.rng

        # Generate candidate inputs
        candidate_inputs = self._generate_candidate_inputs(rng)

        # Compute all I/O pairs
        all_io_pairs = self._compute_io_pairs(program, candidate_inputs)

        # Need at least num_io_pairs valid pairs
        if len(all_io_pairs) < self.num_io_pairs:
            return None

        # Select best pairs
        selected = self._select_best_io_pairs(all_io_pairs, self.num_io_pairs, rng)

        return selected

    def _generate_candidate_inputs(self, rng: random.Random) -> list[list[int]]:
        """
        Generate candidate input lists for I/O pair selection.

        Args:
            rng: Random number generator

        Returns:
            List of candidate input lists
        """
        candidates: list[list[int]] = []

        for _ in range(self.num_candidate_inputs):
            length = rng.randint(self.min_list_length, self.max_list_length)
            input_list = [
                rng.randint(self.min_element_value, self.max_element_value)
                for _ in range(length)
            ]
            candidates.append(input_list)

        return candidates

    def _execute_program(
        self,
        program: ASTNode,
        input_list: list[int],
        program_str: Optional[str] = None
    ) -> Optional[list[int]]:
        """
        Execute a program on an input list with caching.

        Uses JIT compilation for significant speedup when executing the same
        program multiple times (100 candidate inputs + held-out inputs).

        Args:
            program: The program AST (should be a lambda)
            input_list: Input list
            program_str: Optional program string for cache key (computed if None)

        Returns:
            Output list, or None if execution fails
        """
        # Create cache key
        if program_str is None:
            from ..ast_nodes import pretty_print
            program_str = pretty_print(program)
        cache_key = (program_str, tuple(input_list))

        # Check execution cache first (still useful for duplicate inputs)
        if cache_key in self._execution_cache:
            self._cache_hits += 1
            cached_result = self._execution_cache[cache_key]
            # Return a copy to avoid mutations affecting the cache
            return cached_result.copy() if cached_result is not None else None

        self._cache_misses += 1

        # Get or compile the function
        if program_str in self._jit_cache:
            self._jit_cache_hits += 1
            compiled_fn = self._jit_cache[program_str]
        else:
            self._jit_cache_misses += 1
            try:
                # Compile the program to a native Python function
                compiled_fn = self.jit_compiler.compile(program)
                self._jit_cache[program_str] = compiled_fn
            except (JITCompilationError, Exception):
                # Fall back to interpreter if JIT compilation fails
                compiled_fn = None

        # Execute program
        try:
            if compiled_fn is not None:
                # Use JIT-compiled function (fast path)
                output = compiled_fn(input_list)
            else:
                # Fall back to interpreter (slow path)
                closure = self.evaluator.eval(program)

                if type(closure) is not Closure:
                    result = None
                    self._execution_cache[cache_key] = result
                    return result
                else:
                    # Apply the closure to the input
                    env = closure.env.extend(closure.param[0], input_list)
                    output = self.evaluator.eval(closure.body, env)

            # Verify result is a list of integers
            if type(output) is not list:
                result = None
            elif not all(type(x) is int for x in output):
                result = None
            elif len(output) > self.max_list_length:
                result = None
            else:
                result = output

        except (EvaluationError, NameError, TypeError, ZeroDivisionError, Exception):
            result = None

        # Cache the result
        self._execution_cache[cache_key] = result
        # Return a copy to avoid mutations affecting the cache
        return result.copy() if result is not None else None

    def _compute_io_pairs(
        self,
        program: ASTNode,
        candidate_inputs: list[list[int]]
    ) -> list[tuple[list[int], list[int]]]:
        """
        Compute I/O pairs for a program.

        Args:
            program: The program AST
            candidate_inputs: Candidate input lists

        Returns:
            List of (input, output) pairs where execution succeeded
        """
        io_pairs: list[tuple[list[int], list[int]]] = []

        for input_list in candidate_inputs:
            output = self._execute_program(program, input_list)
            if output is not None:
                io_pairs.append((input_list, output))

        return io_pairs

    def _score_io_set(
        self,
        io_pairs: list[tuple[list[int], list[int]]]
    ) -> float:
        """
        Score a set of I/O pairs according to Rule's criteria.

        Higher score is better. The criteria are:
        (i) variance in input and output length
        (ii) variance in the elements of the lists
        (iii) a high number of unique outputs
        (iv) a low number of examples where input == output

        Args:
            io_pairs: List of (input, output) pairs

        Returns:
            Combined score (higher is better)
        """
        if len(io_pairs) < 2:
            return float('-inf')

        # (i) Variance in input and output length
        input_lengths = [len(inp) for inp, _ in io_pairs]
        output_lengths = [len(out) for _, out in io_pairs]

        input_length_var = _fast_variance(input_lengths)
        output_length_var = _fast_variance(output_lengths)
        length_variance_score = input_length_var + output_length_var

        # (ii) Variance in the elements of the lists
        all_input_elements = [x for inp, _ in io_pairs for x in inp]
        all_output_elements = [x for _, out in io_pairs for x in out]

        input_element_var = _fast_variance(all_input_elements)
        output_element_var = _fast_variance(all_output_elements)
        element_variance_score = input_element_var + output_element_var

        # (iii) Number of unique outputs (higher is better)
        unique_outputs = len(set(tuple(out) for _, out in io_pairs))
        unique_output_score = unique_outputs / len(io_pairs)

        # (iv) Low number of input == output (lower is better, so we negate)
        num_identity = sum(1 for inp, out in io_pairs if inp == out)
        non_identity_score = 1.0 - (num_identity / len(io_pairs))

        # Combine scores (weights can be tuned)
        # Normalise length variance by expected max variance
        normalised_length_var = length_variance_score / (self._max_len_var + 1)

        # Normalise element variance by expected max variance
        normalised_element_var = element_variance_score / (self._max_elem_var + 1)

        # Combined score with equal weights
        score = (
            0.25 * normalised_length_var +
            0.25 * normalised_element_var +
            0.25 * unique_output_score +
            0.25 * non_identity_score
        )

        return score

    def _select_best_io_pairs(
        self,
        all_io_pairs: list[tuple[list[int], list[int]]],
        n: int,
        rng: random.Random
    ) -> list[tuple[list[int], list[int]]]:
        """
        Select the best n I/O pairs from candidates using stratified sampling.

        Uses O(n) stratified sampling instead of O(n²) greedy selection:
        1. Bucket pairs by input length into 4 strata
        2. Sample from each bucket, prioritizing unique outputs and non-identity pairs
        3. This ensures length diversity by construction

        Args:
            all_io_pairs: All candidate I/O pairs
            n: Number of pairs to select
            rng: Random number generator for tie-breaking

        Returns:
            Selected I/O pairs
        """
        if len(all_io_pairs) <= n:
            return all_io_pairs

        # Define length buckets for stratification
        # With max_list_length=15, we get 4 buckets: [0-3], [4-7], [8-11], [12-15]
        bucket_size = (self.max_list_length + 1) // 4
        num_buckets = 4

        # Bucket pairs by input length - O(n)
        buckets: list[list[tuple[list[int], list[int]]]] = [[] for _ in range(num_buckets)]
        for pair in all_io_pairs:
            input_len = len(pair[0])
            bucket_idx = min(input_len // bucket_size, num_buckets - 1)
            buckets[bucket_idx].append(pair)

        # Track seen outputs to prioritise uniqueness
        seen_outputs: set[tuple[int, ...]] = set()
        selected: list[tuple[list[int], list[int]]] = []

        # Calculate how many to sample from each bucket
        # Distribute evenly, then handle remainder
        base_per_bucket = n // num_buckets
        remainder = n % num_buckets

        # Sort pairs within each bucket by priority:
        # - Unique output (not seen) > seen output
        # - Non-identity > identity
        def pair_priority(pair: tuple[list[int], list[int]]) -> tuple[int, int, float]:
            inp, out = pair
            out_tuple = tuple(out)
            is_unique = out_tuple not in seen_outputs
            is_non_identity = inp != out
            # Add small random tiebreaker for variety
            return (int(is_unique), int(is_non_identity), rng.random())

        # Sample from each bucket
        for bucket_idx in range(num_buckets):
            bucket = buckets[bucket_idx]
            if not bucket:
                continue

            # How many to take from this bucket
            # Give extra to first 'remainder' buckets
            target = base_per_bucket + (1 if bucket_idx < remainder else 0)

            # Sort by priority (highest first) and take top candidates
            bucket_sorted = sorted(bucket, key=pair_priority, reverse=True)

            for pair in bucket_sorted:
                if len(selected) >= n:
                    break
                if target <= 0:
                    break

                selected.append(pair)
                seen_outputs.add(tuple(pair[1]))
                target -= 1

        # If we didn't get enough (some buckets were empty), fill from remaining
        if len(selected) < n:
            # Collect all unselected pairs
            selected_set = set(id(p) for p in selected)
            remaining = [p for p in all_io_pairs if id(p) not in selected_set]

            # Sort by priority and fill
            remaining_sorted = sorted(remaining, key=pair_priority, reverse=True)
            for pair in remaining_sorted:
                if len(selected) >= n:
                    break
                selected.append(pair)
                seen_outputs.add(tuple(pair[1]))

        return selected

    def _is_identity_like(
        self,
        io_pairs: list[tuple[list[int], list[int]]]
    ) -> bool:
        """
        Check if a function behaves like the identity function.

        A function is identity-like if input == output for all I/O pairs.

        Args:
            io_pairs: The I/O pairs to check

        Returns:
            True if the function is identity-like
        """
        if not io_pairs:
            return False

        return all(inp == out for inp, out in io_pairs)

    def _compute_behaviour_signature(
        self,
        program: ASTNode
    ) -> Optional[tuple[tuple[int, ...], ...]]:
        """
        Compute a behavioural signature for a program.

        The signature is a tuple of outputs on the held-out inputs.
        Programs with the same signature behave identically.

        Args:
            program: The program AST

        Returns:
            Tuple of output tuples, or None if execution fails
        """
        outputs: list[tuple[int, ...]] = []

        for input_list in self.held_out_inputs:
            output = self._execute_program(program, input_list)
            if output is None:
                # Use a sentinel value for failed executions
                outputs.append(tuple())
            else:
                outputs.append(tuple(output))

        return tuple(outputs)

    def _create_identity_program(self) -> ASTNode:
        """
        Create the canonical identity function: (λ (x) x)

        Returns:
            AST for the identity function
        """
        return LambdaNode(param=["x"], body=VariableNode("x"))

    def sample(
        self,
        target_type: TypeType,
        n: int,
        depth: int,
        context: Optional[dict[str, TypeType]] = None
    ) -> list[SampledProgram]:
        """
        Sample a batch of n programs with I/O pairs.

        Args:
            target_type: The desired output type (should be Callable[[list[int]], list[int]])
            n: Number of programs to generate
            depth: Base depth for program generation
            context: Variable bindings in scope (name -> type)

        Returns:
            List of SampledProgram objects

        Raises:
            ValueError: If unable to generate n unique programs
        """
        if n <= 0:
            return []

        if context is None:
            context = {}

        programs: list[SampledProgram] = []
        seen_strings: set[str] = set()
        seen_behaviours: set[tuple[tuple[int, ...], ...]] = set()
        has_identity_like = False

        # Create a separate RNG for sampling
        sampler_rng = random.Random(hash(tuple(self.composer.rng.getstate()[1])))

        max_attempts = n * self.max_attempts_multiplier
        attempts = 0
        seed_offset = 0

        while len(programs) < n and attempts < max_attempts:
            attempts += 1

            # Select depth with variation
            min_depth = max(1, depth - self.depth_variation)  # Min 1 for lambdas
            max_depth = depth + self.depth_variation
            current_depth = sampler_rng.randint(min_depth, max_depth)

            # Create a new seed for this attempt
            current_seed = hash((tuple(self.composer.rng.getstate()[1]), seed_offset)) % (2**32)
            seed_offset += 1

            # Temporarily update the composer's seed
            original_state = self.composer.rng.getstate()
            self.composer.rng.seed(current_seed)
            self.composer.reset_var_counter()

            try:
                # Generate a program
                program = self.composer.generate(
                    target_type=target_type,
                    depth=current_depth,
                    context=context,
                    substitutions=SubstitutionTable()
                )

                program_str = pretty_print(program, inline=True)

                # Check string uniqueness
                if self.uniqueness_mode == UniquenessMode.STRING:
                    if program_str in seen_strings:
                        continue

                # Generate candidate inputs and compute I/O pairs
                candidate_inputs = self._generate_candidate_inputs(sampler_rng)
                all_io_pairs = self._compute_io_pairs(program, candidate_inputs)

                # Need at least num_io_pairs successful executions
                if len(all_io_pairs) < self.num_io_pairs:
                    continue

                # Select best I/O pairs
                selected_io_pairs = self._select_best_io_pairs(
                    all_io_pairs,
                    self.num_io_pairs,
                    sampler_rng
                )

                # Filter out "boring" functions with low-quality I/O pairs
                # Use lower threshold for programs containing required functions
                quality_score = self._score_io_set(selected_io_pairs)
                
                # Check if program contains required functions (lower quality ok)
                min_score = self.min_quality_score
                required_fns = getattr(self.composer, '_required_functions', None)
                if required_fns:
                    program_fns = program.function_names()
                    if program_fns & required_fns:
                        # Program has a required function - use lower threshold
                        min_score = self.min_quality_score_required
                
                if quality_score < min_score:
                    continue

                # Check if identity-like
                is_identity_like = self._is_identity_like(selected_io_pairs)

                # Handle identity-like functions
                if is_identity_like:
                    if has_identity_like:
                        # Already have an identity-like function, skip
                        continue
                    # Replace with canonical identity
                    program = self._create_identity_program()
                    program_str = pretty_print(program, inline=True)
                    # Recompute I/O pairs for the canonical identity
                    all_io_pairs = self._compute_io_pairs(program, candidate_inputs)
                    if len(all_io_pairs) < self.num_io_pairs:
                        continue
                    selected_io_pairs = self._select_best_io_pairs(
                        all_io_pairs,
                        self.num_io_pairs,
                        sampler_rng
                    )
                    has_identity_like = True

                # Compute behavioural signature if needed
                behaviour_signature = None
                if self.uniqueness_mode == UniquenessMode.BEHAVIOURAL:
                    behaviour_signature = self._compute_behaviour_signature(program)
                    if behaviour_signature is None:
                        continue
                    if behaviour_signature in seen_behaviours:
                        continue
                    seen_behaviours.add(behaviour_signature)

                # Add to results
                seen_strings.add(program_str)
                programs.append(SampledProgram(
                    program=program,
                    program_str=program_str,
                    io_pairs=selected_io_pairs,
                    is_identity_like=is_identity_like,
                    behaviour_signature=behaviour_signature
                ))

            except Exception:
                # Generation failed, try again
                pass
            finally:
                # Restore the composer's original state
                self.composer.rng.setstate(original_state)

        if len(programs) < n:
            raise ValueError(
                f"Could not generate {n} unique programs after {attempts} attempts. "
                f"Only generated {len(programs)} unique programs. "
                f"Try increasing depth, depth_variation, or max_attempts_multiplier."
            )

        # Advance the composer's RNG so subsequent sample() calls produce
        # different results. Without this, the RNG state restoration in the
        # loop above causes every sample() call to generate identical programs.
        self.composer.rng.seed(
            hash((tuple(self.composer.rng.getstate()[1]), attempts, len(programs))) % (2**32)
        )

        return programs

    def sample_batch(
        self,
        target_type: TypeType,
        batch_size: int,
        depth: int,
        context: Optional[dict[str, TypeType]] = None
    ) -> list[SampledProgram]:
        """
        Alias for sample() with clearer naming for batch generation.

        Args:
            target_type: The desired output type
            batch_size: Number of programs to generate
            depth: Base depth for program generation
            context: Variable bindings in scope

        Returns:
            List of SampledProgram objects
        """
        return self.sample(target_type, batch_size, depth, context)

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get execution and JIT compilation cache statistics.

        Returns:
            Dictionary with cache hits, misses, hit rate, and cache sizes
        """
        exec_total = self._cache_hits + self._cache_misses
        exec_hit_rate = self._cache_hits / exec_total if exec_total > 0 else 0.0

        jit_total = self._jit_cache_hits + self._jit_cache_misses
        jit_hit_rate = self._jit_cache_hits / jit_total if jit_total > 0 else 0.0

        return {
            'execution_cache': {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'total': exec_total,
                'hit_rate': exec_hit_rate,
                'cache_size': len(self._execution_cache)
            },
            'jit_cache': {
                'hits': self._jit_cache_hits,
                'misses': self._jit_cache_misses,
                'total': jit_total,
                'hit_rate': jit_hit_rate,
                'cache_size': len(self._jit_cache)
            }
        }

    def reset_cache_stats(self) -> None:
        """Reset cache statistics counters (both execution and JIT)."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._jit_cache_hits = 0
        self._jit_cache_misses = 0

    def clear_cache(self) -> None:
        """Clear both execution and JIT compilation caches and reset statistics."""
        self._execution_cache.clear()
        self._jit_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._jit_cache_hits = 0
        self._jit_cache_misses = 0

    def set_execution_grammar(self, grammar) -> None:
        """
        Set a new grammar for program execution (I/O generation).
        
        This allows changing the semantics of functions without changing
        the program composition logic. Useful for semantic variation
        meta-learning where each episode has different function semantics.
        
        Args:
            grammar: The grammar to use for execution (can be a SemanticGrammar
                    or any grammar-like object with the same interface).
        
        Note:
            This clears all caches since the execution semantics have changed.
        """
        self._execution_grammar = grammar
        self.evaluator = Evaluator(grammar)
        self.jit_compiler = JITCompiler(grammar)
        # Clear caches since semantics have changed
        self.clear_cache()

    def get_execution_grammar(self):
        """Get the current execution grammar."""
        return self._execution_grammar


def create_list_to_list_type():
    """
    Create the type Callable[[list[int]], list[int]].

    Returns:
        The callable type for list-to-list functions
    """
    from typing import Callable as TypingCallable
    return TypingCallable[[list[int]], list[int]]
