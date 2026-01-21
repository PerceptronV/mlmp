"""
Query-First Dataset Generator for Program Induction Meta-Learning

This module implements a query-first generation strategy where:
1. Query programs are generated first using the full grammar
2. Support examples are generated to cover all functions in the query
3. Additional support examples maximize grammar coverage

This approach ensures the query distribution is not biased by support generation.

Key differences from the original DatasetGenerator:
- Uses CoverageGuidedComposer for support generation
- Generates query first, then support
- Support examples are guided to cover query functions + maximize diversity
"""

import argparse
import json
import random
import signal
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from tqdm import tqdm

from ..lang.grammar import Grammar, DefaultGrammar
from ..lang.composers import get_composer, list_composers, Composer
from ..lang.composers.template_with_inverse import CoverageGuidedComposer
from ..lang.samplers import RuleSampler, SampledProgram, UniquenessMode, create_list_to_list_type
from ..lang.ast_nodes import ASTNode, pretty_print
from ..lang.evaluator import Evaluator
from ..lang.parser import parse
from ..lang.compiler import JITCompiler
from ..lang.variation_templates import TemplateSemanticGrammar, TemplateVariationRegistry


# ============================================================================
# Data Classes (same as original)
# ============================================================================

def apply_symbol_mapping(program_str: str, mapping: Dict[str, str]) -> str:
    """
    Apply symbol mapping to a program string.
    
    Uses placeholder tokens to avoid double-replacement when canonical names
    appear as shuffled names (e.g., cons->take, take->foldi).
    """
    import re
    
    # Use a two-pass approach with placeholders to avoid double-replacement
    # Pass 1: Replace canonical names with unique placeholders
    placeholders = {}
    result = program_str
    sorted_names = sorted(mapping.keys(), key=len, reverse=True)
    
    for idx, canonical_name in enumerate(sorted_names):
        shuffled_name = mapping[canonical_name]
        if canonical_name != shuffled_name:
            placeholder = f"__PLACEHOLDER_{idx}__"
            placeholders[placeholder] = shuffled_name
            
            # Match function names (word boundaries in Lisp: preceded by '(' or space, followed by space or ')')
            pattern = r'(?<=[(\s])' + re.escape(canonical_name) + r'(?=[\s)]|$)'
            result = re.sub(pattern, placeholder, result)
            pattern = r'\(' + re.escape(canonical_name) + r'(?=\s)'
            result = re.sub(pattern, '(' + placeholder, result)
    
    # Pass 2: Replace placeholders with shuffled names
    for placeholder, shuffled_name in placeholders.items():
        result = result.replace(placeholder, shuffled_name)
    
    return result


@dataclass
class IOPair:
    """Represents a single input-output pair."""
    input: List[int]
    output: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {'input': self.input, 'output': self.output}


@dataclass
class PIExample:
    """Represents a program induction example."""
    io_pairs: List[IOPair]
    program_canonical: str
    program_shuffled: str
    functions_used: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'io_pairs': [pair.to_dict() for pair in self.io_pairs],
            'program_canonical': self.program_canonical,
            'program_shuffled': self.program_shuffled,
            'functions_used': list(self.functions_used)
        }


@dataclass
class MetaLearningEpisode:
    """Represents a complete meta-learning episode."""
    episode_id: int
    symbol_mapping: Dict[str, str]
    support_functions: Set[str]
    support_functions_count: int
    support_examples: List[PIExample]
    query: PIExample
    # Optional semantic variation info (for training with semantic variations)
    # Maps function_name -> {name, template_id, variant_id, description, program, param_values}
    semantic_variants: Optional[Dict[str, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'episode_id': self.episode_id,
            'symbol_mapping': self.symbol_mapping,
            'support_functions': list(self.support_functions),
            'support_functions_count': self.support_functions_count,
            'support_examples': [ex.to_dict() for ex in self.support_examples],
            'query': self.query.to_dict()
        }
        # Include semantic variants if present (with full info including DSL programs)
        if self.semantic_variants is not None:
            result['semantic_variants'] = self.semantic_variants
        return result


# ============================================================================
# Query-First Dataset Generator
# ============================================================================

class QueryFirstDatasetGenerator:
    """
    Dataset generator using query-first generation strategy.
    
    The key innovation is generating the query FIRST, then generating
    support examples that:
    1. Cover all functions used in the query (guaranteed)
    2. Maximize coverage of the remaining grammar functions
    
    This ensures the query distribution is independent of support generation
    and is determined solely by the composer's natural distribution.
    """
    
    def __init__(
        self,
        seed: int,
        n_support: int = 4,
        n_io: int = 11,
        max_program_depth: int = 4,
        gold_grammar: Grammar = DefaultGrammar,
        composer_name: str = 'template',
        uniqueness_mode: UniquenessMode = UniquenessMode.STRING,
        noise: float = 0.3,
        max_query_attempts: int = 10,
        coverage_strength: float = 2.0,
        strict_uniqueness_episodes: int = 0,
        excluded_queries: Optional[Set[str]] = None,
        noise_annealing: bool = False,
        noise_start: Optional[float] = None,
        noise_end: Optional[float] = None,
        noise_anneal_episodes: Optional[int] = None,
        semantic_variations: bool = False,
        canonical_prob: float = 0.0
    ):
        """
        Initialize the query-first dataset generator.
        
        Args:
            seed: Random seed for reproducibility
            n_support: Number of support examples per episode
            n_io: Number of I/O pairs per program
            max_program_depth: Maximum depth for generated programs
            gold_grammar: The gold grammar to use
            composer_name: Name of the base composer for query generation
            uniqueness_mode: How to check program uniqueness
            noise: Base noise parameter (used when noise_annealing=False)
            max_query_attempts: Max attempts to find a valid query
            coverage_strength: How strongly to bias support toward coverage
            strict_uniqueness_episodes: Number of episodes with strict uniqueness.
                                        After this, accept queries if they appear
                                        less than average (allows controlled duplicates).
                                        If 0, always enforce strict uniqueness.
            excluded_queries: Set of query program strings to exclude (e.g., validation queries).
                             These will NEVER be accepted as training queries.
            noise_annealing: If True, gradually increase noise from noise_start to noise_end
            noise_start: Starting noise value (default: 0.1)
            noise_end: Ending noise value (default: 0.8)
            noise_anneal_episodes: Number of episodes over which to anneal (default: 50000)
            semantic_variations: If True, use different function semantics per episode.
                               Each function will have a randomly selected variant,
                               creating a more challenging meta-learning task where
                               the model must learn function semantics from support examples.
            canonical_prob: Probability of selecting canonical variant when semantic_variations=True.
                          0.0 = always random variants, 1.0 = always canonical.
                          Default 0.0 for maximum variation.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_support = n_support
        self.n_io = n_io
        self.max_program_depth = max_program_depth
        self.gold_grammar = gold_grammar
        self.composer_name = composer_name
        self.uniqueness_mode = uniqueness_mode
        self.noise = noise
        self.max_query_attempts = max_query_attempts
        self.coverage_strength = coverage_strength
        self.strict_uniqueness_episodes = strict_uniqueness_episodes
        self._excluded_queries: Set[str] = excluded_queries or set()
        self._episodes_generated = 0
        
        # Semantic variation settings
        self.semantic_variations = semantic_variations
        self.canonical_prob = canonical_prob
        
        # Noise annealing settings
        self.noise_annealing = noise_annealing
        self.noise_start = noise_start if noise_start is not None else 0.1
        self.noise_end = noise_end if noise_end is not None else 0.8
        self.noise_anneal_episodes = noise_anneal_episodes if noise_anneal_episodes is not None else 50000
        
        # Create query composer (use coverage-guided to ensure all functions are covered)
        # Shared coverage tracking initialized early (needed before query_composer)
        self._shared_coverage_counts: Counter = Counter()
        self._coverage_lock = threading.Lock()
        
        # Initial noise depends on whether annealing is enabled
        initial_noise = self.noise_start if noise_annealing else noise
        self.query_composer = CoverageGuidedComposer(
            seed=seed,
            grammar=gold_grammar,
            noise=initial_noise,
            coverage_strength=coverage_strength,
            shared_coverage=(self._shared_coverage_counts, self._coverage_lock)
        )
        
        # Create query sampler
        self.query_sampler = RuleSampler(
            composer=self.query_composer,
            uniqueness_mode=uniqueness_mode,
            num_io_pairs=n_io,
            num_candidate_inputs=100,
            depth_variation=2,
            max_attempts_multiplier=100
        )
        
        # Grammar function names
        self.grammar_names = set(gold_grammar.names)
        
        # Target type for all programs
        self.target_type = create_list_to_list_type()
        
        # Track query counts for diversity
        self._query_counts: Counter = Counter()
        
        # Thread safety lock for shared state
        self._lock = threading.Lock()
        
        # Total query counter (thread-safe)
        self._total_queries = 0
    
    def reset_seen_queries(self) -> None:
        """Reset query counts. Call between dataset splits."""
        with self._lock:
            self._query_counts.clear()
            self._episodes_generated = 0
    
    def _get_current_noise(self, episode_num: Optional[int] = None) -> float:
        """
        Get the current noise level based on annealing schedule.
        
        Args:
            episode_num: Episode number to compute noise for.
                        If None, uses self._episodes_generated.
        
        Returns:
            Current noise value
        """
        if not self.noise_annealing:
            return self.noise
        
        if episode_num is None:
            with self._lock:
                episode_num = self._episodes_generated
        
        # Linear interpolation from noise_start to noise_end
        progress = min(1.0, episode_num / self.noise_anneal_episodes)
        return self.noise_start + progress * (self.noise_end - self.noise_start)
    
    def set_excluded_queries(self, excluded: Set[str]) -> None:
        """
        Set queries to exclude (e.g., validation queries).
        
        These queries will NEVER be accepted, ensuring no overlap
        between training and validation sets.
        
        Args:
            excluded: Set of program strings to exclude
        """
        with self._lock:
            self._excluded_queries = excluded.copy()
    
    def add_excluded_queries(self, excluded: Set[str]) -> None:
        """Add additional queries to the exclusion set."""
        with self._lock:
            self._excluded_queries.update(excluded)
    
    def _is_query_acceptable_locked(self, program_str: str) -> bool:
        """
        Check if a query program is acceptable (assumes lock is held).
        
        A query is acceptable if:
        1. It is NOT in the excluded set (e.g., not a validation query)
        2. During strict uniqueness phase: has never been seen
        3. After strict uniqueness phase: appears less than average frequency
        
        Returns:
            True if the query is acceptable, False otherwise
        """
        # First check: never accept excluded queries (e.g., validation queries)
        if program_str in self._excluded_queries:
            return False
        
        current_count = self._query_counts[program_str]
        
        # Strict uniqueness phase: reject any duplicates
        if (self.strict_uniqueness_episodes == 0 or
                self._episodes_generated < self.strict_uniqueness_episodes):
            return current_count == 0
        
        # Relaxed uniqueness: accept if below average frequency
        n_unique = len(self._query_counts)
        if n_unique == 0:
            return True  # First query is always acceptable
        
        # Average count across all seen programs
        # Accept if this program appears less than expected under uniform distribution
        average_count = self._episodes_generated / n_unique
        return current_count < average_count
    
    def _is_query_acceptable(self, program_str: str) -> bool:
        """Thread-safe version of query acceptability check."""
        with self._lock:
            return self._is_query_acceptable_locked(program_str)
    
    def _try_accept_query(self, program_str: str) -> bool:
        """
        Atomically check if query is acceptable and record it if so.
        
        This is the thread-safe way to accept a query - it checks
        acceptability and records the query in one atomic operation.
        
        Args:
            program_str: The program string to check and potentially accept
        
        Returns:
            True if the query was accepted, False otherwise
        """
        with self._lock:
            if not self._is_query_acceptable_locked(program_str):
                return False
            # Accept the query
            self._query_counts[program_str] += 1
            self._episodes_generated += 1
            return True
    
    def _generate_symbol_mapping(
        self,
        use_identity: bool = False,
        rng: Optional[random.Random] = None
    ) -> Dict[str, str]:
        """
        Generate a symbol mapping for the episode.
        
        Args:
            use_identity: If True, use identity mapping (no shuffling)
            rng: Random generator to use (defaults to self.rng)
        """
        names = list(self.grammar_names)
        
        if use_identity:
            return {name: name for name in names}
        
        shuffled = names.copy()
        (rng or self.rng).shuffle(shuffled)
        return dict(zip(names, shuffled))
    
    def _sampled_to_pi_example(
        self,
        sampled: SampledProgram,
        symbol_mapping: Dict[str, str]
    ) -> PIExample:
        """Convert a SampledProgram to a PIExample."""
        io_pairs = [
            IOPair(input=inp, output=out)
            for inp, out in sampled.io_pairs
        ]
        
        program_canonical = sampled.program_str
        program_shuffled = apply_symbol_mapping(program_canonical, symbol_mapping)
        functions_used = sampled.program.function_names() & self.grammar_names
        
        return PIExample(
            io_pairs=io_pairs,
            program_canonical=program_canonical,
            program_shuffled=program_shuffled,
            functions_used=functions_used
        )
    
    def _generate_support_with_coverage(
        self,
        required_functions: Set[str],
        symbol_mapping: Dict[str, str],
        exclude_programs: Set[str],
        depth: int,
        support_seed: Optional[int] = None,
        execution_grammar=None
    ) -> Optional[List[PIExample]]:
        """
        Generate support examples that cover required functions and maximize diversity.
        
        Args:
            required_functions: Functions that MUST be covered (from query)
            symbol_mapping: Symbol mapping for the episode
            exclude_programs: Program strings to avoid (includes query)
            depth: Program depth
            support_seed: Seed for support generation (for thread-safety)
            execution_grammar: Optional grammar for I/O generation (for semantic variations).
                             If None, uses the canonical gold grammar.
        
        Returns:
            List of support examples, or None if generation fails
        """
        if support_seed is None:
            support_seed = self.rng.randint(0, 2**31 - 1)
        
        # Create coverage-guided composer for support generation
        # Use higher coverage strength for better function coverage
        # Share coverage tracking across episodes for better global coverage
        support_composer = CoverageGuidedComposer(
            seed=support_seed,
            grammar=self.gold_grammar,
            noise=self.noise,
            coverage_strength=max(self.coverage_strength, 3.0),
            shared_coverage=(self._shared_coverage_counts, self._coverage_lock)
        )
        
        # Create sampler for support
        # If execution_grammar is provided, use it for I/O generation (semantic variations)
        support_sampler = RuleSampler(
            composer=support_composer,
            uniqueness_mode=self.uniqueness_mode,
            num_io_pairs=self.n_io,
            num_candidate_inputs=100,
            depth_variation=2,
            max_attempts_multiplier=50,
            execution_grammar=execution_grammar
        )
        
        support_examples: List[PIExample] = []
        covered_functions: Set[str] = set()
        seen_programs: Set[str] = exclude_programs.copy()
        
        # Phase 1: Cover required functions from query
        remaining_required = required_functions.copy()
        # More attempts since some functions are harder to cover
        max_attempts = self.n_support * 50
        attempts = 0
        
        while remaining_required and len(support_examples) < self.n_support and attempts < max_attempts:
            attempts += 1
            
            # Set required functions for next generation
            # This biases the composer toward templates using these functions
            support_composer.set_required_functions(remaining_required)
            
            try:
                # Sample one program at a time
                sampled = support_sampler.sample(
                    target_type=self.target_type,
                    n=1,
                    depth=depth
                )
                
                if not sampled:
                    continue
                
                program = sampled[0]
                
                # Check for uniqueness
                if program.program_str in seen_programs:
                    continue
                
                # Convert to PIExample
                pi_example = self._sampled_to_pi_example(program, symbol_mapping)
                
                # Update tracking
                seen_programs.add(program.program_str)
                covered_functions.update(pi_example.functions_used)
                remaining_required -= pi_example.functions_used
                support_examples.append(pi_example)
                
            except ValueError:
                continue
        
        # Check if we covered all required functions
        if remaining_required:
            # Failed to cover all query functions
            return None
        
        # Phase 2: Fill remaining slots with diverse programs
        while len(support_examples) < self.n_support:
            attempts += 1
            if attempts > max_attempts * 2:
                break
            
            # Clear required functions, let coverage bias guide selection
            support_composer.set_required_functions(None)
            
            try:
                sampled = support_sampler.sample(
                    target_type=self.target_type,
                    n=1,
                    depth=depth
                )
                
                if not sampled:
                    continue
                
                program = sampled[0]
                
                if program.program_str in seen_programs:
                    continue
                
                pi_example = self._sampled_to_pi_example(program, symbol_mapping)
                seen_programs.add(program.program_str)
                covered_functions.update(pi_example.functions_used)
                support_examples.append(pi_example)
                
            except ValueError:
                continue
        
        if len(support_examples) < self.n_support:
            # Didn't get enough support examples
            return None
        
        return support_examples
    
    def generate_episode(
        self,
        episode_id: int,
        use_identity_mapping: bool = False
    ) -> Optional[MetaLearningEpisode]:
        """
        Generate a single meta-learning episode using query-first strategy.

        This is the single-threaded version that uses shared query_sampler.
        For multi-threaded generation, use generate_dataset_parallel().

        Steps:
        1. Generate query program using full grammar
        2. Extract functions used by query
        3. Generate support examples covering query functions + maximizing diversity
        4. Apply symbol mapping

        Args:
            episode_id: Unique identifier for this episode
            use_identity_mapping: If True, don't shuffle symbols

        Returns:
            MetaLearningEpisode, or None if generation fails
        """
        # Generate symbol mapping for this episode
        symbol_mapping = self._generate_symbol_mapping(use_identity_mapping)

        # Sample semantic grammar if semantic variations are enabled
        semantic_grammar = None
        semantic_variant_info = None
        if self.semantic_variations:
            semantic_grammar = TemplateSemanticGrammar.sample(
                self.gold_grammar,
                self.rng,
                canonical_prob=self.canonical_prob
            )
            semantic_variant_info = semantic_grammar.get_variant_info()

        try:
            # Step 1: Generate query using full grammar
            # If semantic variations enabled, use semantic grammar for execution
            if semantic_grammar is not None:
                self.query_sampler.set_execution_grammar(semantic_grammar)

            query_candidates = self.query_sampler.sample(
                target_type=self.target_type,
                n=self.max_query_attempts,
                depth=self.max_program_depth
            )

            # Find a valid query and accept it atomically
            # This ensures: (1) not in excluded set, (2) not overrepresented
            query_sampled = None
            for candidate in query_candidates:
                if self._try_accept_query(candidate.program_str):
                    query_sampled = candidate
                    # Update coverage tracking ONLY for accepted queries
                    self.query_composer.update_coverage_from_program(candidate.program)
                    break

            if query_sampled is None:
                return None

            # Step 2: Extract query functions
            query_example = self._sampled_to_pi_example(query_sampled, symbol_mapping)
            query_functions = query_example.functions_used

            # Step 3: Generate support with coverage
            support_examples = self._generate_support_with_coverage(
                required_functions=query_functions,
                symbol_mapping=symbol_mapping,
                exclude_programs={query_example.program_canonical},
                depth=self.max_program_depth,
                execution_grammar=semantic_grammar
            )

            if support_examples is None:
                return None

            # Collect all support functions
            support_functions: Set[str] = set()
            for ex in support_examples:
                support_functions.update(ex.functions_used)

            return MetaLearningEpisode(
                episode_id=episode_id,
                symbol_mapping=symbol_mapping,
                support_functions=support_functions,
                support_functions_count=len(support_functions),
                support_examples=support_examples,
                query=query_example,
                semantic_variants=semantic_variant_info
            )

        except (ValueError, KeyError):
            return None

        finally:
            # Reset execution grammar to canonical for next episode
            # (since query_sampler is shared in single-threaded mode)
            if semantic_grammar is not None:
                self.query_sampler.set_execution_grammar(self.gold_grammar)

    def _generate_episode_with_seed(
        self,
        episode_id: int,
        episode_seed: int,
        use_identity_mapping: bool = False,
        progress_episode_num: Optional[int] = None
    ) -> Optional[MetaLearningEpisode]:
        """
        Generate an episode with a specific seed (thread-safe).
        
        This version creates thread-local samplers to avoid race conditions
        when called from multiple threads. Uses shared state (protected by locks)
        for uniqueness checking.
        
        Args:
            episode_id: Unique identifier for this episode  
            episode_seed: Seed for this specific episode's generation
            use_identity_mapping: If True, don't shuffle symbols
            progress_episode_num: Episode number for noise annealing calculation
        
        Returns:
            MetaLearningEpisode, or None if generation fails
        """
        # Thread-local RNG
        local_rng = random.Random(episode_seed)
        
        # Generate symbol mapping with local RNG
        symbol_mapping = self._generate_symbol_mapping(use_identity_mapping, rng=local_rng)
        
        # Get current noise level (may be annealed)
        current_noise = self._get_current_noise(progress_episode_num)
        
        # Sample semantic grammar if semantic variations are enabled
        semantic_grammar = None
        semantic_variant_info = None
        if self.semantic_variations:
            semantic_grammar = TemplateSemanticGrammar.sample(
                self.gold_grammar,
                local_rng,
                canonical_prob=self.canonical_prob
            )
            # Get full variant info including DSL programs
            semantic_variant_info = semantic_grammar.get_variant_info()
        
        try:
            # Create thread-local composer and sampler for query generation
            # Use coverage-guided to ensure all functions are covered
            # Pass shared coverage tracking so coverage accumulates across ALL queries
            query_composer = CoverageGuidedComposer(
                seed=episode_seed,
                grammar=self.gold_grammar,
                noise=current_noise,
                coverage_strength=self.coverage_strength,
                shared_coverage=(self._shared_coverage_counts, self._coverage_lock)
            )
            query_sampler = RuleSampler(
                composer=query_composer,
                uniqueness_mode=self.uniqueness_mode,
                num_io_pairs=self.n_io,
                num_candidate_inputs=100,
                depth_variation=2,
                max_attempts_multiplier=100,
                # Use semantic grammar for execution if semantic variations enabled
                execution_grammar=semantic_grammar
            )
            
            # Step 1: Generate query using full grammar
            query_candidates = query_sampler.sample(
                target_type=self.target_type,
                n=self.max_query_attempts,
                depth=self.max_program_depth
            )
            
            # Find a valid query and accept it atomically (thread-safe)
            query_sampled = None
            for candidate in query_candidates:
                if self._try_accept_query(candidate.program_str):
                    query_sampled = candidate
                    # Update coverage tracking ONLY for accepted queries
                    query_composer.update_coverage_from_program(candidate.program)
                    break
            
            if query_sampled is None:
                return None
            
            # Step 2: Extract query functions
            query_example = self._sampled_to_pi_example(query_sampled, symbol_mapping)
            query_functions = query_example.functions_used
            
            # Step 3: Generate support with coverage (use derived seed)
            support_seed = local_rng.randint(0, 2**31 - 1)
            support_examples = self._generate_support_with_coverage(
                required_functions=query_functions,
                symbol_mapping=symbol_mapping,
                exclude_programs={query_example.program_canonical},
                depth=self.max_program_depth,
                support_seed=support_seed,
                execution_grammar=semantic_grammar  # Pass semantic grammar for I/O generation
            )
            
            if support_examples is None:
                return None
            
            # Collect all support functions
            support_functions: Set[str] = set()
            for ex in support_examples:
                support_functions.update(ex.functions_used)
            
            return MetaLearningEpisode(
                episode_id=episode_id,
                symbol_mapping=symbol_mapping,
                support_functions=support_functions,
                support_functions_count=len(support_functions),
                support_examples=support_examples,
                query=query_example,
                semantic_variants=semantic_variant_info
            )
            
        except (ValueError, KeyError):
            return None
    
    def generate_dataset(
        self,
        n_episodes: int,
        output_dir: Path,
        split: str = 'train',
        first_is_identity: bool = False
    ):
        """
        Generate a dataset of meta-learning episodes.
        
        Args:
            n_episodes: Number of episodes to generate
            output_dir: Directory to save the episodes
            split: Dataset split name ('train' or 'eval')
            first_is_identity: If True, first episode uses identity mapping
        """
        output_dir = Path(output_dir) / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {n_episodes} {split} episodes (QUERY-FIRST strategy)")
        print(f"Output directory: {output_dir}")
        print(f"Parameters: n_support={self.n_support}, n_io={self.n_io}, "
              f"max_depth={self.max_program_depth}")
        print(f"Coverage strength: {self.coverage_strength}")
        
        successful = 0
        attempts = 0
        max_attempts = n_episodes * 10
        
        with tqdm(total=n_episodes, desc=f"{split}", unit="episode") as pbar:
            while successful < n_episodes and attempts < max_attempts:
                use_identity = first_is_identity and split == 'eval' and successful == 0
                
                episode = self.generate_episode(
                    episode_id=successful,
                    use_identity_mapping=use_identity
                )
                
                attempts += 1
                
                if episode is not None:
                    # Save the episode atomically (write to temp file, then rename)
                    output_path = output_dir / f"episode_{successful:06d}.json"
                    temp_path = output_dir / f".episode_{successful:06d}.json.tmp"
                    
                    with open(temp_path, 'w') as f:
                        json.dump(episode.to_dict(), f, indent=2)
                    
                    # Atomic rename (prevents corrupted files if interrupted)
                    temp_path.rename(output_path)
                    
                    successful += 1
                    pbar.update(1)
                    pbar.set_postfix(attempts=attempts, success_rate=f"{successful/attempts:.1%}")
        
        if successful < n_episodes:
            print(f"\nWarning: Only generated {successful}/{n_episodes} episodes "
                  f"after {attempts} attempts")
        else:
            print(f"\nSuccessfully generated {n_episodes} {split} episodes!")
    
    def generate_dataset_parallel(
        self,
        n_episodes: int,
        output_dir: Path,
        split: str = 'train',
        first_is_identity: bool = False,
        num_workers: int = 4
    ):
        """
        Generate a dataset of meta-learning episodes using multiple threads.
        
        This method uses a thread pool to generate episodes in parallel while
        maintaining thread-safe uniqueness checking through shared state with locks.
        
        Args:
            n_episodes: Number of episodes to generate
            output_dir: Directory to save the episodes
            split: Dataset split name ('train' or 'eval')
            first_is_identity: If True, first episode uses identity mapping
            num_workers: Number of worker threads
        """
        output_dir = Path(output_dir) / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {n_episodes} {split} episodes (QUERY-FIRST strategy, {num_workers} workers)")
        print(f"Output directory: {output_dir}")
        print(f"Parameters: n_support={self.n_support}, n_io={self.n_io}, "
              f"max_depth={self.max_program_depth}")
        print(f"Coverage strength: {self.coverage_strength}")
        if self.noise_annealing:
            print(f"Noise annealing: {self.noise_start:.2f} -> {self.noise_end:.2f} "
                  f"over {self.noise_anneal_episodes} episodes")
        else:
            print(f"Noise: {self.noise:.2f} (fixed)")
        if self.semantic_variations:
            print(f"Semantic variations: ENABLED (canonical_prob={self.canonical_prob:.2f})")
        
        # Pre-generate all seeds for reproducibility
        # Each attempt gets a unique seed derived from base seed
        max_attempts = n_episodes * 20
        attempt_seeds = [self.seed + i * 1000 for i in range(max_attempts)]
        
        successful = 0
        attempts = 0
        next_episode_id = 0
        
        # Track pending futures and their metadata
        pending_futures: Dict[Any, Tuple[int, int, bool]] = {}  # future -> (episode_id, seed, use_identity)
        
        # Flag to signal shutdown on Ctrl+C
        shutdown_flag = threading.Event()
        
        def signal_handler(signum, frame):
            """Handle Ctrl+C gracefully"""
            print("\n\n⚠️  Interrupted! Shutting down workers...")
            shutdown_flag.set()
        
        # Install signal handler for Ctrl+C
        old_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            with tqdm(total=n_episodes, desc=f"{split}", unit="episode") as pbar:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Initial batch of submissions
                    batch_size = min(num_workers * 2, n_episodes)
                    for i in range(batch_size):
                        if attempts >= max_attempts:
                            break
                        
                        use_identity = first_is_identity and split == 'eval' and next_episode_id == 0
                        seed = attempt_seeds[attempts]
                        
                        future = executor.submit(
                            self._generate_episode_with_seed,
                            episode_id=next_episode_id,
                            episode_seed=seed,
                            use_identity_mapping=use_identity,
                            progress_episode_num=successful
                        )
                        pending_futures[future] = (next_episode_id, seed, use_identity)
                        attempts += 1
                    
                    # Process completions and submit new tasks
                    while pending_futures and successful < n_episodes and not shutdown_flag.is_set():
                        # Wait for at least one future to complete
                        done_futures = []
                        try:
                            for future in as_completed(pending_futures, timeout=0.5):
                                done_futures.append(future)
                                break  # Process one at a time to maintain order where possible
                        except TimeoutError:
                            # No futures completed yet, check shutdown flag
                            continue
                        
                        # Check shutdown flag
                        if shutdown_flag.is_set() or not done_futures:
                            break
                        
                        for future in done_futures:
                            episode_id, seed, use_identity = pending_futures.pop(future)
                            
                            try:
                                episode = future.result()
                            except Exception:
                                episode = None
                            
                            if episode is not None:
                                # Save the episode atomically (write to temp file, then rename)
                                output_path = output_dir / f"episode_{successful:06d}.json"
                                temp_path = output_dir / f".episode_{successful:06d}.json.tmp"
                                
                                with open(temp_path, 'w') as f:
                                    json.dump(episode.to_dict(), f, indent=2)
                                
                                # Atomic rename (prevents corrupted files if interrupted)
                                temp_path.rename(output_path)
                                
                                successful += 1
                                pbar.update(1)
                                
                                # Update postfix with current noise if annealing
                                postfix_dict = {
                                    'attempts': attempts,
                                    'success': f"{successful/max(attempts,1):.1%}",
                                }
                                if self.noise_annealing:
                                    postfix_dict['noise'] = f"{self._get_current_noise(successful):.2f}"
                                pbar.set_postfix(**postfix_dict)
                            
                            # If we still need more episodes, submit another task
                            if successful < n_episodes and attempts < max_attempts:
                                next_episode_id = successful  # Use successful count as next ID
                                use_identity_new = first_is_identity and split == 'eval' and next_episode_id == 0
                                seed_new = attempt_seeds[attempts]
                                
                                new_future = executor.submit(
                                    self._generate_episode_with_seed,
                                    episode_id=next_episode_id,
                                    episode_seed=seed_new,
                                    use_identity_mapping=use_identity_new,
                                    progress_episode_num=successful
                                )
                                pending_futures[new_future] = (next_episode_id, seed_new, use_identity_new)
                                attempts += 1
                        
                        # Cancel any remaining pending futures if interrupted or done
                        if shutdown_flag.is_set():
                            print(f"\n⚠️  Cancelling {len(pending_futures)} pending tasks...")
                            for future in pending_futures:
                                future.cancel()
                            break
                    
                    # Cancel any remaining pending futures if we hit target
                    for future in pending_futures:
                        future.cancel()
        
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, old_handler)
        
        if shutdown_flag.is_set():
            print(f"\n❌ Generation interrupted by user!")
            print(f"   Generated {successful}/{n_episodes} episodes before shutdown")
            sys.exit(1)
        elif successful < n_episodes:
            print(f"\nWarning: Only generated {successful}/{n_episodes} episodes "
                  f"after {attempts} attempts")
        else:
            print(f"\nSuccessfully generated {n_episodes} {split} episodes!")


# ============================================================================
# Query-First Validation Dataset Generator
# ============================================================================

class QueryFirstValidationGenerator:
    """
    Validation dataset generator using canonical programs with query-first strategy.
    
    Similar to ValidationDatasetGenerator but:
    - Selects query from canonical programs FIRST
    - Generates support to cover query functions
    """
    
    def __init__(
        self,
        seed: int,
        canonical_programs_path: Path,
        n_support: int = 4,
        n_io: int = 11,
        max_program_depth: int = 4,
        gold_grammar: Grammar = DefaultGrammar,
        composer_name: str = 'template',
        uniqueness_mode: UniquenessMode = UniquenessMode.STRING,
        coverage_strength: float = 2.0
    ):
        """Initialize the query-first validation generator."""
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_support = n_support
        self.n_io = n_io
        self.max_program_depth = max_program_depth
        self.gold_grammar = gold_grammar
        self.composer_name = composer_name
        self.uniqueness_mode = uniqueness_mode
        self.coverage_strength = coverage_strength
        
        # Load canonical programs
        self.canonical_programs = self._load_canonical_programs(canonical_programs_path)
        print(f"Loaded {len(self.canonical_programs)} canonical programs")
        
        # Create compiler for I/O generation
        self.compiler = JITCompiler(gold_grammar)
        
        # Grammar function names
        self.grammar_names = set(gold_grammar.names)
        
        # Target type
        self.target_type = create_list_to_list_type()
        
        # Track used queries
        self._used_query_indices: Set[int] = set()
        self._used_query_strings: Set[str] = set()
    
    def reset_used_queries(self) -> None:
        """Reset tracking of used queries."""
        self._used_query_indices.clear()
        self._used_query_strings: Set[str] = set()
    
    def get_used_query_strings(self) -> Set[str]:
        """
        Get the set of query program strings used in validation.
        
        Use this to exclude validation queries from training to ensure
        no overlap between training and validation sets.
        
        Returns:
            Set of canonical program strings used as validation queries
        """
        return self._used_query_strings.copy()
    
    def _load_canonical_programs(self, path: Path) -> List[str]:
        """Load canonical programs from file."""
        programs = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    programs.append(line)
        return programs
    
    def _extract_functions_from_program_str(self, program_str: str) -> Set[str]:
        """Extract grammar function names from a program string."""
        import re
        found_functions = set()
        
        for name in self.grammar_names:
            pattern = r'(?<=\()\s*' + re.escape(name) + r'\s+|(?<=\s)' + re.escape(name) + r'(?=\s|\))'
            if re.search(pattern, program_str):
                found_functions.add(name)
        
        return found_functions
    
    def _generate_io_pairs(self, program_node: ASTNode, n: int) -> List[IOPair]:
        """Generate I/O pairs for a program."""
        io_pairs = []
        attempts = 0
        max_attempts = n * 100
        seen_inputs = set()
        seen_outputs = set()
        
        try:
            program_func = self.compiler.compile(program_node)
        except Exception:
            return []
        
        while len(io_pairs) < n and attempts < max_attempts:
            list_length = self.rng.randint(0, 15)
            input_list = [self.rng.randint(-10, 99) for _ in range(list_length)]
            
            input_tuple = tuple(input_list)
            if input_tuple in seen_inputs:
                attempts += 1
                continue
            
            try:
                output = program_func(input_list)
                if isinstance(output, list):
                    output_tuple = tuple(output)
                    if output_tuple not in seen_outputs or len(io_pairs) < n // 2:
                        io_pairs.append(IOPair(input=input_list, output=output))
                        seen_inputs.add(input_tuple)
                        seen_outputs.add(output_tuple)
            except Exception:
                pass
            
            attempts += 1
        
        return io_pairs
    
    def _sampled_to_pi_example(
        self,
        sampled: SampledProgram,
        symbol_mapping: Dict[str, str]
    ) -> PIExample:
        """Convert SampledProgram to PIExample."""
        io_pairs = [
            IOPair(input=inp, output=out)
            for inp, out in sampled.io_pairs
        ]
        
        program_canonical = sampled.program_str
        program_shuffled = apply_symbol_mapping(program_canonical, symbol_mapping)
        functions_used = sampled.program.function_names() & self.grammar_names
        
        return PIExample(
            io_pairs=io_pairs,
            program_canonical=program_canonical,
            program_shuffled=program_shuffled,
            functions_used=functions_used
        )
    
    def generate_episode(
        self,
        episode_id: int,
        allow_query_reuse: bool = False
    ) -> Optional[MetaLearningEpisode]:
        """
        Generate a validation episode using query-first strategy.
        
        1. Select a canonical program as query
        2. Generate support covering query functions
        """
        # Identity mapping for validation
        identity_mapping = {name: name for name in self.grammar_names}
        
        # Step 1: Select a canonical query
        available_indices = [
            i for i in range(len(self.canonical_programs))
            if allow_query_reuse or i not in self._used_query_indices
        ]
        
        if not available_indices:
            return None
        
        self.rng.shuffle(available_indices)
        
        query_idx = None
        query_str = None
        query_functions = None
        
        for idx in available_indices:
            candidate = self.canonical_programs[idx]
            functions = self._extract_functions_from_program_str(candidate)
            
            # Skip if uses functions not in grammar
            if not functions.issubset(self.grammar_names):
                continue
            
            # Valid query candidate
            query_idx = idx
            query_str = candidate
            query_functions = functions
            break
        
        if query_idx is None:
            return None
        
        # Mark query as used
        self._used_query_indices.add(query_idx)
        
        # Parse query and generate I/O pairs
        try:
            query_node = parse(query_str)
        except Exception:
            self._used_query_indices.discard(query_idx)
            return None
        
        query_io_pairs = self._generate_io_pairs(query_node, self.n_io)
        
        if len(query_io_pairs) < self.n_io // 2:
            self._used_query_indices.discard(query_idx)
            return None
        
        query_example = PIExample(
            io_pairs=query_io_pairs,
            program_canonical=query_str,
            program_shuffled=query_str,
            functions_used=query_functions
        )
        
        # Step 2: Generate support covering query functions
        support_seed = self.rng.randint(0, 2**31 - 1)
        
        support_composer = CoverageGuidedComposer(
            seed=support_seed,
            grammar=self.gold_grammar,
            noise=0.3,
            coverage_strength=self.coverage_strength
        )
        
        support_sampler = RuleSampler(
            composer=support_composer,
            uniqueness_mode=self.uniqueness_mode,
            num_io_pairs=self.n_io,
            num_candidate_inputs=100,
            depth_variation=2,
            max_attempts_multiplier=50
        )
        
        support_examples: List[PIExample] = []
        covered_functions: Set[str] = set()
        seen_programs: Set[str] = {query_str}
        remaining_required = query_functions.copy()
        
        max_attempts = self.n_support * 30
        attempts = 0
        
        # Phase 1: Cover required functions
        while remaining_required and len(support_examples) < self.n_support and attempts < max_attempts:
            attempts += 1
            support_composer.set_required_functions(remaining_required)
            
            try:
                sampled = support_sampler.sample(
                    target_type=self.target_type,
                    n=1,
                    depth=self.max_program_depth
                )
                
                if not sampled:
                    continue
                
                program = sampled[0]
                if program.program_str in seen_programs:
                    continue
                
                pi_example = self._sampled_to_pi_example(program, identity_mapping)
                seen_programs.add(program.program_str)
                covered_functions.update(pi_example.functions_used)
                remaining_required -= pi_example.functions_used
                support_examples.append(pi_example)
                
            except ValueError:
                continue
        
        # Check coverage
        if remaining_required:
            self._used_query_indices.discard(query_idx)
            return None
        
        # Phase 2: Fill remaining slots
        while len(support_examples) < self.n_support and attempts < max_attempts * 2:
            attempts += 1
            support_composer.set_required_functions(None)
            
            try:
                sampled = support_sampler.sample(
                    target_type=self.target_type,
                    n=1,
                    depth=self.max_program_depth
                )
                
                if not sampled:
                    continue
                
                program = sampled[0]
                if program.program_str in seen_programs:
                    continue
                
                pi_example = self._sampled_to_pi_example(program, identity_mapping)
                seen_programs.add(program.program_str)
                covered_functions.update(pi_example.functions_used)
                support_examples.append(pi_example)
                
            except ValueError:
                continue
        
        if len(support_examples) < self.n_support:
            self._used_query_indices.discard(query_idx)
            return None
        
        # Track the query string for exclusion from training
        self._used_query_strings.add(query_str)
        
        return MetaLearningEpisode(
            episode_id=episode_id,
            symbol_mapping=identity_mapping,
            support_functions=covered_functions,
            support_functions_count=len(covered_functions),
            support_examples=support_examples,
            query=query_example
        )
    
    def generate_dataset(
        self,
        n_episodes: int,
        output_dir: Path,
        split: str = 'validation',
        allow_query_reuse: bool = False
    ):
        """Generate a validation dataset."""
        output_dir = Path(output_dir) / split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {n_episodes} {split} episodes (QUERY-FIRST strategy)")
        print(f"Output directory: {output_dir}")
        print(f"Coverage strength: {self.coverage_strength}")
        
        successful = 0
        attempts = 0
        max_attempts = n_episodes * 20
        
        with tqdm(total=n_episodes, desc=f"{split}", unit="episode") as pbar:
            while successful < n_episodes and attempts < max_attempts:
                episode = self.generate_episode(
                    episode_id=successful,
                    allow_query_reuse=allow_query_reuse
                )
                
                attempts += 1
                
                if episode is not None:
                    # Save the episode atomically (write to temp file, then rename)
                    output_path = output_dir / f"episode_{successful:06d}.json"
                    temp_path = output_dir / f".episode_{successful:06d}.json.tmp"
                    
                    with open(temp_path, 'w') as f:
                        json.dump(episode.to_dict(), f, indent=2)
                    
                    # Atomic rename (prevents corrupted files if interrupted)
                    temp_path.rename(output_path)
                    
                    successful += 1
                    pbar.update(1)
                    pbar.set_postfix(attempts=attempts, success_rate=f"{successful/attempts:.1%}")
        
        if successful < n_episodes:
            print(f"\nWarning: Only generated {successful}/{n_episodes} episodes "
                  f"after {attempts} attempts")
        else:
            print(f"\nSuccessfully generated {n_episodes} {split} episodes!")


def main():
    """Main entry point for query-first dataset generation."""
    parser = argparse.ArgumentParser(
        description='Generate program induction meta-learning dataset (query-first strategy)'
    )
    
    # Dataset size parameters
    parser.add_argument('--n-train', type=int, default=100000)
    parser.add_argument('--n-validation', type=int, default=210)
    
    # Episode structure parameters
    parser.add_argument('--n-support', type=int, default=30)
    parser.add_argument('--n-io', type=int, default=11)
    
    # Program generation parameters
    parser.add_argument('--max-program-depth', type=int, default=6)
    parser.add_argument('--composer', type=str, default='template', choices=list_composers())
    parser.add_argument('--uniqueness', type=str, default='string', choices=['string', 'behavioral'])
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='datasets')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=float, default=0.3,
                        help='Fixed noise level (used when --noise-annealing is False)')
    
    # Parallelization
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help='Number of worker threads for training generation. '
             '1 = single-threaded (default), >1 = multi-threaded.'
    )
    
    # Noise annealing parameters
    parser.add_argument(
        '--noise-annealing', action='store_true',
        help='Enable noise annealing: gradually increase noise from --noise-start to --noise-end'
    )
    parser.add_argument(
        '--noise-start', type=float, default=0.2,
        help='Starting noise value for annealing (default: 0.2)'
    )
    parser.add_argument(
        '--noise-end', type=float, default=0.5,
        help='Ending noise value for annealing (default: 0.5)'
    )
    parser.add_argument(
        '--noise-anneal-episodes', type=int, default=50000,
        help='Number of episodes over which to anneal noise (default: 50000)'
    )
    
    # Coverage parameters
    parser.add_argument(
        '--coverage-strength', type=float, default=3.0,
        help='How strongly to bias support generation toward coverage (default: 3.0)'
    )
    
    # Uniqueness parameters
    parser.add_argument(
        '--strict-uniqueness-episodes', type=int, default=5000,
        help='Episodes with strict uniqueness before allowing controlled duplicates. '
             '0 = always strict (default: 0)'
    )
    
    # Canonical programs for validation
    parser.add_argument(
        '--canonical-programs', type=str, default='src/data/rule/functions.txt'
    )
    
    # Semantic variation parameters (TRAINING ONLY)
    parser.add_argument(
        '--semantic-variations', action='store_true',
        help='Enable semantic variations for training. Each function will have '
             'randomly varying semantics per episode (e.g., + might mean max, '
             'map might skip every other element). Creates a more challenging '
             'meta-learning task. NOT applied to validation (canonical semantics).'
    )
    parser.add_argument(
        '--canonical-prob', type=float, default=0.0,
        help='Probability of selecting canonical variant when --semantic-variations is enabled. '
             '0.0 = always random variants, 1.0 = always canonical (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    uniqueness_mode = (
        UniquenessMode.BEHAVIORAL if args.uniqueness == 'behavioral'
        else UniquenessMode.STRING
    )
    
    # Build output directory name
    output_name = f"query_first_{args.composer}_seed{args.seed}"
    if args.semantic_variations:
        output_name += "_semvar"
    output_dir = Path(args.output_dir) / output_name
    
    # Generate validation set first (single-threaded)
    print("\n" + "=" * 80)
    print("VALIDATION SET (QUERY-FIRST)")
    print("=" * 80)
    
    validation_generator = QueryFirstValidationGenerator(
        seed=args.seed,
        canonical_programs_path=Path(args.canonical_programs),
        n_support=args.n_support,
        n_io=args.n_io,
        max_program_depth=args.max_program_depth,
        gold_grammar=DefaultGrammar,
        composer_name=args.composer,
        uniqueness_mode=uniqueness_mode,
        coverage_strength=args.coverage_strength
    )
    
    # Generate validation dataset (one episode per canonical program ideally)
    print(f"Note: {len(validation_generator.canonical_programs)} canonical programs available")
    if args.n_validation < len(validation_generator.canonical_programs):
        print(f"Warning: Requesting {args.n_validation} episodes but there are "
              f"{len(validation_generator.canonical_programs)} canonical programs. "
              f"Not all canonical programs will be covered.")
    
    validation_generator.generate_dataset(
        n_episodes=args.n_validation,
        output_dir=output_dir,
        split='validation',
        allow_query_reuse=False
    )
    
    # Get validation queries to exclude from training
    validation_queries = validation_generator.get_used_query_strings()
    print(f"\nExcluding {len(validation_queries)} validation queries from training")
    
    # Generate training set (supports multi-threading)
    print("\n" + "=" * 80)
    print("TRAINING SET (QUERY-FIRST)")
    print("=" * 80)
    
    training_generator = QueryFirstDatasetGenerator(
        seed=args.seed,
        n_support=args.n_support,
        n_io=args.n_io,
        max_program_depth=args.max_program_depth,
        gold_grammar=DefaultGrammar,
        composer_name=args.composer,
        uniqueness_mode=uniqueness_mode,
        noise=args.noise,
        max_query_attempts=10,
        coverage_strength=args.coverage_strength,
        strict_uniqueness_episodes=args.strict_uniqueness_episodes,
        excluded_queries=validation_queries,  # Ensure no overlap with validation
        noise_annealing=args.noise_annealing,
        noise_start=args.noise_start,
        noise_end=args.noise_end,
        noise_anneal_episodes=args.noise_anneal_episodes,
        semantic_variations=args.semantic_variations,
        canonical_prob=args.canonical_prob
    )
    
    # Use parallel or single-threaded generation based on num_workers
    if args.num_workers > 1:
        training_generator.generate_dataset_parallel(
            n_episodes=args.n_train,
            output_dir=output_dir,
            split='train',
            first_is_identity=False,
            num_workers=args.num_workers
        )
    else:
        training_generator.generate_dataset(
            n_episodes=args.n_train,
            output_dir=output_dir,
            split='train',
            first_is_identity=False
        )
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE (QUERY-FIRST)")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Coverage strength: {args.coverage_strength}")
    print(f"Strict uniqueness episodes: {args.strict_uniqueness_episodes} (0 = always strict)")
    print(f"Validation queries excluded from training: {len(validation_queries)}")
    print(f"Training parallelization: {args.num_workers} worker(s)")
    if args.noise_annealing:
        print(f"Noise annealing: {args.noise_start:.2f} -> {args.noise_end:.2f} "
              f"over {args.noise_anneal_episodes} episodes")
    else:
        print(f"Noise: {args.noise:.2f} (fixed)")
    if args.semantic_variations:
        print(f"Semantic variations: ENABLED (canonical_prob={args.canonical_prob:.2f})")
        print(f"  - Each training episode has unique function semantics")
        print(f"  - Model must learn semantics from support examples")
        print(f"  - Validation uses canonical semantics (not affected)")
    else:
        print(f"Semantic variations: DISABLED (standard symbol shuffling only)")


if __name__ == '__main__':
    main()
