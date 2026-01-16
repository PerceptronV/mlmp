"""
Dataset Generator for Program Induction Meta-Learning

This module generates datasets of meta-learning episodes for program induction.
Each episode consists of:
1. A symbol mapping (shuffled function names for the episode)
2. Support examples (I/O pairs + program in both canonical and shuffled forms)
3. Query example (I/O pairs + target program in both canonical and shuffled forms)

All programs have type list[int] -> list[int], representing list transformations.
Programs are generated using the RuleSampler which follows Josh Rule's methodology
for selecting diverse I/O pairs.

Key constraints:
- Query programs only use functions that appear in support examples
- Symbol shuffling is applied as post-processing to canonical programs
- Both canonical (Rule DSL) and shuffled versions are stored
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

from ..lang.grammar import Grammar, DefaultGrammar
from ..lang.composers import get_composer, list_composers, Composer
from ..lang.samplers import RuleSampler, SampledProgram, UniquenessMode, create_list_to_list_type
from ..lang.ast_nodes import (
    ASTNode, pretty_print, LambdaNode, ApplicationNode,
    ListNode, NumberNode, VariableNode, IfNode, BooleanNode
)
from ..lang.evaluator import Evaluator


# ============================================================================
# AST Utilities
# ============================================================================


def apply_symbol_mapping(program_str: str, mapping: Dict[str, str]) -> str:
    """
    Apply symbol mapping to a program string.

    Replaces function names according to the mapping. Uses word boundaries
    to avoid partial replacements.

    Args:
        program_str: The canonical program string
        mapping: Dict mapping canonical names to shuffled names

    Returns:
        Program string with symbols replaced
    """
    result = program_str

    # Sort by length (longest first) to avoid partial replacements
    # e.g., replace "is_even" before "is"
    sorted_names = sorted(mapping.keys(), key=len, reverse=True)

    for canonical_name in sorted_names:
        shuffled_name = mapping[canonical_name]
        if canonical_name != shuffled_name:
            # Use word boundary regex to avoid partial matches
            # Match the name when it's followed by space, ), or end
            # and preceded by space, (, or start
            pattern = r'(?<=[(\s])' + re.escape(canonical_name) + r'(?=[\s)]|$)'
            result = re.sub(pattern, shuffled_name, result)

            # Also handle at start of expression (after open paren)
            pattern = r'\(' + re.escape(canonical_name) + r'(?=\s)'
            result = re.sub(pattern, '(' + shuffled_name, result)

    return result


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class IOPair:
    """Represents a single input-output pair."""
    input: List[int]
    output: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input': self.input,
            'output': self.output
        }


@dataclass
class PIExample:
    """
    Represents a program induction example.

    Contains I/O pairs and the program in both canonical and shuffled forms.
    """
    io_pairs: List[IOPair]
    program_canonical: str  # Program in Rule DSL (original function names)
    program_shuffled: str   # Program with shuffled function names
    functions_used: Set[str] = field(default_factory=set)  # Canonical function names used

    def to_dict(self) -> Dict[str, Any]:
        return {
            'io_pairs': [pair.to_dict() for pair in self.io_pairs],
            'program_canonical': self.program_canonical,
            'program_shuffled': self.program_shuffled,
            'functions_used': list(self.functions_used)
        }


@dataclass
class MetaLearningEpisode:
    """
    Represents a complete meta-learning episode.

    The symbol_mapping maps canonical (Rule DSL) names to shuffled names.
    Support and query examples contain both canonical and shuffled program forms.
    The query program is guaranteed to only use functions that appear in support.
    """
    episode_id: int
    symbol_mapping: Dict[str, str]  # canonical_name -> shuffled_name
    support_functions: Set[str]  # functions used in the episode
    support_functions_count: int  # number of support functions
    support_examples: List[PIExample]
    query: PIExample

    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'symbol_mapping': self.symbol_mapping,
            'support_functions': list(self.support_functions),
            'support_functions_count': self.support_functions_count,
            'support_examples': [ex.to_dict() for ex in self.support_examples],
            'query': self.query.to_dict()
        }


# ============================================================================
# Dataset Generator
# ============================================================================

class DatasetGenerator:
    """
    Dataset generator for meta-learning episodes.

    Uses RuleSampler to generate programs with good I/O pairs following
    Josh Rule's methodology. Ensures query programs only use functions
    that appear in support examples.
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
        noise_warmup_episodes: int = 0,
        max_noise: float = 0.9,
        strict_uniqueness_episodes: int = 0
    ):
        """
        Initialise the dataset generator.

        Args:
            seed: Random seed for reproducibility
            n_support: Number of support examples per episode
            n_io: Number of I/O pairs per program (Rule uses 11)
            max_program_depth: Maximum depth for generated programs
            gold_grammar: The gold grammar to use
            composer_name: Name of the composer to use
            uniqueness_mode: How to check program uniqueness
            noise: Base noise parameter for composers that support it
            max_query_attempts: Max attempts to find a novel query per episode
            noise_warmup_episodes: Number of episodes over which to warm up noise.
                                   If 0, use constant noise. Otherwise, noise
                                   increases linearly from base to max_noise.
            max_noise: Maximum noise value (reached after warmup_episodes)
            strict_uniqueness_episodes: Number of episodes with strict uniqueness.
                                        After this, accept queries if they appear
                                        less than average (allows controlled duplicates).
                                        If 0, always enforce strict uniqueness.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_support = n_support
        self.n_io = n_io
        self.max_program_depth = max_program_depth
        self.gold_grammar = gold_grammar
        self.composer_name = composer_name
        self.uniqueness_mode = uniqueness_mode
        self.base_noise = noise
        self.max_query_attempts = max_query_attempts
        self.noise_warmup_episodes = noise_warmup_episodes
        self.max_noise = max_noise
        self.strict_uniqueness_episodes = strict_uniqueness_episodes
        self._episodes_generated = 0

        # Create composer and sampler
        self.composer = get_composer(composer_name, seed, gold_grammar, noise=noise)
        self.sampler = RuleSampler(
            composer=self.composer,
            uniqueness_mode=uniqueness_mode,
            num_io_pairs=n_io,
            num_candidate_inputs=100,
            depth_variation=2,
            max_attempts_multiplier=100
        )

        # Get grammar function names for extraction
        self.grammar_names = set(gold_grammar.names)

        # Target type for all programs
        self.target_type = create_list_to_list_type()

        # Track query counts across episodes for diversity
        self._query_counts: Counter = Counter()

    def reset_seen_queries(self) -> None:
        """Reset query counts. Call between dataset splits."""
        self._query_counts.clear()
        self._episodes_generated = 0

    def _get_current_noise(self) -> float:
        """
        Compute the current noise level based on generation progress.

        Uses linear warmup from base_noise to max_noise over noise_warmup_episodes.
        If noise_warmup_episodes is 0, returns constant base_noise.
        """
        if self.noise_warmup_episodes <= 0:
            return self.base_noise

        progress = min(1.0, self._episodes_generated / self.noise_warmup_episodes)
        return self.base_noise + progress * (self.max_noise - self.base_noise)

    def _is_query_acceptable(self, program_str: str) -> bool:
        """
        Check if a query program is acceptable based on uniqueness constraints.

        During strict uniqueness phase (first strict_uniqueness_episodes):
            Only accept if program has never been seen.

        After strict uniqueness phase:
            Accept if program count is below average (allows controlled duplicates).
            This ensures no single program is overrepresented.
        """
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
        total_queries = sum(self._query_counts.values())
        average_count = total_queries / n_unique
        return current_count <= average_count

    def _generate_symbol_mapping(self, use_identity: bool = False) -> Dict[str, str]:
        """
        Generate a symbol mapping for the episode.

        Args:
            use_identity: If True, use identity mapping (no shuffling)

        Returns:
            Dict mapping canonical names to (possibly shuffled) names
        """
        names = list(self.grammar_names)

        if use_identity:
            return {name: name for name in names}

        shuffled = names.copy()
        self.rng.shuffle(shuffled)

        return dict(zip(names, shuffled))

    def _sampled_to_pi_example(
        self,
        sampled: SampledProgram,
        symbol_mapping: Dict[str, str]
    ) -> PIExample:
        """
        Convert a SampledProgram to a PIExample with both program forms.

        Args:
            sampled: The sampled program from RuleSampler
            symbol_mapping: The symbol mapping for this episode

        Returns:
            PIExample with canonical and shuffled program strings
        """
        # Convert I/O pairs
        io_pairs = [
            IOPair(input=inp, output=out)
            for inp, out in sampled.io_pairs
        ]

        # Get canonical program string
        program_canonical = sampled.program_str

        # Apply symbol mapping to get shuffled version
        program_shuffled = apply_symbol_mapping(program_canonical, symbol_mapping)

        # Extract functions used
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
        use_identity_mapping: bool = False
    ) -> Optional[MetaLearningEpisode]:
        """
        Generate a single meta-learning episode.

        The episode consists of:
        1. Support examples with programs and I/O pairs
        2. A query example whose program only uses functions from support
        3. Both canonical and shuffled program forms

        The query is generated using a restricted grammar containing only
        the functions that appear in support examples, guaranteeing that
        the query only uses seen functions.

        Args:
            episode_id: Unique identifier for this episode
            use_identity_mapping: If True, don't shuffle symbols

        Returns:
            MetaLearningEpisode, or None if generation fails
        """
        # Generate symbol mapping for this episode
        symbol_mapping = self._generate_symbol_mapping(use_identity_mapping)

        try:
            # Step 1: Generate support examples using full grammar
            sampled_support = self.sampler.sample(
                target_type=self.target_type,
                n=self.n_support,
                depth=self.max_program_depth
            )

            # Convert to PIExamples
            support_examples = [
                self._sampled_to_pi_example(prog, symbol_mapping)
                for prog in sampled_support
            ]

            # Step 2: Collect functions used in support
            support_functions: Set[str] = set()
            for ex in support_examples:
                support_functions.update(ex.functions_used)

            # Step 3: Create restricted grammar with only support functions
            restricted_grammar = self.gold_grammar.subset(support_functions)

            # Ensure we have enough functions for meaningful programs
            if len(restricted_grammar) < 2:
                return None

            # Step 4: Generate query using restricted grammar
            # Create composer/sampler once, then sample a batch of candidates
            support_program_strs = {ex.program_canonical for ex in support_examples}
            query_example = None

            query_seed = self.rng.randint(0, 2**31 - 1)
            current_noise = self._get_current_noise()
            restricted_composer = get_composer(
                self.composer_name,
                query_seed,
                restricted_grammar,
                noise=current_noise
            )

            restricted_sampler = RuleSampler(
                composer=restricted_composer,
                uniqueness_mode=self.uniqueness_mode,
                num_io_pairs=self.n_io,
                num_candidate_inputs=100,
                depth_variation=2,
                max_attempts_multiplier=50
            )

            # Sample queries until we find a novel one
            # Use small batches for efficiency (sampler needs multiple attempts internally)
            try:
                candidates = restricted_sampler.sample(
                    target_type=self.target_type,
                    n=self.max_query_attempts,
                    depth=self.max_program_depth
                )

                for candidate in candidates:
                    pi_example = self._sampled_to_pi_example(candidate, symbol_mapping)

                    # Check: different from support programs
                    if pi_example.program_canonical in support_program_strs:
                        continue

                    # Check: acceptable based on uniqueness constraints
                    if not self._is_query_acceptable(pi_example.program_canonical):
                        continue

                    # Found an acceptable query
                    query_example = pi_example
                    self._query_counts[pi_example.program_canonical] += 1
                    break

            except ValueError:
                # Sampling failed
                pass

            if query_example is None:
                return None

            # Track progress for noise warmup
            self._episodes_generated += 1

            return MetaLearningEpisode(
                episode_id=episode_id,
                symbol_mapping=symbol_mapping,
                support_functions=support_functions,
                support_functions_count=len(support_functions),
                support_examples=support_examples,
                query=query_example
            )

        except (ValueError, KeyError):
            # Sampling failed or restricted grammar issues
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

        print(f"Generating {n_episodes} {split} episodes...")
        print(f"Output directory: {output_dir}")
        print(f"Parameters: n_support={self.n_support}, n_io={self.n_io}, "
              f"max_depth={self.max_program_depth}")
        print(f"Composer: {self.composer_name}, Uniqueness: {self.uniqueness_mode.value}")

        successful = 0
        attempts = 0
        max_attempts = n_episodes * 10

        while successful < n_episodes and attempts < max_attempts:
            # First episode in eval uses identity mapping
            use_identity = first_is_identity and split == 'eval' and successful == 0

            episode = self.generate_episode(
                episode_id=successful,
                use_identity_mapping=use_identity
            )

            attempts += 1

            if episode is not None:
                # Save episode to JSON
                output_path = output_dir / f"episode_{successful:06d}.json"
                with open(output_path, 'w') as f:
                    json.dump(episode.to_dict(), f, indent=2)

                successful += 1

                if successful % 100 == 0:
                    print(f"Generated {successful}/{n_episodes} episodes "
                          f"({attempts} attempts)")

        if successful < n_episodes:
            print(f"Warning: Only generated {successful}/{n_episodes} episodes "
                  f"after {attempts} attempts")
        else:
            print(f"Successfully generated {n_episodes} {split} episodes!")


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description='Generate program induction meta-learning dataset'
    )

    # Dataset size parameters
    parser.add_argument(
        '--n-train',
        type=int,
        default=10000,
        help='Number of training episodes (default: 10,000)'
    )
    parser.add_argument(
        '--n-eval',
        type=int,
        default=100,
        help='Number of evaluation episodes (default: 100)'
    )

    # Episode structure parameters
    parser.add_argument(
        '--n-support',
        type=int,
        default=30,
        help='Number of support examples per episode (default: 4)'
    )
    parser.add_argument(
        '--n-io',
        type=int,
        default=11,
        help='Number of I/O pairs per program (default: 11, per Rule paper)'
    )

    # Program generation parameters
    parser.add_argument(
        '--max-program-depth',
        type=int,
        default=4,
        help='Maximum depth for generated programs (default: 4)'
    )
    parser.add_argument(
        '--composer',
        type=str,
        default='template',
        choices=list_composers(),
        help=f'Composer for program generation (choices: {", ".join(list_composers())}; default: template)'
    )
    parser.add_argument(
        '--uniqueness',
        type=str,
        default='string',
        choices=['string', 'behavioral'],
        help='Uniqueness mode: string (fast) or behavioral (thorough) (default: string)'
    )

    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets',
        help='Output directory for datasets (default: datasets/)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.3,
        help='Base noise parameter for composers (default: 0.3)'
    )
    parser.add_argument(
        '--max-noise',
        type=float,
        default=0.9,
        help='Maximum noise after warmup (default: 0.9)'
    )
    parser.add_argument(
        '--noise-warmup-episodes',
        type=int,
        default=0,
        help='Episodes over which to warm up noise from base to max. '
             '0 = constant noise (default: 0)'
    )
    parser.add_argument(
        '--strict-uniqueness-episodes',
        type=int,
        default=0,
        help='Episodes with strict uniqueness before allowing controlled duplicates. '
             '0 = always strict (default: 0)'
    )
    parser.add_argument(
        '--max-query-attempts',
        type=int,
        default=10,
        help='Max attempts to find a novel query per episode (default: 10)'
    )

    args = parser.parse_args()

    # Determine uniqueness mode
    uniqueness_mode = (
        UniquenessMode.BEHAVIORAL if args.uniqueness == 'behavioral'
        else UniquenessMode.STRING
    )

    # Set up output directory
    output_dir = Path(args.output_dir) / f"{args.composer}_seed{args.seed}"

    # Create dataset generator
    generator = DatasetGenerator(
        seed=args.seed,
        n_support=args.n_support,
        n_io=args.n_io,
        max_program_depth=args.max_program_depth,
        gold_grammar=DefaultGrammar,
        composer_name=args.composer,
        uniqueness_mode=uniqueness_mode,
        noise=args.noise,
        max_query_attempts=args.max_query_attempts,
        noise_warmup_episodes=args.noise_warmup_episodes,
        max_noise=args.max_noise,
        strict_uniqueness_episodes=args.strict_uniqueness_episodes
    )

    # Generate training set
    print("\n" + "=" * 80)
    print("TRAINING SET")
    print("=" * 80)
    generator.generate_dataset(
        n_episodes=args.n_train,
        output_dir=output_dir,
        split='train',
        first_is_identity=False
    )

    # Generate evaluation set
    # Reset seen queries so eval can have independent programs
    generator.reset_seen_queries()
    print("\n" + "=" * 80)
    print("EVALUATION SET")
    print("=" * 80)
    generator.generate_dataset(
        n_episodes=args.n_eval,
        output_dir=output_dir,
        split='eval',
        first_is_identity=True
    )

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Composer: {args.composer}")
    print(f"Uniqueness mode: {args.uniqueness}")
    print(f"Output directory: {output_dir}")
    print(f"Training episodes: {args.n_train}")
    print(f"Evaluation episodes: {args.n_eval}")
    print(f"Noise: {args.noise} -> {args.max_noise} (warmup: {args.noise_warmup_episodes} episodes)")
    print(f"Strict uniqueness: {args.strict_uniqueness_episodes} episodes (0 = always strict)")
    print(f"First eval episode uses identity mapping (no symbol shuffling)")


if __name__ == '__main__':
    main()
