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
from ..lang.parser import parse
from ..lang.compiler import JITCompiler


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
    # Use a two-pass approach with placeholders to avoid double-replacement
    # when canonical names appear as shuffled names (e.g., cons->take, take->foldi)
    result = program_str

    # Sort by length (longest first) to avoid partial replacements
    # e.g., replace "is_even" before "is"
    sorted_names = sorted(mapping.keys(), key=len, reverse=True)

    # Pass 1: Replace canonical names with unique placeholders
    placeholders = {}
    for idx, canonical_name in enumerate(sorted_names):
        shuffled_name = mapping[canonical_name]
        if canonical_name != shuffled_name:
            placeholder = f"__PLACEHOLDER_{idx}__"
            placeholders[placeholder] = shuffled_name
            
            # Use word boundary regex to avoid partial matches
            # Match the name when it's followed by space, ), or end
            # and preceded by space, (, or start
            pattern = r'(?<=[(\s])' + re.escape(canonical_name) + r'(?=[\s)]|$)'
            result = re.sub(pattern, placeholder, result)

            # Also handle at start of expression (after open paren)
            pattern = r'\(' + re.escape(canonical_name) + r'(?=\s)'
            result = re.sub(pattern, '(' + placeholder, result)

    # Pass 2: Replace placeholders with shuffled names
    for placeholder, shuffled_name in placeholders.items():
        result = result.replace(placeholder, shuffled_name)

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
        return current_count < average_count

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


# ============================================================================
# Validation Dataset Generator
# ============================================================================

class ValidationDatasetGenerator:
    """
    Validation dataset generator using canonical programs from functions.txt.

    Generates episodes where:
    - Support examples are randomly generated
    - Query programs come from a predefined canonical set
    - Query only uses functions that appear in support examples
    - No symbol shuffling (identity mapping only)
    """

    def __init__(
        self,
        seed: int,
        canonical_programs_path: Path,
        n_support: int = 4,
        n_io: int = 11,
        max_program_depth: int = 4,
        gold_grammar: Grammar = DefaultGrammar,
        composer_name: str = 'random',
        uniqueness_mode: UniquenessMode = UniquenessMode.STRING
    ):
        """
        Initialize the validation dataset generator.

        Args:
            seed: Random seed for reproducibility
            canonical_programs_path: Path to functions.txt with canonical programs
            n_support: Number of support examples per episode
            n_io: Number of I/O pairs per program
            max_program_depth: Maximum depth for generated support programs
            gold_grammar: The gold grammar to use
            composer_name: Name of the composer for support generation
            uniqueness_mode: How to check program uniqueness
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_support = n_support
        self.n_io = n_io
        self.max_program_depth = max_program_depth
        self.gold_grammar = gold_grammar
        self.composer_name = composer_name
        self.uniqueness_mode = uniqueness_mode

        # Load canonical programs
        self.canonical_programs = self._load_canonical_programs(canonical_programs_path)
        print(f"Loaded {len(self.canonical_programs)} canonical programs")

        # Create composer and sampler for support examples
        self.composer = get_composer(composer_name, seed, gold_grammar)
        self.sampler = RuleSampler(
            composer=self.composer,
            uniqueness_mode=uniqueness_mode,
            num_io_pairs=n_io,
            num_candidate_inputs=100,
            depth_variation=2,
            max_attempts_multiplier=100
        )

        # Create compiler for generating I/O pairs
        self.compiler = JITCompiler(gold_grammar)

        # Get grammar function names for extraction
        self.grammar_names = set(gold_grammar.names)

        # Target type for all programs
        self.target_type = create_list_to_list_type()

        # Track which canonical programs have been used as queries
        self._used_query_indices: Set[int] = set()

    def reset_used_queries(self) -> None:
        """Reset tracking of used queries. Call between dataset splits."""
        self._used_query_indices.clear()

    def _load_canonical_programs(self, path: Path) -> List[str]:
        """
        Load canonical programs from file.

        Args:
            path: Path to functions.txt

        Returns:
            List of program strings
        """
        programs = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    programs.append(line)
        return programs

    def _extract_functions_from_program_str(self, program_str: str) -> Set[str]:
        """
        Extract grammar function names from a program string.

        Uses regex to find function names that match the grammar.

        Args:
            program_str: The program as a string

        Returns:
            Set of function names used in the program
        """
        found_functions = set()

        for name in self.grammar_names:
            # Use word boundary regex to match function names
            # Need to handle Lisp syntax: functions appear after '(' or whitespace
            pattern = r'(?<=\()\s*' + re.escape(name) + r'\s+|(?<=\s)' + re.escape(name) + r'(?=\s|\))'
            if re.search(pattern, program_str):
                found_functions.add(name)

        return found_functions

    def _generate_io_pairs(self, program_node: ASTNode, n: int) -> List[IOPair]:
        """
        Generate I/O pairs by compiling and executing program on random inputs.

        Compiles the program to a native function, then generates random list
        inputs and evaluates the compiled function on them.
        Uses similar methodology to Rule's approach.

        Args:
            program_node: The parsed program AST
            n: Number of I/O pairs to generate

        Returns:
            List of IOPair objects
        """
        io_pairs = []
        attempts = 0
        max_attempts = n * 100  # Allow many attempts to find valid I/O pairs

        # Track seen inputs and outputs for diversity
        seen_inputs = set()
        seen_outputs = set()

        # Compile the program to a native function
        try:
            program_func = self.compiler.compile(program_node)
        except Exception as e:
            print(f"Failed to compile program: {e}")
            return []

        while len(io_pairs) < n and attempts < max_attempts:
            # Generate random input list
            # Vary length and values to get diverse tests
            list_length = self.rng.randint(0, 15)
            input_list = [self.rng.randint(-10, 99) for _ in range(list_length)]

            # Skip if we've seen this input
            input_tuple = tuple(input_list)
            if input_tuple in seen_inputs:
                attempts += 1
                continue

            try:
                # Apply the compiled function to the input
                output = program_func(input_list)

                # Check output is a list
                if isinstance(output, list):
                    # Check for diversity (avoid duplicate outputs if possible)
                    output_tuple = tuple(output)

                    # Accept if output is novel, or if we're struggling to find pairs
                    if output_tuple not in seen_outputs or len(io_pairs) < n // 2:
                        io_pairs.append(IOPair(input=input_list, output=output))
                        seen_inputs.add(input_tuple)
                        seen_outputs.add(output_tuple)
            except Exception:
                # Evaluation failed, skip this input
                pass

            attempts += 1

        return io_pairs

    def _sampled_to_pi_example(
        self,
        sampled_program: SampledProgram,
        symbol_mapping: Dict[str, str]
    ) -> PIExample:
        """
        Convert a sampled program to a PIExample.

        For validation, symbol_mapping should always be identity.

        Args:
            sampled_program: SampledProgram from RuleSampler
            symbol_mapping: Symbol mapping (identity for validation)

        Returns:
            PIExample with canonical and shuffled forms
        """
        # Convert I/O pairs
        io_pairs = [
            IOPair(input=inp, output=out)
            for inp, out in sampled_program.io_pairs
        ]

        # Get canonical program string
        program_canonical = sampled_program.program_str

        # Apply symbol mapping (should be identity for validation)
        program_shuffled = apply_symbol_mapping(program_canonical, symbol_mapping)

        # Extract functions used
        functions_used = sampled_program.program.function_names() & self.grammar_names

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
        Generate a single validation episode.

        The episode consists of:
        1. Support examples generated randomly
        2. A query selected from canonical programs that only uses support functions
        3. Identity mapping (no symbol shuffling)

        Args:
            episode_id: Unique identifier for this episode
            allow_query_reuse: If True, allow reusing canonical programs as queries

        Returns:
            MetaLearningEpisode, or None if no valid query found
        """
        # Identity mapping (no shuffling for validation)
        identity_mapping = {name: name for name in self.grammar_names}

        try:
            # Step 1: Generate support examples using full grammar
            sampled_support = self.sampler.sample(
                target_type=self.target_type,
                n=self.n_support,
                depth=self.max_program_depth
            )

            # Convert to PIExamples
            support_examples = [
                self._sampled_to_pi_example(prog, identity_mapping)
                for prog in sampled_support
            ]

            # Step 2: Collect functions used in support
            support_functions: Set[str] = set()
            for ex in support_examples:
                support_functions.update(ex.functions_used)

            # Step 3: Collect support program strings
            support_program_strs = {ex.program_canonical for ex in support_examples}

            # Step 4: Find valid query programs from canonical set
            valid_query_indices = []

            for idx, prog_str in enumerate(self.canonical_programs):
                # Skip if already used (unless reuse is allowed)
                if not allow_query_reuse and idx in self._used_query_indices:
                    continue

                # Extract functions from this program
                prog_functions = self._extract_functions_from_program_str(prog_str)

                # Check if only uses support functions
                if not prog_functions.issubset(support_functions):
                    continue

                # Check if different from support programs
                if prog_str in support_program_strs:
                    continue

                # Valid query candidate
                valid_query_indices.append(idx)

            # No valid queries found
            if not valid_query_indices:
                return None

            # Step 5: Select a random valid query
            query_idx = self.rng.choice(valid_query_indices)
            query_str = self.canonical_programs[query_idx]

            # Mark as used
            self._used_query_indices.add(query_idx)

            # Step 6: Parse query program and generate I/O pairs
            try:
                query_node = parse(query_str)
            except Exception as e:
                print(f"Failed to parse canonical program {query_idx}: {e}")
                # Unmark as used since we failed
                self._used_query_indices.discard(query_idx)
                return None

            # Generate I/O pairs for query
            query_io_pairs = self._generate_io_pairs(query_node, self.n_io)

            if len(query_io_pairs) < self.n_io:
                print(f"Warning: Only generated {len(query_io_pairs)}/{self.n_io} "
                      f"I/O pairs for query {query_idx}")
                # Accept with fewer I/O pairs if we got at least some
                if len(query_io_pairs) == 0:
                    # Unmark as used since we failed
                    self._used_query_indices.discard(query_idx)
                    return None

            # Extract functions from query
            query_functions = self._extract_functions_from_program_str(query_str)

            # Create query example
            query_example = PIExample(
                io_pairs=query_io_pairs,
                program_canonical=query_str,
                program_shuffled=query_str,  # No shuffling
                functions_used=query_functions
            )

            return MetaLearningEpisode(
                episode_id=episode_id,
                symbol_mapping=identity_mapping,
                support_functions=support_functions,
                support_functions_count=len(support_functions),
                support_examples=support_examples,
                query=query_example
            )

        except (ValueError, KeyError) as e:
            # Sampling failed
            print(f"Episode generation failed: {e}")
            return None

    def generate_dataset(
        self,
        n_episodes: int,
        output_dir: Path,
        split: str = 'validation',
        allow_query_reuse: bool = False
    ):
        """
        Generate a validation dataset.

        Args:
            n_episodes: Number of episodes to generate
            output_dir: Directory to save the episodes
            split: Dataset split name (default: 'validation')
            allow_query_reuse: If True, allow reusing canonical programs
        """
        output_dir = Path(output_dir) / split
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {n_episodes} {split} episodes...")
        print(f"Output directory: {output_dir}")
        print(f"Parameters: n_support={self.n_support}, n_io={self.n_io}, "
              f"max_depth={self.max_program_depth}")
        print(f"Composer: {self.composer_name}, Uniqueness: {self.uniqueness_mode.value}")
        print(f"Using canonical programs from functions.txt (no shuffling)")

        successful = 0
        attempts = 0
        max_attempts = n_episodes * 20  # Allow more attempts since we need matching

        while successful < n_episodes and attempts < max_attempts:
            episode = self.generate_episode(
                episode_id=successful,
                allow_query_reuse=allow_query_reuse
            )

            attempts += 1

            if episode is not None:
                # Save episode to JSON
                output_path = output_dir / f"episode_{successful:06d}.json"
                with open(output_path, 'w') as f:
                    json.dump(episode.to_dict(), f, indent=2)

                successful += 1

                if successful % 10 == 0:
                    print(f"Generated {successful}/{n_episodes} episodes "
                          f"({attempts} attempts)")

        if successful < n_episodes:
            print(f"Warning: Only generated {successful}/{n_episodes} episodes "
                  f"after {attempts} attempts")
            print(f"This may happen if canonical programs don't match support functions frequently")
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
    parser.add_argument(
        '--n-validation',
        type=int,
        default=200,
        help='Number of validation episodes (default: 200)'
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
        default=6,
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
    parser.add_argument(
        '--canonical-programs',
        type=str,
        default='src/data/rule/functions.txt',
        help='Path to canonical programs for validation (default: src/data/rule/functions.txt)'
    )

    args = parser.parse_args()

    # Determine uniqueness mode
    uniqueness_mode = (
        UniquenessMode.BEHAVIORAL if args.uniqueness == 'behavioral'
        else UniquenessMode.STRING
    )

    # Set up output directory
    output_dir = Path(args.output_dir) / f"{args.composer}_seed{args.seed}"

    # Generate validation set using canonical programs
    print("\n" + "=" * 80)
    print("VALIDATION SET")
    print("=" * 80)

    validation_generator = ValidationDatasetGenerator(
        seed=args.seed,
        canonical_programs_path=Path(args.canonical_programs),
        n_support=args.n_support,
        n_io=args.n_io,
        max_program_depth=args.max_program_depth,
        gold_grammar=DefaultGrammar,
        composer_name=args.composer,
        uniqueness_mode=uniqueness_mode
    )

    validation_generator.generate_dataset(
        n_episodes=args.n_validation,
        output_dir=output_dir,
        split='validation',
        allow_query_reuse=False
    )

    # Create training dataset generator
    training_generator = DatasetGenerator(
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
    training_generator.generate_dataset(
        n_episodes=args.n_train,
        output_dir=output_dir,
        split='train',
        first_is_identity=False
    )

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Composer: {args.composer}")
    print(f"Uniqueness mode: {args.uniqueness}")
    print(f"Output directory: {output_dir}")
    print(f"Training episodes: {args.n_train}")
    print(f"Evaluation episodes: {args.n_eval}")
    print(f"Validation episodes: {args.n_validation} (from canonical programs)")
    print(f"Noise: {args.noise} -> {args.max_noise} (warmup: {args.noise_warmup_episodes} episodes)")
    print(f"Strict uniqueness: {args.strict_uniqueness_episodes} episodes (0 = always strict)")
    print(f"First eval episode uses identity mapping (no symbol shuffling)")
    print(f"Validation uses canonical programs from {args.canonical_programs} (no shuffling)")


if __name__ == '__main__':
    main()
