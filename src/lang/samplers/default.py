"""
Default Program Sampler

This module implements the DefaultSampler which generates batches of programs
with the following strategies:
- Changes seed for each program to ensure diversity
- Varies program depth around the base depth
- Ensures uniqueness by rejecting duplicate programs
"""

from typing import Optional
import random

from .base import Sampler
from ..composers.base import Composer
from ..ast_nodes import ASTNode, pretty_print
from ..type_utils import TypeType, SubstitutionTable


class DefaultSampler(Sampler):
    """
    Default sampler that ensures program diversity through seed variation,
    depth variation, and uniqueness checking.

    Features:
    - Changes seed for each attempt to generate different programs
    - Varies depth within a configurable range around the base depth
    - Tracks generated programs to reject duplicates
    - Continues sampling until n unique programs are generated
    """

    def __init__(
        self,
        composer: Composer,
        depth_variation: int = 2,
        max_attempts_multiplier: int = 100
    ):
        """
        Initialize the default sampler.

        Args:
            composer: The composer to use for program generation
            depth_variation: Maximum depth variation from base depth (±depth_variation).
                           For example, if depth=5 and depth_variation=2, depths will
                           range from 3 to 7.
            max_attempts_multiplier: Maximum attempts = n * max_attempts_multiplier.
                                    After this many attempts, sampling gives up.
        """
        super().__init__(composer)
        self.depth_variation = depth_variation
        self.max_attempts_multiplier = max_attempts_multiplier
        # Track the base seed for reproducibility
        self.base_seed = composer.rng.getstate()

    def sample(
        self,
        target_type: TypeType,
        n: int,
        depth: int,
        context: Optional[dict[str, TypeType]] = None
    ) -> list[ASTNode]:
        """
        Sample a batch of n unique programs.

        This method generates programs by:
        1. Varying the seed for each attempt
        2. Randomly selecting depth in range [depth - variation, depth + variation]
        3. Generating a program using the composer
        4. Checking if the program is unique (not already generated)
        5. If unique, adding it to the batch; otherwise, retrying

        Args:
            target_type: The desired output type for all programs
            n: Number of unique programs to generate
            depth: Base depth for program generation
            context: Variable bindings in scope (name -> type)

        Returns:
            List of n unique AST nodes, each of the target type

        Raises:
            ValueError: If unable to generate n unique programs after max_attempts
        """
        if n <= 0:
            return []

        if context is None:
            context = {}

        programs: list[ASTNode] = []
        seen_programs: set[str] = set()

        # Create a separate RNG for depth and seed selection
        # This ensures reproducibility independent of the composer's RNG
        sampler_rng = random.Random(hash(tuple(self.composer.rng.getstate()[1])))

        max_attempts = n * self.max_attempts_multiplier
        attempts = 0
        seed_offset = 0

        while len(programs) < n and attempts < max_attempts:
            attempts += 1

            # Select a depth randomly within the variation range
            min_depth = max(0, depth - self.depth_variation)
            max_depth = depth + self.depth_variation
            current_depth = sampler_rng.randint(min_depth, max_depth)

            # Create a new seed for this attempt
            # We use the composer's original seed plus an offset to ensure
            # deterministic but varied generation
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

                # Convert to string representation for uniqueness checking
                program_str = pretty_print(program, inline=True)

                # Check if this program is unique
                if program_str not in seen_programs:
                    programs.append(program)
                    seen_programs.add(program_str)

            except Exception as e:
                # If generation fails (e.g., type error, impossible constraints),
                # just try again with a different seed
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

        return programs

    def sample_with_varied_depths(
        self,
        target_type: TypeType,
        n: int,
        depths: list[int],
        context: Optional[dict[str, TypeType]] = None
    ) -> list[tuple[ASTNode, int]]:
        """
        Sample programs with explicitly specified depths.

        This is useful when you want more control over the depth distribution.
        Unlike the main sample() method which randomly varies depth, this method
        generates programs at the exact depths specified.

        Args:
            target_type: The desired output type for all programs
            n: Number of unique programs to generate
            depths: List of depths to cycle through (length should equal n, or will cycle)
            context: Variable bindings in scope (name -> type)

        Returns:
            List of (program, depth) tuples, one for each unique program

        Raises:
            ValueError: If unable to generate n unique programs after max_attempts
        """
        if n <= 0:
            return []

        if not depths:
            raise ValueError("depths list cannot be empty")

        if context is None:
            context = {}

        programs: list[tuple[ASTNode, int]] = []
        seen_programs: set[str] = set()

        # Create a separate RNG for seed selection
        sampler_rng = random.Random(hash(tuple(self.composer.rng.getstate()[1])))

        max_attempts = n * self.max_attempts_multiplier
        attempts = 0
        seed_offset = 0
        depth_index = 0

        while len(programs) < n and attempts < max_attempts:
            attempts += 1

            # Select depth from the provided list (cycling if necessary)
            current_depth = depths[depth_index % len(depths)]
            depth_index += 1

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

                # Convert to string representation for uniqueness checking
                program_str = pretty_print(program, inline=True)

                # Check if this program is unique
                if program_str not in seen_programs:
                    programs.append((program, current_depth))
                    seen_programs.add(program_str)

            except Exception as e:
                # If generation fails, try again with a different seed
                pass
            finally:
                # Restore the composer's original state
                self.composer.rng.setstate(original_state)

        if len(programs) < n:
            raise ValueError(
                f"Could not generate {n} unique programs after {attempts} attempts. "
                f"Only generated {len(programs)} unique programs. "
                f"Try adjusting depths or increasing max_attempts_multiplier."
            )

        return programs
