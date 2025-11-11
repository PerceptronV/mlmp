#!/usr/bin/env python3
"""
Dataset generation script for list-to-list function tasks.

This script generates n distinct tasks, where each task consists of:
- A randomly sampled list-to-list function (program)
- 5 input-output examples (using random list inputs)
- String representation of the program

Each task is saved as a JSON file with the following structure:
{
    "program": "<string representation of the program>",
    "examples": [
        {"input": [list], "output": [list]},
        ...
    ]
}
"""

import argparse
import json
import random
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.composer import sample_program
from lang.type_system import list_of, func, INT
from lang.ast_nodes import pretty_print, ListNode
from lang.evaluator import Evaluator


def generate_random_list(rng: random.Random, max_len: int = 10, max_val: int = 100) -> List[int]:
    """
    Generate a random list of integers.

    Args:
        rng: Random number generator
        max_len: Maximum length of the list
        max_val: Maximum value for list elements

    Returns:
        A list of random integers
    """
    length = rng.randint(0, max_len)
    return [rng.randint(0, max_val) for _ in range(length)]


def evaluate_program_on_input(
    evaluator: Evaluator,
    program_ast,
    input_list: List[int]
) -> Any:
    """
    Evaluate a program (which should be a list-to-list function) on an input list.

    Args:
        evaluator: The evaluator instance
        program_ast: The AST node representing the program
        input_list: The input list to apply the program to

    Returns:
        The result of applying the program to the input list

    Raises:
        Exception: If evaluation fails
    """
    try:
        from lang.environment import Closure

        # Evaluate the program to get a function
        func_value = evaluator.eval(program_ast)

        # Check if we got a closure (function)
        if not isinstance(func_value, Closure):
            raise Exception(f"Program did not evaluate to a function, got {type(func_value)}")

        # Apply the function to the input list (as a raw Python list)
        result = evaluator._apply(func_value, [input_list], evaluator.global_env)

        # The result should be a list of integers
        if isinstance(result, list) and all(isinstance(x, int) for x in result):
            return result
        else:
            raise Exception(f"Program did not return a list of integers, got {type(result)}: {result}")
    except Exception as e:
        # Re-raise with context
        raise Exception(f"Failed to evaluate program on input {input_list}: {str(e)}")


def generate_examples(
    evaluator: Evaluator,
    program_ast,
    rng: random.Random,
    num_examples: int = 5,
    max_list_len: int = 10,
    max_val: int = 100
) -> List[Dict[str, List[int]]]:
    """
    Generate input-output examples for a program.

    Args:
        evaluator: The evaluator instance
        program_ast: The AST node representing the program
        rng: Random number generator
        num_examples: Number of examples to generate
        max_list_len: Maximum length of generated lists
        max_val: Maximum value for list elements

    Returns:
        A list of {input, output} dictionaries with unique inputs

    Raises:
        Exception: If unable to generate enough valid examples
    """
    examples = []
    seen_inputs = set()
    attempts = 0
    max_attempts = num_examples * 20  # Allow multiple attempts per example

    while len(examples) < num_examples and attempts < max_attempts:
        attempts += 1
        try:
            input_list = generate_random_list(rng, max_list_len, max_val)

            # Convert to tuple for hashing
            input_tuple = tuple(input_list)

            # Skip if we've already seen this input
            if input_tuple in seen_inputs:
                continue

            output = evaluate_program_on_input(evaluator, program_ast, input_list)

            # Ensure output is a list of integers
            if isinstance(output, list) and all(isinstance(x, int) for x in output):
                examples.append({
                    "input": input_list,
                    "output": output
                })
                seen_inputs.add(input_tuple)
        except Exception:
            # Skip this example and try again
            continue

    if len(examples) < num_examples:
        raise Exception(
            f"Could only generate {len(examples)}/{num_examples} valid examples "
            f"after {max_attempts} attempts"
        )

    return examples


def generate_task(
    task_id: int,
    evaluator: Evaluator,
    rng: random.Random,
    max_depth: int,
    seen_programs: set,
    program_attempt: int = 0
) -> Dict[str, Any]:
    """
    Generate a single task: a program and its input-output examples.

    Args:
        task_id: Identifier for this task
        evaluator: The evaluator instance
        rng: Random number generator
        max_depth: Maximum depth for program generation
        seen_programs: Set of previously generated program strings (for uniqueness)
        program_attempt: Attempt number for this task (for retries)

    Returns:
        A task dictionary with 'program' and 'examples' keys

    Raises:
        Exception: If task generation fails
    """
    # Sample a list-to-list function
    # Target type: [Int] -> [Int]
    target_type = func(list_of(INT), list_of(INT))

    # Use a seed based on task_id and attempt number
    program_seed = task_id * 10000 + program_attempt
    program_ast = sample_program(seed=program_seed, max_depth=max_depth, target_type=target_type)

    # Get program string
    program_str = pretty_print(program_ast)

    # Check for uniqueness
    if program_str in seen_programs:
        raise Exception(f"Generated duplicate program: {program_str}")

    # Generate examples
    examples = generate_examples(evaluator, program_ast, rng, num_examples=5)

    # Create task dictionary
    task = {
        "program": program_str,
        "examples": examples
    }

    # Mark this program as seen
    seen_programs.add(program_str)

    return task


def main():
    """Main entry point for the dataset generation script."""
    parser = argparse.ArgumentParser(
        description="Generate a dataset of list-to-list function tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="datasets",
        help="Output directory for the generated dataset"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000000,
        help="Number of tasks to generate"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=15,
        help="Maximum depth of sampled programs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Convert n to int
    n = int(args.n)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RNG
    rng = random.Random(args.seed)

    # Initialize evaluator
    evaluator = Evaluator()

    print(f"Generating {n} tasks with max_depth={args.max_depth}, seed={args.seed}")
    print(f"Output directory: {out_dir}")

    # Track statistics
    successful = 0
    seen_programs = set()
    max_retries = 10  # Maximum retry attempts per task

    task_id = 0
    while successful < n:
        attempt = 0
        last_error = None

        # Retry loop: try up to max_retries times to generate a valid task
        while attempt < max_retries:
            try:
                # Generate task
                task = generate_task(task_id, evaluator, rng, args.max_depth, seen_programs, attempt)

                # Save to JSON file
                task_file = out_dir / f"task_{successful:08d}.json"
                with open(task_file, 'w') as f:
                    json.dump(task, f, indent=2)

                successful += 1

                # Print progress every 1000 tasks
                if successful % 1000 == 0:
                    print(f"Generated {successful}/{n} tasks ({len(seen_programs)} unique programs)")

                # Break out of retry loop on success
                break

            except Exception as e:
                last_error = str(e)
                attempt += 1
                # Continue to next attempt

        # If we exhausted retries for this task, report error
        if attempt >= max_retries:
            print(
                f"Failed to generate task (after {max_retries} retries): {last_error}",
                file=sys.stderr
            )

        task_id += 1

    print(f"\nDataset generation complete!")
    print(f"Successfully generated: {successful} tasks")
    print(f"Unique programs: {len(seen_programs)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()