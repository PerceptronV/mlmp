#!/usr/bin/env python3
"""
Example script showing how to use the random program generator.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import sample_program
from lang.type_system import INT, BOOL, list_of, func
from lang.ast_nodes import pretty_print
from lang.type_checker import TypeChecker
from lang.evaluator import Evaluator


def main():
    """Generate and display various random programs."""

    print("=" * 70)
    print("Random Typed Program Generator Examples")
    print("=" * 70)

    # Create type checker and evaluator
    checker = TypeChecker()
    evaluator = Evaluator()

    examples = [
        ("Integer Expression", 42, 3, INT),
        ("Boolean Expression", 100, 2, BOOL),
        ("List Expression", 200, 3, list_of(INT)),
        ("Function (Int → Int)", 300, 4, func(INT, INT)),
        ("Function (Int → Bool)", 400, 3, func(INT, BOOL)),
        ("Function ([Int] → Int)", 500, 3, func(list_of(INT), INT)),
    ]

    for name, seed, depth, target_type in examples:
        print(f"\n{name}")
        print("-" * 70)

        # Generate program
        ast = sample_program(seed=seed, max_depth=depth, target_type=target_type)

        # Print the program
        print("Program:")
        print(pretty_print(ast))

        # Type check
        inferred_type = checker.check_program(ast)
        print(f"\nInferred Type: {inferred_type}")

        # Evaluate (for non-function types)
        if not str(inferred_type).count("→"):
            try:
                result = evaluator.eval(ast)
                print(f"Evaluated Result: {result}")
            except Exception as e:
                print(f"Evaluation Error: {e}")
        else:
            print("(Function - not evaluated)")

    # Generate multiple programs at different depths
    print("\n" + "=" * 70)
    print("Programs at Different Depths (seed=999, type=Int)")
    print("=" * 70)

    for depth in range(6):
        ast = sample_program(seed=999, max_depth=depth, target_type=INT)
        print(f"\nDepth {depth}: {pretty_print(ast)}")


if __name__ == "__main__":
    main()