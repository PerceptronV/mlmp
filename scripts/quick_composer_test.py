#!/usr/bin/env python3
"""
Quick Composer Test Script

A simpler script for quickly testing individual composers and seeing
their output with detailed examples.

Usage:
    python scripts/quick_composer_test.py random
    python scripts/quick_composer_test.py template
    python scripts/quick_composer_test.py mcts
"""

import sys
import os
from typing import Callable

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lang.grammar import DefaultGrammar
from lang.composers.random import RandomComposer
from lang.composers.template import TemplateComposer
from lang.composers.mcts import train_mcts_composer
from lang.composers.random_guarded import RandomGuardedComposer
from lang.composers.empirical import EmpiricalComposer
from lang.ast_nodes import pretty_print
from lang.type_utils import SubstitutionTable
from lang.compiler import JITCompiler


def test_composer(composer_name: str, num_examples: int = 10):
    """Test a composer and show detailed examples."""
    
    target_type = Callable[[list[int]], list[int]]
    seed = 42
    
    print(f"\n{'='*70}")
    print(f"Testing {composer_name.upper()} Composer")
    print(f"{'='*70}\n")
    
    # Create composer
    if composer_name == 'random':
        composer = RandomComposer(seed=seed, grammar=DefaultGrammar)
        print("Description: Uniform random sampling from all type-valid candidates")
        print("Expected: Syntactically valid but often degenerate programs")
    
    elif composer_name == 'random_guarded':
        composer = RandomGuardedComposer(seed=seed, grammar=DefaultGrammar)
        print("Description: Random sampling with guard rules to prevent trivial patterns")
        print("Expected: Better than random, avoids some degenerate cases")
    
    elif composer_name == 'template':
        composer = TemplateComposer(seed=seed, grammar=DefaultGrammar, noise=0.0)
        print("Description: Template-based generation with hand-tuned weights")
        print("Expected: Semantically meaningful programs following compositional patterns")
    
    elif composer_name == 'mcts':
        print("Description: Monte Carlo Tree Search with reinforcement learning")
        print("Expected: Programs that vary meaningfully with input")
        print("\nTraining MCTS composer (500 episodes)...")
        composer = train_mcts_composer(
            grammar=DefaultGrammar,
            target_type=target_type,
            num_episodes=500,
            depth=4,
            seed=seed,
            verbose=True
        )
    
    elif composer_name == 'empirical':
        from pathlib import Path
        default_file = Path('src/data/rule/functions.txt')
        if default_file.exists():
            print(f"Description: Learns distributions from example programs")
            print(f"Training file: {default_file} ({len(open(default_file).readlines())} programs)")
            print(f"Expected: Programs matching learned patterns from training data")
            composer = EmpiricalComposer(
                seed=seed,
                grammar=DefaultGrammar,
                functions_path=default_file
            )
        else:
            print(f"Description: Learns distributions from example programs")
            print(f"Expected: Programs matching learned patterns (if trained on data)")
            print(f"Note: Default training file not found, using empty distributions (random fallback)")
            composer = EmpiricalComposer(seed=seed, grammar=DefaultGrammar)
    
    else:
        print(f"Unknown composer: {composer_name}")
        print("Available: random, random_guarded, template, mcts")
        return
    
    # Setup compiler for testing
    compiler = JITCompiler(DefaultGrammar)
    
    print(f"\n{'='*70}")
    print(f"Generated Programs")
    print(f"{'='*70}\n")
    
    # Generate and display programs
    for i in range(num_examples):
        composer.reset_var_counter()
        
        try:
            program = composer.generate(
                target_type=target_type,
                depth=4,
                context={},
                substitutions=SubstitutionTable()
            )
            
            prog_str = pretty_print(program)
            print(f"Program {i+1}:")
            print(f"  {prog_str}")
            
            # Try to compile and test
            try:
                fn = compiler.compile(program)
                
                # Test on sample inputs
                test_inputs = [
                    [1, 2, 3],
                    [10, 20, 30],
                    [],
                    [5, 4, 3, 2, 1],
                    [1, 1, 2, 2, 3],
                    [-1, 0, 1],
                ]
                
                print(f"  Behavior:")
                for test_input in test_inputs:
                    try:
                        result = fn(test_input)
                        print(f"    {test_input} -> {result}")
                    except Exception as e:
                        print(f"    {test_input} -> ERROR: {str(e)[:40]}")
                
            except Exception as e:
                print(f"  Compilation Error: {str(e)[:60]}")
            
            print()
        
        except Exception as e:
            print(f"Program {i+1}: Generation Error - {str(e)[:60]}\n")
    
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    print(f"\nComposer: {composer_name}")
    print(f"Programs generated: {num_examples}")
    print(f"Target type: Callable[[list[int]], list[int]]")
    print(f"Depth: 4")


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_composer_test.py <composer>")
        print("\nAvailable composers:")
        print("  random          - Uniform random sampling")
        print("  random_guarded  - Random with guard rules")
        print("  template        - Template-based generation")
        print("  mcts            - Monte Carlo Tree Search")
        print("  empirical       - Empirical distribution learning")
        print("\nExample:")
        print("  python scripts/quick_composer_test.py template")
        sys.exit(1)
    
    composer_name = sys.argv[1].lower()
    num_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    test_composer(composer_name, num_examples)


if __name__ == '__main__':
    main()
