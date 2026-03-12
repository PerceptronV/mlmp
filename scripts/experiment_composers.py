#!/usr/bin/env python3
"""
Composer Experimentation Script

This script allows you to experiment with different program composers and
analyze their output quality and behavior. It generates programs, compares
their characteristics, and provides detailed statistics.

Usage:
    python scripts/experiment_composers.py --composer random --num-samples 100
    python scripts/experiment_composers.py --composer template --depth 5
    python scripts/experiment_composers.py --compare random,template,mcts
"""

import sys
import os
import argparse
import json
import time
from collections import defaultdict, Counter
from typing import Callable, Any
from pathlib import Path
import random

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lang.grammar import DefaultGrammar
from lang.parser import parse
from lang.composers.random import RandomComposer
from lang.composers.template import TemplateComposer
from lang.composers.mcts import MCTSComposer, train_mcts_composer
from lang.composers.random_guarded import RandomGuardedComposer
from lang.composers.empirical import EmpiricalComposer
from lang.ast_nodes import pretty_print, ASTNode
from lang.type_utils import SubstitutionTable
from lang.type_checker import TypeChecker
from lang.evaluator import Evaluator
from lang.compiler import JITCompiler


class ProgramAnalyzer:
    """Analyzes generated programs for quality metrics."""
    
    def __init__(self, grammar, seed):
        self.grammar = grammar
        self.type_checker = TypeChecker()
        self.evaluator = Evaluator()
        self.jit_compiler = JITCompiler(grammar)
        self.seed = seed
        
        # Define test inputs once - used for both variability calculation and display
        self.test_inputs = self._generate_test_inputs()
    
    def _generate_test_inputs(self) -> list:
        """Generate the standard test input set for variability calculation."""
        random.seed(self.seed)
        
        # Fixed test inputs
        inputs = [
            # Edge cases
            [],
            [0],
            [5],
            
            # Basic variety
            [1, 2, 3],
            [10, 20, 30],
            [0, 0, 0],
            [99, 50, 1],
            
            # More patterns
            [1, 2, 3, 4, 5],          # Standard sequence
            [5, 4, 3, 2, 1],          # Descending
            [1, 1, 2, 2, 3, 3],       # Duplicates
            [7, 7, 7],                # All same
            [70, 80, 90],             # Large numbers
            list(range(1, 11)),       # [1..10]
            list(range(10, 0, -1)),   # [10..1]
            [2, 4, 6, 8, 10],         # Even only
            [1, 3, 5, 7, 9],          # Odd only
            [5, 1, 4, 2, 3],          # Unordered
        ]
        
        # Add random patterns (with fixed seed for reproducibility)
        for _ in range(10):
            inputs.append([
                random.randint(0, 100) 
                for _ in range(random.randint(3, 10))
            ])
        
        return inputs
    
    def analyze_program(self, program: ASTNode) -> dict[str, Any]:
        """
        Analyze a single program and return metrics.
        
        Returns:
            Dictionary with metrics like:
            - size: Number of AST nodes
            - depth: Maximum depth of AST
            - functions_used: Set of functions used
            - uses_input: Whether program uses input variable
            - variability: Output variability across inputs
            - type_checks: Whether program type-checks
            - compiles: Whether program compiles
        """
        metrics = {}
        
        # Size and depth
        metrics['size'] = self._count_nodes(program)
        metrics['depth'] = self._compute_depth(program)
        
        # Functions used
        metrics['functions_used'] = program.function_names() & set(self.grammar.names)
        metrics['num_functions'] = len(metrics['functions_used'])
        
        # Check if uses input variable
        prog_str = pretty_print(program)
        metrics['uses_input'] = self._uses_input_variable(prog_str)
        
        # Type checking
        try:
            self.type_checker.check(program)
            metrics['type_checks'] = True
        except Exception as e:
            metrics['type_checks'] = False
            metrics['type_error'] = str(e)
        
        # Compilation
        try:
            self.jit_compiler.compile(program)
            metrics['compiles'] = True
        except Exception as e:
            metrics['compiles'] = False
            metrics['compile_error'] = str(e)
        
        # Variability (only if compiles)
        if metrics['compiles']:
            metrics['variability'] = self._compute_variability(program)
        else:
            metrics['variability'] = 0.0
        
        # Program string representation
        metrics['program'] = prog_str
        
        return metrics
    
    def _count_nodes(self, node: ASTNode) -> int:
        """Count total AST nodes."""
        count = 1
        if node.ast_type == 'Lambda':
            count += self._count_nodes(node.body)
        elif node.ast_type == 'Application':
            count += self._count_nodes(node.function)
            for arg in node.arguments:
                count += self._count_nodes(arg)
        elif node.ast_type == 'If':
            count += self._count_nodes(node.condition)
            count += self._count_nodes(node.then_expr)
            count += self._count_nodes(node.else_expr)
        elif node.ast_type == 'List':
            for elem in node.elements:
                count += self._count_nodes(elem)
        return count
    
    def _compute_depth(self, node: ASTNode) -> int:
        """Compute maximum depth of AST."""
        if node.ast_type in ('Number', 'Boolean', 'Variable'):
            return 1
        elif node.ast_type == 'Lambda':
            return 1 + self._compute_depth(node.body)
        elif node.ast_type == 'Application':
            fn_depth = self._compute_depth(node.function)
            arg_depths = [self._compute_depth(arg) for arg in node.arguments]
            return 1 + max([fn_depth] + arg_depths)
        elif node.ast_type == 'If':
            return 1 + max(
                self._compute_depth(node.condition),
                self._compute_depth(node.then_expr),
                self._compute_depth(node.else_expr)
            )
        elif node.ast_type == 'List':
            if not node.elements:
                return 1
            return 1 + max(self._compute_depth(e) for e in node.elements)
        return 1
    
    def _uses_input_variable(self, program_str: str) -> bool:
        """Check if program uses input variable."""
        import re
        # Skip lambda header and check for x in body
        body = re.sub(r'^\(λ \([^)]+\) ', '', program_str)
        return bool(re.search(r'\bx\b', body))
    
    def _compute_variability(self, program: ASTNode) -> float:
        """
        Compute output variability by running program on different inputs.
        
        Errors count as the same output (reducing variability).
        Identity outputs (where output == input) also count as the same (reducing variability).
        
        Returns:
            Float in [0, 1] representing output variability
        """
        try:
            compiled_fn = self.jit_compiler.compile(program)
            
            outputs = []
            for inp in self.test_inputs:
                try:
                    result = compiled_fn(inp)
                    # Check if output equals input (identity behavior)
                    if self._equals(result, inp):
                        outputs.append(('identity', None))
                    else:
                        outputs.append(('success', self._to_hashable(result)))
                except Exception as e:
                    # All errors count as the same output type
                    outputs.append(('error', None))
            
            if not outputs or len(outputs) < 2:
                return 0.0
            
            # Compute variability as ratio of unique outputs
            # Note: All errors are identical, all identity cases are identical
            unique_outputs = len(set(outputs))
            return (unique_outputs - 1) / (len(outputs) - 1)
        
        except:
            return 0.0
    
    def _equals(self, a, b) -> bool:
        """Check if two values are equal (handles lists recursively)."""
        if type(a) != type(b):
            return False
        if isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(self._equals(x, y) for x, y in zip(a, b))
        return a == b
    
    def _to_hashable(self, value: Any) -> Any:
        """Convert value to hashable representation."""
        if isinstance(value, list):
            return tuple(self._to_hashable(x) for x in value)
        return value


class ComposerExperiment:
    """Runs experiments comparing different composers."""
    
    def __init__(
        self,
        grammar,
        target_type,
        depth: int,
        num_samples: int,
        seed: int,
        empirical_file: str = 'src/data/rule/functions.txt'
    ):
        self.grammar = grammar
        self.target_type = target_type
        self.depth = depth
        self.num_samples = num_samples
        self.seed = seed
        self.empirical_file = empirical_file
        self.analyzer = ProgramAnalyzer(grammar, seed)
        self.composers = {}
    
    def register_composer(self, name: str, composer):
        """Register a composer for experiments."""
        self.composers[name] = composer
    
    def run_experiment(self, composer_name: str) -> dict:
        """
        Run experiment for a single composer.
        
        Returns:
            Dictionary with:
            - programs: List of analyzed programs
            - stats: Aggregate statistics
            - timing: Generation timing info
        """
        composer = self.composers[composer_name]
        
        print(f"\n{'='*60}")
        print(f"Experimenting with {composer_name} composer")
        print(f"{'='*60}")
        
        programs = []
        total_time = 0
        
        for i in range(self.num_samples):
            composer.reset_var_counter()
            
            start_time = time.time()
            try:
                program = composer.generate(
                    target_type=self.target_type,
                    depth=self.depth,
                    context={},
                    substitutions=SubstitutionTable()
                )
                elapsed = time.time() - start_time
                total_time += elapsed
                
                # Analyze program
                metrics = self.analyzer.analyze_program(program)
                metrics['generation_time'] = elapsed
                programs.append(metrics)
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i+1}/{self.num_samples} programs...")
            
            except Exception as e:
                print(f"  Error generating program {i}: {e}")
        
        # Compute aggregate statistics
        stats = self._compute_stats(programs)
        stats['total_time'] = total_time
        stats['avg_time'] = total_time / len(programs) if programs else 0
        
        return {
            'composer': composer_name,
            'programs': programs,
            'stats': stats
        }
    
    def _compute_stats(self, programs: list[dict]) -> dict:
        """Compute aggregate statistics from program metrics."""
        if not programs:
            return {}
        
        stats = {
            'num_programs': len(programs),
            'type_check_rate': sum(p['type_checks'] for p in programs) / len(programs),
            'compile_rate': sum(p['compiles'] for p in programs) / len(programs),
            'uses_input_rate': sum(p['uses_input'] for p in programs) / len(programs),
        }
        
        # Size statistics
        sizes = [p['size'] for p in programs]
        stats['avg_size'] = sum(sizes) / len(sizes)
        stats['min_size'] = min(sizes)
        stats['max_size'] = max(sizes)
        
        # Depth statistics
        depths = [p['depth'] for p in programs]
        stats['avg_depth'] = sum(depths) / len(depths)
        stats['min_depth'] = min(depths)
        stats['max_depth'] = max(depths)
        
        # Function usage statistics
        all_functions = []
        for p in programs:
            all_functions.extend(p['functions_used'])
        stats['function_usage'] = dict(Counter(all_functions).most_common(10))
        stats['unique_functions_used'] = len(set(all_functions))
        
        # Variability statistics
        variabilities = [p['variability'] for p in programs if p['compiles']]
        if variabilities:
            stats['avg_variability'] = sum(variabilities) / len(variabilities)
            stats['high_variability_rate'] = sum(v > 0.5 for v in variabilities) / len(variabilities)
        else:
            stats['avg_variability'] = 0.0
            stats['high_variability_rate'] = 0.0
        
        return stats
    
    def print_stats(self, result: dict):
        """Print formatted statistics."""
        stats = result['stats']
        composer = result['composer']
        
        print(f"\n{'='*60}")
        print(f"Results for {composer}")
        print(f"{'='*60}")
        
        print(f"\n## Generation Quality")
        print(f"  Type Check Rate:     {stats['type_check_rate']*100:.1f}%")
        print(f"  Compile Rate:        {stats['compile_rate']*100:.1f}%")
        print(f"  Uses Input Rate:     {stats['uses_input_rate']*100:.1f}%")
        
        print(f"\n## Program Size")
        print(f"  Average Size:        {stats['avg_size']:.1f} nodes")
        print(f"  Size Range:          [{stats['min_size']}, {stats['max_size']}]")
        
        print(f"\n## Program Depth")
        print(f"  Average Depth:       {stats['avg_depth']:.1f}")
        print(f"  Depth Range:         [{stats['min_depth']}, {stats['max_depth']}]")
        
        print(f"\n## Variability")
        print(f"  Average Variability: {stats['avg_variability']:.3f}")
        print(f"  High Variability:    {stats['high_variability_rate']*100:.1f}%")
        
        print(f"\n## Function Usage (Top 10)")
        for func, count in stats['function_usage'].items():
            print(f"  {func:15s} {count:4d} times")
        
        print(f"\n## Performance")
        print(f"  Total Time:          {stats['total_time']:.2f}s")
        print(f"  Avg Time/Program:    {stats['avg_time']*1000:.1f}ms")
    
    def compare_composers(self, composer_names: list[str]) -> dict:
        """Run experiments for multiple composers and compare."""
        results = {}
        
        for name in composer_names:
            if name != 'rule' and name not in self.composers:
                print(f"Warning: Composer '{name}' not registered, skipping")
                continue

            if name == 'rule':
                rule_programs = []
                with open(self.empirical_file, 'r') as f:
                    for line in f:
                        p = line.strip()
                        if p != '':
                            metrics = self.analyzer.analyze_program(parse(p))
                            rule_programs.append(metrics)
                
                stats = self._compute_stats(rule_programs)
                stats['total_time'] = 0
                stats['avg_time'] = 0

                results['rule'] = {
                    'composer': 'rule',
                    'stats': stats,
                    'programs': rule_programs
                }
                self.print_stats(results['rule'])
            
            else:
                results[name] = self.run_experiment(name)
                self.print_stats(results[name])
        
        # Comparison summary
        if len(results) > 1:
            self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: dict):
        """Print comparison between composers."""
        print(f"\n{'='*60}")
        print(f"Comparison Summary")
        print(f"{'='*60}")
        
        # Create comparison table
        metrics = [
            ('Type Check Rate', 'type_check_rate', '%'),
            ('Compile Rate', 'compile_rate', '%'),
            ('Uses Input Rate', 'uses_input_rate', '%'),
            ('Avg Size', 'avg_size', ''),
            ('Avg Depth', 'avg_depth', ''),
            ('Avg Variability', 'avg_variability', ''),
            ('High Var Rate', 'high_variability_rate', '%'),
            ('Avg Time (ms)', 'avg_time', 'ms'),
        ]
        
        print(f"\n{'Metric':<20}", end='')
        for name in results.keys():
            print(f"{name:>15}", end='')
        print()
        print('-' * (20 + 15 * len(results)))
        
        for metric_name, metric_key, unit in metrics:
            print(f"{metric_name:<20}", end='')
            for name, result in results.items():
                value = result['stats'].get(metric_key, 0)
                if unit == '%':
                    print(f"{value*100:>14.1f}%", end='')
                elif unit == 'ms':
                    print(f"{value*1000:>14.1f}", end='')
                else:
                    print(f"{value:>15.2f}", end='')
            print()
    
    def save_results(self, results: dict, output_file: str):
        """Save results to JSON file."""
        # Convert sets to lists for JSON serialization
        serializable_results = {}
        for composer_name, result in results.items():
            serializable_result = {
                'composer': result['composer'],
                'stats': result['stats'].copy()
            }
            
            # Convert function sets to lists
            if 'function_usage' in serializable_result['stats']:
                serializable_result['stats']['function_usage'] = \
                    dict(serializable_result['stats']['function_usage'])
            
            # Only save program strings, not full metrics
            serializable_result['sample_programs'] = [
                p['program'] for p in result['programs'][:10]
            ]
            
            serializable_results[composer_name] = serializable_result
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def visualize_programs_by_variability(self, composer_name: str, num_display: int = None):
        """
        Generate and display programs ordered by variability.
        
        Args:
            composer_name: Name of the composer to use
            num_display: Number of programs to display (default: all)
        """
        from lang.ast_nodes import pretty_print
        from lang.parser import parse
        
        if composer_name not in self.composers:
            print(f"Error: Composer '{composer_name}' not registered")
            return
        
        composer = self.composers[composer_name]
        
        print(f"\n{'='*70}")
        print(f"Generating programs with {composer_name} ordered by variability")
        print(f"{'='*70}\n")
        
        # Use the same test inputs as variability calculation
        display_test_inputs = self.analyzer.test_inputs
        
        # Generate programs and collect metrics
        programs_with_metrics = []
        
        for i in range(self.num_samples):
            composer.reset_var_counter()
            
            try:
                program = composer.generate(
                    self.target_type,
                    depth=self.depth,
                    context={},
                    substitutions=SubstitutionTable()
                )
                
                metrics = self.analyzer.analyze_program(program)
                prog_str = pretty_print(program)
                
                # Collect input-output pairs
                io_pairs = []
                if metrics['compiles']:
                    try:
                        compiled_fn = self.analyzer.jit_compiler.compile(program)
                        for test_input in display_test_inputs:
                            try:
                                output = compiled_fn(test_input)
                                # Show full output without truncation
                                output_str = str(output)
                                io_pairs.append((test_input, output_str))
                            except Exception as e:
                                io_pairs.append((test_input, f"ERROR: {str(e)[:50]}"))
                    except:
                        pass
                
                programs_with_metrics.append({
                    'index': i + 1,
                    'program': prog_str,
                    'variability': metrics['variability'],
                    'type_checks': metrics['type_checks'],
                    'compiles': metrics['compiles'],
                    'uses_input': metrics['uses_input'],
                    'size': metrics['size'],
                    'io_pairs': io_pairs
                })
                
            except Exception as e:
                print(f"Warning: Failed to generate program {i+1}: {e}")
        
        # Sort by variability (descending)
        programs_with_metrics.sort(key=lambda p: p['variability'], reverse=True)
        
        # Determine how many to display
        display_count = num_display if num_display else len(programs_with_metrics)
        
        # Display programs
        for i, prog_data in enumerate(programs_with_metrics[:display_count]):
            var = prog_data['variability']
            bar_length = int(var * 40)
            print(f"Program {i+1} (original #{prog_data['index']}):")
            print(f"  Variability: {var:.3f} {'█' * bar_length}")
            print(f"  Type checks: {'✓' if prog_data['type_checks'] else '✗'} | "
                  f"Compiles: {'✓' if prog_data['compiles'] else '✗'} | "
                  f"Uses input: {'✓' if prog_data['uses_input'] else '✗'} | "
                  f"Size: {prog_data['size']} nodes")
            print(f"  Program: {prog_data['program']}")
            
            # Display input-output pairs
            if prog_data['io_pairs']:
                print(f"  Behavior ({len(prog_data['io_pairs'])} test cases):")
                for inp, out in prog_data['io_pairs']:
                    # Show full input without truncation
                    inp_str = str(inp)
                    print(f"    {inp_str:40} -> {out}")
            
            print()
        
        # Summary statistics
        variabilities = [p['variability'] for p in programs_with_metrics]
        print(f"{'='*70}")
        print(f"Summary:")
        print(f"  Total programs: {len(programs_with_metrics)}")
        print(f"  Avg variability: {sum(variabilities) / len(variabilities):.3f}")
        print(f"  Min variability: {min(variabilities):.3f}")
        print(f"  Max variability: {max(variabilities):.3f}")
        print(f"  High variability (>0.5): {sum(1 for v in variabilities if v > 0.5)}/{len(variabilities)}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment with different program composers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single composer
  python scripts/experiment_composers.py --composer random --num-samples 100
  
  # Compare multiple composers
  python scripts/experiment_composers.py --compare random,template,mcts --num-samples 50
  
  # Adjust generation parameters
  python scripts/experiment_composers.py --composer template --depth 5 --num-samples 200
  
  # Train MCTS and test
  python scripts/experiment_composers.py --composer mcts --train-episodes 500 --num-samples 50
  
  # Visualize programs ordered by variability
  python scripts/experiment_composers.py --composer random --num-samples 30 --visualize
  python scripts/experiment_composers.py --composer template --num-samples 20 --visualize --show-count 10
        """
    )
    
    parser.add_argument(
        '--composer',
        type=str,
        choices=['random', 'random_guarded', 'template', 'mcts', 'empirical'],
        help='Single composer to experiment with'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        help='Comma-separated list of composers to compare (e.g., random,template,mcts)'
    )
    
    parser.add_argument(
        '--depth',
        type=int,
        default=4,
        help='Maximum program depth (default: 4)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of programs to generate per composer (default: 50)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--train-episodes',
        type=int,
        default=1000,
        help='Number of training episodes for MCTS (default: 1000)'
    )
    
    parser.add_argument(
        '--template-noise',
        type=float,
        default=0.0,
        help='Noise parameter for template composer (default: 0.0)'
    )
    
    parser.add_argument(
        '--empirical-file',
        type=str,
        default='src/data/rule/functions.txt',
        help='Program file for empirical composer (default: src/data/rule/functions.txt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='composer_results.json',
        help='Output file for results (default: composer_results.json)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize programs ordered by variability (only with --composer)'
    )
    
    parser.add_argument(
        '--show-count',
        type=int,
        default=None,
        help='Number of programs to display in visualization (default: all)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.composer and not args.compare:
        parser.error("Must specify either --composer or --compare")
    
    # Setup
    target_type = Callable[[list[int]], list[int]]
    experiment = ComposerExperiment(
        grammar=DefaultGrammar,
        target_type=target_type,
        depth=args.depth,
        num_samples=args.num_samples,
        seed=args.seed,
        empirical_file=args.empirical_file,
    )
    
    # Register composers
    print(f"Setting up composers...")
    print(f"  Random seed: {args.seed}")
    print(f"  Target depth: {args.depth}")
    print(f"  Samples per composer: {args.num_samples}")
    
    # Random composer
    experiment.register_composer(
        'random',
        RandomComposer(seed=args.seed, grammar=DefaultGrammar)
    )
    
    # Random guarded composer
    experiment.register_composer(
        'random_guarded',
        RandomGuardedComposer(seed=args.seed, grammar=DefaultGrammar)
    )
    
    # Template composer
    experiment.register_composer(
        'template',
        TemplateComposer(seed=args.seed, grammar=DefaultGrammar, noise=args.template_noise)
    )
    
    # Empirical composer
    if args.composer == 'empirical' or (args.compare and 'empirical' in args.compare):
        print(f"\nSetting up empirical composer...")
        from pathlib import Path
        if args.empirical_file and Path(args.empirical_file).exists():
            print(f"  Using program file: {args.empirical_file}")
            empirical_composer = EmpiricalComposer(
                seed=args.seed,
                grammar=DefaultGrammar,
                functions_path=Path(args.empirical_file)
            )
        else:
            if args.empirical_file:
                print(f"  Warning: File {args.empirical_file} not found")
            print(f"  Using empty distributions (will use fallback generation)")
            empirical_composer = EmpiricalComposer(
                seed=args.seed,
                grammar=DefaultGrammar
            )
        experiment.register_composer('empirical', empirical_composer)
    
    # MCTS composer (with optional training)
    if args.composer == 'mcts' or (args.compare and 'mcts' in args.compare):
        print(f"\nTraining MCTS composer ({args.train_episodes} episodes)...")
        mcts_composer = train_mcts_composer(
            grammar=DefaultGrammar,
            target_type=target_type,
            num_episodes=args.train_episodes,
            depth=args.depth,
            seed=args.seed,
            verbose=True
        )
        experiment.register_composer('mcts', mcts_composer)
    
    # Run experiments
    if args.compare:
        composer_names = [name.strip() for name in args.compare.split(',')] + ['rule']
        results = experiment.compare_composers(composer_names)
    elif args.visualize:
        # Visualization mode: show programs ordered by variability
        experiment.visualize_programs_by_variability(args.composer, args.show_count)
        return  # Skip normal experiment flow
    else:
        results = {args.composer: experiment.run_experiment(args.composer)}
        experiment.print_stats(results[args.composer])
    
        # Add rule et al programs
        rule_programs = []
        with open(args.empirical_file, 'r') as f:
            for line in f:
                p = line.strip()
                if p != '':
                    metrics = experiment.analyzer.analyze_program(parse(p))
                    rule_programs.append(metrics)
        
        results['rule'] = {
            'composer': 'rule',
            'stats': experiment._compute_stats(rule_programs),
            'programs': rule_programs
        }
    
    # Save results
    experiment.save_results(results, args.output)
    
    print(f"\nExperiment complete!")


if __name__ == '__main__':
    main()
