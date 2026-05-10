#!/usr/bin/env python3
"""
Convert Rule-MPS Programs to Standard Syntax

This script reads all programs from functions.csv and converts them from
Rule et al.'s syntax to our standard syntax, outputting one program per line.

Usage:
    python scripts/convert_rule_programs.py [--input INPUT] [--output OUTPUT]

Arguments:
    --input, -i    Input CSV file path (default: src/lang/composer/functions.csv)
    --output, -o   Output text file path (default: src/lang/composer/functions.txt)
"""

import argparse
import csv
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lang.parser import parse as parse_program
from src.lang.ast_nodes import pretty_print, LambdaNode, ApplicationNode, VariableNode
from src.lang.grammar import DefaultGrammar
from typing import get_origin, get_args
from src.lang.type_utils import CallableOrig


def uncurry_lambdas(ast):
    """
    Uncurry nested lambdas that are arguments to multi-argument functions.

    For example, converts:
      (mapi (λ y (λ z (+ z y))) x)
    to:
      (mapi (λ y z (+ z y)) x)
    """
    if isinstance(ast, ApplicationNode):
        # Check if this is a function application
        func = ast.function
        args = ast.arguments

        if isinstance(func, VariableNode) and func.name in DefaultGrammar.names:
            # Get the function's expected argument types
            func_info = DefaultGrammar[func.name]
            arg_types = func_info['arg_types']

            # Process each argument
            new_args = []
            for i, arg in enumerate(args):
                if i < len(arg_types):
                    expected_type = arg_types[i]
                    # Check if this argument should be a multi-param callable
                    if get_origin(expected_type) == CallableOrig:
                        type_args = get_args(expected_type)
                        if len(type_args) == 2 and isinstance(type_args[0], list):
                            # This should be a callable with N parameters
                            expected_params = type_args[0]
                            num_params = len(expected_params)

                            # Try to uncurry if arg is nested lambdas
                            uncurried = uncurry_nested_lambda(arg, num_params)
                            new_args.append(uncurried)
                            continue

                # Recursively process argument
                new_args.append(uncurry_lambdas(arg))

            # Recursively process the function itself
            new_func = uncurry_lambdas(func)
            return ApplicationNode(new_func, new_args)
        else:
            # Recursively process all parts
            new_func = uncurry_lambdas(func)
            new_args = [uncurry_lambdas(arg) for arg in args]
            return ApplicationNode(new_func, new_args)

    elif isinstance(ast, LambdaNode):
        # Recursively process lambda body
        new_body = uncurry_lambdas(ast.body)
        return LambdaNode(ast.param, new_body)

    else:
        # Base case: return as-is
        return ast


def uncurry_nested_lambda(ast, num_params):
    """
    Uncurry a nested lambda to have num_params parameters.

    Converts (λ x (λ y (λ z body))) to (λ (x y z) body) when num_params=3.
    """
    if not isinstance(ast, LambdaNode):
        # Not a lambda, can't uncurry
        return uncurry_lambdas(ast)

    # Collect parameters from nested lambdas
    params = ast.param.copy()  # param is a list
    current = ast.body

    while isinstance(current, LambdaNode) and len(params) < num_params:
        params.extend(current.param)
        current = current.body

    # Recursively process the final body
    final_body = uncurry_lambdas(current)

    # Build multi-parameter lambda
    return LambdaNode(params, final_body)


def convert_rule_syntax(program_str: str) -> str:
    """
    Convert Rule et al. syntax to our syntax.

    Transformations:
    - (lambda (body)) → (lambda x body)
    - (lambda $0) → (lambda x x)  [identity function]
    - Nested lambdas: (lambda (lambda ...)) gets converted properly
    - $N indexing: De Bruijn-style where $0 is innermost, $1 is one level out, etc.
      With N nested lambdas, $N-1 refers to outermost, $0 refers to innermost.
    - empty → []

    Example:
      (lambda (mapi (lambda (lambda (max (take $1 $2)))) $0))
      The outermost lambda is depth 0 from root
      At location of $0 (in main lambda body): depth=1, so $0 refers to depth 1-0=1 -> x
      At location of $1, $2 (inside double-nested lambda): depth=3
        $1 refers to depth 3-1=2 -> y (first inner lambda)
        $2 refers to depth 3-2=1 -> x (but wait, that's wrong direction)

      Actually: $N counts outward from current position.
      In (lambda (lambda (lambda ... $0 ... $1 ... $2 ...))):
        $0 = innermost lambda's param
        $1 = middle lambda's param
        $2 = outermost lambda's param
    """
    var_names = ['x', 'y', 'z', 'w', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    def parse_and_convert(s: str, pos: int, depth: int) -> tuple[str, int]:
        """
        Parse from position pos and return (converted_string, end_position).
        depth is the current lambda nesting depth.
        """
        result = []
        i = pos

        while i < len(s):
            # Skip whitespace
            while i < len(s) and s[i] in ' \t\n':
                result.append(s[i])
                i += 1

            if i >= len(s):
                break

            if s[i] == '(':
                # Check if it's a lambda
                if s[i:].startswith('(lambda '):
                    # It's a lambda - add variable name and process body
                    var_name = var_names[depth % len(var_names)]
                    result.append(f'(λ ({var_name}) ')
                    i += len('(lambda ')

                    # Skip whitespace
                    while i < len(s) and s[i] in ' \t\n':
                        i += 1

                    # Check if the body is just a variable reference (like $0)
                    # In this case, it's an identity function
                    if s[i] == '$':
                        # Peek ahead to see if it's just $N followed by )
                        j = i + 1
                        while j < len(s) and s[j].isdigit():
                            j += 1
                        # Check if next non-whitespace is )
                        k = j
                        while k < len(s) and s[k] in ' \t\n':
                            k += 1
                        if k < len(s) and s[k] == ')':
                            # It's an identity lambda like (lambda $0)
                            # The $N should refer to the current lambda's parameter when
                            # N points to current depth
                            n = int(s[i+1:j])
                            # At depth d, $n refers to depth (d - 1 - n)
                            # For (lambda $0) at depth 0, target = 0-1-0 = -1
                            # But we want it to be the current lambda's param (depth 0)
                            # So when target_depth < 0, use current lambda's param
                            target_depth = depth - 1 - n
                            if target_depth < 0:
                                # Refers to current lambda parameter
                                result.append(var_names[depth % len(var_names)])
                            elif target_depth < len(var_names):
                                result.append(var_names[target_depth])
                            else:
                                result.append(s[i:j])
                            result.append(')')
                            i = k + 1
                            return ''.join(result), i

                    # Parse the body at depth+1
                    body_str, i = parse_and_convert(s, i, depth + 1)
                    result.append(body_str)

                    # Should end with )
                    if i < len(s) and s[i] == ')':
                        result.append(')')
                        i += 1

                    return ''.join(result), i
                else:
                    # Regular s-expression
                    result.append('(')
                    i += 1

                    # Parse contents
                    while i < len(s) and s[i] != ')':
                        inner_str, i = parse_and_convert(s, i, depth)
                        result.append(inner_str)

                    if i < len(s) and s[i] == ')':
                        result.append(')')
                        i += 1

                    return ''.join(result), i

            elif s[i] == ')':
                # End of current expression
                return ''.join(result), i

            elif s[i] == '$':
                # Variable reference - $n means n levels up from innermost
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                n = int(s[i+1:j])
                # At depth d, $n refers to the lambda at depth (d - 1 - n)
                # depth-1 is innermost ($0), depth-2 is next out ($1), etc.
                target_depth = depth - 1 - n
                if 0 <= target_depth < len(var_names):
                    result.append(var_names[target_depth])
                else:
                    result.append(s[i:j])  # Keep original if invalid
                i = j
                return ''.join(result), i

            elif s[i:].startswith('empty'):
                result.append('[]')
                i += 5
                return ''.join(result), i

            else:
                # Regular identifier or number
                j = i
                while j < len(s) and s[j] not in ' \t\n()':
                    j += 1
                result.append(s[i:j])
                i = j
                return ''.join(result), i

        return ''.join(result), i

    converted, _ = parse_and_convert(program_str, 0, 0)

    # Replace λ back to lambda for consistency with parser
    converted = converted.replace('λ', 'lambda')

    return converted


def parse_args():
    """Parse command line arguments."""
    default_input = Path(__file__).parent.parent/'src'/'lang'/'composer'/'data'/'functions.csv'
    default_output = Path(__file__).parent.parent/'src'/'lang'/'composer'/'data'/'functions.txt'
    
    parser = argparse.ArgumentParser(
        description='Convert Rule-MPS Programs to Standard Syntax',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=str(default_input),
        help=f'Input CSV file path (default: {default_input.relative_to(Path(__file__).parent.parent)})'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(default_output),
        help=f'Output text file path (default: {default_output.relative_to(Path(__file__).parent.parent)})'
    )
    
    return parser.parse_args()


def main():
    """Convert all programs from functions.csv to standard syntax."""
    args = parse_args()

    # Input and output paths
    csv_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Reading programs from: {csv_path}")
    print(f"Writing converted programs to: {output_path}")
    print()

    converted_programs = []
    successful = 0
    failed = 0
    failed_examples = []

    # Read and convert programs
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            program_id = row['id']
            original_program = row['program']
            gloss = row.get('gloss', '')

            try:
                # Convert syntax
                converted = convert_rule_syntax(original_program)

                # Parse to validate
                ast = parse_program(converted)

                # Uncurry nested lambdas that are arguments to multi-param functions
                ast = uncurry_lambdas(ast)

                # Pretty print to get final form
                final_program = pretty_print(ast)

                # Store result
                converted_programs.append({
                    'id': program_id,
                    'original': original_program,
                    'converted': final_program,
                    'gloss': gloss
                })

                successful += 1

            except Exception as e:
                failed += 1
                failed_examples.append({
                    'id': program_id,
                    'original': original_program,
                    'error': str(e)
                })

    # Write converted programs to file
    with open(output_path, 'w') as f:
        for item in converted_programs:
            f.write(f"{item['converted']}\n")

    # Print summary
    print("=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total programs:     {successful + failed}")
    print(f"Successfully converted: {successful} ({100*successful/(successful+failed):.1f}%)")
    print(f"Failed:             {failed}")
    print()
    print(f"✓ Converted programs written to: {output_path}")

    # Show some examples
    if converted_programs:
        print()
        print("=" * 80)
        print("SAMPLE CONVERTED PROGRAMS")
        print("=" * 80)
        for i, item in enumerate(converted_programs[:10], 1):
            print(f"\n{i}. {item['id']}: {item['gloss']}")
            print(f"   Original:  {item['original']}")
            print(f"   Converted: {item['converted']}")

    # Show failures if any
    if failed_examples:
        print()
        print("=" * 80)
        print("FAILED CONVERSIONS")
        print("=" * 80)
        for item in failed_examples[:5]:
            
            print(f"\n{item['id']}:")
            print(f"  Original: {item['original']}")
            print(f"  Error:    {item['error']}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
