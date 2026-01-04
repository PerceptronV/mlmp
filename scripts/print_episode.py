#!/usr/bin/env python3
"""
Helper script to print a metalearning episode in human-readable format.

Usage:
    python scripts/print_episode.py <path_to_episode.json>

Example:
    python scripts/print_episode.py datasets/template_seed42/train/episode_000000.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


def format_io_pair(io_pair: Dict[str, List[int]], index: int, total: int) -> str:
    """Format a single I/O pair."""
    input_str = str(io_pair['input'])
    output_str = str(io_pair['output'])
    return f"    [{index+1}/{total}] {input_str} → {output_str}"


def format_example(example: Dict[str, Any], example_type: str, index: int = None,
                   show_all_io: bool = False) -> str:
    """Format a support or query example."""
    lines = []

    # Header
    if index is not None:
        header = f"{example_type} Example #{index + 1}"
    else:
        header = example_type
    lines.append("\n" + "=" * 80)
    lines.append(header)
    lines.append("=" * 80)

    # Programs
    lines.append("\nPrograms:")
    lines.append(f"  Canonical: {example['program_canonical']}")
    lines.append(f"  Shuffled:  {example['program_shuffled']}")

    # Functions used
    functions = ", ".join(sorted(example['functions_used']))
    lines.append(f"\nFunctions used: {functions}")

    # I/O pairs
    io_pairs = example['io_pairs']
    lines.append(f"\nI/O Pairs ({len(io_pairs)} total):")

    # Show first 3 and last 1 if there are many (unless show_all_io is True)
    if show_all_io or len(io_pairs) <= 5:
        for i, pair in enumerate(io_pairs):
            lines.append(format_io_pair(pair, i, len(io_pairs)))
    else:
        # Show first 3
        for i in range(3):
            lines.append(format_io_pair(io_pairs[i], i, len(io_pairs)))
        lines.append("    ...")
        # Show last one
        lines.append(format_io_pair(io_pairs[-1], len(io_pairs) - 1, len(io_pairs)))

    return "\n".join(lines)


def format_symbol_mapping(mapping: Dict[str, str], show_all: bool = False) -> str:
    """Format the symbol mapping dictionary."""
    lines = []
    lines.append("\nSymbol Mapping (Canonical → Shuffled):")
    lines.append("-" * 80)

    # Sort by canonical name for readability
    sorted_items = sorted(mapping.items())

    if show_all or len(sorted_items) <= 20:
        # Show all mappings in columns
        for i in range(0, len(sorted_items), 3):
            row_items = sorted_items[i:i+3]
            row = "  ".join(f"{k:12} → {v:12}" for k, v in row_items)
            lines.append(f"  {row}")
    else:
        # Show first 10
        for i in range(0, 10, 3):
            row_items = sorted_items[i:i+3]
            row = "  ".join(f"{k:12} → {v:12}" for k, v in row_items)
            lines.append(f"  {row}")

        lines.append(f"  ... ({len(sorted_items) - 10} more mappings)")
        lines.append(f"  (Use --show-all-mappings to see all {len(sorted_items)} mappings)")

    return "\n".join(lines)


def print_episode(episode_path: Path, show_all_mappings: bool = False,
                  show_all_io: bool = False) -> None:
    """Print a metalearning episode in human-readable format."""

    # Load the episode
    with open(episode_path, 'r') as f:
        episode = json.load(f)

    # Header
    print("\n" + "=" * 80)
    print(f"METALEARNING EPISODE #{episode['episode_id']}")
    print("=" * 80)
    print(f"Source: {episode_path}")

    # Symbol mapping
    print(format_symbol_mapping(episode['symbol_mapping'], show_all=show_all_mappings))

    # Support examples
    print("\n" + "=" * 80)
    print(f"SUPPORT SET ({len(episode['support_examples'])} examples)")
    print("=" * 80)

    for i, support_ex in enumerate(episode['support_examples']):
        print(format_example(support_ex, "Support", i, show_all_io=show_all_io))

    # Query example
    print(format_example(episode['query'], "QUERY", show_all_io=show_all_io))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Collect all functions used
    all_functions = set()
    for ex in episode['support_examples']:
        all_functions.update(ex['functions_used'])

    query_functions = set(episode['query']['functions_used'])

    print(f"\nSupport examples: {len(episode['support_examples'])}")
    print(f"I/O pairs per example: {len(episode['support_examples'][0]['io_pairs'])}")
    print(f"Functions in support: {', '.join(sorted(all_functions))}")
    print(f"Functions in query: {', '.join(sorted(query_functions))}")

    # Check if query uses only support functions
    unseen_functions = query_functions - all_functions
    if unseen_functions:
        print(f"\n⚠️  WARNING: Query uses unseen functions: {', '.join(sorted(unseen_functions))}")
    else:
        print("\n✓ Query uses only functions seen in support examples")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Print a metalearning episode in human-readable format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print a specific episode
  python scripts/print_episode.py datasets/template_seed42/train/episode_000000.json

  # Show all symbol mappings
  python scripts/print_episode.py datasets/template_seed42/train/episode_000000.json --show-all-mappings

  # Show all I/O pairs
  python scripts/print_episode.py datasets/template_seed42/train/episode_000000.json --show-all-io
        """
    )

    parser.add_argument(
        'episode_path',
        type=str,
        help='Path to the episode JSON file'
    )

    parser.add_argument(
        '--show-all-mappings',
        action='store_true',
        help='Show all symbol mappings (default: show first 10)'
    )

    parser.add_argument(
        '--show-all-io',
        action='store_true',
        help='Show all I/O pairs (default: show first 3 and last 1)'
    )

    args = parser.parse_args()

    # Validate path
    episode_path = Path(args.episode_path)
    if not episode_path.exists():
        print(f"Error: File not found: {episode_path}")
        return 1

    if not episode_path.suffix == '.json':
        print(f"Error: File must be a JSON file: {episode_path}")
        return 1

    # Print the episode
    try:
        print_episode(
            episode_path,
            show_all_mappings=args.show_all_mappings,
            show_all_io=args.show_all_io
        )
        return 0
    except Exception as e:
        print(f"Error reading episode: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
