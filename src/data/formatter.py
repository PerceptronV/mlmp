from pathlib import Path
from typing import Dict, List, Any


def format_io_pair(io_pair: Dict[str, List[int]], index: int, total: int, structured: bool = False) -> str:
    """Format a single I/O pair."""
    input_str = str(io_pair['input'])
    output_str = str(io_pair['output'])
    if structured:
        return f"    [{index+1}/{total}] {input_str} → {output_str}"
    else:
        return f"{input_str} → {output_str}"


def format_example(example: Dict[str, Any], example_type: str, index: int = None,
                   fold: bool = False, structured: bool = False) -> str:
    """Format a support or query example."""
    lines = []

    # Header
    if structured:
        if index is not None:
            header = f"{example_type} Example #{index + 1}"
        else:
            header = example_type
        lines.append("\n" + "=" * 80)
        lines.append(header)
        lines.append("=" * 80)

    # Programs
    if not structured and example_type == "QUERY":
        lines.append(f"{example_type} Program: ?")
    elif not structured:
        lines.append(f"{example_type} Program: {example['program_shuffled']}")
    else:
        lines.append("\nPrograms:")
        lines.append(f"  Canonical: {example['program_canonical']}")
        lines.append(f"  Shuffled:  {example['program_shuffled']}")

    # Functions used
    functions = ", ".join(sorted(example['functions_used']))
    if structured:
        lines.append(f"\nFunctions used: {functions}")

    # I/O pairs
    io_pairs = example['io_pairs']
    if structured:
        lines.append(f"\nI/O Pairs ({len(io_pairs)} total):")

    # Show first 3 and last 1 if there are many
    if not fold or len(io_pairs) <= 5 or not structured:
        for i, pair in enumerate(io_pairs):
            lines.append(format_io_pair(pair, i, len(io_pairs), structured=structured))
    else:
        # Show first 3
        for i in range(3):
            lines.append(format_io_pair(io_pairs[i], i, len(io_pairs), structured=structured))
        lines.append("    ...")
        # Show last one
        lines.append(format_io_pair(io_pairs[-1], len(io_pairs) - 1, len(io_pairs), structured=structured))

    if not structured and example_type == "QUERY":
        lines.append("\nTARGET OUTPUT:")
        lines.append(example['program_shuffled'])
    
    return "\n".join(lines)


def format_symbol_mapping(mapping: Dict[str, str], fold: bool = False) -> str:
    """Format the symbol mapping dictionary."""
    lines = []
    lines.append("\nSymbol Mapping (Canonical → Shuffled):")
    lines.append("-" * 80)

    # Sort by canonical name for readability
    sorted_items = sorted(mapping.items())

    if not fold or len(sorted_items) <= 20:
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

    return "\n".join(lines)


def format_episode(episode: Dict[str, Any], episode_path: Path, fold_mappings: bool = False,
                   fold_io: bool = False, structured: bool = False) -> str:
    """Format a metalearning episode in structured format."""
    lines = []

    # Header
    if structured:
        lines.append("\n" + "=" * 80)
        lines.append(f"METALEARNING EPISODE #{episode['episode_id']}")
        lines.append("=" * 80)
        lines.append(f"Source: {episode_path}")

    # Symbol mapping
    if structured:
        lines.append(format_symbol_mapping(episode['symbol_mapping'], fold=fold_mappings))

    # Support examples
    if structured:
        lines.append("\n" + "=" * 80)
        lines.append(f"SUPPORT SET ({len(episode['support_examples'])} examples)")
        lines.append("=" * 80)

    for i, support_ex in enumerate(episode['support_examples']):
        lines.append(format_example(support_ex, "Support", i, fold=fold_io, structured=structured))

    # Query example
    lines.append(format_example(episode['query'], "QUERY", fold=fold_io, structured=structured))

    # Summary
    if structured:
        lines.append("\n" + "=" * 80)
        lines.append("SUMMARY")
        lines.append("=" * 80)

        # Collect all functions used
        all_functions = set()
        for ex in episode['support_examples']:
            all_functions.update(ex['functions_used'])

        query_functions = set(episode['query']['functions_used'])

        lines.append(f"\nSupport examples: {len(episode['support_examples'])}")
        lines.append(f"I/O pairs per example: {len(episode['support_examples'][0]['io_pairs'])}")
        lines.append(f"Functions in support: {', '.join(sorted(all_functions))}")
        lines.append(f"Functions in query: {', '.join(sorted(query_functions))}")

        # Check if query uses only support functions
        unseen_functions = query_functions - all_functions
        if unseen_functions:
            lines.append(f"\n!  WARNING: Query uses unseen functions: {', '.join(sorted(unseen_functions))}")
        else:
            lines.append("\n  Query uses only functions seen in support examples")

        lines.append("\n" + "=" * 80 + "\n")

    return "\n".join(lines)
