#!/usr/bin/env python3
"""
Visualize Composer Experiment Results

Generates plots and visualizations from experiment result JSON files.

Usage:
    python scripts/visualize_results.py results.json
    python scripts/visualize_results.py results.json --output plots/
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_results(filepath: str) -> dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_comparison_bars(results: dict, output_dir: Path):
    """Create bar chart comparing key metrics."""
    if not HAS_MATPLOTLIB:
        return
    
    composers = list(results.keys())
    metrics = [
        ('uses_input_rate', 'Uses Input Rate', '%'),
        ('avg_variability', 'Avg Variability', ''),
        ('high_variability_rate', 'High Variability Rate', '%'),
        ('compile_rate', 'Compile Rate', '%'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name, unit) in enumerate(metrics):
        ax = axes[idx]
        values = [results[c]['stats'][metric_key] for c in composers]
        
        if unit == '%':
            values = [v * 100 for v in values]
        
        bars = ax.bar(composers, values, alpha=0.8)
        ax.set_ylabel(f"{metric_name} ({unit})" if unit else metric_name)
        ax.set_title(metric_name)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = output_dir / 'comparison_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_size_depth(results: dict, output_dir: Path):
    """Create scatter plot of program size vs depth."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for composer, data in results.items():
        stats = data['stats']
        ax.scatter(
            [stats['avg_depth']],
            [stats['avg_size']],
            s=200,
            alpha=0.7,
            label=composer
        )
        
        # Add error bars for range
        ax.errorbar(
            [stats['avg_depth']],
            [stats['avg_size']],
            xerr=[[stats['avg_depth'] - stats['min_depth']],
                  [stats['max_depth'] - stats['avg_depth']]],
            yerr=[[stats['avg_size'] - stats['min_size']],
                  [stats['max_size'] - stats['avg_size']]],
            fmt='none',
            alpha=0.3
        )
    
    ax.set_xlabel('Average Program Depth')
    ax.set_ylabel('Average Program Size (nodes)')
    ax.set_title('Program Size vs Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'size_vs_depth.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_function_usage(results: dict, output_dir: Path):
    """Create bar charts of function usage for each composer."""
    if not HAS_MATPLOTLIB:
        return
    
    num_composers = len(results)
    fig, axes = plt.subplots(1, num_composers, figsize=(7*num_composers, 6))
    
    if num_composers == 1:
        axes = [axes]
    
    for idx, (composer, data) in enumerate(results.items()):
        ax = axes[idx]
        usage = data['stats'].get('function_usage', {})
        
        if not usage:
            continue
        
        funcs = list(usage.keys())
        counts = list(usage.values())
        
        bars = ax.barh(funcs, counts, alpha=0.8)
        ax.set_xlabel('Usage Count')
        ax.set_title(f'{composer.capitalize()} - Top Functions')
        ax.grid(axis='x', alpha=0.3)
        
        # Color bars by count
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.tight_layout()
    output_path = output_dir / 'function_usage.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_variability_distribution(results: dict, output_dir: Path):
    """Create violin/box plot of variability distribution."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    composers = list(results.keys())
    variability_data = []
    
    for composer in composers:
        # We only have aggregate stats, so approximate distribution
        stats = results[composer]['stats']
        avg = stats['avg_variability']
        # Create synthetic data points around average
        # This is just for visualization - real data would be better
        data = [avg] * 10  # Placeholder
        variability_data.append(data)
    
    positions = range(1, len(composers) + 1)
    parts = ax.violinplot(variability_data, positions=positions, showmeans=True)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(composers)
    ax.set_ylabel('Variability Score')
    ax.set_title('Program Variability Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    output_path = output_dir / 'variability_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_table(results: dict, output_dir: Path):
    """Generate markdown summary table."""
    output_path = output_dir / 'summary_table.md'
    
    with open(output_path, 'w') as f:
        f.write("# Composer Comparison Summary\n\n")
        
        # Main metrics table
        f.write("## Key Metrics\n\n")
        f.write("| Composer | Type Check | Compile | Uses Input | Avg Variability | Avg Size | Avg Depth |\n")
        f.write("|----------|------------|---------|------------|-----------------|----------|----------|\n")
        
        for composer, data in results.items():
            stats = data['stats']
            f.write(f"| {composer} | "
                   f"{stats['type_check_rate']*100:.1f}% | "
                   f"{stats['compile_rate']*100:.1f}% | "
                   f"{stats['uses_input_rate']*100:.1f}% | "
                   f"{stats['avg_variability']:.3f} | "
                   f"{stats['avg_size']:.1f} | "
                   f"{stats['avg_depth']:.1f} |\n")
        
        # Function usage table
        f.write("\n## Top Function Usage\n\n")
        for composer, data in results.items():
            f.write(f"\n### {composer.capitalize()}\n\n")
            usage = data['stats'].get('function_usage', {})
            if usage:
                f.write("| Function | Count |\n")
                f.write("|----------|-------|\n")
                for func, count in usage.items():
                    f.write(f"| `{func}` | {count} |\n")
        
        # Sample programs
        f.write("\n## Sample Programs\n\n")
        for composer, data in results.items():
            f.write(f"\n### {composer.capitalize()}\n\n")
            for i, prog in enumerate(data.get('sample_programs', [])[:3], 1):
                f.write(f"{i}. `{prog}`\n")
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize composer experiment results'
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input JSON file with experiment results'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='plots',
        help='Output directory for plots (default: plots/)'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}")
    results = load_results(args.input)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}")
    
    # Generate all visualizations
    if HAS_MATPLOTLIB:
        plot_comparison_bars(results, output_dir)
        plot_size_depth(results, output_dir)
        plot_function_usage(results, output_dir)
        plot_variability_distribution(results, output_dir)
    else:
        print("Skipping plots (matplotlib not installed)")
    
    generate_summary_table(results, output_dir)
    
    print(f"\n✓ Visualization complete!")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
