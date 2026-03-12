#!/usr/bin/env python3
"""
Batch Experiment Runner

Runs multiple experiments with different configurations automatically.
Useful for parameter sweeps, reproducibility studies, or comprehensive evaluations.

Usage:
    python scripts/batch_experiments.py --preset quick
    python scripts/batch_experiments.py --preset full
    python scripts/batch_experiments.py --config experiments.json
"""

import sys
import os
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


PRESETS = {
    'quick': {
        'description': 'Quick comparison for testing',
        'experiments': [
            {
                'name': 'quick_random_vs_template',
                'composers': ['random', 'template'],
                'num_samples': 30,
                'depth': 4,
                'seed': 42,
            }
        ]
    },
    
    'standard': {
        'description': 'Standard comparison with all composers',
        'experiments': [
            {
                'name': 'standard_comparison',
                'composers': ['random', 'random_guarded', 'template', 'mcts', 'empirical'],
                'num_samples': 100,
                'depth': 4,
                'seed': 42,
                'train_episodes': 1000,
            }
        ]
    },
    
    'full': {
        'description': 'Comprehensive evaluation suite',
        'experiments': [
            {
                'name': 'full_comparison',
                'composers': ['random', 'random_guarded', 'template', 'mcts', 'empirical'],
                'num_samples': 200,
                'depth': 4,
                'seed': 42,
                'train_episodes': 1000,
            },
            {
                'name': 'depth_sweep_shallow',
                'composers': ['random', 'template'],
                'num_samples': 100,
                'depth': 3,
                'seed': 42,
            },
            {
                'name': 'depth_sweep_deep',
                'composers': ['random', 'template'],
                'num_samples': 100,
                'depth': 6,
                'seed': 42,
            },
        ]
    },
    
    'reproducibility': {
        'description': 'Multiple seeds for reproducibility analysis',
        'experiments': [
            {
                'name': f'seed_{seed}',
                'composers': ['random', 'template', 'mcts', 'empirical'],
                'num_samples': 100,
                'depth': 4,
                'seed': seed,
                'train_episodes': 1000,
            }
            for seed in [42, 123, 456, 789, 1337]
        ]
    },
    
    'parameter_sweep': {
        'description': 'Sweep over different parameters',
        'experiments': [
            # Vary number of samples
            {
                'name': f'samples_{n}',
                'composers': ['template'],
                'num_samples': n,
                'depth': 4,
                'seed': 42,
            }
            for n in [50, 100, 200, 500]
        ] + [
            # Vary depth
            {
                'name': f'depth_{d}',
                'composers': ['template'],
                'num_samples': 100,
                'depth': d,
                'seed': 42,
            }
            for d in [3, 4, 5, 6]
        ] + [
            # Vary MCTS training episodes
            {
                'name': f'mcts_train_{episodes}',
                'composers': ['mcts'],
                'num_samples': 50,
                'depth': 4,
                'seed': 42,
                'train_episodes': episodes,
            }
            for episodes in [200, 500, 1000, 2000]
        ]
    }
}


def run_experiment(exp_config: dict, output_dir: Path) -> dict:
    """
    Run a single experiment.
    
    Returns:
        Dictionary with experiment results and metadata
    """
    exp_name = exp_config['name']
    print(f"\n{'='*70}")
    print(f"Running: {exp_name}")
    print(f"{'='*70}")
    
    # Build command
    cmd = ['python', 'scripts/experiment_composers.py']
    
    # Add composers
    composers = ','.join(exp_config['composers'])
    cmd.extend(['--compare', composers])
    
    # Add parameters
    cmd.extend(['--num-samples', str(exp_config['num_samples'])])
    cmd.extend(['--depth', str(exp_config['depth'])])
    cmd.extend(['--seed', str(exp_config['seed'])])
    
    if 'train_episodes' in exp_config:
        cmd.extend(['--train-episodes', str(exp_config['train_episodes'])])
    
    if 'template_noise' in exp_config:
        cmd.extend(['--template-noise', str(exp_config['template_noise'])])
    
    # Output file
    output_file = output_dir / f"{exp_name}.json"
    cmd.extend(['--output', str(output_file)])
    
    # Run experiment
    print(f"Command: {' '.join(cmd)}")
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(result.stdout)
        
        return {
            'name': exp_name,
            'config': exp_config,
            'output_file': str(output_file),
            'status': 'success',
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }
    
    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"ERROR: Experiment failed")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        
        return {
            'name': exp_name,
            'config': exp_config,
            'status': 'failed',
            'error': str(e),
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat()
        }


def run_batch(preset_name: str, output_dir: Path) -> dict:
    """
    Run a batch of experiments from a preset.
    
    Returns:
        Dictionary with batch results
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    print(f"\n{'='*70}")
    print(f"Batch: {preset_name}")
    print(f"Description: {preset['description']}")
    print(f"Experiments: {len(preset['experiments'])}")
    print(f"{'='*70}")
    
    results = []
    for exp_config in preset['experiments']:
        result = run_experiment(exp_config, output_dir)
        results.append(result)
    
    # Save batch metadata
    batch_metadata = {
        'preset': preset_name,
        'description': preset['description'],
        'num_experiments': len(results),
        'experiments': results,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_file = output_dir / 'batch_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Batch Complete")
    print(f"{'='*70}")
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Metadata saved to: {metadata_file}")
    
    return batch_metadata


def run_custom_config(config_file: str, output_dir: Path):
    """Run experiments from custom JSON config file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    experiments = config.get('experiments', [])
    
    print(f"\n{'='*70}")
    print(f"Custom Config: {config_file}")
    print(f"Experiments: {len(experiments)}")
    print(f"{'='*70}")
    
    results = []
    for exp_config in experiments:
        result = run_experiment(exp_config, output_dir)
        results.append(result)
    
    # Save results
    batch_metadata = {
        'config_file': config_file,
        'num_experiments': len(results),
        'experiments': results,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_file = output_dir / 'batch_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    print(f"\nBatch complete. Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run batch experiments with different configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available presets:
{chr(10).join(f'  {name:20s} - {preset["description"]}' for name, preset in PRESETS.items())}

Examples:
  # Run quick test batch
  python scripts/batch_experiments.py --preset quick
  
  # Run full evaluation
  python scripts/batch_experiments.py --preset full --output results/full/
  
  # Run custom configuration
  python scripts/batch_experiments.py --config my_experiments.json
  
Custom config JSON format:
{{
  "experiments": [
    {{
      "name": "exp1",
      "composers": ["random", "template"],
      "num_samples": 100,
      "depth": 4,
      "seed": 42
    }}
  ]
}}
        """
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=list(PRESETS.keys()),
        help='Preset experiment batch to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Custom JSON config file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='batch_results',
        help='Output directory for results (default: batch_results/)'
    )
    
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available presets and exit'
    )
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        print("\nAvailable Presets:")
        print("="*70)
        for name, preset in PRESETS.items():
            print(f"\n{name}:")
            print(f"  {preset['description']}")
            print(f"  Experiments: {len(preset['experiments'])}")
        return
    
    # Validate arguments
    if not args.preset and not args.config:
        parser.error("Must specify either --preset or --config")
    
    if args.preset and args.config:
        parser.error("Cannot specify both --preset and --config")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Run batch
    if args.preset:
        run_batch(args.preset, output_dir)
    else:
        run_custom_config(args.config, output_dir)


if __name__ == '__main__':
    main()
