# Composer Experimentation Scripts - Complete Summary

This directory contains a comprehensive suite of scripts for experimenting with program composers, analyzing their output quality, and comparing their behavior.

## 📋 All Scripts

### Core Scripts

1. **`quick_composer_test.py`** - Quick testing with live examples
2. **`experiment_composers.py`** - Comprehensive statistical analysis
3. **`visualize_results.py`** - Generate plots and tables from results
4. **`batch_experiments.py`** - Run multiple experiments automatically

### Documentation

- **`README_EXPERIMENTS.md`** - Detailed technical documentation
- **`USAGE_GUIDE.md`** - Complete usage guide with examples
- **`SCRIPTS_SUMMARY.md`** - This file

## 🚀 Quick Start

### 1. See what a composer produces
```bash
python scripts/quick_composer_test.py template 5
```

### 2. Compare composers statistically
```bash
python scripts/experiment_composers.py \
    --compare random,template \
    --num-samples 50 \
    --output results.json
```

### 3. Visualize results
```bash
python scripts/visualize_results.py results.json --output plots/
```

### 4. Run batch experiments
```bash
python scripts/batch_experiments.py --preset quick
```

## 📊 What Each Script Does

### `quick_composer_test.py`

**Purpose:** Quickly see what a composer generates with live behavior examples

**Input:**
- Composer name (random, random_guarded, template, mcts)
- Number of examples (default: 10)

**Output:**
- Program syntax for each example
- Behavior on sample inputs `[1,2,3]`, `[10,20,30]`, `[]`
- Compilation status

**When to use:**
- Quick sanity check of a composer
- Visually comparing different composers
- Debugging program generation issues
- Demonstrating composer behavior

**Example:**
```bash
python scripts/quick_composer_test.py template 3
```

**Output:**
```
Program 1:
  (λ (x) (cons 94 x))
  Behavior:
    [1, 2, 3] -> [94, 1, 2, 3]
    [10, 20, 30] -> [94, 10, 20, 30]
    [] -> [94]
```

---

### `experiment_composers.py`

**Purpose:** Comprehensive statistical analysis and comparison

**Input:**
- Single composer or comparison list
- Number of samples per composer
- Generation parameters (depth, seed, etc.)
- Composer-specific parameters (MCTS training, template noise)

**Output:**
- JSON file with detailed statistics
- Console output with comparison tables
- Metrics: type check rate, compile rate, uses input rate, variability, size, depth, function usage

**When to use:**
- Rigorous comparison of composers
- Generating data for papers/reports
- Parameter tuning and optimization
- Large-scale evaluation

**Example:**
```bash
python scripts/experiment_composers.py \
    --compare random,template,mcts \
    --num-samples 100 \
    --train-episodes 500 \
    --output full_comparison.json
```

**Output (console):**
```
Comparison Summary
================================================================
Metric              random       template          mcts
----------------------------------------------------------------
Type Check Rate     100.0%         100.0%        100.0%
Compile Rate        100.0%         100.0%        100.0%
Uses Input Rate      95.0%         100.0%         98.0%
Avg Variability       0.10           0.90          0.75
High Var Rate        10.0%          95.0%         85.0%
```

---

### `visualize_results.py`

**Purpose:** Generate plots and tables from experiment JSON results

**Input:**
- JSON file from `experiment_composers.py`
- Output directory for plots

**Output:**
- `comparison_metrics.png` - Bar charts of key metrics
- `size_vs_depth.png` - Scatter plot with error bars
- `function_usage.png` - Bar charts of function distributions
- `variability_distribution.png` - Violin plots
- `summary_table.md` - Markdown summary table

**When to use:**
- Creating figures for papers/presentations
- Visual analysis of results
- Generating summary tables
- Sharing results with collaborators

**Example:**
```bash
python scripts/visualize_results.py results.json --output figures/
```

**Requirements:**
- matplotlib
- numpy

If not installed: `pip install matplotlib numpy`

---

### `batch_experiments.py`

**Purpose:** Run multiple experiments with different configurations automatically

**Input:**
- Preset name (quick, standard, full, reproducibility, parameter_sweep)
- OR custom JSON config file
- Output directory

**Output:**
- Individual JSON files for each experiment
- `batch_metadata.json` with batch summary
- Console logs for all experiments

**When to use:**
- Parameter sweeps (vary depth, samples, training episodes)
- Reproducibility studies (multiple seeds)
- Comprehensive evaluations (multiple configurations)
- Automated testing pipelines

**Available Presets:**
- **quick**: 1 experiment, 30 samples (for testing)
- **standard**: 1 experiment, 100 samples, all composers
- **full**: 3 experiments, 200 samples, depth sweep
- **reproducibility**: 5 experiments with different seeds
- **parameter_sweep**: 12 experiments varying parameters

**Example:**
```bash
# Run quick preset
python scripts/batch_experiments.py --preset quick

# Run full evaluation suite
python scripts/batch_experiments.py --preset full --output results/

# Custom configuration
python scripts/batch_experiments.py --config my_config.json
```

**Custom Config Format:**
```json
{
  "experiments": [
    {
      "name": "experiment_1",
      "composers": ["random", "template"],
      "num_samples": 100,
      "depth": 4,
      "seed": 42,
      "train_episodes": 200
    }
  ]
}
```

## 🔄 Typical Workflows

### Research Paper Workflow

1. **Exploration** - See what composers produce
   ```bash
   python scripts/quick_composer_test.py template
   python scripts/quick_composer_test.py random
   ```

2. **Pilot Study** - Small-scale comparison
   ```bash
   python scripts/experiment_composers.py \
       --compare random,template \
       --num-samples 50
   ```

3. **Full Evaluation** - Large-scale with multiple seeds
   ```bash
   python scripts/batch_experiments.py --preset reproducibility
   ```

4. **Analysis** - Generate figures
   ```bash
   for file in batch_results/*/*.json; do
       python scripts/visualize_results.py "$file" --output "figures/$(basename $file .json)/"
   done
   ```

### Parameter Tuning Workflow

1. **Baseline** - Test default parameters
   ```bash
   python scripts/experiment_composers.py --composer template --num-samples 100
   ```

2. **Sweep** - Try different parameters
   ```bash
   python scripts/batch_experiments.py --preset parameter_sweep
   ```

3. **Compare** - Analyze results
   ```bash
   python scripts/visualize_results.py batch_results/*/samples_*.json
   ```

### Quick Debugging Workflow

1. **Generate examples**
   ```bash
   python scripts/quick_composer_test.py template 10
   ```

2. **Check specific issue**
   ```bash
   # Modify quick_composer_test.py to test specific case
   python scripts/quick_composer_test.py template 1
   ```

## 📈 Metrics Reference

### Quality Metrics
| Metric | Range | Good Value | Description |
|--------|-------|------------|-------------|
| Type Check Rate | 0-100% | 100% | Programs that type-check correctly |
| Compile Rate | 0-100% | 100% | Programs that compile to executable code |
| Uses Input Rate | 0-100% | >80% | Programs that use input variable (not constant) |

### Behavior Metrics
| Metric | Range | Good Value | Description |
|--------|-------|------------|-------------|
| Avg Variability | 0-1 | >0.5 | How much output varies across inputs |
| High Var Rate | 0-100% | >50% | Percentage with variability > 0.5 |

### Size Metrics
| Metric | Range | Good Value | Description |
|--------|-------|------------|-------------|
| Avg Size | 1-∞ | 10-30 | Average number of AST nodes |
| Avg Depth | 1-∞ | 3-6 | Average nesting depth |

## 🎯 Expected Benchmark Results

Based on testing with `Callable[[list[int]], list[int]]` at depth 4:

| Composer | Variability | Uses Input | Avg Size | Common Functions |
|----------|-------------|------------|----------|------------------|
| Random | 0.10 | 95% | 38.8 | Scattered, many structural ops |
| Random Guarded | 0.15 | 95% | 35.0 | Similar to random, fewer trivial patterns |
| Template | 0.90 | 100% | 9.8 | map, filter, fold, cons, singleton |
| MCTS (trained) | 0.75 | 98% | 15.0 | Learned from rewards, adaptive |

## 🛠️ Extending the Scripts

### Add a New Composer

1. Implement composer in `src/lang/composers/`
2. Register in `experiment_composers.py`:
   ```python
   from lang.composers.my_composer import MyComposer
   
   experiment.register_composer(
       'my_composer',
       MyComposer(seed=args.seed, grammar=DefaultGrammar)
   )
   ```

### Add a New Metric

1. Edit `ProgramAnalyzer.analyze_program()` in `experiment_composers.py`
2. Add visualization in `visualize_results.py`

### Add a New Preset

Edit `batch_experiments.py`:
```python
PRESETS['my_preset'] = {
    'description': 'My custom preset',
    'experiments': [
        {
            'name': 'my_experiment',
            'composers': ['template'],
            'num_samples': 100,
            'depth': 4,
            'seed': 42,
        }
    ]
}
```

## 📦 Dependencies

**Required:**
- Python 3.10+
- Core mlmp dependencies (see main README)

**Optional (for visualization):**
- matplotlib
- numpy

Install with:
```bash
pip install matplotlib numpy
```

## 💡 Tips

1. **Start small** - Use quick test script first
2. **Use fixed seeds** - For reproducibility
3. **Save results** - JSON files are your data
4. **Visualize early** - Plots reveal patterns
5. **Document parameters** - Keep track of what you tried
6. **Batch experiments** - Run overnight for large studies

## 🐛 Common Issues

**"Module not found"**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**MCTS training slow**
- Reduce `--train-episodes`
- Reduce `--num-samples`
- Use smaller `--depth`

**Out of memory**
- Reduce `--num-samples`
- Process results in batches
- Use `--depth 3` or `--depth 4`

## 📚 See Also

- Main README: Project overview and setup
- `README_EXPERIMENTS.md`: Detailed technical documentation
- `USAGE_GUIDE.md`: Complete usage guide with examples
- Test files: `tests/test_mcts_composer.py`, etc.

---

**Quick Commands:**
```bash
# Quick test
python scripts/quick_composer_test.py template

# Compare composers
python scripts/experiment_composers.py --compare random,template --num-samples 50

# Visualize
python scripts/visualize_results.py results.json

# Batch run
python scripts/batch_experiments.py --preset quick
```
