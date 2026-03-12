# Composer Experimentation Scripts - Complete Guide

I've created comprehensive scripts for experimenting with and comparing different program composers. Here's what you have:

## 📁 Scripts Overview

### 1. **Quick Test Script** (`quick_composer_test.py`)
Fast way to see what each composer generates with live examples.

```bash
# See 10 examples from template composer
python scripts/quick_composer_test.py template

# See 5 examples from MCTS (includes training)
python scripts/quick_composer_test.py mcts 5

# Compare random vs template behavior
python scripts/quick_composer_test.py random 3
python scripts/quick_composer_test.py template 3
```

**Shows:**
- Generated program syntax
- Behavior on sample inputs `[1,2,3]`, `[10,20,30]`, `[]`
- Compilation errors if any
- Easy visual comparison

### 2. **Full Experiment Script** (`experiment_composers.py`)
Comprehensive analysis with statistics and comparisons.

```bash
# Single composer analysis (100 samples)
python scripts/experiment_composers.py --composer template --num-samples 100

# Compare multiple composers
python scripts/experiment_composers.py \
    --compare random,template,mcts \
    --num-samples 50 \
    --depth 4

# Full comparison with custom parameters
python scripts/experiment_composers.py \
    --compare random,random_guarded,template,mcts \
    --num-samples 200 \
    --train-episodes 500 \
    --output full_comparison.json
```

**Provides:**
- Type check rate, compile rate, uses input rate
- Program size and depth statistics
- Output variability analysis
- Function usage distributions
- Performance metrics (generation time)
- Comparison tables across composers
- JSON output for further analysis

### 3. **Visualization Script** (`visualize_results.py`)
Generate plots and summary tables from experiment results.

```bash
# Generate visualizations from experiment results
python scripts/visualize_results.py results.json --output plots/

# Creates:
# - comparison_metrics.png (bar charts)
# - size_vs_depth.png (scatter plot)
# - function_usage.png (usage distributions)
# - variability_distribution.png (violin plots)
# - summary_table.md (markdown summary)
```

## 🎯 Available Composers

| Composer | Description | Best For |
|----------|-------------|----------|
| `random` | Uniform random sampling | Baseline comparison |
| `random_guarded` | Random with guard rules | Testing guard effectiveness |
| `template` | Template-based with hand-tuned weights | Meaningful programs, benchmark generation |
| `mcts` | Monte Carlo Tree Search + RL | Learning from rewards, adaptive generation |
| `empirical` | Learns from example program corpus | Domain-specific generation, pattern matching |

## 📊 Metrics Explained

### Quality Metrics
- **Type Check Rate**: Should be 100% (all programs type-check)
- **Compile Rate**: Programs that compile to executable code
- **Uses Input Rate**: Programs that use input variable (not constant)

### Behavior Metrics
- **Avg Variability**: Score 0-1 measuring output diversity across inputs
  - 0.0 = constant output (bad)
  - 0.5 = moderate variation
  - 1.0 = maximum variation (good)
- **High Variability Rate**: % of programs with variability > 0.5

### Size Metrics
- **Program Size**: Number of AST nodes
- **Program Depth**: Maximum nesting depth

## 🔬 Example Workflows

### Quick Sanity Check
```bash
# See what template composer produces
python scripts/quick_composer_test.py template 5
```

### Small-Scale Comparison
```bash
# Compare random vs template (50 samples each)
python scripts/experiment_composers.py \
    --compare random,template \
    --num-samples 50 \
    --output comparison.json

# Visualize results
python scripts/visualize_results.py comparison.json
```

### Full Evaluation
```bash
# Comprehensive comparison (200+ samples)
python scripts/experiment_composers.py \
    --compare random,random_guarded,template,mcts \
    --num-samples 200 \
    --train-episodes 500 \
    --depth 5 \
    --output full_eval.json

# Generate plots and summary
python scripts/visualize_results.py full_eval.json --output figures/
```

### Custom Experiments

**Test MCTS with more training:**
```bash
python scripts/experiment_composers.py \
    --composer mcts \
    --train-episodes 1000 \
    --num-samples 100
```

**Test template with noise:**
```bash
python scripts/experiment_composers.py \
    --composer template \
    --template-noise 0.2 \
    --num-samples 100
```

**Test at different depths:**
```bash
# Shallow programs
python scripts/experiment_composers.py --compare random,template --depth 3 --num-samples 50

# Deep programs  
python scripts/experiment_composers.py --compare random,template --depth 6 --num-samples 50
```

## 📈 Expected Results

Based on the test run with 20 samples:

### Random Composer
- ❌ Low variability: 0.10 (only 10% high variability)
- 📏 Large programs: 38.8 nodes average
- 🎲 Uses many functions randomly
- ⚠️ Often generates runtime errors (index out of bounds, etc.)

### Template Composer
- ✅ High variability: 0.90 (95% high variability)
- 📏 Compact programs: 9.8 nodes average
- 🎯 Uses meaningful functions (map, filter, fold, cons)
- ✅ Generates semantic patterns (reverse via fold, etc.)

### MCTS Composer (after training)
- 🎓 Learns to avoid constant expressions
- 📈 Improves variability over time
- 🎯 Adapts function usage based on rewards
- 🔄 Now trains for 1000 episodes by default (was 200)
- ⚡ More training = better quality programs

### Empirical Composer
- 📊 Learns from example programs
- 🎯 Matches patterns in training data
- 🔄 Falls back to random when no training data
- 📈 Quality depends on training corpus

## 🛠️ Customization

### Add New Metrics
Edit `ProgramAnalyzer.analyze_program()` in `experiment_composers.py`:

```python
def analyze_program(self, program: ASTNode) -> dict:
    metrics = {}
    # ... existing metrics ...
    
    # Add your custom metric
    metrics['my_metric'] = self._compute_my_metric(program)
    return metrics
```

### Add New Visualizations
Edit `visualize_results.py`:

```python
def plot_my_visualization(results: dict, output_dir: Path):
    # Your plotting code
    pass
```

### Change Target Type
Edit the `target_type` variable in script main functions:

```python
# Current: list[int] -> list[int]
target_type = Callable[[list[int]], list[int]]

# Change to: int -> int
target_type = Callable[[int], int]

# Or: list[int] -> int
target_type = Callable[[list[int]], int]
```

## 🐛 Troubleshooting

**Import errors:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**MCTS training too slow:**
- Reduce `--train-episodes` (try 100 instead of 500)
- Reduce `--num-samples`
- Run on smaller `--depth`

**Matplotlib not found:**
```bash
pip install matplotlib numpy
```

## 📝 Output Files

### JSON Format
```json
{
  "composer_name": {
    "composer": "template",
    "stats": {
      "num_programs": 50,
      "type_check_rate": 0.98,
      "compile_rate": 1.0,
      "avg_variability": 0.85,
      ...
    },
    "sample_programs": ["(λ (x) ...)", ...]
  }
}
```

### Generated Files
- `composer_results.json` - Experiment data
- `plots/comparison_metrics.png` - Bar charts
- `plots/size_vs_depth.png` - Scatter plot
- `plots/function_usage.png` - Usage distributions
- `plots/summary_table.md` - Markdown summary

## 🎓 For Research/Papers

If using these scripts for research:

1. **Document parameters**: Save all flags used
2. **Use fixed seeds**: For reproducibility
3. **Generate multiple runs**: Average over different seeds
4. **Save raw data**: Keep JSON files for later analysis
5. **Document environment**: Python version, dependencies

Example reproducible experiment:
```bash
# Documented, reproducible experiment
python scripts/experiment_composers.py \
    --compare random,template,mcts \
    --num-samples 500 \
    --depth 4 \
    --seed 42 \
    --train-episodes 1000 \
    --output experiment_seed42.json \
    > experiment_seed42.log 2>&1
```

## 📚 Further Reading

- See `README_EXPERIMENTS.md` for detailed documentation
- Check test files in `tests/` for usage examples
- Read composer implementations in `src/lang/composers/`

---

**Quick Reference:**
- Quick test: `python scripts/quick_composer_test.py template`
- Full experiment: `python scripts/experiment_composers.py --compare random,template --num-samples 50`
- Visualize: `python scripts/visualize_results.py results.json`
