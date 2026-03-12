# Composer Experimentation Scripts

This directory contains scripts for experimenting with different program composers and analyzing their output quality and behavior.

## Available Composers

1. **Random** (`random`) - Uniform random sampling from all type-valid candidates
   - Produces syntactically valid but often semantically degenerate programs
   - Baseline composer for comparison

2. **Random Guarded** (`random_guarded`) - Random sampling with guard rules
   - Uses guard rules to prevent trivial patterns (e.g., literal booleans in if-conditions)
   - Better than random but still doesn't guarantee meaningful programs

3. **Template** (`template`) - Template-based generation with hand-tuned weights
   - Uses compositional templates (map, filter, fold, etc.)
   - Context-sensitive weights produce semantically meaningful programs
   - Tunable noise parameter for controlling randomness

4. **MCTS** (`mcts`) - Monte Carlo Tree Search with reinforcement learning
   - Learns which generation strategies produce variable programs
   - Uses variability reward to avoid degenerate patterns
   - Requires training phase before use

## Quick Testing Script

For quickly seeing what each composer produces:

```bash
# Test template composer (10 examples)
python scripts/quick_composer_test.py template

# Test MCTS composer (5 examples)
python scripts/quick_composer_test.py mcts 5

# Test random composer (20 examples)
python scripts/quick_composer_test.py random 20
```

The quick test script shows:
- Generated program syntax
- Behavior on sample inputs
- Compilation status
- Simple readability

## Full Experimentation Script

For comprehensive analysis and comparison:

### Single Composer Experiment

```bash
# Experiment with template composer (100 samples)
python scripts/experiment_composers.py --composer template --num-samples 100

# Test MCTS with custom training
python scripts/experiment_composers.py --composer mcts \
    --train-episodes 500 \
    --num-samples 50 \
    --depth 5
```

### Compare Multiple Composers

```bash
# Compare random, template, and MCTS
python scripts/experiment_composers.py \
    --compare random,template,mcts \
    --num-samples 50 \
    --depth 4

# Compare with custom parameters
python scripts/experiment_composers.py \
    --compare random_guarded,template \
    --num-samples 100 \
    --template-noise 0.1 \
    --output comparison_results.json
```

### Advanced Options

```bash
# Full comparison with all options
python scripts/experiment_composers.py \
    --compare random,random_guarded,template,mcts,empirical \
    --depth 5 \
    --num-samples 200 \
    --seed 123 \
    --train-episodes 2000 \
    --template-noise 0.0 \
    --output full_comparison.json
```

## Metrics Reported

The experiment script analyzes programs on multiple dimensions:

### Quality Metrics
- **Type Check Rate**: Percentage of programs that type-check
- **Compile Rate**: Percentage of programs that compile to executable code
- **Uses Input Rate**: Percentage that use the input variable (not constant)

### Size Metrics
- **Average Size**: Mean number of AST nodes
- **Size Range**: Min and max program sizes
- **Average Depth**: Mean AST depth

### Behavior Metrics
- **Average Variability**: How much program output varies across different inputs
- **High Variability Rate**: Percentage with variability > 0.5

### Function Usage
- Distribution of which grammar functions are used
- Top 10 most frequently used functions

### Performance
- Total generation time
- Average time per program

## Output Files

Results are saved to JSON files (default: `composer_results.json`) with:
- Aggregate statistics for each composer
- Sample programs (first 10 generated)
- Function usage distributions
- Performance metrics

## Example Workflow

1. **Quick sanity check** - See what a composer produces:
   ```bash
   python scripts/quick_composer_test.py template
   ```

2. **Small comparison** - Compare 2 composers with 50 samples:
   ```bash
   python scripts/experiment_composers.py \
       --compare random,template \
       --num-samples 50
   ```

3. **Full evaluation** - Comprehensive comparison with 200+ samples:
   ```bash
   python scripts/experiment_composers.py \
       --compare random,random_guarded,template,mcts \
       --num-samples 200 \
       --train-episodes 500 \
       --output full_eval.json
   ```

4. **Analyze results** - Load JSON and further analyze or visualize

## Interpreting Results

### What to Look For

**Random Composer:**
- Low variability (many constant/identity functions)
- Random function usage
- May have high compilation rate but poor behavior

**Template Composer:**
- High uses-input rate (>80%)
- Balanced function usage (map, filter, fold, etc.)
- Moderate to high variability
- Programs follow compositional patterns

**MCTS Composer:**
- Should improve after training
- High uses-input rate (due to constant expression penalty)
- Function usage learns from rewards
- Higher variability than random

### Red Flags

- **Type check rate < 100%**: Bug in composer (should always type-check)
- **Uses input rate < 50%**: Generating too many constant programs
- **Avg variability < 0.2**: Programs not meaningfully varying with input
- **All programs use same function**: Lack of diversity

## Customization

You can extend the scripts to:

1. **Add new composers**: Register in `ComposerExperiment.__init__`
2. **Add new metrics**: Extend `ProgramAnalyzer.analyze_program`
3. **Change target type**: Modify `target_type` in main functions
4. **Add visualization**: Use the JSON output with matplotlib/seaborn

## Troubleshooting

**Script not found:**
```bash
# Make sure you're in the repo root
cd /path/to/mlmp
python scripts/experiment_composers.py --help
```

**Import errors:**
```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python scripts/experiment_composers.py --composer template
```

**MCTS training too slow:**
```bash
# Reduce training episodes or samples
python scripts/experiment_composers.py \
    --composer mcts \
    --train-episodes 100 \
    --num-samples 20
```

## Examples from Paper

To reproduce experiments from the paper or README:

```bash
# Distribution analysis
python scripts/experiment_composers.py \
    --compare random,template \
    --num-samples 1000 \
    --depth 4 \
    --output distribution_analysis.json

# Quality comparison
python scripts/experiment_composers.py \
    --compare random_guarded,template,mcts \
    --num-samples 200 \
    --train-episodes 500 \
    --output quality_comparison.json
```
