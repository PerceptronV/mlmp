# Scripts Directory

This directory contains experimentation and analysis scripts for comparing program composers.

## 🎯 Quick Start

```bash
# See what template composer generates (5 examples)
python scripts/quick_composer_test.py template 5

# Compare random vs template (50 samples each)
python scripts/experiment_composers.py --compare random,template --num-samples 50

# Generate plots from results
python scripts/visualize_results.py results.json --output plots/

# Run batch experiments
python scripts/batch_experiments.py --preset quick
```

## 📁 Scripts

| Script | Purpose |
|--------|---------|
| `quick_composer_test.py` | Quick testing with live examples |
| `experiment_composers.py` | Comprehensive statistical analysis |
| `visualize_results.py` | Generate plots and tables |
| `batch_experiments.py` | Run multiple experiments |

## 📚 Documentation

- **[SCRIPTS_SUMMARY.md](SCRIPTS_SUMMARY.md)** - Complete overview of all scripts
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage guide with examples
- **[README_EXPERIMENTS.md](README_EXPERIMENTS.md)** - Technical documentation

## 🚀 Common Tasks

**See composer output:**
```bash
python scripts/quick_composer_test.py template
```

**Compare composers:**
```bash
python scripts/experiment_composers.py \
    --compare random,template,mcts \
    --num-samples 100 \
    --output results.json
```

**Visualize results:**
```bash
python scripts/visualize_results.py results.json
```

**Run preset experiments:**
```bash
python scripts/batch_experiments.py --preset standard
```

## 📖 Learn More

- See [SCRIPTS_SUMMARY.md](SCRIPTS_SUMMARY.md) for complete documentation
- See [USAGE_GUIDE.md](USAGE_GUIDE.md) for workflow examples
- Run any script with `--help` for options

## 🎓 Available Composers

- `random` - Uniform random sampling (baseline)
- `random_guarded` - Random with guard rules
- `template` - Template-based with hand-tuned weights
- `mcts` - Monte Carlo Tree Search with RL

## 📊 Output

Scripts generate:
- JSON files with statistics
- PNG plots (requires matplotlib)
- Markdown summary tables
- Console comparison tables
