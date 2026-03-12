# Quick Reference: Composer Experimentation

## 🚀 Quick Commands

### Test Single Composer
```bash
python scripts/quick_composer_test.py template 5
python scripts/quick_composer_test.py mcts 3        # 500 episodes training
python scripts/quick_composer_test.py empirical 5   # Uses random fallback
```

### Compare Composers (NEW: includes empirical)
```bash
python scripts/experiment_composers.py \
    --compare random,template,mcts,empirical \
    --num-samples 50
```

### MCTS with Custom Training (NEW: default 1000 episodes)
```bash
python scripts/experiment_composers.py \
    --composer mcts \
    --train-episodes 2000 \
    --num-samples 100
```

### Empirical with Data (NEW)
```bash
python scripts/experiment_composers.py \
    --composer empirical \
    --empirical-file programs.txt \
    --num-samples 100
```

## 📊 Available Composers (5 total)

| Name | Training | Speed | Quality | Description |
|------|----------|-------|---------|-------------|
| `random` | None | ⚡⚡⚡ | ⭐ | Baseline (low variability) |
| `random_guarded` | None | ⚡⚡⚡ | ⭐⭐ | With guard rules |
| `template` | None | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Hand-tuned patterns |
| `mcts` | **1000 episodes** | ⚡ | ⭐⭐⭐⭐ | Reinforcement learning |
| `empirical` | Data loading | ⚡⚡⚡ | ⭐⭐⭐⭐ | Learns from corpus |

## 🎯 Training Duration (NEW DEFAULTS)

| Script | MCTS Default | Can Override |
|--------|--------------|--------------|
| `quick_composer_test.py` | 500 episodes | No |
| `experiment_composers.py` | 1000 episodes | `--train-episodes N` |
| `batch_experiments.py` (standard) | 1000 episodes | Edit preset |
| `batch_experiments.py` (sweep) | 200-2000 | Edit preset |

## 📦 Batch Presets (ALL UPDATED)

```bash
# Quick test (1 experiment, 30 samples)
python scripts/batch_experiments.py --preset quick

# Standard (1 experiment, 100 samples, 5 composers)
python scripts/batch_experiments.py --preset standard

# Full (3 experiments, 200 samples, depth sweep)
python scripts/batch_experiments.py --preset full

# Reproducibility (5 seeds, 4 composers)
python scripts/batch_experiments.py --preset reproducibility

# Parameter sweep (12 experiments, MCTS: 200-2000)
python scripts/batch_experiments.py --preset parameter_sweep
```

## 🔍 Common Workflows

### Workflow 1: Quick Check
```bash
# See what each composer produces
python scripts/quick_composer_test.py template 3
python scripts/quick_composer_test.py mcts 3
python scripts/quick_composer_test.py empirical 3
```

### Workflow 2: Compare All
```bash
# Small comparison
python scripts/experiment_composers.py \
    --compare random,template,mcts,empirical \
    --num-samples 50 \
    --output comparison.json

# Visualize
python scripts/visualize_results.py comparison.json
```

### Workflow 3: MCTS Study
```bash
# Test different training durations
python scripts/experiment_composers.py --composer mcts --train-episodes 500 --num-samples 30 --output mcts_500.json
python scripts/experiment_composers.py --composer mcts --train-episodes 1000 --num-samples 30 --output mcts_1000.json
python scripts/experiment_composers.py --composer mcts --train-episodes 2000 --num-samples 30 --output mcts_2000.json
```

### Workflow 4: Production Evaluation
```bash
# Comprehensive evaluation
python scripts/batch_experiments.py --preset full --output results/

# Generate all plots
for file in results/*/*.json; do
    python scripts/visualize_results.py "$file" --output "plots/$(basename $file .json)/"
done
```

## 📈 Expected Performance

### With 1000 Episode MCTS Training (NEW)

```
Metric                Random  Template  MCTS      Empirical
─────────────────────────────────────────────────────────────
Variability           0.10    0.90      0.75-0.80 varies
Uses Input Rate       95%     100%      98%       varies
Avg Program Size      38      10        15        varies
Training Time         0s      0s        ~40s      ~1s
Generation Time/prog  3ms     <1ms      <1ms      <1ms
High Var Rate         10%     95%       85%       varies
```

### Quality Improvements vs 200 Episodes

```
MCTS Metric           200 Ep    1000 Ep   Improvement
────────────────────────────────────────────────────
Variability           0.65      0.78      +20%
Uses Input Rate       96%       98%       +2%
High Var Rate         75%       85%       +10%
Tree Size             ~100      ~200      (expected)
```

## ⚙️ Key Parameters

```bash
--composer NAME              # random, random_guarded, template, mcts, empirical
--compare NAMES              # Comma-separated list
--num-samples N              # Programs per composer (default: 50)
--depth N                    # Max program depth (default: 4)
--seed N                     # Random seed (default: 42)
--train-episodes N           # MCTS training (default: 1000)
--template-noise F           # Template noise 0-1 (default: 0.0)
--empirical-file PATH        # Program corpus for empirical
--output FILE                # JSON output file
```

## 💡 Tips

1. **Start small**: Test with 10-20 samples first
2. **Use fixed seeds**: For reproducibility
3. **Save results**: JSON files are your data
4. **Compare incrementally**: Start with 2-3 composers
5. **Longer MCTS**: Use 2000+ episodes for production
6. **Empirical needs data**: Provide program file for best results

## 🐛 Troubleshooting

**MCTS training slow?**
```bash
--train-episodes 100  # Quick test
```

**Need empirical composer?**
```bash
--empirical-file path/to/programs.txt
```

**Out of memory?**
```bash
--num-samples 20 --depth 3
```

## 📚 Documentation

- `README.md` - Overview
- `README_EXPERIMENTS.md` - Technical details
- `USAGE_GUIDE.md` - Complete guide
- `SCRIPTS_SUMMARY.md` - All scripts
- `UPDATE_SUMMARY.md` - Recent changes

---

**Most Common Commands:**

```bash
# Quick test
python scripts/quick_composer_test.py template

# Full comparison (5 composers)
python scripts/experiment_composers.py --compare random,template,mcts,empirical --num-samples 50

# Batch run
python scripts/batch_experiments.py --preset standard
```
