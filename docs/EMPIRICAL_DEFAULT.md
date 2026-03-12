# Empirical Composer - Default Training Data

The empirical composer now uses a default training file with 250 example programs.

## Default Training File

**Location:** `src/data/rule/functions.txt`

**Content:** 250 hand-crafted programs from the Rule-MPS-DSL benchmark, including:
- List manipulation (map, filter, fold, zip)
- Structural operations (slice, swap, cut, insert)
- Aggregations (max, min, sum, length)
- Higher-order functions with lambdas
- Complex compositions

**Source:** Rule-MPS-DSL benchmark (already in repo)

## Usage

### With Default File (Automatic)

The empirical composer now automatically uses the default training file:

```bash
# Quick test - uses default file automatically
python scripts/quick_composer_test.py empirical 5

# Experiment - uses default file automatically
python scripts/experiment_composers.py --composer empirical --num-samples 50

# Compare - includes empirical with default training
python scripts/experiment_composers.py --compare template,empirical --num-samples 50
```

### With Custom File

You can still provide your own training file:

```bash
python scripts/experiment_composers.py \
    --composer empirical \
    --empirical-file path/to/my_programs.txt \
    --num-samples 100
```

## Training File Format

One program per line, in s-expression format:

```
(λ (x) (map (λ (y) (+ y 1)) x))
(λ (x) (filter (λ (y) (> y 0)) x))
(λ (x) (fold (λ (y z) (cons z y)) [] x))
...
```

## Expected Performance

With the default training file (250 programs):

| Metric | Expected Value |
|--------|---------------|
| Uses Input Rate | 85% |
| Variability | 0.53 |
| High Var Rate | 50% |
| Avg Size | ~7 nodes |
| Generation Speed | ~1.5ms/program |

## Comparison with Other Composers

From testing with 20 samples:

| Composer | Variability | Uses Input | Avg Size | Common Functions |
|----------|-------------|------------|----------|------------------|
| template | 0.90 | 100% | 9.8 | cons, first, append, singleton |
| **empirical** | **0.53** | **85%** | **7.2** | **reverse, cons, singleton, filteri** |

The empirical composer:
- ✅ Generates real patterns from training data
- ✅ More compact than template (7.2 vs 9.8 nodes)
- ✅ Good variability (0.53, between random 0.10 and template 0.90)
- ✅ Fast generation (~1.5ms per program)
- ✅ Learns function distributions from examples

## Creating Custom Training Files

### From Template Composer

Generate training data using the template composer:

```python
from lang.composers.template import TemplateComposer
from lang.grammar import DefaultGrammar
from lang.type_utils import SubstitutionTable
from lang.ast_nodes import pretty_print
from typing import Callable

composer = TemplateComposer(seed=42, grammar=DefaultGrammar)
target_type = Callable[[list[int]], list[int]]

with open('my_training.txt', 'w') as f:
    for i in range(500):
        composer.reset_var_counter()
        program = composer.generate(target_type, depth=4, context={}, substitutions=SubstitutionTable())
        f.write(pretty_print(program) + '\n')
```

### From Existing Programs

Convert your existing programs to the required format:

```python
from lang.ast_nodes import pretty_print
from lang.parser import parse

# Load and normalize your programs
with open('my_programs.txt') as f:
    programs = [line.strip() for line in f]

with open('normalized.txt', 'w') as f:
    for prog_str in programs:
        try:
            ast = parse(prog_str)
            normalized = pretty_print(ast)
            f.write(normalized + '\n')
        except:
            pass  # Skip invalid programs
```

## File Location

Uses the existing benchmark file directly:

```
Location:    src/data/rule/functions.txt
Programs:    250
Type:        Callable[[list[int]], list[int]]
```

## Notes

- The empirical composer learns statistical patterns from the training data
- Quality depends on the size and diversity of the training corpus
- Default file contains 250 high-quality programs from the benchmark
- Falls back to random generation when no training data or unseen contexts
- Automatically loaded when you use the empirical composer

## Testing

Verify the default file is working:

```bash
# Check file exists
ls -lh src/data/rule/functions.txt

# Count programs
wc -l src/data/rule/functions.txt

# Test composer
python scripts/quick_composer_test.py empirical 5
```

You should see:
```
Description: Learns distributions from example programs
Training file: src/data/rule/functions.txt (250 programs)
Expected: Programs matching learned patterns from training data
```
