# Program Variability Visualization Feature

## Overview

The experiment script now includes a `--visualize` mode that generates programs and displays them **ordered by variability**, making it easy to see the range of behaviors a composer produces.

## Usage

### Basic Command

```bash
python scripts/experiment_composers.py --composer <name> --num-samples <N> --visualize
```

### Arguments

- `--composer`: Required. Choose from `random`, `random_guarded`, `template`, `mcts`, or `empirical`
- `--num-samples`: Number of programs to generate (default: 50)
- `--visualize`: Enable visualization mode (shows programs ordered by variability)
- `--show-count`: Optional. Limit display to top N programs (default: show all)
- `--depth`: Program depth (default: 4)
- `--seed`: Random seed (default: 42)

### Examples

**View all programs from random composer:**
```bash
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer random --num-samples 30 --visualize
```

**View top 10 programs from template composer:**
```bash
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer template --num-samples 50 --visualize --show-count 10
```

**Compare different composers:**
```bash
# Random composer (low variability)
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer random --num-samples 20 --visualize --show-count 5

# Template composer (high variability)
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer template --num-samples 20 --visualize --show-count 5
```

## Output Format

For each program, the visualization shows:

```
Program 1 (original #5):
  Variability: 0.847 █████████████████████████████████
  Type checks: ✓ | Compiles: ✓ | Uses input: ✓ | Size: 12 nodes
  Program: (λ (x) (map (λ (y) (* y 2)) x))
  Behavior:
    []                   -> []
    [1, 2, 3]            -> [2, 4, 6]
    [5, 4, 3, 2, 1]      -> [10, 8, 6, 4, 2]
    [0, 0, 0]            -> [0, 0, 0]
```

### Fields Explained

- **Program N**: Rank by variability (1 = highest)
- **(original #N)**: Original generation order (before sorting)
- **Variability**: Score from 0.0 to 1.0, with visual bar
  - 0.0 = constant output (all inputs produce same output)
  - 1.0 = maximally diverse (all inputs produce unique outputs)
- **Type checks**: ✓ if program passes static type checking
- **Compiles**: ✓ if program compiles successfully
- **Uses input**: ✓ if program references input variable
- **Size**: Number of AST nodes
- **Program**: The actual program code
- **Behavior**: Representative input-output pairs showing program behavior
  - 4 test cases: `[]`, `[1,2,3]`, `[5,4,3,2,1]`, `[0,0,0]`
  - Shows actual outputs or error messages
  - Long outputs truncated with `...` and item count

### Summary Statistics

After displaying programs, shows:

```
======================================================================
Summary:
  Total programs: 30
  Avg variability: 0.068
  Min variability: 0.000
  Max variability: 1.000
  High variability (>0.5): 2/30
======================================================================
```

## Example Outputs

### Random Composer (Low Variability)

```
Program 1 (original #10):
  Variability: 1.000 ████████████████████████████████████████
  Type checks: ✓ | Compiles: ✓ | Uses input: ✓ | Size: 43 nodes
  Program: (if false ... (λ (d) d))
  Behavior:
    []                   -> []
    [1, 2, 3]            -> [1, 2, 3]
    [5, 4, 3, 2, 1]      -> [5, 4, 3, 2, 1]
    [0, 0, 0]            -> [0, 0, 0]

Program 2 (original #9):
  Variability: 0.808 ████████████████████████████████
  Type checks: ✓ | Compiles: ✓ | Uses input: ✓ | Size: 25 nodes
  Program: (λ (x) (filteri ... x))
  Behavior:
    []                   -> [52]
    [1, 2, 3]            -> [1, 2, 3, 52]
    [5, 4, 3, 2, 1]      -> [1, 2, 3, 4, 5, ...] (6 items)
    [0, 0, 0]            -> [52, 0]

Program 3 (original #1):
  Variability: 0.000 
  Type checks: ✓ | Compiles: ✓ | Uses input: ✓ | Size: 27 nodes
  Program: (if false ... [])
  Behavior:
    []                   -> []
    [1, 2, 3]            -> []
    [5, 4, 3, 2, 1]      -> []
    [0, 0, 0]            -> []

Program 4 (original #2):
  Variability: 0.000 
  Type checks: ✓ | Compiles: ✓ | Uses input: ✓ | Size: 37 nodes
  Program: (λ (x) (swap ... x))
  Behavior:
    []                   -> ERROR: min: empty list
    [1, 2, 3]            -> ERROR: nth: index 87 out of bounds
    [5, 4, 3, 2, 1]      -> ERROR: nth: index 87 out of bounds
    [0, 0, 0]            -> ERROR: Division by zero

...

Summary:
  Avg variability: 0.090  ← Very low!
  High variability (>0.5): 2/20 (10%)
```

**Observation**: 
- Most random programs have 0.0 variability (constant outputs or all failures)
- Program #3: Always returns `[]` (constant condition in if-statement)
- Program #4: Crashes on all inputs (hardcoded index 87, division by zero)
- Only rare high-variability programs

### Template Composer (High Variability)

```
Program 1 (original #1):
  Variability: 1.000 ████████████████████████████████████████
  Type checks: ✓ | Compiles: ✓ | Uses input: ✓ | Size: 5 nodes
  Program: (λ (x) (cons 94 x))
  Behavior:
    []                   -> [94]
    [1, 2, 3]            -> [94, 1, 2, 3]
    [5, 4, 3, 2, 1]      -> [94, 5, 4, 3, 2, ...] (6 items)
    [0, 0, 0]            -> [94, 0, 0, 0]

Program 2 (original #2):
  Variability: 1.000 ████████████████████████████████████████
  Type checks: ✓ | Compiles: ✓ | Uses input: ✓ | Size: 19 nodes
  Program: (λ (x) (filteri (λ (b c) ...) (mapi (λ (y z) ...) x)))
  Behavior:
    []                   -> []
    [1, 2, 3]            -> [9, 18, 27]
    [5, 4, 3, 2, 1]      -> [45, 36, 27, 18, 9]
    [0, 0, 0]            -> [0, 0, 0]

Program 3 (original #3):
  Variability: 1.000 ████████████████████████████████████████
  Type checks: ✓ | Compiles: ✓ | Uses input: ✓ | Size: 10 nodes
  Program: (λ (x) (fold (λ (y z) (cons z y)) [] x))
  Behavior:
    []                   -> []
    [1, 2, 3]            -> [3, 2, 1]
    [5, 4, 3, 2, 1]      -> [1, 2, 3, 4, 5]
    [0, 0, 0]            -> [0, 0, 0]

...

Summary:
  Avg variability: 0.849  ← Very high!
  High variability (>0.5): 13/15 (87%)
```

**Observation**: 
- Template programs consistently achieve high variability
- Program #1: Prepends constant, but output varies with input length/content
- Program #2: Maps and filters - output completely determined by input
- Program #3: Reverses list - different output for each input
- Most programs producing diverse outputs

## Use Cases

### 1. Debugging Composer Behavior

**Problem**: Why does my composer generate low-variability programs?

**Solution**: Use `--visualize` to inspect actual programs and their behavior:

```bash
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer my_new_composer --num-samples 50 --visualize --show-count 20
```

Look for patterns in the **Behavior** section:
- **Constant outputs**: All inputs produce same output → check for constant branches
- **All errors**: Programs crash on all inputs → check for hardcoded indices/constants
- **One success, rest errors**: Fragile programs → check edge case handling
- **Identical outputs**: Input not actually used → check if input variable referenced

**Example diagnosis from random composer:**

```
Program 3:
  Variability: 0.000
  Program: (if false ... [])
  Behavior:
    []                   -> []
    [1, 2, 3]            -> []
    [5, 4, 3, 2, 1]      -> []
    [0, 0, 0]            -> []
```
**Diagnosis**: Constant condition `false` means only else-branch executes, always returning `[]`

```
Program 4:
  Variability: 0.000
  Program: (λ (x) (swap ... (nth 87 x) ...))
  Behavior:
    []                   -> ERROR: min: empty list
    [1, 2, 3]            -> ERROR: nth: index 87 out of bounds
    [5, 4, 3, 2, 1]      -> ERROR: nth: index 87 out of bounds
    [0, 0, 0]            -> ERROR: Division by zero
```
**Diagnosis**: Hardcoded index 87 fails on realistic inputs, plus other errors

### 2. Comparing Composers

Generate visualizations for multiple composers and compare:

```bash
# Random
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer random --num-samples 30 --visualize > random_viz.txt

# Template
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer template --num-samples 30 --visualize > template_viz.txt
```

### 3. Finding Representative Examples

Identify programs that showcase composer characteristics:

- **Highest variability**: Top programs show best-case behavior
- **Lowest variability**: Bottom programs show failure modes
- **Middle range**: Typical expected behavior

### 4. Validating Changes

After modifying a composer:

```bash
# Before changes
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer template --num-samples 50 --seed 42 --visualize > before.txt

# After changes
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer template --num-samples 50 --seed 42 --visualize > after.txt

# Compare
diff before.txt after.txt
```

## Technical Details

### Variability Calculation

For each program, variability is computed as:

```python
variability = (unique_outputs - 1) / (total_successful_outputs - 1)
```

Where:
- `unique_outputs`: Number of distinct outputs across test inputs
- `total_successful_outputs`: Number of test cases that didn't crash

**Test inputs** (27 total):
- Edge cases: `[]`, `[0]`, `[5]`
- Basic: `[1,2,3]`, `[10,20,30]`, `[0,0,0]`, etc.
- Patterns: sorted, reversed, duplicates, negatives, etc.
- Random: 10 additional random inputs

### Sorting

Programs are sorted by variability in **descending order**:
1. Highest variability programs first (most diverse)
2. Lowest variability programs last (most constant)

### Display Truncation

Use `--show-count` to limit output:
- Large sample sizes can produce verbose output
- Top 10-20 programs usually capture the range
- Full statistics always shown regardless of truncation

## Integration with Existing Workflow

The visualization mode:
- ✓ Uses same composers and configurations as experiments
- ✓ Respects `--seed`, `--depth`, `--num-samples` arguments
- ✓ Works with all composer types
- ✗ Does NOT save results to JSON (visualization only)
- ✗ Cannot be combined with `--compare` mode

For full experiments with saved results, use normal mode:
```bash
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer template --num-samples 100 --output results.json
```

For quick visual inspection, use `--visualize`:
```bash
PYTHONPATH=src python3 scripts/experiment_composers.py \
  --composer template --num-samples 30 --visualize
```

## Tips

1. **Start small**: Use `--num-samples 20-30` for quick feedback
2. **Use show-count**: Add `--show-count 10` to see just the extremes
3. **Compare seeds**: Try different `--seed` values to verify consistency
4. **Save output**: Redirect to file for later analysis (`> output.txt`)
5. **Watch the bars**: Visual bars make trends obvious at a glance

## Known Limitations

- Very long programs may wrap in terminal display
- Bar length limited to 40 characters (scaled by variability)
- Failed programs (compilation errors) are excluded
- Visualization mode doesn't produce JSON output files
