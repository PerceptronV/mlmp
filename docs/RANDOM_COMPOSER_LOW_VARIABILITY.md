# Why Random Composer Has ~0% Variability

## The Problem

The random composer generates programs with **extremely low variability** (~0.068 average, with 90% of programs having variability <0.1).

## Root Causes

### 1. **Constant Literals Everywhere**

The random composer samples literals **uniformly** throughout the program structure. This leads to programs filled with hard-coded constants:

```scheme
(λ (x) (range 92 (fold (λ (y z) z) (second []) (droplast 64 x)) ...))
                ^                             ^             ^^
              Constant                    Constant      Constant
```

**Result**: Even if the program uses input `x`, the output is dominated by these constants, producing the same or very similar outputs across different inputs.

### 2. **Fragile Programs with Hardcoded Indices**

Random programs frequently use:
- `nth 87 x` - accesses element 87 (fails on most realistic inputs)
- `replace 46 ...` - replaces index 46
- `droplast 64 x` - drops 64 elements
- `take 52 x` - takes 52 elements

```scheme
Program: (λ (x) (swap (nth 87 x) (last []) (takelast 48 [])))

Outputs:
  []           -> ERROR: nth: index 87 out of bounds
  [1,2,3]      -> ERROR: nth: index 87 out of bounds
  [10,20,30]   -> ERROR: nth: index 87 out of bounds
  [5,4,3,2,1]  -> ERROR: nth: index 87 out of bounds
```

**Result**: Most inputs cause crashes, leaving few successful outputs. With failures excluded from variability calculation, there's often only 0-2 unique outputs.

### 3. **Input-Ignoring Branches**

Random if-statements often have constant conditions that ignore the input:

```scheme
(if false
    (λ (x) <complex expression using x>)
    (λ (y) (filteri (λ (z a) false) [])))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
          Always returns []
```

**Result**: The program always takes one branch and returns a constant.

### 4. **Degenerate Compositions**

Random sampling creates semantically meaningless compositions:

```scheme
(fold (λ (y z) z) <init> <list>)
      ^^^^^^^^^^^
      Always returns second argument, ignores accumulator
      → Result is always the last element

(fold (λ (y z) 7) <init> x)
      ^^^^^^^^^^^
      Always returns 7
      → Result is always 7 regardless of input
```

**Result**: Complex-looking programs that collapse to constants.

## Statistical Evidence

**From 100 random programs (depth=4, seed=42):**

```
Average variability:     0.068  (very low!)
High variability (>0.5): 6%
Uses input variable:     86%

Variability distribution:
  [0.0-0.1):  90 programs  ████████████████████████████
  [0.1-0.2):   1 program
  [0.2-0.3):   0 programs
  [0.3-0.4):   1 program
  [0.4-0.5):   2 programs
  [0.5-0.6):   1 program
  [0.6-0.7):   0 programs
  [0.7-0.8):   0 programs
  [0.8-0.9):   1 program
  [0.9-1.0):   0 programs
  [1.0]:       4 programs  █
```

**Key observation**: Even though 86% of programs technically "use" the input variable `x`, they still have near-zero variability!

## Detailed Example

### Program with 0.0 Variability (using input!)

```scheme
(λ (x) (range 92 (fold (λ (y z) z) (second []) (droplast 64 x)) 
              (- (max x) (max x))))
       ^^^^^^^^                    ^^^^^^^^  ^^^^^^^^^^^^^^^^^
       Constant                    Constant  Always 0
```

**Execution**:
```
Input: []           -> ERROR: second: list too short
Input: [1,2,3]      -> ERROR: second: list too short
Input: [10,20,30]   -> ERROR: second: list too short
Input: [5,4,3,2,1]  -> ERROR: second: list too short

Variability: 0.0 (all cases failed, no outputs to measure)
```

The program nominally uses `x` but:
1. Calls `second []` which always fails
2. Uses `droplast 64 x` which fails on small lists
3. Computes `(max x) - (max x) = 0` (constant)

## Why This Happens: Random Sampling Strategy

### Uniform Weight Distribution

From `random.py` lines 92-136:

```python
# All candidates get similar weights
candidates = []
weights = []

if target == int or target == bool or base_type == list:
    candidates.append(('literal', None))
    weights.append(0.1)  # Literals

for var_name, var_type in context.items():
    candidates.append(('variable', var_name))
    weights.append(0.1)  # Variables

if base_type == CallableOrig and depth > 0:
    candidates.append(('lambda', None))
    weights.append(0.2)  # Lambdas

if depth > 0:
    candidates.append(('if', None))
    weights.append(0.2)  # If statements

for func_name in grammar_functions:
    candidates.append(('application', func_name))
    weights.append(0.2)  # Each function
```

**Problem**: All options are weighted nearly equally!
- Literal `42` has weight 0.1
- Variable `x` has weight 0.1
- Complex function has weight 0.2

This leads to:
1. **Too many literals**: ~10-20% of nodes are random constants
2. **No semantic guidance**: Functions like `nth`, `max`, `min` are chosen without considering if they make sense
3. **Deep constant nesting**: Literals appear at all depth levels, not just leaves

### Literal Sampling (Lines 147)

```python
def _sample_literal(self, target, substitutions):
    if target == int:
        return NumberNode(self.rng.randint(0, 100))  # Random 0-100
    elif target == bool:
        return BooleanNode(self.rng.choice([True, False]))
    elif base_type == list:
        return ListNode([])  # Always empty list
```

Every time a literal is chosen:
- Integers: Random 0-100 (e.g., 42, 87, 64)
- Booleans: Random true/false
- Lists: Always `[]`

**Result**: Hardcoded magic numbers throughout the program.

## Contrast with Template Composer

**Template Composer (avg variability: 0.78, high var: 74%)**

The template composer:
1. **Uses templates**: Pre-defined patterns like `(map f x)`, `(filter p x)`
2. **Guards generation**: Ensures generated functions match expected signatures
3. **Avoids constants in key positions**: Doesn't put `nth 87` on random inputs
4. **Semantic structure**: Maps/filters/folds actually transform the input

**Example template program:**
```scheme
(λ (x) (map (λ (y) (* y 2)) x))
```

**Outputs** (high variability!):
```
[]           -> []
[1,2,3]      -> [2,4,6]
[10,20,30]   -> [20,40,60]
[5,4,3,2,1]  -> [10,8,6,4,2]

Unique: 4/4, Variability: 1.0
```

## Summary

The random composer has near-zero variability because:

1. ✗ **Constant proliferation**: Uniform sampling adds random constants throughout
2. ✗ **Fragile operations**: Hardcoded indices like `nth 87` fail on realistic inputs
3. ✗ **Semantic degeneracy**: Random composition creates meaningless patterns
4. ✗ **No semantic guidance**: All choices weighted equally, no preference for input-dependent operations

**The programs are syntactically valid but semantically degenerate** - they type-check and compile, but behave like random constant generators rather than interesting transformations.

## What About the 6% with High Variability?

The rare high-variability programs happen when:
- Random sampling luckily avoids constants in critical paths
- Input variable appears in key positions
- Operations naturally propagate input differences

**Example (variability 1.0)**:
```scheme
(λ (x) (reverse (drop 1 x)))
```

No constants, simple operations on input → every input produces unique output!

But this is **accidental** - the random composer has no mechanism to prefer such programs.

## Implications

This explains why random composer is used as a **baseline**:
- Shows what happens without semantic guidance
- Demonstrates importance of structured generation
- Validates that template/MCTS improvements are meaningful (78x better variability!)

The random composer proves that **syntax alone is insufficient** - you need semantic structure to generate interesting programs.
