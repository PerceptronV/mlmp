# Rule Programs Type-Check Failures

## Overview

The "rule" programs (canonical programs from `src/data/rule/functions.txt`) have a **94.4% type-check rate** with **14 out of 251 programs failing** type-checking.

These failures are **not related to the lambda fix** - they represent actual bugs in the canonical program corpus.

## Root Cause

All 14 failures involve the `sort` function with an incorrect key function:

```python
# Sort signature (from grammar.py line 596):
def sort(f: Callable[[T1], int], xs: list[T1]) -> list[T1]
```

The sort function expects:
- **Key function type**: `Callable[[T], int]` - a function that takes an element and returns an integer
- **Purpose**: The integer is used as the sort key

## The Bug

The failing programs use lambdas like `(λ (y) x)` as the sort key function:

```scheme
(λ (x) (sort (λ (y) x) x))
```

**Problem**:
- The key function `(λ (y) x)` ignores its parameter `y` and returns `x`
- Since `x` has type `list[int]`, the key function returns `list[int]`
- But `sort` expects the key function to return `int`, not `list[int]`

**Type mismatch**: Expected `Callable[[T], int]`, got `Callable[[T], list[int]]`

## Examples of Failing Programs

### Line 106
```scheme
(λ (x) (sort (λ (y) x) x))
```
- Key function `(λ (y) x)` returns `x: list[int]`
- Expected: returns `int`

### Line 128
```scheme
(λ (x) (sort (λ (y) x) (cut_idx 3 (drop 2 x))))
```
- Same issue: key function returns a list instead of int

### Line 179
```scheme
(λ (x) (fold (λ (y z) (append (reverse y) z)) [] (reverse (sort (λ (y) x) x))))
```
- Nested within fold, same sort issue

### Line 198
```scheme
(λ (x) (reverse (sort (λ (y) x) (unique x))))
```
- Another variant of the same bug

### Line 199
```scheme
(λ (x) (flatten (zip (range 1 1 (length x)) (sort (λ (y) x) x))))
```
- Same pattern

### Line 200
```scheme
(λ (x) (sort (λ (y) x) (map (λ (y) (/ y 10)) x)))
```
- Same pattern

### All Other Failures (Lines 110, 180, 193, 209, ...)
All follow the same pattern: `(sort (λ (y) x) ...)` where the key function incorrectly returns a list.

## Why This Compiles But Fails Type-Checking

The JIT compiler (runtime) is more lenient:
- Python's `sorted(xs, key=lambda x: f(x))` will execute even if `f` returns unexpected types
- It will use whatever `f` returns for comparison (Python can compare lists)

However, the static type checker enforces the declared type signature:
- `sort` is declared to expect `Callable[[T], int]`
- The type checker rejects `Callable[[T], list[int]]` as incompatible

## Impact

**Compile Rate**: 99.6% (250/251) - Only 1 fails to compile  
**Type Check Rate**: 94.4% (237/251) - 14 fail type-checking

The programs still compile and may even run, but they violate the type signature contract.

## Likely Intent

These programs probably intended one of:
1. **Sort by element value**: `(λ (y) y)` - sort by the element itself
2. **Constant sort key**: `(λ (y) 0)` - all elements get same key (no-op sort)
3. **Sort by input property**: `(λ (y) (length x))` - but this makes all elements equal (no-op)

The most likely fix would be:
```scheme
# Before (wrong):
(λ (x) (sort (λ (y) x) x))

# After (correct):
(λ (x) (sort (λ (y) y) x))  # Sort by element value
```

## Distribution of Failures

All 14 failures (100%) are due to this single pattern: incorrect sort key function returning a list instead of an integer.

## Recommendations

### For the Canonical Corpus
1. **Fix the bugs**: Update the 14 programs in `src/data/rule/functions.txt`
2. **Add validation**: Run type-checking on all canonical programs as part of data pipeline
3. **Document intent**: Add comments explaining what each sort operation should achieve

### For Composers
The composers should avoid generating `(sort (λ (y) x) ...)` patterns:
- Template composer: Ensure sort key functions generate appropriate int-valued expressions
- Empirical composer: Will learn from canonical data, so fixing the corpus will fix this
- Random/MCTS: May occasionally generate this by chance, but it's relatively rare

## Summary

The rule programs' type-check failures are **not related to lambda syntax** (curried vs multi-parameter). They represent actual semantic bugs where:
- Sort key functions return lists instead of integers
- This violates the type signature but may still compile/run
- The static type checker correctly identifies these as type errors

This is distinct from the lambda fix, which addressed a syntactic issue (nested single-parameter lambdas vs multi-parameter lambdas) that affected composers' generated programs.
