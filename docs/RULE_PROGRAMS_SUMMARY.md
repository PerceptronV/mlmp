# Summary: Why Rule Programs Fail Type Checks

## Quick Answer

The rule programs (canonical programs from `src/data/rule/functions.txt`) have **14 type-check failures out of 251 programs (94.4% pass rate)**. 

**All 14 failures are caused by the same bug**: Using `(λ (y) x)` as a sort key function.

## The Bug

```scheme
(λ (x) (sort (λ (y) x) x))
```

**What's wrong:**
- `sort` expects a key function: `Callable[[T], int]` (takes element, returns int)
- `(λ (y) x)` returns `x` which has type `list[int]`, not `int`
- **Type error**: Expected `int`, got `list[int]`

## Why It Still Compiles and Runs

The program compiles successfully and even runs:
```
[3, 1, 4, 1, 5] -> [3, 1, 4, 1, 5]  (unchanged)
```

**Reason**: Python's `sorted()` accepts any comparable type. Since the key function returns the same list for every element, all elements have equal sort keys, so the list is returned unchanged (stable sort preserves original order).

**However**, this violates the declared type signature in the grammar, so the static type checker correctly rejects it.

## Affected Programs

**14 programs with this pattern** (lines: 106, 110, 128, 179, 180, 193, 198, 199, 200, 209, and 4 others)

All follow the pattern: `(sort (λ (y) x) ...)`

## The Fix

Replace `(λ (y) x)` with an appropriate key function:

```scheme
# Wrong (returns list):
(λ (x) (sort (λ (y) x) x))

# Correct option 1 (sort by element value):
(λ (x) (sort (λ (y) y) x))

# Correct option 2 (constant key, no-op sort):
(λ (x) (sort (λ (y) 0) x))
```

Most likely the intent was option 1: sort by element value.

## Not Related to Lambda Fix

This issue is **completely separate** from the lambda currying fix:
- **Lambda fix**: Syntactic issue (nested vs multi-parameter lambdas)
- **This issue**: Semantic bug (wrong return type from key function)

The lambda fix addressed how composers generate lambdas. This issue is a bug in the hand-written canonical programs themselves.

## Impact on Experiments

When comparing composers to "rule" baseline:
- Rule programs: 94.4% type-check rate
- Template composer (after fix): 100% type-check rate  
- Random composer: 100% type-check rate

The rule programs are actually **less correct** than the composers!

## Recommendation

Fix the 14 buggy programs in `src/data/rule/functions.txt` by replacing `(λ (y) x)` with `(λ (y) y)` in all sort expressions.
