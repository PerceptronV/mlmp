# Rule Programs Fixed - Summary

## Results

✓ **All 251 programs now pass type-checking (100%)**

**Before fix**: 94.4% (237/251 passed, 14 failed)  
**After fix**: 100.0% (251/251 passed, 0 failed)

## Changes Made

### 1. Fixed Sort Key Functions (13 programs)

Replaced all instances of `(sort (λ (y) x) ...)` with `(sort (λ (y) y) ...)`:

**Lines fixed**: 106, 110, 128, 179, 180, 193, 198, 199, 200, 209, 213, 231, 242

**The bug**: Key function `(λ (y) x)` returned `list[int]` instead of `int`  
**The fix**: Changed to `(λ (y) y)` which returns the element value (int)

#### Example Fix - Line 200:
```scheme
# Before (wrong):
(λ (x) (sort (λ (y) x) (map (λ (y) (/ y 10)) x)))

# After (correct):
(λ (x) (sort (λ (y) y) (map (λ (y) (/ y 10)) x)))
```

**Behavioral change**: 
- Before: No sorting (all elements had same key)
- After: Properly sorts elements by value

#### Additional Fix - Line 231:
This line had **two** bugs - both the `sort` and `group` used `(λ (y) x)`:

```scheme
# Before:
(λ (x) (sort (λ (y) x) (map length (group (λ (y) x) x))))

# After:
(λ (x) (sort (λ (y) y) (map length (group (λ (y) y) x))))
```

### 2. Fixed Negative Literal (1 program)

**Line 235**: Replaced `-1` with `(- 0 1)` to work around parser limitation

```scheme
# Before:
(λ (x) (fold (λ (y z) (concat y (drop 1 (range (last y) (if (> z (last y)) 1 -1) z)))) (take 1 x) (drop 1 x)))

# After:
(λ (x) (fold (λ (y z) (concat y (drop 1 (range (last y) (if (> z (last y)) 1 (- 0 1)) z)))) (take 1 x) (drop 1 x)))
```

**Note**: The parser doesn't support negative number literals directly, so we use `(- 0 1)` to represent -1.

## Impact

### Type Checking
- **Before**: 14 type errors
- **After**: 0 type errors ✓

### Semantics
The fixed programs now:
1. Actually sort their data (instead of no-op)
2. Group correctly by element value
3. Properly handle negative step in range

### Experiment Results
Now when running experiments with `--compare`, the rule programs will show:
```
Type Check Rate: 100.0% ✓  (was 94.4%)
```

This makes the canonical baseline fully consistent with the type system.

## Verification

Run this to verify:
```bash
PYTHONPATH=src python3 << 'EOF'
from lang.parser import parse
from lang.type_checker import TypeChecker

checker = TypeChecker()
failures = 0

with open('src/data/rule/functions.txt') as f:
    for line in f:
        if line.strip():
            try:
                checker.check(parse(line.strip()))
            except:
                failures += 1

print(f"Type check rate: {(251-failures)/251*100:.1f}%")
EOF
```

Expected output: `Type check rate: 100.0%`

## Files Modified

- `src/data/rule/functions.txt` - Fixed 14 lines (13 sort bugs + 1 negative literal)

## Summary

All type-check failures in the canonical program corpus have been resolved. The rule programs now serve as a proper baseline for comparing against composer-generated programs.
