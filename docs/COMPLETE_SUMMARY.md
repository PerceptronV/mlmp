# Complete Summary: Test Input Expansion & Lambda Fix

## Changes Made

### 1. Fixed Multi-Parameter Lambda Generation

**Problem**: Template composers were generating nested single-parameter lambdas instead of multi-parameter lambdas, causing type-check failures.

**Files Modified**:
- `src/lang/composers/template.py`
- `src/lang/composers/template_with_inverse.py`

**Impact**:
- Template composer type-check rate: **50-70% → 100%** ✓
- Fixed all lambda-related type errors

### 2. Expanded Test Inputs

**Problem**: Limited test input variety (5-7 inputs) didn't adequately test program behavior diversity.

**Files Modified**:
- `scripts/experiment_composers.py`: 7 → 24 test inputs
- `scripts/quick_composer_test.py`: 3 → 6 test inputs

**New test patterns include**:
- Edge cases: `[]`, `[0]`, single elements
- Negative numbers: `[-1, -2, -3]`
- Mixed signs: `[1, -1, 2, -2, 3]`
- Duplicates: `[1, 1, 2, 2, 3, 3]`
- Sorted/reversed sequences
- Large numbers, symmetric ranges
- Even/odd only sequences

**Impact**: Better variability measurements and more comprehensive behavior testing.

## Results Summary

### Template Composer (After Fix)
```
Type Check Rate:     100.0% ✓
Compile Rate:        100.0% ✓
Uses Input Rate:     100.0% ✓
Average Variability: 0.747
High Variability:    74.0%
```

### Random Composer (Unchanged)
```
Type Check Rate:     100.0% ✓
Compile Rate:        100.0% ✓
Uses Input Rate:     93.3%
Average Variability: 0.056
High Variability:    6.7%
```

### Rule Programs (Canonical Baseline)
```
Type Check Rate:     94.4%  ← Has bugs!
Compile Rate:        99.6%
Uses Input Rate:     97.2%
Average Variability: 0.740
High Variability:    76.3%
```

## User Question: Why Do Rule Programs Fail Type Checks?

**Answer**: The rule programs have 14 type-check failures (out of 251 programs) due to a **bug in the canonical corpus**, not an issue with the type system or composers.

**Root Cause**: All 14 failures use incorrect sort key functions:
```scheme
(λ (x) (sort (λ (y) x) x))  # ✗ Key function returns list[int], not int
```

**The Bug**:
- `sort` signature: `Callable[[T], int] -> list[T] -> list[T]`
- Expected: Key function returns `int`
- Actual: `(λ (y) x)` returns `list[int]`
- **Type mismatch**: Returns wrong type

**Why it compiles**: Python's `sorted()` accepts any comparable type, so the program runs (though it doesn't sort properly).

**Why it fails type-check**: The static type checker correctly identifies the type signature violation.

## Key Distinctions

### Lambda Fix (Syntactic Issue)
- **Problem**: Nested `(λ x (λ y body))` vs multi-param `(λ (x y) body)`
- **Where**: Composer-generated programs
- **Solution**: Fixed in template composers
- **Result**: 100% type-check rate

### Rule Program Bugs (Semantic Issue)
- **Problem**: Sort key returns `list[int]` instead of `int`
- **Where**: Hand-written canonical programs
- **Solution**: Fix the 14 buggy programs in `functions.txt`
- **Impact**: Currently 94.4% type-check rate

## Testing

Verify the fixes work:

```bash
# Quick test with expanded inputs
PYTHONPATH=src python3 scripts/quick_composer_test.py template 5

# Full experiment
PYTHONPATH=src python3 scripts/experiment_composers.py --composer template --num-samples 100

# Batch comparison
PYTHONPATH=src python3 scripts/batch_experiments.py --preset quick
```

All should show **100% type-check rate** for template and random composers.

## Documentation Created

1. **LAMBDA_FIX_SUMMARY.md** - Detailed technical documentation of the lambda fix
2. **RULE_PROGRAMS_TYPE_ERRORS.md** - Comprehensive analysis of rule program bugs
3. **RULE_PROGRAMS_SUMMARY.md** - Quick reference for rule program issues
4. **THIS_FILE.md** - Complete overview of all changes

## Next Steps (Optional)

1. **Fix rule programs**: Update the 14 buggy programs in `src/data/rule/functions.txt`
   - Replace `(sort (λ (y) x) ...)` with `(sort (λ (y) y) ...)`
   
2. **Add validation**: Run type-checker on canonical programs as part of data pipeline

3. **Document expected behavior**: Add comments to canonical programs explaining intent

## Conclusion

✓ **Lambda fix complete**: Template composers now generate 100% type-correct programs  
✓ **Test inputs expanded**: Better coverage for variability measurement  
✓ **Rule program bugs identified**: 14 programs with incorrect sort key functions  

The composers are now **more correct** than the canonical baseline!
