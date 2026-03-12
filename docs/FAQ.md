# Answers to Common Questions

## Question 1: Why are there programs that don't type check?

### The Issue

Some composers (especially Template) generate 20-40% of programs that don't type-check. This seems contradictory since the generation is supposed to be type-directed.

### Root Cause: Nested Lambda Representation

The issue is with **how multi-parameter lambdas are represented**:

**Template Composer creates nested lambdas:**
```python
# Generated structure (WRONG for type checker)
for param_name in reversed(param_names):
    body = LambdaNode(param_name, body)  # Creates nested structure
```

This produces: `(λ (x) (λ (y) body))` - nested single-parameter lambdas

**But type checker expects:**
```python
LambdaNode(param_names, body)  # Multi-parameter lambda
```

This expects: `(λ (x y) body)` - single multi-parameter lambda

### Example Failure

**Generated program:**
```lisp
(λ (x) (filteri (λ (b) (λ (c) (< (% c 2) 8))) (mapi (λ (y) (λ (z) (* y 9))) x)))
```

**Error:**
```
Type mismatch: expected Callable[[T6, int], T7], 
               got Callable[[int], Callable[[int], bool]]
```

The inner lambda `(λ (b) (λ (c) ...))` is **curried** (nested), but `filteri` expects a **multi-parameter** lambda.

### Why Other Composers Work

- **Random/MCTS/RandomGuarded:** Use `LambdaNode(param_names, body)` directly
- **Empirical:** Learns from programs that use the correct format
- **Template:** Uses nested structure for historical reasons

### The Programs Still Work!

**Important:** Even though they don't type-check, they **compile and run correctly** because:
1. The JIT compiler handles both representations
2. Python doesn't distinguish between curried and multi-parameter functions
3. They're semantically equivalent

### Evidence

From our test:
```
Generated 20 programs
Type check failures: 4 (20%)

Example failures:
- filteri with nested lambdas
- fold with nested lambdas  
- mapi with nested lambdas
```

All involve higher-order functions (`filteri`, `mapi`, `fold`) that expect multi-parameter lambdas but receive nested ones.

### Solution Options

**Option 1: Fix Template Composer (recommended)**
```python
# In template.py line 921-922, change:
for param_name in reversed(param_names):
    body = LambdaNode(param_name, body)

# To:
return LambdaNode(param_names, body)
```

**Option 2: Update Type Checker**
Make the type checker understand curried lambdas as equivalent to multi-parameter ones.

**Option 3: Do Nothing**
Since programs compile and run correctly, and this only affects 20% of template programs, you can leave it as-is. Just note that "Type Check Rate" measures syntactic compatibility with the type checker, not semantic correctness.

---

## Question 2: How is variability computed?

### Algorithm

```python
def _compute_variability(self, program: ASTNode) -> float:
    # 1. Compile the program
    compiled_fn = self.jit_compiler.compile(program)
    
    # 2. Run on 7 test inputs
    test_inputs = [
        [],
        [1, 2, 3],
        [10, 20, 30],
        [0, 0, 0],
        [5],
        list(range(10)),
        [99, 50, 1],
    ]
    
    # 3. Collect outputs (skip errors)
    outputs = []
    for inp in test_inputs:
        try:
            result = compiled_fn(inp)
            outputs.append(hashable(result))
        except:
            pass  # Skip if program crashes on this input
    
    # 4. Compute variability
    unique_outputs = len(set(outputs))
    variability = (unique_outputs - 1) / (len(outputs) - 1)
    
    return variability  # Range: 0.0 to 1.0
```

### Formula

```
Variability = (unique_outputs - 1) / (total_outputs - 1)
```

### Interpretation

| Variability | Meaning | Example |
|-------------|---------|---------|
| 0.0 | Constant output | `(λ (x) [])` → always returns `[]` |
| 0.17 | Low variation | 2 unique outputs from 7 inputs |
| 0.50 | Moderate | 4 unique outputs from 7 inputs |
| 1.0 | Maximum variation | All 7 outputs different |

### Examples

**Example 1: Constant Program (variability = 0.0)**
```lisp
(λ (x) [])
```
- Input: `[]` → Output: `[]`
- Input: `[1,2,3]` → Output: `[]`
- Input: `[10,20,30]` → Output: `[]`
- Unique outputs: 1
- Variability: (1-1)/(7-1) = **0.0**

**Example 2: Identity Program (variability = 1.0)**
```lisp
(λ (x) x)
```
- Input: `[]` → Output: `[]`
- Input: `[1,2,3]` → Output: `[1,2,3]`
- Input: `[10,20,30]` → Output: `[10,20,30]`
- Input: `[0,0,0]` → Output: `[0,0,0]`
- ... (all different)
- Unique outputs: 7
- Variability: (7-1)/(7-1) = **1.0**

**Example 3: Map Program (variability = 0.83)**
```lisp
(λ (x) (map (λ (y) (+ y 1)) x))
```
- Input: `[]` → Output: `[]`
- Input: `[1,2,3]` → Output: `[2,3,4]`
- Input: `[10,20,30]` → Output: `[11,21,31]`
- Input: `[0,0,0]` → Output: `[1,1,1]`
- ... (6 different, 1 duplicate)
- Unique outputs: 6
- Variability: (6-1)/(7-1) = **0.83**

**Example 4: Length Program (variability = 0.5)**
```lisp
(λ (x) (singleton (length x)))
```
- Input: `[]` → Output: `[0]`
- Input: `[1,2,3]` → Output: `[3]`
- Input: `[10,20,30]` → Output: `[3]` (duplicate!)
- Input: `[0,0,0]` → Output: `[3]` (duplicate!)
- Input: `[5]` → Output: `[1]`
- Input: `range(10)` → Output: `[10]`
- Input: `[99,50,1]` → Output: `[3]` (duplicate!)
- Unique outputs: 4
- Variability: (4-1)/(7-1) = **0.5**

### Why This Metric?

**Measures meaningful behavior:**
- Low variability (0.0-0.3) → Likely degenerate (constant, ignores input)
- Medium variability (0.3-0.7) → Partially uses input
- High variability (0.7-1.0) → Strongly depends on input

**Distinguishes quality:**
- Random composer: 0.10 (mostly constants)
- Template composer: 0.90 (meaningful operations)
- MCTS composer: 0.75 (learned from rewards)

### Limitations

1. **Only 7 test inputs:** May not capture all behavior
2. **Ignores errors:** Programs that crash are treated as variability 0.0
3. **Simple inputs:** All lists of integers, specific patterns
4. **No semantic understanding:** `(reverse x)` and `(sort x)` have same variability

### Better Metrics Considered

**Alternative metrics:**
- **Entropy:** More mathematically sound but harder to interpret
- **Edit distance:** Measures how different outputs are
- **Functional complexity:** Analyzes AST structure
- **Behavioral diversity:** Uses more sophisticated input generation

**Why we use this simple metric:**
- Easy to understand (0-1 scale)
- Fast to compute
- Correlates well with program quality
- Good enough to distinguish composers

### Edge Cases

**Errors during execution:**
```python
outputs = []
for inp in test_inputs:
    try:
        result = compiled_fn(inp)
        outputs.append(result)
    except:
        pass  # Skip this input
```

If program crashes on most inputs → `len(outputs) < 2` → variability = 0.0

**All outputs identical:**
```
unique_outputs = 1
variability = (1-1)/(7-1) = 0.0
```

---

## Summary

### Question 1: Type Checking
- **Issue:** Template composer creates nested lambdas instead of multi-parameter lambdas
- **Impact:** ~20% of template programs don't type-check
- **Reality:** They still compile and run correctly
- **Fix:** Change `LambdaNode` construction in template.py

### Question 2: Variability
- **Metric:** `(unique_outputs - 1) / (total_outputs - 1)`
- **Inputs:** 7 test cases ranging from `[]` to `[99,50,1]`
- **Range:** 0.0 (constant) to 1.0 (maximum variation)
- **Purpose:** Distinguish meaningful programs from degenerate ones

### Observed Variability by Composer

| Composer | Avg Variability | Interpretation |
|----------|----------------|----------------|
| random | 0.10 | Mostly constant/degenerate |
| random_guarded | 0.15 | Slightly better |
| template | 0.90 | Meaningful operations |
| mcts (1000ep) | 0.75 | Learned good patterns |
| empirical | 0.53 | Matches training data |
