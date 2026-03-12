# Variability Computation and Case Success/Failure

## Overview

There are **two different variability computations** in the codebase:

1. **Experiment Analysis Variability** (in `scripts/experiment_composers.py`)
2. **MCTS Reward Variability** (in `src/lang/composers/mcts.py`)

They handle execution success/failure differently.

---

## 1. Experiment Analysis Variability (ProgramAnalyzer)

**Location**: `scripts/experiment_composers.py`, lines 153-208

### Algorithm

```python
def _compute_variability(self, program: ASTNode) -> float:
    try:
        compiled_fn = self.jit_compiler.compile(program)
        
        outputs = []
        for inp in test_inputs:  # 27 test inputs
            try:
                result = compiled_fn(inp)
                outputs.append(self._to_hashable(result))
            except:
                pass  # SILENTLY SKIP FAILURES
        
        if not outputs or len(outputs) < 2:
            return 0.0
        
        # Compute variability as ratio of unique outputs
        unique_outputs = len(set(outputs))
        return (unique_outputs - 1) / (len(outputs) - 1)
    
    except:
        return 0.0  # Compilation failure
```

### Handling of Failures

**Key behavior**: `except: pass` on line 197-198

#### Failed Cases Are Completely Ignored
- If a test input causes an exception (e.g., index out of bounds, division by zero)
- The failure is **silently skipped**
- That test case **does not contribute** to the variability calculation

#### Impact on Variability Score

**Example 1: Program that sometimes fails**
```python
test_inputs = [[1,2,3], [10,20,30], [], [5,4,3]]  # 4 inputs
results:
  [1,2,3] -> [1,2,3]     ✓ (succeeds)
  [10,20,30] -> [10,20,30]  ✓ (succeeds)
  [] -> ERROR            ✗ (skipped)
  [5,4,3] -> [5,4,3]     ✓ (succeeds)

outputs = [[1,2,3], [10,20,30], [5,4,3]]  # 3 successful
unique = 3
variability = (3 - 1) / (3 - 1) = 2/2 = 1.0
```

**Example 2: Same program, all succeed**
```python
test_inputs = [[1,2,3], [10,20,30], [5], [5,4,3]]  # No empty list
results:
  [1,2,3] -> [1,2,3]     ✓
  [10,20,30] -> [10,20,30]  ✓
  [5] -> [5]             ✓
  [5,4,3] -> [5,4,3]     ✓

outputs = [[1,2,3], [10,20,30], [5], [5,4,3]]  # 4 successful
unique = 4
variability = (4 - 1) / (4 - 1) = 3/3 = 1.0
```

### Edge Cases

#### All Cases Fail
```python
outputs = []  # Everything threw exceptions
return 0.0
```

#### Only 1 Case Succeeds
```python
outputs = [[1,2,3]]  # Only one success
len(outputs) < 2
return 0.0
```

#### All Same Output
```python
outputs = [[1,2,3], [1,2,3], [1,2,3]]
unique = 1
variability = (1 - 1) / (3 - 1) = 0/2 = 0.0
```

### Key Insight

**Failures reduce the effective sample size** but don't directly penalize the score:

- A program that fails on 20 out of 27 inputs and produces 7 different outputs gets:
  - `variability = (7-1)/(7-1) = 1.0` ✓ High score!
  
- A program that succeeds on all 27 inputs but produces only 3 outputs gets:
  - `variability = (3-1)/(27-1) = 2/26 = 0.077` ✗ Low score

**This means: Programs that fail on many edge cases can still score HIGH variability if they produce different outputs on their successful cases.**

---

## 2. MCTS Reward Variability (VariabilityScorer)

**Location**: `src/lang/composers/mcts.py`, lines 459-515

### Algorithm

```python
def compute_variability(self, program, input_type, substitutions) -> float:
    inputs = self.input_sampler.sample_many(input_type, self.num_samples, substitutions)
    
    try:
        compiled_fn = self.jit_compiler.compile(program)
    except JITCompilationError:
        return self._score_by_structure(program) * 0.3  # Compilation fails
    
    outputs = []
    identity_count = 0
    valid_count = 0
    
    for inp in inputs:
        try:
            result = compiled_fn(inp)
            valid_count += 1
            if self._equals(result, inp):  # Check for identity function
                identity_count += 1
            outputs.append(self._to_hashable(result))
        except Exception:
            outputs.append(None)  # RECORD FAILURES AS None
    
    valid_outputs = [o for o in outputs if o is not None]
    
    if not valid_outputs:
        return self._score_by_structure(program) * 0.2  # All failed
    
    # Compute variability from VALID outputs only
    unique_outputs = len(set(valid_outputs))
    if len(valid_outputs) == 1:
        variability = 0.0
    else:
        variability = (unique_outputs - 1) / (len(valid_outputs) - 1)
    
    # Penalize identity functions
    if valid_count > 0:
        identity_ratio = identity_count / valid_count
        variability *= (1.0 - self.identity_penalty * identity_ratio)
    
    structure = self._score_by_structure(program)
    success_rate = valid_count / self.num_samples
    
    # Weighted combination
    return (variability * 0.3 + structure * 0.5 + success_rate * 0.2)
```

### Handling of Failures

**Key difference**: Failures are tracked separately!

#### Failed Cases Are Recorded as `None`
```python
except Exception:
    outputs.append(None)  # Keep track of failures
```

#### Success Rate is Part of the Score
```python
success_rate = valid_count / self.num_samples
final_score = variability * 0.3 + structure * 0.5 + success_rate * 0.2
```

### Impact on MCTS Score

**Example: Program that fails on half the inputs**

```python
num_samples = 10
valid_count = 5  # Half succeed
unique valid outputs = 4

# Variability component
variability = (4-1)/(5-1) = 0.75

# Success rate component
success_rate = 5/10 = 0.5

# Structure (hypothetical)
structure = 0.6

# Final score
final = 0.75 * 0.3 + 0.6 * 0.5 + 0.5 * 0.2
      = 0.225 + 0.3 + 0.1
      = 0.625
```

**Same program with all successes:**

```python
num_samples = 10
valid_count = 10  # All succeed
unique valid outputs = 4

variability = (4-1)/(10-1) = 0.33

success_rate = 10/10 = 1.0

structure = 0.6

final = 0.33 * 0.3 + 0.6 * 0.5 + 1.0 * 0.2
      = 0.099 + 0.3 + 0.2
      = 0.599
```

**Interesting**: The first program (with failures) scored HIGHER because:
- Variability is computed only from valid outputs (higher ratio: 3/4 vs 3/9)
- Success rate penalty (0.1) < Variability gain (0.126)

### Additional Penalties

#### Identity Function Penalty
```python
if self._equals(result, inp):
    identity_count += 1

identity_ratio = identity_count / valid_count
variability *= (1.0 - self.identity_penalty * identity_ratio)
```

If 50% of outputs equal inputs and `identity_penalty = 0.5`:
```python
variability *= (1.0 - 0.5 * 0.5) = 0.75 * variability
```

#### Compilation Failure Fallback
```python
except JITCompilationError:
    return self._score_by_structure(program) * 0.3
```
Programs that don't compile get 30% of their structure score.

#### All Execution Failures Fallback
```python
if not valid_outputs:
    return self._score_by_structure(program) * 0.2
```
Programs that compile but always fail get 20% of structure score.

---

## Summary Table

| Aspect | Experiment Variability | MCTS Variability |
|--------|------------------------|------------------|
| **Failure handling** | Silently skip | Record as `None` |
| **Success rate tracked?** | No | Yes (20% weight) |
| **Variability basis** | Successful outputs only | Successful outputs only |
| **Failure penalty** | Indirect (smaller denominator) | Direct (success_rate term) |
| **Identity penalty** | No | Yes (configurable) |
| **Structure score** | No | Yes (50% weight) |
| **Compilation failure** | Return 0.0 | Structure * 0.3 |
| **All failures** | Return 0.0 | Structure * 0.2 |

## Key Insights

### 1. Both Compute Variability Only From Successful Cases
Neither system treats failures as "another output variant". Failures are excluded from the unique output count.

### 2. Failures Can Increase Apparent Variability
Since variability = `(unique - 1) / (valid - 1)`, having fewer successful cases with diverse outputs yields higher variability than many successes with few unique outputs.

### 3. MCTS Balances Multiple Objectives
The MCTS scorer cares about:
- **Variability** (30%): Diverse outputs
- **Structure** (50%): Interesting functions used
- **Success rate** (20%): Robustness

So a highly variable but failure-prone program might still score well due to structure.

### 4. Experiment Analyzer Focuses Pure Variability
The experiment variability is a **pure diversity metric** - it doesn't penalize failures except by reducing sample size.

## Recommendation

If you want to **penalize fragile programs** in experiments, you could modify the formula to:

```python
success_rate = len(outputs) / len(test_inputs)
variability = (unique_outputs - 1) / (len(outputs) - 1) if len(outputs) > 1 else 0
adjusted_variability = variability * success_rate
```

This would give lower scores to programs that fail on many test cases.
