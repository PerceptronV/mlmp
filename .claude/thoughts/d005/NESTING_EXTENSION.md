# Extension: Nested Higher-Order Functions in Lambda Bodies

## Overview

This document describes the changes needed to support nested higher-order functions inside lambda bodies — e.g., `map` inside a `filter` lambda, `filter` inside a `fold` accumulator, etc. This replaces the V1 restriction that disallowed higher-order functions inside lambda bodies entirely.

The core change is replacing the special-cased lambda enumeration with a single recursive enumerator parameterised by a variable context, and replacing the flat `ProgramBank` with a hierarchical `ContextualBank` that mirrors lexical scoping.

---

## 1. Motivation

### What V1 Can Express

V1 enumerates programs like:
```scheme
(filter (λ y (> y 0)) x)                    ;; filter positives
(map (λ y (* y 2)) x)                       ;; double each element
(sort (λ y y) x)                            ;; identity sort
(fold (λ acc elem (+ acc elem)) 0 x)        ;; sum via fold
```

### What V1 Cannot Express

V1 cannot enumerate programs with higher-order functions inside lambda bodies:
```scheme
(map (λ y (filter (λ z (> z y)) x)) x)
;; for each element y, filter x for elements greater than y

(filter (λ y (> (count (λ z (== z y)) x) 1)) x)
;; keep elements that appear more than once in x

(map (λ y (sum (filter (λ z (< z y)) x))) x)
;; for each element, sum all elements less than it

(fold (λ acc elem (concat acc (map (λ z (* z elem)) acc))) [1] x)
;; complex fold with inner map
```

These are natural, interesting list transformations that require one or two levels of higher-order nesting. The "extract duplicates" program from the research notes:
```scheme
(map (λ g (first g))
  (filter (λ g (> (length g) 1))
    (group (λ y y) (sort (λ y y) x))))
```
requires size ~12 in the raw DSL but only nesting depth 1 (the lambdas passed to `map`, `filter`, `group`, `sort` are all first-order).

---

## 2. Architectural Change: Contextual Program Banks

### Problem

In V1, the `ProgramBank` is global — it stores programs that may reference only `x : list[int]`. A program like `(> _p0 0)` that references a lambda-bound variable `_p0 : int` has no place in this bank. V1 worked around this by doing a separate mini-enumeration for lambda bodies, but that mini-enumeration couldn't recurse.

### Solution

Replace `ProgramBank` with `ContextualBank`, a hierarchical structure where child banks inherit from parents, mirroring lexical scoping.

```python
class ContextualBank:
    """
    Program bank parameterised by a variable context.
    
    The bank for context {x: list[int], _p0: int} extends the bank
    for context {x: list[int]} — it contains everything the parent
    has, plus new programs that use _p0.
    
    This structure mirrors lexical scoping: entering a lambda body
    creates a child bank that inherits all outer programs.
    """
    
    def __init__(self, parent: 'ContextualBank | None' = None):
        self.parent = parent
        # Only stores programs NEW to this context level
        self._local: dict[TypeType, dict[int, list[TypedProgram]]] = \
            defaultdict(lambda: defaultdict(list))
        self._local_fingerprints: dict[TypeType, FingerprintTable] = \
            defaultdict(FingerprintTable)
    
    def get(self, type_: TypeType, size: int) -> list[TypedProgram]:
        """Get programs from this context AND all ancestor contexts."""
        results = list(self._local.get(type_, {}).get(size, []))
        if self.parent is not None:
            results.extend(self.parent.get(type_, size))
        return results
    
    def add_local(self, prog: TypedProgram) -> bool:
        """
        Add a program that is new to this context level.
        Checks for fingerprint duplicates against this level AND all ancestors.
        """
        # Check ancestor banks first — don't re-add programs already known
        if self.parent and self.parent.contains_fingerprint(prog.type, prog.fingerprint):
            return False
        fp_table = self._local_fingerprints[prog.type]
        if fp_table.insert(prog.fingerprint, prog.ast):
            self._local[prog.type][prog.size].append(prog)
            return True
        return False
    
    def contains_fingerprint(self, type_: TypeType, fp: Fingerprint) -> bool:
        """Check if a fingerprint exists at this level or any ancestor."""
        if self._local_fingerprints[type_].contains(fp):
            return True
        if self.parent:
            return self.parent.contains_fingerprint(type_, fp)
        return False
    
    def local_count(self) -> int:
        """Count programs added at this context level only."""
        return sum(
            len(progs)
            for by_size in self._local.values()
            for progs in by_size.values()
        )
    
    def total_count(self) -> int:
        """Count all accessible programs (this level + ancestors)."""
        count = self.local_count()
        if self.parent:
            count += self.parent.total_count()
        return count
```

### Hierarchy Example

For the program `(filter (λ _p0 (> (count (λ _p1 (== _p1 _p0)) x) 1)) x)`:

```
Level 0 (top-level): context = {x: list[int]}
  Bank contains: x, 0, 1, 2, 3, [], (length x), (reverse x), (sum x), ...
  
  └─ Level 1 (filter lambda): context = {x: list[int], _p0: int}
       Inherits everything from Level 0, plus:
       _p0, (+ _p0 1), (> _p0 0), (is_even _p0), ...
       
       └─ Level 2 (count lambda): context = {x: list[int], _p0: int, _p1: int}
            Inherits everything from Levels 0 and 1, plus:
            _p1, (== _p1 _p0), (> _p1 _p0), (+ _p1 _p0), ...
```

---

## 3. Unified Recursive Enumerator

### Replace Three Methods With One

V1 has three separate enumeration methods:
- `_enumerate_first_order()` — for functions with no Callable arguments
- `_enumerate_higher_order()` — for functions with Callable arguments
- `_enumerate_lambdas()` — flat mini-enumeration for lambda bodies

Replace all three with a single recursive method:

```python
def _enumerate_in_context(
    self,
    bank: ContextualBank,
    context: dict[str, TypeType],
    max_size: int,
    nesting_depth: int = 0,
) -> ContextualBank:
    """
    Enumerate all programs up to max_size in the given variable context.
    
    This is the single recursive entry point. Higher-order function
    arguments trigger recursive calls with extended contexts and
    child banks.
    
    Args:
        bank: The contextual bank to populate
        context: Variable name -> type mapping for all bound variables
        max_size: Maximum program size to enumerate
        nesting_depth: Current depth of lambda nesting (for explosion control)
    
    Returns:
        The populated bank
    """
    # Base case: add context-specific variables as size-1 atoms
    # (Only variables NOT already in a parent bank)
    for var_name, var_type in context.items():
        node = VariableNode(var_name)
        fp = self._fingerprint_in_context(node, context)
        if fp is not None:
            bank.add_local(TypedProgram(node, var_type, fp, size=1))
    
    # Inductive case
    for size in range(2, max_size + 1):
        for func_name in self.grammar.names:
            func_info = self.grammar[func_name]
            arg_types = self._resolve_types(func_info['arg_types'])
            ret_type = self._resolve_type(func_info['ret_type'])
            if ret_type is None:
                continue
            
            arity = len(arg_types)
            remaining = size - 1
            
            for partition in integer_partitions(remaining, arity):
                self._enumerate_application(
                    bank, context, func_name,
                    arg_types, ret_type, partition, size,
                    nesting_depth,
                )
    
    return bank
```

### Application Enumeration Handles All Cases Uniformly

```python
def _enumerate_application(
    self, bank, context, func_name, arg_types, ret_type,
    partition, total_size, nesting_depth,
):
    """
    Enumerate one function application.
    
    For Callable-typed arguments, recursively enumerate lambda bodies
    in an extended context via _enumerate_lambda_arg.
    For non-Callable arguments, look up existing programs in the bank.
    """
    arg_candidate_lists = []
    
    for i, (arg_type, arg_size) in enumerate(zip(arg_types, partition)):
        if get_origin(arg_type) == CallableOrig:
            lambdas = self._enumerate_lambda_arg(
                bank, context, arg_type, arg_size, nesting_depth,
            )
            if not lambdas:
                return
            arg_candidate_lists.append(lambdas)
        else:
            candidates = bank.get(arg_type, arg_size)
            if not candidates:
                return
            arg_candidate_lists.append(candidates)
    
    for combo in itertools.product(*arg_candidate_lists):
        node = ApplicationNode(
            VariableNode(func_name),
            [c.ast if isinstance(c, TypedProgram) else c for c in combo]
        )
        fp = self._fingerprint_in_context(node, context)
        if fp is not None:
            prog = TypedProgram(node, ret_type, fp, total_size)
            bank.add_local(prog)
```

### Lambda Argument Enumeration via Recursive Bank Construction

```python
def _enumerate_lambda_arg(
    self,
    parent_bank: ContextualBank,
    parent_context: dict[str, TypeType],
    callable_type: TypeType,
    available_size: int,
    nesting_depth: int,
) -> list[TypedProgram]:
    """
    Enumerate lambda expressions by recursively enumerating bodies
    in an extended context with a child bank.
    
    Args:
        parent_bank: Bank for the enclosing scope
        parent_context: Variable context for the enclosing scope
        callable_type: The Callable type required (e.g. Callable[[int], bool])
        available_size: Total size budget for the lambda node
        nesting_depth: Current nesting depth
    
    Returns:
        List of TypedProgram entries wrapping LambdaNode ASTs
    """
    args = get_args(callable_type)
    param_types = args[0]    # list of parameter types
    body_type = args[1]      # return type
    
    body_budget = available_size - 1  # lambda node itself costs 1
    if body_budget < 1:
        return []
    
    # Generate fresh parameter names (avoid collision with existing context)
    param_names = self._fresh_param_names(len(param_types), parent_context)
    
    # Extend context with lambda parameters
    child_context = parent_context.copy()
    for pname, ptype in zip(param_names, param_types):
        child_context[pname] = ptype
    
    # Get or build the child bank (with caching — see Section 5)
    child_bank = self._get_or_build_child_bank(
        parent_bank, child_context, body_budget, nesting_depth + 1
    )
    
    # Collect all body terms of the target type, wrap as lambdas
    results = []
    for body_size in range(1, body_budget + 1):
        for prog in child_bank.get(body_type, body_size):
            lambda_node = LambdaNode(param_names, prog.ast)
            results.append(TypedProgram(
                lambda_node, callable_type, prog.fingerprint, available_size
            ))
    
    return results
```

---

## 4. Fingerprinting in Extended Contexts

### Problem

A term like `(> _p0 0)` is not a closed expression — it references the lambda-bound variable `_p0`. It cannot be evaluated in isolation. To fingerprint it, we need to supply values for all free variables.

### Solution

Fingerprint terms in extended contexts by wrapping them in lambdas for all context variables and evaluating with **probe values** for inner parameters:

```python
# Probe values for lambda-bound variables, by type
PROBE_VALUES = {
    int: [0, 1, 3],
    bool: [True, False],
    # list[int] probes not needed — lambda params are rarely list-typed
    # in practice (only fold's accumulator when T2 = list[int])
}

def _fingerprint_in_context(
    self, node: ASTNode, context: dict[str, TypeType]
) -> Fingerprint | None:
    """
    Fingerprint a term that may reference any variables in context.
    
    The test suite provides values for the input variable x.
    Probe values provide representative values for lambda-bound parameters.
    The fingerprint is the tuple of results across all combinations of
    test inputs × probe values.
    """
    # Separate the input variable from lambda-bound parameters
    inner_params = [
        (name, typ) for name, typ in context.items()
        if name != self.input_var_name
    ]
    
    if not inner_params:
        # Top-level context: standard fingerprinting via (λ x <term>)
        return self._fingerprint(node)
    
    # Build closed term: (λ x (λ _p0 (λ _p1 ... <term>)))
    # Outermost lambda binds x, inner lambdas bind the parameters
    closed = node
    for pname, _ in reversed(inner_params):
        closed = LambdaNode([pname], closed)
    closed = LambdaNode([self.input_var_name], closed)
    
    try:
        compiled = self.jit.compile(closed)
    except Exception:
        return None
    
    # Compute probe value combinations for inner parameters
    inner_probes = [PROBE_VALUES.get(typ, [0]) for _, typ in inner_params]
    probe_combos = list(itertools.product(*inner_probes))
    
    values = []
    for inp in self.test_suite:
        for probes in probe_combos:
            try:
                result = compiled(inp)
                for p in probes:
                    result = result(p)
                values.append(make_hashable(result))
            except Exception:
                values.append(FAIL)
    
    return Fingerprint(tuple(values))
```

### Fingerprint Length Analysis

The fingerprint length is `|test_suite| × product(|probes| for each inner param)`:

| Context | Inner params | Probe combos | × 10 inputs | Total |
|---------|-------------|-------------|-------------|-------|
| Top-level `{x}` | 0 | 1 | 10 | **10** |
| filter/map lambda `{x, _p0: int}` | 1 | 3 | 10 | **30** |
| fold accumulator `{x, _acc: int, _elem: int}` | 2 | 9 | 10 | **90** |
| Nested: filter inside map `{x, _p0: int, _p1: int}` | 2 | 9 | 10 | **90** |
| Double-nested `{x, _p0, _p1, _p2}` | 3 | 27 | 10 | **270** |

All of these are fine for hashing. The cost is in *evaluation*: each candidate program at nesting depth 2 requires 90 evaluations instead of 10. This is the main performance cost of nesting and is unavoidable — it's the price of correctly distinguishing programs in extended contexts.

### Probe Value Selection

The probes `{0, 1, 3}` for `int` are chosen to distinguish:
- Additive vs multiplicative operations: `(+ _p0 1)` gives `{1, 2, 4}`, `(* _p0 1)` gives `{0, 1, 3}`
- Identity vs constant: `_p0` gives `{0, 1, 3}`, `0` gives `{0, 0, 0}`
- Even/odd sensitivity: 0 and 1 differ in parity, 3 adds a non-trivial odd value
- Comparison ordering: 0 < 1 < 3 provides three distinct comparison outcomes

The probes `{True, False}` for `bool` are exhaustive — they cover the entire domain.

---

## 5. Explosion Control

### The Danger

Without limits, nesting is exponential. At each level, you enumerate up to `body_budget` programs, and each higher-order function at that level triggers another nested enumeration. With ~7 higher-order functions in the grammar, and each potentially appearing at multiple size levels, the recursion tree fans out rapidly.

### Control Mechanism 1: Nesting Depth Limit

```python
MAX_NESTING_DEPTH = 2  # Configurable

def _enumerate_application(self, bank, context, func_name, arg_types, 
                            ret_type, partition, total_size, nesting_depth):
    for i, (arg_type, arg_size) in enumerate(zip(arg_types, partition)):
        if get_origin(arg_type) == CallableOrig:
            if nesting_depth >= MAX_NESTING_DEPTH:
                # At max depth: only enumerate first-order lambda bodies
                lambdas = self._enumerate_lambda_first_order_only(
                    bank, context, arg_type, arg_size
                )
            else:
                # Below max depth: full recursive enumeration
                lambdas = self._enumerate_lambda_arg(
                    bank, context, arg_type, arg_size, nesting_depth
                )
            ...
```

Recommended starting values:
- `max_nesting = 1`: catches the most important cases (map-inside-filter, count-inside-filter, etc.) while keeping cost manageable
- `max_nesting = 2`: needed for three-level compositions; expect ~10× more computation than depth 1

### Control Mechanism 2: Size Budget Decay

The size budget naturally decays through recursion. When the outer program has size 6 and uses `filter` (cost 1) with lambda budget 3, the inner enumeration goes to size 2. At size 2, the inner body can only be `(f atom)` for some unary function — there's no room for another higher-order function application. This natural decay is the primary explosion limiter.

The dangerous case is when the outer size budget is large (7+). At that point, the lambda body budget can be 4–5, which leaves room for inner higher-order functions with their own lambda bodies of budget 2–3. The nesting depth limit catches this.

### Control Mechanism 3: Child Bank Caching

Multiple higher-order functions at the same size level need lambda bodies in the same extended context. Without caching, you'd enumerate the same child bank once per higher-order function:

```python
def _get_or_build_child_bank(
    self,
    parent_bank: ContextualBank,
    child_context: dict[str, TypeType],
    body_budget: int,
    nesting_depth: int,
) -> ContextualBank:
    """
    Return a cached child bank if one exists for this context and budget,
    otherwise build one by recursive enumeration.
    """
    # Cache key: the parent bank identity, the context (as frozen set),
    # the body budget, and the nesting depth
    key = (
        id(parent_bank),
        frozenset(child_context.items()),
        body_budget,
        nesting_depth,
    )
    
    if key not in self._child_bank_cache:
        child_bank = ContextualBank(parent=parent_bank)
        self._enumerate_in_context(
            child_bank, child_context, body_budget, nesting_depth
        )
        self._child_bank_cache[key] = child_bank
    
    return self._child_bank_cache[key]
```

This is critical for performance. At any given size level, the higher-order functions `map`, `filter`, `sort`, `group`, `count`, `find`, `mapi`, `filteri`, `foldi` all need lambda bodies in overlapping contexts. Without caching, you'd do 7–10× redundant work. With caching, you enumerate each unique (context, budget) pair exactly once.

**Cache invalidation.** The cache is valid for the lifetime of a single `enumerate()` call. Clear it at the start of each enumeration run:

```python
def enumerate(self) -> ContextualBank:
    self._child_bank_cache = {}
    # ... proceed with enumeration ...
```

### Expected Cost

Rough estimates for `max_nesting = 1`, `max_size = 5`:

| What | V1 (no nesting) | With nesting |
|------|-----------------|--------------|
| Child bank builds (level 1) | 0 | ~5–8 (one per unique context × budget) |
| Total candidate evaluations | ~10⁶ | ~10⁷ (10× from inner enumeration) |
| Fingerprint evaluations per candidate | 10 | 10–30 (depending on level) |
| Wall-clock time (estimate) | ~10 min | ~1–2 hours |
| Distinct programs found | ~10³–10⁴ | ~10⁴–10⁵ |

The 10× cost increase is the price for accessing a qualitatively richer program space. The caching ensures it's only 10× and not 70×.

---

## 6. Changes to the RL Component

### 6.1 State Representation

Add `nesting_depth` to the MDP state:

```python
@dataclass
class SynthesisState:
    target_type: TypeType
    context: dict[str, TypeType]
    parent_func: str | None
    arg_index: int | None
    siblings: list[tuple[ASTNode, Fingerprint | None]]
    depth_budget: int
    nesting_depth: int = 0  # NEW: current lambda nesting level
```

### 6.2 Valid Action Masking

At `nesting_depth >= MAX_NESTING_DEPTH`, exclude higher-order functions from the action space:

```python
def valid_actions(state, grammar, seed_constants):
    actions = []
    # ... literals, variables as before ...
    
    if state.depth_budget > 0:
        for func_name in grammar.names:
            func_info = grammar[func_name]
            resolved_ret = _resolve_type(func_info['ret_type'])
            if resolved_ret != state.target_type:
                continue
            
            # NEW: check nesting depth for higher-order functions
            arg_types = func_info['arg_types']
            is_higher_order = any(
                get_origin(t) == CallableOrig for t in arg_types
            )
            if is_higher_order and state.nesting_depth >= MAX_NESTING_DEPTH:
                continue  # Skip this function at max nesting
            
            actions.append(Action(ActionType.APPLY, func_name))
    
    # ... lambda, if as before ...
    return actions
```

### 6.3 State Transitions

When generating a lambda body, increment `nesting_depth`:

```python
def _generate_lambda(self, state):
    # ... extract param types, body type, param names ...
    
    body_state = SynthesisState(
        target_type=body_type,
        context=new_context,
        parent_func=None,
        arg_index=None,
        siblings=[],
        depth_budget=state.depth_budget - 1,
        nesting_depth=state.nesting_depth + 1,  # INCREMENT
    )
    body_node = self._generate(body_state)
    # ...
```

### 6.4 Policy Network

Add one embedding for nesting depth:

```python
class StateEncoder(nn.Module):
    def __init__(self, type_vocab_size, func_vocab_size, embed_dim=64):
        super().__init__()
        self.type_embed = nn.Embedding(type_vocab_size, embed_dim)
        self.func_embed = nn.Embedding(func_vocab_size + 1, embed_dim)
        self.arg_index_embed = nn.Embedding(8, embed_dim)
        self.depth_embed = nn.Embedding(16, embed_dim)
        self.nesting_embed = nn.Embedding(4, embed_dim)  # NEW: max nesting ~3
        self.context_proj = nn.Linear(16, embed_dim)
        
        self.combine = nn.Linear(6 * embed_dim, embed_dim)  # 5 -> 6 inputs
```

### 6.5 Trajectory Extraction

When walking into lambda bodies, track nesting depth:

```python
def _walk(node, state):
    # ... other cases ...
    
    elif isinstance(node, LambdaNode):
        trajectory.append((state, Action(ActionType.LAMBDA, None)))
        
        args = get_args(state.target_type)
        param_types = args[0]
        body_type = args[1]
        
        new_context = state.context.copy()
        for pname, ptype in zip(node.param, param_types):
            new_context[pname] = ptype
        
        body_state = SynthesisState(
            target_type=body_type,
            context=new_context,
            parent_func=None,
            arg_index=None,
            siblings=[],
            depth_budget=state.depth_budget - 1,
            nesting_depth=state.nesting_depth + 1,  # INCREMENT
        )
        _walk(node.body, body_state)
```

---

## 7. Testing Strategy

### 7.1 Correctness Tests

**Test 1: Nesting depth 1 programs are enumerated.**
Verify that the following programs (or observationally equivalent ones) appear in the corpus:

```scheme
(filter (λ _p0 (> (length x) 3)) x)           ;; uses outer x in inner body
(map (λ _p0 (count (λ _p1 (== _p1 _p0)) x)) x) ;; count occurrences of each element
(filter (λ _p0 (is_in x (+ _p0 1))) x)          ;; keep elements where element+1 is also in x
```

**Test 2: Observational equivalence works across nesting levels.**
Verify that `(map (λ y y) x)` still collapses with `x` (the identity), and that `(filter (λ y true) x)` collapses with `x`.

**Test 3: Nesting depth limit is respected.**
With `max_nesting = 1`, verify that no program in the corpus contains a higher-order function call at depth 2 (a map inside a filter inside a map).

**Test 4: Child bank caching works.**
Instrument the cache to count hits vs misses. Verify that for a given size level, the cache hit rate is > 50% (since multiple higher-order functions share contexts).

### 7.2 Performance Tests

**Test 5: Enumeration completes in reasonable time.**
With `max_size = 5, max_nesting = 1`:
- Target: completes in < 2 hours
- Measure: wall-clock time, peak memory, programs per size level

**Test 6: Corpus size increases.**
Compare corpus sizes:
- V1 (no nesting): baseline
- With nesting depth 1: expect 2–5× more distinct programs
- With nesting depth 2: expect 5–20× more (if tractable)

### 7.3 Semantic Quality Tests

**Test 7: Novel behaviours are discovered.**
Take the fingerprint sets from V1 and the nested version. Verify that the nested version contains fingerprints not present in V1 — these are genuinely new program behaviours that nesting enables.

**Test 8: Degenerate programs are still pruned.**
Verify that nested degenerate programs like `(map (λ y (filter (λ z false) [])) x)` (returns `[[], [], ...]`) are correctly identified and pruned by observational equivalence.

---

## 8. Summary of All File Changes

| File | Change type | Description |
|------|------------|-------------|
| `enumeration/enumerator.py` | **Major rewrite** | Replace `ProgramBank` with `ContextualBank`. Replace three enumeration methods with unified `_enumerate_in_context`. Add `_get_or_build_child_bank` with caching. Add `nesting_depth` parameter throughout. |
| `enumeration/fingerprint.py` | **Extend** | Add `_fingerprint_in_context` with probe values for inner parameters. Add `PROBE_VALUES` constant. |
| `rl/mdp.py` | **Minor update** | Add `nesting_depth: int = 0` field to `SynthesisState`. Update `valid_actions` to gate higher-order functions on nesting depth. |
| `rl/policy.py` | **Minor update** | Add `nesting_embed` to `StateEncoder`. Change `combine` layer input size from `5 * embed_dim` to `6 * embed_dim`. |
| `rl/trajectory.py` | **Minor update** | Increment `nesting_depth` when walking into lambda bodies. |
| `rl/trainer.py` | No change | Training loop is agnostic to nesting depth. |
| `rl/reward.py` | No change | Reward functions work on fingerprints regardless of nesting. |
| `rl/priority_queue.py` | No change | Buffer is agnostic to program structure. |
| `pipeline.py` | **Minor update** | Add `max_nesting` parameter. Pass through to enumerator. |
| `utils.py` | No change | Program size computation is unchanged. |

---

## 9. Configuration Recommendations

### Conservative Start (recommended)
```python
max_size = 5
max_nesting = 1
probe_values = {int: [0, 1, 3], bool: [True, False]}
```
Expected: 2–5× more programs than V1, completes in 1–2 hours.

### Aggressive
```python
max_size = 6
max_nesting = 2
probe_values = {int: [0, 1, 3, -1], bool: [True, False]}
```
Expected: 10–50× more programs, may take 6–12 hours. Only attempt after conservative run succeeds and you've verified memory usage is acceptable.

---

*Depends on: `IMPLEMENTATION_PLAN.md` (base implementation)*
*Updates: Section 6 (enumerator) and Section 7 (MDP) of the implementation plan*
