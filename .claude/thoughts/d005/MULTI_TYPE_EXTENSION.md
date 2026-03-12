# Extension: Multi-Type Instantiation

## Overview

This document describes the changes needed to enumerate programs over multiple concrete type instantiations, replacing the fixed `T1=int, T2=int` assumption. This unlocks programs that operate on `list[list[int]]` (groups), `list[bool]` (predicate results), and list-accumulating folds — all of which are blocked by the fixed instantiation.

This extension is orthogonal to the nesting extension. It changes the **type universe** over which enumeration operates, while nesting changed the **structure** of enumeration. All nesting machinery (ContextualBank, recursive lambda enumeration, child bank caching, probe-based fingerprinting) is unchanged.

In the RL component, each (function, instantiation) pair is a distinct action. This gives the policy explicit control over type instantiation — for example, the policy can learn that `fold` with T2=list[int] is more likely to produce interesting programs than `fold` with T2=bool, and allocate its exploration budget accordingly.

**Depends on:** `IMPLEMENTATION_PLAN.md`, `NESTING_EXTENSION.md` (both assumed fully implemented).

---

## 1. What the Fixed Instantiation Blocks

### 1.1 List-Accumulating Folds

`fold` has type `(T2 → T1 → T2) → T2 → [T1] → T2`. With T2=int, the accumulator is always `int`. This blocks:

```scheme
(fold (λ acc elem (cons elem acc)) [] x)
;; reverse via fold — requires T2 = list[int]

(fold (λ acc elem (if (> elem 0) (cons elem acc) acc)) [] x)
;; filter positives via fold — requires T2 = list[int]

(fold (λ acc elem (append acc (* elem elem))) [] x)
;; map-squares via fold — requires T2 = list[int]
```

### 1.2 Group-Then-Aggregate

`group` returns `list[list[T1]]` = `list[list[int]]`. Mapping over groups requires a lambda of type `list[int] → T2`:

```scheme
(map (λ g (length g)) (group (λ y y) x))
;; count occurrences of each value — lambda param is list[int]

(map (λ g (sum g)) (group (λ y (% y 2)) x))
;; sum even-indexed and odd-indexed groups

(map (λ g (first g)) (filter (λ g (> (length g) 1)) (group (λ y y) x)))
;; extract duplicate values
```

These require T1=list[int] in the outer `map`/`filter` (since the list elements are themselves lists).

### 1.3 Zip Patterns

`zip` returns `list[list[T1]]`. Processing zipped pairs requires the same T1=list[int] lambda parameters:

```scheme
(map (λ pair (+ (first pair) (last pair))) (zip x (reverse x)))
;; pairwise sum with reverse — lambda param is list[int]
```

### 1.4 Boolean Lists

`map` with a predicate produces `list[bool]`:

```scheme
(map (λ y (> y 0)) x)
;; returns [true, false, true, ...] — type list[bool]
```

Boolean lists are less critical than the above, but including them costs very little and completes the type universe.

---

## 2. Type Universe

### 2.1 Concrete Types

Define the set of concrete types the enumerator operates over:

```python
TYPE_UNIVERSE = {
    # Atomic types
    int,
    bool,
    # List types
    list[int],
    list[bool],
    list[list[int]],
}
```

This is a closed, finite set. Every program in the bank has a type drawn from this universe. Every function argument and return type, after instantiation, must fall within this universe.

**Why not `list[list[list[int]]]` or deeper?** Diminishing returns. Programs producing triply-nested lists are rarely interesting for list transformation tasks. The universe can be extended later if needed.

### 2.2 Ground Types for Type Variable Instantiation

Type variables T1 and T2 can each be instantiated to:

```python
GROUND_TYPES = [int, bool, list[int]]
```

Note that `list[list[int]]` is NOT in `GROUND_TYPES` — it arises as `list[T1]` when T1=list[int], but T1 itself is never assigned to `list[list[int]]`. This prevents runaway nesting of list types.

### 2.3 Callable Types

Callable types are not in `TYPE_UNIVERSE` directly — they arise as argument types of higher-order functions and are handled by lambda enumeration. The set of callable types that can appear is determined by the instantiations:

```
Callable[[int], int]              — map/sort/mapi with T1=int, T2=int
Callable[[int], bool]             — filter/count/find with T1=int
Callable[[int], list[int]]        — map with T1=int, T2=list[int] (rare but valid)
Callable[[int, int], int]         — fold with T1=int, T2=int
Callable[[int, int], list[int]]   — fold with T1=int, T2=list[int] (list-accumulating)
Callable[[list[int]], int]        — map over groups with T1=list[int], T2=int
Callable[[list[int]], bool]       — filter over groups with T1=list[int]
Callable[[list[int]], list[int]]  — map over groups with T1=list[int], T2=list[int]
...
```

These are not enumerated explicitly — they're generated on the fly when a higher-order function is instantiated. The lambda enumeration machinery from the nesting extension handles them.

---

## 3. Changes to Type Resolution

### 3.1 Replace `_resolve_type` With Instantiation-Parameterised Version

**Before (fixed instantiation):**
```python
def _resolve_type(self, type_: TypeType) -> TypeType | None:
    subs = SubstitutionTable()
    subs[T1] = int
    subs[T2] = int
    try:
        return substitute_type_vars(type_, subs)
    except Exception:
        return None
```

**After (parameterised):**
```python
def _resolve_type(
    self, type_: TypeType, instantiation: dict[TypeVar, TypeType]
) -> TypeType | None:
    """
    Resolve type variables using the given instantiation.
    
    Args:
        type_: Type possibly containing T1, T2
        instantiation: Mapping from type variables to concrete types
                       e.g. {T1: int, T2: list[int]}
    
    Returns:
        Concrete type with all type variables resolved, or None if
        the resolved type falls outside TYPE_UNIVERSE.
    """
    subs = SubstitutionTable()
    for tv, concrete in instantiation.items():
        subs[tv] = concrete
    try:
        resolved = substitute_type_vars(type_, subs)
    except Exception:
        return None
    
    # Validate: non-callable types must be in TYPE_UNIVERSE
    if get_origin(resolved) != CallableOrig and resolved not in TYPE_UNIVERSE:
        return None
    
    return resolved

def _resolve_types(
    self, types: tuple, instantiation: dict[TypeVar, TypeType]
) -> list[TypeType | None]:
    return [self._resolve_type(t, instantiation) for t in types]
```

Update **every call site** that previously called `_resolve_type(t)` to pass an instantiation dict. This is a mechanical find-and-replace across `enumerator.py`.

### 3.2 Compute Valid Instantiations Per Function

At initialisation time, compute and cache the set of valid instantiations for each grammar function:

```python
from .grammar import T1, T2

def _compute_valid_instantiations(self) -> dict[str, list[dict[TypeVar, TypeType]]]:
    """
    For each grammar function, find all type variable assignments
    such that all argument types and the return type resolve to
    types within TYPE_UNIVERSE (or valid Callable types).
    
    Called once at initialisation and cached.
    """
    result = {}
    
    for func_name in self.grammar.names:
        func_info = self.grammar[func_name]
        
        # Collect free type variables in this function's signature
        free_tvs = set()
        subs = SubstitutionTable()
        for t in func_info['arg_types']:
            free_tvs |= get_free_types(t, subs)
        free_tvs |= get_free_types(func_info['ret_type'], subs)
        
        free_tvs = sorted(free_tvs, key=str)  # Deterministic ordering
        
        if not free_tvs:
            # Monomorphic function — one trivial instantiation
            result[func_name] = [{}]
            continue
        
        # Enumerate all assignments of free type variables to ground types
        valid = []
        for assignment in itertools.product(GROUND_TYPES, repeat=len(free_tvs)):
            inst = dict(zip(free_tvs, assignment))
            
            # Check that all resolved types are valid
            try:
                resolved_args = [self._resolve_type(t, inst) for t in func_info['arg_types']]
                resolved_ret = self._resolve_type(func_info['ret_type'], inst)
            except Exception:
                continue
            
            if None in resolved_args or resolved_ret is None:
                continue
            
            valid.append(inst)
        
        result[func_name] = valid
    
    return result
```

**Cache this in `__init__`:**
```python
def __init__(self, grammar, test_suite, ...):
    # ... existing init ...
    self._valid_instantiations = self._compute_valid_instantiations()
```

### 3.3 Example: What Instantiations Each Function Gets

| Function | Type | Free TVs | Valid instantiations |
|----------|------|----------|---------------------|
| `+` | `int → int → int` | none | `[{}]` (monomorphic) |
| `length` | `[T1] → int` | T1 | `[{T1:int}, {T1:bool}, {T1:list[int]}]` |
| `reverse` | `[T1] → [T1]` | T1 | `[{T1:int}, {T1:bool}, {T1:list[int]}]` |
| `map` | `(T1→T2) → [T1] → [T2]` | T1, T2 | `[{T1:int,T2:int}, {T1:int,T2:bool}, {T1:int,T2:list[int]}, {T1:bool,T2:int}, ..., {T1:list[int],T2:list[int]}]` — up to 9 |
| `fold` | `(T2→T1→T2) → T2 → [T1] → T2` | T1, T2 | same 9 combinations |
| `group` | `(T1→T2) → [T1] → [[T1]]` | T1, T2 | `[{T1:int,T2:int}, {T1:int,T2:bool}, ...]` — ret type is `list[list[int]]` when T1=int |
| `flatten` | `[[T1]] → [T1]` | T1 | `[{T1:int}, {T1:bool}, {T1:list[int]}]` |
| `sum` | `[int] → int` | none | `[{}]` |
| `max` | `[int] → int` | none | `[{}]` |

Most functions have 1–3 valid instantiations. `map` and `fold` have up to 9, but many of these produce argument types that have no programs in the bank (e.g., `map` with T1=bool, T2=list[int] needs a `Callable[[bool], list[int]]` lambda, which is unlikely to produce interesting programs). The Cartesian product enumeration naturally handles this: if no programs exist at the required type, that branch produces nothing.

---

## 4. Changes to the Enumerator

### 4.1 Base Case

The base case must now seed atoms for all types in the universe:

```python
def _enumerate_base_case(self, bank: ContextualBank, context: dict[str, TypeType]):
    """Populate the bank with all size-1 atoms across all types."""
    
    # Integer constants
    for c in self.seed_constants:
        node = NumberNode(c)
        self._try_add(node, int, size=1, bank=bank, context=context)
    
    # Boolean constants
    for b in [True, False]:
        node = BooleanNode(b)
        self._try_add(node, bool, size=1, bank=bank, context=context)
    
    # Empty lists — one per list type
    for list_type in [list[int], list[bool], list[list[int]]]:
        node = ListNode([])
        self._try_add(node, list_type, size=1, bank=bank, context=context)
    
    # Context variables (input variable + any lambda-bound params)
    for var_name, var_type in context.items():
        node = VariableNode(var_name)
        fp = self._fingerprint_in_context(node, context)
        if fp is not None:
            prog = TypedProgram(ast=node, type=var_type, fingerprint=fp, size=1)
            bank.add_local(prog)
```

**Note on empty lists:** `ListNode([])` can inhabit any list type. We add it once per list type in the universe. The fingerprints will differ contextually, but in practice the values are identical (`[]`). They're stored under different type keys in the bank and will only be considered as arguments to functions that expect that specific type.

### 4.2 Inductive Case

The main enumeration loop now iterates over instantiations per function:

```python
def _enumerate_at_size(
    self, bank: ContextualBank, context: dict[str, TypeType],
    size: int, nesting_depth: int
):
    """Enumerate all programs of exactly the given size."""
    
    for func_name in self.grammar.names:
        # Iterate over all valid type instantiations for this function
        for inst in self._valid_instantiations[func_name]:
            func_info = self.grammar[func_name]
            arg_types = self._resolve_types(func_info['arg_types'], inst)
            ret_type = self._resolve_type(func_info['ret_type'], inst)
            
            # Skip if any type couldn't be resolved
            if None in arg_types or ret_type is None:
                continue
            
            arity = len(arg_types)
            remaining = size - 1
            
            for partition in integer_partitions(remaining, arity):
                self._enumerate_application(
                    bank, context, func_name,
                    arg_types, ret_type, partition, size,
                    nesting_depth,
                )
```

This is the only structural change to the enumeration loop. The `_enumerate_application` method, the lambda enumeration, and the nesting machinery are all unchanged — they already work with arbitrary concrete types.

### 4.3 What Happens Inside Lambda Enumeration (No Change Needed)

When `_enumerate_lambda_arg` is called for, say, `map` with instantiation {T1: list[int], T2: int}, the callable type is `Callable[[list[int]], int]`. The lambda enumeration:

1. Extracts param types: `[list[int]]`
2. Extracts body type: `int`
3. Creates a child context: `{x: list[int], _p0: list[int]}`
4. Recursively enumerates bodies of type `int` in this context

The child bank will now contain programs like `(length _p0)`, `(sum _p0)`, `(max _p0)`, `(first _p0)` — all of which take `_p0 : list[int]` and return `int`. These are exactly the group-aggregation functions we wanted.

No changes to `_enumerate_lambda_arg`, `_get_or_build_child_bank`, or `ContextualBank` are needed. They're already parameterised by types.

---

## 5. Changes to Fingerprinting

### 5.1 Probe Values for New Types

Add probe values for `list[int]` and `list[bool]` parameters (used when lambda-bound variables have these types):

```python
PROBE_VALUES = {
    int: [0, 1, 3],
    bool: [True, False],
    list[int]: [[], [1], [2, 1, 3]],       # NEW
    list[bool]: [[], [True], [True, False]], # NEW
}
```

**Probe value design for `list[int]`:**
- `[]` — tests empty-list handling
- `[1]` — singleton
- `[2, 1, 3]` — short unsorted list with distinct elements

These distinguish `length`, `sum`, `max`, `first`, `reverse`, `sort`, etc. from each other when applied to `_p0 : list[int]`.

### 5.2 Fingerprint Length Impact

With `list[int]` probes (3 values), a lambda over groups has fingerprint length 10 × 3 = 30 — same as the existing `int` probe case. A fold with T2=list[int] has accumulator type `list[int]` and element type `int`, so the fold body lambda has two parameters with probe counts 3 and 3, giving 10 × 9 = 90. This is within the bounds already handled by the nesting extension.

### 5.3 Fingerprint Type Sensitivity

**Important:** Two programs with different types but identical fingerprints should NOT be collapsed. The existing design handles this correctly: the `ContextualBank` indexes fingerprints by type, so two programs are only compared if they share the same resolved type. No change needed.

---

## 6. Changes to Quality Filters

No changes to the filter predicates themselves. Quality filters operate on fingerprints, which are type-agnostic tuples. A program of type `list[list[int]] → list[int]` is filtered by the same variability/non-constancy criteria as a program of type `list[int] → list[int]`.

The **corpus extraction** must filter for programs whose return type is interesting:

```python
def extract_corpus(self, min_variability=0.3, min_successes=3):
    """
    Extract final corpus of quality-filtered programs.
    
    Programs are open terms in the top-level bank. Each term of type τ
    in context {x: list[int]} corresponds to a program of type
    list[int] → τ.
    """
    INTERESTING_RETURN_TYPES = {
        list[int],           # [Int] → [Int]
        int,                 # [Int] → Int
        bool,                # [Int] → Bool
        list[list[int]],     # [Int] → [[Int]]
        list[bool],          # [Int] → [Bool]
    }
    
    corpus = []
    for ret_type in INTERESTING_RETURN_TYPES:
        for size in range(1, self.max_size + 1):
            for prog in self.bank.get(ret_type, size):
                if passes_quality_filter(
                    prog.fingerprint, min_successes, min_variability
                ):
                    corpus.append(prog)
    return corpus
```

---

## 7. Changes to the RL Component

The central design decision: **each (function_name, instantiation) pair is a separate action in the RL action space.** The policy explicitly chooses which type instantiation to use when it applies a function. This gives the policy fine-grained control over type-level decisions — it can learn, for example, that `fold` with T2=list[int] is valuable for building list-accumulating reductions, or that `map` with T2=bool produces predicate lists that are useful as intermediate values.

### 7.1 Action Representation

Update the `Action` dataclass to carry the instantiation as part of `APPLY` actions:

```python
@dataclass(frozen=True)
class Action:
    action_type: ActionType
    payload: Any = None
    instantiation: tuple | None = None  # NEW: frozen dict as tuple of pairs
    
    def __hash__(self):
        return hash((self.action_type, self.payload, self.instantiation))
    
    def __eq__(self, other):
        return (isinstance(other, Action) and 
                self.action_type == other.action_type and
                self.payload == other.payload and
                self.instantiation == other.instantiation)
```

The instantiation is stored as a `tuple` of `(TypeVar, TypeType)` pairs (frozen for hashability) rather than a dict. Helper to convert:

```python
def freeze_instantiation(inst: dict[TypeVar, TypeType]) -> tuple:
    """Convert instantiation dict to a hashable tuple for use in Action."""
    return tuple(sorted(inst.items(), key=lambda kv: str(kv[0])))

def thaw_instantiation(frozen: tuple) -> dict[TypeVar, TypeType]:
    """Convert frozen instantiation back to a dict."""
    return dict(frozen)
```

An `APPLY` action is constructed as:

```python
Action(
    ActionType.APPLY,
    payload="fold",
    instantiation=freeze_instantiation({T1: int, T2: list[int]})
)
```

All other action types (`LITERAL_*`, `VARIABLE`, `LAMBDA`, `IF`) have `instantiation=None`.

### 7.2 Action Vocabulary

The action vocabulary now includes one entry per (function, instantiation) pair:

```python
def build_action_vocab(
    grammar: Grammar,
    seed_constants: list[int],
    valid_instantiations: dict[str, list[dict[TypeVar, TypeType]]],
) -> dict[Action, int]:
    """
    Build mapping from Action -> integer index.
    
    The vocabulary includes one entry per (function, instantiation) pair,
    so polymorphic functions contribute multiple entries.
    """
    vocab = {}
    idx = 0
    
    # Literal actions
    for c in seed_constants:
        vocab[Action(ActionType.LITERAL_INT, c)] = idx; idx += 1
    vocab[Action(ActionType.LITERAL_BOOL, True)] = idx; idx += 1
    vocab[Action(ActionType.LITERAL_BOOL, False)] = idx; idx += 1
    vocab[Action(ActionType.LITERAL_EMPTY_LIST, None)] = idx; idx += 1
    
    # Variable actions (fixed set of possible names)
    for var_name in ["x", "_p0", "_p1", "_p2"]:
        vocab[Action(ActionType.VARIABLE, var_name)] = idx; idx += 1
    
    # Function application actions: one per (function, instantiation)
    for func_name in grammar.names:
        for inst in valid_instantiations[func_name]:
            frozen = freeze_instantiation(inst)
            action = Action(ActionType.APPLY, func_name, frozen)
            vocab[action] = idx; idx += 1
    
    # Lambda and if
    vocab[Action(ActionType.LAMBDA, None)] = idx; idx += 1
    vocab[Action(ActionType.IF, None)] = idx; idx += 1
    
    return vocab
```

**Expected vocabulary size:**

| Category | Count (before) | Count (after) |
|----------|---------------|---------------|
| Literals (int) | 4 | 4 |
| Literals (bool) | 2 | 2 |
| Literal (empty list) | 1 | 1 |
| Variables | 4 | 4 |
| APPLY (monomorphic functions, ~15) | 15 | 15 × 1 = 15 |
| APPLY (unary polymorphic, ~25) | 25 | 25 × 3 = 75 |
| APPLY (binary polymorphic, ~15) | 15 | 15 × 9 = 135 (upper bound; many filtered) |
| APPLY (other polymorphic, ~15) | 15 | 15 × 3 = 45 |
| Lambda | 1 | 1 |
| If | 1 | 1 |
| **Total** | **~83** | **~200–280** |

The action space roughly triples. This is manageable — the policy network's output head grows from ~83 to ~250 logits, which is a trivial increase in parameters. The valid action mask ensures that at any given state, only a small fraction of these are valid (typically 10–30 actions).

### 7.3 Valid Actions

The `valid_actions` function now emits a separate action per valid (function, instantiation) pair:

```python
def valid_actions(
    state: SynthesisState,
    grammar: Grammar,
    seed_constants: list[int],
    valid_instantiations: dict[str, list[dict[TypeVar, TypeType]]],
) -> list[Action]:
    """
    Enumerate all valid actions from the current state.
    
    For function applications, emits one action per (function, instantiation)
    pair whose return type matches the target type.
    """
    actions = []
    t = state.target_type
    
    # --- Literals ---
    if t == int:
        for c in seed_constants:
            actions.append(Action(ActionType.LITERAL_INT, c))
    if t == bool:
        actions.append(Action(ActionType.LITERAL_BOOL, True))
        actions.append(Action(ActionType.LITERAL_BOOL, False))
    if t in {list[int], list[bool], list[list[int]]}:
        actions.append(Action(ActionType.LITERAL_EMPTY_LIST, None))
    
    # --- Variables ---
    for var_name, var_type in state.context.items():
        if var_type == t:
            actions.append(Action(ActionType.VARIABLE, var_name))
    
    # --- Function applications ---
    if state.depth_budget > 0:
        for func_name in grammar.names:
            func_info = grammar[func_name]
            arg_types_raw = func_info['arg_types']
            
            # Check nesting constraint for higher-order functions
            is_higher_order = any(
                get_origin(at) == CallableOrig for at in arg_types_raw
            )
            if is_higher_order and state.nesting_depth >= MAX_NESTING_DEPTH:
                continue
            
            # Emit one action per instantiation whose return type matches t
            for inst in valid_instantiations[func_name]:
                resolved_ret = _resolve_type(func_info['ret_type'], inst)
                if resolved_ret == t:
                    frozen = freeze_instantiation(inst)
                    actions.append(Action(ActionType.APPLY, func_name, frozen))
    
    # --- Lambda ---
    if get_origin(t) == CallableOrig:
        actions.append(Action(ActionType.LAMBDA, None))
    
    # --- If ---
    if state.depth_budget > 0:
        actions.append(Action(ActionType.IF, None))
    
    return actions
```

**Why this matters for the policy.** Consider a state where `target_type = int` and we want to apply `fold`. Under the fixed instantiation, there was one `APPLY(fold)` action. Now there are potentially several:

- `APPLY(fold, {T1:int, T2:int})` — fold produces `int`, accumulator is `int`, standard arithmetic fold
- `APPLY(fold, {T1:bool, T2:int})` — fold over a `list[bool]` with `int` accumulator (unlikely but valid)
- `APPLY(fold, {T1:list[int], T2:int})` — fold over `list[list[int]]` with `int` accumulator

Each of these leads to a different set of child argument types, so the policy's downstream decisions are fundamentally different. By making the instantiation explicit in the action, the policy can learn that `APPLY(fold, {T1:int, T2:int})` is the common useful case when producing an `int`, while `APPLY(fold, {T1:list[int], T2:int})` is useful specifically when processing group results.

### 7.4 Type Vocabulary

The type vocabulary must include all types in the universe plus all callable types that can appear:

```python
def build_type_vocab(
    grammar: Grammar,
    valid_instantiations: dict[str, list[dict[TypeVar, TypeType]]],
) -> dict[TypeType, int]:
    """Build mapping from concrete types to integer indices."""
    types = set()
    
    # Base types
    types |= TYPE_UNIVERSE
    
    # Callable types from all valid instantiations
    for func_name, instantiations in valid_instantiations.items():
        func_info = grammar[func_name]
        for inst in instantiations:
            for arg_type in func_info['arg_types']:
                resolved = _resolve_type(arg_type, inst)
                if resolved is not None:
                    types.add(resolved)
            resolved_ret = _resolve_type(func_info['ret_type'], inst)
            if resolved_ret is not None:
                types.add(resolved_ret)
    
    return {t: i for i, t in enumerate(sorted(types, key=str))}
```

### 7.5 Episode Runner: Application Generation

When the policy selects an `APPLY` action, the instantiation is encoded directly in the action. No inference needed:

```python
def _generate_application(self, state, action):
    func_name = action.payload
    inst = thaw_instantiation(action.instantiation)
    
    func_info = self.grammar[func_name]
    arg_types = [_resolve_type(t, inst) for t in func_info['arg_types']]
    
    arg_nodes = []
    for i, arg_type in enumerate(arg_types):
        child_state = SynthesisState(
            target_type=arg_type,
            context=state.context,
            parent_func=func_name,
            arg_index=i,
            siblings=[(n, None) for n in arg_nodes],
            depth_budget=state.depth_budget - 1,
            nesting_depth=state.nesting_depth,
        )
        arg_node = self._generate(child_state)
        if arg_node is None:
            return None
        arg_nodes.append(arg_node)
    
    return ApplicationNode(VariableNode(func_name), arg_nodes)
```

### 7.6 Trajectory Extraction

When extracting trajectories from existing programs, we must reconstruct which instantiation was used. Since we know the return type (from the parent state's target type) and the argument types (from the child nodes), we can determine the instantiation:

```python
def _infer_instantiation_from_types(
    func_name: str,
    target_ret_type: TypeType,
    valid_instantiations: dict[str, list[dict[TypeVar, TypeType]]],
) -> dict[TypeVar, TypeType] | None:
    """
    Given a function and the desired return type, find the unique
    instantiation that produces that return type.
    
    Used during trajectory extraction (not during episode running,
    where the action already carries the instantiation).
    """
    for inst in valid_instantiations[func_name]:
        func_info = grammar[func_name]
        resolved_ret = _resolve_type(func_info['ret_type'], inst)
        if resolved_ret == target_ret_type:
            return inst
    return None
```

Then in the trajectory walk:

```python
def _walk(node, state):
    # ...
    elif isinstance(node, ApplicationNode):
        if isinstance(node.function, VariableNode):
            func_name = node.function.name
            
            # Infer which instantiation was used
            inst = _infer_instantiation_from_types(
                func_name, state.target_type, valid_instantiations
            )
            if inst is None:
                raise ValueError(
                    f"Cannot infer instantiation for {func_name} "
                    f"with return type {state.target_type}"
                )
            
            frozen = freeze_instantiation(inst)
            trajectory.append((
                state,
                Action(ActionType.APPLY, func_name, frozen)
            ))
            
            arg_types = [
                _resolve_type(t, inst)
                for t in grammar[func_name]['arg_types']
            ]
            
            generated_siblings = []
            for i, (arg_node, arg_type) in enumerate(zip(node.arguments, arg_types)):
                child_state = SynthesisState(
                    target_type=arg_type,
                    context=state.context,
                    parent_func=func_name,
                    arg_index=i,
                    siblings=list(generated_siblings),
                    depth_budget=state.depth_budget - 1,
                    nesting_depth=state.nesting_depth,
                )
                _walk(arg_node, child_state)
                generated_siblings.append((arg_node, None))
    # ...
```

**Ambiguity in instantiation inference.** It is possible that multiple instantiations of the same function produce the same return type. For example, `reverse` with T1=int and `reverse` with T1=bool both return `list[int]` when applied to `x : list[int]` — wait, `reverse` with T1=bool would expect `list[bool]`, not `list[int]`, so it wouldn't match. In general, the return type combined with the input list type `list[int]` uniquely determines the instantiation for most functions. If genuine ambiguity arises (the same function under two different instantiations produces the same return type AND accepts the same argument types), both instantiations produce identical programs, so picking either one is correct.

To handle this robustly: if multiple instantiations match, prefer the one where all argument types match the types of the actual child nodes in the AST. This can be checked by type-inspecting the children. As a simpler fallback, just pick the first match — document this as a known approximation.

### 7.7 Policy Network

The `StateEncoder` is unchanged from the nesting extension. The only change is the sizes of the vocabularies passed to the constructor:

```python
type_vocab = build_type_vocab(grammar, valid_instantiations)  # Larger
action_vocab = build_action_vocab(grammar, seed_constants, valid_instantiations)  # ~3x larger

policy = PolicyNetwork(
    action_vocab_size=len(action_vocab),   # ~250 instead of ~83
    type_vocab_size=len(type_vocab),       # ~20 instead of ~5
    func_vocab_size=len(func_vocab),       # Unchanged
)
```

The output head (`nn.Linear(hidden_dim, action_vocab_size)`) grows proportionally, but this is a negligible parameter increase (~250 × 128 = 32K additional parameters if hidden_dim=128).

**The valid action mask is critical.** At any given state, only 10–30 of the ~250 actions are valid. The mask ensures the policy doesn't waste probability mass on invalid actions. The masking code is unchanged from the implementation plan — it just operates on a larger vocabulary:

```python
def compute_valid_mask(
    state: SynthesisState,
    grammar: Grammar,
    seed_constants: list[int],
    valid_instantiations: dict,
    action_vocab: dict[Action, int],
) -> torch.BoolTensor:
    """Compute a boolean mask over the action vocabulary for valid actions."""
    valid = valid_actions(state, grammar, seed_constants, valid_instantiations)
    mask = torch.zeros(len(action_vocab), dtype=torch.bool)
    for a in valid:
        if a in action_vocab:
            mask[action_vocab[a]] = True
    return mask
```

### 7.8 What the Policy Can Learn

With explicit instantiation actions, the policy has access to signals that would be invisible under inferred instantiation:

**Instantiation preference by context.** The policy can learn that when `parent_func = "map"` and `arg_index = 1` (the list argument), choosing `APPLY(group, {T1:int, T2:int})` produces a `list[list[int]]` that `map` can then process with a group-aggregation lambda. Under inferred instantiation, the policy would just see `APPLY(group)` and the instantiation would be determined by the target type — but the policy wouldn't have explicitly *chosen* to produce a `list[list[int]]` here.

**Type-level exploration.** With novelty-driven reward, the policy can discover that instantiations producing `list[list[int]]` or `list[bool]` lead to underexplored regions of fingerprint space, and deliberately steer toward those instantiations. Under inferred instantiation, this type-level steering would be indirect (the policy would have to choose functions whose only valid instantiation produces the desired type).

**Fold accumulator type choice.** When generating a `fold`, the policy can explicitly choose between `APPLY(fold, {T1:int, T2:int})` and `APPLY(fold, {T1:int, T2:list[int]})`. This is a meaningful semantic decision: the first produces arithmetic reductions, the second produces list-building reductions. Making this explicit lets the policy learn the value of each.

---

## 8. Performance Impact

### 8.1 Enumeration Cost

The main cost increase comes from iterating over multiple instantiations per function:

| Function category | Functions | Avg instantiations (before) | Avg instantiations (after) |
|---|---|---|---|
| Monomorphic (`+`, `-`, `sum`, `max`, ...) | ~15 | 1 | 1 |
| Unary polymorphic (`length`, `reverse`, `first`, ...) | ~25 | 1 | 3 |
| Binary polymorphic (`map`, `filter`, `fold`, ...) | ~15 | 1 | up to 9 |
| Other polymorphic (`cons`, `append`, ...) | ~15 | 1 | 3 |

**Total candidates generated per size level:** roughly 3× more than the fixed instantiation, because most of the new candidates are for the polymorphic functions with 3–9× more instantiations. However, the majority of new candidates will be pruned by observational equivalence (e.g., `reverse` at T1=bool on int inputs produces no programs because the types don't match). The number of *survivors* grows more modestly.

**Estimated overhead:** 2–3× more wall-clock time for enumeration, primarily from the additional candidate generation and evaluation. The nesting extension was already the expensive change (10×); this is relatively cheap on top of it.

### 8.2 RL Training Cost

The larger action space has minimal impact on training speed. The forward pass through the policy is dominated by the encoder, not the output head. The valid action mask means softmax is computed over the same ~10–30 valid actions regardless of total vocabulary size. The main cost is computing the valid action mask itself, which involves iterating over instantiations — cache this per-state if it becomes a bottleneck.

### 8.3 Bank Size

The bank now stores programs at more types. New type slots:

| Type | Typical programs at size 2–3 |
|------|------------------------------|
| `bool` | `(> (length x) 0)`, `(is_even (first x))`, ... |
| `list[bool]` | `(map (λ y (is_even y)) x)`, `(map (λ y (> y 0)) x)`, ... |
| `list[list[int]]` | `(group (λ y y) x)`, `(group (λ y (% y 2)) x)`, `(zip x x)`, ... |

These are all new intermediate types that feed into larger compositions. The `list[list[int]]` programs are particularly valuable as arguments to `flatten`, `map`-over-groups, and `filter`-over-groups.

### 8.4 Memory

Each new type adds entries to the bank. With 5 types instead of 3, and programs stored per-type-per-size, memory grows roughly proportionally. Monitor memory usage and add bank size limits if needed.

---

## 9. Testing Strategy

### 9.1 Validate New Instantiations

**Test 1: Instantiation computation is correct.**
For each grammar function, print its valid instantiations and manually verify:
- `+` should have exactly 1 (empty, monomorphic)
- `map` should have 9 (3 × 3 ground types for T1, T2)
- `fold` should have 9
- `sum` should have 1 (monomorphic)
- `group` should have at most 9 (but only a few produce return types in TYPE_UNIVERSE)

**Test 2: New program types appear in the bank.**
After enumeration at size 3, verify the bank contains programs of type:
- `list[list[int]]`: at minimum `(group (λ y y) x)` and `(zip x x)`
- `list[bool]`: at minimum `(map (λ y (is_even y)) x)` and `(map (λ y (> y 0)) x)`
- `bool`: several predicate expressions

**Test 3: Group-then-aggregate programs are enumerated (with nesting).**
At size 5+, verify the bank contains programs like:
```scheme
(map (λ g (length g)) (group (λ y y) x))
(map (λ g (sum g)) (group (λ y y) x))
(flatten (group (λ y (% y 2)) x))
```
These require both multi-type instantiation (T1=list[int] in outer map) and nesting (lambda body contains `length`/`sum`).

**Test 4: List-accumulating folds are enumerated.**
At size 5+, verify:
```scheme
(fold (λ acc elem (cons elem acc)) [] x)       ;; reverse
(fold (λ acc elem (append acc elem)) [] x)     ;; identity (copy)
```
These require T2=list[int] in fold.

### 9.2 Validate Action Vocabulary

**Test 5: Action vocab has expected size.**
Print the action vocabulary and verify:
- Monomorphic functions contribute 1 action each
- `map` contributes 9 actions (one per valid instantiation)
- `fold` contributes 9 actions
- Total is in the range 200–280

**Test 6: Valid action masks are correct.**
For several concrete states, enumerate valid actions and verify:
- State with `target_type = int`: should include `APPLY(fold, {T1:int, T2:int})` but NOT `APPLY(fold, {T1:int, T2:list[int]})` (since fold with T2=list[int] returns `list[int]`, not `int`)
- State with `target_type = list[int]`: should include `APPLY(fold, {T1:int, T2:list[int]})` and `APPLY(map, {T1:int, T2:int})` and `APPLY(reverse, {T1:int})`
- State with `target_type = list[list[int]]`: should include `APPLY(group, {T1:int, T2:*})` and `APPLY(zip, {T1:int})`

**Test 7: Trajectory extraction recovers correct instantiations.**
Take 10 programs from the enumeration corpus. Extract trajectories. For each `APPLY` action in the trajectory, verify the instantiation is consistent with the parent state's target type and the child states' argument types.

### 9.3 Validate Observational Equivalence

**Test 8: Cross-instantiation deduplication is correct.**
The type system prevents most spurious duplication — `reverse` at T1=bool expects `list[bool]` input and won't match `x : list[int]`. Verify this by checking that no two programs in the bank at the same type have identical fingerprints.

### 9.4 Performance Validation

**Test 9: Enumeration completes in reasonable time.**
With `max_size = 5, max_nesting = 1` (same as nesting extension, now with multi-type):
- Target: completes in < 3 hours (1.5× overhead over nesting-only)
- Measure: wall-clock, memory, candidates per size per type

**Test 10: Corpus diversity increases.**
Compare corpus sizes:
- Fixed instantiation with nesting: baseline
- Multi-type with nesting: expect 1.5–3× more distinct programs, primarily from group/fold patterns

**Test 11: RL policy learns instantiation preferences.**
After warm-start + 5000 RL iterations, sample 100 programs from the policy. Verify that the distribution of instantiations is non-uniform — the policy should prefer `{T1:int, T2:int}` and `{T1:int, T2:list[int]}` for `fold`, and rarely use `{T1:bool, T2:bool}`.

---

## 10. Summary of All File Changes

| File | Change type | Description |
|------|------------|-------------|
| `enumeration/enumerator.py` | **Moderate rewrite** | Replace fixed `_resolve_type` with instantiation-parameterised version. Add `_compute_valid_instantiations` at init. Change `_enumerate_at_size` to iterate over instantiations per function. Update base case to seed all types in TYPE_UNIVERSE. Update `extract_corpus` for multiple return types. |
| `enumeration/fingerprint.py` | **Minor update** | Add `list[int]` and `list[bool]` entries to `PROBE_VALUES`. |
| `enumeration/filters.py` | No change | |
| `enumeration/test_suite.py` | No change | |
| `rl/mdp.py` | **Moderate update** | Update `Action` to carry `instantiation` field. Update `valid_actions` to emit one action per (function, instantiation) pair. Add `freeze_instantiation` / `thaw_instantiation` helpers. |
| `rl/policy.py` | **Minor update** | Rebuild type and action vocabularies with expanded sets. Output head grows to ~250 logits. No architectural change. |
| `rl/trajectory.py` | **Moderate update** | Add `_infer_instantiation_from_types` for recovering instantiations during trajectory extraction. Record instantiation in `APPLY` actions. |
| `rl/trainer.py` | **Minor update** | Pass `valid_instantiations` to valid action computation. No training loop changes. |
| `rl/reward.py` | No change | |
| `rl/priority_queue.py` | No change | |
| `pipeline.py` | **Minor update** | Compute `valid_instantiations` and pass to all components. |
| `utils.py` | **Minor update** | Add `TYPE_UNIVERSE`, `GROUND_TYPES` constants. Add `freeze_instantiation`, `thaw_instantiation`. |

---

## 11. Constants Reference

Collect all new constants in `utils.py` for single-source-of-truth:

```python
from typing import TypeVar

T1, T2 = TypeVar('T1'), TypeVar('T2')

# The closed set of concrete types the enumerator operates over
TYPE_UNIVERSE = frozenset({
    int,
    bool,
    list[int],
    list[bool],
    list[list[int]],
})

# Ground types that type variables can be instantiated to
GROUND_TYPES = [int, bool, list[int]]

# Return types of interest for the final corpus
INTERESTING_RETURN_TYPES = frozenset({
    list[int],
    int,
    bool,
    list[list[int]],
    list[bool],
})

# Probe values for fingerprinting lambda-bound variables
PROBE_VALUES = {
    int: [0, 1, 3],
    bool: [True, False],
    list[int]: [[], [1], [2, 1, 3]],
    list[bool]: [[], [True], [True, False]],
}
```

Import `T1` and `T2` from `grammar.py` (where they're already defined) rather than redefining them. The constants above should be the canonical source; update any hardcoded references in the enumerator and fingerprinting code to use these.

---

*Depends on: `IMPLEMENTATION_PLAN.md`, `NESTING_EXTENSION.md`*
*Updates: Sections 3, 4, 6, 7 of the implementation plan*
