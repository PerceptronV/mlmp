# Program Generation via Reinforcement Learning, Bootstrapped by Enumeration

## Problem Statement

We are interested in the following problem :

> Suppose we have access to:
>
> - A set $F$ of functions $f_1, f_2, \ldots, f_n$ with types $\tau_1 \to \upsilon_1, \tau_2 \to \upsilon_2, \ldots, \tau_n \to \upsilon_n$, where we can query the function implementation (i.e. $f_i(x)$ can be computed for arbitrary well-typed inputs $x$) but not necessarily the function description (e.g. we do not know the mathematical or programmatic form of the functions).
> - A reward function $R: \mathrm{program} \to \mathbb{R}$ that maps a program to a real-valued reward, where a program is a well-typed composition of functions from $F$ (e.g. this can reward a function for desirable properties such as being non-constant).
> - How do we synthesise a large number of programs that are diverse and achieve high rewards?

This document describes the two-phase program synthesis pipeline: bottom-up enumeration followed by reinforcement learning. Both phases share the same grammar, type system, and fingerprinting infrastructure.

---

## Overview

The pipeline has two sequential phases:

1. **Bottom-up enumeration** — exhaustively generates all observationally distinct programs up to a size bound, pruning semantic duplicates via fingerprinting. The result is a *corpus* of behaviorally diverse programs.
2. **RL exploration** — a neural policy learns from the corpus (behavioral cloning warm-start), then continues searching beyond the enumeration horizon via reward-weighted policy gradient, using a priority queue buffer of the best programs found so far.

---

## Part 1: Bottom-Up Enumeration

### Size Metric

Every AST node is assigned an integer *size*:


| Node                          | Size                            |
| ----------------------------- | ------------------------------- |
| Literal, variable, empty list | 1                               |
| Application `(f a1 ... ak)`   | 1 + size(a1) + ... + size(ak)   |
| Lambda `(λ (p1 ... pk) body)` | 1 + size(body)                  |
| If `(if c t e)`               | 1 + size(c) + size(t) + size(e) |


Enumeration proceeds in increasing order of size. At each level, the enumerator only needs programs from strictly smaller levels, giving a clean bottom-up stratification.

### Program Bank

The `ProgramBank` stores programs indexed by `(type, size)`:

```
bank[resolved_type][size] → list[TypedProgram]
```

Each `TypedProgram` carries its AST, resolved type, fingerprint, and size. The bank enforces **observational equivalence**: a new program is only added if its fingerprint is novel for its type. Two programs with identical behavior on the test suite are considered the same; only the first encountered is kept.

### Fingerprinting

A fingerprint is the tuple of outputs produced by evaluating a program on every input in the *test suite* (10 fixed `list[int]` inputs). Open terms (e.g., a sub-expression referencing `x`) are closed by wrapping them in `(λ x <term>)` before evaluation.

Terms inside lambda bodies that reference additional parameters (e.g., `y` inside `(map (λ (y) ...) x)`) are closed more carefully: the system wraps the body in nested lambdas for all free parameters and evaluates across the Cartesian product of the test suite and a set of *probe values* for each parameter type. This ensures that sub-expressions inside HOF arguments are still deduplicated correctly.

Failed evaluations (exceptions, type errors) are recorded as a special `FAIL` sentinel. A program's fingerprint may contain a mix of successful outputs and `FAIL` entries.

**Default test suite** (10 inputs):

```python
[],                        # empty
[0],                       # singleton zero
[3, 1, 2],                 # small unsorted
[1, 1, 1, 1],              # duplicates
[5, 4, 3, 2, 1],           # reverse-sorted
[1, 2, 3, 4, 5, 6, 7, 8],  # longer sorted
[10, -3, 7, 7, 0],         # negatives
[2, 8, 3, 8, 2, 3],        # multiple repeats
[0, 1, 0, 1, 0],           # binary pattern
[42],                      # singleton nonzero
```

### Base Case (Size 1)

The enumerator seeds the bank with:

- **Integer constants** — one `NumberNode(c)` per value in `seed_constants` (default `[0, 1, 2, 3]`)
- **Boolean constants** — `True`, `False`
- **Empty lists** — `[]` with type `list[int]`, `list[bool]`, `list[list[int]]`
- **Input variable** — `x : list[int]` with fingerprint equal to the identity (each test input maps to itself)

### First-Order Applications

For each grammar function `f` with arity `k`, return type `R`, and argument types `A1 ... Ak`, the enumerator tries to build `(f a1 ... ak)` at total size `s`. The cost of the function node itself is 1, so the arguments must partition the remaining budget `s - 1` into `k` positive parts.

For each such partition `(s1, ..., sk)`:

1. Look up all programs of type `Ai` at size `si` in the bank.
2. Take the Cartesian product across all argument positions.
3. For each combination, build the `ApplicationNode`, compute its fingerprint, and add to the bank if novel.

Polymorphic functions (e.g., `map : (T1→T2) → list[T1] → list[T2]`) are handled by pre-computing a table of valid type instantiations (`T1=int, T2=int`; `T1=int, T2=bool`; etc.). Each instantiation is enumerated separately, with argument types resolved before lookup.

### Higher-Order Applications and Nested Lambdas

When one or more argument types are `Callable[..., ...]` (detected by checking `get_origin(t) == CallableOrig`), the enumerator calls `_enumerate_lambda_arg` to synthesize the lambda argument inline rather than looking it up in the flat bank.

`_enumerate_lambda_arg` does the following:

1. Extracts the parameter types `[P1, ..., Pk]` and body type `B` from the `Callable` type.
2. Generates fresh parameter names `_p0, _p1, ...` that don't clash with the current context.
3. Creates an extended context `{x: list[int], _p0: P1, ..., _pk: Pk}`.
4. Builds a **child bank** (`ContextualBank`) by recursively enumerating programs in the extended context up to the available size budget minus 1 (for the lambda node itself).
5. Collects all programs of type `B` from the child bank and wraps each as `LambdaNode([_p0, ...], body)`.

**ContextualBank** mirrors lexical scoping: its `get()` method returns programs from the local level plus all ancestor banks. Deduplication uses a *local* fingerprint table (not the parent's), because programs in a child context are evaluated on a larger probe space (test suite × parameter probes) and their fingerprint lengths differ from the parent's.

Child banks are **cached** by `(parent_bank_id, frozenset(context.items()), body_budget, nesting_depth)`. This avoids rebuilding the same child bank when the same lambda type appears as an argument to multiple functions at the same size.

**Nesting depth** tracks how many lambda scopes deep the enumeration currently is. Higher-order function calls within a child context are only attempted if `nesting_depth <= max_nesting` (default `max_nesting=1`). This prevents combinatorial explosion from deeply stacked lambda contexts while still allowing the common pattern of one level of nested lambda (e.g., `map (λ (y) ...) x`).

### Quality Filtering

After enumeration, `extract_corpus` filters the bank down to programs that pass a quality gate:

- **Non-crashing**: at least 3 test inputs produce a non-`FAIL` output
- **Non-constant**: at least 2 distinct output values across the test suite
- **Variability**: `unique_outputs / total_outputs >= min_variability` (default `0.3`)

Only programs passing all three filters are included in the corpus used for RL warm-start.

### Key Parameters


| Parameter         | Default     | Effect                                         |
| ----------------- | ----------- | ---------------------------------------------- |
| `max_size`        | 5           | Maximum AST size enumerated                    |
| `max_nesting`     | 1           | Maximum lambda nesting depth for HOF arguments |
| `seed_constants`  | `[0,1,2,3]` | Integer literals available as base-case atoms  |
| `min_variability` | 0.3         | Corpus quality threshold                       |
| `test_suite`      | 10 inputs   | Used for fingerprinting and deduplication      |


---

## Part 2: Reinforcement Learning

The RL phase uses a top-down generative model: a neural policy builds programs recursively from the root, making one decision per AST node. This is framed as an MDP.

### MDP Formulation

**State** (`SynthesisState`):


| Field           | Type                   | Description                                              |
| --------------- | ---------------------- | -------------------------------------------------------- |
| `target_type`   | `TypeType`             | The type the current node must produce                   |
| `context`       | `dict[str, TypeType]`  | Variables in scope (e.g., `{x: list[int], _p0: int}`)    |
| `parent_func`   | `str                   | None`                                                    |
| `arg_index`     | `int                   | None`                                                    |
| `siblings`      | `list[(ASTNode, Any)]` | Already-generated earlier sibling arguments              |
| `depth_budget`  | `int`                  | Remaining AST depth (decremented at each recursive call) |
| `nesting_depth` | `int`                  | Current lambda nesting level                             |


**Actions** (`ActionType`):


| Action               | Payload                       | Condition                                                                                                                                    |
| -------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `LITERAL_INT`        | integer value                 | `target_type == int`                                                                                                                         |
| `LITERAL_BOOL`       | `True` / `False`              | `target_type == bool`                                                                                                                        |
| `LITERAL_EMPTY_LIST` | —                             | `target_type` is a list type                                                                                                                 |
| `VARIABLE`           | variable name                 | variable of matching type exists in context                                                                                                  |
| `APPLY`              | function name + instantiation | `depth_budget > 0`; function return type matches `target_type`; HOF functions additionally require `nesting_depth < MAX_NESTING_DEPTH` (= 2) |
| `LAMBDA`             | —                             | `target_type` is a `Callable` type                                                                                                           |
| `IF`                 | —                             | `depth_budget > 0`                                                                                                                           |


**Episode** (`Episode.run()`):

An episode starts with:

```
SynthesisState(
    target_type = Callable[[list[int]], list[int]],
    context = {},
    depth_budget = max_depth,   # default 6
)
```

The policy selects an action; the episode then recurses into child states for each sub-expression that needs to be generated:

- `APPLY f` → one child state per argument of `f`, each with `depth_budget - 1` and the resolved argument type
- `LAMBDA` → one child state for the body, with `depth_budget - 1`, `nesting_depth + 1`, and context extended with the lambda parameters
- `IF` → three child states (condition as `bool`, then-branch and else-branch both at `target_type`), each with `depth_budget - 1`

If any recursive call returns `None` (no valid actions), the episode fails and returns `None`.

### Reward

After a completed episode, the generated AST is wrapped as `(λ x <body>)` if not already a lambda, and evaluated on the test suite to produce a fingerprint. The reward is:

```
reward = variability(fp) × novelty_bonus
```

where:

- `variability(fp)` = fraction of test suite positions with distinct output values
- `novelty_bonus` = `1.0` if the fingerprint is not in `corpus_fingerprints`, else `0.1`

Programs that crash on fewer than 3 inputs, or that produce only a single distinct output, receive reward `0.0` and are discarded.

The novelty bonus means the policy is incentivized to explore behaviors not already covered by the enumeration corpus or earlier RL discoveries. Programs that duplicate known behaviors still receive a small reward (`0.1 × variability`) so the policy is not penalized for rediscovering them.

### Priority Queue Buffer

The RL buffer stores the top-K programs by reward (default capacity 5000). Internally it is a **bounded min-heap**: when full, a new program only enters if its reward exceeds the current minimum, which is then evicted. Fingerprints of all buffered programs are tracked to prevent duplicates.

The buffer is sampled uniformly at random during training. There is no recency or priority weighting on samples — the buffer simply maintains the highest-reward programs seen so far.

### Policy Network

The policy network maps a `SynthesisState` to a distribution over the action vocabulary.

**Architecture**:

```
StateEncoder → Linear(embed_dim, hidden_dim) → ReLU → Linear(hidden_dim, |actions|)
```

**StateEncoder** encodes 6 features, each via a learned embedding or projection, then combines them:


| Feature           | Encoding                 | Range                                 |
| ----------------- | ------------------------ | ------------------------------------- |
| `target_type`     | Embedding(               | type_vocab                            |
| `parent_func`     | Embedding(               | func_vocab                            |
| `arg_index`       | Embedding(8, embed_dim)  | 0–7, clamped                          |
| `depth_budget`    | Embedding(16, embed_dim) | 0–15, clamped                         |
| `nesting_depth`   | Embedding(4, embed_dim)  | 0–3, clamped                          |
| context variables | Linear(16, embed_dim)    | count of variables per type (16 bins) |


The six vectors are concatenated and passed through `Linear(6×embed_dim, embed_dim)` + ReLU. Default `embed_dim=64`, `hidden_dim=128`.

**Action masking**: before softmax, logits for invalid actions are set to `-inf`. Validity is determined by `valid_actions(state, grammar, ...)` as described above, and is cached by `(target_type, frozenset(context), depth_budget > 0, nesting_depth)` for efficiency.

**Inference**: during episode rollout, the policy samples from the masked softmax distribution using `torch.multinomial`.

### Training Phase 1: Behavioral Cloning Warm-Start

Before RL exploration begins, the policy is pre-trained on the enumeration corpus via imitation learning.

For each program `p` in the corpus, `extract_trajectory` performs a DFS over `p`'s AST. At each node it records the `SynthesisState` that would have been active at that point in a top-down generation, and the `Action` corresponding to the node. This produces a list of `(state, action)` pairs.

The full set of transitions across all corpus programs is assembled into a `TransitionDataset`. The policy is trained with negative log-likelihood (cross-entropy) loss:

```
loss = -E[ log π(action | state) ]
```

over `epochs` passes (default 50), batch size 64, Adam optimizer at lr=1e-3.

Transitions whose action is not in the vocabulary (e.g., from programs using constants not in `seed_constants`) are silently dropped.

### Training Phase 2: RL Loop

The RL training loop alternates between a sampling phase and a training phase.

**Sampling phase** (32 episodes per iteration by default):

1. Run an episode under the current policy.
2. Compute fingerprint and reward.
3. If reward > 0, attempt to insert into the priority queue buffer.
4. If the fingerprint is novel to the corpus, add it to `corpus_fingerprints` so future episodes that rediscover it receive only the 0.1× novelty bonus.

**Training phase** (8 gradient steps per iteration by default):

1. Sample a batch of `(reward, program, trajectory)` tuples from the buffer.
2. Flatten all `(state, action, reward)` triples across the batch.
3. Compute the policy gradient loss:

```
loss = -mean( reward_weight × log π(action | state) )
```

This is reward-weighted maximum likelihood: each transition is up-weighted by the reward of its parent program. No baseline subtraction is applied.

1. Apply gradient clipping at norm 1.0 before the optimizer step (Adam at lr=1e-4).

### Trajectory Extraction

`extract_trajectory` is the inverse of `Episode.run`. Given a complete AST, it reconstructs the sequence of `(SynthesisState, Action)` pairs that a top-down policy would have produced:

- `LambdaNode` → `Action(LAMBDA)` followed by a recursive walk of the body at `nesting_depth + 1`
- `ApplicationNode(VariableNode(f), args)` → `Action(APPLY, f, inst)` followed by recursive walks of each argument with the appropriate type and `parent_func`/`arg_index` set
- `IfNode` → `Action(IF)` followed by condition, then-branch, else-branch
- Leaf nodes → `LITERAL_INT`, `LITERAL_BOOL`, `LITERAL_EMPTY_LIST`, or `VARIABLE`

For polymorphic functions, the instantiation is inferred by finding which entry in `valid_instantiations[f]` produces a return type matching the current `state.target_type`.

---

## End-to-End Pipeline

```
seed_constants, test_suite, grammar
        │
        ▼
BottomUpEnumerator.enumerate()
  ├── base case (size 1 atoms)
  └── sizes 2 → max_size
       ├── first-order: lookup in bank
       └── higher-order: enumerate lambda args via ContextualBank
        │
        ▼ (fingerprinting + quality filter)
corpus: list[TypedProgram]
        │
        ├─── warm_start(policy, corpus)
        │      └── extract_trajectory × |corpus|
        │          → supervised training (NLL loss)
        │
        └─── train_rl(policy, buffer, corpus_fingerprints)
               └── for each iteration:
                    ├── episodes: sample programs → compute reward → insert buffer
                    └── training: sample buffer → reward-weighted MLL loss
```

---

## Limitations

**Partial application / currying** — The grammar treats all functions as taking a fixed number of arguments. `(== y)` as a `Callable[[int], bool]` value is not generatable; a lambda wrapper `(λ (z) (== z y))` is needed instead. This affects a small number of programs in the target corpus that use `(== val)` as a predicate directly.

**Lambda in function position** — Neither the enumerator nor the RL MDP can generate `((λ (y) ...) arg)` (an anonymous lambda called immediately). The APPLY action always names a grammar function; there is no action for applying an arbitrary expression as a function. This pattern is effectively a let-binding workaround and is uncommon in the target corpus.

`**seed_constants` coverage** — Integer literals are only available at the values listed in `seed_constants`. Programs using constants outside this list (e.g., 5, 6, 7, 8, 9, or large numbers like 18, 42, 77) cannot be generated without explicitly including those values. The default `[0, 1, 2, 3]` is insufficient for most non-trivial programs.

**Size and depth limits** — Enumeration stops at `max_size=5` by default. The RL's `max_depth=6` limits program depth. Complex programs (e.g., nested folds with multi-expression bodies) may exceed both limits. These are configuration parameters, not architectural constraints.

**Nesting depth cap** — `MAX_NESTING_DEPTH=2` in the MDP and `max_nesting=1` in the enumerator prevent HOF arguments from themselves containing HOF calls. This is sufficient for single-level nesting (`map (λ (y) ...) x`) but would block doubly-nested HOFs like `map (λ (y) (filter ...)) xs`.