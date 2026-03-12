# Implementation Plan: Bottom-Up Enumeration + RL for the Rule DSL

## Instructions for Claude Code Agent

This document specifies a complete implementation plan for synthesising semantically diverse programs over the Rule DSL. The system has two components: (1) bottom-up enumeration with observational equivalence pruning, and (2) RL-based generation with priority queue training. Read this document in full before writing any code.

---

## 0. Codebase Orientation

The existing codebase is a Python package (referred to as `mlmp` in `__init__.py`) implementing a typed functional DSL with ~70 primitives. The key modules are:

| Module | Purpose | Key classes/functions you will use |
|--------|---------|-------------------------------------|
| `grammar.py` | Function registry with types | `Grammar`, `DefaultGrammar`, `Grammar.find_matching_functions()`, `Grammar.functions`, `Grammar.names` |
| `ast_nodes.py` | AST node definitions | `NumberNode`, `BooleanNode`, `VariableNode`, `LambdaNode`, `ApplicationNode`, `ListNode`, `IfNode` |
| `type_utils.py` | Type manipulation and unification | `SubstitutionTable`, `matchable()`, `substitute_type_vars()`, `get_base_type()`, `get_args()`, `isvariable()`, `CallableOrig` |
| `evaluator.py` | Tree-walking interpreter | `Evaluator`, `Evaluator.eval()`, `EvaluationError` |
| `compiler.py` | JIT compiler (AST → Python bytecode) | `JITCompiler`, `JITCompiler.compile()` |
| `type_checker.py` | Hindley-Milner type inference | `TypeChecker`, `TypeChecker.check()` |
| `environment.py` | Variable scoping, closures | `Environment`, `Closure` |
| `parser.py` | S-expression parser | `parse()` |
| `lexer.py` | Tokeniser | `tokenise()` |

### Critical implementation details to understand before coding:

**Grammar function access.** Each function in `DefaultGrammar.functions` is a dict with keys: `'fn'` (raw Python callable), `'__call__'` (evaluator-wrapped callable), `'arg_names'` (tuple of strings), `'arg_types'` (tuple of types), `'ret_type'` (type). Type variables are `TypeVar('T1')` and `TypeVar('T2')`.

**Type system.** Types are Python typing constructs: `int`, `bool`, `list[T1]`, `Callable[[T1], T2]`, etc. The `CallableOrig` constant is `get_origin(Callable)`. Use `matchable(type1, type2, substitutions)` to check if two types can unify, which mutates the `SubstitutionTable`. Always `.copy()` the substitution table before speculative matching.

**AST construction.** To build `(map (λ y (* y 2)) x)` programmatically:
```python
ApplicationNode(
    VariableNode("map"),
    [
        LambdaNode(["y"], ApplicationNode(VariableNode("*"), [VariableNode("y"), NumberNode(2)])),
        VariableNode("x")
    ]
)
```

**Evaluation.** The `Evaluator` walks the AST. For bulk evaluation, prefer the `JITCompiler`: compile once with `jit.compile(ast)`, then call the returned function repeatedly. This is critical for performance since each program will be evaluated on 10+ test inputs.

**Higher-order function types.** `map` has type `Callable[[T1], T2] → list[T1] → list[T2]`. In the grammar dict, `arg_types` is `(Callable[[T1], T2], list[T1])` and `ret_type` is `list[T2]`. The first arg type is a `Callable` origin — detect this with `get_origin(arg_type) == CallableOrig`.

---

## 1. File Structure

Create the following files. All new files go in the same package directory as the existing modules.

```
mlmp/
├── (existing files: grammar.py, ast_nodes.py, evaluator.py, compiler.py, ...)
├── enumeration/
│   ├── __init__.py
│   ├── enumerator.py          # Core bottom-up enumerator
│   ├── fingerprint.py         # Observational equivalence and fingerprinting
│   ├── test_suite.py          # Test input generation and management
│   └── filters.py             # Quality filters (non-constant, variability, etc.)
├── rl/
│   ├── __init__.py
│   ├── mdp.py                 # MDP state/action definitions
│   ├── policy.py              # Neural policy network
│   ├── priority_queue.py      # Priority queue buffer
│   ├── trainer.py             # Training loop (priority queue training)
│   ├── reward.py              # Reward functions (variability, novelty)
│   └── trajectory.py          # Trajectory extraction from programs
├── pipeline.py                # Top-level orchestrator (enum → warm-start → RL)
└── utils.py                   # Shared utilities (program size, AST helpers)
```

---

## 2. Shared Utilities (`utils.py`)

### 2.1 Program Size

Implement `program_size(node: ASTNode) -> int`:

```
size(NumberNode(v))           = 1
size(BooleanNode(v))          = 1
size(VariableNode(x))         = 1
size(ListNode([]))             = 1
size(ListNode([e1, ..., en]))  = 1 + sum(size(ei))
size(LambdaNode(params, body)) = 1 + size(body)
size(IfNode(c, t, e))         = 1 + size(c) + size(t) + size(e)
size(ApplicationNode(f, [a1, ..., an])) = 1 + size(f) + sum(size(ai))
```

Note: `size(ApplicationNode(VariableNode("map"), [lambda, x]))` = 1 + 1 + size(lambda) + size(x). The function name node `VariableNode("map")` contributes size 1, but since the `ApplicationNode` itself already costs 1, the total for a function application `(f e1 ... en)` is `1 + sum(size(ei))` where the function name is NOT double-counted. Define this carefully. Recommended: for `ApplicationNode`, the cost is `1 + sum(size(arg) for arg in arguments)` and the `function` field (which is a `VariableNode` for primitive calls) is subsumed into the `1`. This means:
- `(length x)` = 1 + 1 = 2
- `(+ x 1)` = 1 + 1 + 1 = 3
- `(map (λ y (* y 2)) x)` = 1 + (1 + (1 + 1 + 1)) + 1 = 6

### 2.2 AST Helpers

- `free_variables(node: ASTNode, bound: set[str]) -> set[str]`: Return the set of free variable names in `node` given the set of already-bound names. This already exists as `_find_free_variables` in `compiler.py`; extract and generalise it.
- `uses_variable(node: ASTNode, var_name: str) -> bool`: Check if a variable name appears free in the AST.

---

## 3. Test Suite (`enumeration/test_suite.py`)

### 3.1 Default Test Suite

Define the test suite as a module-level constant:

```python
DEFAULT_TEST_SUITE: list[list[int]] = [
    [],                         # i0: empty list
    [0],                        # i1: singleton zero
    [3, 1, 2],                  # i2: small unsorted
    [1, 1, 1, 1],              # i3: all duplicates
    [5, 4, 3, 2, 1],           # i4: reverse-sorted
    [1, 2, 3, 4, 5, 6, 7, 8], # i5: longer sorted
    [10, -3, 7, 7, 0],         # i6: negatives, duplicates, zero
    [2, 8, 3, 8, 2, 3],       # i7: multiple repeated values
    [0, 1, 0, 1, 0],          # i8: binary pattern
    [42],                       # i9: singleton nonzero
]
```

Store this as a list of lists. Programs of type `[t1] → [t2]` will be evaluated by applying them to each test input. Programs of type `[Int] → Int` similarly.

### 3.2 Evaluation Function

Implement `evaluate_program(ast: ASTNode, test_suite: list, timeout_ms: int = 100) -> list[Any | None]`:

1. JIT-compile the AST using `JITCompiler.compile(ast)`. If compilation fails, return `[None] * len(test_suite)`.
2. For each test input `inp`, call the compiled function inside a try/except block. Catch `Exception` (including `ValueError`, `ZeroDivisionError`, `RecursionError`, `ListSizeExceeded`, etc.). On exception or timeout, record `None` for that input. On success, record the return value.
3. Return the list of results (values or `None`).

**Timeout handling.** Python does not have trivial per-call timeouts. For v1, use a simple recursion/size guard: if any intermediate list exceeds `MAX_LIST_SIZE` (already defined in `grammar.py` as 1000), the grammar functions raise `ListSizeExceeded`. This is sufficient. If needed later, use `signal.alarm` on Unix or multiprocessing with a timeout.

**Important: JIT vs Evaluator.** Use the JIT compiler (`JITCompiler`) rather than the tree-walking `Evaluator` for fingerprinting. The JIT compiles the AST to native Python once and then executes it as a plain function call per input. This is critical because the enumerator will evaluate potentially millions of candidate programs.

---

## 4. Fingerprinting (`enumeration/fingerprint.py`)

### 4.1 Fingerprint Representation

A fingerprint is a tuple of results, where each result is either a hashable value or a sentinel `FAIL`:

```python
FAIL = object()  # Sentinel for failed evaluations

class Fingerprint:
    """Immutable fingerprint for a program's behaviour on the test suite."""
    
    __slots__ = ('values', '_hash')
    
    def __init__(self, values: tuple):
        # values is a tuple where each entry is a hashable value or FAIL
        self.values = values
        self._hash = hash(values)
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return isinstance(other, Fingerprint) and self.values == other.values
```

**Making values hashable.** Evaluation results may be lists (which are unhashable). Convert lists to tuples recursively before storing in the fingerprint. Booleans and ints are already hashable. Nested lists `[[1,2],[3]]` become `((1,2),(3,))`.

```python
def make_hashable(value):
    if value is None or value is FAIL:
        return FAIL
    if isinstance(value, list):
        return tuple(make_hashable(v) for v in value)
    return value
```

### 4.2 Fingerprint Table

```python
class FingerprintTable:
    """Hash table mapping fingerprints to canonical program representatives."""
    
    def __init__(self):
        self.table: dict[Fingerprint, ASTNode] = {}
    
    def contains(self, fp: Fingerprint) -> bool:
        return fp in self.table
    
    def insert(self, fp: Fingerprint, program: ASTNode) -> bool:
        """Insert if novel. Returns True if inserted, False if duplicate."""
        if fp in self.table:
            return False
        self.table[fp] = program
        return True
    
    def __len__(self):
        return len(self.table)
    
    def programs(self) -> list[ASTNode]:
        return list(self.table.values())
    
    def items(self) -> list[tuple[Fingerprint, ASTNode]]:
        return list(self.table.items())
```

---

## 5. Quality Filters (`enumeration/filters.py`)

Implement the following predicates on `Fingerprint` objects:

### 5.1 `is_non_crashing(fp: Fingerprint, min_successes: int = 3) -> bool`

Return `True` if at least `min_successes` entries in `fp.values` are not `FAIL`.

### 5.2 `is_non_constant(fp: Fingerprint) -> bool`

Return `True` if there are at least 2 distinct non-`FAIL` values in `fp.values`.

### 5.3 `variability(fp: Fingerprint) -> float`

Compute the fraction of unique non-`FAIL` values among all non-`FAIL` values:

```python
non_fail = [v for v in fp.values if v is not FAIL]
if len(non_fail) <= 1:
    return 0.0
return len(set(non_fail)) / len(non_fail)
```

### 5.4 `passes_quality_filter(fp: Fingerprint, min_successes: int = 3, min_variability: float = 0.3) -> bool`

Conjunction of:
1. `is_non_crashing(fp, min_successes)`
2. `is_non_constant(fp)`
3. `variability(fp) >= min_variability`

---

## 6. Bottom-Up Enumerator (`enumeration/enumerator.py`)

This is the core of Component 1. It is the most complex module and requires careful implementation.

### 6.1 Data Structures

```python
@dataclass
class TypedProgram:
    """A program with its type and fingerprint."""
    ast: ASTNode
    type: TypeType           # The resolved type of this program
    fingerprint: Fingerprint
    size: int

class ProgramBank:
    """
    Stores semantically distinct programs indexed by type and size.
    
    Access pattern: bank[resolved_type][size] -> list[TypedProgram]
    """
    def __init__(self):
        self._bank: dict[TypeType, dict[int, list[TypedProgram]]] = defaultdict(lambda: defaultdict(list))
        self._fingerprint_table: dict[TypeType, FingerprintTable] = defaultdict(FingerprintTable)
    
    def add(self, prog: TypedProgram) -> bool:
        """Add program if its fingerprint is novel for its type. Returns True if added."""
        fp_table = self._fingerprint_table[prog.type]
        if fp_table.insert(prog.fingerprint, prog.ast):
            self._bank[prog.type][prog.size].append(prog)
            return True
        return False
    
    def get(self, type_: TypeType, size: int) -> list[TypedProgram]:
        """Get all programs of a given type and exact size."""
        return self._bank.get(type_, {}).get(size, [])
    
    def get_up_to(self, type_: TypeType, max_size: int) -> list[TypedProgram]:
        """Get all programs of a given type up to a given size."""
        result = []
        for s in range(1, max_size + 1):
            result.extend(self.get(type_, s))
        return result
    
    def count(self) -> int:
        """Total number of stored programs."""
        return sum(
            len(progs)
            for by_size in self._bank.values()
            for progs in by_size.values()
        )
```

**Critical: type normalisation.** The bank is indexed by *resolved* types — i.e., types with all type variables substituted away. When you enumerate programs at type `list[T1]` instantiated to `list[int]`, the resolved type is `list[int]`. Two programs are in the same observational equivalence pool only if they have the same resolved type.

However, the enumerator must also handle polymorphic programs that work at multiple type instantiations. For the initial implementation, **fix the type instantiation to `T1 = int, T2 = int`** (since the test suite consists of `list[int]` inputs). This dramatically simplifies the type machinery: every `list[T1]` becomes `list[int]`, every `Callable[[T1], T2]` becomes `Callable[[int], int]`, etc. Document this as a simplification that can be generalised later.

### 6.2 Enumeration Algorithm

```python
class BottomUpEnumerator:
    def __init__(
        self,
        grammar: Grammar,
        test_suite: list[list[int]],
        seed_constants: list[int] = [0, 1, 2, 3],
        max_size: int = 5,
        min_variability: float = 0.3,
        input_var_name: str = "x",
        input_type: TypeType = list[int],  # Fixed instantiation
    ):
        self.grammar = grammar
        self.test_suite = test_suite
        self.seed_constants = seed_constants
        self.max_size = max_size
        self.min_variability = min_variability
        self.input_var_name = input_var_name
        self.input_type = input_type
        
        self.bank = ProgramBank()
        self.jit = JITCompiler(grammar)
    
    def enumerate(self) -> ProgramBank:
        """Run bottom-up enumeration and return the populated program bank."""
        self._enumerate_base_case()
        for size in range(2, self.max_size + 1):
            self._enumerate_at_size(size)
            # Log progress
            print(f"Size {size}: {self.bank.count()} total programs")
        return self.bank
```

### 6.3 Base Case (size 1)

```python
def _enumerate_base_case(self):
    """Populate the bank with all size-1 atoms."""
    
    # Integer constants
    for c in self.seed_constants:
        node = NumberNode(c)
        self._try_add(node, int, size=1)
    
    # Boolean constants
    for b in [True, False]:
        node = BooleanNode(b)
        self._try_add(node, bool, size=1)
    
    # Empty list
    node = ListNode([])
    self._try_add(node, list[int], size=1)
    
    # Input variable — this is NOT a closed term, so we cannot fingerprint
    # it in isolation. Instead, we store it with a special fingerprint
    # that is just the identity mapping on the test suite.
    # The input variable has type list[int] (our fixed instantiation).
    # Its fingerprint is computed by treating it as the identity program.
    var_node = VariableNode(self.input_var_name)
    fp = self._compute_var_fingerprint()
    prog = TypedProgram(ast=var_node, type=self.input_type, fingerprint=fp, size=1)
    self.bank.add(prog)
```

**Critical subtlety: open vs closed terms.** The bank stores *open* terms — terms that may reference the input variable `x`. They are not directly evaluable. To compute a fingerprint, you must wrap the term in a lambda `(λ x <term>)` and then evaluate it on each test input.

```python
def _fingerprint(self, node: ASTNode, node_type: TypeType) -> Fingerprint | None:
    """
    Compute the fingerprint of a (possibly open) term.
    
    Wraps the term in (λ x <term>) and evaluates on the test suite.
    Returns None if compilation fails.
    """
    # Wrap in lambda to close over the input variable
    closed = LambdaNode([self.input_var_name], node)
    
    try:
        compiled = self.jit.compile(closed)
    except Exception:
        return None
    
    values = []
    for inp in self.test_suite:
        try:
            result = compiled(inp)
            values.append(make_hashable(result))
        except Exception:
            values.append(FAIL)
    
    return Fingerprint(tuple(values))

def _try_add(self, node: ASTNode, resolved_type: TypeType, size: int) -> bool:
    """Compute fingerprint and add to bank if novel and passes filters."""
    fp = self._fingerprint(node, resolved_type)
    if fp is None:
        return False
    prog = TypedProgram(ast=node, type=resolved_type, fingerprint=fp, size=size)
    return self.bank.add(prog)
```

**Important: fingerprinting terms that don't use x.** A term like `NumberNode(3)` wrapped in `(λ x 3)` will return 3 for every test input. Its fingerprint is `(3, 3, 3, 3, ...)`. This is correct — it represents a constant function. The fingerprint table will collapse all constant-3 programs to one representative. The quality filters will then exclude it (non-constant check fails).

BUT: we still want constant subexpressions in the bank because they serve as arguments to larger programs. The quality filters should be applied only to *top-level* programs that are candidates for the final corpus, not to intermediate subexpressions stored in the bank. So:

- **The bank stores ALL semantically distinct programs (including constants).**
- **The quality filters are applied at the end, when extracting the final corpus.**

### 6.4 Inductive Case (size k > 1)

This is the core loop. For each grammar function, enumerate all well-typed argument combinations that sum to the right size.

```python
def _enumerate_at_size(self, size: int):
    """Enumerate all programs of exactly the given size."""
    
    for func_name in self.grammar.names:
        func_info = self.grammar[func_name]
        arg_types = func_info['arg_types']
        ret_type = func_info['ret_type']
        arity = len(arg_types)
        
        # Resolve type variables to our fixed instantiation
        resolved_arg_types = self._resolve_types(arg_types)
        resolved_ret_type = self._resolve_type(ret_type)
        
        if resolved_ret_type is None:
            continue  # Skip functions whose types can't be resolved
        
        # The function node costs 1, so arguments must sum to (size - 1)
        remaining = size - 1
        
        # Handle higher-order arguments specially
        ho_indices = [
            i for i, t in enumerate(resolved_arg_types)
            if get_origin(t) == CallableOrig
        ]
        
        if ho_indices:
            self._enumerate_higher_order(
                func_name, resolved_arg_types, resolved_ret_type,
                ho_indices, remaining, size
            )
        else:
            self._enumerate_first_order(
                func_name, resolved_arg_types, resolved_ret_type,
                remaining, size
            )
```

### 6.5 First-Order Function Enumeration

For a function like `drop : Int → list[Int] → list[Int]` with 2 arguments and remaining size budget `r`:

```python
def _enumerate_first_order(
    self, func_name, arg_types, ret_type, remaining, total_size
):
    """Enumerate applications of a first-order function."""
    arity = len(arg_types)
    
    # Generate all partitions of `remaining` into `arity` positive parts
    for partition in integer_partitions(remaining, arity):
        # partition is e.g. (1, 2) meaning arg0 has size 1, arg1 has size 2
        
        # Get candidate programs for each argument position
        arg_candidates = []
        for i, (arg_type, arg_size) in enumerate(zip(arg_types, partition)):
            candidates = self.bank.get(arg_type, arg_size)
            if not candidates:
                break  # No candidates at this size for this type
            arg_candidates.append(candidates)
        else:
            # All argument positions have candidates — enumerate the Cartesian product
            for combo in itertools.product(*arg_candidates):
                node = ApplicationNode(
                    VariableNode(func_name),
                    [c.ast for c in combo]
                )
                self._try_add(node, ret_type, total_size)
```

### 6.6 Higher-Order Function Enumeration

For functions like `map : Callable[[int], int] → list[int] → list[int]`, the `Callable`-typed argument must be a lambda. We enumerate lambda bodies in an extended context.

```python
def _enumerate_higher_order(
    self, func_name, arg_types, ret_type, ho_indices, remaining, total_size
):
    """Enumerate applications of a higher-order function."""
    arity = len(arg_types)
    
    for partition in integer_partitions(remaining, arity):
        arg_candidates = []
        skip = False
        
        for i, (arg_type, arg_size) in enumerate(zip(arg_types, partition)):
            if i in ho_indices:
                # This argument is a lambda — enumerate lambda bodies
                lambdas = self._enumerate_lambdas(arg_type, arg_size)
                if not lambdas:
                    skip = True
                    break
                arg_candidates.append(lambdas)
            else:
                candidates = self.bank.get(arg_type, arg_size)
                if not candidates:
                    skip = True
                    break
                arg_candidates.append(candidates)
        
        if skip:
            continue
        
        for combo in itertools.product(*arg_candidates):
            node = ApplicationNode(
                VariableNode(func_name),
                [c.ast if isinstance(c, TypedProgram) else c for c in combo]
            )
            self._try_add(node, ret_type, total_size)
```

### 6.7 Lambda Enumeration

This is where nested enumeration contexts arise. For a `Callable[[int], int]` argument, we need to enumerate all lambda bodies `(λ y <body>)` where `body` has type `int` and may reference `y : int` and the outer input `x : list[int]`.

```python
def _enumerate_lambdas(self, callable_type, available_size) -> list[ASTNode]:
    """
    Enumerate all lambda expressions of a given callable type and size.
    
    A lambda (λ params body) costs 1 + size(body), so the body budget is
    available_size - 1.
    
    Args:
        callable_type: e.g. Callable[[int], int]
        available_size: total size budget for the lambda node
    
    Returns:
        List of LambdaNode ASTs
    """
    # Extract parameter types and return type from the Callable type
    args = get_args(callable_type)
    param_types = args[0]  # list of parameter types
    body_type = args[1]    # return type
    
    body_budget = available_size - 1  # lambda node costs 1
    if body_budget < 1:
        return []
    
    # Generate fresh parameter names
    param_names = [f"_p{i}" for i in range(len(param_types))]
    
    # Build the extended context: input var x + lambda params
    # We need a *separate* mini-enumeration for the lambda body.
    # This is a nested bottom-up enumeration in an extended variable context.
    
    # IMPORTANT: We reuse programs from the outer bank that don't depend
    # on the lambda parameters (they're still valid in the extended context).
    # But we also need new programs that USE the lambda parameters.
    
    # Approach: build a temporary bank for the lambda body.
    # Seed it with:
    #   - The lambda parameters as size-1 atoms
    #   - All programs from the outer bank (they're still valid here)
    # Then enumerate combinations up to body_budget.
    
    # For efficiency, rather than a full nested enumeration, do a simpler
    # version: only enumerate body terms up to a small depth.
    
    results = []
    
    # Size 1 bodies: just a parameter or an outer-bank size-1 term
    if body_budget >= 1:
        # Lambda parameters as body
        for pname, ptype in zip(param_names, param_types):
            if ptype == body_type or self._types_match(ptype, body_type):
                body = VariableNode(pname)
                results.append(LambdaNode(param_names, body))
        
        # Outer bank atoms of the right type
        for prog in self.bank.get(body_type, 1):
            results.append(LambdaNode(param_names, prog.ast))
    
    # Size 2+ bodies: apply functions to lambda params and outer terms
    for body_size in range(2, body_budget + 1):
        for func_name in self.grammar.names:
            func_info = self.grammar[func_name]
            f_arg_types = self._resolve_types(func_info['arg_types'])
            f_ret_type = self._resolve_type(func_info['ret_type'])
            
            if f_ret_type is None or f_ret_type != body_type:
                continue  # Return type doesn't match body type needed
            
            f_arity = len(f_arg_types)
            f_remaining = body_size - 1
            
            # Skip higher-order functions inside lambda bodies (for v1)
            # This prevents deep nesting. Can be relaxed later.
            ho = any(get_origin(t) == CallableOrig for t in f_arg_types)
            if ho:
                continue
            
            for partition in integer_partitions(f_remaining, f_arity):
                arg_candidates = []
                skip = False
                for j, (at, s) in enumerate(zip(f_arg_types, partition)):
                    # Candidates: outer bank programs + lambda parameters
                    cands = list(self.bank.get(at, s))
                    # Add lambda params if they match and size is 1
                    if s == 1:
                        for pname, ptype in zip(param_names, param_types):
                            if ptype == at or self._types_match(ptype, at):
                                cands.append(TypedProgram(
                                    ast=VariableNode(pname),
                                    type=ptype,
                                    fingerprint=None,  # Placeholder
                                    size=1
                                ))
                    if not cands:
                        skip = True
                        break
                    arg_candidates.append(cands)
                
                if skip:
                    continue
                
                for combo in itertools.product(*arg_candidates):
                    body = ApplicationNode(
                        VariableNode(func_name),
                        [c.ast for c in combo]
                    )
                    results.append(LambdaNode(param_names, body))
    
    # Deduplicate lambda results by fingerprint
    # Fingerprinting lambdas: they will be fingerprinted as part of the
    # enclosing application (e.g., map (λ y ...) x) at the call site.
    # So we return all candidates and let the caller's _try_add handle dedup.
    return results
```

**IMPORTANT SIMPLIFICATION FOR V1:** The lambda enumeration above skips higher-order functions inside lambda bodies (no `map` inside a `filter` lambda, etc.). This prevents exponential nesting and is sufficient for the initial implementation. Document this as a known limitation.

### 6.8 Integer Partitions

Implement a generator for ordered partitions of `n` into `k` positive parts:

```python
def integer_partitions(n: int, k: int) -> Iterator[tuple[int, ...]]:
    """
    Generate all ordered partitions of n into k parts, each >= 1.
    
    E.g., integer_partitions(4, 2) yields (1,3), (2,2), (3,1).
    """
    if k == 1:
        if n >= 1:
            yield (n,)
        return
    for first in range(1, n - k + 2):  # Leave at least 1 for each remaining part
        for rest in integer_partitions(n - first, k - 1):
            yield (first,) + rest
```

### 6.9 Type Resolution

For the fixed instantiation `T1 = int, T2 = int`:

```python
def _resolve_type(self, type_: TypeType) -> TypeType | None:
    """Resolve type variables to the fixed instantiation T1=int, T2=int."""
    subs = SubstitutionTable()
    # Use the grammar's T1, T2 type variables
    from .grammar import T1, T2
    subs[T1] = int
    subs[T2] = int
    try:
        return substitute_type_vars(type_, subs)
    except Exception:
        return None

def _resolve_types(self, types: tuple) -> list[TypeType | None]:
    return [self._resolve_type(t) for t in types]

def _types_match(self, type1: TypeType, type2: TypeType) -> bool:
    subs = SubstitutionTable()
    return matchable(type1, type2, subs, update=False)
```

**Note on polymorphism.** The fixed `T1=int, T2=int` instantiation means we only enumerate programs over integer lists. This misses programs that work on `list[bool]` or `list[list[int]]`, but it covers the vast majority of interesting list transformations and dramatically simplifies the enumerator. The type system can be generalised later by running the enumerator multiple times with different instantiations.

### 6.10 Final Corpus Extraction

```python
def extract_corpus(
    self,
    min_variability: float = 0.3,
    min_successes: int = 3,
) -> list[TypedProgram]:
    """Extract the final corpus of quality-filtered programs."""
    corpus = []
    for type_key, by_size in self.bank._bank.items():
        for size, progs in by_size.items():
            for prog in progs:
                if passes_quality_filter(
                    prog.fingerprint,
                    min_successes=min_successes,
                    min_variability=min_variability,
                ):
                    corpus.append(prog)
    return corpus
```

---

## 7. MDP Definition (`rl/mdp.py`)

### 7.1 State

```python
@dataclass
class SynthesisState:
    """
    State in the program synthesis MDP.
    
    Attributes:
        target_type: The type that must be generated at this AST node
        context: Dict mapping bound variable names to their types
        parent_func: Name of the function this term is an argument to (None if top-level)
        arg_index: Index of the argument position within parent_func (None if top-level)
        siblings: List of (ASTNode, Fingerprint) for already-generated sibling args
        depth_budget: Remaining depth budget
    """
    target_type: TypeType
    context: dict[str, TypeType]
    parent_func: str | None
    arg_index: int | None
    siblings: list[tuple[ASTNode, Fingerprint | None]]
    depth_budget: int
```

### 7.2 Actions

```python
class ActionType(Enum):
    LITERAL_INT = auto()     # Emit an integer literal
    LITERAL_BOOL = auto()    # Emit a boolean literal
    LITERAL_EMPTY_LIST = auto()  # Emit []
    VARIABLE = auto()        # Emit a bound variable
    APPLY = auto()           # Apply a grammar function
    LAMBDA = auto()          # Introduce a lambda
    IF = auto()              # Introduce an if-expression

@dataclass
class Action:
    action_type: ActionType
    # For LITERAL_INT: the integer value (from seed_constants)
    # For VARIABLE: the variable name
    # For APPLY: the function name
    payload: Any = None
```

### 7.3 Valid Action Enumeration

```python
def valid_actions(state: SynthesisState, grammar: Grammar, seed_constants: list[int]) -> list[Action]:
    """
    Enumerate all valid actions from the current state.
    
    An action is valid if it can produce a term of state.target_type
    given the current context and depth budget.
    """
    actions = []
    t = state.target_type
    
    # Literals
    if t == int:
        for c in seed_constants:
            actions.append(Action(ActionType.LITERAL_INT, c))
    if t == bool:
        actions.append(Action(ActionType.LITERAL_BOOL, True))
        actions.append(Action(ActionType.LITERAL_BOOL, False))
    if get_origin(t) == list or t == list:
        actions.append(Action(ActionType.LITERAL_EMPTY_LIST, None))
    
    # Variables in context with matching type
    for var_name, var_type in state.context.items():
        if var_type == t:  # Use exact match for the fixed instantiation
            actions.append(Action(ActionType.VARIABLE, var_name))
    
    # Function applications (only if depth budget > 0)
    if state.depth_budget > 0:
        for func_name in grammar.names:
            func_info = grammar[func_name]
            resolved_ret = _resolve_type(func_info['ret_type'])
            if resolved_ret == t:
                actions.append(Action(ActionType.APPLY, func_name))
        
        # If-expression (only if depth budget > 0)
        actions.append(Action(ActionType.IF, None))
    
    # Lambda (only if target type is Callable)
    if get_origin(t) == CallableOrig:
        actions.append(Action(ActionType.LAMBDA, None))
    
    return actions
```

### 7.4 State Transitions

Implement `step(state: SynthesisState, action: Action) -> list[SynthesisState]`:

Taking an action at a state produces zero or more *child states* (the holes that still need to be filled):

- `LITERAL_*` and `VARIABLE`: return `[]` (leaf node, no children).
- `APPLY(func_name)`: return one child state per argument of `func_name`, each with `target_type` set to the argument's type, `parent_func = func_name`, `arg_index = i`, `depth_budget = state.depth_budget - 1`.
- `LAMBDA`: extract param types and return type from `state.target_type`. Return one child state for the body with extended context and `target_type = return_type`.
- `IF`: return three child states: condition (type `bool`), then-branch (type `state.target_type`), else-branch (type `state.target_type`), each with `depth_budget = state.depth_budget - 1`.

### 7.5 Episode Runner

```python
class Episode:
    """
    Runs a single episode of the synthesis MDP.
    
    Uses a policy to make decisions at each state, building an AST
    top-down. Records the full trajectory for training.
    """
    
    def __init__(self, policy, grammar, test_suite, seed_constants, max_depth=6):
        self.policy = policy
        self.grammar = grammar
        self.test_suite = test_suite
        self.seed_constants = seed_constants
        self.max_depth = max_depth
        self.trajectory = []  # list of (state, action) pairs
    
    def run(self) -> tuple[ASTNode | None, list[tuple[SynthesisState, Action]]]:
        """
        Run one episode.
        
        Returns:
            (completed_ast, trajectory) or (None, trajectory) if generation fails.
        """
        initial_state = SynthesisState(
            target_type=Callable[[list[int]], list[int]],  # [Int] -> [Int]
            context={},
            parent_func=None,
            arg_index=None,
            siblings=[],
            depth_budget=self.max_depth,
        )
        
        ast = self._generate(initial_state)
        return ast, self.trajectory
    
    def _generate(self, state: SynthesisState) -> ASTNode | None:
        """Recursively generate an AST node by querying the policy."""
        actions = valid_actions(state, self.grammar, self.seed_constants)
        if not actions:
            return None  # Dead end
        
        action = self.policy.select_action(state, actions)
        self.trajectory.append((state, action))
        
        if action.action_type == ActionType.LITERAL_INT:
            return NumberNode(action.payload)
        elif action.action_type == ActionType.LITERAL_BOOL:
            return BooleanNode(action.payload)
        elif action.action_type == ActionType.LITERAL_EMPTY_LIST:
            return ListNode([])
        elif action.action_type == ActionType.VARIABLE:
            return VariableNode(action.payload)
        elif action.action_type == ActionType.APPLY:
            return self._generate_application(state, action)
        elif action.action_type == ActionType.LAMBDA:
            return self._generate_lambda(state)
        elif action.action_type == ActionType.IF:
            return self._generate_if(state)
        return None
    
    def _generate_application(self, state, action):
        func_name = action.payload
        func_info = self.grammar[func_name]
        arg_types = [_resolve_type(t) for t in func_info['arg_types']]
        
        arg_nodes = []
        for i, arg_type in enumerate(arg_types):
            child_state = SynthesisState(
                target_type=arg_type,
                context=state.context,
                parent_func=func_name,
                arg_index=i,
                siblings=[(n, None) for n in arg_nodes],  # Already-generated siblings
                depth_budget=state.depth_budget - 1,
            )
            arg_node = self._generate(child_state)
            if arg_node is None:
                return None
            arg_nodes.append(arg_node)
        
        return ApplicationNode(VariableNode(func_name), arg_nodes)
    
    def _generate_lambda(self, state):
        args = get_args(state.target_type)
        param_types = args[0]
        body_type = args[1]
        param_names = [f"_p{i}" for i in range(len(param_types))]
        
        new_context = state.context.copy()
        for pname, ptype in zip(param_names, param_types):
            new_context[pname] = ptype
        
        body_state = SynthesisState(
            target_type=body_type,
            context=new_context,
            parent_func=None,
            arg_index=None,
            siblings=[],
            depth_budget=state.depth_budget - 1,
        )
        body_node = self._generate(body_state)
        if body_node is None:
            return None
        return LambdaNode(param_names, body_node)
    
    def _generate_if(self, state):
        cond_state = SynthesisState(
            target_type=bool,
            context=state.context,
            parent_func=None, arg_index=None, siblings=[],
            depth_budget=state.depth_budget - 1,
        )
        cond = self._generate(cond_state)
        if cond is None:
            return None
        
        then_state = SynthesisState(
            target_type=state.target_type,
            context=state.context,
            parent_func=None, arg_index=None, siblings=[],
            depth_budget=state.depth_budget - 1,
        )
        then_node = self._generate(then_state)
        if then_node is None:
            return None
        
        else_state = SynthesisState(
            target_type=state.target_type,
            context=state.context,
            parent_func=None, arg_index=None, siblings=[],
            depth_budget=state.depth_budget - 1,
        )
        else_node = self._generate(else_state)
        if else_node is None:
            return None
        
        return IfNode(cond, then_node, else_node)
```

---

## 8. Policy Network (`rl/policy.py`)

Use PyTorch. The policy is a simple MLP that maps state embeddings to action logits.

### 8.1 State Embedding

```python
class StateEncoder(nn.Module):
    """Encode a SynthesisState into a fixed-size vector."""
    
    def __init__(self, type_vocab_size, func_vocab_size, embed_dim=64):
        super().__init__()
        self.type_embed = nn.Embedding(type_vocab_size, embed_dim)
        self.func_embed = nn.Embedding(func_vocab_size + 1, embed_dim)  # +1 for None
        self.arg_index_embed = nn.Embedding(8, embed_dim)  # max arity ~4, padded
        self.depth_embed = nn.Embedding(16, embed_dim)     # max depth ~10
        
        # Context: variable count features
        self.context_proj = nn.Linear(16, embed_dim)  # one-hot variable type counts
        
        # Combine
        self.combine = nn.Linear(5 * embed_dim, embed_dim)
    
    def forward(self, state_batch):
        """
        Args:
            state_batch: dict with keys 'target_type', 'parent_func', 
                         'arg_index', 'depth_budget', 'context_features'
                         each as a tensor of shape (batch_size,) or (batch_size, feat_dim)
        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        t = self.type_embed(state_batch['target_type'])
        f = self.func_embed(state_batch['parent_func'])
        i = self.arg_index_embed(state_batch['arg_index'])
        d = self.depth_embed(state_batch['depth_budget'])
        c = self.context_proj(state_batch['context_features'])
        
        combined = torch.cat([t, f, i, d, c], dim=-1)
        return torch.relu(self.combine(combined))
```

### 8.2 Vocabulary Construction

Before training, build vocabularies mapping types and function names to integer indices:

```python
def build_type_vocab(grammar: Grammar) -> dict[TypeType, int]:
    """Collect all resolved types that appear in the grammar."""
    types = {int, bool, list[int], list[list[int]], list[bool],
             Callable[[int], int], Callable[[int], bool],
             Callable[[int, int], int],  # for fold
             Callable[[list[int]], int],  # etc.
             }
    # Add more from grammar inspection
    return {t: i for i, t in enumerate(sorted(types, key=str))}

def build_func_vocab(grammar: Grammar) -> dict[str | None, int]:
    """Map function names to indices. None maps to 0."""
    vocab = {None: 0}
    for i, name in enumerate(grammar.names, start=1):
        vocab[name] = i
    return vocab
```

### 8.3 Action Head

```python
class PolicyNetwork(nn.Module):
    """
    Full policy network: state -> distribution over actions.
    
    Uses a shared state encoder and a linear head that scores all
    possible actions. Invalid actions are masked to -inf before softmax.
    """
    
    def __init__(self, action_vocab_size, type_vocab_size, func_vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = StateEncoder(type_vocab_size, func_vocab_size, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_vocab_size),
        )
    
    def forward(self, state_batch, valid_action_mask):
        """
        Args:
            state_batch: dict of tensors
            valid_action_mask: BoolTensor of shape (batch_size, action_vocab_size)
                              True for valid actions, False for invalid.
        Returns:
            log_probs: Tensor of shape (batch_size, action_vocab_size)
        """
        h = self.encoder(state_batch)
        logits = self.head(h)
        logits[~valid_action_mask] = float('-inf')
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
```

### 8.4 Action Vocabulary

Build a fixed action vocabulary where each action is an integer index:

```python
def build_action_vocab(grammar, seed_constants):
    """Build a mapping from Action -> int index."""
    vocab = {}
    idx = 0
    
    for c in seed_constants:
        vocab[Action(ActionType.LITERAL_INT, c)] = idx; idx += 1
    vocab[Action(ActionType.LITERAL_BOOL, True)] = idx; idx += 1
    vocab[Action(ActionType.LITERAL_BOOL, False)] = idx; idx += 1
    vocab[Action(ActionType.LITERAL_EMPTY_LIST, None)] = idx; idx += 1
    
    # Variables: use a fixed set of possible variable names
    for var_name in ["x", "_p0", "_p1", "_p2"]:
        vocab[Action(ActionType.VARIABLE, var_name)] = idx; idx += 1
    
    for func_name in grammar.names:
        vocab[Action(ActionType.APPLY, func_name)] = idx; idx += 1
    
    vocab[Action(ActionType.LAMBDA, None)] = idx; idx += 1
    vocab[Action(ActionType.IF, None)] = idx; idx += 1
    
    return vocab
```

---

## 9. Priority Queue Buffer (`rl/priority_queue.py`)

```python
class PriorityQueueBuffer:
    """
    Bounded buffer of the top-K highest-reward programs.
    
    Programs are stored as (reward, program_ast, trajectory) tuples.
    When the buffer is full, inserting a program with reward higher
    than the current minimum evicts the minimum.
    """
    
    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.buffer: list[tuple[float, ASTNode, list]] = []
        self.fingerprints: set[Fingerprint] = set()  # Prevent exact duplicates
    
    def insert(self, reward: float, program: ASTNode, trajectory: list, fingerprint: Fingerprint) -> bool:
        """
        Insert a program if it improves the buffer.
        
        Returns True if inserted.
        """
        if fingerprint in self.fingerprints:
            return False  # Exact duplicate
        
        if len(self.buffer) < self.capacity:
            heapq.heappush(self.buffer, (reward, id(program), program, trajectory, fingerprint))
            self.fingerprints.add(fingerprint)
            return True
        
        # Buffer full — check if new program beats the minimum
        if reward > self.buffer[0][0]:
            evicted = heapq.heapreplace(self.buffer, (reward, id(program), program, trajectory, fingerprint))
            self.fingerprints.discard(evicted[4])
            self.fingerprints.add(fingerprint)
            return True
        
        return False
    
    def sample(self, batch_size: int) -> list[tuple[float, ASTNode, list]]:
        """Sample a batch of (reward, program, trajectory) tuples."""
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        return [(self.buffer[i][0], self.buffer[i][2], self.buffer[i][3]) for i in indices]
    
    def min_reward(self) -> float:
        if not self.buffer:
            return 0.0
        return self.buffer[0][0]
    
    def __len__(self):
        return len(self.buffer)
```

---

## 10. Trajectory Extraction (`rl/trajectory.py`)

To warm-start the policy from the enumeration corpus, we need to extract trajectories — the sequence of `(state, action)` pairs that would produce a given program under the MDP.

```python
def extract_trajectory(
    program: ASTNode,
    target_type: TypeType,
    grammar: Grammar,
    initial_context: dict[str, TypeType] = None,
    initial_depth: int = 8,
) -> list[tuple[SynthesisState, Action]]:
    """
    Given a complete program AST, extract the trajectory of (state, action)
    pairs that would produce it under the top-down MDP.
    
    This is a deterministic tree walk: at each AST node, the state is
    determined by the node's position in the tree, and the action is
    determined by the node's type.
    """
    if initial_context is None:
        initial_context = {}
    
    trajectory = []
    
    def _walk(node: ASTNode, state: SynthesisState):
        if isinstance(node, NumberNode):
            trajectory.append((state, Action(ActionType.LITERAL_INT, node.value)))
        
        elif isinstance(node, BooleanNode):
            trajectory.append((state, Action(ActionType.LITERAL_BOOL, node.value)))
        
        elif isinstance(node, ListNode) and len(node.elements) == 0:
            trajectory.append((state, Action(ActionType.LITERAL_EMPTY_LIST, None)))
        
        elif isinstance(node, VariableNode):
            trajectory.append((state, Action(ActionType.VARIABLE, node.name)))
        
        elif isinstance(node, ApplicationNode):
            if isinstance(node.function, VariableNode):
                func_name = node.function.name
                trajectory.append((state, Action(ActionType.APPLY, func_name)))
                
                func_info = grammar[func_name]
                arg_types = [_resolve_type(t) for t in func_info['arg_types']]
                
                generated_siblings = []
                for i, (arg_node, arg_type) in enumerate(zip(node.arguments, arg_types)):
                    child_state = SynthesisState(
                        target_type=arg_type,
                        context=state.context,
                        parent_func=func_name,
                        arg_index=i,
                        siblings=list(generated_siblings),
                        depth_budget=state.depth_budget - 1,
                    )
                    _walk(arg_node, child_state)
                    generated_siblings.append((arg_node, None))
        
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
            )
            _walk(node.body, body_state)
        
        elif isinstance(node, IfNode):
            trajectory.append((state, Action(ActionType.IF, None)))
            
            cond_state = SynthesisState(
                target_type=bool, context=state.context,
                parent_func=None, arg_index=None, siblings=[],
                depth_budget=state.depth_budget - 1,
            )
            _walk(node.condition, cond_state)
            
            then_state = SynthesisState(
                target_type=state.target_type, context=state.context,
                parent_func=None, arg_index=None, siblings=[],
                depth_budget=state.depth_budget - 1,
            )
            _walk(node.then_expr, then_state)
            
            else_state = SynthesisState(
                target_type=state.target_type, context=state.context,
                parent_func=None, arg_index=None, siblings=[],
                depth_budget=state.depth_budget - 1,
            )
            _walk(node.else_expr, else_state)
    
    initial_state = SynthesisState(
        target_type=target_type,
        context=initial_context,
        parent_func=None,
        arg_index=None,
        siblings=[],
        depth_budget=initial_depth,
    )
    _walk(program, initial_state)
    return trajectory
```

---

## 11. Reward Functions (`rl/reward.py`)

```python
def compute_reward(
    fingerprint: Fingerprint,
    corpus_fingerprints: set[Fingerprint],
    alpha: float = 0.5,
) -> float:
    """
    Compute the reward for a generated program.
    
    reward = variability(fp) * novelty_bonus
    
    where novelty_bonus = 1.0 if fp is not in corpus_fingerprints, else 0.1
    (small positive reward for rediscovering known programs, to maintain
    the policy's ability to generate them).
    
    Args:
        fingerprint: The program's fingerprint
        corpus_fingerprints: Set of all fingerprints in the current corpus
        alpha: Weight for novelty vs quality tradeoff
    """
    var = variability(fingerprint)
    
    if not is_non_crashing(fingerprint, min_successes=3):
        return 0.0
    
    if not is_non_constant(fingerprint):
        return 0.0
    
    novelty = 1.0 if fingerprint not in corpus_fingerprints else 0.1
    
    return var * novelty
```

---

## 12. Training Loop (`rl/trainer.py`)

### 12.1 Behavioural Cloning (Warm-Start)

```python
def warm_start(
    policy: PolicyNetwork,
    corpus: list[TypedProgram],
    grammar: Grammar,
    action_vocab: dict,
    type_vocab: dict,
    func_vocab: dict,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """
    Pre-train the policy via behavioural cloning on the enumeration corpus.
    
    For each program in the corpus, extract its trajectory and train the
    policy to predict each action given the corresponding state.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    # Extract all trajectories
    all_transitions = []  # list of (state, action) pairs
    for prog in corpus:
        target_type = Callable[[list[int]], list[int]]
        traj = extract_trajectory(prog.ast, target_type, grammar)
        all_transitions.extend(traj)
    
    print(f"Warm-start: {len(all_transitions)} transitions from {len(corpus)} programs")
    
    dataset = TransitionDataset(all_transitions, action_vocab, type_vocab, func_vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            state_batch, action_indices, valid_masks = batch
            log_probs = policy(state_batch, valid_masks)
            loss = F.nll_loss(log_probs, action_indices)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss = {total_loss/n_batches:.4f}")
```

### 12.2 Priority Queue Training Loop

```python
def train_rl(
    policy: PolicyNetwork,
    buffer: PriorityQueueBuffer,
    grammar: Grammar,
    test_suite: list[list[int]],
    action_vocab: dict,
    type_vocab: dict,
    func_vocab: dict,
    corpus_fingerprints: set[Fingerprint],
    n_iterations: int = 10000,
    episodes_per_iter: int = 32,
    train_steps_per_iter: int = 8,
    batch_size: int = 64,
    lr: float = 1e-4,
    max_depth: int = 8,
    seed_constants: list[int] = [0, 1, 2, 3],
):
    """
    Main RL training loop with priority queue training.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    jit = JITCompiler(grammar)
    
    stats = {'novel_found': 0, 'total_generated': 0, 'buffer_min': 0.0}
    
    for iteration in range(n_iterations):
        # === Sampling Phase ===
        for _ in range(episodes_per_iter):
            episode = Episode(policy, grammar, test_suite, seed_constants, max_depth)
            ast, trajectory = episode.run()
            stats['total_generated'] += 1
            
            if ast is None:
                continue
            
            # Wrap in lambda and fingerprint
            closed_ast = LambdaNode(["x"], ast) if not isinstance(ast, LambdaNode) else ast
            fp = compute_fingerprint(closed_ast, test_suite, jit)
            if fp is None:
                continue
            
            reward = compute_reward(fp, corpus_fingerprints)
            if reward > 0:
                inserted = buffer.insert(reward, closed_ast, trajectory, fp)
                if inserted and fp not in corpus_fingerprints:
                    corpus_fingerprints.add(fp)
                    stats['novel_found'] += 1
        
        # === Training Phase ===
        if len(buffer) < batch_size:
            continue
        
        for _ in range(train_steps_per_iter):
            batch = buffer.sample(batch_size)
            
            # Extract transitions from trajectories
            all_transitions = []
            for reward, program, trajectory in batch:
                for state, action in trajectory:
                    all_transitions.append((state, action, reward))
            
            if not all_transitions:
                continue
            
            # Convert to tensors and compute loss
            states, actions, rewards = zip(*all_transitions)
            state_batch = encode_states(states, type_vocab, func_vocab)
            action_indices = torch.tensor([action_vocab[a] for a in actions])
            reward_weights = torch.tensor([r for r in rewards], dtype=torch.float32)
            valid_masks = compute_valid_masks(states, grammar, seed_constants, action_vocab)
            
            log_probs = policy(state_batch, valid_masks)
            per_action_log_prob = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            
            # Reward-weighted maximum likelihood
            loss = -(reward_weights * per_action_log_prob).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        
        # === Logging ===
        if (iteration + 1) % 100 == 0:
            stats['buffer_min'] = buffer.min_reward()
            print(
                f"Iter {iteration+1}: buffer={len(buffer)}, "
                f"novel={stats['novel_found']}, "
                f"generated={stats['total_generated']}, "
                f"buffer_min={stats['buffer_min']:.3f}"
            )
```

---

## 13. Pipeline Orchestrator (`pipeline.py`)

```python
def run_pipeline(
    grammar: Grammar = DefaultGrammar,
    enum_max_size: int = 5,
    enum_min_variability: float = 0.3,
    rl_iterations: int = 10000,
    rl_max_depth: int = 8,
    buffer_capacity: int = 5000,
    seed_constants: list[int] = [0, 1, 2, 3],
    output_dir: str = "output",
):
    """
    Full pipeline: enumeration → warm-start → RL exploration.
    """
    test_suite = DEFAULT_TEST_SUITE
    
    # === Phase 1: Enumeration ===
    print("=" * 60)
    print("Phase 1: Bottom-Up Enumeration")
    print("=" * 60)
    
    enumerator = BottomUpEnumerator(
        grammar=grammar,
        test_suite=test_suite,
        seed_constants=seed_constants,
        max_size=enum_max_size,
        min_variability=enum_min_variability,
    )
    bank = enumerator.enumerate()
    corpus = enumerator.extract_corpus(min_variability=enum_min_variability)
    
    print(f"\nEnumeration complete:")
    print(f"  Total programs in bank: {bank.count()}")
    print(f"  Quality-filtered corpus: {len(corpus)}")
    
    # Save enumeration results
    save_corpus(corpus, f"{output_dir}/enumeration_corpus.json")
    
    # === Phase 2: Warm-Start ===
    print("\n" + "=" * 60)
    print("Phase 2: Warm-Start via Behavioural Cloning")
    print("=" * 60)
    
    action_vocab = build_action_vocab(grammar, seed_constants)
    type_vocab = build_type_vocab(grammar)
    func_vocab = build_func_vocab(grammar)
    
    policy = PolicyNetwork(
        action_vocab_size=len(action_vocab),
        type_vocab_size=len(type_vocab),
        func_vocab_size=len(func_vocab),
    )
    
    warm_start(policy, corpus, grammar, action_vocab, type_vocab, func_vocab)
    
    # Seed the priority queue buffer with the enumeration corpus
    buffer = PriorityQueueBuffer(capacity=buffer_capacity)
    corpus_fingerprints = set()
    
    jit = JITCompiler(grammar)
    for prog in corpus:
        target_type = Callable[[list[int]], list[int]]
        traj = extract_trajectory(prog.ast, target_type, grammar)
        reward = compute_reward(prog.fingerprint, corpus_fingerprints)
        buffer.insert(reward, prog.ast, traj, prog.fingerprint)
        corpus_fingerprints.add(prog.fingerprint)
    
    print(f"Buffer seeded with {len(buffer)} programs")
    
    # === Phase 3: RL Exploration ===
    print("\n" + "=" * 60)
    print("Phase 3: RL Exploration")
    print("=" * 60)
    
    train_rl(
        policy=policy,
        buffer=buffer,
        grammar=grammar,
        test_suite=test_suite,
        action_vocab=action_vocab,
        type_vocab=type_vocab,
        func_vocab=func_vocab,
        corpus_fingerprints=corpus_fingerprints,
        n_iterations=rl_iterations,
        max_depth=rl_max_depth,
        seed_constants=seed_constants,
    )
    
    # === Final Output ===
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    
    final_corpus = [prog for _, prog, _ in buffer.buffer]  # All programs in buffer
    print(f"Total unique programs: {len(corpus_fingerprints)}")
    print(f"Buffer size: {len(buffer)}")
    
    save_corpus_asts(final_corpus, f"{output_dir}/final_corpus.json")
```

---

## 14. Implementation Order and Testing Strategy

Implement in this order, testing each component before moving to the next:

### Stage 1: Foundation (estimate: 1-2 days)
1. `utils.py` — program size, free variables
2. `enumeration/test_suite.py` — test suite definition and evaluation function
3. `enumeration/fingerprint.py` — Fingerprint class, FingerprintTable
4. `enumeration/filters.py` — quality predicates

**Test:** Manually construct 5-10 ASTs (including known-degenerate ones like `fold (λ y z. z) 0 x`), compute fingerprints, verify that degenerate programs are correctly identified and that distinct programs get distinct fingerprints.

### Stage 2: Enumerator (estimate: 3-5 days)
5. `enumeration/enumerator.py` — ProgramBank, BottomUpEnumerator

**Test at each size level:**
- Size 1: verify bank contains `{0, 1, 2, 3, true, false, [], x}` with correct types
- Size 2: verify `(length x)`, `(reverse x)`, `(sum x)`, `(max x)`, `(unique x)`, `(first x)`, `(last x)` are present. Verify `(length [])` collapses with `NumberNode(0)`.
- Size 3: verify `(drop 1 x)`, `(take 2 x)`, `(cons 0 x)`, `(+ (length x) 1)` are present. Verify `(map (λ y y) x)` collapses with `x` (identity).
- Size 4-5: log counts per type per size. Expect 100s-1000s of programs. Verify wall-clock time is reasonable (< 1 hour for size 5).

**Critical benchmark:** Run enumeration up to size 5 and measure:
- Total candidate programs generated per size level
- Total survivors after observational equivalence pruning
- Wall-clock time per size level
- Memory usage

### Stage 3: MDP and Episode Runner (estimate: 2-3 days)
6. `rl/mdp.py` — state, action, valid_actions, step
7. `rl/trajectory.py` — trajectory extraction
8. `rl/reward.py` — reward computation

**Test:** Extract trajectories from 10 enumerated programs, verify they reconstruct the original ASTs. Run the episode runner with a random policy, verify it produces well-typed ASTs (even if they're mostly degenerate).

### Stage 4: Policy and Training (estimate: 3-4 days)
9. `rl/policy.py` — StateEncoder, PolicyNetwork, vocabulary builders
10. `rl/priority_queue.py` — PriorityQueueBuffer
11. `rl/trainer.py` — warm_start, train_rl

**Test:** Run warm-start on the enumeration corpus, verify the policy loss decreases. Sample 100 programs from the warm-started policy, verify >50% are non-degenerate (vs <6% from random sampling). Run 1000 RL iterations, verify the buffer's minimum reward increases over time.

### Stage 5: Integration (estimate: 1-2 days)
12. `pipeline.py` — full orchestrator

**Test:** Run the full pipeline with `enum_max_size=4` (faster) and `rl_iterations=1000`. Verify the final corpus is larger and more diverse than the enumeration corpus alone.

---

## 15. Known Simplifications and Future Extensions

Document these in the code as TODO comments:

1. **Fixed type instantiation (T1=int, T2=int).** Generalise to enumerate over multiple type instantiations.
2. **No higher-order functions inside lambda bodies.** Allow nested map/filter/fold for richer lambda expressions.
3. **No if-expressions in the enumerator.** Add these as a separate enumeration case (they require a Bool-typed condition sub-enumeration).
4. **Sibling encoding in policy state.** Currently `siblings` is not fully encoded by the neural network. Add an RNN/attention layer over generated siblings.
5. **Novelty reward is binary.** Replace with continuous distance to nearest neighbour in fingerprint space for smoother gradient signal.
6. **Library learning.** See Section 8 of the research notes document for the deferred extension plan and trigger conditions.
7. **Parallel enumeration.** The Cartesian product enumeration at each size is embarrassingly parallel. Use multiprocessing.
8. **Adaptive test suite.** Add inputs that distinguish programs currently in the same equivalence class.

---

## 16. Dependencies

Add to `requirements.txt`:

```
torch>=2.0
numpy
```

The existing codebase has no external dependencies beyond the Python standard library. The RL component requires PyTorch. Do not introduce other ML frameworks.

---

## 17. Performance Targets

| Metric | Target | How to measure |
|--------|--------|----------------|
| Enumeration size 5 completes | < 1 hour | Wall-clock time |
| Enumeration corpus size | > 1000 distinct programs | `len(corpus)` |
| Warm-started policy non-degeneracy | > 50% of sampled programs | Sample 100, check variability |
| RL discovers novel programs | > 100 new fingerprints in 10k iters | Count novel insertions into buffer |
| Random baseline non-degeneracy | ~6% | Confirm the improvement is real |
