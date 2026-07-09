# Primitive Subset Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/lang/subset_search/`, a three-stage search that finds 5–6 primitive subsets of a curated 18-primitive pool maximizing behavioral diversity of bounded enumeration.

**Architecture:** Stage 0 enumerates the full pool once and proxy-scores all 27,132 subsets via witness support sets; Stage 1 deep-enumerates the top ~200 subsets exactly and computes a 6-scalar score vector; Stage 2 takes the Pareto front and emits qualitative markdown reports. Reuses `BottomUpEnumerator`, `Grammar.subset`, `Fingerprint`, and `passes_quality_filter` unchanged (two tiny additive changes to `enumerator.py`).

**Tech Stack:** Pure Python (stdlib only: `itertools`, `json`, `signal`, `concurrent.futures`, `argparse`, `random`). Tests with pytest.

**Spec:** `docs/superpowers/specs/2026-07-08-primitive-subset-search-design.md`

## Global Constraints

- Run all Python via the ml13 micromamba env: `micromamba run -n ml13 python -m pytest ...` from the repo root `/Users/yiding/Desktop/Research/mlmp`.
- Imports use the `from src.lang. ...` absolute style in tests, relative (`from ..grammar import ...`) inside `src/lang/`.
- The 18-primitive pool is exactly: `+ - * % < == and not singleton cons concat take drop length range map filter fold`.
- Subset sizes searched: 5 and 6. C(18,5)+C(18,6) = 8,568+18,564 = 27,132.
- Quality-filter thresholds are reused as-is: `min_successes=3`, `min_variability=0.3`.
- Results are written under `outputs/subset_search/` (the `outputs/` dir already exists and is not tracked by git).
- Do not modify the test suite, filters, or fingerprints. Only additive changes to `enumerator.py` (Task 2).

## File Structure

```
src/lang/subset_search/
  __init__.py          # empty package marker
  pool.py              # POOL_NAMES, pool grammar, subset iteration, subset keys
  support.py           # support-set extraction, Stage 0 proxy scoring
  scoring.py           # Stage 1 score vector from a ProgramBank
  search.py            # stage orchestration, multiprocessing, JSON persistence
  report.py            # Pareto front + Stage 2 markdown reports
  __main__.py          # argparse CLI: --stage 0|1|2
  tests/
    __init__.py
    test_pool.py
    test_bank_helpers.py
    test_support.py
    test_scoring.py
    test_search.py
    test_report.py
src/lang/enumeration/enumerator.py   # modified: ProgramBank.all_programs(), attempts counter
```

---

### Task 1: Pool module

**Files:**
- Create: `src/lang/subset_search/__init__.py` (empty)
- Create: `src/lang/subset_search/pool.py`
- Create: `src/lang/subset_search/tests/__init__.py` (empty)
- Test: `src/lang/subset_search/tests/test_pool.py`

**Interfaces:**
- Consumes: `DefaultGrammar.subset(names: set[str]) -> Grammar` from `src/lang/grammar.py`.
- Produces: `POOL_NAMES: tuple[str, ...]` (18 names), `pool_grammar(pool_names=None) -> Grammar`, `iter_subsets(sizes=(5, 6), pool_names=None) -> Iterator[frozenset[str]]`, `subset_key(subset: frozenset[str]) -> str` (space-joined sorted names).

- [ ] **Step 1: Write the failing test**

```python
# src/lang/subset_search/tests/test_pool.py
"""Tests for the curated primitive pool and subset generation."""

from math import comb

from src.lang.grammar import DefaultGrammar
from src.lang.subset_search.pool import (
    POOL_NAMES,
    pool_grammar,
    iter_subsets,
    subset_key,
)


def test_pool_has_18_primitives_all_in_default_grammar():
    assert len(POOL_NAMES) == 18
    assert len(set(POOL_NAMES)) == 18
    for name in POOL_NAMES:
        assert name in DefaultGrammar.functions, name


def test_pool_grammar_contains_exactly_the_pool():
    g = pool_grammar()
    assert set(g.names) == set(POOL_NAMES)


def test_iter_subsets_counts():
    subsets = list(iter_subsets(sizes=(5, 6)))
    assert len(subsets) == comb(18, 5) + comb(18, 6)  # 27,132
    assert all(isinstance(s, frozenset) for s in subsets[:10])
    assert len(set(subsets)) == len(subsets)


def test_iter_subsets_respects_pool_override():
    small_pool = ('+', '*', 'map', 'fold', 'take', 'concat')
    subsets = list(iter_subsets(sizes=(5,), pool_names=small_pool))
    assert len(subsets) == comb(6, 5)
    for s in subsets:
        assert s <= frozenset(small_pool)


def test_subset_key_is_sorted_and_stable():
    s = frozenset({'map', '+', 'fold'})
    assert subset_key(s) == '+ fold map'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_pool.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'src.lang.subset_search'`

- [ ] **Step 3: Write the implementation**

Create empty `src/lang/subset_search/__init__.py` and `src/lang/subset_search/tests/__init__.py`, then:

```python
# src/lang/subset_search/pool.py
"""Curated primitive pool and subset generation for the subset search."""

from itertools import combinations
from typing import Iterator, Optional, Sequence

from ..grammar import DefaultGrammar, Grammar


POOL_NAMES: tuple[str, ...] = (
    '+', '-', '*', '%', '<', '==', 'and', 'not',
    'singleton', 'cons', 'concat', 'take', 'drop', 'length', 'range',
    'map', 'filter', 'fold',
)


def pool_grammar(pool_names: Optional[Sequence[str]] = None) -> Grammar:
    """Grammar restricted to the pool (or an explicit override, for tests)."""
    names = tuple(pool_names) if pool_names is not None else POOL_NAMES
    return DefaultGrammar.subset(set(names))


def iter_subsets(
    sizes: tuple[int, ...] = (5, 6),
    pool_names: Optional[Sequence[str]] = None,
) -> Iterator[frozenset[str]]:
    """Yield every subset of the pool with a cardinality in ``sizes``."""
    names = tuple(pool_names) if pool_names is not None else POOL_NAMES
    for k in sizes:
        for combo in combinations(names, k):
            yield frozenset(combo)


def subset_key(subset: frozenset[str]) -> str:
    """Stable string key for a subset (names contain no spaces)."""
    return ' '.join(sorted(subset))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_pool.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/lang/subset_search
git commit -m "feat(subset_search): curated 18-primitive pool and subset generation"
```

---

### Task 2: ProgramBank/enumerator helpers

**Files:**
- Modify: `src/lang/enumeration/enumerator.py` (class `ProgramBank` ~line 34, class `BottomUpEnumerator.__init__` ~line 171, `_try_add` ~line 288)
- Test: `src/lang/subset_search/tests/test_bank_helpers.py`

**Interfaces:**
- Consumes: existing `ProgramBank._bank` layout (`dict[type][size] -> list[TypedProgram]`), `BottomUpEnumerator._try_add`.
- Produces: `ProgramBank.all_programs() -> Iterator[TypedProgram]` (yields every stored program; each distinct behavior appears exactly once and `TypedProgram.size` is its minimal size, guaranteed by bottom-up insertion order), and `BottomUpEnumerator.attempts: int` (count of candidate programs fingerprinted, i.e. `_try_add` calls — the denominator for semantic density).

- [ ] **Step 1: Write the failing test**

```python
# src/lang/subset_search/tests/test_bank_helpers.py
"""Tests for ProgramBank.all_programs() and the enumerator attempts counter."""

from src.lang.enumeration.enumerator import BottomUpEnumerator
from src.lang.subset_search.pool import pool_grammar


def _tiny_enum():
    g = pool_grammar(('+', 'length', 'take'))
    enum = BottomUpEnumerator(grammar=g, max_size=3)
    bank = enum.enumerate()
    return enum, bank


def test_all_programs_yields_every_stored_program():
    enum, bank = _tiny_enum()
    programs = list(bank.all_programs())
    assert len(programs) == bank.count()
    assert len(programs) > 0


def test_all_programs_unique_per_type_and_fingerprint():
    _, bank = _tiny_enum()
    seen = set()
    for p in bank.all_programs():
        if p.fingerprint is None:
            continue
        key = (str(p.type), p.fingerprint)
        assert key not in seen, f"duplicate behavior: {p.ast}"
        seen.add(key)


def test_attempts_counts_at_least_stored_programs():
    enum, bank = _tiny_enum()
    assert enum.attempts >= bank.count()
    assert enum.attempts > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_bank_helpers.py -v`
Expected: FAIL with `AttributeError: 'ProgramBank' object has no attribute 'all_programs'`

- [ ] **Step 3: Implement the two additions**

In `src/lang/enumeration/enumerator.py`, add to class `ProgramBank` (after `count()`):

```python
    def all_programs(self) -> Iterator[TypedProgram]:
        """Yield every stored program, sizes ascending within each type.

        Each distinct (type, fingerprint) appears exactly once; a program's
        ``size`` is the minimal size for its behavior because bottom-up
        enumeration inserts the first (smallest) witness and dedups the rest.
        """
        for by_size in self._bank.values():
            for size in sorted(by_size):
                yield from by_size[size]
```

In `BottomUpEnumerator.__init__`, after `self.bank = ProgramBank()`:

```python
        self.attempts = 0  # candidate programs fingerprinted (incl. duplicates)
```

In `BottomUpEnumerator._try_add`, as the first line of the method body:

```python
        self.attempts += 1
```

- [ ] **Step 4: Run the new test and the existing enumeration/grammar tests**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_bank_helpers.py src/lang/tests -v`
Expected: all PASSED (no regressions)

- [ ] **Step 5: Commit**

```bash
git add src/lang/enumeration/enumerator.py src/lang/subset_search/tests/test_bank_helpers.py
git commit -m "feat(enumeration): ProgramBank.all_programs and attempts counter"
```

---

### Task 3: Support sets and Stage 0 proxy scoring

**Files:**
- Create: `src/lang/subset_search/support.py`
- Test: `src/lang/subset_search/tests/test_support.py`

**Interfaces:**
- Consumes: `ASTNode.function_names() -> set[str]` (`src/lang/ast_nodes.py`; note it includes variable names like `x` and lambda params, hence the pool intersection), `ProgramBank.all_programs()` (Task 2), `passes_quality_filter(fp, min_successes, min_variability)` (`src/lang/enumeration/filters.py`).
- Produces: `support_set(ast, pool_names: frozenset[str]) -> frozenset[str]`, `support_buckets(bank, pool_names, min_successes=3, min_variability=0.3) -> dict[frozenset[str], int]`, `proxy_score(subset: frozenset[str], buckets) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# src/lang/subset_search/tests/test_support.py
"""Tests for support-set extraction and Stage 0 proxy scoring."""

from src.lang.ast_nodes import ApplicationNode, LambdaNode, VariableNode
from src.lang.enumeration.enumerator import BottomUpEnumerator
from src.lang.subset_search.pool import pool_grammar
from src.lang.subset_search.support import (
    support_set,
    support_buckets,
    proxy_score,
)


POOL = frozenset({'+', 'map', 'length'})


def test_support_set_intersects_pool_and_ignores_variables():
    # (map (λ (p) (+ p p)) x) — uses map and + from the pool; x and p are vars
    ast = ApplicationNode(
        VariableNode('map'),
        [
            LambdaNode(['p'], ApplicationNode(
                VariableNode('+'), [VariableNode('p'), VariableNode('p')]
            )),
            VariableNode('x'),
        ],
    )
    assert support_set(ast, POOL) == frozenset({'map', '+'})


def test_support_set_of_bare_variable_is_empty():
    assert support_set(VariableNode('x'), POOL) == frozenset()


def test_proxy_score_sums_contained_buckets():
    buckets = {
        frozenset(): 1,                    # e.g. the identity program
        frozenset({'+'}): 4,
        frozenset({'map', '+'}): 7,
        frozenset({'length'}): 2,
        frozenset({'map', 'length'}): 3,
    }
    assert proxy_score(frozenset({'+', 'map'}), buckets) == 1 + 4 + 7
    assert proxy_score(frozenset({'length'}), buckets) == 1 + 2
    assert proxy_score(frozenset(), buckets) == 1


def test_support_buckets_counts_only_quality_fingerprints():
    g = pool_grammar(('+', 'length', 'take'))
    enum = BottomUpEnumerator(grammar=g, max_size=3)
    bank = enum.enumerate()
    pool = frozenset({'+', 'length', 'take'})
    buckets = support_buckets(bank, pool)
    assert all(s <= pool for s in buckets)
    assert all(n > 0 for n in buckets.values())
    total_quality = sum(buckets.values())
    assert 0 < total_quality <= bank.count()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_support.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `src.lang.subset_search.support`

- [ ] **Step 3: Write the implementation**

```python
# src/lang/subset_search/support.py
"""Support-set extraction and Stage 0 proxy scoring.

The proxy score of a subset S is the number of quality-passing distinct
behaviors whose *witness* program uses only primitives in S. It is a lower
bound on S's true yield (a behavior may be reachable in S via a program other
than the first-found full-pool witness); Stage 0 only ranks, so a consistent
lower bound is acceptable.
"""

from collections import defaultdict

from ..ast_nodes import ASTNode
from ..enumeration.enumerator import ProgramBank
from ..enumeration.filters import passes_quality_filter


def support_set(ast: ASTNode, pool_names: frozenset[str]) -> frozenset[str]:
    """Grammar functions used by a program (variables filtered out by the
    pool intersection — function_names() includes variable references)."""
    return frozenset(ast.function_names()) & pool_names


def support_buckets(
    bank: ProgramBank,
    pool_names: frozenset[str],
    min_successes: int = 3,
    min_variability: float = 0.3,
) -> dict[frozenset[str], int]:
    """Count quality-passing distinct behaviors per witness support set."""
    buckets: dict[frozenset[str], int] = defaultdict(int)
    for prog in bank.all_programs():
        if prog.fingerprint is None:
            continue
        if not passes_quality_filter(prog.fingerprint, min_successes, min_variability):
            continue
        buckets[support_set(prog.ast, pool_names)] += 1
    return dict(buckets)


def proxy_score(
    subset: frozenset[str],
    buckets: dict[frozenset[str], int],
) -> int:
    """Lower bound on the subset's distinct-behavior yield."""
    return sum(n for support, n in buckets.items() if support <= subset)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_support.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/lang/subset_search/support.py src/lang/subset_search/tests/test_support.py
git commit -m "feat(subset_search): support-set extraction and proxy scoring"
```

---

### Task 4: Stage 1 score vector

**Files:**
- Create: `src/lang/subset_search/scoring.py`
- Test: `src/lang/subset_search/tests/test_scoring.py`

**Interfaces:**
- Consumes: `ProgramBank.all_programs()`, `TypedProgram` fields (`ast`, `type`, `fingerprint`, `size`), `passes_quality_filter`, `Fingerprint.values` (tuple, entries may be the `FAIL` sentinel).
- Produces: `score_vector(bank, attempts, max_size, min_successes=3, min_variability=0.3, tail_offset=2, seed=0) -> dict` with keys `n_distinct` (int), `slope` (float), `density` (float), `type_coverage` (int), `spread` (float), `tail` (float), `curve` (list[int], `curve[k-1] == N(k)`). The six scalar keys (all higher-is-better) are the Pareto dimensions; `curve` is informational. Also `PARETO_DIMS: tuple[str, ...]` naming the six scalars.

- [ ] **Step 1: Write the failing test**

```python
# src/lang/subset_search/tests/test_scoring.py
"""Tests for the Stage 1 score vector."""

import pytest

from src.lang.enumeration.enumerator import BottomUpEnumerator
from src.lang.subset_search.pool import pool_grammar
from src.lang.subset_search.scoring import score_vector, PARETO_DIMS


@pytest.fixture(scope="module")
def scored_bank():
    g = pool_grammar(('+', 'length', 'take', 'map'))
    enum = BottomUpEnumerator(grammar=g, max_size=4)
    bank = enum.enumerate()
    vec = score_vector(bank, enum.attempts, max_size=4)
    return bank, enum, vec


def test_score_vector_has_all_pareto_dims(scored_bank):
    _, _, vec = scored_bank
    assert set(PARETO_DIMS) == {
        'n_distinct', 'slope', 'density', 'type_coverage', 'spread', 'tail'
    }
    for dim in PARETO_DIMS:
        assert dim in vec


def test_curve_is_cumulative_and_ends_at_n_distinct(scored_bank):
    _, _, vec = scored_bank
    curve = vec['curve']
    assert len(curve) == 4
    assert all(curve[i] <= curve[i + 1] for i in range(len(curve) - 1))
    assert curve[-1] == vec['n_distinct']


def test_scalar_ranges(scored_bank):
    _, enum, vec = scored_bank
    assert vec['n_distinct'] > 0
    assert vec['slope'] >= 1.0          # cumulative counts cannot shrink
    assert 0.0 < vec['density'] <= 1.0  # distinct/attempted
    assert vec['type_coverage'] >= 1
    assert 0.0 <= vec['spread'] <= 1.0  # normalized Hamming
    assert 0.0 <= vec['tail'] <= 1.0


def test_score_vector_is_deterministic(scored_bank):
    bank, enum, vec = scored_bank
    vec2 = score_vector(bank, enum.attempts, max_size=4)
    assert vec == vec2


def test_empty_bank_gives_zero_vector():
    # Note: any *enumerated* bank contains the input variable x, whose
    # identity fingerprint passes the quality filter — so a truly empty
    # bank must be constructed directly to exercise the zero-guards.
    from src.lang.enumeration.enumerator import ProgramBank
    bank = ProgramBank()
    vec = score_vector(bank, attempts=0, max_size=3)
    assert vec['n_distinct'] == 0
    assert vec['density'] == 0.0
    assert vec['spread'] == 0.0
    assert vec['tail'] == 0.0
    assert vec['curve'] == [0, 0, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_scoring.py -v`
Expected: FAIL with ImportError for `src.lang.subset_search.scoring`

- [ ] **Step 3: Write the implementation**

```python
# src/lang/subset_search/scoring.py
"""Stage 1 score vector: behavioral diversity of a deep-enumerated subset.

All six Pareto dimensions are higher-is-better:
  n_distinct    — quality-passing distinct behaviors at the size bound
  slope         — N(K) / max(1, N(K - tail_offset)): still discovering?
  density       — N(K) / programs attempted: primitives compose non-degenerately
  type_coverage — distinct output type signatures among quality behaviors
  spread        — mean pairwise normalized Hamming distance between fingerprints
  tail          — fraction of behaviors with minimal size >= K - tail_offset
"""

import random

from ..enumeration.enumerator import ProgramBank, TypedProgram
from ..enumeration.filters import passes_quality_filter


PARETO_DIMS: tuple[str, ...] = (
    'n_distinct', 'slope', 'density', 'type_coverage', 'spread', 'tail'
)


def _quality_programs(
    bank: ProgramBank, min_successes: int, min_variability: float
) -> list[TypedProgram]:
    return [
        p for p in bank.all_programs()
        if p.fingerprint is not None
        and passes_quality_filter(p.fingerprint, min_successes, min_variability)
    ]


def yield_curve(programs: list[TypedProgram], max_size: int) -> list[int]:
    """Cumulative distinct-behavior counts; result[k-1] == N(k)."""
    per_size = [0] * (max_size + 1)
    for p in programs:
        per_size[min(p.size, max_size)] += 1
    curve, total = [], 0
    for k in range(1, max_size + 1):
        total += per_size[k]
        curve.append(total)
    return curve


def mean_pairwise_hamming(
    programs: list[TypedProgram],
    sample_size: int = 2000,
    n_pairs: int = 5000,
    seed: int = 0,
) -> float:
    """Mean normalized Hamming distance over sampled fingerprint pairs.

    Positions compared element-wise (FAIL is its own value). Pairs with
    different fingerprint lengths are compared up to the shorter length.
    """
    fps = [p.fingerprint.values for p in programs]
    rng = random.Random(seed)
    if len(fps) > sample_size:
        fps = rng.sample(fps, sample_size)
    if len(fps) < 2:
        return 0.0
    dists = []
    for _ in range(n_pairs):
        a, b = rng.sample(fps, 2)
        n = min(len(a), len(b))
        if n == 0:
            continue
        dists.append(sum(1 for i in range(n) if a[i] != b[i]) / n)
    return sum(dists) / len(dists) if dists else 0.0


def score_vector(
    bank: ProgramBank,
    attempts: int,
    max_size: int,
    min_successes: int = 3,
    min_variability: float = 0.3,
    tail_offset: int = 2,
    seed: int = 0,
) -> dict:
    programs = _quality_programs(bank, min_successes, min_variability)
    curve = yield_curve(programs, max_size)
    n_k = curve[-1] if curve else 0

    earlier_idx = max_size - tail_offset - 1
    n_earlier = curve[earlier_idx] if 0 <= earlier_idx < len(curve) else 0
    slope = n_k / max(1, n_earlier)

    tail_threshold = max(1, max_size - tail_offset)
    tail = (
        sum(1 for p in programs if p.size >= tail_threshold) / n_k
        if n_k else 0.0
    )

    return {
        'n_distinct': n_k,
        'slope': slope,
        'density': n_k / attempts if attempts else 0.0,
        'type_coverage': len({str(p.type) for p in programs}),
        'spread': mean_pairwise_hamming(programs, seed=seed),
        'tail': tail,
        'curve': curve,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_scoring.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/lang/subset_search/scoring.py src/lang/subset_search/tests/test_scoring.py
git commit -m "feat(subset_search): six-dimension Stage 1 score vector"
```

---

### Task 5: Stage orchestration (Stage 0 + Stage 1)

**Files:**
- Create: `src/lang/subset_search/search.py`
- Test: `src/lang/subset_search/tests/test_search.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4; `BottomUpEnumerator(grammar, max_size)`.
- Produces:
  - `run_stage0(out_dir, max_size=7, sizes=(5, 6), pool_names=None) -> list[tuple[str, int]]` — writes `<out_dir>/stage0.json` with `{'max_size', 'pool', 'scores': [[key, score], ...]}` sorted descending by score; returns the scores list.
  - `run_stage1(out_dir, top_n=200, max_size=10, timeout_s=600, workers=None) -> dict[str, dict]` — reads `stage0.json`, deep-evals the top `top_n` subsets in parallel, incrementally writes `<out_dir>/stage1.json` mapping `key -> {'status': 'ok'|'timeout'|'error', 'score': vector, 'max_size': int}`; resumable (skips keys already present).
  - Subset keys are `subset_key()` strings; `key.split(' ')` recovers the names.

- [ ] **Step 1: Write the failing test**

```python
# src/lang/subset_search/tests/test_search.py
"""Integration tests for Stage 0 and Stage 1 on a tiny pool."""

import json
from math import comb
from pathlib import Path

import pytest

from src.lang.enumeration.enumerator import BottomUpEnumerator
from src.lang.subset_search.pool import pool_grammar
from src.lang.subset_search.search import run_stage0, run_stage1
from src.lang.subset_search.scoring import score_vector

TINY_POOL = ('+', '*', 'length', 'take', 'concat', 'map')
TINY_MAX_SIZE = 4


@pytest.fixture(scope="module")
def stage0_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("subset_search")
    run_stage0(str(out), max_size=TINY_MAX_SIZE, sizes=(5,), pool_names=TINY_POOL)
    return out


def test_stage0_scores_all_subsets_sorted(stage0_dir):
    data = json.loads((stage0_dir / "stage0.json").read_text())
    scores = data['scores']
    assert len(scores) == comb(len(TINY_POOL), 5)
    values = [s for _, s in scores]
    assert values == sorted(values, reverse=True)
    assert data['pool'] == list(TINY_POOL)


def test_proxy_is_lower_bound_on_exact_yield(stage0_dir):
    """Spec sanity check: proxy(S) <= exact N(S) at the same size bound."""
    data = json.loads((stage0_dir / "stage0.json").read_text())
    for key, proxy in data['scores'][:3]:
        g = pool_grammar(tuple(key.split(' ')))
        enum = BottomUpEnumerator(grammar=g, max_size=TINY_MAX_SIZE)
        bank = enum.enumerate()
        exact = score_vector(bank, enum.attempts, TINY_MAX_SIZE)['n_distinct']
        assert proxy <= exact, f"{key}: proxy {proxy} > exact {exact}"


def test_stage1_runs_and_is_resumable(stage0_dir):
    results = run_stage1(
        str(stage0_dir), top_n=2, max_size=TINY_MAX_SIZE, timeout_s=120, workers=2,
    )
    assert len(results) == 2
    for key, res in results.items():
        assert res['status'] == 'ok'
        assert res['score']['n_distinct'] >= 0
        assert res['max_size'] == TINY_MAX_SIZE

    on_disk = json.loads((stage0_dir / "stage1.json").read_text())
    assert on_disk == results

    # Resumability: a second call re-runs nothing (all keys present) and
    # returns the same results.
    again = run_stage1(
        str(stage0_dir), top_n=2, max_size=TINY_MAX_SIZE, timeout_s=120, workers=2,
    )
    assert again == results
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_search.py -v`
Expected: FAIL with ImportError for `src.lang.subset_search.search`

- [ ] **Step 3: Write the implementation**

```python
# src/lang/subset_search/search.py
"""Stage orchestration for the primitive subset search.

Stage 0: enumerate the full pool once, proxy-score every subset combinatorially.
Stage 1: exact deep enumeration of the top-N subsets, in worker processes with
a per-subset wall-clock timeout (SIGALRM), results persisted incrementally.
"""

import json
import os
import signal
from concurrent.futures import ProcessPoolExecutor

from ..enumeration.enumerator import BottomUpEnumerator
from .pool import POOL_NAMES, pool_grammar, iter_subsets, subset_key
from .support import support_buckets, proxy_score
from .scoring import score_vector


class SubsetTimeout(Exception):
    """Raised inside a worker when the per-subset wall clock expires."""


def _on_alarm(signum, frame):
    raise SubsetTimeout()


def run_stage0(out_dir, max_size=7, sizes=(5, 6), pool_names=None):
    """One full-pool enumeration; proxy-score every candidate subset."""
    os.makedirs(out_dir, exist_ok=True)
    names = tuple(pool_names) if pool_names is not None else POOL_NAMES

    enum = BottomUpEnumerator(grammar=pool_grammar(names), max_size=max_size)
    bank = enum.enumerate()
    buckets = support_buckets(bank, frozenset(names))
    # A support larger than the biggest subset can never be contained in one.
    buckets = {s: n for s, n in buckets.items() if len(s) <= max(sizes)}

    scores = sorted(
        (
            (subset_key(s), proxy_score(s, buckets))
            for s in iter_subsets(sizes=sizes, pool_names=names)
        ),
        key=lambda kv: (-kv[1], kv[0]),
    )
    with open(os.path.join(out_dir, 'stage0.json'), 'w') as f:
        json.dump({'max_size': max_size, 'pool': list(names), 'scores': scores}, f)
    return scores


def _stage1_worker(task):
    """Deep-enumerate one subset under a SIGALRM wall-clock guard."""
    key, max_size, timeout_s = task
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_s)
    try:
        grammar = pool_grammar(tuple(key.split(' ')))
        enum = BottomUpEnumerator(grammar=grammar, max_size=max_size)
        bank = enum.enumerate()
        vec = score_vector(bank, enum.attempts, max_size)
        return key, {'status': 'ok', 'score': vec, 'max_size': max_size}
    except SubsetTimeout:
        return key, {'status': 'timeout', 'max_size': max_size}
    except Exception as e:  # never let one subset kill the sweep
        return key, {'status': 'error', 'error': repr(e), 'max_size': max_size}
    finally:
        signal.alarm(0)


def run_stage1(out_dir, top_n=200, max_size=10, timeout_s=600, workers=None):
    """Exact deep evaluation of the Stage 0 top-N. Incremental + resumable."""
    with open(os.path.join(out_dir, 'stage0.json')) as f:
        stage0 = json.load(f)

    results_path = os.path.join(out_dir, 'stage1.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    todo = [
        (key, max_size, timeout_s)
        for key, _score in stage0['scores'][:top_n]
        if key not in results
    ]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for key, res in ex.map(_stage1_worker, todo):
                results[key] = res
                with open(results_path, 'w') as f:
                    json.dump(results, f)

    return {
        key: results[key]
        for key, _score in stage0['scores'][:top_n]
        if key in results
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_search.py -v`
Expected: 3 PASSED (module fixture runs one tiny Stage 0; total under ~2 min)

- [ ] **Step 5: Commit**

```bash
git add src/lang/subset_search/search.py src/lang/subset_search/tests/test_search.py
git commit -m "feat(subset_search): Stage 0 proxy sweep and Stage 1 parallel deep eval"
```

---

### Task 6: Pareto front and Stage 2 reports

**Files:**
- Create: `src/lang/subset_search/report.py`
- Test: `src/lang/subset_search/tests/test_report.py`

**Interfaces:**
- Consumes: `PARETO_DIMS` and score vectors (Task 4), `run_stage1` output format (Task 5), `BottomUpEnumerator`, `ProgramBank.all_programs()`, `TypedProgram.ast.pretty_print()`, `DEFAULT_TEST_SUITE` (`src/lang/enumeration/test_suite.py`), `FAIL` sentinel.
- Produces:
  - `pareto_front(vectors: dict[str, dict]) -> list[str]` — keys whose score vectors are not strictly dominated on `PARETO_DIMS`, sorted by descending `n_distinct`.
  - `write_reports(out_dir, max_finalists=10, samples_per_size=5, n_hardest=10) -> list[str]` — reads `stage1.json`, selects the Pareto front (capped at `max_finalists` by `n_distinct`), re-enumerates each finalist at its Stage 1 `max_size`, writes `<out_dir>/reports/<key with spaces replaced by _>.md` and a `<out_dir>/reports/summary.md`; returns finalist keys.

- [ ] **Step 1: Write the failing test**

```python
# src/lang/subset_search/tests/test_report.py
"""Tests for Pareto selection and Stage 2 report generation."""

import json
from pathlib import Path

from src.lang.subset_search.search import run_stage0, run_stage1
from src.lang.subset_search.report import pareto_front, write_reports


def _vec(**overrides):
    base = {
        'n_distinct': 10, 'slope': 1.5, 'density': 0.1,
        'type_coverage': 3, 'spread': 0.5, 'tail': 0.2, 'curve': [1, 5, 10],
    }
    base.update(overrides)
    return base


def test_pareto_front_removes_dominated():
    vectors = {
        'a': _vec(),
        'b': _vec(n_distinct=5, slope=1.0, density=0.05,
                  type_coverage=2, spread=0.3, tail=0.1),   # dominated by a
        'c': _vec(n_distinct=8, spread=0.9),                # trade-off vs a
    }
    front = pareto_front(vectors)
    assert set(front) == {'a', 'c'}
    assert front[0] == 'a'  # sorted by n_distinct descending


def test_pareto_front_keeps_equal_vectors():
    vectors = {'a': _vec(), 'b': _vec()}
    assert set(pareto_front(vectors)) == {'a', 'b'}


def test_write_reports_end_to_end(tmp_path):
    pool = ('+', '*', 'length', 'take', 'concat', 'map')
    run_stage0(str(tmp_path), max_size=4, sizes=(5,), pool_names=pool)
    run_stage1(str(tmp_path), top_n=3, max_size=4, timeout_s=120, workers=2)

    finalists = write_reports(str(tmp_path), max_finalists=2)
    assert 1 <= len(finalists) <= 2

    reports_dir = tmp_path / "reports"
    assert (reports_dir / "summary.md").exists()
    for key in finalists:
        report = (reports_dir / f"{key.replace(' ', '_')}.md").read_text()
        assert "Yield curve" in report
        assert "Sample programs" in report
        assert "Hardest behaviors" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_report.py -v`
Expected: FAIL with ImportError for `src.lang.subset_search.report`

- [ ] **Step 3: Write the implementation**

```python
# src/lang/subset_search/report.py
"""Stage 2: Pareto selection and qualitative markdown reports."""

import json
import os
from collections import defaultdict

from ..enumeration.enumerator import BottomUpEnumerator
from ..enumeration.filters import passes_quality_filter
from ..enumeration.fingerprint import FAIL
from ..enumeration.test_suite import DEFAULT_TEST_SUITE
from .pool import pool_grammar
from .scoring import PARETO_DIMS


def pareto_front(vectors: dict[str, dict]) -> list[str]:
    """Keys not strictly dominated on PARETO_DIMS, by descending n_distinct."""
    def dominates(a: dict, b: dict) -> bool:
        return (
            all(a[d] >= b[d] for d in PARETO_DIMS)
            and any(a[d] > b[d] for d in PARETO_DIMS)
        )

    front = [
        k for k in vectors
        if not any(dominates(vectors[o], vectors[k]) for o in vectors if o != k)
    ]
    return sorted(front, key=lambda k: -vectors[k]['n_distinct'])


def _format_value(v) -> str:
    if v is FAIL:
        return "FAIL"
    if isinstance(v, tuple):
        return "[" + ", ".join(_format_value(x) for x in v) + "]"
    return str(v)


def _finalist_report(
    key: str, vec: dict, max_size: int,
    samples_per_size: int, n_hardest: int,
) -> str:
    enum = BottomUpEnumerator(grammar=pool_grammar(tuple(key.split(' '))),
                              max_size=max_size)
    bank = enum.enumerate()
    quality = [
        p for p in bank.all_programs()
        if p.fingerprint is not None and passes_quality_filter(p.fingerprint)
    ]

    lines = [f"# Subset: `{key}`", ""]

    lines += ["## Score vector", ""]
    for dim in PARETO_DIMS:
        lines.append(f"- **{dim}**: {vec[dim]}")
    lines.append("")

    lines += ["## Yield curve", "", "| size k | N(k) |", "|---|---|"]
    for k, n in enumerate(vec['curve'], start=1):
        lines.append(f"| {k} | {n} |")
    lines.append("")

    lines += ["## Sample programs", ""]
    by_size = defaultdict(list)
    for p in quality:
        by_size[p.size].append(p)
    for size in sorted(by_size):
        lines.append(f"### Size {size}")
        lines.append("")
        for p in by_size[size][:samples_per_size]:
            lines.append(f"- `{p.ast.pretty_print(0, True)}` : `{p.type}`")
        lines.append("")

    lines += ["## Hardest behaviors", "",
              "Behaviors with the largest minimal program size.", ""]
    hardest = sorted(quality, key=lambda p: -p.size)[:n_hardest]
    for p in hardest:
        lines.append(f"### `{p.ast.pretty_print(0, True)}` (size {p.size})")
        lines.append("")
        n_shown = min(len(DEFAULT_TEST_SUITE), len(p.fingerprint.values))
        for inp, out in zip(DEFAULT_TEST_SUITE[:n_shown], p.fingerprint.values):
            lines.append(f"- `{inp}` → `{_format_value(out)}`")
        lines.append("")

    return "\n".join(lines)


def write_reports(
    out_dir: str,
    max_finalists: int = 10,
    samples_per_size: int = 5,
    n_hardest: int = 10,
) -> list[str]:
    with open(os.path.join(out_dir, 'stage1.json')) as f:
        stage1 = json.load(f)

    ok = {k: v for k, v in stage1.items() if v['status'] == 'ok'}
    vectors = {k: v['score'] for k, v in ok.items()}
    finalists = pareto_front(vectors)[:max_finalists]

    reports_dir = os.path.join(out_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    summary = ["# Subset search — Pareto front", "",
               "| subset | " + " | ".join(PARETO_DIMS) + " |",
               "|" + "---|" * (len(PARETO_DIMS) + 1)]
    for key in finalists:
        vec = vectors[key]
        row = " | ".join(
            f"{vec[d]:.4g}" if isinstance(vec[d], float) else str(vec[d])
            for d in PARETO_DIMS
        )
        summary.append(f"| `{key}` | {row} |")
        report = _finalist_report(
            key, vec, ok[key]['max_size'], samples_per_size, n_hardest,
        )
        path = os.path.join(reports_dir, f"{key.replace(' ', '_')}.md")
        with open(path, 'w') as f:
            f.write(report)
    summary.append("")
    summary.append(f"Full Stage 1 results: {len(vectors)} subsets evaluated; "
                   f"{len(finalists)} on the (capped) Pareto front.")

    with open(os.path.join(reports_dir, 'summary.md'), 'w') as f:
        f.write("\n".join(summary))
    return finalists
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_report.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/lang/subset_search/report.py src/lang/subset_search/tests/test_report.py
git commit -m "feat(subset_search): Pareto front and Stage 2 qualitative reports"
```

---

### Task 7: CLI entry point

**Files:**
- Create: `src/lang/subset_search/__main__.py`
- Test: extend `src/lang/subset_search/tests/test_search.py` (append the CLI test below)

**Interfaces:**
- Consumes: `run_stage0`, `run_stage1`, `write_reports`.
- Produces: `python -m src.lang.subset_search --stage {0,1,2}` with options `--out` (default `outputs/subset_search`), `--max-size` (default 7 for stage 0, 10 for stage 1), `--top-n` (default 200), `--timeout` (seconds, default 600), `--workers` (default: executor default), `--max-finalists` (default 10). Exposed as `main(argv=None)` for testing.

- [ ] **Step 1: Write the failing test (append to test_search.py)**

```python
def test_cli_stage0_smoke(tmp_path, monkeypatch):
    from src.lang.subset_search.__main__ import main
    import src.lang.subset_search.search as search_mod

    calls = {}

    def fake_stage0(out_dir, max_size=7, sizes=(5, 6), pool_names=None):
        calls['args'] = (out_dir, max_size, sizes)
        return []

    monkeypatch.setattr(search_mod, 'run_stage0', fake_stage0)
    main(['--stage', '0', '--out', str(tmp_path), '--max-size', '5'])
    assert calls['args'] == (str(tmp_path), 5, (5, 6))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search/tests/test_search.py::test_cli_stage0_smoke -v`
Expected: FAIL with `ModuleNotFoundError` for `src.lang.subset_search.__main__`

- [ ] **Step 3: Write the implementation**

```python
# src/lang/subset_search/__main__.py
"""CLI for the primitive subset search.

    python -m src.lang.subset_search --stage 0            # proxy sweep
    python -m src.lang.subset_search --stage 1 --top-n 200
    python -m src.lang.subset_search --stage 2
"""

import argparse

from . import search


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, required=True, choices=[0, 1, 2])
    parser.add_argument('--out', default='outputs/subset_search')
    parser.add_argument('--max-size', type=int, default=None,
                        help='enumeration size bound (default: 7 stage 0, 10 stage 1)')
    parser.add_argument('--top-n', type=int, default=200)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--max-finalists', type=int, default=10)
    args = parser.parse_args(argv)

    if args.stage == 0:
        scores = search.run_stage0(
            args.out, max_size=args.max_size if args.max_size else 7,
        )
        print(f"Stage 0 done: {len(scores)} subsets scored; "
              f"top: {scores[0] if scores else 'n/a'}")
    elif args.stage == 1:
        results = search.run_stage1(
            args.out,
            top_n=args.top_n,
            max_size=args.max_size if args.max_size else 10,
            timeout_s=args.timeout,
            workers=args.workers,
        )
        n_ok = sum(1 for r in results.values() if r['status'] == 'ok')
        print(f"Stage 1 done: {n_ok}/{len(results)} subsets scored ok")
    else:
        from .report import write_reports
        finalists = write_reports(args.out, max_finalists=args.max_finalists)
        print(f"Stage 2 done: {len(finalists)} finalist reports in "
              f"{args.out}/reports/")


if __name__ == '__main__':
    main()
```

Note the CLI test monkeypatches `search_mod.run_stage0`, so `__main__` must call it as `search.run_stage0(...)` (module attribute), not import the function directly — as written above.

- [ ] **Step 4: Run the full subset_search test suite**

Run: `micromamba run -n ml13 python -m pytest src/lang/subset_search -v`
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add src/lang/subset_search/__main__.py src/lang/subset_search/tests/test_search.py
git commit -m "feat(subset_search): CLI entry point for the three stages"
```

---

### Task 8: Real Stage 0 run + budget calibration

Not a code task — the first production run, validating the max_size=7 budget from the spec ("tune to keep the run under ~an hour").

**Files:**
- None created in git; results land in `outputs/subset_search/`.

- [ ] **Step 1: Time a cheaper full-pool enumeration first**

Run: `micromamba run -n ml13 python -m src.lang.subset_search --stage 0 --max-size 6 --out outputs/subset_search_ms6`
Record the wall-clock time. Bottom-up enumeration cost grows steeply with size; if max_size=6 takes over ~5 minutes, stay at 6 for the real Stage 0. Otherwise proceed to 7.

- [ ] **Step 2: Run the real Stage 0**

Run: `micromamba run -n ml13 python -m src.lang.subset_search --stage 0 --max-size 7`
Expected: completes under ~an hour; prints `Stage 0 done: 27132 subsets scored; top: (...)`. If it blows past the budget, fall back to the max_size=6 results (`cp outputs/subset_search_ms6/stage0.json outputs/subset_search/`).

- [ ] **Step 3: Sanity-check the ranking**

Run: `micromamba run -n ml13 python -c "
import json
d = json.load(open('outputs/subset_search/stage0.json'))
for k, s in d['scores'][:15]: print(s, ' ', k)
print('...')
for k, s in d['scores'][-3:]: print(s, ' ', k)
"`
Expected: top subsets contain at least one higher-order function (map/filter/fold) and a mix of arithmetic + list ops; bottom subsets are degenerate (e.g. all-boolean). If the top looks degenerate, stop and investigate before Stage 1.

- [ ] **Step 4: Report findings to the user before launching Stage 1**

Stage 1 at max_size=10 for 200 subsets is the expensive step (up to 200 × 10 min worst case, embarrassingly parallel). Report Stage 0 timing + top-15 ranking, confirm Stage 1 budget (`--top-n`, `--max-size`, `--timeout`), then run:

`micromamba run -n ml13 python -m src.lang.subset_search --stage 1 --top-n 200 --max-size 10`

followed by:

`micromamba run -n ml13 python -m src.lang.subset_search --stage 2`

---

## Self-Review Notes

- **Spec coverage:** pool (Task 1), support proxy incl. bucket-by-support optimization (Task 3), score vector with all 5 spec components — behavioral spread split into `type_coverage` + `spread` (Task 4), stages/persistence/resume/timeout (Task 5), Pareto + reports with yield curve, samples, hardest behaviors (Task 6), CLI (Task 7), proxy≤exact sanity check (Task 5 test), budget tuning + production run (Task 8). `results/subset_search/` from the spec is realized as `outputs/subset_search/` to match the repo's existing untracked `outputs/` directory.
- **Known behavior, not a bug:** bool-typed programs rarely pass `min_variability=0.3` (≤2 distinct values over 10 tests), so predicates earn their score through lambda positions in `filter`/`fold`, not as top-level behaviors. Spec says reuse thresholds as-is.
- **Multiprocessing:** `_stage1_worker` is module-level (picklable); SIGALRM works per worker process on darwin/POSIX. Enumerator per-size prints will interleave across workers — cosmetic only.
