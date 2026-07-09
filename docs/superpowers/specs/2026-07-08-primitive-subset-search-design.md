# Primitive Subset Search — Design

**Date:** 2026-07-08
**Goal:** Find a set of 5–6 grammar primitives (from a curated pool of ~18) that still
generates *qualitatively interesting* programs, where interesting is operationalized as
intrinsic behavioral diversity of bounded-size enumeration.

## Background

`SmallGrammar` (src/lang/grammar.py) is currently a hand-picked 10-function subset of
`DefaultGrammar`. We want a principled, search-based selection of 5–6 primitives instead.
The existing enumeration machinery does the heavy lifting:

- `BottomUpEnumerator(grammar, max_size=k)` — bottom-up enumeration with observational
  equivalence pruning (src/lang/enumeration/enumerator.py).
- `Fingerprint` / `FingerprintTable` — a program's behavior = its output vector on the
  10-input `DEFAULT_TEST_SUITE` (src/lang/enumeration/fingerprint.py, test_suite.py).
- `passes_quality_filter` — non-crashing, non-constant, minimum variability
  (src/lang/enumeration/filters.py).
- `Grammar.subset(names)` — builds a candidate grammar from a name set.

Integer literals, booleans, lambdas, and the input variable come from the language
itself, not the grammar, so the pool favors *generators* over derivable specializations
(e.g. `is_even` is derivable from `%` + `==` and is excluded).

## Candidate pool (18 primitives)

```
+  -  *  %  <  ==  and  not
singleton  cons  concat  take  drop  length  range
map  filter  fold
```

Subset sizes searched: all 5-subsets and 6-subsets → C(18,5) + C(18,6) = 8,568 + 18,564
= 27,132 candidates.

## Architecture: three stages

### Stage 0 — single-pass proxy scoring (all 27k subsets)

Run `BottomUpEnumerator` **once** over the full 18-primitive pool at a moderate size
bound (start at `max_size=7`; tune to keep the run under ~an hour). Extend fingerprint
recording so that each distinct behavior stores its **support set**: the frozenset of
grammar function names appearing in its first-found witness program (extracted by
walking the witness AST).

Proxy score of a candidate subset S:

```
proxy(S) = #{ fingerprint fp : support(fp) ⊆ S and passes_quality_filter(fp) }
```

Computed combinatorially for all 27,132 subsets from the one enumeration — no
re-enumeration. Implementation detail: bucket fingerprints by support set (there are
few distinct support sets relative to fingerprints), then for each subset sum the
buckets it contains. This is a **lower bound** on the subset's true yield (a subset may
reach a behavior via an alternative program whose first-found witness used other
primitives), acceptable because Stage 0 only ranks.

Output: ranked list of subsets with proxy scores, persisted as JSON.

### Stage 1 — exact deep evaluation (top ~200 subsets)

For each surviving subset, run `BottomUpEnumerator(grammar=pool.subset(S), max_size=K)`
with K as deep as budget allows (target K=9–11; smaller grammars branch less, so they
enumerate deeper for the same budget — which is where their interestingness lives).
Embarrassingly parallel across subsets (multiprocessing pool).

Each run produces a **score vector** (see Scoring below), persisted per subset as JSON.

### Stage 2 — Pareto front + qualitative report (top ~10)

Select the Pareto front of the Stage 1 score vectors (no scalar weighting). For each
finalist, generate a qualitative report:

- the yield curve N(k) for k = 1..K,
- sample programs at each size (pretty-printed),
- the "hardest" behaviors: fingerprints with the largest minimal program size, shown
  as (program, input→output examples) pairs,
- the score vector alongside the pool-wide distribution for context.

Human eyeballs make the final call.

## Scoring vector (Stage 1)

Raw fingerprint count is gameable; each subset gets a vector:

1. **Semantic yield** `N(K)` — distinct fingerprints passing `passes_quality_filter`
   at the size bound.
2. **Yield-curve slope** `N(K) / N(K−2)` — is the grammar still discovering new
   behaviors at the bound? Early flattening = boring basis.
3. **Semantic density** `N(K) / total programs enumerated` — high density means
   primitives compose non-degenerately (few observational collisions).
4. **Behavioral spread** — (a) count of output type signatures covered (list→list,
   list→int, list→bool, …); (b) mean pairwise normalized Hamming distance between
   fingerprint value tuples (positions compared element-wise, FAIL counts as its own
   value), computed on a uniform sample of ≤ 2,000 fingerprints per subset.
5. **Nontriviality tail** — fraction of behaviors whose minimal program size ≥
   threshold (default: K−2). Interesting bases have behaviors reachable only by real
   composition.

Pareto selection uses (1)–(5); ties broken by human review in Stage 2.

## Code layout

New module `src/lang/subset_search/`:

- `pool.py` — the 18-name pool constant and subset generation.
- `support.py` — support-set extraction from witness ASTs; Stage 0 proxy scoring.
- `scoring.py` — the Stage 1 score vector computed from a `ProgramBank`.
- `search.py` — orchestration: stage entry points, multiprocessing, JSON persistence.
- `report.py` — Stage 2 Pareto selection and report generation.
- CLI entry point (e.g. `python -m src.lang.subset_search --stage 0|1|2`), results
  under a git-ignored `results/subset_search/` directory.

Existing code changes are minimal: Stage 0 needs witness ASTs per fingerprint, which
`FingerprintTable` already stores (`fp → ASTNode`); support extraction is a read-only
walk. Minimal size per fingerprint is recoverable from `ProgramBank` (programs are
indexed by size), so no enumerator changes are needed.

## Error handling

- Per-subset enumeration runs in a worker process with a wall-clock timeout
  (default 10 min); timed-out subsets are recorded as failed with a partial score if
  available, never crash the sweep.
- Evaluation crashes inside enumeration are already handled by the FAIL sentinel.
- All stages are resumable: results JSON is written incrementally; re-running a stage
  skips subsets that already have results.

## Testing

- Unit: support-set extraction on hand-built ASTs; proxy scoring on a toy fingerprint
  table; score-vector components on tiny synthetic banks; Pareto selection on known
  vectors.
- Integration: end-to-end Stage 0→1→2 on a tiny pool (e.g. 6 primitives, max_size=4)
  asserting sane, deterministic output.
- Sanity check: proxy(S) ≤ exact N for the same size bound, verified on a few subsets.

## Out of scope

- Downstream trainability evaluation (Model A learning curves) — a possible later
  filter on the finalists, not part of this search.
- Searching the full ~50-function DefaultGrammar.
- Changing the test suite or quality-filter thresholds (reused as-is).
