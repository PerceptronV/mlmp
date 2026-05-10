# Probing for meta-primitive use — v1.1 delta plan

This document records the **deltas** from [`analysis_subpackage_design.md`](analysis_subpackage_design.md) needed to land `ProbingAnalysis`. Read the main plan first; this one only covers what changes / is added.

## 1. Goal

For each transformer method (and any other `EmbeddingMethod` in scope), train a linear probe on per-task encoder embeddings to predict whether MPL's best solution for that task uses each meta-primitive in a fixed vocabulary. Compare per-method probe AUROC across primitives.

The hypothesis being tested (from `.claude/thoughts/research/research_proposal.md`): a meta-learned model's representations should linearly separate tasks by which meta-primitives MPL uses to solve them, *more* than a non-meta-learned baseline's. If the symbol-shuffling model's probe AUROCs systematically beat the in-weight model's on primitives like `Recurse` or `AntiUnify`, that's converging evidence for functionally-meta-primitive structure in the meta-learned representations.

## 2. Empirical facts that shape the design

Confirmed from `mpl_best.csv` (424k rows total, ~5,500 winning rows after best-per-(id,run,order,trial)):

- The `metaprogram` column is non-null for every row (zero nulls in a 200k-row sample).
- It's a parseable dot-separated sequence of `Token(...)` calls, e.g.
  `MemorizeAll.SampleRule.SampleAtom(Some(Operator(Operator(8)))).SampleAtom(None).SampleAtom(Some(Variable(Variable(0)))).Stop`.
- The token vocabulary observed (counts from a 200k-row sample):

  ```
  SampleAtom 1,044,768   Variablize 619,304   Stop 200,000
  MemorizeAll 153,026    MemorizeDatum 133,892   AntiUnify 81,021
  RegenerateRule 60,217  RegenerateThisRule 60,217  RegenerateThisPart 60,217  RegenerateThisPlace 60,217
  SampleRule 58,732      DeleteRule 49,658
  Recurse 29,478         Generalize 2,781   Compose 2,096
  ```

- Per-`(id, run, order)` MCMC produces **different best metaprograms across replicates** for the same trial. Verified on `c001`: three distinct winning metaprograms across `(order, run)` pairs at `trial=11`. Treat the label as a *distribution over (run, order) replicates*, not a single ground truth.

## 3. Vocabulary mapping (paper meta-primitives → logged tokens)

This is a judgement call worth pinning down once and committing to:

| Paper meta-primitive | Logged token(s) used as label | Notes |
|---|---|---|
| `MemorizeAll` | `MemorizeAll` | clean 1-to-1 |
| `Memorize p ψ` | `MemorizeDatum` | clean 1-to-1 |
| `AntiUnify` | `AntiUnify` | clean 1-to-1 |
| `Recurse` | `Recurse` | clean 1-to-1 |
| `Variable` | `Variablize` | 1-to-1; rename in code to `Variable` to match the paper |
| `Compose` | `Compose` | rare (~1% of tokens) — drop from probe vocab if base rate < 5% post-filter |
| `Subproblem` | *(none)* | not directly logged; **excluded** from v1.1 |
| `Delete` | `DeleteRule` | 1-to-1 |

Excluded from labels (search-control, not meta-primitives): `SampleAtom`, `SampleRule`, `Stop`, `RegenerateRule`, `RegenerateThisRule`, `RegenerateThisPart`, `RegenerateThisPlace`, `Generalize`.

The mapping lives as a module constant `META_PRIMITIVE_VOCAB` in `methods/mpl.py` (a `dict[str, tuple[str, ...]]` so we can re-map if Rule's logging conventions change).

## 4. Filter-to-MPL-acquired tasks

Probe is trained only on tasks where MPL acquired the rule (i.e. correct on every trial ≥ *n* for some *n* ≤ 11 — Rule's strict criterion — on at least one of the 25 (run × order) replicates). Rationale:

- For tasks MPL fails to solve, the "best" metaprogram is a posterior sample from a chain that never found a high-likelihood region. Its meta-primitive content is mostly noise.
- Reproducing Rule Fig 3A from CSVs gives MPL (500K) ≈ 73/100 tasks acquired by trial 11 in our pipeline. The paper's curve sits in the same band. So expect N ≈ 70-75 tasks for the probe, not 50-60 as an earlier draft of this doc claimed.
- N must be reported alongside every result (and the per-primitive base rate). A probe AUROC of 0.85 on a primitive present in 90% of tasks is uninteresting; one of 0.85 on a primitive present in 30% is.

**Source of truth.** As of the v1 standardisation, `RuleAcquisitionAnalysis` already produces `outputs/analysis/<run_name>/rule_acquisition/acquired.parquet` with columns `(method, function, order, acquired_on)`, where `acquired_on ∈ [1, 11]` is Rule's strict criterion (smallest *n* such that `correct` is True on every trial ≥ *n*) and `12` is the never-acquired sentinel. **Probing reuses this parquet** rather than recomputing — duplicating the criterion creates two slightly-divergent definitions of "acquired" over time.

Operational definition: a task is *MPL-acquired* iff `acquired.parquet` filtered to `method == mpl` has at least one `(function, order)` row with `acquired_on ≤ 11`. The strict criterion is computed from `Prediction.correct` (binary), not from the continuous `accuracy` column; accuracy continues to be the metric for everything else (per-trial means, pairwise Pearson). See main plan §7.1 for the binary-vs-continuous split.

`MPLBestMethod.acquired_tasks()` reads the parquet (cached at `outputs/analysis/<run_name>/rule_acquisition/acquired.parquet`) and returns the filtered set. If the parquet doesn't exist yet (e.g. probing run before rule_acquisition), the method either errors with a clear message pointing the user at the dependency or falls back to recomputing from `mpl.csv` (decide at build time; erroring is cleaner).

## 5. Code deltas from the main plan

### 5.1 `methods/mpl.py` (extend, not new)

Add:

```python
META_PRIMITIVE_VOCAB: dict[str, tuple[str, ...]] = {
    "MemorizeAll": ("MemorizeAll",),
    "Memorize":    ("MemorizeDatum",),
    "AntiUnify":   ("AntiUnify",),
    "Recurse":     ("Recurse",),
    "Variable":    ("Variablize",),
    "Compose":     ("Compose",),
    "Delete":      ("DeleteRule",),
}

class MPLBestMethod(CSVMethod):
    ...

    def acquired_tasks(self, rule_acq_dir: Path | None = None) -> set[str]:
        """Task IDs where ≥1 (function, order) row in
        ``rule_acquisition/acquired.parquet`` (filtered to ``method == mpl``)
        has ``acquired_on ≤ 11``. Reads the parquet rather than recomputing
        so the strict-acquisition criterion lives in one place
        (RuleAcquisitionAnalysis, main plan §7.1).

        ``rule_acq_dir`` defaults to ``self.root / ".." / "rule_acquisition"``
        following the standard ``outputs/analysis/<run_name>/`` layout."""

    def metaprimitives_for(
        self,
        task_id: str,
        *,
        trial: int = 11,
        vocab: dict[str, tuple[str, ...]] = META_PRIMITIVE_VOCAB,
    ) -> np.ndarray:
        """Multi-hot label matrix of shape (n_replicates, len(vocab)) for the
        given task. Each row is one (run, order) replicate's best metaprogram
        at the given trial; entry [r, p] = 1 iff any token in vocab[p]
        appears in that metaprogram. Returns empty (0, len(vocab)) if the
        task is not MPL-acquired."""
```

A small `_parse_metaprogram(s) -> list[str]` helper splits on `.` and extracts the alphabetic prefix from each part; lives next to the vocab. ~50 LOC + tests.

### 5.2 `metrics/probing.py` (replace stub)

```python
@dataclass
class ProbingResult(AnalysisResult):
    auroc:  pd.DataFrame   # (method, primitive, fold, auroc)
    null:   pd.DataFrame   # (method, primitive, perm_idx, auroc)  — label-shuffle null
    base:   pd.DataFrame   # (primitive, base_rate, n_acquired_tasks)
    config: dict
    def save(self, outdir): ...
    def plot(self, outdir): ...

class ProbingAnalysis(Analysis):
    def __init__(
        self,
        methods: list[str],          # subset that must implement Capability.EMBEDDINGS
        primitives: list[str] | None = None,   # default: all in META_PRIMITIVE_VOCAB
        n_folds: int = 5,
        n_perm: int = 200,
        embedding_pool: Literal["mean", "last"] = "mean",
        n_io_shown: int = 11,
        order: int = 1,
        label_aggregation: Literal["majority", "soft"] = "soft",
        majority_threshold: float = 0.5,    # fraction of replicates that must use a primitive for label=1
    ): ...
    def run(self, methods, bundle, cache) -> ProbingResult: ...
```

Logic:

1. Find `MPLBestMethod` in the method list (error if absent). Compute `acquired = mpl.acquired_tasks()`. Reduce `bundle.tasks` to `bundle.tasks_subset(acquired)`. Log `N = len(acquired)`.
2. For each `(method, task_id)`, embed once with `(n_io_shown, order)` → matrix `X[method] ∈ R^{N × d}`.
3. For each `task_id`, build label vector `y[task_id] ∈ [0, 1]^|primitives|` by aggregating `mpl.metaprimitives_for(task_id)` across replicates (i.e. `y[p]` = fraction of the (run, order) replicates whose best metaprogram contains a token in `META_PRIMITIVE_VOCAB[p]`).

   `majority_threshold ∈ (0, 1)` (default `0.5`) is the cutoff turning `y` into a binary label `b = (y ≥ majority_threshold)`. This threshold applies to:
   - the training labels under `label_aggregation: "majority"` mode,
   - the stratified-CV split key (so folds are balanced on `b`),
   - the AUROC ground truth in both `majority` *and* `soft` modes.

   Why a hparam: with 25 (run × order) replicates per task, MPL frequently uses a primitive in some-but-not-all chains. The default `0.5` asks "is the probe finding tasks where MPL uses this primitive most of the time?", which is the standard probing question. Raising it (e.g. `0.7`) asks the stricter version "is the probe finding tasks where MPL uses this primitive *consistently* across MCMC runs?", which is more conservative and trims out tasks where MPL flip-flops. Lowering it (e.g. `0.3`) asks "does the probe find tasks where the primitive shows up at all?", which is sensitive to rare primitives like `Compose` that might never cross 50% but still carry signal. Pick once per study and report the value alongside results — sweeping the threshold mid-experiment is p-hacking.

   - `majority`: train on hard binary labels `b`, equal sample weights. Standard probing-literature convention; useful as a sanity reference but throws away MCMC uncertainty (under `majority_threshold=0.5`, a primitive used in 14/25 replicates and one used in 25/25 both get label 1).
   - `soft`: train against `y` directly. The probe learns `P(primitive used | embedding)` instead of "consistently-uses-it". Implemented via the duplicate-with-weights trick — for each example emit two rows `(x, label=1, weight=y)` and `(x, label=0, weight=1-y)` — which is mathematically identical to minimizing BCE against the fractional `y`. Sklearn's `LogisticRegression` accepts the resulting `sample_weight` natively. This is the unbiased estimator and the recommended default; `majority` exists for comparison.

   AUROC evaluation always uses `b` for ground truth (rounded by `majority_threshold`) — AUROC is a ranking metric over the probe's scores, decoupled from the training target. Training-vs-evaluation labels can therefore differ without contradiction.

   **Base-rate sanity.** After applying `majority_threshold`, drop any primitive whose positive base rate is below 5% or above 95% (too few examples on one side to fit a probe meaningfully). Log a one-line warning when this trims the primitive list. The default 5% threshold matches the existing rule for dropping `Compose` if it's too rare (§3, ~1% of tokens).
4. For each `(method, primitive)`:
   - 5-fold CV with stratified split on `b = (y ≥ majority_threshold)` (used regardless of `label_aggregation` — stratification needs a binary key). The same `b` is the AUROC ground truth.
   - Logistic regression with L2 (sklearn) on standardised embeddings, fit with the per-mode targets and weights from step 3.
   - Report cross-validated AUROC + per-fold scores.
5. Run baselines (§6).
6. Statistical tests:
   - Within-method: AUROC vs label-shuffle null (one-tailed).
   - Across-method per-primitive: paired Wilcoxon over folds (or DeLong's test if you want a more principled AUROC comparison).
7. Plot:
   - Heatmap (method × primitive → AUROC).
   - Per-primitive grouped bars across methods, with null-distribution dashed lines and base-rate annotations.
   - N-and-base-rate sidecar table baked into the heatmap caption.

Estimated ~250–350 LOC for the analysis itself, ~50 LOC for tests.

### 5.3 `methods/transformer.py` (no change)

Already exposes `embed(trial)` per the main plan. The probe doesn't need new transformer-side code.

Implementation note: `TransformerMethod.embed` is a thin wrapper over `src/data/program_io.py:ProgramIO.encode_pool`, the single source of truth for transformer-side tokenisation / decode / pooling. If a v1.1 stretch ever needs raw encoder output (e.g. token-level probes instead of pooled task-level), reach into `ProgramIO` rather than re-implementing.

### 5.4 `stats.py` (extend)

`stats.py` already houses bootstrap CIs, `pearson_with_ci` (added during the v1 rule-acquisition standardisation), `spearman_with_ci`, paired Wilcoxon, log-rank, ARI/NMI, χ² + FDR. Probing only needs one new entry: `delong_auroc_test(y_true, scores_a, scores_b) -> p_value` for principled paired-AUROC comparisons across methods. Paired-Wilcoxon over per-fold AUROCs is the cheap fallback if we don't want to add the dependency.

### 5.5 YAML schema (extend)

```yaml
analyses:
  - kind: probing
    methods: [tx_in_weight, tx_easy_shuf, tx_hard_shuf]
    mpl_method: mpl              # name of the MPLBestMethod entry
    primitives: [MemorizeAll, Memorize, AntiUnify, Recurse, Variable, Delete]   # Compose dropped if base rate < 5%
    n_folds: 5
    n_perm: 200
    label_aggregation: soft       # "soft" (BCE-against-y, recommended) or "majority" (round at threshold)
    majority_threshold: 0.5       # fraction of replicates needed for label=1; also AUROC ground truth + CV stratification key
    embedding_pool: mean
    n_io_shown: 11
    order: 1
```

## 6. Required baselines

Probing-paper hygiene says you can't interpret a number without these three baselines:

1. **Label-shuffle null.** For each (method, primitive), retrain the probe on shuffled labels `n_perm = 200` times. Report the empirical p-value (fraction of null AUROCs ≥ true AUROC) and the 95th-percentile null AUROC as the "chance" line on bar charts.

2. **Per-primitive base rates.** For each primitive, log the fraction of acquired tasks with the positive label *under the chosen `majority_threshold`*. A probe at AUROC 0.7 on a primitive with 20% base rate is more informative than the same AUROC on a 80%-base-rate primitive. Always plotted as an annotation on the bar. Bake the threshold into the figure caption so a reader doesn't have to dig through the config to know what "positive" meant.

3. **Surface-feature baseline.** Train an identical probe on a *non-embedding* feature vector for each task: bag-of-tokens over the encoder input for that task (or just `functions.csv`'s feature columns). If the surface baseline matches the embedding probe, the embedding contributes no information beyond what's already in the surface form — that's a finding worth reporting honestly.

The `n_perm = 200` permutation pass is the dominant cost; ~200 × n_methods × n_primitives × 5 folds = ~50k logistic regressions per run. At ~1ms each that's 50s; tolerable. Cache the null distribution per `(method, primitive)` on disk so re-plotting is free.

## 7. Verification (v1.1 done-when)

1. `python -m src.analysis configs/probing.yaml` runs to completion against checkpoints listed in `methods:` and outputs `outputs/analysis/<run_name>/probing/{auroc.parquet, null.parquet, base.parquet, heatmap.png, bars_per_primitive.png}`.
2. `MPLBestMethod.acquired_tasks()` returns a set of size matching the in-pipeline reproduction of Rule Fig 3A within ±2 functions. The csv_only_smoke run (May 2026) produces 73/100 acquired-by-trial-11 for MPL (500K), so expect N ≈ 70–75. Cross-check against `outputs/analysis/<run_name>/rule_acquisition/acquired.parquet` directly — if probing's count diverges from a one-line `(acquired_on ≤ 11).any()` aggregate over that parquet, the wrapper has drifted from the source of truth.
3. `MPLBestMethod.metaprimitives_for("c003")` returns a `(25, |vocab|)` array (5 runs × 5 orders) where the column for `Memorize` has at least one 1 (sanity: c003 is memorisation-heavy by inspection of the program).
4. Label-shuffle null AUROCs concentrate around 0.5 ± 0.05 (sanity: probe is not data-leaking).
5. The surface-feature baseline on `functions.csv` features achieves above-chance AUROC for primitives strongly correlated with feature columns (e.g. `Recurse` ↔ `recursive`), and the embedding probe matches or exceeds it.
   - **Pre-condition:** the loader fix in `task.py` (extended `_TRUE`/`_FALSE` for the `y/n` shorthand) must be in place; before that fix the `recursive` and `counting` columns were silently dropped from `feature_cols`, which would have made this baseline vacuous (every task gets `recursive=False`). See main plan §3a / §14 item 13.
6. With `n_perm=0` (skip null) and `n_folds=2` (fast smoke), the analysis completes in under 30 seconds end-to-end on a single GPU.

## 8. Open decisions for v1.1 build time

- **Subproblem.** Currently excluded; if Rule's logging is updated or we find a proxy in the metaprogram tuple (e.g. specific `Variablize` argument shapes), reconsider.
- **`Compose` retention.** Will likely fall below the 5% base-rate threshold post-filter. Either drop, or merge with `AntiUnify` into a coarser "structural-rewrite" macro-label.
- **`majority_threshold` value.** Default `0.5` matches probing-literature convention. `0.6`–`0.7` may be worth checking once we have empirical replicate distributions per primitive — if most primitives have either ≥80% or ≤20% replicates per task (i.e. MPL is consistent), the threshold barely matters; if the distribution is uniform around 0.5, results will be threshold-sensitive and we should report a sweep over `{0.4, 0.5, 0.6, 0.7}` rather than a single value. Decide after one pilot run with default settings.
- **DeLong vs Wilcoxon for cross-method tests.** DeLong is the principled paired-AUROC test but adds a stats dependency; Wilcoxon over per-fold AUROCs is OK if folds are stratified consistently across methods.
- **Multi-trial probing (stretch²).** Probe at `n_io_shown ∈ {1, 3, 5, 7, 11}` to ask: at what context length do meta-primitive distinctions emerge in the embedding space? Trivially supported by the existing `n_io_shown` config knob; just becomes a 5×-fanout output.

## 9. LOC budget

| Component | New LOC | Test LOC |
|---|---:|---:|
| `methods/mpl.py` extensions (vocab, parser, `acquired_tasks`, `metaprimitives_for`) | 90 | 60 |
| `metrics/probing.py` (replace stub) | 300 | 80 |
| `stats.py` extensions (DeLong optional) | 40 | 20 |
| Surface-feature baseline | 50 | 20 |
| YAML schema + config validation | 20 | 10 |
| Plotting | 80 | — |
| **Total** | **~580** | **~190** |

So: ~3 days of focused work assuming no scope creep, dominated by getting the metaprogram parser right and choosing the right baselines.
