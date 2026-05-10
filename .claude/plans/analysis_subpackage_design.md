# `src/analysis` subpackage — design plan

## 1. Goal

Compare *algorithmic strategies* across program-induction methods on Rule et al.'s 100-list-function benchmark. Methods include:

- Transformers trained under three curricula (`in-weight`, `easy-symbol-shuffling`, `symbol-shuffling`), selectable by run name + checkpoint mode (`best_acc`, `best_loss`, `latest`, `epoch_<N>`).
- Humans (Rule's `predictions.csv`).
- Symbolic / neurosymbolic baselines from Rule's CSVs: MPL, Fleet, Codex, Enumerate, Metagol, RobustFill (and `*_best.csv` variants where applicable).

Four core analyses in scope for v1:

1. **Rule-acquisition efficiency** — accuracy as a function of I/O examples observed (trial 1..11), reported via Rule's strict acquisition criterion (acquired on trial *n* iff correct on all trials ≥ *n*; sentinel = 12) and as a per-trial mean.
2. **Failure modes** — per-task accuracy stratified by the boolean feature columns of `functions.csv` (`recursive`, `higher`, `conditional`, `arithmetic`, `mapping`, `filtering`, `indexing`, `unfolding`, `counting`, `uniqueness`, …), compared across methods.
3. **Clustering of tasks** — by per-task final-layer encoder representation (PCA / t-SNE / UMAP, then k-means / hierarchical), with cluster characterisation against `functions.csv` features and cross-method ARI/NMI.
4. **Error similarity to humans** — per-(function, order, trial) match probability against the human response distribution, aggregated to a length-100 human-likeness profile per method, compared across methods by Spearman.

One analysis kept as **stretch** (plumbing acknowledged in the design but not built in v1; see `.claude/plans/probing.md` for the full delta plan):

- **Probing for meta-primitive use** — linear probe on per-task embeddings predicting MPL's meta-primitive usage (AntiUnify, Recurse, Compose, …) from MPL's solutions, filtered to MPL-acquired tasks and reported with N.

The pipeline is modular along three axes — **method**, **analysis**, **visualisation** (visualisation is folded into each analysis result; see §3) — and YAML-configurable for replicability. Outputs land under `outputs/analysis/<run_name>/`.

## 2. Module tree

```
src/analysis/
  __init__.py
  cli.py                 # python -m src.analysis path/to/config.yaml
  config.py              # PyYAML loader + @dataclass tree (no pydantic)
  task.py                # Task / Trial / TaskBundle (functions.csv + stimuli.csv)
  capability.py          # Capability flag enum + CapabilityMissing exception
  cache.py               # per-method parquet + npz, mtime-fingerprinted
  symbol_map.py          # extracted helper for fn-name (un)mapping
  stats.py               # bootstrap, Pearson, Spearman, paired Wilcoxon, log-rank, ARI/NMI, χ² + FDR
  plotting.py            # shared mpl rcParams, palette, save_fig
  methods/
    __init__.py          # 10-line dict mapping config "kind" → class
    base.py              # Method ABC + Prediction dataclass
    csv_method.py        # CSVMethod base (column-mapping ClassVars)
    transformer.py       # TransformerMethod (live model + cache)
    human.py             # HumanMethod (predictions.csv) — exposes subjects()
    mpl.py               # MPLMethod, MPLBestMethod
    fleet.py             # FleetMethod, FleetBestMethod
    codex.py             # CodexMethod (filters on `source` ∈ {G, P})
    enumeration.py       # EnumerationMethod
    metagol.py           # MetagolMethod
    robustfill.py        # RobustFillMethod
  analyses/
    __init__.py          # dict mapping config "kind" → class
    base.py              # Analysis ABC + AnalysisResult ABC (.save, .plot)
    rule_acquisition.py
    failure_modes.py
    clustering.py
    error_similarity.py
    probing.py           # stretch (skeleton only in v1; see probing.md)
  configs/
    example.yaml
```

~15–17 small files. **Explicitly not building**: plugin/registry decorators, DI containers, generic `ExperimentRunner`/`Pipeline`/`Step`, async/multiprocessing, Hydra/OmegaConf, pydantic, separate `Visualizer`, separate `Comparison` layer, `Method.fit()`, live MPL re-runner.

## 3. Why no separate `comparisons/` or `viz/` layer

Cross-method statistics (Pearson r on difficulty profiles, Spearman ρ on human-likeness profiles, ARI between clusterings, paired Wilcoxon, log-rank) are first-class outputs of each analysis, not a separate stage. Each `AnalysisResult` carries:

- a **per-method tidy table** (e.g. `(method, function, run, trial, ...) → accuracy`),
- a **pairwise comparison table** computed during `Analysis.run` (e.g. Pearson r + bootstrap CI per method pair),
- `.save(outdir)` writing both to parquet/csv,
- `.plot(outdir)` writing the figures the analysis is responsible for.

This preserves statistical rigour and reuse without adding modules. `stats.py` houses the shared estimators (bootstrap CIs, Pearson, Spearman, paired Wilcoxon, log-rank, ARI/NMI, χ² with FDR, sparse logistic regression).

**Design note: Pearson vs Spearman, what slice.** Rule et al. report cross-method correlations on **per-function mean accuracy averaged across all 11 trials** (length-100 vector) using **Pearson r** (Supp. Fig. 6, e.g. humans-MPL r ≈ 0.715). `RuleAcquisitionAnalysis` matches this exactly. An earlier draft of this design said "trial-11 difficulty profile, Spearman ρ" — that compresses variance once MPL/Fleet plateau and tied 0/1 entries deflate the rank correlation, so the numbers were systematically lower than the paper's. Reverted to all-trial Pearson. `ErrorSimilarityAnalysis` keeps Spearman because it correlates **rank** orderings of human-likeness across functions, not raw means; the paper does not report a directly comparable number there.

## 3a. Source-data hygiene

The OSF release (`src/data/osf_gq2hj/`) is **read-only** for the analysis pipeline — we never edit upstream CSVs to fix typos or normalize encodings. Quirks are absorbed in the loader instead.

The known quirk affecting v1: `functions.csv` mixes boolean encodings across columns. Most use Python-bool `True/False`; `recursive` and `counting` use upper-case `TRUE/FALSE` plus a handful of `"y"/"n"` shorthand cells (5 across the 250 functions, 5 in c001..c100). `_TRUE`/`_FALSE` in `task.py` accept both casings and the `y/n/yes/no` abbreviations, so feature columns with stray-but-recoverable values aren't silently dropped. `TaskBundle.load` emits a one-line `WARNING` per column when it encounters non-canonical values (anything outside `{TRUE, True, true, 1, FALSE, False, false, 0}`), so the typo remains visible in logs even though the column parses.

If a future column shows a *new* category of quirk, prefer extending the loader (and bumping the warning) over editing OSF data. Genuinely free-text or numeric columns still skip cleanly because `vals.issubset(_TRUE | _FALSE)` fails.

## 4. Core data model

```python
# task.py
@dataclass(frozen=True)
class Trial:
    task_id: str                                  # "c001"
    order: int                                    # 1..5
    trial: int                                    # 1..11
    observed_examples: tuple[tuple[list[int], list[int]], ...]  # the first (trial-1) IO pairs
    query_input: list[int]
    expected_output: list[int]

@dataclass(frozen=True)
class Task:
    task_id: str
    program: str
    gloss: str
    features: dict[str, bool]                     # ~60 boolean columns from functions.csv
    trials: tuple[Trial, ...]                     # 5 orders × 11 trials = 55 per task

class TaskBundle:
    tasks: dict[str, Task]
    @classmethod
    def load(cls, root: Path) -> "TaskBundle": ...
    def iter_trials(self) -> Iterator[Trial]: ...

# capability.py
class Capability(Flag):
    PREDICTIONS = auto()
    EMBEDDINGS  = auto()
    EFFORT      = auto()                          # cpu / count / lposterior etc.

class CapabilityMissing(Exception):
    def __init__(self, method_name: str, cap: Capability): ...

# methods/base.py
@dataclass
class Prediction:
    response: list[int] | None
    program: str | None
    correct: bool
    effort: dict | None                           # {"cpu": float, "count": int, "lposterior": float, ...} or None

class Method(ABC):
    name: str
    capabilities: ClassVar[Capability]

    @abstractmethod
    def predict(self, trial: Trial) -> Prediction: ...

    def embed(self, trial: Trial) -> np.ndarray:
        raise CapabilityMissing(self.name, Capability.EMBEDDINGS)

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities
```

Analyses use `method.supports(Capability.X)` for feature gating and `try/except CapabilityMissing` for runtime catches; one warning per (method, capability) pair, never per-trial.

## 5. TransformerMethod adapter

Reuses, in order:

- `src/data/dataloader.py:270` `ProgramDataset.tokenise_program_item` — single source of truth for encoder input formatting. Instantiate `ProgramDataset` purely as a tokeniser/grammar handle (corpus path comes from the checkpoint's stored `args`), then call `tokenise_program_item(program_str=task.program, io_pairs=trial.observed_examples + ((trial.query_input, trial.expected_output),)[:trial.trial], name_map=...)` and slice off the program target portion to get encoder input only.
- `src/data/dataloader.py:231` `_sample_name_map`, `:240` `_sample_partial_name_map` — sampled deterministically per-trial via `_episode_rng` when `mode in {symbol-shuffling, easy-symbol-shuffling}`.
  - For `easy-symbol-shuffling`, K is derived from the checkpoint's epoch via `src/train.py:358` `_easy_shuffle_k_for_epoch(epoch, args, n_total_fns)` so eval matches the K active when the checkpoint was saved (per design decision).
- `src/train.py:131` `greedy_decode` for autoregressive prediction.
- `src/train.py:163` `_check_program_match` as a reference for reverse-mapping permuted fn names back to canonical ones; the reverse-map helper is **factored out** into `src/analysis/symbol_map.py` and shared between `TransformerMethod` and `train.py` (small refactor PR alongside this work).
- `src/lang/parser.py` `parse` + `src/lang/compiler.py` `JITCompiler` to compile the predicted program and execute on `trial.query_input`. Apply `% 100` to numeric outputs (project convention; see `project_int_mod`). Wrap execution with a small SIGALRM timeout (mirroring `_alarm` in `train.py`).
- `src/train.py:325` `load_checkpoint` for state restoration. Build `Seq2SeqTransformer` with the constructor args present in `ckpt['args']`.

**Constructor**:

```python
TransformerMethod(
    name: str,
    run_name: str,
    mode: Literal["in-weight", "symbol-shuffling", "easy-symbol-shuffling"],
    ckpt_select: Literal["latest", "best_loss", "best_acc"] | str = "best_acc",
                              # str supports "epoch_<N>"
    device: str = "cuda",
    max_program_tokens: int = 80,
    embedding_pool: Literal["mean", "last", "attn"] = "mean",   # design decision: configurable, default = mean
    corpus: Path | None = None,
)
```

Validate `ckpt['args']['mode'] == mode` and error clearly otherwise.

**`embed(trial)`** returns `(d_model,)`. Pool the final-layer post-norm output of `Seq2SeqTransformer.encode` (`src/models/seq2seq.py:274`), mode-controlled by `embedding_pool`:

- `mean` (default): average over the encoder sequence.
- `last`: take the final encoder position.
- `attn`: a learned single-query attention pool — implemented as a deterministic key-norm-weighted average so we don't introduce trained parameters at inference time. (Mostly a slot for future experimentation; v1 ships `mean` and `last`.)

**Capabilities**: `PREDICTIONS | EMBEDDINGS`. (`EFFORT` is not produced; we don't have search-step counts for transformers.)

## 6. CSV-backed methods

One shared base, thin per-file subclasses (5–15 lines each). Justified because each CSV has a different schema (`response` vs `predictions`, presence of `program`/`cpu`/`count`/`lposterior`, Codex's two `source` rows per trial, `*_best.csv` vs full files), so per-source column-mapping ClassVars are clearer than parameterising a single mega-class via YAML.

```python
class CSVMethod(Method):
    capabilities = Capability.PREDICTIONS | Capability.EFFORT
    csv_filename: ClassVar[str]
    join_keys:    ClassVar[tuple[str, ...]] = ("id", "order", "trial")
    response_col: ClassVar[str] = "response"
    program_col:  ClassVar[str | None] = "program"
    effort_cols:  ClassVar[tuple[str, ...]] = ()
    correct_col:  ClassVar[str] = "accuracy"

    def __init__(self, name, root, **filters): ...   # filters select run/source/mode/lesion/steps rows
    def predict(self, trial) -> Prediction: ...      # O(1) dict lookup on join_keys
```

- `HumanMethod`: capabilities = `PREDICTIONS` only; aggregates over `subject` and exposes `subjects()` for stratified analyses; `predict(trial)` returns one `Prediction` aggregated across subjects (mean accuracy, modal response). **At load time applies Rule et al.'s `identify_excluded_subjects` filter (`analysis.R:50`): exclude any subject with `max_same ≥ 20`, `sum(accuracy) ≤ 10`, or `total_time_s < 1200`.** This drops 106/498 subjects on Rule's data and reproduces the paper's headline 392-subject sample with mean accuracy 0.521 (95% CI [0.479, 0.559]) on the first 100 functions. `Prediction.effort["mean_correct"]` carries the un-thresholded subject-mean accuracy; analyses that need a continuous accuracy use this rather than `Prediction.correct` (which is `mean_correct ≥ 0.5`).
- `MPLMethod`/`MPLBestMethod`, `FleetMethod`/`FleetBestMethod`: pointed at full vs `_best.csv`.
- `CodexMethod`: filters `source ∈ {"G", "P"}` (greedy vs pass@50); pick via filter dict.
- `EnumerationMethod`, `MetagolMethod`, `RobustFillMethod`: straightforward.

Filters propagate from YAML (`filters: {steps: 500000, mode: Online, lesion: None}`).

## 7. Analyses

Each `Analysis` is a class:

```python
class Analysis(ABC):
    @abstractmethod
    def run(self, methods: list[Method], bundle: TaskBundle, cache: Cache) -> AnalysisResult: ...

class AnalysisResult(ABC):
    def save(self, outdir: Path) -> None: ...     # write parquet/csv
    def plot(self, outdir: Path) -> None: ...     # write figures
```

### 7.1 `RuleAcquisitionAnalysis`

Per-method outputs:

- `(method, function, run, trial) → (correct ∈ {0,1}, accuracy ∈ [0, 1])` long-form table. `correct` is binarized for Rule's strict-acquisition criterion; `accuracy` is the un-thresholded run/subject-mean from `Prediction.effort["mean_correct"]` (falls back to `float(correct)` if a method doesn't expose it). All continuous-valued aggregates (per-trial mean, per-function profile for Pearson, violin) read `accuracy`. The strict-acquisition trial reads `correct`.
- **Strict acquisition trial** `acquired_on[method, function, run, order] ∈ [1, 12]`, where 12 = never. (Rule's criterion: rule acquired on trial *n* iff correct on every trial ≥ *n*.)
- **Per-trial mean accuracy** `acc[method, trial] ∈ [0, 1]` with bootstrap CI at run/subject level. Computed from `accuracy`, not `correct`.

Cross-method outputs (computed during `run`):

- Pearson r between length-100 difficulty profiles (per-function mean accuracy averaged across **all 11 trials**), with paired-bootstrap CIs. Matches Rule Supp. Fig. 6 (humans-MPL r ≈ 0.715).
- Paired Wilcoxon signed-rank on the same length-100 vector of mean accuracies.
- Log-rank between acquisition-trial KM curves.

Pairwise table columns: `method_a, method_b, pearson, pearson_lo, pearson_hi, wilcoxon_stat, wilcoxon_p, logrank_chi2, logrank_p, n_functions` (saved as `pairwise_stats.csv`).

`.plot()` writes:

- Cumulative-acquisition curve per method (Rule Fig 3A reproduction). One curve per method on a single axes; **humans get a per-subject median curve plus 25-75% (dark gray) and min-max (light gray) bands**, computed by walking `HumanMethod.predict_per_subject(trial)` for every (function, order, trial) the subject saw, applying Rule's strict criterion per (subject, function, order), then computing the across-subject distribution of "fraction of functions acquired by trial *t*". Subjects with fewer than 11 trials for a (function, order) are skipped to avoid inflating the criterion. Result is saved separately as `human_per_subject_acquired.parquet` for downstream analyses.
- Per-trial mean-accuracy curve (sibling figure).
- Human-relative-score violin (per-function all-trial model mean accuracy / per-function all-trial human mean accuracy; Rule Fig 3B reproduction).

Aggregation respects repeated measures: bootstrap at run/subject level before bootstrapping across functions.

### 7.2 `FailureModesAnalysis(features=None)`

Per-method outputs:

- `accuracy[method, feature, value ∈ {True, False}]` averaged over tasks where `task.features[feature] == value`, with bootstrap CIs. Per-task accuracy is the un-thresholded run/subject mean from `Prediction.effort["mean_correct"]` averaged across **all 11 trials** and 5 orders (fallback `float(Prediction.correct)` when a method doesn't expose `mean_correct`). Earlier the analysis read binarized `Prediction.correct` at trial 11 only — that compresses humans to "≥50% subjects correct on the final trial" and similarly for run-averaging models, and diverges from Rule's reported feature-stratified accuracies.
- FDR-corrected χ² test of feature ↔ correctness per (method, feature). The χ² 2×2 table is built by thresholding the per-(method, function) accuracy at 0.5 (i.e. "method got the function right on average"); this preserves the binary χ² semantics while letting the per-task accuracy vector be continuous.

Cross-method outputs:

- Per-feature difference profile: rank tasks by `acc_A − acc_B`, surface top/bottom 10 with their gloss + feature signature.

`.plot()` writes:

- Method × feature accuracy heatmap, with humans/MPL/Fleet pinned reference rows.
- Difference-profile bars (per method pair on demand; `pairs` field in YAML).

### 7.3 `ClusteringAnalysis`

Embeds per task at `n_io_shown=11`, `order=1` by default. Optionally per-(task, order) replicates if `replicates: true` (5× cost; gives within-task variance for cluster robustness). Per-(task, trial) trajectories (`level: trajectory`) are supported but not v1 default.

Per-method outputs:

- Embedding matrix `(n_tasks [, n_replicates], d_model)`.
- Reductions (PCA, t-SNE, UMAP) at requested perplexities; PCA always kept for ARI / Procrustes comparisons.
- Cluster assignments via k-means (k chosen from `k_search` list by silhouette; gap statistic optional) and hierarchical (Ward) for robustness check.
- **Cluster characterisation**:
  - Descriptive: most-enriched binary feature per cluster, χ² with FDR.
  - Inferential: sparse multinomial logistic regression of cluster id on `functions.csv` features → short signatures like "cluster 3 ≈ recursive ∧ indexing".

Cross-method outputs:

- ARI / NMI between any two methods' clusterings.
- CCA / Procrustes alignment of raw embedding spaces (PCA-projected).
- Hungarian-algorithm cluster matching for an interpretable per-cluster correspondence.

`.plot()` writes:

- 2-D scatter per method (PCA and t-SNE), colour-by `cluster` plus user-selected feature columns.
- ARI heatmap across method pairs.
- Cluster-feature enrichment heatmap.

### 7.4 `ErrorSimilarityAnalysis`

Asks: when methods are wrong, are their wrong answers the wrong answers humans give?

**Inputs.** Parsed responses from every method that has `Capability.PREDICTIONS` plus humans. Responses are normalised to a tuple of ints via `_parse_response(s)` in `methods/csv_method.py`:

- `nan` / null → sentinel `NO_RESPONSE` (distinct from `EMPTY`); kept as a label, not dropped, because Codex emits ~50% nulls (timeouts / API errors) and dropping them would inflate its apparent human-likeness.
- `"[]"` → sentinel `EMPTY`.
- `"C([1, 2, 3])"` (MPL's TRS-style wrapper) → strip and `literal_eval` the inside.
- All other strings → `ast.literal_eval` to a tuple of ints.
- Bad parses → `NO_RESPONSE` with a one-warning-per-method log.

**Per-cell metric.** For each cell `(function_id, order, trial)`:

- Build the human response distribution `P_human(r)` over the ~20 subjects who answered that cell (median 19, max 30 across c001–c100; verified empirically). Smoothing not needed — we use match probability rather than KL.
- For each method, look up its (run-aggregated for CSV models, single for transformers) response `r_model`. The cell-level human-likeness is `P_human(r_model)` ∈ [0, 1] — i.e. "what fraction of humans gave the same answer the model gave?". `EMPTY` and `NO_RESPONSE` are valid keys in this distribution; humans do produce both.
- Where a model has multiple runs (MPL/Fleet: 5; Codex/RobustFill/Enumerate/Metagol: per the `run` column), report the mean over runs and a run-level bootstrap CI for the cell.

**Per-method aggregate.** Mean `P_human(r_model)` over the 5 orders × 11 trials per function → length-100 human-likeness profile. Mean over functions → scalar human-likeness per method, with bootstrap CI bootstrapping at the function level (functions are the units of independent variation).

The per-method table also carries an `accuracy` column = mean accuracy across **all 11 trials and 5 orders** using the un-thresholded run/subject mean (from `Prediction.effort["mean_correct"]`, fallback `float(Prediction.correct)`). This drives the `scatter_humanness_vs_accuracy.png` x-axis and lines up with the per-method accuracy reported in `RuleAcquisitionAnalysis`. The earlier implementation used binarized `correct` at trial 11 only, which placed plateaued models at the same x-coordinate and didn't match Rule's headline accuracy figures.

The `cond_correct` / `cond_incorrect` decomposition keeps `Prediction.correct` (binary) for the cell-level split: the question "when the model is correct on this cell, is it human-like?" is intrinsically binary at the cell level. For humans this corresponds to "the modal subject got it right"; for run-averaging models, "≥50% of runs got it right". This is a deliberate design choice — making the split continuous would require redefining the metric.

**Cross-method outputs (computed during `run`).**

- Spearman ρ between length-100 profiles for every method pair, with bootstrap CIs.
- Paired Wilcoxon between humans-likeness vectors of any two methods.
- Per-trial profiles (length-11 per method): does human-likeness sharpen with more I/O examples?
- Conditional decomposition: human-likeness on cells where the model is **correct** vs **incorrect**. ("When model X is wrong, are its wrongs human-like?" is the more interesting question, and the unconditional metric obscures it.)

`.plot()` writes:

- Bar chart of mean human-likeness per method with bootstrap CIs, sorted.
- Per-trial human-likeness curves (one per method).
- Scatter of human-likeness vs accuracy across methods (does "more accurate" track "more human-like"?).
- Conditional bars (correct vs incorrect cells) per method.

Capabilities required: `PREDICTIONS` only (no embedding requirement). HumanMethod is the reference; it skips itself in the cross-method comparison or trivially scores 1.0 — the analysis emits a single warning "humans excluded from comparison set as the reference" rather than a per-trial warning.

### 7.5 Stretch analyses (v1: skeleton + tests, no plotting)

- `ProbingAnalysis`: linear probe on per-task embeddings predicting MPL meta-primitive usage from MPL's `metaprogram` column (one-vs-rest per primitive), filtered to MPL-acquired tasks. Method-side hook reserved (`MPLMethod.metaprimitives_for(task_id)`); `run()` raises `NotImplementedError` until v1.1. Full delta plan in [.claude/plans/probing.md](probing.md).

## 8. Trial loop

```python
for trial in bundle.iter_trials():
    for method in methods:
        cache.get_or_compute(method, trial, fn=method.predict)
        if analysis.needs_embeddings and method.supports(Capability.EMBEDDINGS):
            cache.get_or_compute_embedding(method, trial, fn=method.embed)
```

`CapabilityMissing` is caught at the analysis level, with one warning per (method, capability).

## 9. YAML config (example)

```yaml
run_name: study_v1
output_dir: outputs/analysis           # final dir = outputs/analysis/study_v1
rule_data_root: src/data/osf_gq2hj/osfstorage/analysis/data
device: cuda

methods:
  - { kind: transformer, name: tx_in_weight,  run_name: rl_iw_baseline, mode: in-weight,             ckpt: best_acc, embedding_pool: mean, corpus: datasets/corpus-a/rl_corpus.json }
  - { kind: transformer, name: tx_easy_shuf,  run_name: rl_easy_shuf,   mode: easy-symbol-shuffling, ckpt: best_acc, embedding_pool: mean, corpus: datasets/corpus-a/rl_corpus.json }
  - { kind: transformer, name: tx_hard_shuf,  run_name: rl_hard_shuf,   mode: symbol-shuffling,      ckpt: best_acc, embedding_pool: mean, corpus: datasets/corpus-a/rl_corpus.json }
  - { kind: human,       name: humans }
  - { kind: mpl_best,    name: mpl,    filters: { lesion: None, mode: Online, steps: 500000 } }
  - { kind: fleet_best,  name: fleet,  filters: { mode: Online, steps: 500000 } }

analyses:
  - kind: rule_acquisition
    pairs: all                     # which method pairs to compute Pearson/Wilcoxon/log-rank for

  - kind: failure_modes
    features: [recursive, higher, conditional, arithmetic, mapping, filtering, indexing, unfolding, counting]
    pairs: [[tx_in_weight, humans], [tx_hard_shuf, humans], [tx_in_weight, tx_hard_shuf]]

  - kind: clustering
    methods: [tx_in_weight, tx_easy_shuf, tx_hard_shuf]   # subset; HumanMethod skipped via CapabilityMissing
    reducer: [pca, tsne]
    cluster: kmeans
    k_search: [4, 5, 6, 7, 8, 9, 10]
    color_by: [cluster, recursive, mapping]
    replicates: false
    level: task
```

`config.py` validates against a `@dataclass` tree; unknown keys fail loudly.

## 10. Caching

- `outputs/analysis/<run_name>/cache/<method.name>/predictions.parquet` — keyed by `(task_id, order, trial)`, columns `(response, program, correct, effort_json)`.
- `outputs/analysis/<run_name>/cache/<method.name>/embeddings.npz` — keyed by `(task_id, order)` (only for methods with `EMBEDDINGS`).
- Cache key fingerprint: `run_name + mode + ckpt mtime + ckpt_select + embedding_pool` for transformers, `csv mtime + filters dict` for CSV methods. Mismatch → invalidate.
- `cache.get_or_compute(method, trial, fn)` reads the parquet, falls back to computing, writes back in batches of 256 trials. CSV methods are basically free; the cache exists so transformer runs aren't re-decoded across analyses or re-runs.

## 11. CLI

```python
def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    bundle = TaskBundle.load(cfg.rule_data_root)
    methods = [build_method(m) for m in cfg.methods]
    cache = Cache(cfg.output_dir / cfg.run_name / "cache")
    outdir = cfg.output_dir / cfg.run_name
    for acfg in cfg.analyses:
        analysis = build_analysis(acfg)
        result = analysis.run([methods_named(m) for m in acfg.methods], bundle, cache)
        result.save(outdir / acfg.kind)
        result.plot(outdir / acfg.kind)
```

`python -m src.analysis path/to/config.yaml`. No subcommands. No daemon. No multiprocessing in v1; if transformer eval is slow, batch trials inside `TransformerMethod.predict_many` later.

## 12. Critical files / hooks into existing code

To **create**:

- `src/analysis/__init__.py`, `cli.py`, `config.py`, `task.py`, `capability.py`, `cache.py`, `symbol_map.py`, `stats.py`, `plotting.py`
- `src/analysis/methods/{__init__,base,csv_method,transformer,human,mpl,fleet,codex,enumeration,metagol,robustfill}.py`
- `src/analysis/analyses/{__init__,base,rule_acquisition,failure_modes,clustering,error_similarity,probing}.py`
- `src/analysis/configs/example.yaml`

To **read and reuse** (no changes needed unless noted):

- `src/data/dataloader.py:270` `tokenise_program_item`
- `src/data/dataloader.py:231` `_sample_name_map`, `:240` `_sample_partial_name_map`, `:225` `_episode_rng`
- `src/train.py:131` `greedy_decode`
- `src/train.py:163` `_check_program_match` (reference for fn-name reverse-map; **factor the reverse-map into `src/analysis/symbol_map.py` and use from both call sites — small refactor PR**)
- `src/train.py:325` `load_checkpoint`
- `src/train.py:358` `_easy_shuffle_k_for_epoch`
- `src/train.py:148` `_alarm` (inference-time timeout)
- `src/models/seq2seq.py:274` `Seq2SeqTransformer.encode`
- `src/lang/parser.py`, `src/lang/compiler.py` (JIT execution)
- `src/data/osf_gq2hj/osfstorage/analysis/data/{functions,stimuli,predictions,mpl,mpl_best,fleet,fleet_best,codex,enumeration,metagol,robustfill}.csv`

## 13. Build order

1. `task.py`, `capability.py`, `methods/base.py`, `methods/csv_method.py`. Add `methods/{human,mpl,fleet}.py`. **Sanity check: per-task mean accuracy at trial 11 for humans/MPL/Fleet matches Rule's published aggregates within 1 percentage point.**
2. `analyses/rule_acquisition.py` with strict-acquisition + per-trial-mean. Reproduce Rule Fig 3A from CSVs alone.
3. `cache.py`, `methods/transformer.py` (PREDICTIONS only first). Run `tx_in_weight` against Rule100; plot its acquisition curve alongside humans/MPL/Fleet.
4. Add cross-method Pearson/Wilcoxon/log-rank to the acquisition analysis (Pearson on per-function mean accuracy averaged across all 11 trials, matching Rule Supp. Fig. 6).
5. `analyses/failure_modes.py` with feature-stratified means + FDR χ² + difference profile.
6. `methods/transformer.py` `embed()`; `analyses/clustering.py` with PCA/t-SNE, k-means + characterisation + ARI.
7. Round out CSV methods (`codex`, `enumeration`, `metagol`, `robustfill`).
8. `_parse_response` in `methods/csv_method.py` (handles nulls, `"[]"`, `"C(...)"` wrapper); `analyses/error_similarity.py` with per-cell match probability, conditional decomposition, and cross-method Spearman.
9. **Stretch:** `analyses/probing.py` (see [probing.md](probing.md)).

## 14. Verification (v1 done-when)

1. `python -m src.analysis src/analysis/configs/example.yaml` runs to completion with three transformer runs + Human + MPL + Fleet on the rule-acquisition, failure-mode, clustering, and error-similarity analyses.
2. `outputs/analysis/study_v1/rule_acquisition/{result.parquet, curves_acquisition.png, curves_per_trial.png, violin_human_relative.png, pairwise_stats.csv}` exist; the cumulative-acquisition curve matches Rule Fig 3A within visual tolerance.
3. `outputs/analysis/study_v1/failure_modes/{result.parquet, heatmap.png, difference_profile_*.png, chi2_fdr.csv}` exist with rows for each method × feature.
4. `outputs/analysis/study_v1/clustering/{embeddings.npz, reductions.parquet, clusters.parquet, scatter_*_pca.png, scatter_*_tsne.png, ari_heatmap.png, cluster_features.csv}` exist.
5. `outputs/analysis/study_v1/error_similarity/{result.parquet, bar_human_likeness.png, curves_per_trial.png, scatter_humanness_vs_accuracy.png, bars_conditional.png, pairwise_spearman.csv}` exist; HumanMethod is excluded from the bar chart with one warning.
6. Re-running the CLI is fast: cache is hit for all methods (only first run pays transformer decode cost).
7. `HumanMethod` per-task accuracy at `trial=11` matches `analysis.R`'s reported aggregate within 1 percentage point.
8. A `clustering` run with only `HumanMethod` in the config emits exactly one `CapabilityMissing` warning ("humans does not support EMBEDDINGS") and skips cleanly.
9. `TransformerMethod.predict` on `c001` order 1 trial 1 produces the same response as a manual `greedy_decode` invocation in a notebook with the same checkpoint.
10. Rule's strict acquisition criterion sanity-checked on a 5-trial fixture (every permutation of correct/incorrect over 5 trials → expected acquired-on value).
11. `_parse_response` round-trips fixtures for: `"[1, 2, 3]"` → `(1,2,3)`; `"[]"` → `EMPTY`; `nan` → `NO_RESPONSE`; `"C([4, 5])"` → `(4,5)`; `"garbage"` → `NO_RESPONSE` + one warning logged.
12. **Rule-paper corroboration (csv_only run, May 2026):**
    - `HumanMethod` retains 392 / excludes 106 of 498 raw subjects. Matches `analysis.R` exactly.
    - Per-function human mean accuracy on c001..c100: n=100, mean=0.521, SD=0.202, range [0.042, 0.868]. Matches paper Results §1 verbatim.
    - 18 of 21 cross-method Pearson correlations match Supp. Fig. 6 within Δ < 0.05 (most to 3 d.p.). The three pairs with larger residuals — `mpl-fleet` (Δ +0.075), `mpl-enumeration` (Δ +0.071), `mpl-metagol` (Δ +0.101) — are MPL pairs and reflect that the live `TaskBundle` iterates one order per task while the paper averages over all 5 orders × 5 runs per cell. Acceptable for v1; revisit if a future analysis demands per-order replicates.
    - Per-subject acquisition: 389 unique subjects (matches paper's "n = 389 people"); median curve hits 0.50 at trial 8 — consistent with paper's "Functions were acquired by ≥50% of human learners within eight trials".

13. **Audit notes — places where binarized `Prediction.correct` was leaking into continuous metrics, fixed in this pass:**
    - `RuleAcquisitionAnalysis`: per-trial mean and pairwise Pearson now use `accuracy` (un-thresholded), not `correct`.
    - `FailureModesAnalysis`: per-task accuracy now mean over all 11 trials × 5 orders using `mean_correct`, not binarized `correct` at trial 11.
    - `ErrorSimilarityAnalysis`: `per_method["accuracy"]` (scatter x-axis) now mean over all trials using `mean_correct`. The `cond_correct`/`cond_incorrect` split intentionally remains binary at the cell level.
    - `ClusteringAnalysis`: unaffected (uses embeddings, not accuracy).
    - **Strict-acquisition trial** (`acquired_on`): keeps binary `correct` per Rule's criterion. The Fig 3A bands now come from a separate per-subject pass via `HumanMethod.predict_per_subject`.
    - **Loader bool parsing**: `task.py` extended `_TRUE`/`_FALSE` to accept `y/n/yes/no` so that `recursive` (39/100) and `counting` (10/100) feature columns load correctly despite stray-shorthand cells in OSF's `functions.csv`. Loader emits a one-line warning per column with non-canonical values; OSF data is never edited.

## 15. Anti-overengineering

Explicitly **not** building:

- Plugin/registry decorators (`@register_method`). Use a 10-line dict.
- DI container or `MethodFactory`. Methods instantiated by `Klass(**cfg)` directly.
- Generic `ExperimentRunner` / `Pipeline` / `Step`.
- Async / multiprocessing layer in v1.
- Hydra / OmegaConf / pydantic. Plain pyyaml + a single `@dataclass` tree.
- A separate `Visualizer` or `Comparison` module — both fold into `AnalysisResult.plot` and `Analysis.run`.
- A live MPL re-runner. We only consume Rule's pre-computed CSVs.
- `Method.fit()` / `Method.train()`.
- Result schemas more elaborate than tidy dataframes + `.plot()`.
- IRT joint-fit and behavioural-clustering-of-humans (out of scope for now).

## 16. Design decisions baked in

- **Inference path:** TransformerMethod runs greedy decode + embedding extraction inside the package, gated by per-method disk cache.
- **Benchmark I/O:** Rule's `stimuli.csv` exactly. No re-sampling at eval time.
- **Embedding pool:** configurable (`mean | last | attn`), default `mean`.
- **Acquisition stat:** report **both** strict acquisition trial (Rule's criterion, headline) and per-trial mean (sibling curve).
- **Easy-shuf K at eval:** derive from the checkpoint's epoch via `_easy_shuffle_k_for_epoch(epoch, args, n_total_fns)`.
- **Embedding scope v1:** per-task at full context (`n_io_shown=11`, `order=1`). Per-(task, order) replicates exposed via `replicates: true`. Per-(task, trial) trajectories supported via `level: trajectory` but not v1 default.
- **Stretch analyses kept in design space:** probing for meta-primitive use (full delta plan in [probing.md](probing.md)).
- **Repeated-measures aggregation:** bootstrap at run/subject level inside the function, then across functions.
- **Error-similarity null handling:** Codex's ~50% null responses are encoded as a `NO_RESPONSE` sentinel and kept in the human-likeness distribution rather than dropped, so methods aren't rewarded for non-response. `EMPTY` (`"[]"`) is a separate sentinel.
- **Error-similarity metric:** per-cell `P_human(model_response)` rather than KL — interpretable as "fraction of humans matching the model" and well-defined without smoothing.
- **Error-similarity reference:** humans are the reference, not a participant in the bar chart; one warning logged at the start of the analysis.
