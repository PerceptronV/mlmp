# Synthesising Semantically Diverse Programs from Black-Box Typed Primitives

## Research Notes — Working Document

---

## 1. Problem Statement

We are given:

- A **typed lambda calculus** over basic types (Int, Bool, [t1], etc.)
- A **library of primitive functions** with known types but **black-box semantics** (we can evaluate them but don't have their mathematical definitions)
- A goal: **synthesise a large corpus of programs** that are semantically non-trivial — non-constant, input-sensitive, and behaviourally diverse

The grammar under consideration includes arithmetic, list operations, higher-order combinators (map, filter, fold, sort, group), and standard accessors/slicers. Programs are closed lambda expressions of type `[t1] → [t2]` (or similar).

## 2. The Core Difficulty: Semantic Degeneracy Under Random Sampling

Uniform random sampling over well-typed terms produces programs that are overwhelmingly semantically degenerate. Empirically, ~90% of randomly generated programs (depth 4) have variability < 0.1. Root causes:

1. **Constant domination.** When literals and variables compete at roughly equal weight at every AST node, the probability that the input variable influences the output decays exponentially with term depth. A term of depth $d$ has $O(2^d)$ leaf positions; if each leaf independently has ~50% chance of being a constant, input-sensitivity vanishes fast.

2. **Fragile indexing.** Randomly sampled integer arguments to partial functions (`nth 87 x`, `drop 64 x`, `take 52 x`) almost always fall outside the valid domain for realistic inputs. Programs are well-typed but operationally undefined on nearly all test cases.

3. **Semantic collapse.** Syntactically rich terms reduce to trivial computations. For example, `fold (λ y z. z) init x` always returns the last element regardless of the fold body; `fold (λ y z. 7) init x` is constant. These are well-typed, syntactically complex, but behaviourally trivial.

4. **Dead branches.** Random if-expressions often have constant conditions, causing one branch to be unreachable. Programs that nominally "use" the input variable may only do so in dead code.

**Key insight:** The space of well-typed terms is dominated by semantically degenerate programs. Syntax alone is insufficient — semantic structure is required to generate interesting programs.

## 3. Recommended Approach: Bottom-Up Enumeration with Observational Equivalence

### 3.1 Bottom-Up Enumeration

Build programs incrementally by size:

- **Size 1:** Variables (`x`), small literals (`0, 1, 2, 3`), empty list (`[]`)
- **Size 2:** `length x`, `reverse x`, `sum x`, `max x`, `unique x`, etc.
- **Size 3:** `map f x`, `filter p x`, `sort f x` (where `f` is a size-1 lambda), `drop 1 x`, `take 2 x`, `cons (first x) x`
- **Size k:** All well-typed applications of primitives to combinations of smaller terms

At each size level, the set of programs is finite and enumerable.

**Higher-order arguments.** Functions like `map`, `filter`, `fold`, `sort`, `group` take function-typed arguments. The preferred strategy (following BUSTLE) is to enumerate the lambda body in a context that includes the bound variable, effectively doing a nested bottom-up search. This avoids generating lambdas that are never used as arguments.

**Constant handling.** Do *not* enumerate all integers 0–99. Use a small set of "interesting" constants (`{0, 1, 2, 3}`) plus constants *derived from evaluation* — if a size-$k$ program evaluates to 5 on some input, then 5 is available as a "discovered constant" at size $k+1$. This eliminates the `nth 87` problem by construction.

### 3.2 Observational Equivalence Pruning

Fix a test suite $I = \{i_1, \ldots, i_n\}$ of diverse input lists. For each enumerated program $e$, compute its **fingerprint**:

$$\beta(e) = \langle \text{eval}(e, i_1), \ldots, \text{eval}(e, i_n) \rangle$$

Maintain a hash table of fingerprints. If a new program's fingerprint matches an existing one, discard it. This is sound for distinguishing programs: if $\beta(e_1) \neq \beta(e_2)$ then $e_1 \not\equiv e_2$.

**Effect:** All degenerate programs collapse. `fold (λ y z. z) init x`, `fold (λ y z. 7) init x`, and the literal `7` all map to the same fingerprint and only one representative survives. Constant programs are trivially filtered (all fingerprint entries identical). Programs that crash on all inputs are discarded.

This typically reduces the program count at each size level by 1–2 orders of magnitude.

**Test suite design.** Use 8–15 input lists of varying lengths and content:

```
[], [1], [3,1,2], [1,1,1], [5,4,3,2,1],
[1,2,3,4,5,6,7,8], [10,-3,7,7,0], ...
```

### 3.3 Diversity Selection via Behavioural Fingerprints

After pruning, each surviving program has a unique fingerprint. To select a *diverse* subset:

- **Simple:** Greedily select programs maximising the number of distinct fingerprints
- **Structured:** Discretise the fingerprint space into cells and use **MAP-Elites** (Mouret & Clune, 2015) to maintain one high-quality representative per cell
- **Metric-based:** Define a distance on fingerprints (Hamming, edit distance, etc.) and select programs maximising pairwise distance or coverage

### 3.4 Additional Static Filtering

Before evaluation, a cheap **data-flow reachability** check can prune programs where the input variable is syntactically present but cannot influence the output (e.g., appears only inside `second []` which always crashes, or in `max(x) - max(x)` which is structurally constant). This is a lightweight relevance/liveness analysis.

### 3.5 Expected Yield

At sizes 3–5, this pipeline should produce thousands to tens of thousands of behaviourally distinct programs, including:

- `reverse (drop 1 x)` — remove first element, then reverse
- `sort (λ y. y) (unique x)` — deduplicate and sort
- `filter (λ y. (is_odd y)) x` — keep odd elements
- `map (λ y. (* y 2)) x` — double every element
- `concat x (reverse x)` — palindromise
- `take (length x) (repeat (sum x) (length x))` — replace all elements with their sum

### 3.6 Key References

- **MagicHaskeller** (Katayama, 2007) — bottom-up enumeration of Haskell programs with observational equivalence
- **Transit** (Udupa et al., PLDI 2013) — formalised observational equivalence pruning in component-based synthesis
- **BUSTLE** (Odena et al., ICLR 2021) — bottom-up synthesis with learned guidance, handles higher-order functions
- **CrossBeam** (Shi et al., ICLR 2022) — neural model decides which subprograms to combine at each enumeration level

## 4. RL-Based Program Generation

### 4.1 Formulation

Define a production-rule MDP over typed derivation trees:

- **State** $S = (t, c, f, i, g)$: the target type $t$, the context $c$ of currently-bound typed variables, the function $f$ being applied (if generating an argument), the argument index $i$, and the list $g$ of already-generated sibling terms
- **Action** $a$: a production choice — apply a specific function, sample a literal, introduce a lambda, use a variable, etc. Actions are constrained by the type system.
- **Episode termination:** when a closed lambda expression is formed
- **Reward** $R(e)$: a semantic quality measure (variability, non-constancy, novelty, etc.) computed by evaluating the completed program on the test suite

A policy $\pi_\theta(a \mid S)$ is trained to maximise expected reward.

### 4.2 The State Design Is Richer Than Prior Work

Most grammar-guided RL for synthesis conditions only on $(t, c)$. Conditioning on $(f, i, g)$ gives the policy information like "I'm generating the second argument to `fold`, and the first argument was `(λ y z. (+ y z))`." This enables the policy to learn:

- When filling a lambda body for `filter`, prefer comparisons involving the bound variable
- When generating an integer argument to `take`, prefer small values or `length`-derived expressions
- When the fold lambda ignores one of its arguments, the overall program is likely degenerate

### 4.3 Central Challenge: Reward Sparsity

~94% of uniformly random programs score near-zero variability. A randomly-initialised policy generates almost exclusively zero-reward episodes, giving REINFORCE no signal. The reward is terminal (only after the full term is formed), and episodes may involve 20–50 sequential decisions. Credit assignment is extremely difficult.

### 4.4 Priority Queue Training (Abolafia et al., 2018)

The key mitigation strategy. Maintain a bounded buffer $\mathcal{B}$ of the top-$K$ highest-reward programs found by any means. Instead of REINFORCE, optimise a maximum likelihood objective over buffer contents:

$$\mathcal{L}(\theta) = \sum_{e \in \mathcal{B}} R(e) \cdot \log \pi_\theta(e)$$

**Training loop:**

1. **Sample:** Roll out $\pi_\theta$ to generate programs, evaluate them, insert any that beat the buffer minimum into $\mathcal{B}$
2. **Train:** Sample mini-batches from $\mathcal{B}$, take gradient steps on $\mathcal{L}$
3. Repeat

The buffer always has positive-reward examples to learn from, breaking the sparse-reward deadlock. As the policy improves, it finds better programs, which improve the buffer, which improves training — a virtuous cycle.

### 4.5 Recommended Hybrid Strategy

1. **Exhaustive enumeration for sizes 1–5** with observational equivalence pruning, collecting a large corpus of high-variability programs
2. **Warm-start the RL policy** via behavioural cloning on the enumerated corpus (supervised learning to imitate the decisions that would produce those programs)
3. **Deploy the policy for biased sampling at sizes 6–8**, using priority queue training with the enumerated corpus as the initial buffer
4. **Shaped intermediate rewards** to provide denser signal: reward choosing input variables over literals in data-flow-critical positions; penalise generating constant-valued subexpressions (detectable by partial evaluation on the test suite)
5. **Curriculum over depth**: train first on depth 2–3 (where random exploration occasionally hits good programs), then increase

### 4.6 Related Work

- **Abolafia et al. (2018)** — "Neural Program Synthesis with Priority Queue Training": RNN policy trained with priority queue mechanism; closest to the proposed formulation
- **Bunel et al. (ICLR 2018)** — grammar-guided policy network with REINFORCE; grammar constrains valid actions (analogous to type constraints)
- **DeepCoder** (Balog et al., ICLR 2017) — neural model predicts which DSL functions are needed from I/O examples, used to prioritise enumeration
- **CrossBeam** (Shi et al., ICLR 2022) — learned model guides bottom-up enumeration by scoring which subprogram pairs to combine
- **Parisotto et al. (ICLR 2017)** — generation network conditioned on partial tree structure

## 5. Most Promising Research Direction: Quality-Diversity RL for Program Synthesis

### 5.1 The Gap in the Literature

Most existing program synthesis RL optimises for correctness against a *single* specification. Our problem is fundamentally different: we want to *fill out* the behavioural space. The reward should not just be "is this program high-variability?" but "does this program occupy a region of behavioural space that isn't already well-covered?"

### 5.2 Proposed Architecture

**MAP-Elites with a Learned Generator:**

1. Discretise the fingerprint space $\beta(e) \in Y^n$ into cells (e.g., by hashing or quantising the output vectors)
2. The RL policy's reward for generating program $e$ is high if $e$ lands in an **empty or under-occupied cell**, and low if it lands in an already-dense cell
3. This creates a **non-stationary reward** that automatically pushes the policy toward generating programs with *novel* behaviour

Over time, the policy learns to target specific behavioural niches on demand — a capability that pure enumeration cannot provide.

### 5.3 Iterative Library Learning

Periodically compress the corpus of discovered programs to extract reusable abstractions (following DreamCoder / Stitch):

- Identify common subprograms via anti-unification
- Add them as new primitives to the library
- This shifts the distribution of enumerable programs toward more interesting regions and enables deeper effective compositions

### 5.4 The Full System

The combination of:

1. **Typed bottom-up enumeration** (completeness at small sizes)
2. **Quality-diversity RL** (targeted exploration at large sizes)
3. **Iterative library learning** (compression → new primitives → deeper compositions)

would be a novel and well-motivated architecture. To our knowledge, no existing work combines all three in this way.

## 6. Theoretical Notes

- **Rice's theorem:** Any non-trivial semantic property of programs is undecidable. Observational equivalence on finite test sets is a decidable approximation — sound for distinguishing programs, incomplete for identifying equivalence.
- **Type inhabitation in STLC** is PSPACE-complete (Statman, 1979). In practice, types provide powerful pruning of the search space.
- **Kolmogorov complexity / MDL:** Program "interestingness" can be formalised as low description length relative to behavioural complexity. This connects to the compression objective in DreamCoder/Stitch.

## 7. Open Questions

- What is the right granularity for fingerprint-space discretisation in MAP-Elites? Too coarse loses diversity; too fine makes coverage intractable.
- How should the RL state representation encode already-generated sibling terms $g$? Options include: direct AST embedding, fingerprint of the partial program, or a learned representation.
- Can the policy generalise compositionally — learning rules at small depths that transfer to larger programs?
- What is the right balance between enumeration (complete but expensive) and RL sampling (incomplete but scalable)?
- How should the test suite $I$ be designed or adapted over time to maximise the discriminative power of fingerprints?

---

*Last updated: March 2026*
