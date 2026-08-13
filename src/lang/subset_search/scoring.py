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

# When the search is restricted to a single output type, type_coverage is
# constant and spread/tail reward variety the restriction deliberately gave
# up, so the front is taken over count and efficiency only.
PARETO_DIMS_TARGETED: tuple[str, ...] = ('n_distinct', 'density')


def _quality_programs(
    bank: ProgramBank,
    min_successes: int,
    min_variability: float,
    target_type: str | None = None,
) -> list[TypedProgram]:
    return [
        p for p in bank.all_programs()
        if p.fingerprint is not None
        and passes_quality_filter(p.fingerprint, min_successes, min_variability)
        and (target_type is None or str(p.type) == target_type)
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
    target_type: str | None = None,
) -> dict:
    """With ``target_type`` (e.g. 'list[int]'), only behaviors of that output
    type count toward every dimension; ``attempts`` stays the full enumeration
    effort, so density charges the subset for work spent on other types."""
    programs = _quality_programs(bank, min_successes, min_variability, target_type)
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
