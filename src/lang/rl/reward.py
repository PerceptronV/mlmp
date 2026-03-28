"""Reward computation for RL-based program synthesis."""

from ..enumeration.fingerprint import Fingerprint
from ..enumeration.filters import variability, is_non_crashing, is_non_constant


def compute_reward(
    fingerprint: Fingerprint,
    corpus_fingerprints: set[Fingerprint],
    alpha: float = 0.5,
) -> float:
    """
    Compute the reward for a generated program.

    reward = variability(fp) * novelty_bonus

    where novelty_bonus = 1.0 if novel, 0.1 if already known.
    Returns 0.0 if crashing or constant.
    """
    if not is_non_crashing(fingerprint, min_successes=3):
        return 0.0

    if not is_non_constant(fingerprint):
        return 0.0

    var = variability(fingerprint)
    novelty = 1.0 if fingerprint not in corpus_fingerprints else 0.1

    return var * novelty
