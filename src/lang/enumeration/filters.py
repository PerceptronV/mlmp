"""Quality predicates for fingerprinted programs."""

from .fingerprint import Fingerprint, FAIL


def is_non_crashing(fp: Fingerprint, min_successes: int = 3) -> bool:
    """True if at least min_successes entries are not FAIL."""
    successes = sum(1 for v in fp.values if v is not FAIL)
    return successes >= min_successes


def is_non_constant(fp: Fingerprint) -> bool:
    """True if there are at least 2 distinct non-FAIL values."""
    non_fail = {v for v in fp.values if v is not FAIL}
    return len(non_fail) >= 2


def variability(fp: Fingerprint) -> float:
    """Fraction of unique non-FAIL values among all non-FAIL values."""
    non_fail = [v for v in fp.values if v is not FAIL]
    if len(non_fail) <= 1:
        return 0.0
    return len(set(non_fail)) / len(non_fail)


def passes_quality_filter(
    fp: Fingerprint,
    min_successes: int = 3,
    min_variability: float = 0.3,
) -> bool:
    """Conjunction of non-crashing, non-constant, and minimum variability."""
    return (
        is_non_crashing(fp, min_successes)
        and is_non_constant(fp)
        and variability(fp) >= min_variability
    )
