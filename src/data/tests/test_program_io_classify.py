"""Tests for ProgramIO.classify_program and the failure-mode taxonomy."""

import pytest

from src.data.program_io import ProgramIO
from src.train import _failure_mode, FAILURE_MODES


@pytest.fixture(scope="module")
def io():
    return ProgramIO()


IDENTITY = "(λ (_p0) _p0)"


def test_malformed_program(io):
    status, n = io.classify_program("(λ (_p0) (undefined_fn", [([1], [1])])
    assert (status, n) == ("malformed", 0)


def test_runtime_error(io):
    # first of an empty list raises ValueError at runtime
    prog = "(λ (_p0) (singleton (first (take 0 _p0))))"
    status, n = io.classify_program(prog, [([1, 2], [1])])
    assert (status, n) == ("runtime_error", 0)


def test_runtime_error_counts_matches_before_the_error(io):
    # first(take 0 xs) raises only when the earlier pairs succeeded... use a
    # program that works on non-empty input but raises on empty input.
    prog = "(λ (_p0) (singleton (first _p0)))"
    pairs = [([5, 2], [5]), ([], [0])]
    status, n = io.classify_program(prog, pairs)
    assert (status, n) == ("runtime_error", 1)


def test_executed_full_match(io):
    status, n = io.classify_program(IDENTITY, [([1], [1]), ([2, 3], [2, 3])])
    assert (status, n) == ("executed", 2)
    assert io.check_program(IDENTITY, [([1], [1]), ([2, 3], [2, 3])])


def test_executed_partial_and_total_mismatch(io):
    pairs = [([1], [1]), ([2], [9])]
    status, n = io.classify_program(IDENTITY, pairs)
    assert (status, n) == ("executed", 1)
    assert not io.check_program(IDENTITY, pairs)

    status, n = io.classify_program(IDENTITY, [([1], [7]), ([2], [9])])
    assert (status, n) == ("executed", 0)


def test_outputs_compared_mod_100(io):
    # The harness convention: model outputs are compared mod 100.
    prog = "(λ (_p0) (map (λ (_p1) (+ _p1 100)) _p0))"
    assert io.classify_program(prog, [([3], [3])]) == ("executed", 1)


def test_non_list_output_is_a_mismatch_not_an_error(io):
    status, n = io.classify_program("(λ (_p0) (length _p0))", [([1, 2], [2])])
    assert (status, n) == ("executed", 0)


def test_nested_list_output_is_a_mismatch_not_a_crash(io):
    # Regression: list[list[int]] outputs used to raise TypeError on the
    # mod-100 comparison instead of counting as a mismatch.
    status, n = io.classify_program(
        "(λ (_p0) (singleton _p0))", [([1, 2], [1, 2])])
    assert (status, n) == ("executed", 0)


def test_failure_mode_taxonomy_is_disjoint_and_complete():
    assert _failure_mode("malformed", 0, 3) == "malformed"
    assert _failure_mode("runtime_error", 2, 3) == "runtime_error"
    assert _failure_mode("executed", 3, 3) == "correct"
    assert _failure_mode("executed", 0, 3) == "total_mismatch"
    assert _failure_mode("executed", 1, 3) == "partial_mismatch"
    assert set(FAILURE_MODES) == {
        "correct", "malformed", "runtime_error", "total_mismatch", "partial_mismatch"
    }
