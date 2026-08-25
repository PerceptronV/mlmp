"""Tests for temperature sampling and the accuracy@k (best-of-k) metrics."""

import pytest
import torch

from src.data.program_io import ProgramIO
from src import train


@pytest.fixture(scope="module")
def io():
    return ProgramIO()


class _StubModel:
    """Stands in for ``Seq2SeqTransformer`` in ``decode_batch``.

    Its next-token logits are a pure function of the current token, so tests
    can pin the distribution exactly without running attention (the jagged
    flex-attention decoder doesn't compile on CPU).
    """

    def __init__(self, n_tokens: int, candidates: dict[int, list[int]]):
        self.n_tokens = n_tokens
        self.candidates = candidates  # token -> tokens that may follow, in rank order

    def encode(self, src):
        return src

    def decode(self, tgt, memory):
        return tgt

    def project(self, h):
        vals = h.values()
        logits = torch.full((vals.numel(), self.n_tokens), -1e9)
        for row, token in enumerate(vals.tolist()):
            # Descending logits so argmax is the first candidate; equal values
            # would make the sampled test order-dependent.
            for rank, candidate in enumerate(self.candidates[token]):
                logits[row, candidate] = -float(rank)
        return torch.nested.nested_tensor_from_jagged(logits, h.offsets())


def _srcs():
    return [torch.tensor([3, 4, 5]), torch.tensor([6, 7]), torch.tensor([8])]


def test_decode_batch_greedy_takes_the_argmax_and_stops_at_end(io):
    n_tokens = len(io.tokeniser.vocab)
    a, b = 20, 21
    model = _StubModel(n_tokens, {io.start: [a, b], a: [io.end], b: [io.end]})

    gens = io.decode_batch(model, _srcs(), max_tokens=6, device="cpu")

    # Every row walks start -> a -> <end> and stops there, well short of max_tokens.
    assert gens == [[a, io.end]] * 3


def test_decode_batch_samples_from_the_distribution_above_zero_temperature(io):
    n_tokens = len(io.tokeniser.vocab)
    a, b = 20, 21
    model = _StubModel(n_tokens, {io.start: [a, b], a: [io.end], b: [io.end]})

    torch.manual_seed(0)
    rows = [io.decode_batch(model, _srcs(), max_tokens=6, device="cpu",
                            temperature=2.0)
            for _ in range(8)]

    first_tokens = {gen[0] for batch in rows for gen in batch}
    assert first_tokens == {a, b}, "sampling should reach both candidates"
    # Only the first token is a choice; the rest of the path is forced.
    assert all(gen[1:] == [io.end] for batch in rows for gen in batch)


def test_greedy_is_temperature_zero(io):
    n_tokens = len(io.tokeniser.vocab)
    a, b = 20, 21
    model = _StubModel(n_tokens, {io.start: [a, b], a: [io.end], b: [io.end]})

    torch.manual_seed(0)
    assert io.decode_batch(model, _srcs(), max_tokens=6, device="cpu",
                           temperature=0.0) == io.decode_batch(
        model, _srcs(), max_tokens=6, device="cpu")


class _StubDataset:
    """Minimal shape ``_val_view_indices`` reads off a validation dataset."""

    def __init__(self, n_programs: int, n_io_views: int):
        self.programs = [None] * n_programs
        self.n_io_views = n_io_views
        self.min_n_io_shown = 1


def test_pass_at_k_keeps_each_programs_best_sample(monkeypatch):
    # Two programs, scored at n_io_shown 1 (round-robin view 0) and 2 (view 1).
    # Neither sample is good at both: each program's best is in a different one.
    samples = [
        [(False, 'malformed', 0, 1), (False, 'executed', 1, 2)],
        [(True, 'executed', 1, 1), (False, 'malformed', 0, 2)],
    ]
    temperatures = []

    def fake_decode_and_classify(*args, **kwargs):
        temperatures.append(kwargs['temperature'])
        return samples[len(temperatures) - 1]

    monkeypatch.setattr(train, '_decode_and_classify', fake_decode_and_classify)

    metrics = train.compute_pass_at_k_metrics(
        model=None, val_dataset=_StubDataset(2, 2), device=None, k=2,
        temperature=0.7)

    assert temperatures == [0.7, 0.7]
    # Program 0 is only correct in sample 2, program 1 only ever partial.
    assert metrics['accuracy'] == 0.5
    assert metrics['accuracy_at_k'] == [0.0, 0.5]
    assert dict(metrics['failure_modes']) == {'correct': 1, 'partial_mismatch': 1}
    assert {n: dict(c) for n, c in metrics['outcomes_by_n_io'].items()} == {
        1: {'match_1': 1}, 2: {'match_1': 1}}


def test_best_outcome_ranks_matches_then_completion():
    malformed = (False, 'malformed', 0, 3)
    errored = (False, 'runtime_error', 0, 3)
    partial = (False, 'executed', 2, 3)
    correct = (True, 'executed', 3, 3)

    assert train._best_outcome(malformed, partial) is partial
    assert train._best_outcome(partial, correct) is correct
    # Same number matched: the sample that ran to completion wins.
    assert train._best_outcome(malformed, errored) is errored
    assert train._best_outcome(errored, (False, 'executed', 0, 3)) == (
        False, 'executed', 0, 3)
    # Items without execution semantics (inverse-mlc / vacuous) rank on correct.
    assert train._best_outcome((False, None, None, 0), (True, None, None, 0))[0]
