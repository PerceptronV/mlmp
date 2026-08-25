"""Constrained decoding: masked generations are always well-formed.

Uses a stub model rather than a trained one — the point is that the *decoder*
cannot emit a malformed program regardless of what the model wants, so a model
that actively pushes for illegal tokens is the strongest case to test.
"""

import pytest
import torch

from src.data.program_io import ProgramIO
from src.lang.parser import parse


@pytest.fixture(scope="module")
def io():
    return ProgramIO()


class _PerverseModel:
    """Ranks tokens in fixed vocab order, so it always wants token id 0 —
    which is never a legal program token. Unconstrained it produces garbage;
    constrained it must still produce a parseable program.
    """

    def __init__(self, n_tokens: int):
        self.n_tokens = n_tokens

    def encode(self, src):
        return src

    def decode(self, tgt, memory):
        return tgt

    def project(self, h):
        vals = h.values() if h.is_nested else h
        n = vals.shape[0] if h.is_nested else vals.numel()
        logits = torch.arange(self.n_tokens, 0, -1, dtype=torch.float).repeat(n, 1)
        if h.is_nested:
            return torch.nested.nested_tensor_from_jagged(logits, h.offsets())
        return logits.view(*vals.shape, self.n_tokens)


def _decoded_program(io, gen):
    assert io.end in gen, "a constrained decode must terminate with <end>"
    return io.detokenise_program(gen, None)


def test_unconstrained_decode_is_malformed(io):
    """Baseline: without the mask this model's output doesn't parse."""
    model = _PerverseModel(len(io.tokeniser.vocab))
    gen = io.decode_one(model, torch.tensor([3, 4, 5]), 40, 'cpu')
    program = io.detokenise_program(gen, None)
    with pytest.raises(Exception):
        parse(program)


def test_constrained_decode_one_parses(io):
    model = _PerverseModel(len(io.tokeniser.vocab))
    gen = io.decode_one(model, torch.tensor([3, 4, 5]), 40, 'cpu', constrain=True)
    program = _decoded_program(io, gen)
    parse(program)  # raises if the mask let anything through


def test_constrained_decode_batch_parses(io):
    model = _PerverseModel(len(io.tokeniser.vocab))
    srcs = [torch.tensor([3, 4, 5]), torch.tensor([6, 7]), torch.tensor([8])]
    gens = io.decode_batch(model, srcs, 40, 'cpu', constrain=True)
    assert len(gens) == 3
    for gen in gens:
        parse(_decoded_program(io, gen))


def test_constrained_decode_is_executable(io):
    """Well-formed *and* runnable: the whole point is to stop losing programs
    to `malformed` before they ever reach the I/O check."""
    model = _PerverseModel(len(io.tokeniser.vocab))
    gen = io.decode_one(model, torch.tensor([3, 4, 5]), 40, 'cpu', constrain=True)
    status, _ = io.classify_program(_decoded_program(io, gen), [([1, 2], [1, 2])])
    assert status != 'malformed'


def test_tight_budget_still_completes(io):
    """A short budget must yield a complete program, not a truncated one.

    The floor is the shortest program that exists — `(λ (_p0) _p0)`, 7 tokens
    — plus one for <end>; below that no decoder could succeed. Production uses
    max_program_tokens=80.
    """
    from src.lang.prefix import PrefixState
    assert PrefixState().min_completion_length() == 7
    model = _PerverseModel(len(io.tokeniser.vocab))
    for budget in (8, 12, 20):
        gen = io.decode_one(model, torch.tensor([3, 4, 5]), budget, 'cpu',
                            constrain=True)
        parse(_decoded_program(io, gen))


def test_constrained_sampling_parses(io):
    """Temperature sampling draws from the masked distribution, so every
    sample is well-formed too — this is what accuracy@k relies on."""
    model = _PerverseModel(len(io.tokeniser.vocab))
    torch.manual_seed(0)
    for _ in range(20):
        gen = io.decode_one(model, torch.tensor([3, 4, 5]), 40, 'cpu',
                            temperature=1.5, constrain=True)
        parse(_decoded_program(io, gen))
