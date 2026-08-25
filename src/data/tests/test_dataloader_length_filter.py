"""Tests for ProgramDataset's max_program_length filter.

The ordering guarantee is the point: filtering has to happen before the
holdout split, the max_programs subsample, and all indexing, so nothing
downstream ever sees an over-length program.
"""

import json

import pytest

from src.data.dataloader import ProgramDataset


def _nest(depth: int, tag: int = 0) -> str:
    """A program whose token length grows with ``depth``. ``tag`` varies the
    integer literal, so programs of equal length are still distinct — the
    holdout split partitions indices, and duplicate strings would make a
    disjointness check meaningless."""
    body = f"(take {tag} _p0)"          # 11 tokens at depth 0
    for _ in range(depth):
        body = f"(concat {body} _p0)"   # +4 tokens per level
    return f"(λ (_p0) {body})"


@pytest.fixture()
def corpus_file(tmp_path):
    entries = [{"program": _nest(d, tag), "type": "list[int]", "size": d + 1}
               for d in range(30) for tag in range(4)]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    return path


def _lengths(ds):
    return [len(ds.tokeniser.tokenise_program(e["program"])) for e in ds.programs]


def test_filter_drops_only_over_length_programs(corpus_file):
    unfiltered = ProgramDataset(corpus_files=corpus_file)
    filtered = ProgramDataset(corpus_files=corpus_file, max_program_length=20)

    assert max(_lengths(filtered)) <= 20
    assert len(filtered.programs) < len(unfiltered.programs)
    # Nothing under the cap was lost.
    kept = {e["program"] for e in filtered.programs}
    for e in unfiltered.programs:
        if len(unfiltered.tokeniser.tokenise_program(e["program"])) <= 20:
            assert e["program"] in kept


def test_filter_runs_before_the_holdout_split(corpus_file):
    """Split, indexing and views all operate on the filtered corpus."""
    kwargs = dict(corpus_files=corpus_file, holdout=10, split_seed=0,
                  max_program_length=20)
    train = ProgramDataset(split="train", **kwargs)
    val = ProgramDataset(split="val", **kwargs)

    assert max(_lengths(train)) <= 20
    assert max(_lengths(val)) <= 20
    assert len(val.programs) == 10
    # Still a partition of the *filtered* corpus, with no overlap.
    filtered = ProgramDataset(corpus_files=corpus_file, max_program_length=20)
    assert len(train.programs) + len(val.programs) == len(filtered.programs)
    assert not ({e["program"] for e in train.programs}
                & {e["program"] for e in val.programs})


def test_filter_runs_before_the_max_programs_cap(corpus_file):
    ds = ProgramDataset(corpus_files=corpus_file, max_program_length=20,
                        max_programs=5, seed=0)
    assert len(ds.programs) == 5
    assert max(_lengths(ds)) <= 20


def test_every_item_respects_the_cap(corpus_file):
    """The filter is upstream of __getitem__, so every emitted target is short."""
    ds = ProgramDataset(corpus_files=corpus_file, max_program_length=20,
                        n_io_per_program=3, min_n_io_shown=1)
    for idx in range(0, len(ds), max(1, len(ds) // 50)):
        seq, loss_mask, info = ds.__getitem__(idx, include_program=True)
        n_program_tokens = sum(loss_mask) - 1   # mask covers program + <end>
        assert n_program_tokens <= 20
        assert len(ds.tokeniser.tokenise_program(info["program"])) <= 20


def test_no_filter_by_default(corpus_file):
    ds = ProgramDataset(corpus_files=corpus_file)
    assert ds.max_program_length is None
    assert max(_lengths(ds)) > 60


def test_fast_path_agrees_with_the_lexer(corpus_file):
    """The length check skips the lexer for strings no longer than the cap;
    that shortcut must never disagree with the exact count."""
    ds = ProgramDataset(corpus_files=corpus_file)
    for cap in (10, 20, 40, 100):
        for entry in ds.programs:
            program = entry["program"]
            exact = len(ds.tokeniser.tokenise_program(program)) <= cap
            assert ds._within_length(program, cap) == exact, (program, cap)


def test_empty_result_is_an_error(corpus_file):
    with pytest.raises(AssertionError, match="filtered out every program"):
        ProgramDataset(corpus_files=corpus_file, max_program_length=3)
