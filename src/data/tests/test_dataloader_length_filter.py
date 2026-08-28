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


# --- min_io_pairs -----------------------------------------------------------
# Programs whose I/O pool is too small are dropped before the split, so a view
# asking for n pairs always gets n.

def _io_corpus(tmp_path):
    """A corpus mixing programs with full I/O pools and programs with none.

    `first` of an empty list raises, so `(first (take 0 ...))` raises on every
    candidate input and the sampler returns nothing. (A program that merely
    returns a constant still yields pairs — the sampler tolerates duplicate
    outputs — so it can't be used to force an empty pool.)
    """
    good = [{"program": _nest(d, tag), "type": "list[int]", "size": 2}
            for d in range(3) for tag in range(4)]
    empty = [{"program": f"(λ (_p0) (singleton (first (take 0 (take {tag} _p0)))))",
              "type": "list[int]", "size": 4} for tag in range(4)]
    path = tmp_path / "io_corpus.json"
    path.write_text(json.dumps(good + empty))
    return path, len(good), len(empty)


def test_min_io_pairs_drops_short_pools(tmp_path):
    path, n_good, _ = _io_corpus(tmp_path)
    ds = ProgramDataset(corpus_files=path, n_io_per_program=4, min_io_pairs=4)
    assert 0 < len(ds.programs) <= n_good
    for idx in range(len(ds.programs)):
        assert len(ds._get_io_pairs(idx)) >= 4


def test_legacy_filter_empty_io_args_replay_as_min_one(tmp_path):
    """--filter-empty-io was replaced by --min-io-pairs, but checkpoints
    predating that store the old flag; rebuilding their val set must still
    apply it, or an offline eval scores a different set of programs."""
    from types import SimpleNamespace

    from src.train import build_val_dataset

    path, _, _ = _io_corpus(tmp_path)
    common = dict(dataset='program', train_corpus=str(path), val_corpus=str(path),
                  val_split=None, split_seed=0, grammar='default', data_seed=0,
                  n_io_per_program=4, min_n_io_shown=1, mode='in-weight',
                  max_program_length=None)
    legacy = build_val_dataset(SimpleNamespace(filter_empty_io=True, **common))
    explicit = build_val_dataset(SimpleNamespace(min_io_pairs=1, **common))
    unfiltered = build_val_dataset(SimpleNamespace(**common))

    assert legacy.min_io_pairs == 1
    assert [e["program"] for e in legacy.programs] == \
           [e["program"] for e in explicit.programs]
    assert len(legacy.programs) < len(unfiltered.programs)


def test_every_view_shows_its_full_number_of_pairs(tmp_path):
    """The point of requiring a full pool: n_io_shown is then real, instead of
    silently collapsing to however many pairs the program could produce."""
    path, _, _ = _io_corpus(tmp_path)
    ds = ProgramDataset(corpus_files=path, n_io_per_program=4, min_n_io_shown=1,
                        min_io_pairs=4)
    for idx in range(len(ds)):
        _seq, _mask, info = ds.__getitem__(idx, include_program=True)
        assert len(info["io_pairs"]) == info["n_io_shown"]


def test_pools_survive_the_split_and_subsample(tmp_path):
    """Filtering runs before the split, so a holdout of N yields exactly N —
    and each kept program keeps the pool it was filtered on."""
    path, _, _ = _io_corpus(tmp_path)
    kwargs = dict(corpus_files=path, n_io_per_program=4, min_io_pairs=4,
                  holdout=3, split_seed=0)
    train = ProgramDataset(split="train", **kwargs)
    val = ProgramDataset(split="val", **kwargs)
    assert len(val.programs) == 3
    assert not ({e["program"] for e in train.programs}
                & {e["program"] for e in val.programs})
    for ds in (train, val):
        # Each survivor still draws the pool it was judged on, so the guarantee
        # holds after re-indexing rather than only at filter time.
        for idx in range(len(ds.programs)):
            assert len(ds._get_io_pairs(idx)) >= 4

    capped = ProgramDataset(corpus_files=path, n_io_per_program=4,
                            min_io_pairs=4, max_programs=2, seed=0)
    assert len(capped.programs) == 2
    for idx in range(2):
        assert len(capped._get_io_pairs(idx)) >= 4


def test_min_io_pairs_cannot_exceed_the_pool_size(tmp_path):
    path, _, _ = _io_corpus(tmp_path)
    with pytest.raises(AssertionError, match="min_io_pairs"):
        ProgramDataset(corpus_files=path, n_io_per_program=4, min_io_pairs=5)


# --- the two optimisations: parallel sampling and the on-disk verdict cache ---

def test_parallel_pool_sizes_match_serial(tmp_path):
    """Pools are seeded per-program, so spreading the pass over processes must
    not change a single verdict."""
    path, _, _ = _io_corpus(tmp_path)
    ds = ProgramDataset(corpus_files=path, n_io_per_program=4)
    serial = ds._io_pool_sizes(workers=1)
    parallel = ds._io_pool_sizes(workers=2)
    assert list(serial) == list(parallel)


def test_verdict_cache_round_trips(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLMP_CACHE_DIR", str(tmp_path / "cache"))
    path, _, _ = _io_corpus(tmp_path)
    kwargs = dict(corpus_files=path, n_io_per_program=4, min_io_pairs=4)

    cold = ProgramDataset(**kwargs)
    assert "reusing cached verdict" not in capsys.readouterr().out

    warm = ProgramDataset(**kwargs)
    assert "reusing cached verdict" in capsys.readouterr().out
    assert [e["program"] for e in warm.programs] == \
           [e["program"] for e in cold.programs]
    # The cached run still draws the pools its verdict was based on.
    for idx in range(len(warm.programs)):
        assert len(warm._get_io_pairs(idx)) >= 4


def test_verdict_cache_is_keyed_on_what_changes_the_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("MLMP_CACHE_DIR", str(tmp_path / "cache"))
    path, _, _ = _io_corpus(tmp_path)
    ds = ProgramDataset(corpus_files=path, n_io_per_program=4, min_io_pairs=4)

    def key(**over):
        args = dict(corpus_files=[path], type_filter="list[int]",
                    max_program_length=None, min_pairs=4)
        args.update(over)
        return ds._io_pool_cache_key(args["corpus_files"], args["type_filter"],
                                     args["max_program_length"], args["min_pairs"])

    base = key()
    assert key() == base                                   # stable
    assert key(min_pairs=2) != base                        # threshold
    assert key(max_program_length=20) != base              # changes what is sampled
    assert key(type_filter=None) != base
    ds.seed = 1                                            # changes every pool
    assert key() != base


def test_verdict_cache_survives_a_corrupt_file(tmp_path, monkeypatch, capsys):
    """A truncated cache must fall back to recomputing, not crash a run."""
    monkeypatch.setenv("MLMP_CACHE_DIR", str(tmp_path / "cache"))
    path, _, _ = _io_corpus(tmp_path)
    kwargs = dict(corpus_files=path, n_io_per_program=4, min_io_pairs=4)
    expected = [e["program"] for e in ProgramDataset(**kwargs).programs]

    cached, = (tmp_path / "cache" / "io_pools").glob("*.npy")
    cached.write_bytes(b"not a numpy file")
    capsys.readouterr()

    ds = ProgramDataset(**kwargs)
    assert "Ignoring unreadable I/O pool cache" in capsys.readouterr().out
    assert [e["program"] for e in ds.programs] == expected


def test_cache_can_be_disabled(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MLMP_CACHE_DIR", str(tmp_path / "cache"))
    path, _, _ = _io_corpus(tmp_path)
    kwargs = dict(corpus_files=path, n_io_per_program=4, min_io_pairs=4,
                  io_pool_cache=False)
    ProgramDataset(**kwargs)
    ProgramDataset(**kwargs)
    assert "reusing cached verdict" not in capsys.readouterr().out
    assert not list((tmp_path / "cache" / "io_pools").glob("*.npy")) \
        if (tmp_path / "cache" / "io_pools").exists() else True
