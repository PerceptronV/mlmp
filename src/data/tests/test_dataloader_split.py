"""Tests for ProgramDataset's deterministic holdout split."""

import json
from pathlib import Path

import pytest

from src.data.dataloader import ProgramDataset


@pytest.fixture()
def corpus_file(tmp_path):
    entries = [
        {"program": f"(lambda (x) (prog{i} x))", "type": "list[int]", "size": 3}
        for i in range(100)
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(entries))
    return path


def _programs(ds):
    return {e["program"] for e in ds.programs}


def test_holdout_split_is_disjoint_and_exhaustive(corpus_file):
    kwargs = dict(corpus_files=corpus_file, holdout=20, split_seed=0)
    train = ProgramDataset(split="train", **kwargs)
    val = ProgramDataset(split="val", **kwargs)
    assert len(train.programs) == 80
    assert len(val.programs) == 20
    assert _programs(train) & _programs(val) == set()
    assert len(_programs(train) | _programs(val)) == 100


def test_holdout_split_is_deterministic_and_seed_sensitive(corpus_file):
    val_a = ProgramDataset(corpus_files=corpus_file, holdout=20, split="val", split_seed=0)
    val_b = ProgramDataset(corpus_files=corpus_file, holdout=20, split="val", split_seed=0)
    val_c = ProgramDataset(corpus_files=corpus_file, holdout=20, split="val", split_seed=1)
    assert _programs(val_a) == _programs(val_b)
    assert _programs(val_a) != _programs(val_c)


def test_holdout_applies_before_max_programs_cap(corpus_file):
    # The train cap must shrink the train side only, never the val set.
    train = ProgramDataset(
        corpus_files=corpus_file, holdout=20, split="train", split_seed=0,
        max_programs=10,
    )
    val = ProgramDataset(corpus_files=corpus_file, holdout=20, split="val", split_seed=0)
    assert len(train.programs) == 10
    assert len(val.programs) == 20
    assert _programs(train) & _programs(val) == set()


def test_holdout_rejects_bad_arguments(corpus_file):
    with pytest.raises(AssertionError):
        ProgramDataset(corpus_files=corpus_file, holdout=100, split="val")
    with pytest.raises(AssertionError):
        ProgramDataset(corpus_files=corpus_file, holdout=10, split="test")


def test_no_holdout_keeps_everything(corpus_file):
    ds = ProgramDataset(corpus_files=corpus_file)
    assert len(ds.programs) == 100
