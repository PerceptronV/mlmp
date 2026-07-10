"""Integration tests for Stage 0 and Stage 1 on a tiny pool."""

import json
from math import comb

import pytest

from src.lang.enumeration.enumerator import BottomUpEnumerator
from src.lang.subset_search.pool import pool_grammar
from src.lang.subset_search.search import run_stage0, run_stage1
from src.lang.subset_search.scoring import score_vector

TINY_POOL = ('+', '*', 'length', 'take', 'concat', 'map')
TINY_MAX_SIZE = 4


@pytest.fixture(scope="module")
def stage0_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("subset_search")
    run_stage0(str(out), max_size=TINY_MAX_SIZE, sizes=(5,), pool_names=TINY_POOL)
    return out


def test_stage0_scores_all_subsets_sorted(stage0_dir):
    data = json.loads((stage0_dir / "stage0.json").read_text())
    scores = data['scores']
    assert len(scores) == comb(len(TINY_POOL), 5)
    values = [s for _, s in scores]
    assert values == sorted(values, reverse=True)
    assert data['pool'] == list(TINY_POOL)


def test_proxy_is_lower_bound_on_exact_yield(stage0_dir):
    """Spec sanity check: proxy(S) <= exact N(S) at the same size bound."""
    data = json.loads((stage0_dir / "stage0.json").read_text())
    for key, proxy in data['scores'][:3]:
        g = pool_grammar(tuple(key.split(' ')))
        enum = BottomUpEnumerator(grammar=g, max_size=TINY_MAX_SIZE)
        bank = enum.enumerate()
        exact = score_vector(bank, enum.attempts, TINY_MAX_SIZE)['n_distinct']
        assert proxy <= exact, f"{key}: proxy {proxy} > exact {exact}"


def test_stage1_runs_and_is_resumable(stage0_dir):
    results = run_stage1(
        str(stage0_dir), top_n=2, max_size=TINY_MAX_SIZE, timeout_s=120, workers=2,
    )
    assert len(results) == 2
    for key, res in results.items():
        assert res['status'] == 'ok'
        assert res['score']['n_distinct'] >= 0
        assert res['max_size'] == TINY_MAX_SIZE

    on_disk = json.loads((stage0_dir / "stage1.json").read_text())
    assert on_disk == results

    # Resumability: a second call re-runs nothing (all keys present) and
    # returns the same results.
    again = run_stage1(
        str(stage0_dir), top_n=2, max_size=TINY_MAX_SIZE, timeout_s=120, workers=2,
    )
    assert again == results


def test_subset_timeout_escapes_broad_exception_handlers():
    from src.lang.subset_search.search import SubsetTimeout
    with pytest.raises(SubsetTimeout):
        try:
            raise SubsetTimeout()
        except Exception:
            pytest.fail("SubsetTimeout must not be caught by 'except Exception'")


def test_stage1_worker_records_error(monkeypatch):
    import src.lang.subset_search.search as search_mod

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(search_mod, 'score_vector', boom)
    key, res = search_mod._stage1_worker(('+ length', 2, 60))
    assert res['status'] == 'error'
    assert 'boom' in res['error']


def test_stage1_worker_records_timeout(monkeypatch):
    import time
    import src.lang.subset_search.search as search_mod

    def slow(*args, **kwargs):
        time.sleep(10)  # SIGALRM interrupts the sleep

    monkeypatch.setattr(search_mod, 'score_vector', slow)
    key, res = search_mod._stage1_worker(('+ length', 2, 1))
    assert res['status'] == 'timeout'
