"""Tests for Pareto selection and Stage 2 report generation."""

from src.lang.subset_search.search import run_stage0, run_stage1
from src.lang.subset_search.report import pareto_front, write_reports


def _vec(**overrides):
    base = {
        'n_distinct': 10, 'slope': 1.5, 'density': 0.1,
        'type_coverage': 3, 'spread': 0.5, 'tail': 0.2, 'curve': [1, 5, 10],
    }
    base.update(overrides)
    return base


def test_pareto_front_removes_dominated():
    vectors = {
        'a': _vec(),
        'b': _vec(n_distinct=5, slope=1.0, density=0.05,
                  type_coverage=2, spread=0.3, tail=0.1),   # dominated by a
        'c': _vec(n_distinct=8, spread=0.9),                # trade-off vs a
    }
    front = pareto_front(vectors)
    assert set(front) == {'a', 'c'}
    assert front[0] == 'a'  # sorted by n_distinct descending


def test_pareto_front_keeps_equal_vectors():
    vectors = {'a': _vec(), 'b': _vec()}
    assert set(pareto_front(vectors)) == {'a', 'b'}


def test_write_reports_end_to_end(tmp_path):
    pool = ('+', '*', 'length', 'take', 'concat', 'map')
    run_stage0(str(tmp_path), max_size=4, sizes=(5,), pool_names=pool)
    run_stage1(str(tmp_path), top_n=3, max_size=4, timeout_s=120, workers=2)

    finalists = write_reports(str(tmp_path), max_finalists=2)
    assert 1 <= len(finalists) <= 2

    reports_dir = tmp_path / "reports"
    assert (reports_dir / "summary.md").exists()
    for key in finalists:
        report = (reports_dir / f"{key.replace(' ', '_')}.md").read_text()
        assert "Yield curve" in report
        assert "Sample programs" in report
        assert "Hardest behaviors" in report


def test_pareto_front_with_targeted_dims_ignores_other_dims():
    vectors = {
        'a': _vec(n_distinct=10, density=0.1, spread=0.1),
        # Dominated on (n_distinct, density) but best on spread — must be
        # dropped when spread is not a Pareto dimension.
        'b': _vec(n_distinct=5, density=0.05, spread=0.9),
    }
    assert pareto_front(vectors, dims=('n_distinct', 'density')) == ['a']


def test_targeted_pipeline_end_to_end(tmp_path):
    pool = ('+', '*', 'length', 'take', 'concat', 'map')
    run_stage0(str(tmp_path), max_size=4, sizes=(5,), pool_names=pool,
               target_type='list[int]')
    results = run_stage1(str(tmp_path), top_n=3, max_size=4, timeout_s=120,
                         workers=2)
    # Stage 1 inherits the restriction from stage0.json: every scored
    # behavior is list[int]-typed, so type_coverage collapses to 1.
    for res in results.values():
        assert res['status'] == 'ok'
        assert res['score']['type_coverage'] == 1

    finalists = write_reports(str(tmp_path), max_finalists=2)
    assert finalists
    summary = (tmp_path / "reports" / "summary.md").read_text()
    assert "list[int]" in summary
    assert "spread" not in summary  # 2D front: n_distinct and density only
    for key in finalists:
        report = (tmp_path / "reports" / f"{key.replace(' ', '_')}.md").read_text()
        assert "Restricted to output type `list[int]`" in report
