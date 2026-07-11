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
