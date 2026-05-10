"""End-to-end tests against the doc's worked examples and the corpus."""

import json
import random

import pytest

from src.lang.parser import parse
from src.lang.simplify import SimplifyError, simplify
from src.lang.simplify.encode import HoleEncountered
from src.lang.utils import program_size
from src.lang.ast_nodes import IntHoleNode


@pytest.mark.parametrize("inp,expected_size", [
    # §3.6: reverse(reverse(concat xs [])) → xs
    ("(λ (_p0) (reverse (reverse (concat _p0 []))))", 2),
    # §5.5: length(reverse(reverse(repeat 0 5))) → 5
    ("(λ (_p0) (length (reverse (reverse (repeat 0 5)))))", 2),
    # §A: layered identities → sum xs
    ("(λ (_p0) (sum (map (λ (_p1) (* _p1 1)) (filter (λ (_p1) true) (reverse (reverse (map (λ (_p1) _p1) _p0)))))))", 3),
    # The motivating §A.3 example: product (repeat 10 8) is a constant
    ("(λ (_p0) (cons (product (repeat 10 8)) _p0))", 4),
])
def test_worked_examples(inp: str, expected_size: int) -> None:
    ast = parse(inp)
    out = simplify(ast)
    assert program_size(out) == expected_size, \
        f"\n  in:  {inp}\n  out: {out}\n  expected size {expected_size}, got {program_size(out)}"


def test_worked_example_with_add_zero_in_lambda() -> None:
    """§6.6 from the doc: length(reverse(concat(map(λx. x+0) xs)(repeat 8 10)))
       should reduce to (+ (length xs) 10)."""
    inp = "(λ (_p0) (length (reverse (concat (map (λ (_p1) (+ _p1 0)) _p0) (repeat 8 10)))))"
    out = simplify(parse(inp))
    assert program_size(out) == 5, f"got: {out}"


@pytest.mark.parametrize("inp", [
    "(λ (_p0) (+ _p0 1))",
    "(λ (_p0) (* _p0 2))",
    "(λ (_p0) (length _p0))",
    "(λ (_p0) (reverse _p0))",
    "(λ (_p0) (if (> (first _p0) 0) (cons 1 _p0) (cons 2 _p0)))",
])
def test_no_spurious_simplification(inp: str) -> None:
    """Programs without applicable rewrites should be returned (canonicalised) unchanged in size."""
    ast = parse(inp)
    out = simplify(ast)
    assert program_size(out) == program_size(ast), \
        f"\n  in:  {inp}\n  out: {out}"


def test_int_hole_rejected() -> None:
    from src.lang.ast_nodes import LambdaNode
    ast = LambdaNode(["x"], IntHoleNode())
    with pytest.raises(HoleEncountered):
        simplify(ast)


def test_phi_preserved_property_corpus_sample() -> None:
    """Φ must be preserved on a random sample of corpus programs."""
    try:
        with open("datasets/corpus-a/rl_corpus_no_rule.json") as f:
            corpus = json.load(f)
    except FileNotFoundError:
        pytest.skip("corpus not available")

    random.seed(42)
    sample = random.sample(corpus, 200)
    n_simplified = 0
    n_unchanged = 0
    for entry in sample:
        ast = parse(entry["program"])
        try:
            out = simplify(ast)
        except SimplifyError as e:
            pytest.fail(f"Phi-mismatch on corpus program: {entry['program'][:120]}\n{e}")
        if program_size(out) < program_size(ast):
            n_simplified += 1
        else:
            n_unchanged += 1

    print(f"  simplified={n_simplified}, unchanged={n_unchanged}/{len(sample)}")
