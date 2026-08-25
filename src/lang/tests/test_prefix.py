"""Tests for the legal-prefix automaton.

The two that matter are the round-trip pair: replaying real corpus programs
proves the mask never blocks a valid program (not too strict), and walking
random legal paths proves everything it allows parses (not too loose).
"""

import json
import random
from pathlib import Path

import pytest

from src.lang.grammar import DefaultGrammar, get_grammar
from src.lang.lexer import tokenise
from src.lang.parser import parse
from src.lang.prefix import PrefixError, PrefixState

CORPUS = Path('datasets/corpus-a/enum_corpus_no_rule.json')


def _tokens(program: str) -> list[str]:
    return [t.value for t in tokenise(program) if t.value]


def _replay(program: str, **kwargs) -> PrefixState:
    """Feed a real program through the automaton, asserting every token it
    actually uses was offered as legal."""
    state = PrefixState(**kwargs)
    for i, tok in enumerate(_tokens(program)):
        legal = state.legal()
        assert tok in legal, (
            f"token {i} ({tok!r}) of {program!r} was not offered; "
            f"legal was {sorted(legal)[:12]}")
        state.advance(tok)
    return state


HAND_WRITTEN = [
    "(λ (_p0) _p0)",
    "(λ (_p0) (take 1 _p0))",
    "(λ (_p0) (concat _p0 (cons (length _p0) [])))",
    "(λ (_p0) (map (λ (_p1) (+ _p1 1)) _p0))",
    "(λ (_p0) (fold + 0 _p0))",
    "(λ (_p0) (if (< (length _p0) 3) _p0 (take 2 _p0)))",
    "(λ (_p0) [1 2 3])",
]


@pytest.mark.parametrize("program", HAND_WRITTEN)
def test_accepts_well_formed_programs(program):
    state = _replay(program)
    assert state.complete
    assert state.legal() == frozenset()  # only <end> may follow


def test_scope_is_tracked():
    state = PrefixState()
    for tok in _tokens("(λ (_p0) (take 1 "):
        state.advance(tok)
    assert '_p0' in state.legal(), "the bound parameter must be referenceable"
    assert '_p1' not in state.legal(), "an unbound variable must be masked out"


def test_arity_is_enforced():
    state = PrefixState()
    # `take` has arity 2: after one argument the closing paren is still illegal.
    for tok in _tokens("(λ (_p0) (take 1"):
        state.advance(tok)
    assert ')' not in state.legal()
    state.advance('_p0')
    assert state.legal() == frozenset({')'}), "arity satisfied -> must close"


def test_comment_marker_is_never_legal():
    """'#' comments out the rest of the line, so a decoder must never emit it."""
    state = PrefixState()
    seen = set()
    for tok in _tokens("(λ (_p0) (concat _p0 (cons (length _p0) [])))"):
        seen |= state.legal()
        state.advance(tok)
    assert '#' not in seen


def test_end_only_offered_once_complete():
    state = PrefixState()
    toks = _tokens("(λ (_p0) _p0)")
    for tok in toks[:-1]:
        state.advance(tok)
        assert not state.complete
    state.advance(toks[-1])
    assert state.complete


def test_illegal_token_raises():
    state = PrefixState()
    with pytest.raises(PrefixError):
        state.advance(')')


def _random_walk(rng: random.Random, state: PrefixState, max_tokens: int = 60):
    """Sample uniformly from the legal set, biased toward closing so the walk
    terminates, and return the emitted token string."""
    out = []
    for _ in range(max_tokens):
        if state.complete:
            break
        legal = sorted(state.legal())
        closers = [t for t in legal if t in (')', ']')]
        # Past a few tokens, prefer closing; otherwise deep nesting rarely ends.
        pick = rng.choice(closers) if (closers and len(out) > 8 and rng.random() < 0.6) \
            else rng.choice(legal)
        state.advance(pick)
        out.append(pick)
    return ' '.join(out)


def test_random_legal_walks_always_parse():
    """Anything the automaton allows must parse — the 'not too loose' half."""
    rng = random.Random(0)
    completed = 0
    for _ in range(300):
        state = PrefixState()
        program = _random_walk(rng, state)
        if not state.complete:
            continue  # hit the token budget mid-program; nothing to assert
        completed += 1
        parse(program)  # raises ParseError if the mask let something through
    assert completed > 50, f"only {completed} walks completed; test is too weak"


def test_random_legal_walks_parse_under_a_subset_grammar():
    rng = random.Random(1)
    tiny = get_grammar('tiny')
    completed = 0
    for _ in range(200):
        state = PrefixState(grammar=tiny)
        program = _random_walk(rng, state)
        if state.complete:
            completed += 1
            parse(program)
    assert completed > 30


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not present")
def test_replays_the_training_corpus():
    """The 'not too strict' half: every program the model is trained to emit
    must be reachable through the mask."""
    with open(CORPUS) as f:
        entries = json.load(f)
    rng = random.Random(0)
    sample = rng.sample(entries, min(20000, len(entries)))
    for entry in sample:
        state = _replay(entry['program'])
        assert state.complete, f"{entry['program']!r} left the automaton open"


def test_arity_resolves_through_a_symbol_shuffling_map():
    """Under symbol shuffling the model emits permuted names, so arity must
    follow the *original* function, not the name it was printed as."""
    # 'length' (arity 1) is printed as 'take', and 'take' (arity 2) as 'length'.
    name_map = {'length': 'take', 'take': 'length'}
    state = PrefixState(name_map=name_map)
    for tok in _tokens("(λ (_p0) (take _p0"):
        state.advance(tok)
    # 'take' here *means* length: one argument, so it must close now.
    assert state.legal() == frozenset({')'})

    state = PrefixState(name_map=name_map)
    for tok in _tokens("(λ (_p0) (length 1"):
        state.advance(tok)
    # 'length' here *means* take: a second argument is still owed.
    assert ')' not in state.legal()
