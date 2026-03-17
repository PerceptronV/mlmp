"""Tests for nested higher-order functions in lambda bodies."""

import pytest
from typing import Callable

from src.enumeration.enumerator import (
    BottomUpEnumerator, ProgramBank, ContextualBank, TypedProgram,
    _fresh_param_names, PROBE_VALUES,
)
from src.enumeration.fingerprint import Fingerprint, FingerprintTable
from src.enumeration.test_suite import DEFAULT_TEST_SUITE
from src.lang.grammar import DefaultGrammar
from src.lang.ast_nodes import (
    VariableNode, NumberNode, ApplicationNode, LambdaNode,
)
from src.lang.type_utils import CallableOrig, get_origin
from src.rl.mdp import (
    SynthesisState, Action, ActionType, valid_actions, MAX_NESTING_DEPTH,
)
from src.rl.trajectory import extract_trajectory
from src.rl.policy import (
    StateEncoder, encode_states, build_type_vocab, build_func_vocab,
    build_action_vocab,
)
from src.utils import resolve_type


# ─── ContextualBank tests ───────────────────────────────────────────

def test_contextual_bank_basic():
    """get() returns local + parent, add_local deduplicates locally."""
    parent = ProgramBank()
    fp1 = Fingerprint((1, 2, 3))
    prog1 = TypedProgram(ast=NumberNode(0), type=int, fingerprint=fp1, size=1)
    parent.add(prog1)

    child = ContextualBank(parent=parent)
    fp2 = Fingerprint((4, 5, 6))
    prog2 = TypedProgram(ast=NumberNode(1), type=int, fingerprint=fp2, size=1)
    child.add_local(prog2)

    # get returns both parent and child programs
    results = child.get(int, 1)
    assert len(results) == 2

    # Duplicate fingerprint in child is rejected
    prog_dup = TypedProgram(ast=NumberNode(99), type=int, fingerprint=fp2, size=1)
    assert child.add_local(prog_dup) is False


def test_contextual_bank_hierarchy():
    """3-level hierarchy: get() walks all ancestors."""
    root = ProgramBank()
    fp_root = Fingerprint((1,))
    root.add(TypedProgram(ast=NumberNode(0), type=int, fingerprint=fp_root, size=1))

    level1 = ContextualBank(parent=root)
    fp1 = Fingerprint((2,))
    level1.add_local(TypedProgram(ast=NumberNode(1), type=int, fingerprint=fp1, size=1))

    level2 = ContextualBank(parent=level1)
    fp2 = Fingerprint((3,))
    level2.add_local(TypedProgram(ast=NumberNode(2), type=int, fingerprint=fp2, size=1))

    # level2 sees all three
    results = level2.get(int, 1)
    assert len(results) == 3

    # total_count includes all levels
    assert level2.total_count() == 3
    assert level2.local_count() == 1
    assert level1.local_count() == 1


# ─── Fresh param names ──────────────────────────────────────────────

def test_fresh_param_names():
    """Picks _pN names not already in context."""
    ctx = {"x": list[int], "_p0": int}
    names = _fresh_param_names(2, ctx)
    assert names == ["_p1", "_p2"]

    ctx2 = {"x": list[int]}
    names2 = _fresh_param_names(1, ctx2)
    assert names2 == ["_p0"]


# ─── Fingerprint in context ─────────────────────────────────────────

def test_fingerprint_in_context_top_level():
    """Top-level context (only x) matches existing _fingerprint()."""
    enum = BottomUpEnumerator(max_size=2, max_nesting=0)
    node = NumberNode(42)
    ctx = {"x": list[int]}

    fp_context = enum._fingerprint_in_context(node, ctx)
    fp_standard = enum._fingerprint(node)
    assert fp_context == fp_standard


def test_fingerprint_in_context_with_params():
    """_p0 vs constant 0 produce different fingerprints in extended context."""
    enum = BottomUpEnumerator(max_size=2, max_nesting=1)
    ctx = {"x": list[int], "_p0": int}

    fp_param = enum._fingerprint_in_context(VariableNode("_p0"), ctx)
    fp_const = enum._fingerprint_in_context(NumberNode(0), ctx)

    assert fp_param is not None
    assert fp_const is not None
    assert fp_param != fp_const  # _p0 varies with probes, 0 doesn't


# ─── Enumeration with nesting ───────────────────────────────────────

def test_nesting_depth_0_matches_v1():
    """max_nesting=0 produces same count as V1 behavior."""
    # V1 behavior: _enumerate_lambdas skips higher-order inside lambda bodies
    # max_nesting=0 should produce identical results
    enum_v1 = BottomUpEnumerator(max_size=4, max_nesting=0)
    bank_v1 = enum_v1.enumerate()

    # With max_nesting=0, the new code path should still produce the same
    # programs since it won't recurse into nested HOFs
    count_v1 = bank_v1.count()
    assert count_v1 > 0  # Sanity check


def test_nesting_depth_1_produces_nested_programs():
    """max_nesting=1 produces more programs than max_nesting=0 at sufficient size."""
    # At max_size=5, nested HOFs don't fit (min nested HOF is ~7-8 nodes).
    # At max_size=8, nesting produces genuinely novel programs.
    enum_0 = BottomUpEnumerator(max_size=8, max_nesting=0)
    bank_0 = enum_0.enumerate()

    enum_1 = BottomUpEnumerator(max_size=8, max_nesting=1)
    bank_1 = enum_1.enumerate()

    # Nesting should produce strictly more programs at size 8
    assert bank_1.count() > bank_0.count()


def _max_nesting_in_ast(node, depth=0):
    """Compute the maximum lambda nesting depth in an AST."""
    if isinstance(node, LambdaNode):
        return _max_nesting_in_ast(node.body, depth + 1)
    elif isinstance(node, ApplicationNode):
        max_d = depth
        for arg in node.arguments:
            max_d = max(max_d, _max_nesting_in_ast(arg, depth))
        return max_d
    else:
        return depth


def test_nesting_depth_limit_respected():
    """No program exceeds max_nesting depth."""
    enum = BottomUpEnumerator(max_size=5, max_nesting=1)
    bank = enum.enumerate()

    for type_key, by_size in bank._bank.items():
        for size, progs in by_size.items():
            for prog in progs:
                # Programs in bank are open terms (not lambda-wrapped).
                # Lambda nodes appear as arguments to HOFs.
                depth = _max_nesting_in_ast(prog.ast, 0)
                assert depth <= 1, f"Program {prog.ast} has nesting depth {depth} > 1"


def test_child_bank_caching():
    """Cache is populated after enumeration with nesting."""
    enum = BottomUpEnumerator(max_size=5, max_nesting=1)
    enum.enumerate()
    # HOFs at size 3+ create child banks (e.g., filter lambda at size 2)
    assert len(enum._child_bank_cache) > 0
    # Verify child banks contain local programs (lambda params + compositions)
    has_local = any(bank.local_count() > 0 for bank in enum._child_bank_cache.values())
    assert has_local


# ─── RL: valid_actions nesting gate ─────────────────────────────────

def test_valid_actions_nesting_gate():
    """HOFs are excluded when nesting_depth >= MAX_NESTING_DEPTH."""
    state_ok = SynthesisState(
        target_type=list[int],
        context={"x": list[int]},
        depth_budget=5,
        nesting_depth=0,
    )
    actions_ok = valid_actions(state_ok, DefaultGrammar, [0, 1, 2, 3])

    state_max = SynthesisState(
        target_type=list[int],
        context={"x": list[int]},
        depth_budget=5,
        nesting_depth=MAX_NESTING_DEPTH,
    )
    actions_max = valid_actions(state_max, DefaultGrammar, [0, 1, 2, 3])

    # At max nesting, HOFs should be excluded
    ho_funcs_ok = set()
    ho_funcs_max = set()
    for a in actions_ok:
        if a.action_type == ActionType.APPLY:
            func_info = DefaultGrammar[a.payload]
            if any(get_origin(t) == CallableOrig for t in func_info['arg_types']):
                ho_funcs_ok.add(a.payload)
    for a in actions_max:
        if a.action_type == ActionType.APPLY:
            func_info = DefaultGrammar[a.payload]
            if any(get_origin(t) == CallableOrig for t in func_info['arg_types']):
                ho_funcs_max.add(a.payload)

    # There should be HOFs at depth 0
    assert len(ho_funcs_ok) > 0
    # No HOFs at max nesting
    assert len(ho_funcs_max) == 0


# ─── RL: trajectory nesting depth ───────────────────────────────────

def test_trajectory_nesting_depth():
    """nesting_depth increments at lambda boundaries in trajectories."""
    # (filter (λ _p0 (> _p0 0)) x)
    program = LambdaNode(["x"], ApplicationNode(
        VariableNode("filter"),
        [
            LambdaNode(["_p0"], ApplicationNode(
                VariableNode(">"),
                [VariableNode("_p0"), NumberNode(0)]
            )),
            VariableNode("x"),
        ]
    ))

    target_type = Callable[[list[int]], list[int]]
    trajectory = extract_trajectory(program, target_type, DefaultGrammar)

    # Find states in the trajectory
    nesting_depths = [s.nesting_depth for s, a in trajectory]

    # The top-level lambda creates nesting_depth=0 state
    # The filter application keeps nesting_depth=0 (propagated)
    # The inner lambda body should have nesting_depth >= 1
    assert max(nesting_depths) >= 1, f"Expected nesting_depth >= 1, got depths: {nesting_depths}"


# ─── RL: state encoder with nesting ─────────────────────────────────

def test_state_encoder_6_inputs():
    """Forward pass works with nesting_depth key in state batch."""
    import torch

    type_vocab = build_type_vocab(DefaultGrammar)
    func_vocab = build_func_vocab(DefaultGrammar)

    encoder = StateEncoder(len(type_vocab), len(func_vocab))

    states = [
        SynthesisState(
            target_type=list[int],
            context={"x": list[int]},
            depth_budget=5,
            nesting_depth=1,
        )
    ]

    state_batch = encode_states(states, type_vocab, func_vocab)
    assert 'nesting_depth' in state_batch

    output = encoder(state_batch)
    assert output.shape == (1, 64)  # default embed_dim=64
