"""Priority queue RL on Rule et al. programs.

Loads pre-converted Rule et al. programs from src/data/rule/functions.txt,
fingerprints and deduplicates them, then runs warm-start + priority queue RL.

Programs whose trajectories contain actions outside the MDP's valid action set
(due to polymorphic type resolution limitations) are filtered out to avoid
corrupted training signal.

Usage:
    python scripts/run_rule_rl.py
    python scripts/run_rule_rl.py --rl-iterations 2000 --episodes-per-iter 64

Initial results:

  ┌──────────────────────────────────────┬─────────────────────┐
  │                Metric                │        Value        │
  ├──────────────────────────────────────┼─────────────────────┤
  │ Rule et al. corpus (all)             │ 212 unique programs │
  ├──────────────────────────────────────┼─────────────────────┤
  │ MDP-compatible (warm-start + buffer) │ 171                 │
  ├──────────────────────────────────────┼─────────────────────┤
  │ Rule et al. unique fingerprints      │ 212                 │
  ├──────────────────────────────────────┼─────────────────────┤
  │ Total fingerprints after RL          │ 25,326              │
  ├──────────────────────────────────────┼─────────────────────┤
  │ Novel programs discovered by RL      │ 25,114              │
  ├──────────────────────────────────────┼─────────────────────┤
  │ RL time                              │ 5,303s (~88 min)    │
  ├──────────────────────────────────────┼─────────────────────┤
  │ Buffer size                          │ 25,286              │
  └──────────────────────────────────────┴─────────────────────┘

  So starting from 212 Rule et al. programs, RL discovered 25,114 novel behaviorally-distinct programs over 1000 iterations (32 episodes each =
  32,000 episodes total). That's a ~120x expansion of the program corpus. The discovery rate was roughly 25 novel programs per iteration,
  sustained throughout the run.

  The warm-start loss dropped from 2.89 to ~1.24 over 50 epochs, confirming the trajectory filtering fixed the inf loss issue.

  The generated programs do tend to be quite large/complex (lots of nested cons, concat, splice). Many have reward=1.0 (full novelty + high
  variability). The script is at scripts/run_rule_rl.py.

"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from typing import Callable

from tqdm import tqdm

from src.lang.enumeration.fingerprint import Fingerprint, compute_fingerprint
from src.lang.enumeration.test_suite import DEFAULT_TEST_SUITE
from src.lang.grammar import DefaultGrammar
from src.lang.ast_nodes import ASTNode, NumberNode, LambdaNode
from src.lang.parser import parse
from src.lang.compiler import JITCompiler
from src.lang.rl.policy import (
    PolicyNetwork, build_action_vocab, build_type_vocab, build_func_vocab,
)
from src.lang.rl.trajectory import extract_trajectory
from src.lang.rl.mdp import valid_actions
from src.lang.rl.reward import compute_reward
from src.lang.rl.priority_queue import PriorityQueueBuffer
from src.lang.rl.trainer import warm_start, train_rl
from src.lang.enumeration.enumerator import TypedProgram


def collect_constants(node: ASTNode) -> set[int]:
    """Walk an AST and collect all integer literal constants."""
    constants = set()
    if isinstance(node, NumberNode):
        constants.add(node.value)
    elif isinstance(node, LambdaNode):
        constants.update(collect_constants(node.body))
    elif hasattr(node, 'arguments'):  # ApplicationNode
        constants.update(collect_constants(node.function))
        for arg in node.arguments:
            constants.update(collect_constants(arg))
    elif hasattr(node, 'condition'):  # IfNode
        constants.update(collect_constants(node.condition))
        constants.update(collect_constants(node.then_expr))
        constants.update(collect_constants(node.else_expr))
    elif hasattr(node, 'elements'):  # ListNode
        for elem in node.elements:
            constants.update(collect_constants(elem))
    return constants


def trajectory_is_valid(traj, grammar, seed_constants):
    """Check that every action in a trajectory is within the valid action set.

    Returns False if any (state, action) pair has the action outside the set
    returned by valid_actions(state, grammar, seed_constants). This catches
    programs where polymorphic type resolution (T1=int, T2=int) misassigns
    types, e.g. map with a lambda returning list[int] instead of int.
    """
    for state, action in traj:
        if action not in valid_actions(state, grammar, seed_constants):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Priority queue RL on Rule et al. programs")
    parser.add_argument("--rl-iterations", type=int, default=1000)
    parser.add_argument("--episodes-per-iter", type=int, default=32)
    parser.add_argument("--rl-max-depth", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--warm-start-epochs", type=int, default=50)
    parser.add_argument("--programs-file", type=str,
                        default=str(project_root / "src" / "data" / "rule" / "functions.txt"))
    args = parser.parse_args()

    grammar = DefaultGrammar
    test_suite = DEFAULT_TEST_SUITE
    target_type = Callable[[list[int]], list[int]]

    # === Phase 1: Load Rule et al. programs ===
    print("=" * 60)
    print("Phase 1: Loading Rule et al. programs")
    print("=" * 60)

    t0 = time.time()
    programs_file = Path(args.programs_file)
    lines = programs_file.read_text().strip().splitlines()
    print(f"Read {len(lines)} program lines from {programs_file.name}")

    # Parse all programs and collect constants
    jit = JITCompiler(grammar)
    parsed_programs: list[tuple[LambdaNode, str]] = []
    all_constants: set[int] = set()
    parse_failures = 0

    for line in tqdm(lines, desc="Parsing"):
        line = line.strip()
        if not line:
            continue
        try:
            ast = parse(line)
            if not isinstance(ast, LambdaNode):
                parse_failures += 1
                continue
            parsed_programs.append((ast, line))
            all_constants.update(collect_constants(ast))
        except Exception:
            parse_failures += 1

    print(f"Parsed {len(parsed_programs)} programs in {time.time() - t0:.1f}s")
    if parse_failures:
        print(f"  ({parse_failures} parse failures)")
    print(f"Unique integer constants found: {sorted(all_constants)}")

    # Build seed_constants from all constants found in programs
    seed_constants = sorted(all_constants)
    print(f"Using {len(seed_constants)} seed constants")

    # Fingerprint and deduplicate
    t1 = time.time()
    seen_fps: set[Fingerprint] = set()
    corpus: list[TypedProgram] = []

    for ast, line in tqdm(parsed_programs, desc="Fingerprinting"):
        fp = compute_fingerprint(ast, test_suite, jit)
        if fp is None:
            continue
        if fp in seen_fps:
            continue
        seen_fps.add(fp)
        # Store the body (without the outer lambda) for consistency with TypedProgram
        corpus.append(TypedProgram(ast=ast.body, type=list[int], fingerprint=fp, size=len(line)))

    print(f"Fingerprinted & deduplicated: {len(corpus)} unique programs in {time.time() - t1:.1f}s")

    # Pre-extract trajectories and filter out programs with invalid actions.
    # Some Rule programs use HOFs with non-int return types (e.g. map with
    # lambda returning list[int]) which the MDP's type resolution doesn't
    # handle, producing trajectory actions outside the valid set.
    valid_corpus: list[TypedProgram] = []
    valid_trajs: list[list] = []
    type_mismatch = 0

    for prog in tqdm(corpus, desc="Validating trajectories"):
        wrapped = LambdaNode(["x"], prog.ast)
        try:
            traj = extract_trajectory(wrapped, target_type, grammar)
        except Exception:
            type_mismatch += 1
            continue
        if not trajectory_is_valid(traj, grammar, seed_constants):
            type_mismatch += 1
            continue
        valid_corpus.append(prog)
        valid_trajs.append(traj)

    print(f"MDP-compatible programs: {len(valid_corpus)}/{len(corpus)}")
    if type_mismatch:
        print(f"  ({type_mismatch} dropped due to type resolution mismatch)")

    enum_fingerprints = {p.fingerprint for p in corpus}  # all Rule fps (for novelty)
    print(f"Unique fingerprints (all Rule): {len(enum_fingerprints)}")

    # === Phase 2: Build policy and warm-start ===
    print("\n" + "=" * 60)
    print("Phase 2: Warm-Start")
    print("=" * 60)

    action_vocab = build_action_vocab(grammar, seed_constants)
    type_vocab = build_type_vocab(grammar)
    func_vocab = build_func_vocab(grammar)

    print(f"Action vocab size: {len(action_vocab)}")
    print(f"Type vocab size: {len(type_vocab)}")
    print(f"Func vocab size: {len(func_vocab)}")

    policy = PolicyNetwork(
        action_vocab_size=len(action_vocab),
        type_vocab_size=len(type_vocab),
        func_vocab_size=len(func_vocab),
    )

    warm_start(
        policy, valid_corpus, grammar, action_vocab, type_vocab, func_vocab,
        seed_constants=seed_constants, epochs=args.warm_start_epochs, batch_size=128,
    )

    # === Phase 3: Seed buffer ===
    print("\n" + "=" * 60)
    print("Phase 3: Seed Buffer")
    print("=" * 60)

    buffer = PriorityQueueBuffer(capacity=args.buffer_capacity)
    corpus_fingerprints: set[Fingerprint] = set()

    seeded = 0
    for prog, traj in tqdm(zip(valid_corpus, valid_trajs), total=len(valid_corpus), desc="Seeding buffer"):
        reward = compute_reward(prog.fingerprint, corpus_fingerprints)
        buffer.insert(reward, prog.ast, traj, prog.fingerprint)
        corpus_fingerprints.add(prog.fingerprint)
        seeded += 1

    # Also register the dropped programs' fingerprints so RL knows about them
    for prog in corpus:
        corpus_fingerprints.add(prog.fingerprint)

    print(f"Buffer seeded with {seeded} programs (of {len(valid_corpus)} valid)")
    print(f"Corpus fingerprints: {len(corpus_fingerprints)}")

    # === Phase 4: RL ===
    print("\n" + "=" * 60)
    print(f"Phase 4: RL Exploration ({args.rl_iterations} iterations)")
    print("=" * 60)

    t_rl = time.time()
    train_rl(
        policy=policy,
        buffer=buffer,
        grammar=grammar,
        test_suite=test_suite,
        action_vocab=action_vocab,
        type_vocab=type_vocab,
        func_vocab=func_vocab,
        corpus_fingerprints=corpus_fingerprints,
        n_iterations=args.rl_iterations,
        episodes_per_iter=args.episodes_per_iter,
        train_steps_per_iter=8,
        batch_size=128,
        lr=1e-4,
        max_depth=args.rl_max_depth,
        seed_constants=seed_constants,
    )
    rl_time = time.time() - t_rl

    # === Report ===
    novel = len(corpus_fingerprints) - len(enum_fingerprints)
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Rule et al. corpus (all):                  {len(corpus)}")
    print(f"MDP-compatible (warm-start + buffer):      {len(valid_corpus)}")
    print(f"Rule et al. unique fingerprints:           {len(enum_fingerprints)}")
    print(f"Total fingerprints after RL:               {len(corpus_fingerprints)}")
    print(f"Novel programs discovered by RL:           {novel}")
    print(f"RL time:                                   {rl_time:.1f}s")
    print(f"Buffer size:                               {len(buffer)}")
    print(f"Buffer min reward:                         {buffer.min_reward():.3f}")

    # Show some novel programs
    print(f"\nSample novel RL-discovered programs:")
    count = 0
    for reward, _id, prog, traj, fp in sorted(buffer.buffer, reverse=True):
        if fp not in enum_fingerprints and count < 20:
            print(f"  reward={reward:.3f}  {prog}")
            count += 1


if __name__ == "__main__":
    main()
