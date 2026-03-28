"""Top-level orchestrator: enumeration -> warm-start -> RL exploration."""

import json
import os
from typing import Callable

from .grammar import Grammar, DefaultGrammar
from .ast_nodes import ASTNode, LambdaNode
from .compiler import JITCompiler
from .enumeration.enumerator import BottomUpEnumerator, TypedProgram
from .enumeration.test_suite import DEFAULT_TEST_SUITE
from .enumeration.fingerprint import Fingerprint
from .rl.policy import (
    PolicyNetwork, build_action_vocab, build_type_vocab, build_func_vocab,
)
from .rl.trajectory import extract_trajectory
from .rl.reward import compute_reward
from .rl.priority_queue import PriorityQueueBuffer
from .rl.trainer import warm_start, train_rl
from .utils import compute_valid_instantiations


def save_corpus(corpus: list[TypedProgram], path: str):
    """Serialize programs as S-expressions to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entries = []
    for prog in corpus:
        entries.append({
            'program': str(prog.ast),
            'type': str(prog.type),
            'size': prog.size,
        })
    with open(path, 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"Saved {len(entries)} programs to {path}")


def save_corpus_asts(asts: list[ASTNode], path: str):
    """Serialize AST list to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entries = [{'program': str(ast)} for ast in asts]
    with open(path, 'w') as f:
        json.dump(entries, f, indent=2)
    print(f"Saved {len(entries)} programs to {path}")


def run_pipeline(
    grammar: Grammar = DefaultGrammar,
    enum_max_size: int = 5,
    enum_min_variability: float = 0.3,
    enum_max_nesting: int = 2,
    rl_iterations: int = 10000,
    rl_max_depth: int = 8,
    buffer_capacity: int = 5000,
    seed_constants: list[int] | None = None,
    output_dir: str = "output",
):
    """
    Full pipeline: enumeration -> warm-start -> RL exploration.
    """
    if seed_constants is None:
        seed_constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    test_suite = DEFAULT_TEST_SUITE

    # Compute valid instantiations once for the whole pipeline
    valid_instantiations = compute_valid_instantiations(grammar)

    # === Phase 1: Enumeration ===
    print("=" * 60)
    print("Phase 1: Bottom-Up Enumeration")
    print("=" * 60)

    enumerator = BottomUpEnumerator(
        grammar=grammar,
        test_suite=test_suite,
        seed_constants=seed_constants,
        max_size=enum_max_size,
        min_variability=enum_min_variability,
        max_nesting=enum_max_nesting,
    )
    bank = enumerator.enumerate()
    corpus = enumerator.extract_corpus(min_variability=enum_min_variability)

    print(f"\nEnumeration complete:")
    print(f"  Total programs in bank: {bank.count()}")
    print(f"  Quality-filtered corpus: {len(corpus)}")

    save_corpus(corpus, f"{output_dir}/enumeration_corpus.json")

    # === Phase 2: Warm-Start ===
    print("\n" + "=" * 60)
    print("Phase 2: Warm-Start via Behavioural Cloning")
    print("=" * 60)

    action_vocab = build_action_vocab(grammar, seed_constants, valid_instantiations)
    type_vocab = build_type_vocab(grammar, valid_instantiations)
    func_vocab = build_func_vocab(grammar)

    policy = PolicyNetwork(
        action_vocab_size=len(action_vocab),
        type_vocab_size=len(type_vocab),
        func_vocab_size=len(func_vocab),
    )

    warm_start(
        policy, corpus, grammar, action_vocab, type_vocab, func_vocab,
        seed_constants=seed_constants,
        valid_instantiations=valid_instantiations,
    )

    # Seed the priority queue buffer with the enumeration corpus
    buffer = PriorityQueueBuffer(capacity=buffer_capacity)
    corpus_fingerprints: set[Fingerprint] = set()

    for prog in corpus:
        target_type = Callable[[list[int]], prog.type]
        wrapped = LambdaNode(["x"], prog.ast)
        try:
            traj = extract_trajectory(
                wrapped, target_type, grammar,
                valid_instantiations=valid_instantiations,
            )
        except Exception:
            corpus_fingerprints.add(prog.fingerprint)
            continue
        reward = compute_reward(prog.fingerprint, corpus_fingerprints)
        buffer.insert(reward, prog.ast, traj, prog.fingerprint)
        corpus_fingerprints.add(prog.fingerprint)

    print(f"Buffer seeded with {len(buffer)} programs")

    # === Phase 3: RL Exploration ===
    print("\n" + "=" * 60)
    print("Phase 3: RL Exploration")
    print("=" * 60)

    train_rl(
        policy=policy,
        buffer=buffer,
        grammar=grammar,
        test_suite=test_suite,
        action_vocab=action_vocab,
        type_vocab=type_vocab,
        func_vocab=func_vocab,
        corpus_fingerprints=corpus_fingerprints,
        n_iterations=rl_iterations,
        max_depth=rl_max_depth,
        seed_constants=seed_constants,
        valid_instantiations=valid_instantiations,
    )

    # === Final Output ===
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    final_asts = [entry[2] for entry in buffer.buffer]
    print(f"Total unique programs: {len(corpus_fingerprints)}")
    print(f"Buffer size: {len(buffer)}")

    save_corpus_asts(final_asts, f"{output_dir}/final_corpus.json")


if __name__ == "__main__":
    run_pipeline()
