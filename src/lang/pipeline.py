"""Top-level orchestrator: enumeration -> warm-start -> RL exploration."""

import json
import os
from typing import Callable

from .grammar import Grammar, DefaultGrammar
from .ast_nodes import ASTNode, LambdaNode, IntHoleNode
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


def expand_sketches(
    sketches: list[TypedProgram],
    seed_constants: list[int],
    test_suite: list,
    jit: 'JITCompiler',
    min_variability: float = 0.3,
    min_successes: int = 3,
) -> list[TypedProgram]:
    """
    Expand sketch programs into concrete programs by trying all constant substitutions.
    Useful for inspection and export; NOT called during RL training.
    """
    import itertools
    from .enumeration.fingerprint import Fingerprint, FingerprintTable, make_hashable, FAIL
    from .enumeration.filters import passes_quality_filter

    def count_holes(node) -> int:
        if isinstance(node, IntHoleNode):
            return 1
        elif hasattr(node, '__dataclass_fields__'):
            total = 0
            for field_name in node.__dataclass_fields__:
                child = getattr(node, field_name)
                if isinstance(child, ASTNode):
                    total += count_holes(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ASTNode):
                            total += count_holes(item)
            return total
        return 0

    results = []
    seen_fps = FingerprintTable()

    for sketch in sketches:
        # Wrap open terms in a lambda so fn(inp) works for evaluation
        if isinstance(sketch.ast, LambdaNode):
            compile_node = sketch.ast
            is_wrapped = False
        else:
            compile_node = LambdaNode(["x"], sketch.ast)
            is_wrapped = True

        k = count_holes(sketch.ast)
        for sigma in itertools.product(seed_constants, repeat=k):
            sigma_list = list(sigma)
            try:
                fn, concrete_lambda = jit.compile(compile_node, sigma_list)
                # Unwrap to get the concrete body if we wrapped it
                concrete_ast = concrete_lambda.body if is_wrapped else concrete_lambda
                outputs = []
                for inp in test_suite:
                    try:
                        outputs.append(make_hashable(fn(inp)))
                    except Exception:
                        outputs.append(FAIL)
                fp = Fingerprint(tuple(outputs))
                if not seen_fps.contains(fp):
                    if passes_quality_filter(fp, min_successes=min_successes, min_variability=min_variability):
                        seen_fps.insert(fp, concrete_ast)
                        results.append(TypedProgram(
                            ast=concrete_ast,
                            type=sketch.type,
                            fingerprint=fp,
                            size=sketch.size,
                            substitution=sigma_list,
                        ))
            except Exception:
                continue

    return results


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
