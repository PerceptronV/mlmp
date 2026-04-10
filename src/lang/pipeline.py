"""Top-level orchestrator: enumeration -> warm-start -> RL exploration."""

import json
import os
import pickle
from typing import Callable

from tqdm import tqdm

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
    max_substitutions: int | None = None,
    target_total: int | None = None,
) -> list[TypedProgram]:
    """
    Expand sketch programs into concrete programs by trying constant substitutions.
    Useful for inspection and export; NOT called during RL training.

    ``target_total``, if set, automatically derives ``max_substitutions`` as
    ``ceil(target_total / len(sketches))`` and stops collecting once the
    result list reaches that size.  Use this to expand N sketches to
    approximately T programs without choosing a per-sketch budget manually.
    Actual output may fall below ``target_total`` due to quality/dedup
    filtering, or slightly exceed it on the last sketch.

    ``max_substitutions`` overrides the derived value when set explicitly.
    When the full enumeration of ``seed_constants`` combinations for a
    sketch exceeds this limit, substitutions are sampled randomly instead.
    """
    import itertools
    import math
    import random as _random
    from .enumeration.fingerprint import Fingerprint, FingerprintTable, make_hashable, FAIL
    from .enumeration.filters import passes_quality_filter

    if target_total is not None and max_substitutions is None and len(sketches) > 0:
        max_substitutions = max(1, math.ceil(target_total / len(sketches)))

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
        if target_total is not None and len(results) >= target_total:
            break

        # Wrap open terms in a lambda so fn(inp) works for evaluation
        if isinstance(sketch.ast, LambdaNode):
            compile_node = sketch.ast
            is_wrapped = False
        else:
            compile_node = LambdaNode(["x"], sketch.ast)
            is_wrapped = True

        k = count_holes(sketch.ast)
        n_combos = len(seed_constants) ** k if k > 0 else 1
        if max_substitutions is not None and n_combos > max_substitutions:
            sigma_iter = (
                [_random.choice(seed_constants) for _ in range(k)]
                for _ in range(max_substitutions)
            )
        else:
            sigma_iter = itertools.product(seed_constants, repeat=k)
        for sigma in sigma_iter:
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


def synthesize_corpus(
    grammar: Grammar = DefaultGrammar,
    enum_max_size: int = 9,
    enum_min_variability: float = 0.3,
    enum_max_nesting: int = 2,
    enum_target_programs: int = 6_000_000,
    seed_constants: list[int] | None = None,
    rl_max_depth: int = 12,
    rl_target_programs: int = 100_000,
    rl_max_iterations: int = 5_000_000,
    rl_buffer_capacity: int = 50_000,
    rl_episodes_per_iter: int = 32,
    rl_train_steps_per_iter: int = 8,
    rl_batch_size: int = 64,
    rl_checkpoint_interval: int = 10_000,
    output_dir: str = "output",
) -> None:
    """
    Synthesise a large program corpus for training Model A.

    Phase 1 – Bottom-up enumeration up to ``enum_max_size``, yielding
              sketches (programs with IntHoleNodes for integer literals).
    Phase 2 – Expand every sketch by substituting all ``seed_constants``
              combinations into its holes; deduplicate by behaviour.
              Writes enum_corpus.json  (~6 M programs expected).
    Phase 3 – Train a synthesis policy via warm-start + RL and collect
              ``rl_target_programs`` novel sketches at nesting depth up to
              ``rl_max_depth``.  Expand those sketches and write
              rl_corpus.json  (~4 M programs expected).

    Note on rl_max_depth: this is the MDP depth budget (number of
    APPLY/LAMBDA/IF nodes along the deepest branch), not AST node count.
    The deepest programs in the Rule et al. benchmark reach depth ~11,
    so the default of 12 covers the full benchmark with one step of
    headroom.
    """
    import torch
    import torch.nn.functional as F
    from .rl.mdp import Episode
    from .rl.policy import encode_states, compute_valid_masks
    from .rl.trainer import _fill_in_sketch
    from .enumeration.fingerprint import compute_fingerprint
    from .utils import program_size as _program_size

    if seed_constants is None:
        seed_constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    test_suite = DEFAULT_TEST_SUITE
    valid_instantiations = compute_valid_instantiations(grammar)
    jit = JITCompiler(grammar)

    enum_ckpt = f"{output_dir}/enum_corpus.pkl"
    policy_ckpt = f"{output_dir}/policy_warmstart.pt"

    if os.path.exists(enum_ckpt):
        # ── Resume: load enum corpus from checkpoint ──────────────────────────
        print(f"[resume] Loading enum corpus from {enum_ckpt}")
        with open(enum_ckpt, 'rb') as f:
            quality_corpus, enum_concrete = pickle.load(f)
        print(f"  quality_corpus: {len(quality_corpus):,}, enum_concrete: {len(enum_concrete):,}")
    else:
        # ── Phase 1: Enumeration ──────────────────────────────────────────────
        print("=" * 60)
        print(f"Phase 1: Enumeration  (max_size={enum_max_size})")
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
        quality_corpus = enumerator.extract_corpus(min_variability=enum_min_variability)
        print(f"\nEnumeration: {bank.count()} sketches, {len(quality_corpus)} quality-filtered")
        save_corpus(quality_corpus, f"{output_dir}/enum_sketches.json")

        # ── Phase 2: Sketch Expansion ─────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Phase 2: Sketch Expansion")
        print("=" * 60)

        all_sketches = [
            prog
            for by_size in bank._bank.values()
            for progs in by_size.values()
            for prog in progs
        ]
        print(f"Expanding {len(all_sketches):,} sketches...")

        enum_concrete = expand_sketches(
            all_sketches, seed_constants, test_suite, jit,
            min_variability=enum_min_variability,
            target_total=enum_target_programs,
        )
        print(f"Concrete programs after dedup: {len(enum_concrete):,}")
        save_corpus(enum_concrete, f"{output_dir}/enum_corpus.json")

        # Save pickle checkpoint so future runs can skip enumeration
        with open(enum_ckpt, 'wb') as f:
            pickle.dump((quality_corpus, enum_concrete), f)
        print(f"Saved enum checkpoint to {enum_ckpt}")

    corpus_fingerprints: set = {
        p.fingerprint for p in enum_concrete if p.fingerprint is not None
    }

    # ── Phase 3: RL Collection ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Phase 3: RL Collection  (max_depth={rl_max_depth}, "
          f"target={rl_target_programs:,})")
    print("=" * 60)

    action_vocab = build_action_vocab(grammar, seed_constants, valid_instantiations)
    type_vocab = build_type_vocab(grammar, valid_instantiations)
    func_vocab = build_func_vocab(grammar)

    policy = PolicyNetwork(
        action_vocab_size=len(action_vocab),
        type_vocab_size=len(type_vocab),
        func_vocab_size=len(func_vocab),
    )
    if os.path.exists(policy_ckpt):
        print(f"[resume] Loading warm-started policy from {policy_ckpt}")
        policy.load_state_dict(torch.load(policy_ckpt, weights_only=True))
    else:
        import random as _random
        warmstart_corpus = quality_corpus
        if len(warmstart_corpus) > 100_000:
            warmstart_corpus = _random.sample(warmstart_corpus, 100_000)
            print(f"Subsampled warm-start corpus to {len(warmstart_corpus):,} programs")
        warm_start(
            policy, warmstart_corpus, grammar, action_vocab, type_vocab, func_vocab,
            seed_constants=seed_constants,
            valid_instantiations=valid_instantiations,
            epochs=5,
        )
        torch.save(policy.state_dict(), policy_ckpt)
        print(f"Saved warm-start policy to {policy_ckpt}")
    policy.setup_for_inference(
        action_vocab, type_vocab, func_vocab, grammar, seed_constants,
        valid_instantiations=valid_instantiations,
    )

    # Seed buffer with quality corpus for initial training diversity
    buffer = PriorityQueueBuffer(capacity=rl_buffer_capacity)
    for prog in quality_corpus:
        wrapped = LambdaNode(["x"], prog.ast)
        try:
            traj = extract_trajectory(
                wrapped, Callable[[list[int]], prog.type], grammar,
                valid_instantiations=valid_instantiations,
            )
        except Exception:
            continue
        buffer.insert(
            compute_reward(prog.fingerprint, corpus_fingerprints),
            prog.ast, traj, prog.fingerprint,
        )
    print(f"Buffer seeded with {len(buffer)} programs")

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    rl_sketches: list[TypedProgram] = []
    novel_count = 0
    total_generated = 0

    os.makedirs(output_dir, exist_ok=True)
    rl_ckpt_path = f"{output_dir}/rl_sketches_ckpt.json"

    pbar = tqdm(range(rl_max_iterations), desc="RL collection")
    for iteration in pbar:
        if novel_count >= rl_target_programs:
            break

        # ── Sampling ──────────────────────────────────────────────────────
        for _ in range(rl_episodes_per_iter):
            if novel_count >= rl_target_programs:
                break

            episode = Episode(
                policy, grammar, test_suite, seed_constants, rl_max_depth,
                valid_instantiations=valid_instantiations,
            )
            sketch_ast, trajectory = episode.run()
            total_generated += 1
            if sketch_ast is None:
                continue

            sketch_lambda = (
                sketch_ast
                if isinstance(sketch_ast, LambdaNode)
                else LambdaNode(["x"], sketch_ast)
            )
            _fn, concrete_body, fp = _fill_in_sketch(
                sketch_lambda, seed_constants, test_suite, jit,
                corpus_fingerprints, n_samples=4,
            )
            if fp is None:
                fp = compute_fingerprint(sketch_lambda, test_suite, jit)
                concrete_body = sketch_lambda
            if fp is None:
                continue

            reward = compute_reward(fp, corpus_fingerprints)
            if reward > 0:
                closed = concrete_body if concrete_body is not None else sketch_lambda
                buffer.insert(reward, closed, trajectory, fp)
                if fp not in corpus_fingerprints:
                    corpus_fingerprints.add(fp)
                    rl_sketches.append(TypedProgram(
                        ast=sketch_ast,
                        type=list[int],
                        fingerprint=fp,
                        size=_program_size(sketch_ast),
                    ))
                    novel_count += 1

        # ── Training ──────────────────────────────────────────────────────
        if len(buffer) >= rl_batch_size:
            for _ in range(rl_train_steps_per_iter):
                batch = buffer.sample(rl_batch_size)
                all_transitions = [
                    (s, a, r)
                    for r, _prog, traj in batch
                    for s, a in traj
                    if a in action_vocab
                ]
                if not all_transitions:
                    continue
                states, actions, rewards_list = zip(*all_transitions)
                state_batch = encode_states(states, type_vocab, func_vocab)
                action_indices = torch.tensor(
                    [action_vocab[a] for a in actions], dtype=torch.long,
                )
                reward_weights = torch.tensor(rewards_list, dtype=torch.float32)
                valid_masks = compute_valid_masks(
                    states, grammar, seed_constants, action_vocab,
                    valid_instantiations=valid_instantiations,
                )
                log_probs = policy(state_batch, valid_masks)
                loss = -(
                    reward_weights
                    * log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

        # ── Checkpoint ────────────────────────────────────────────────────
        if (iteration + 1) % rl_checkpoint_interval == 0:
            save_corpus_asts([p.ast for p in rl_sketches], rl_ckpt_path)

        pbar.set_postfix(novel=novel_count, gen=total_generated, buf=len(buffer))

    # ── Post-RL Expansion ─────────────────────────────────────────────────────
    print(f"\nExpanding {len(rl_sketches):,} RL sketches...")
    rl_concrete = expand_sketches(
        rl_sketches, seed_constants, test_suite, jit,
        min_variability=enum_min_variability,
        max_substitutions=100,  # cap per sketch; RL programs can have many holes
        target_total=4_000_000,
    )
    # Deduplicate against everything already discovered (enum + RL sampling)
    rl_unique = [p for p in rl_concrete if p.fingerprint not in corpus_fingerprints]
    print(f"RL concrete programs (novel vs all prior): {len(rl_unique):,}")
    save_corpus(rl_unique, f"{output_dir}/rl_corpus.json")

    print("\n" + "=" * 60)
    print("Synthesis complete")
    print("=" * 60)
    print(f"  Enum corpus:  {len(enum_concrete):,}  →  {output_dir}/enum_corpus.json")
    print(f"  RL corpus:    {len(rl_unique):,}  →  {output_dir}/rl_corpus.json")
    print(f"  Total:        {len(enum_concrete) + len(rl_unique):,}")


if __name__ == "__main__":
    run_pipeline()
