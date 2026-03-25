"""Run RL on top of enumerated list[int] -> list[int] corpus and report novel discoveries.

Loads pre-enumerated programs from datasets/enumerated_list_int/ JSON batches,
fingerprints them, then runs warm-start + RL.

Usage:
    python scripts/run_rl_experiment.py                  # all sizes
    python scripts/run_rl_experiment.py --max-size 5     # only programs up to size 5
"""

import sys
import json
import glob
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from typing import Callable

from tqdm import tqdm

from src.enumeration.enumerator import TypedProgram
from src.enumeration.fingerprint import Fingerprint, compute_fingerprint
from src.enumeration.test_suite import DEFAULT_TEST_SUITE
from src.lang.grammar import DefaultGrammar
from src.lang.ast_nodes import LambdaNode
from src.lang.parser import parse
from src.lang.compiler import JITCompiler
from src.rl.policy import (
    PolicyNetwork, build_action_vocab, build_type_vocab, build_func_vocab,
)
from src.rl.trajectory import extract_trajectory
from src.rl.reward import compute_reward
from src.rl.priority_queue import PriorityQueueBuffer
from src.rl.trainer import warm_start, train_rl
from src.utils import compute_valid_instantiations


def main():
    parser = argparse.ArgumentParser(description="Run RL on enumerated programs")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Only load programs up to this size (default: all)")
    parser.add_argument("--dataset-dir", type=str,
                        default=str(project_root / "datasets" / "enumerated_list_int"),
                        help="Path to enumerated dataset directory")
    parser.add_argument("--rl-iterations", type=int, default=1000)
    parser.add_argument("--episodes-per-iter", type=int, default=32)
    parser.add_argument("--rl-max-depth", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--warm-start-epochs", type=int, default=50)
    args = parser.parse_args()

    seed_constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    grammar = DefaultGrammar
    test_suite = DEFAULT_TEST_SUITE
    dataset_dir = Path(args.dataset_dir)

    # Compute valid instantiations once
    valid_instantiations = compute_valid_instantiations(grammar)

    # === Phase 1: Load programs from JSON batches ===
    print("=" * 60)
    print("Phase 1: Loading enumerated programs from disk")
    print("=" * 60)

    t0 = time.time()

    with open(dataset_dir / "manifest.json") as f:
        manifest = json.load(f)
    print(f"Manifest: {manifest['total_programs']} programs, {manifest['n_batches']} batches")
    if args.max_size is not None:
        print(f"Filtering to max size: {args.max_size}")

    # Load all batches and parse S-expressions back to ASTs
    batch_files = sorted(glob.glob(str(dataset_dir / "batch_*.json")))
    jit = JITCompiler(grammar)

    raw_programs = []
    skipped_by_size = 0
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        with open(batch_file) as f:
            batch = json.load(f)
        for entry in batch:
            if args.max_size is not None and entry["size"] > args.max_size:
                skipped_by_size += 1
                continue
            raw_programs.append((entry["program"], entry["size"]))

    print(f"Loaded {len(raw_programs)} program strings in {time.time() - t0:.1f}s")
    if skipped_by_size:
        print(f"  ({skipped_by_size} skipped, size > {args.max_size})")

    # Parse and fingerprint
    t1 = time.time()
    corpus: list[TypedProgram] = []
    parse_failures = 0

    for prog_str, size in tqdm(raw_programs, desc="Parsing & fingerprinting"):
        try:
            ast = parse(prog_str)
        except Exception:
            parse_failures += 1
            continue

        closed = LambdaNode(["x"], ast)
        fp = compute_fingerprint(closed, test_suite, jit)
        if fp is None:
            continue

        corpus.append(TypedProgram(ast=ast, type=list[int], fingerprint=fp, size=size))

    print(f"Parsed & fingerprinted {len(corpus)} programs in {time.time() - t1:.1f}s")
    if parse_failures:
        print(f"  ({parse_failures} parse failures)")

    # Collect all fingerprints from enumeration
    enum_fingerprints = {p.fingerprint for p in corpus}
    print(f"Unique enumeration fingerprints: {len(enum_fingerprints)}")

    # === Phase 2: Build policy and warm-start ===
    print("\n" + "=" * 60)
    print("Phase 2: Warm-Start")
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
        seed_constants=seed_constants, epochs=args.warm_start_epochs, batch_size=128,
        valid_instantiations=valid_instantiations,
    )

    # === Phase 3: Seed buffer ===
    print("\n" + "=" * 60)
    print("Phase 3: Seed Buffer")
    print("=" * 60)

    buffer = PriorityQueueBuffer(capacity=args.buffer_capacity)
    corpus_fingerprints: set[Fingerprint] = set()

    seeded = 0
    for prog in tqdm(corpus, desc="Seeding buffer"):
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
        seeded += 1

    print(f"Buffer seeded with {seeded} programs (of {len(corpus)} corpus)")
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
        valid_instantiations=valid_instantiations,
    )
    rl_time = time.time() - t_rl

    # === Report ===
    novel = len(corpus_fingerprints) - len(enum_fingerprints)
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Enumeration corpus (list[int]->list[int]): {len(corpus)}")
    print(f"Enumeration unique fingerprints:           {len(enum_fingerprints)}")
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
