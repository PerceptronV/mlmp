"""Run RL on top of enumerated list[int] -> list[int] corpus and report novel discoveries."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import time
from typing import Callable

from src.enumeration.enumerator import BottomUpEnumerator
from src.enumeration.fingerprint import Fingerprint, compute_fingerprint
from src.enumeration.test_suite import DEFAULT_TEST_SUITE
from src.lang.grammar import DefaultGrammar
from src.lang.ast_nodes import LambdaNode
from src.lang.compiler import JITCompiler
from src.rl.policy import (
    PolicyNetwork, build_action_vocab, build_type_vocab, build_func_vocab,
)
from src.rl.trajectory import extract_trajectory
from src.rl.reward import compute_reward
from src.rl.priority_queue import PriorityQueueBuffer
from src.rl.trainer import warm_start, train_rl


seed_constants = [0, 1, 2, 3]
grammar = DefaultGrammar
test_suite = DEFAULT_TEST_SUITE

# === Phase 1: Re-enumerate to get fingerprinted corpus ===
print("=" * 60)
print("Phase 1: Enumeration (seeding RL)")
print("=" * 60)

t0 = time.time()
enumerator = BottomUpEnumerator(max_size=8)
bank = enumerator.enumerate()
corpus_all = enumerator.extract_corpus()
# Filter to list[int] -> list[int] only
corpus = [p for p in corpus_all if p.type == list[int]]
print(f"Enumeration: {time.time() - t0:.1f}s")
print(f"list[int] -> list[int] corpus: {len(corpus)} programs")

# Collect all fingerprints from enumeration
enum_fingerprints = {p.fingerprint for p in corpus}
print(f"Unique enumeration fingerprints: {len(enum_fingerprints)}")

# === Phase 2: Build policy and warm-start ===
print("\n" + "=" * 60)
print("Phase 2: Warm-Start")
print("=" * 60)

action_vocab = build_action_vocab(grammar, seed_constants)
type_vocab = build_type_vocab(grammar)
func_vocab = build_func_vocab(grammar)

policy = PolicyNetwork(
    action_vocab_size=len(action_vocab),
    type_vocab_size=len(type_vocab),
    func_vocab_size=len(func_vocab),
)

warm_start(
    policy, corpus, grammar, action_vocab, type_vocab, func_vocab,
    seed_constants=seed_constants, epochs=50, batch_size=128,
)

# === Phase 3: Seed buffer ===
print("\n" + "=" * 60)
print("Phase 3: Seed Buffer")
print("=" * 60)

buffer = PriorityQueueBuffer(capacity=50000)
corpus_fingerprints = set()
target_type = Callable[[list[int]], list[int]]

seeded = 0
for prog in corpus:
    wrapped = LambdaNode(["x"], prog.ast)
    try:
        traj = extract_trajectory(wrapped, target_type, grammar)
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
print("Phase 4: RL Exploration (1000 iterations)")
print("=" * 60)

t1 = time.time()
train_rl(
    policy=policy,
    buffer=buffer,
    grammar=grammar,
    test_suite=test_suite,
    action_vocab=action_vocab,
    type_vocab=type_vocab,
    func_vocab=func_vocab,
    corpus_fingerprints=corpus_fingerprints,
    n_iterations=1000,
    episodes_per_iter=32,
    train_steps_per_iter=8,
    batch_size=128,
    lr=1e-4,
    max_depth=8,
    seed_constants=seed_constants,
)
rl_time = time.time() - t1

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
jit = JITCompiler(grammar)
print(f"\nSample novel RL-discovered programs:")
count = 0
for reward, _id, prog, traj, fp in sorted(buffer.buffer, reverse=True):
    if fp not in enum_fingerprints and count < 20:
        print(f"  reward={reward:.3f}  {prog}")
        count += 1
