"""Compare BC-only sampling against the full RL loop, under equal episode budgets.

Both paths share a single warm-started policy (deep-copied before each path runs).
BC-only: freeze the warm-started policy, sample N episodes, count novel
fingerprints (∉ enumeration corpus).
RL: run `train_rl` for N total episodes starting from the same warm-start.

Usage:
    python scripts/compare_bc_vs_rl.py --max-size 5 --rl-iterations 200

The total sampling budget for both paths is `rl-iterations * episodes-per-iter`.
"""

import argparse
import copy
import glob
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm

from src.lang.ast_nodes import LambdaNode
from src.lang.compiler import JITCompiler
from src.lang.enumeration.enumerator import TypedProgram
from src.lang.enumeration.fingerprint import Fingerprint, compute_fingerprint
from src.lang.enumeration.test_suite import DEFAULT_TEST_SUITE
from src.lang.enumeration.filters import variability
from src.lang.grammar import DefaultGrammar
from src.lang.parser import parse
from src.lang.rl.mdp import Episode
from src.lang.rl.policy import (
    PolicyNetwork, build_action_vocab, build_type_vocab, build_func_vocab,
)
from src.lang.rl.priority_queue import PriorityQueueBuffer
from src.lang.rl.reward import compute_reward
from src.lang.rl.trainer import warm_start, train_rl, _fill_in_sketch
from src.lang.rl.trajectory import extract_trajectory
from src.lang.utils import compute_valid_instantiations, program_size


class CountTablePolicy:
    """Non-parametric empirical p(a|s) as a count table.

    State key: a canonical, hashable projection of SynthesisState that matches
    what the neural policy uses — target type, per-type variable counts from Γ,
    parent function, arg index, depth budget (clamped to [0,15]), nesting level
    (clamped to [0,3]). Unseen keys fall back to uniform-over-valid.
    """

    def __init__(self):
        self.counts: dict = defaultdict(lambda: defaultdict(int))
        self.marginal: dict = defaultdict(int)  # action -> total count (fallback)

    @staticmethod
    def state_key(state):
        type_counts = tuple(sorted(
            (str(t), sum(1 for v in state.context.values() if v == t))
            for t in set(state.context.values())
        ))
        return (
            str(state.target_type),
            type_counts,
            state.parent_func,
            state.arg_index,
            min(state.depth_budget, 15),
            min(state.nesting_depth, 3),
        )

    def fit(self, transitions):
        for state, action in transitions:
            self.counts[self.state_key(state)][action] += 1
            self.marginal[action] += 1
        total = sum(sum(row.values()) for row in self.counts.values())
        print(f"  CountTable: {len(self.counts)} state keys, "
              f"{len(self.marginal)} marginal actions, {total} transitions")

    def select_action(self, state, actions):
        if not actions:
            return None
        # Tier 1: state-conditional counts
        row = self.counts.get(self.state_key(state))
        if row is not None:
            weights = [row.get(a, 0) for a in actions]
            if sum(weights) > 0:
                return random.choices(actions, weights=weights, k=1)[0]
        # Tier 2: global action marginal (still empirical)
        weights = [self.marginal.get(a, 0) for a in actions]
        if sum(weights) > 0:
            return random.choices(actions, weights=weights, k=1)[0]
        # Tier 3: uniform over valid
        return random.choice(actions)


def load_corpus(dataset_dir: Path, max_size: int | None, grammar, test_suite, jit):
    with open(dataset_dir / "manifest.json") as f:
        manifest = json.load(f)
    print(f"Manifest: {manifest['total_programs']} programs, {manifest['n_batches']} batches")

    raw = []
    for bf in tqdm(sorted(glob.glob(str(dataset_dir / "batch_*.json"))), desc="Loading batches"):
        with open(bf) as f:
            batch = json.load(f)
        for entry in batch:
            if max_size is not None and entry["size"] > max_size:
                continue
            raw.append((entry["program"], entry["size"]))

    corpus: list[TypedProgram] = []
    for prog_str, size in tqdm(raw, desc="Parsing & fingerprinting"):
        try:
            ast = parse(prog_str)
        except Exception:
            continue
        fp = compute_fingerprint(LambdaNode(["x"], ast), test_suite, jit)
        if fp is None:
            continue
        corpus.append(TypedProgram(ast=ast, type=list[int], fingerprint=fp, size=size))

    return corpus


def _run_episode_loop(
    policy, grammar, test_suite, jit, n_episodes, max_depth,
    seed_constants, valid_instantiations, enum_fingerprints, desc,
):
    """Shared sampling loop. Returns dict[fp] = (reward, size)."""
    known = set(enum_fingerprints)
    discovered: dict[Fingerprint, tuple[float, int]] = {}

    for _ in tqdm(range(n_episodes), desc=desc):
        episode = Episode(
            policy, grammar, test_suite, seed_constants, max_depth,
            valid_instantiations=valid_instantiations,
        )
        sketch_ast, _ = episode.run()
        if sketch_ast is None:
            continue

        sketch_lambda = (
            sketch_ast if isinstance(sketch_ast, LambdaNode)
            else LambdaNode(["x"], sketch_ast)
        )
        _fn, _ast, fp = _fill_in_sketch(
            sketch_lambda, seed_constants, test_suite, jit, known, n_samples=10,
        )
        if fp is None:
            fp = compute_fingerprint(sketch_lambda, test_suite, jit)
        if fp is None:
            continue

        r = compute_reward(fp, known)
        if r > 0 and fp not in discovered:
            discovered[fp] = (r, program_size(sketch_ast))
            known.add(fp)

    return discovered


def sample_bc_only(
    policy, grammar, test_suite, jit, n_episodes, max_depth,
    seed_constants, valid_instantiations, enum_fingerprints,
    action_vocab, type_vocab, func_vocab,
):
    """Sample n_episodes from the (frozen) warm-started neural policy."""
    import torch
    policy.setup_for_inference(
        action_vocab=action_vocab, type_vocab=type_vocab, func_vocab=func_vocab,
        grammar=grammar, seed_constants=seed_constants,
        valid_instantiations=valid_instantiations,
    )
    policy.eval()
    with torch.no_grad():
        return _run_episode_loop(
            policy, grammar, test_suite, jit, n_episodes, max_depth,
            seed_constants, valid_instantiations, enum_fingerprints,
            desc="BC-only sampling",
        )


def sample_count_table(
    policy, grammar, test_suite, jit, n_episodes, max_depth,
    seed_constants, valid_instantiations, enum_fingerprints,
):
    """Sample n_episodes from the non-parametric count-table policy."""
    return _run_episode_loop(
        policy, grammar, test_suite, jit, n_episodes, max_depth,
        seed_constants, valid_instantiations, enum_fingerprints,
        desc="CountTable sampling",
    )


def summarise(
    name: str, novel_fps: set, n_episodes: int,
    valid: int | None = None,
    sizes: dict | None = None,  # fp -> size (for novel fps)
    corpus_max_size: int | None = None,
):
    vars_ = [variability(fp) for fp in novel_fps]
    mean_var = statistics.mean(vars_) if vars_ else 0.0
    print(f"\n[{name}]")
    print(f"  episodes sampled:              {n_episodes}")
    if valid is not None:
        print(f"  valid programs (reward > 0):   {valid}")
    print(f"  novel (∉ enumeration corpus):  {len(novel_fps)}")
    print(f"  novelty rate (novel/episodes): {len(novel_fps) / max(n_episodes, 1):.3f}")
    print(f"  mean variability (novel):      {mean_var:.3f}")

    if sizes and corpus_max_size is not None:
        novel_sizes = [sizes[fp] for fp in novel_fps if fp in sizes]
        in_dist = sum(1 for s in novel_sizes if s <= corpus_max_size)
        extrap = sum(1 for s in novel_sizes if s > corpus_max_size)
        max_s = max(novel_sizes) if novel_sizes else 0
        mean_s = statistics.mean(novel_sizes) if novel_sizes else 0.0
        print(f"  novel in-dist  (size ≤ {corpus_max_size}):    {in_dist}")
        print(f"  novel extrapol (size > {corpus_max_size}):    {extrap}")
        print(f"  extrapolation rate:            {extrap / max(len(novel_sizes), 1):.3f}")
        print(f"  novel size mean / max:         {mean_s:.2f} / {max_s}")


def main():
    parser = argparse.ArgumentParser(description="Compare BC-only vs RL sampling")
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--dataset-dir", type=str,
                        default=str(project_root / "datasets" / "enumerated_list_int"))
    parser.add_argument("--rl-iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iter", type=int, default=32)
    parser.add_argument("--rl-max-depth", type=int, default=16)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    parser.add_argument("--warm-start-epochs", type=int, default=50)
    parser.add_argument("--skip-bc", action="store_true", help="Skip BC-only path")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL path")
    parser.add_argument("--skip-count", action="store_true",
                        help="Skip non-parametric count-table path")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON path to dump summary metrics")
    args = parser.parse_args()

    seed_constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    grammar = DefaultGrammar
    test_suite = DEFAULT_TEST_SUITE
    jit = JITCompiler(grammar)
    valid_instantiations = compute_valid_instantiations(grammar)
    total_episodes = args.rl_iterations * args.episodes_per_iter

    # === Load corpus ===
    print("=" * 60)
    print("Loading enumeration corpus")
    print("=" * 60)
    corpus = load_corpus(Path(args.dataset_dir), args.max_size, grammar, test_suite, jit)
    enum_fingerprints = {p.fingerprint for p in corpus}
    corpus_max_size = max(p.size for p in corpus) if corpus else 0
    print(f"Corpus: {len(corpus)} programs, {len(enum_fingerprints)} unique fingerprints, max size {corpus_max_size}")

    # === Warm-start once ===
    print("\n" + "=" * 60)
    print("Warm-start (shared)")
    print("=" * 60)
    action_vocab = build_action_vocab(grammar, seed_constants, valid_instantiations)
    type_vocab = build_type_vocab(grammar, valid_instantiations)
    func_vocab = build_func_vocab(grammar)

    warm_policy = PolicyNetwork(
        action_vocab_size=len(action_vocab),
        type_vocab_size=len(type_vocab),
        func_vocab_size=len(func_vocab),
    )
    warm_start(
        warm_policy, corpus, grammar, action_vocab, type_vocab, func_vocab,
        seed_constants=seed_constants, epochs=args.warm_start_epochs, batch_size=128,
        valid_instantiations=valid_instantiations,
    )

    bc_novel: set = set()
    rl_novel: set = set()
    ct_novel: set = set()
    bc_time = rl_time = ct_time = 0.0

    # === Count-table (non-parametric empirical p(a|s)) ===
    if not args.skip_count:
        print("\n" + "=" * 60)
        print(f"Count-table sampling ({total_episodes} episodes, non-parametric)")
        print("=" * 60)
        ct_policy = CountTablePolicy()
        all_transitions = []
        for prog in tqdm(corpus, desc="Extracting trajectories"):
            target_type = Callable[[list[int]], prog.type]
            wrapped = LambdaNode(["x"], prog.ast)
            try:
                traj = extract_trajectory(
                    wrapped, target_type, grammar,
                    valid_instantiations=valid_instantiations,
                )
                all_transitions.extend(traj)
            except Exception:
                continue
        ct_policy.fit(all_transitions)

        t0 = time.time()
        ct_discovered = sample_count_table(
            ct_policy, grammar, test_suite, jit,
            n_episodes=total_episodes, max_depth=args.rl_max_depth,
            seed_constants=seed_constants,
            valid_instantiations=valid_instantiations,
            enum_fingerprints=enum_fingerprints,
        )
        ct_time = time.time() - t0
        ct_novel = {fp for fp in ct_discovered if fp not in enum_fingerprints}
        ct_sizes = {fp: sz for fp, (_r, sz) in ct_discovered.items()}
        summarise(
            "CountTable", ct_novel, n_episodes=total_episodes,
            valid=len(ct_discovered), sizes=ct_sizes, corpus_max_size=corpus_max_size,
        )
        print(f"  time: {ct_time:.1f}s")

    # === BC-only ===
    if not args.skip_bc:
        print("\n" + "=" * 60)
        print(f"BC-only sampling ({total_episodes} episodes, policy frozen)")
        print("=" * 60)
        bc_policy = copy.deepcopy(warm_policy)
        bc_policy.eval()
        t0 = time.time()
        bc_discovered = sample_bc_only(
            bc_policy, grammar, test_suite, jit,
            n_episodes=total_episodes, max_depth=args.rl_max_depth,
            seed_constants=seed_constants,
            valid_instantiations=valid_instantiations,
            enum_fingerprints=enum_fingerprints,
            action_vocab=action_vocab, type_vocab=type_vocab, func_vocab=func_vocab,
        )
        bc_time = time.time() - t0
        bc_novel = {fp for fp in bc_discovered if fp not in enum_fingerprints}
        bc_sizes = {fp: sz for fp, (_r, sz) in bc_discovered.items()}
        summarise(
            "BC-only", bc_novel, n_episodes=total_episodes,
            valid=len(bc_discovered), sizes=bc_sizes, corpus_max_size=corpus_max_size,
        )
        print(f"  time: {bc_time:.1f}s")

    # === RL ===
    if not args.skip_rl:
        print("\n" + "=" * 60)
        print(f"RL training ({args.rl_iterations} iter × {args.episodes_per_iter} eps = {total_episodes})")
        print("=" * 60)
        rl_policy = copy.deepcopy(warm_policy)
        buffer = PriorityQueueBuffer(capacity=args.buffer_capacity)
        rl_fingerprints = set(enum_fingerprints)

        # Seed buffer with corpus (same as run_rl_experiment)
        for prog in tqdm(corpus, desc="Seeding buffer"):
            target_type = Callable[[list[int]], prog.type]
            wrapped = LambdaNode(["x"], prog.ast)
            try:
                traj = extract_trajectory(
                    wrapped, target_type, grammar,
                    valid_instantiations=valid_instantiations,
                )
            except Exception:
                continue
            r = compute_reward(prog.fingerprint, rl_fingerprints)
            buffer.insert(r, prog.ast, traj, prog.fingerprint)

        t0 = time.time()
        train_rl(
            policy=rl_policy, buffer=buffer, grammar=grammar, test_suite=test_suite,
            action_vocab=action_vocab, type_vocab=type_vocab, func_vocab=func_vocab,
            corpus_fingerprints=rl_fingerprints,
            n_iterations=args.rl_iterations,
            episodes_per_iter=args.episodes_per_iter,
            train_steps_per_iter=8, batch_size=128, lr=1e-4,
            max_depth=args.rl_max_depth,
            seed_constants=seed_constants,
            valid_instantiations=valid_instantiations,
        )
        rl_time = time.time() - t0

        # Novel discoveries = fingerprints added to the set during RL sampling.
        # "Valid" and "rediscovered" counts aren't recoverable without modifying
        # train_rl (buffer is pre-seeded with the corpus).
        rl_novel = rl_fingerprints - enum_fingerprints
        # Recover program sizes for novel fps from the buffer.
        rl_sizes = {
            fp: program_size(prog) for (_r, _id, prog, _t, fp) in buffer.buffer
            if fp in rl_novel
        }
        summarise(
            "RL", rl_novel, n_episodes=total_episodes, valid=None,
            sizes=rl_sizes, corpus_max_size=corpus_max_size,
        )
        print(f"  time: {rl_time:.1f}s")

    # === Comparison ===
    runs = {"CountTable": ct_novel, "BC-only": bc_novel, "RL": rl_novel}
    runs = {k: v for k, v in runs.items() if v}
    if len(runs) >= 2:
        print("\n" + "=" * 60)
        print("Comparison")
        print("=" * 60)
        for name, fps in runs.items():
            print(f"  novel by {name}: {len(fps)}")
        names = list(runs)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ov = runs[a] & runs[b]
                print(f"  {a} ∩ {b}: {len(ov)}  "
                      f"({a}-exclusive {len(runs[a] - runs[b])}, "
                      f"{b}-exclusive {len(runs[b] - runs[a])})")
        union = set().union(*runs.values())
        print(f"  union across {len(runs)} runs: {len(union)}")

    if args.output:
        out = {
            "total_episodes": total_episodes,
            "enum_fingerprints": len(enum_fingerprints),
            "corpus_max_size": corpus_max_size,
            "count_table": {"novel": len(ct_novel), "time_s": ct_time},
            "bc": {"novel": len(bc_novel), "time_s": bc_time},
            "rl": {"novel": len(rl_novel), "time_s": rl_time},
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote summary to {args.output}")


if __name__ == "__main__":
    main()
