"""Training loops: behavioural cloning warm-start and priority queue RL."""

import logging
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

from .policy import PolicyNetwork, encode_states, compute_valid_masks, build_action_vocab
from .mdp import SynthesisState, Action, Episode, valid_actions
from .trajectory import extract_trajectory
from .reward import compute_reward
from .priority_queue import PriorityQueueBuffer
from ..grammar import Grammar
from ..ast_nodes import LambdaNode, IntHoleNode
from ..compiler import JITCompiler
from ..enumeration.fingerprint import Fingerprint, FingerprintTable, make_hashable, FAIL, compute_fingerprint
from ..enumeration.enumerator import TypedProgram
from ..utils import RANDINT_PROBE_SEQUENCE


class TransitionDataset(Dataset):
    """Dataset of (state, action) pairs for supervised training."""

    def __init__(
        self,
        transitions: list[tuple[SynthesisState, Action]],
        action_vocab: dict[Action, int],
        type_vocab: dict,
        func_vocab: dict,
        grammar: Grammar,
        seed_constants: list[int],
        valid_instantiations: dict | None = None,
    ):
        self.transitions = transitions
        self.action_vocab = action_vocab
        self.type_vocab = type_vocab
        self.func_vocab = func_vocab
        self.grammar = grammar
        self.seed_constants = seed_constants
        self.valid_instantiations = valid_instantiations

        # Filter out transitions whose action is missing from vocab
        n_before = len(transitions)
        transitions = [(s, a) for s, a in transitions if a in action_vocab]
        n_dropped = n_before - len(transitions)
        if n_dropped:
            logger.warning(
                f"Dropped {n_dropped}/{n_before} transitions with unknown actions"
            )
        self.transitions = transitions

        # Pre-encode all states
        print(f"  Encoding {len(transitions)} transitions...")
        states = [s for s, _ in transitions]
        self.state_data = encode_states(states, type_vocab, func_vocab)
        self.action_indices = torch.tensor(
            [action_vocab[a] for _, a in transitions], dtype=torch.long,
        )
        self.valid_masks = compute_valid_masks(
            states, grammar, seed_constants, action_vocab,
            valid_instantiations=valid_instantiations,
        )

        # Drop transitions where the target action is masked (would produce loss=inf).
        # This catches any remaining type-inference mismatches in trajectory extraction.
        valid_for_action = self.valid_masks.gather(
            1, self.action_indices.unsqueeze(1),
        ).squeeze(1)
        keep = valid_for_action.bool()
        n_masked = int((~keep).sum().item())
        if n_masked:
            logger.warning(
                f"Dropped {n_masked}/{len(transitions)} transitions with masked target actions"
            )
            self.transitions = [t for t, k in zip(transitions, keep.tolist()) if k]
            self.state_data = {k: v[keep] for k, v in self.state_data.items()}
            self.action_indices = self.action_indices[keep]
            self.valid_masks = self.valid_masks[keep]

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state_batch = {k: v[idx] for k, v in self.state_data.items()}
        return state_batch, self.action_indices[idx], self.valid_masks[idx]


def _collate_transitions(batch):
    """Collate function for TransitionDataset."""
    state_dicts, action_indices, valid_masks = zip(*batch)
    collated_state = {}
    for key in state_dicts[0]:
        collated_state[key] = torch.stack([s[key] for s in state_dicts])
    return collated_state, torch.stack(action_indices), torch.stack(valid_masks)


def _count_holes(node) -> int:
    """Count the number of IntHoleNodes in an AST via pre-order traversal."""
    if isinstance(node, IntHoleNode):
        return 1
    elif hasattr(node, '__dataclass_fields__'):
        total = 0
        for field_name in node.__dataclass_fields__:
            child = getattr(node, field_name)
            if hasattr(child, 'ast_type'):  # ASTNode
                total += _count_holes(child)
            elif isinstance(child, list):
                for item in child:
                    if hasattr(item, 'ast_type'):  # ASTNode
                        total += _count_holes(item)
        return total
    return 0


def _fill_in_sketch(
    sketch_ast,
    seed_constants: list[int],
    test_suite: list,
    jit: JITCompiler,
    known_fingerprints: set,
    n_samples: int = 10,
):
    """
    Given a sketch AST (which may contain IntHoleNodes), sample substitutions
    and return the best (fn, concrete_ast, fp) by reward.

    Returns (fn, concrete_ast, fp) or (None, None, None) if all attempts fail.
    """
    import random
    from ..rl.reward import compute_reward

    k = _count_holes(sketch_ast)
    if k == 0:
        # No holes — compile directly and fingerprint via probe sequence
        try:
            fn, concrete_ast = jit.compile(sketch_ast)
            return fn, concrete_ast, None  # caller will fingerprint
        except Exception:
            return None, None, None

    best_reward = -1
    best_fn = None
    best_ast = None
    best_fp = None

    for _ in range(n_samples):
        sigma = [random.choice(seed_constants) for _ in range(k)]
        try:
            fn, concrete_ast = jit.compile(sketch_ast, sigma)
            outputs = []
            for inp in test_suite:
                try:
                    outputs.append(make_hashable(fn(inp)))
                except Exception:
                    outputs.append(FAIL)
            fp = Fingerprint(tuple(outputs))
            reward = compute_reward(fp, known_fingerprints)
            if reward > best_reward:
                best_reward = reward
                best_fn = fn
                best_ast = concrete_ast
                best_fp = fp
        except Exception:
            continue

    return best_fn, best_ast, best_fp


def warm_start(
    policy: PolicyNetwork,
    corpus: list[TypedProgram],
    grammar: Grammar,
    action_vocab: dict,
    type_vocab: dict,
    func_vocab: dict,
    seed_constants: list[int] | None = None,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    valid_instantiations: dict | None = None,
):
    """
    Pre-train the policy via behavioural cloning on the enumeration corpus.

    For each program in the corpus, extract its trajectory and train the
    policy to predict each action given the corresponding state.
    """
    if seed_constants is None:
        seed_constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Warm-start device: {device}")
    policy = policy.to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)

    # Extract all trajectories from corpus programs
    all_transitions = []
    for prog in tqdm(corpus, desc="Extracting trajectories"):
        # Use the program's actual return type to build the correct Callable type
        target_type = Callable[[list[int]], prog.type]
        wrapped = LambdaNode(["x"], prog.ast)
        try:
            traj = extract_trajectory(
                wrapped, target_type, grammar,
                valid_instantiations=valid_instantiations,
            )
            all_transitions.extend(traj)
        except Exception:
            continue  # Skip programs that fail trajectory extraction

    if not all_transitions:
        print("Warm-start: no transitions extracted from corpus")
        return

    print(f"Warm-start: {len(all_transitions)} transitions from {len(corpus)} programs")

    dataset = TransitionDataset(
        all_transitions, action_vocab, type_vocab, func_vocab,
        grammar, seed_constants,
        valid_instantiations=valid_instantiations,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_transitions,
    )

    epoch_pbar = tqdm(range(epochs), desc="Warm-start")
    for epoch in epoch_pbar:
        total_loss = 0.0
        n_batches = 0
        for state_batch, action_indices, valid_masks in loader:
            state_batch = {k: v.to(device) for k, v in state_batch.items()}
            action_indices = action_indices.to(device)
            valid_masks = valid_masks.to(device)

            log_probs = policy(state_batch, valid_masks)
            loss = F.nll_loss(log_probs, action_indices)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")

    policy.to('cpu')


def train_rl(
    policy: PolicyNetwork,
    buffer: PriorityQueueBuffer,
    grammar: Grammar,
    test_suite: list[list[int]],
    action_vocab: dict,
    type_vocab: dict,
    func_vocab: dict,
    corpus_fingerprints: set[Fingerprint],
    n_iterations: int = 10000,
    episodes_per_iter: int = 32,
    train_steps_per_iter: int = 8,
    batch_size: int = 64,
    lr: float = 1e-4,
    max_depth: int = 8,
    seed_constants: list[int] | None = None,
    valid_instantiations: dict | None = None,
):
    """Main RL training loop with priority queue training."""
    if seed_constants is None:
        seed_constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    jit = JITCompiler(grammar)

    # Ensure policy is set up for inference
    policy.setup_for_inference(
        action_vocab, type_vocab, func_vocab, grammar, seed_constants,
        valid_instantiations=valid_instantiations,
    )

    stats = {'novel_found': 0, 'total_generated': 0, 'buffer_min': 0.0}

    pbar = tqdm(range(n_iterations), desc="RL training")
    for iteration in pbar:
        # === Sampling Phase ===
        for _ in range(episodes_per_iter):
            episode = Episode(
                policy, grammar, test_suite, seed_constants, max_depth,
                valid_instantiations=valid_instantiations,
            )
            sketch_ast, trajectory = episode.run()
            stats['total_generated'] += 1

            if sketch_ast is None:
                continue

            # Wrap sketch in lambda for fingerprinting context
            sketch_lambda = LambdaNode(["x"], sketch_ast) if not isinstance(sketch_ast, LambdaNode) else sketch_ast

            # Fill integer holes and pick best substitution by reward
            _fn, concrete_body, fp = _fill_in_sketch(
                sketch_lambda, seed_constants, test_suite, jit,
                corpus_fingerprints, n_samples=10,
            )

            if fp is None:
                # No holes: fall back to probe-based fingerprint
                fp = compute_fingerprint(sketch_lambda, test_suite, jit)
                concrete_body = sketch_lambda

            if fp is None:
                continue

            reward = compute_reward(fp, corpus_fingerprints)
            if reward > 0:
                closed_ast = concrete_body if concrete_body is not None else sketch_lambda
                buffer.insert(reward, closed_ast, trajectory, fp)
                if fp not in corpus_fingerprints:
                    corpus_fingerprints.add(fp)
                    stats['novel_found'] += 1

        # === Training Phase ===
        if len(buffer) < batch_size:
            continue

        for _ in range(train_steps_per_iter):
            batch = buffer.sample(batch_size)

            all_transitions = []
            for reward, program, trajectory in batch:
                for state, action in trajectory:
                    if action in action_vocab:
                        all_transitions.append((state, action, reward))

            if not all_transitions:
                continue

            states, actions, rewards = zip(*all_transitions)
            state_batch = encode_states(states, type_vocab, func_vocab)
            action_indices = torch.tensor(
                [action_vocab[a] for a in actions], dtype=torch.long,
            )
            reward_weights = torch.tensor(rewards, dtype=torch.float32)
            valid_masks = compute_valid_masks(
                states, grammar, seed_constants, action_vocab,
                valid_instantiations=valid_instantiations,
            )

            log_probs = policy(state_batch, valid_masks)
            per_action_log_prob = log_probs.gather(
                1, action_indices.unsqueeze(1),
            ).squeeze(1)

            # Reward-weighted maximum likelihood
            loss = -(reward_weights * per_action_log_prob).mean()

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()

        # === Progress bar update ===
        stats['buffer_min'] = buffer.min_reward()
        pbar.set_postfix(
            buf=len(buffer),
            novel=stats['novel_found'],
            gen=stats['total_generated'],
            min_r=f"{stats['buffer_min']:.3f}",
        )
