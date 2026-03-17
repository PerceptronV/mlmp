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
from ..lang.grammar import Grammar
from ..lang.ast_nodes import LambdaNode
from ..lang.compiler import JITCompiler
from ..enumeration.fingerprint import Fingerprint, compute_fingerprint
from ..enumeration.enumerator import TypedProgram


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


def warm_start(
    policy: PolicyNetwork,
    corpus: list[TypedProgram],
    grammar: Grammar,
    action_vocab: dict,
    type_vocab: dict,
    func_vocab: dict,
    seed_constants: list[int] | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    valid_instantiations: dict | None = None,
):
    """
    Pre-train the policy via behavioural cloning on the enumeration corpus.

    For each program in the corpus, extract its trajectory and train the
    policy to predict each action given the corresponding state.
    """
    if seed_constants is None:
        seed_constants = [0, 1, 2, 3]

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

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
            log_probs = policy(state_batch, valid_masks)
            loss = F.nll_loss(log_probs, action_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}")


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
        seed_constants = [0, 1, 2, 3]

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
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
            ast, trajectory = episode.run()
            stats['total_generated'] += 1

            if ast is None:
                continue

            # Wrap in lambda and fingerprint
            closed_ast = LambdaNode(["x"], ast) if not isinstance(ast, LambdaNode) else ast
            fp = compute_fingerprint(closed_ast, test_suite, jit)
            if fp is None:
                continue

            reward = compute_reward(fp, corpus_fingerprints)
            if reward > 0:
                inserted = buffer.insert(reward, closed_ast, trajectory, fp)
                if inserted and fp not in corpus_fingerprints:
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

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        # === Progress bar update ===
        stats['buffer_min'] = buffer.min_reward()
        pbar.set_postfix(
            buf=len(buffer),
            novel=stats['novel_found'],
            gen=stats['total_generated'],
            min_r=f"{stats['buffer_min']:.3f}",
        )
