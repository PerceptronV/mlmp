"""Neural policy network for program synthesis."""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mdp import SynthesisState, Action, ActionType, valid_actions
from ..lang.grammar import Grammar
from ..lang.type_utils import CallableOrig, TypeType
from ..utils import resolve_type, freeze_instantiation, TYPE_UNIVERSE


def build_type_vocab(grammar: Grammar, valid_instantiations: dict | None = None) -> dict[TypeType, int]:
    """Collect all resolved types that appear in the grammar + common types."""
    if valid_instantiations is not None:
        types = set(TYPE_UNIVERSE)
        for func_name, instantiations in valid_instantiations.items():
            func_info = grammar[func_name]
            for inst in instantiations:
                for arg_type in func_info['arg_types']:
                    resolved = resolve_type(arg_type, instantiation=inst)
                    if resolved is not None:
                        types.add(resolved)
                resolved_ret = resolve_type(func_info['ret_type'], instantiation=inst)
                if resolved_ret is not None:
                    types.add(resolved_ret)
    else:
        types = {
            int, bool, list[int], list[list[int]], list[bool],
            Callable[[int], int], Callable[[int], bool],
            Callable[[int, int], int],
            Callable[[list[int]], int],
            Callable[[int], list[int]],
            Callable[[bool], bool],
        }
    return {t: i for i, t in enumerate(sorted(types, key=str))}


def build_func_vocab(grammar: Grammar) -> dict[str | None, int]:
    """Map function names to indices. None maps to 0."""
    vocab = {None: 0}
    for i, name in enumerate(grammar.names, start=1):
        vocab[name] = i
    return vocab


def build_action_vocab(
    grammar: Grammar,
    seed_constants: list[int],
    valid_instantiations: dict | None = None,
) -> dict[Action, int]:
    """
    Build a mapping from Action -> int index.

    When valid_instantiations is provided, emits one APPLY entry per
    (function, instantiation) pair instead of one per function.
    """
    vocab = {}
    idx = 0

    for c in seed_constants:
        vocab[Action(ActionType.LITERAL_INT, c)] = idx
        idx += 1
    vocab[Action(ActionType.LITERAL_BOOL, True)] = idx
    idx += 1
    vocab[Action(ActionType.LITERAL_BOOL, False)] = idx
    idx += 1
    vocab[Action(ActionType.LITERAL_EMPTY_LIST, None)] = idx
    idx += 1

    for var_name in ["x", "_p0", "_p1", "_p2", "_p3", "_p4", "_p5"]:
        vocab[Action(ActionType.VARIABLE, var_name)] = idx
        idx += 1

    if valid_instantiations is not None:
        for func_name in grammar.names:
            for inst in valid_instantiations[func_name]:
                frozen = freeze_instantiation(inst)
                vocab[Action(ActionType.APPLY, func_name, frozen)] = idx
                idx += 1
    else:
        for func_name in grammar.names:
            vocab[Action(ActionType.APPLY, func_name)] = idx
            idx += 1

    vocab[Action(ActionType.LAMBDA, None)] = idx
    idx += 1
    vocab[Action(ActionType.IF, None)] = idx
    idx += 1

    return vocab


class StateEncoder(nn.Module):
    """Encode a SynthesisState into a fixed-size vector."""

    def __init__(self, type_vocab_size, func_vocab_size, embed_dim=64):
        super().__init__()
        self.type_embed = nn.Embedding(type_vocab_size, embed_dim)
        self.func_embed = nn.Embedding(func_vocab_size + 1, embed_dim)  # +1 for None
        self.arg_index_embed = nn.Embedding(8, embed_dim)
        self.depth_embed = nn.Embedding(16, embed_dim)
        self.nesting_embed = nn.Embedding(4, embed_dim)
        self.context_proj = nn.Linear(16, embed_dim)
        self.combine = nn.Linear(6 * embed_dim, embed_dim)

    def forward(self, state_batch):
        """
        Args:
            state_batch: dict with keys 'target_type', 'parent_func',
                         'arg_index', 'depth_budget', 'context_features'
        Returns:
            Tensor of shape (batch_size, embed_dim)
        """
        t = self.type_embed(state_batch['target_type'])
        f = self.func_embed(state_batch['parent_func'])
        i = self.arg_index_embed(state_batch['arg_index'])
        d = self.depth_embed(state_batch['depth_budget'])
        n = self.nesting_embed(state_batch['nesting_depth'])
        c = self.context_proj(state_batch['context_features'])

        combined = torch.cat([t, f, i, d, n, c], dim=-1)
        return torch.relu(self.combine(combined))


class PolicyNetwork(nn.Module):
    """
    Full policy network: state -> distribution over actions.

    Uses a shared state encoder and a linear head that scores all
    possible actions. Invalid actions are masked to -inf before softmax.
    """

    def __init__(
        self,
        action_vocab_size,
        type_vocab_size,
        func_vocab_size,
        embed_dim=64,
        hidden_dim=128,
    ):
        super().__init__()
        self.encoder = StateEncoder(type_vocab_size, func_vocab_size, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_vocab_size),
        )

        # Store vocab refs for select_action
        self._action_vocab = None
        self._action_vocab_inv = None
        self._type_vocab = None
        self._func_vocab = None
        self._grammar = None
        self._seed_constants = None
        self._valid_instantiations = None

    def setup_for_inference(
        self,
        action_vocab: dict[Action, int],
        type_vocab: dict,
        func_vocab: dict,
        grammar: Grammar,
        seed_constants: list[int],
        valid_instantiations: dict | None = None,
    ):
        """Store vocabs needed for select_action during episodes."""
        self._action_vocab = action_vocab
        self._action_vocab_inv = {v: k for k, v in action_vocab.items()}
        self._type_vocab = type_vocab
        self._func_vocab = func_vocab
        self._grammar = grammar
        self._seed_constants = seed_constants
        self._valid_instantiations = valid_instantiations

    def forward(self, state_batch, valid_action_mask):
        """
        Args:
            state_batch: dict of tensors
            valid_action_mask: BoolTensor of shape (batch_size, action_vocab_size)
        Returns:
            log_probs: Tensor of shape (batch_size, action_vocab_size)
        """
        h = self.encoder(state_batch)
        logits = self.head(h)
        logits[~valid_action_mask] = float('-inf')
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def select_action(
        self,
        state: SynthesisState,
        valid_action_list: list[Action],
    ) -> Action:
        """Select an action for a single state during episode rollout."""
        with torch.no_grad():
            state_batch = encode_states(
                [state], self._type_vocab, self._func_vocab,
            )
            mask = compute_valid_masks(
                [state], self._grammar, self._seed_constants, self._action_vocab,
                valid_instantiations=self._valid_instantiations,
            )
            log_probs = self.forward(state_batch, mask)
            probs = torch.exp(log_probs[0])

            # Sample from the distribution
            action_idx = torch.multinomial(probs, 1).item()
            return self._action_vocab_inv[action_idx]


def encode_states(
    states: list[SynthesisState] | tuple[SynthesisState, ...],
    type_vocab: dict[TypeType, int],
    func_vocab: dict[str | None, int],
) -> dict[str, torch.Tensor]:
    """Batch-encode a list of SynthesisStates into tensors."""
    default_type_id = 0
    n = len(states)

    target_types = []
    parent_funcs = []
    arg_indices = []
    depth_budgets = []
    nesting_depths = []
    context_features = []

    for s in states:
        target_types.append(type_vocab.get(s.target_type, default_type_id))
        parent_funcs.append(func_vocab.get(s.parent_func, 0))
        arg_indices.append(min(s.arg_index or 0, 7))
        depth_budgets.append(max(0, min(s.depth_budget, 15)))
        nesting_depths.append(max(0, min(getattr(s, 'nesting_depth', 0), 3)))

        # Context features: count of variables per type
        feat = [0.0] * 16
        for i, (vname, vtype) in enumerate(s.context.items()):
            tid = type_vocab.get(vtype, 0)
            if tid < 16:
                feat[tid] += 1.0
        context_features.append(feat)

    return {
        'target_type': torch.tensor(target_types, dtype=torch.long),
        'parent_func': torch.tensor(parent_funcs, dtype=torch.long),
        'arg_index': torch.tensor(arg_indices, dtype=torch.long),
        'depth_budget': torch.tensor(depth_budgets, dtype=torch.long),
        'nesting_depth': torch.tensor(nesting_depths, dtype=torch.long),
        'context_features': torch.tensor(context_features, dtype=torch.float32),
    }


def _mask_cache_key(s: SynthesisState) -> tuple:
    """Cache key: valid actions depend on target_type, context types, and whether depth > 0."""
    return (s.target_type, frozenset(s.context.items()), s.depth_budget > 0, getattr(s, 'nesting_depth', 0))


def compute_valid_masks(
    states: list[SynthesisState] | tuple[SynthesisState, ...],
    grammar: Grammar,
    seed_constants: list[int],
    action_vocab: dict[Action, int],
    valid_instantiations: dict | None = None,
) -> torch.BoolTensor:
    """Compute valid action masks for a batch of states (cached by state signature)."""
    n = len(states)
    vocab_size = len(action_vocab)
    masks = torch.zeros(n, vocab_size, dtype=torch.bool)

    cache: dict[tuple, torch.Tensor] = {}
    for i, s in enumerate(states):
        key = _mask_cache_key(s)
        if key not in cache:
            row = torch.zeros(vocab_size, dtype=torch.bool)
            for a in valid_actions(s, grammar, seed_constants, valid_instantiations):
                if a in action_vocab:
                    row[action_vocab[a]] = True
            cache[key] = row
        masks[i] = cache[key]

    return masks
