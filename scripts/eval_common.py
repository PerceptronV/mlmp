"""Shared plumbing for the offline evaluation scripts.

Every eval script does the same three things before it can measure anything:
resolve a checkpoint, rebuild the val dataset the run was trained against, and
reload the model. All of it comes from the checkpoint's stored args, so the
scripts take a bare list of checkpoints. Used by ``eval_pass_at_k.py`` and
``eval_by_length.py``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import torch

from src.train import build_model, build_val_dataset

# Everything build_val_dataset reads. Checkpoints agreeing on all of it share
# one val dataset — loading (and IO-sampling) the same corpus per model is
# pure cost.
VAL_ARG_KEYS = ('dataset', 'inverse_mlc_episode_type', 'inverse_mlc_data_root',
                'val_corpus', 'val_split', 'train_corpus', 'data_seed',
                'n_io_per_program', 'min_n_io_shown', 'mode', 'min_io_pairs',
                'filter_empty_io',  # legacy: pre---min-io-pairs checkpoints
                'grammar', 'split_seed', 'max_program_length')


@dataclass
class LoadedRun:
    """One checkpoint, ready to evaluate."""

    name: str
    path: Path
    ckpt: dict
    args: SimpleNamespace
    dataset: object
    model: torch.nn.Module
    n_programs: int

    @property
    def stored_accuracy(self):
        """The greedy val accuracy recorded on the checkpoint, if any — the
        reference point any offline number should be read against."""
        value = self.ckpt.get('val_accuracy')
        return value if isinstance(value, (int, float)) else None

    def header(self) -> str:
        return (f"=== {self.name} ({self.path.name}, epoch {self.ckpt.get('epoch')}, "
                f"mode={self.args.mode}) ===")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags every eval script needs: which checkpoints, which val set,
    and how to decode."""
    parser.add_argument('checkpoints', nargs='+',
                        help='Checkpoint .pt files and/or run directories')
    parser.add_argument('--ckpt-select', type=str, default='best_acc',
                        help="Which checkpoint to take from a run directory: "
                             "best_acc (default) | best_loss | latest | epoch_<N>")
    parser.add_argument('--val-corpus', type=str, default=None,
                        help='Override the val corpus stored in the checkpoint args')
    parser.add_argument('--val-examples', type=int, default=None,
                        help='Cap on validation programs (default: all)')
    parser.add_argument('--max-program-tokens', type=int, default=80)
    parser.add_argument('--decode-batch-size', type=int, default=64)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--constrain', action='store_true',
                        help='Mask each decode step to tokens that can still '
                             'complete a well-formed program, which removes the '
                             'malformed outcome by construction')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds the sampler, so a rerun draws the same programs')
    parser.add_argument('--json', type=str, default=None,
                        help='Write the full per-model metrics to this JSON file')
    return parser


def resolve_ckpt(path: Path, select: str) -> Path:
    """A checkpoint file, or a run directory + which checkpoint to take from it
    ('best_acc' | 'best_loss' | 'latest' | 'epoch_<N>')."""
    return path / f"checkpoint_{select}.pt" if path.is_dir() else path


def load_model(ckpt: dict, saved_args, n_tokens: int, device) -> torch.nn.Module:
    model = build_model(saved_args, n_tokens)
    state = ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state):
        # --compile-layers wraps each layer, prefixing its keys.
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def val_dataset_for(saved_args, cache: dict):
    key = tuple(str(getattr(saved_args, k, None)) for k in VAL_ARG_KEYS)
    if key not in cache:
        dataset = build_val_dataset(saved_args)
        if dataset is None:
            raise SystemExit(
                "Checkpoint was trained without a validation set "
                "(no --val-corpus / --val-split); pass --val-corpus")
        cache[key] = dataset
    return cache[key]


def load_runs(args) -> Iterator[LoadedRun]:
    """Yield each checkpoint ready to evaluate, reusing one val dataset across
    checkpoints that agree on everything it depends on."""
    device = torch.device(args.device)
    cache: dict = {}
    seen: set[str] = set()

    for raw in args.checkpoints:
        path = resolve_ckpt(Path(raw).expanduser(), args.ckpt_select)
        if not path.exists():
            raise SystemExit(f"No such checkpoint: {path}")
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        saved_args = SimpleNamespace(**ckpt['args'])
        if args.val_corpus is not None:
            saved_args.val_corpus = args.val_corpus
        name = getattr(saved_args, 'run_name', None) or path.parent.name
        if name in seen:  # several checkpoints of the same run
            name = f"{name}/{path.stem}"
        seen.add(name)

        dataset = val_dataset_for(saved_args, cache)
        model = load_model(ckpt, saved_args, len(dataset.tokeniser.vocab), device)
        n_programs = len(dataset.programs)
        if args.val_examples is not None:
            n_programs = min(n_programs, args.val_examples)

        yield LoadedRun(name=name, path=path, ckpt=ckpt, args=saved_args,
                        dataset=dataset, model=model, n_programs=n_programs)


def jsonable(metrics: dict) -> dict:
    """Counters and int keys -> JSON-safe equivalents."""
    bins = metrics.get('outcomes_by_n_io')
    out = {k: v for k, v in metrics.items()
           if k not in ('failure_modes', 'outcomes_by_n_io')}
    if 'failure_modes' in metrics:
        out['failure_modes'] = dict(metrics['failure_modes'])
    if 'outcomes_by_n_io' in metrics:
        out['outcomes_by_n_io'] = ({str(n): dict(c) for n, c in sorted(bins.items())}
                                   if bins else None)
    return out
