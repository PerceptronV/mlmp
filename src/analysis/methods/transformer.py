"""TransformerMethod — runs a trained ``Seq2SeqTransformer`` against Rule100.

All transformer-side tokenisation, decode, reverse-map, and execute logic
lives on :class:`src.data.program_io.ProgramIO`; this method is a thin
adapter. The only things added on top are:

- a deterministic per-trial RNG for the symbol-shuffling preamble (so the
  same trial gets the same permutation every run),
- a checkpoint-aware K for ``easy-symbol-shuffling`` (so eval matches the K
  the checkpoint was last trained at),
- encoder-side embedding extraction (``mean | last``).

Embeddings are extracted with ``n_io_shown=11, order=1`` (per design §16),
which makes them per-task. We pool the final-layer post-norm encoder output.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import torch

from ...data.program_io import ProgramIO
from ..capability import Capability
from ..task import Trial
from .base import Method, Prediction

logger = logging.getLogger(__name__)


@dataclass
class TransformerMethod(Method):
    capabilities: ClassVar[Capability] = Capability.PREDICTIONS | Capability.EMBEDDINGS

    name: str = ""
    run_name: str = ""
    mode: Literal["in-weight", "symbol-shuffling", "easy-symbol-shuffling"] = "in-weight"
    ckpt_select: str = "best_acc"     # "latest" | "best_loss" | "best_acc" | "epoch_<N>"
    device: str = "cuda"
    max_program_tokens: int = 80
    embedding_pool: Literal["mean", "last"] = "mean"
    checkpoint_dir: str | Path = "checkpoints"
    exec_timeout: float = 1.0
    embed_n_io_shown: int = 11
    embed_order: int = 1
    predict_batch_size: int = 128

    _model: object = field(init=False, repr=False, default=None)
    _io: ProgramIO | None = field(init=False, repr=False, default=None)
    _ckpt_path: Path = field(init=False, repr=False, default_factory=lambda: Path())
    _ckpt_args: dict = field(init=False, repr=False, default_factory=dict)
    _easy_k: int | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # Lazy: only load when first used.
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_model()
        self._loaded = True

    def _resolve_ckpt(self) -> Path:
        ckpt_dir = Path(self.checkpoint_dir).expanduser() / self.run_name
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"No checkpoint dir {ckpt_dir}")
        select = self.ckpt_select
        if select == "latest":
            return ckpt_dir / "checkpoint_latest.pt"
        if select == "best_loss":
            p = ckpt_dir / "checkpoint_best_loss.pt"
            if not p.exists():
                p = ckpt_dir / "checkpoint_best.pt"  # legacy
            return p
        if select == "best_acc":
            return ckpt_dir / "checkpoint_best_acc.pt"
        if select.startswith("epoch_"):
            n = int(select.split("_", 1)[1])
            return ckpt_dir / f"checkpoint_epoch_{n}.pt"
        raise ValueError(f"Unknown ckpt_select={select!r}")

    def _load_model(self) -> None:
        # Lazy imports keep ``import src.analysis`` torch-free for CSV-only runs.
        from ...models.seq2seq import Seq2SeqTransformer
        from ...train import _easy_shuffle_k_for_epoch

        ckpt_path = self._resolve_ckpt()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
        self._ckpt_path = ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        args = ckpt.get("args", {})
        self._ckpt_args = dict(args)

        ckpt_mode = args.get("mode", "in-weight")
        if ckpt_mode != self.mode:
            raise ValueError(
                f"{self.name}: checkpoint mode is {ckpt_mode!r} but config says {self.mode!r}"
            )

        # Single source of truth for tokenisation / decode / execute. We don't
        # need a corpus here — ``ProgramIO`` is corpus-free; only the grammar
        # and tokeniser matter for inference.
        self._io = ProgramIO(exec_timeout=self.exec_timeout)
        n_tokens = len(self._io.tokeniser.vocab)  # type: ignore[union-attr]

        model = Seq2SeqTransformer(
            n_tokens=n_tokens,
            d_model=int(args.get("d_model", 256)),
            n_heads=int(args.get("n_heads", 8)),
            n_layers=int(args.get("n_layers", 4)),
            d_ff=args.get("d_ff"),
            max_seq_len=int(args.get("max_seq_len", 2048)),
            compile_layers=False,  # decode path doesn't benefit from compile here
        )
        # Handle compile-prefixed keys in the checkpoint.
        sd = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._device = torch.device(device)
        model = model.to(self._device).eval()
        self._model = model

        if self.mode == "easy-symbol-shuffling":
            class _NS:
                pass

            ns = _NS()
            for k, v in args.items():
                setattr(ns, k.replace("-", "_"), v)
            n_fns = len(self._io.fn_names)
            epoch = int(ckpt.get("epoch", 0))
            self._easy_k = int(_easy_shuffle_k_for_epoch(epoch, ns, n_fns))
            logger.info("[%s] easy-symbol-shuffling K=%d (epoch %d)", self.name, self._easy_k, epoch)

    # ---- Cache fingerprint (used by Cache to invalidate stale entries) ----
    def cache_fingerprint(self) -> str:
        try:
            ckpt = self._resolve_ckpt()
            mtime = ckpt.stat().st_mtime if ckpt.exists() else 0.0
        except Exception:
            mtime = 0.0
        return f"transformer::{self.run_name}::{self.mode}::{self.ckpt_select}::mtime={mtime}::pool={self.embedding_pool}"

    # ---- Per-trial deterministic RNG for symbol shuffling ----
    def _episode_rng(self, trial: Trial) -> random.Random:
        seed = hash((self.name, trial.task_id, trial.order, trial.trial)) & 0xFFFF_FFFF
        return random.Random(seed)

    def _name_map_for(self, trial: Trial) -> dict[str, str] | None:
        if self.mode == "in-weight":
            return None
        io = self._io
        assert io is not None  # _ensure_loaded was called by predict / embed
        rng = self._episode_rng(trial)
        if self.mode == "symbol-shuffling":
            return io.sample_name_map(rng)
        if self.mode == "easy-symbol-shuffling":
            k = self._easy_k if self._easy_k is not None else len(io.fn_names)
            return io.sample_partial_name_map(rng, k)
        raise ValueError(f"Unknown mode {self.mode!r}")

    # ---- predict (single) ----
    def predict(self, trial: Trial) -> Prediction:
        self._ensure_loaded()
        io = self._io
        assert io is not None
        if not trial.observed_examples:
            # 0 IOs → empty src crashes RoPE in dense path. Treat as vacuously
            # incorrect since the model hasn't seen anything.
            return Prediction(response=None, program=None, correct=False)

        observed = [(list(inp), list(out)) for inp, out in trial.observed_examples]
        name_map = self._name_map_for(trial)
        src_tokens = torch.tensor(io.tokenise_input(observed, name_map), dtype=torch.long)
        if src_tokens.numel() == 0:
            return Prediction(response=None, program=None, correct=False)

        gen_tokens = io.greedy_decode(self._model, src_tokens, self.max_program_tokens, self._device)
        program_str = io.detokenise_program(gen_tokens, name_map)
        response = io.execute(program_str, list(trial.query_input), timeout=self.exec_timeout)
        correct = response is not None and response == list(trial.expected_output)
        return Prediction(response=response, program=program_str, correct=correct)

    # ---- predict (batched) ----
    def predict_many(self, trials: list[Trial]) -> list[Prediction]:
        """Batched encode + greedy decode. Per-trial CPU work (parse + JIT
        compile + SIGALRM execute) stays sequential — only the GPU pass is
        amortised across the batch. Trials with empty observed_examples or
        empty src tokens collapse to a vacuous Prediction without touching
        the GPU.
        """
        self._ensure_loaded()
        io = self._io
        assert io is not None
        if not trials:
            return []

        # Tokenise per trial. ``None`` entries skip the GPU path and produce
        # a vacuous Prediction below.
        srcs: list = []
        name_maps: list = []
        for trial in trials:
            if not trial.observed_examples:
                srcs.append(None)
                name_maps.append(None)
                continue
            observed = [(list(inp), list(out)) for inp, out in trial.observed_examples]
            nm = self._name_map_for(trial)
            tok_list = io.tokenise_input(observed, nm)
            if not tok_list:
                srcs.append(None)
                name_maps.append(None)
                continue
            srcs.append(torch.tensor(tok_list, dtype=torch.long))
            name_maps.append(nm)

        gen_lists = io.greedy_decode_batch(
            self._model, srcs, self.max_program_tokens, self._device
        )

        out: list[Prediction] = []
        for trial, src, nm, gen in zip(trials, srcs, name_maps, gen_lists):
            if src is None:
                out.append(Prediction(response=None, program=None, correct=False))
                continue
            program_str = io.detokenise_program(gen, nm)
            response = io.execute(program_str, list(trial.query_input), timeout=self.exec_timeout)
            correct = response is not None and response == list(trial.expected_output)
            out.append(Prediction(response=response, program=program_str, correct=correct))
        return out

    # ---- embed ----
    def embed(self, trial: Trial) -> np.ndarray:
        """Per-task embedding pooled from the encoder. ``trial`` is the
        sentinel last trial of order=1; we override its observed_examples to
        the full ``embed_n_io_shown`` pairs of order=1.
        """
        self._ensure_loaded()
        io = self._io
        assert io is not None

        all_pairs = [(list(inp), list(out)) for inp, out in trial.observed_examples]
        all_pairs.append((list(trial.query_input), list(trial.expected_output)))
        all_pairs = all_pairs[: self.embed_n_io_shown]

        name_map = self._name_map_for(trial)
        src_tokens = torch.tensor(io.tokenise_input(all_pairs, name_map), dtype=torch.long)
        vec = io.encode_pool(self._model, src_tokens, self._device, pool=self.embedding_pool)
        return vec.numpy().astype(np.float32)
