"""Single source of truth for transformer-side program I/O.

Everything project-specific about how programs and I/O pairs are presented to
``Seq2SeqTransformer`` lives here. Other modules (``ProgramDataset``,
``train._check_program_match``, ``analysis.methods.transformer.TransformerMethod``)
are thin callers; they don't independently re-implement any of:

- encoder-input formatting (preamble + I/O pairs),
- symbol-shuffling permutation sampling (full and partial),
- decoder-target formatting (``<start> + program + <end>`` with optional
  fn-name remapping),
- greedy decoding,
- detokenise + reverse-map back to canonical fn names,
- parse + JIT-compile + execute with timeout, mod-100 (project_int_mod).

If you find yourself writing tokenisation or decode logic outside this file,
you're forking the source of truth — push the change in here instead.
"""
from __future__ import annotations

import random
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from ..lang.compiler import JITCompiler
from ..lang.grammar import DefaultGrammar, Grammar
from ..lang.parser import parse
from .tokeniser import (
    DEFINED_AS_TOKEN,
    END_TOKEN,
    NEWLINE_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    START_TOKEN,
    TO_TOKEN,
    Tokeniser,
)


@contextmanager
def alarm(seconds: float):
    """SIGALRM-based execution timeout.

    Main-thread / POSIX only — both hold for our training and analysis loops.
    SIGALRM is the only way to interrupt a runaway pure-Python program (a
    user-supplied ``(lambda (fix ...))`` for example) without forking.
    """

    def _handler(_signum, _frame):
        raise TimeoutError(f"call exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


@dataclass
class ProgramIO:
    """Stateless (w.r.t. corpus) handle for transformer input/output formatting.

    Construct once per (tokeniser, grammar) pair. ``ProgramDataset`` builds one
    in ``__init__``; ``TransformerMethod`` builds one from a fresh ``Tokeniser``;
    ``train._check_program_match`` reuses the dataset's.

    The constructor accepts an existing tokeniser/grammar so that the same
    instance can be shared across training and inference, but defaults make it
    a one-liner: ``ProgramIO()``.
    """

    tokeniser: Tokeniser | None = None
    grammar: Grammar = DefaultGrammar
    exec_timeout: float = 1.0

    def __post_init__(self) -> None:
        if self.tokeniser is None:
            self.tokeniser = Tokeniser(grammar=self.grammar)
        # Cache the special-token ids; matches ProgramDataset's old layout.
        v = self.tokeniser.vocab
        self.pad = v.stoi[PAD_TOKEN]
        self.start = v.stoi[START_TOKEN]
        self.end = v.stoi[END_TOKEN]
        self.to = v.stoi[TO_TOKEN]
        self.defined_as = v.stoi[DEFINED_AS_TOKEN]
        self.sep = v.stoi[SEP_TOKEN]
        self.newline = v.stoi[NEWLINE_TOKEN]
        self.fn_names: list[str] = list(self.grammar.functions.keys())
        # Lazy compiler; expensive to construct.
        self._compiler: JITCompiler | None = None

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    def tokenise_io_pairs(self, io_pairs: list[tuple[list[int], list[int]]]) -> list[int]:
        """``[input] → [output] \\n`` for each pair."""
        x: list[int] = []
        for inp, out in io_pairs:
            x.extend(self.tokeniser.tokenise_list(inp) + [self.to])  # type: ignore[union-attr]
            x.extend(self.tokeniser.tokenise_list(out) + [self.newline])  # type: ignore[union-attr]
        return x

    def tokenise_preamble(self, name_map: dict[str, str]) -> list[int]:
        """``<mapped> ≜ <orig> \\n`` lines in alphabetical-by-mapped order,
        terminated by ``<SEP> \\n``. The fixed order means the only signal the
        model gets about the permutation is the ``≜`` lines themselves, not
        positional cues.
        """
        order = sorted(name_map.items(), key=lambda kv: kv[1])
        toks: list[int] = []
        for orig, mapped in order:
            toks.append(self.tokeniser.tokenise_element(mapped))  # type: ignore[union-attr]
            toks.append(self.defined_as)
            toks.append(self.tokeniser.tokenise_element(orig))  # type: ignore[union-attr]
            toks.append(self.newline)
        toks.append(self.sep)
        toks.append(self.newline)
        return toks

    def tokenise_input(
        self,
        io_pairs: list[tuple[list[int], list[int]]],
        name_map: dict[str, str] | None = None,
    ) -> list[int]:
        """Encoder src = ``[preamble?] + [io_pairs]``."""
        x: list[int] = []
        if name_map is not None:
            x.extend(self.tokenise_preamble(name_map))
        x.extend(self.tokenise_io_pairs(io_pairs))
        return x

    def tokenise_program_item(
        self,
        program_str: str,
        io_pairs: list[tuple[list[int], list[int]]],
        name_map: dict[str, str] | None = None,
    ) -> tuple[list[int], list[int]]:
        """``(x, y)`` pair: ``x`` = encoder src, ``y`` = ``<start> + program + <end>``."""
        x = self.tokenise_input(io_pairs, name_map)
        y = (
            [self.start]
            + self.tokeniser.tokenise_program(program_str, name_map)  # type: ignore[union-attr]
            + [self.end]
        )
        return x, y

    # ------------------------------------------------------------------
    # Symbol-shuffling permutations
    # ------------------------------------------------------------------
    def sample_name_map(self, rng: random.Random) -> dict[str, str]:
        """Full permutation over all grammar function names. Returns
        ``{orig_name: mapped_name}``; the mapped name is what appears in the
        rewritten program; the preamble lists each ``mapped ≜ orig``.
        """
        permuted = list(self.fn_names)
        rng.shuffle(permuted)
        return dict(zip(self.fn_names, permuted))

    def sample_partial_name_map(self, rng: random.Random, k: int) -> dict[str, str]:
        """Permutation over a random K-subset of grammar function names.

        Functions outside the subset are absent from the map and pass through
        unchanged in both the preamble and program rewrites. ``k`` is clamped
        to ``[0, len(fn_names)]``.
        """
        k = max(0, min(k, len(self.fn_names)))
        if k == 0:
            return {}
        chosen = rng.sample(self.fn_names, k)
        permuted = list(chosen)
        rng.shuffle(permuted)
        return dict(zip(chosen, permuted))

    # ------------------------------------------------------------------
    # Decoder-side reverse map
    # ------------------------------------------------------------------
    @staticmethod
    def reverse_program_names(
        program_str: str, name_map: dict[str, str] | None
    ) -> str:
        """Undo a per-episode fn-name permutation on a tokenised program string.

        In symbol-shuffling modes the model emits the program with mapped fn
        names; reverse the permutation so the parser/compiler sees the
        canonical names. ``None`` / empty map is a pass-through.
        """
        if not name_map:
            return program_str
        mapped_to_orig = {v: k for k, v in name_map.items()}
        return " ".join(mapped_to_orig.get(tok, tok) for tok in program_str.split(" "))

    def detokenise_program(
        self, gen_tokens: list[int], name_map: dict[str, str] | None = None
    ) -> str:
        """Token ids → program string, with the trailing ``<end>`` (and
        anything after) trimmed and the per-episode fn-name permutation
        reversed.
        """
        if self.end in gen_tokens:
            gen_tokens = gen_tokens[: gen_tokens.index(self.end)]
        program_str = self.tokeniser.detokenise(gen_tokens)  # type: ignore[union-attr]
        return self.reverse_program_names(program_str, name_map)

    # ------------------------------------------------------------------
    # Greedy decode
    # ------------------------------------------------------------------
    def greedy_decode(
        self,
        model,
        src_tokens,  # 1-D LongTensor on CPU; we move it to ``device``
        max_tokens: int,
        device,
    ) -> list[int]:
        """Greedy-decode a single sequence. Returns the predicted token ids,
        excluding ``<start>`` but including ``<end>`` if it was emitted.
        """
        import torch  # local import: program_io stays importable without torch installed

        # Dense path: jagged SDPA in PyTorch 2.11 fails on single-sequence (B=1)
        # NestedTensors, so we explicitly use a dense (B=1, L) tensor here.
        src = src_tokens.to(device).unsqueeze(0)
        memory = model.encode(src)

        out = [self.start]
        for _ in range(max_tokens):
            tgt = torch.tensor(out, dtype=torch.long, device=device).unsqueeze(0)
            logits = model.project(model.decode(tgt, memory))  # (1, len(out), n_tokens)
            next_token = int(logits[0, -1].argmax())
            out.append(next_token)
            if next_token == self.end:
                break
        return out[1:]  # drop <start>

    def greedy_decode_batch(
        self,
        model,
        src_tokens_list,  # list of 1-D LongTensors (any device); empty / ``None`` entries are handled
        max_tokens: int,
        device,
    ) -> list[list[int]]:
        """Batched greedy decode over the *jagged* path. Each row is one
        independent (src, gen) pair; rows can have different src lengths and
        finish at different times. Returns a list of generated token-id lists
        (excluding ``<start>``, including ``<end>`` if emitted), one per
        non-empty input. Empty / ``None`` entries in ``src_tokens_list`` map
        to empty output lists.

        Falls through to ``greedy_decode`` for the single-row case because
        jagged SDPA in PyTorch 2.11 is broken at ``B=1`` (see ``greedy_decode``
        for the dense workaround). Above ``B=1`` the jagged path is the same
        one training uses every step.
        """
        import torch
        from ..models.seq2seq import from_token_ids

        valid_idx: list[int] = []
        valid_srcs: list = []
        for i, s in enumerate(src_tokens_list):
            if s is None or s.numel() == 0:
                continue
            valid_idx.append(i)
            valid_srcs.append(s.to(device))

        out_lists: list[list[int]] = [[] for _ in src_tokens_list]
        if not valid_srcs:
            return out_lists
        if len(valid_srcs) == 1:
            # B=1: jagged SDPA is broken, take the dense path.
            out_lists[valid_idx[0]] = self.greedy_decode(
                model, valid_srcs[0], max_tokens, device
            )
            return out_lists

        src_batch = from_token_ids(valid_srcs)
        memory = model.encode(src_batch)

        n = len(valid_srcs)
        per_row: list[list[int]] = [[self.start] for _ in range(n)]
        done = [False] * n

        for _ in range(max_tokens):
            if all(done):
                break
            # Each row's tgt grows in lockstep for active rows; done rows stay
            # frozen at their previous length. Jagged NT tolerates ragged
            # lengths so we don't need to resize the batch.
            tgt_batch = from_token_ids(
                [torch.tensor(o, dtype=torch.long, device=device) for o in per_row]
            )
            logits_nt = model.project(model.decode(tgt_batch, memory))
            vals = logits_nt.values()                       # (sum_len, n_tokens)
            offsets = logits_nt.offsets()                   # (B+1,)
            last_idx = (offsets[1:] - 1).tolist()
            next_tokens = vals[last_idx].argmax(dim=-1).tolist()

            for i, tok in enumerate(next_tokens):
                if done[i]:
                    continue
                per_row[i].append(int(tok))
                if int(tok) == self.end:
                    done[i] = True

        for slot, gen in zip(valid_idx, per_row):
            out_lists[slot] = gen[1:]                       # drop <start>
        return out_lists

    # ------------------------------------------------------------------
    # Encoder pooling (for analysis-time embeddings)
    # ------------------------------------------------------------------
    def encode_pool(
        self,
        model,
        src_tokens,
        device,
        pool: str = "mean",
    ):
        """Run the encoder on ``src_tokens`` and pool the final-layer
        post-norm output to a ``(d_model,)`` vector. ``pool`` is ``"mean"`` or
        ``"last"``. Returns a torch tensor on CPU. Empty src returns a zero
        ``d_model`` vector instead of raising — keeps embedding loops uniform.
        """
        import torch  # local: program_io stays importable without torch

        if src_tokens.numel() == 0:
            return torch.zeros(model.d_model)
        with torch.no_grad():
            src = src_tokens.to(device).unsqueeze(0)
            mem = model.encode(src)  # (1, L, d)
            if pool == "mean":
                vec = mem.mean(dim=1)
            elif pool == "last":
                vec = mem[:, -1, :]
            else:
                raise ValueError(f"Unknown pool={pool!r}")
        return vec.squeeze(0).cpu()

    # ------------------------------------------------------------------
    # Execute / check
    # ------------------------------------------------------------------
    def _get_compiler(self) -> JITCompiler:
        if self._compiler is None:
            self._compiler = JITCompiler(self.grammar)
        return self._compiler

    def execute(
        self,
        program_str: str,
        query_input: list[int],
        timeout: float | None = None,
    ) -> list[int] | None:
        """Parse + JIT-compile + run on ``query_input`` with a SIGALRM timeout.

        Returns ``None`` on parse failure, compile failure, runtime exception,
        timeout, or non-list output. Numeric output is reduced ``% 100`` per
        project convention (see project_int_mod memory).
        """
        timeout = timeout if timeout is not None else self.exec_timeout
        try:
            fn, _ = self._get_compiler().compile(parse(program_str))
            with alarm(timeout):
                output = fn(list(query_input))
            if not isinstance(output, list):
                return None
            return [int(x) % 100 for x in output]
        except Exception:
            return None

    def check_program(
        self,
        program_str: str,
        io_pairs: list[tuple[list[int], list[int]]],
        timeout: float | None = None,
    ) -> bool:
        """``True`` iff compiling ``program_str`` and applying it to each
        input reproduces the corresponding expected output for every
        ``io_pair``. Failure modes (parse, compile, runtime, timeout, mismatch)
        all collapse to ``False``.
        """
        timeout = timeout if timeout is not None else self.exec_timeout
        try:
            fn, _ = self._get_compiler().compile(parse(program_str))
            for inp, expected in io_pairs:
                with alarm(timeout):
                    output = fn(list(inp))
                if not isinstance(output, list) or [x % 100 for x in output] != list(expected):
                    return False
            return True
        except Exception:
            return False
