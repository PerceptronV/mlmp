"""Inverse-MLC dataset adapter.

Wraps Brenden Lake's MLC ``data_algebraic`` dataset (the one stored in
``src/data/inverse-mlc/``) so each query becomes a single ``(seq, loss_mask)``
item shaped like the items emitted by :class:`src.data.dataloader.ProgramDataset`,
i.e. consumable by ``src.train.collate_fn`` without changes.

Each MLC episode file has ``ns_max=14`` support pairs and ``nq=10`` queries.
We expose one item per (episode, query), so
``len(self) == len(self.programs) * self.n_io_views`` with
``self.n_io_views == 10``. ``programs`` lists episodes (one entry per file),
matching the prog-idx / view-idx scheme ``train.py`` already uses for its
validation accuracy sweep.

Sequence layout per item:
    encoder src = xq_context  (the per-query encoder input that
                  ``bundle_biml_episode`` builds: the query plus the support
                  context separated by ``|`` and ``->`` tokens)
    decoder tgt = <SOS> + yq + <EOS>
    seq         = encoder_src + decoder_tgt
    loss_mask   = [0] * len(encoder_src) + [1] * (len(decoder_tgt) - 1)

The MLC vocabulary is small (~24 tokens). All tokens — separators, grammar
variables, colours, pseudowords, specials — share a single :class:`Lang`
(MLC's ``DataAlg`` already builds it that way), so encoder and decoder use
the same embedding / projection.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .inverse_mlc import (
    DataAlg,
    DataAlgAndBias,
    DataRetrieve,
    SOS_token,
    EOS_token,
    PAD_token,
    input_symbols_list_default,
    str_to_grammar,
)


_INVERSE_MLC_DIR = Path(__file__).resolve().parent / "inverse_mlc"


INVERSE_MLC_EPISODE_TYPES: tuple[str, ...] = (
    "algebraic",
    "algebraic_noise",
    "algebraic+biases",
    "retrieve",
)


def _build_underlying_dataset(mode: str, episode_type: str, mydir: str):
    """Instantiate the MLC ``Dataset`` subclass that matches ``episode_type``.

    Mirrors the train/val factories in MLC's ``get_dataset`` but takes ``mydir``
    as an absolute path (so we don't depend on the caller's cwd, unlike MLC's
    own ``get_dataset`` which assumes the inverse-mlc dir is the cwd).
    """
    train = mode == "train"
    if episode_type == "algebraic":
        return DataAlg(mode, mydir, p_noise=0.0, min_ns=14 if train else 0)
    if episode_type == "algebraic_noise":
        # Match MLC's own setup: noise enabled in train, val without noise.
        return DataAlg(
            mode, mydir, p_noise=0.01 if train else 0.0, min_ns=14 if train else 0,
        )
    if episode_type == "algebraic+biases":
        # MLC pairs ``DataAlgAndBias`` for train with plain ``DataAlg`` for val,
        # which keeps the validation distribution aligned with the gold grammar.
        if train:
            return DataAlgAndBias(mode, mydir, p_noise=0.0, min_ns=14)
        return DataAlg(mode, mydir)
    if episode_type == "retrieve":
        # ``DataRetrieve`` already pins ``inc_support_in_query=True`` internally.
        return DataRetrieve(mode, mydir, min_ns=14, max_ns=14)
    raise ValueError(
        f"Unknown episode_type={episode_type!r}; "
        f"expected one of {INVERSE_MLC_EPISODE_TYPES}"
    )


class _VocabShim:
    """Tiny adapter so callers using ``ProgramDataset.tokeniser.vocab`` (``len``,
    ``itos``, ``stoi``) work unchanged on top of an MLC :class:`Lang`."""

    def __init__(self, lang):
        self._lang = lang

    def __len__(self) -> int:
        return self._lang.n_symbols

    @property
    def itos(self):
        # Construct each access — vocab is tiny and this avoids stale state if
        # the lang is mutated (MLC code doesn't mutate, but cheap insurance).
        return [self._lang.index2symbol[i] for i in range(self._lang.n_symbols)]

    @property
    def stoi(self):
        return self._lang.symbol2index


class _TokeniserShim:
    """Mimics ``Tokeniser`` to the extent ``train.py`` and accuracy code use it
    (``vocab`` and ``detokenise``). The underlying conversion is just a Lang
    index lookup."""

    def __init__(self, lang):
        self._lang = lang
        self.vocab = _VocabShim(lang)

    def detokenise(self, toks: list[int]) -> str:
        return ' '.join(self._lang.index2symbol[t] for t in toks)


class InverseMLCDataset(Dataset):
    """Inverse-MLC algebraic episodes presented as flat ``(seq, mask)`` items.

    Parameters
    ----------
    mode : ``"train"`` or ``"val"``.
    episode_type : one of ``INVERSE_MLC_EPISODE_TYPES``.
    data_root : path to ``data_algebraic`` (containing ``train/`` and ``val/``
        sub-dirs of ``.txt`` episode files). Defaults to the copy that lives
        at ``src/data/inverse-mlc/data_algebraic/`` in this repo.

    Notes
    -----
    * The underlying MLC dataset re-samples a random support subset on every
      read in train mode; in val mode it returns the full support unchanged.
      Because we expose 10 items per file (one per query), the same file can
      see different support subsets across its 10 items in a single epoch.
      That's a slight divergence from MLC's per-episode sampling but is fine
      meta-learning signal in practice.
    * MLC numbers files starting at 1 and there's no per-file metadata beyond
      the filename, so ``programs[i]`` is a small dict ``{"file": <path>,
      "type": "inverse-mlc-episode"}``. ``size`` is omitted (no analogue).
    """

    def __init__(
        self,
        mode: str = "train",
        episode_type: str = "algebraic",
        data_root: Path | str | None = None,
    ):
        assert mode in ("train", "val"), f"mode must be 'train' or 'val', got {mode!r}"
        if data_root is None:
            data_root = _INVERSE_MLC_DIR / "data_algebraic"
        data_root = Path(data_root)
        assert (data_root / mode).exists(), (
            f"Expected {data_root / mode} to exist (it should contain the .txt episode files)"
        )

        self.mode = mode
        self.episode_type = episode_type
        self.data_root = data_root
        self.D = _build_underlying_dataset(mode, episode_type, str(data_root))

        # Shared input/output vocab (DataAlg / DataAlgAndBias / DataRetrieve all
        # construct ``langs`` from ``combine_input_output_symb``, so input and
        # output langs use identical indices — safe to use one for both).
        self.langs = self.D.langs
        self._lang = self.langs["input"]
        self.tokeniser = _TokeniserShim(self._lang)

        self.start = self._lang.symbol2index[SOS_token]
        self.end = self._lang.symbol2index[EOS_token]
        self.pad = self._lang.symbol2index[PAD_token]

        # All inverse-mlc episode files have exactly 10 queries.
        self._queries_per_episode = 10
        self.programs = [
            {"file": p, "type": "inverse-mlc-episode"} for p in self.D.list_items
        ]

    @property
    def n_io_views(self) -> int:
        return self._queries_per_episode

    def __len__(self) -> int:
        return len(self.programs) * self._queries_per_episode

    def _tokenise(self, symbols: list[str]) -> list[int]:
        # ``symbol2index`` raises KeyError on unknown symbols — that's the
        # correct failure mode (it'd mean the vocab and the file disagree).
        return [self._lang.symbol2index[s] for s in symbols]

    def __getitem__(self, idx: int, include_program: bool = False):
        prog_idx = idx // self._queries_per_episode
        q_idx = idx % self._queries_per_episode

        sample = self.D[prog_idx]
        nq = len(sample["xq"])
        # Belt-and-braces: MLC's algebraic files all have nq=10, but DataAlgAndBias
        # can in principle change query count via heuristics. Fold to a valid q.
        q_idx = q_idx % nq

        x = self._tokenise(sample["xq_context"][q_idx])
        yq_ids = self._tokenise(sample["yq"][q_idx])
        y = [self.start] + yq_ids + [self.end]

        seq = x + y
        loss_mask = [0] * len(x) + [1] * (len(y) - 1)

        if include_program:
            info = {
                "file": self.programs[prog_idx]["file"],
                "q_idx": q_idx,
                "xq": sample["xq"][q_idx],
                "yq": sample["yq"][q_idx],
                "yq_token_ids": yq_ids,
                # Per-episode gold grammar string. ``check_prediction`` parses
                # this lazily into a ``Grammar`` and applies it to the model's
                # predicted pseudoword sequence to recover the colour string,
                # which is then compared against ``xq``.
                "grammar_str": sample["aux"]["grammar_str"],
                # Empty list short-circuits the program-compile accuracy path
                # in train.py — we provide ``check_prediction`` instead, which
                # train.py prefers when present.
                "io_pairs": [],
            }
            return seq, loss_mask, info
        return seq, loss_mask

    def check_prediction(self, generated_token_ids: list[int], info: dict) -> bool:
        """Functional-equivalence accuracy (matches MLC's ``batch_acc`` default,
        ``exact_match=False`` in eval.py).

        Pseudowords ARE the program: each episode file ships a gold grammar
        (the ``*GRAMMAR*`` block) of rules like ``dax -> BLUE`` and
        ``u1 fep u2 -> [u1] [u2]``. To check correctness we

        1. detokenise the greedy-decoded ids into pseudoword strings,
        2. parse the per-episode grammar string into a ``Grammar``,
        3. apply the grammar to the predicted pseudoword sequence to get
           a colour sequence, and
        4. compare that colour sequence to ``info["xq"]`` — the query's
           input colours, which is what the model was conditioned on.

        Multiple distinct pseudoword sequences can map to the same colour
        sequence, so functional equivalence (not byte-identity to ``yq``) is
        the right notion of correctness — and the one MLC reports in the
        paper. Returns ``False`` on any grammar parse / apply failure
        (e.g. the model emits a malformed program, or hits the grammar's
        recursion cap).
        """
        toks = generated_token_ids
        if toks and toks[-1] == self.end:
            toks = toks[:-1]
        predicted_program = ' '.join(self._lang.index2symbol[t] for t in toks)
        grammar = str_to_grammar(info["grammar_str"], input_symbols_list_default)
        if not grammar:
            # ``str_to_grammar`` returns ``[]`` (falsy) on parse failure. The
            # gold grammar should always parse, so this is mostly defensive.
            return False
        try:
            reconstructed_colours = grammar.apply(predicted_program)
        except Exception:
            # ``Grammar.apply`` can raise on malformed programs (unbound vars,
            # recursion cap, etc.) — those count as wrong answers.
            return False
        return reconstructed_colours == ' '.join(info["xq"])


if __name__ == "__main__":
    # Smoke-test: build a small train/val pair and dump one example.
    ds = InverseMLCDataset(mode="train", episode_type="algebraic")
    print(f"# programs: {len(ds.programs):,}; total items: {len(ds):,}; vocab: {len(ds.tokeniser.vocab)}")
    seq, mask, info = ds.__getitem__(0, include_program=True)
    print(f"seq len: {len(seq)}, x len: {mask.count(0)}, y len: {sum(mask) + 1}")
    print(f"file: {info['file']}, q_idx: {info['q_idx']}")
    print(f"xq    : {info['xq']}")
    print(f"yq    : {info['yq']}")
    print(f"src   : {ds.tokeniser.detokenise(seq[:mask.count(0)])}")
    print(f"tgt   : {ds.tokeniser.detokenise(seq[mask.count(0):])}")
