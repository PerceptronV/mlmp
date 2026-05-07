import json
import random
from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset
from tqdm import tqdm

from .tokeniser import (
    Tokeniser,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    TO_TOKEN,
    DEFINED_AS_TOKEN,
    SEP_TOKEN,
    NEWLINE_TOKEN,
)
from .io_sampler import RuleIOSampler

TrainingMode = Literal["in-weight", "symbol-shuffling", "easy-symbol-shuffling"]
TRAINING_MODES: tuple[TrainingMode, ...] = (
    "in-weight",
    "symbol-shuffling",
    "easy-symbol-shuffling",
)


class ProgramDataset(Dataset):
    """Corpus-A program dataset.

    Loads ``list[int] -> list[int]`` programs from one or more corpus JSON files
    (each entry ``{"program": str, "type": str, "size": int}``) and, for each
    item, samples I/O pairs on the fly via ``RuleIOSampler``.

    Each program is seen ``n_io_views`` times across the dataset, with
    ``n_io_shown`` ranging from ``min_n_io_shown..max_n_io_shown``. The same
    program always samples the same I/O pool (seed = ``base_seed * 1000003 +
    prog_idx``); the n-th view simply takes the first ``n`` pairs of that fixed
    pool.

    Sequence layout depends on ``mode``:

    ``in-weight`` (default):
        [io_1.input] → [io_1.output] \n
        ...
        [io_n.input] → [io_n.output] \n
        <start> <program tokens> <end>

    ``symbol-shuffling``:
        <mapped_1> ≜ <orig_1> \n   ... <mapped_K> ≜ <orig_K> \n
        <SEP>
        [io_1.input] → [io_1.output] \n  ...
        <start> <program with orig fn names rewritten to mapped names> <end>
        A fresh random permutation over all grammar function names is drawn for
        every ``__getitem__`` call ("per episode"). Lambda parameters and ints
        are not renamed.

    ``easy-symbol-shuffling``:
        Same layout as ``symbol-shuffling`` but only ``n_permuted`` functions
        (a fresh random subset per episode) are permuted; the rest pass
        through unchanged in both preamble and program. Set ``n_permuted``
        externally (e.g. by the training loop) to drive a curriculum from
        a small K up to ``len(fn_names)``. ``None`` ≡ all functions, i.e.
        equivalent to ``symbol-shuffling``.

    Loss mask is 0 over the prefix (preamble + I/O context) and 1 over the
    program tokens.
    """

    def __init__(
        self,
        corpus_files: Path | list[Path],
        seed: int = 0,
        n_io_per_program: int = 11,
        min_n_io_shown: int = 1,
        type_filter: str | None = "list[int]",
        io_sampler: RuleIOSampler | None = None,
        mode: TrainingMode = "in-weight",
        filter_empty_io: bool = False,
        max_programs: int | None = None,
    ):
        assert 1 <= min_n_io_shown <= n_io_per_program, (
            f"min_n_io_shown={min_n_io_shown} must be in [1, n_io_per_program={n_io_per_program}]"
        )
        assert mode in TRAINING_MODES, f"mode={mode!r} must be one of {TRAINING_MODES}"
        self.tokeniser = Tokeniser()
        self.seed = seed
        self.n_io_per_program = n_io_per_program
        self.min_n_io_shown = min_n_io_shown
        self.mode: TrainingMode = mode
        self.fn_names: list[str] = list(self.tokeniser.grammar.functions.keys())

        if isinstance(corpus_files, Path):
            corpus_files = [corpus_files]
        self.corpus_files = corpus_files

        self.programs: list[dict] = []
        for path in corpus_files:
            assert path.exists(), f"Corpus file not found: {path}"
            with open(path, "r") as f:
                entries = json.load(f)
            if type_filter is not None:
                entries = [e for e in entries if e.get("type") == type_filter]
            self.programs.extend(entries)
        assert len(self.programs) > 0, f"No programs loaded from {corpus_files}"

        if max_programs is not None and len(self.programs) > max_programs:
            # Random subsample (not slice) since the corpus is likely stored in
            # enumeration order — taking the first N would skew toward small programs.
            import random as _random
            n_before = len(self.programs)
            _random.Random(seed).shuffle(self.programs)
            self.programs = self.programs[:max_programs]
            print(f"Subsampled corpus: {len(self.programs):,} / {n_before:,} programs (cap={max_programs:,}, seed={seed})")

        self.io_sampler = io_sampler or RuleIOSampler(num_io_pairs=n_io_per_program)

        self.pad = self.tokeniser.vocab.stoi[PAD_TOKEN]
        self.to = self.tokeniser.vocab.stoi[TO_TOKEN]
        self.defined_as = self.tokeniser.vocab.stoi[DEFINED_AS_TOKEN]
        self.newline = self.tokeniser.vocab.stoi[NEWLINE_TOKEN]
        self.sep = self.tokeniser.vocab.stoi[SEP_TOKEN]
        self.start = self.tokeniser.vocab.stoi[START_TOKEN]
        self.end = self.tokeniser.vocab.stoi[END_TOKEN]

        self._io_cache: dict[int, list[tuple[list[int], list[int]]]] = {}
        self._prog_idx_redirect: dict[int, int] = {}

        # easy-symbol-shuffling curriculum knob: number of grammar functions
        # to permute per episode. ``None`` means "all of them" (i.e. behaves
        # exactly like ``symbol-shuffling``). The training loop is expected
        # to mutate this between epochs to ramp difficulty.
        self.n_permuted: int | None = None

        if filter_empty_io:
            self._filter_empty_io_programs()

    @property
    def max_n_io_shown(self) -> int:
        return self.n_io_per_program

    @property
    def n_io_views(self) -> int:
        return self.max_n_io_shown - self.min_n_io_shown + 1

    def __len__(self) -> int:
        return len(self.programs) * self.n_io_views

    def _get_io_pairs(self, prog_idx: int) -> list[tuple[list[int], list[int]]]:
        """Sample (and cache) the I/O pool for a given program."""
        if prog_idx not in self._io_cache:
            rng = random.Random(self.seed * 1000003 + prog_idx)
            self._io_cache[prog_idx] = self.io_sampler.sample(
                self.programs[prog_idx]["program"], rng
            )
        return self._io_cache[prog_idx]

    def _resolve_prog_idx(self, prog_idx: int) -> int:
        """Return ``prog_idx`` if its IO pool is non-empty; otherwise walk
        forward (mod ``len(self.programs)``) until we find one that is, and
        cache the redirect.

        An empty-IO program would otherwise yield a malformed item: in
        in-weight mode the encoder src would be 0-length, producing NaN logits
        from cross-attention over an empty memory; in symbol-shuffling mode the
        item would still train but with no I/O signal at all. Redirecting
        keeps ``len(self)`` stable while guaranteeing every item has at least
        one valid I/O pair to condition on.
        """
        if prog_idx in self._prog_idx_redirect:
            return self._prog_idx_redirect[prog_idx]
        n = len(self.programs)
        for offset in range(n):
            cur = (prog_idx + offset) % n
            if self._get_io_pairs(cur):
                self._prog_idx_redirect[prog_idx] = cur
                return cur
        raise RuntimeError("No programs in the corpus have non-empty IO pools")

    def _filter_empty_io_programs(self) -> None:
        """Drop programs whose IO sampler returns an empty pool.

        Pre-samples each program's IO pool with the same seed scheme that
        ``_get_io_pairs`` would use, removes programs with no valid pairs, and
        keeps the surviving pools warm in ``_io_cache`` (re-keyed against the
        post-filter indices) so the work isn't repeated on first access.

        Pre-existing bug context: ``RuleIOSampler.sample`` returns ``[]`` when a
        program fails to compile or raises on every candidate input. Such
        programs end up with a 0-length encoder source in in-weight mode,
        which crashes dense-path RoPE inside ``greedy_decode``.
        """
        n_before = len(self.programs)
        kept_programs: list[dict] = []
        kept_cache: dict[int, list[tuple[list[int], list[int]]]] = {}
        for prog_idx, program in enumerate(
            tqdm(self.programs, desc="Filtering empty-IO programs")
        ):
            rng = random.Random(self.seed * 1000003 + prog_idx)
            pairs = self.io_sampler.sample(program["program"], rng)
            if pairs:
                kept_cache[len(kept_programs)] = pairs
                kept_programs.append(program)
        self.programs = kept_programs
        self._io_cache = kept_cache
        n_after = len(self.programs)
        n_dropped = n_before - n_after
        pct = 100.0 * n_after / n_before if n_before else 0.0
        print(
            f"Filtered empty-IO programs: kept {n_after:,} / {n_before:,} "
            f"({pct:.2f}%); dropped {n_dropped:,}"
        )
        # NOTE: post-filter, ``prog_idx`` in ``_get_io_pairs`` indexes the kept
        # list, so the seed for any future re-sample (e.g. cache eviction) is
        # tied to the program's *new* position. This is fine for training but
        # means IO pools differ between filtered and unfiltered runs.

    def _tokenise_io_pairs(self, io_pairs: list[tuple[list[int], list[int]]]) -> list[int]:
        x: list[int] = []
        for inp, out in io_pairs:
            x.extend(self.tokeniser.tokenise_list(inp) + [self.to])
            x.extend(self.tokeniser.tokenise_list(out) + [self.newline])
        return x

    def _episode_rng(self, idx: int) -> random.Random:
        """Per-episode RNG, deterministic in ``(self.seed, idx)``. Distinct from
        the I/O sampler's seed scheme so the I/O pool and the symbol permutation
        don't share a stream."""
        return random.Random(self.seed * 1000037 + idx * 7919 + 13)

    def _sample_name_map(self, rng: random.Random) -> dict[str, str]:
        """Draw a permutation over all grammar function names from ``rng``.
        Returns a dict mapping orig_name -> mapped_name (the mapped name is
        what appears in the rewritten program; the preamble lists each
        ``mapped ≜ orig``)."""
        permuted = list(self.fn_names)
        rng.shuffle(permuted)
        return dict(zip(self.fn_names, permuted))

    def _sample_partial_name_map(self, rng: random.Random, k: int) -> dict[str, str]:
        """Draw a permutation over a random K-subset of the grammar function
        names. Functions outside the subset are absent from the map and
        pass through unchanged in both preamble and program rewrites.
        ``k`` is clamped to ``[0, len(fn_names)]``."""
        k = max(0, min(k, len(self.fn_names)))
        if k == 0:
            return {}
        chosen = rng.sample(self.fn_names, k)
        permuted = list(chosen)
        rng.shuffle(permuted)
        return dict(zip(chosen, permuted))

    def _tokenise_preamble(self, name_map: dict[str, str]) -> list[int]:
        """Emit ``<mapped> ≜ <orig> \\n`` lines in a fixed canonical order
        (alphabetical by mapped_fn_name), terminated by ``<SEP> \\n``. The fixed
        order means the only signal the model gets about the permutation is
        the ``≜`` lines themselves, not positional cues. The trailing newline
        gives a clean visual / token-level break before the I/O examples."""
        order = sorted(name_map.items(), key=lambda kv: kv[1])
        toks: list[int] = []
        for orig, mapped in order:
            toks.append(self.tokeniser.tokenise_element(mapped))
            toks.append(self.defined_as)
            toks.append(self.tokeniser.tokenise_element(orig))
            toks.append(self.newline)
        toks.append(self.sep)
        toks.append(self.newline)
        return toks

    def tokenise_program_item(
        self,
        program_str: str,
        io_pairs: list[tuple[list[int], list[int]]],
        name_map: dict[str, str] | None = None,
    ) -> tuple[list[int], list[int]]:
        x: list[int] = []
        if name_map is not None:
            x.extend(self._tokenise_preamble(name_map))
        x.extend(self._tokenise_io_pairs(io_pairs))
        y = [self.start] + self.tokeniser.tokenise_program(program_str, name_map) + [self.end]
        return x, y

    def __getitem__(self, idx: int, include_program: bool = False):
        prog_idx = self._resolve_prog_idx(idx // self.n_io_views)
        n_io_shown = idx % self.n_io_views + self.min_n_io_shown

        program = self.programs[prog_idx]
        io_pairs = self._get_io_pairs(prog_idx)[:n_io_shown]

        name_map = None
        if self.mode == "symbol-shuffling":
            name_map = self._sample_name_map(self._episode_rng(idx))
        elif self.mode == "easy-symbol-shuffling":
            k = self.n_permuted if self.n_permuted is not None else len(self.fn_names)
            name_map = self._sample_partial_name_map(self._episode_rng(idx), k)
        x, y = self.tokenise_program_item(program["program"], io_pairs, name_map)
        # loss mask has length seq_len - 1 (no prediction at first token);
        # 1 over the predictions of program tokens following <start>.
        loss_mask = [0] * len(x) + [1] * (len(y) - 1)

        if include_program:
            info = {**program, "n_io_shown": n_io_shown, "io_pairs": io_pairs, "name_map": name_map}
            return x + y, loss_mask, info
        return x + y, loss_mask


if __name__ == "__main__":
    while 1:
        try:
            corpus_input = input(
                "Enter corpus JSON file(s) (comma-separated, default datasets/corpus-a/rl_corpus.json): "
            ).strip()
            if corpus_input == "":
                corpus_input = "datasets/corpus-a/rl_corpus.json"
            corpus_files = [Path(p.strip()) for p in corpus_input.split(",")]

            seed_input = input("Seed (default 0): ").strip()
            seed = int(seed_input) if seed_input else 0

            mode_input = input(f"Mode {TRAINING_MODES} (default in-weight): ").strip() or "in-weight"
            assert mode_input in TRAINING_MODES, f"Unknown mode: {mode_input}"

            dataset = ProgramDataset(corpus_files=corpus_files, seed=seed, mode=mode_input)
            print(f"\nLoaded {len(dataset.programs):,} programs"
                  f" -> {len(dataset):,} items"
                  f" (n_io_shown range: {dataset.min_n_io_shown}..{dataset.max_n_io_shown})"
                  f" mode={dataset.mode}")

            while 1:
                idx = input("\nEnter index (-1 to exit): ")
                try:
                    idx = int(idx)
                except ValueError:
                    print("Invalid index. Please enter a valid integer.")
                    continue
                if idx < 0:
                    print("\n")
                    break
                if idx >= len(dataset):
                    print(f"Index out of range [0, {len(dataset)-1}].")
                    continue

                seq, loss_mask, info = dataset.__getitem__(idx, include_program=True)
                loss_mask = [0] + loss_mask  # align with seq

                GREEN, DIM, RESET = "\033[92m", "\033[2m", "\033[0m"
                print(f"\n{DIM}--- raw ---{RESET}")
                print(f"  prog_idx   : {idx // dataset.n_io_views}")
                print(f"  n_io_shown : {info['n_io_shown']}")
                print(f"  type       : {info.get('type')}    size: {info.get('size')}")
                print(f"  program    : {info['program']}")
                print(f"  io_pairs   :")
                for inp, out in info["io_pairs"]:
                    print(f"      {inp} → {out}")

                print(f"\n{DIM}--- token ids (len {len(seq)}, mask sum {sum(loss_mask)}) ---{RESET}")
                print(" ", " ".join(str(t) for t in seq))

                print(f"\n{DIM}--- detokenised (green = predicted / loss-masked tokens) ---{RESET}")
                print(" ", " ".join(
                    f"{GREEN}{dataset.tokeniser.vocab.itos[t]}{RESET}" if m
                    else dataset.tokeniser.vocab.itos[t]
                    for t, m in zip(seq, loss_mask)
                ))
                print()
        except KeyboardInterrupt:
            break
