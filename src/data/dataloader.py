import hashlib
import json
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .program_io import ProgramIO
from .sampler import RuleIOSampler
from ..lang.grammar import DefaultGrammar, Grammar

TrainingMode = Literal["in-weight", "symbol-shuffling", "easy-symbol-shuffling"]
TRAINING_MODES: tuple[TrainingMode, ...] = (
    "in-weight",
    "symbol-shuffling",
    "easy-symbol-shuffling",
)


# Per-worker sampler for the parallel I/O-pool pass. Built once per process
# rather than pickled: RuleIOSampler holds a JITCompiler and a compiled-function
# cache, neither of which survives the process boundary.
_WORKER_SAMPLER: RuleIOSampler | None = None


def _init_pool_worker(n_io_per_program: int) -> None:
    global _WORKER_SAMPLER
    _WORKER_SAMPLER = RuleIOSampler(num_io_pairs=n_io_per_program)


def _pool_size_worker(task: tuple[str, int]) -> int:
    """Only the pool *size* crosses back — the pools themselves are gigabytes."""
    program_str, seed = task
    assert _WORKER_SAMPLER is not None
    return len(_WORKER_SAMPLER.sample(program_str, random.Random(seed)))


class ProgramDataset(Dataset):
    """Corpus-A program dataset.

    Loads ``list[int] -> list[int]`` programs from one or more corpus JSON files
    (each entry ``{"program": str, "type": str, "size": int}``) and, for each
    item, samples I/O pairs on the fly via ``RuleIOSampler``.

    Two corpus filters run before anything else — split, subsample, indexing
    and per-episode sampling all see the filtered corpus, and train and val
    must be built with the same values or a holdout split stops being a
    partition:

    ``max_program_length``
        drops programs whose target exceeds N decoder tokens.
    ``min_io_pairs``
        drops programs whose sampled I/O pool is smaller than N pairs.
        ``min_io_pairs=n_io_per_program`` keeps only programs that can fill
        every view, i.e. "train on programs with all N examples", while
        ``min_io_pairs=1`` just drops the programs with no usable I/O at all.
        Deciding it costs one pass of program execution over the corpus, so it
        runs across ``io_workers`` processes and its verdict is cached on disk
        (disable with ``io_pool_cache=False``; relocate with ``MLMP_CACHE_DIR``).

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
        min_io_pairs: int | None = None,
        io_workers: int | None = None,
        io_pool_cache: bool = True,
        max_programs: int | None = None,
        max_program_length: int | None = None,
        grammar: Grammar = DefaultGrammar,
        holdout: int | None = None,
        split: str = "train",
        split_seed: int = 0,
    ):
        assert 1 <= min_n_io_shown <= n_io_per_program, (
            f"min_n_io_shown={min_n_io_shown} must be in [1, n_io_per_program={n_io_per_program}]"
        )
        assert mode in TRAINING_MODES, f"mode={mode!r} must be one of {TRAINING_MODES}"
        assert min_io_pairs is None or 1 <= min_io_pairs <= n_io_per_program, (
            f"min_io_pairs={min_io_pairs} must be in [1, n_io_per_program={n_io_per_program}]"
        )
        min_pairs = min_io_pairs
        # Single source of truth for tokenisation, symbol-shuffling preamble
        # sampling, decode, and program execution. ``ProgramDataset`` is a thin
        # wrapper around an instance of it (plus corpus iteration / IO-sampler
        # plumbing). See ``src/data/program_io.py`` for the canonical home.
        # ``grammar`` flows into the tokeniser vocab AND the symbol-shuffling
        # name permutation, so the shuffled fn names stay confined to this
        # grammar's functions (matters for the *-symbol-shuffling modes).
        self.io: ProgramIO = ProgramIO(grammar=grammar)
        self.tokeniser = self.io.tokeniser
        self.seed = seed
        self.n_io_per_program = n_io_per_program
        self.min_n_io_shown = min_n_io_shown
        self.mode: TrainingMode = mode
        self.max_program_length = max_program_length
        self.min_io_pairs = min_pairs
        self.fn_names: list[str] = self.io.fn_names

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

        if max_program_length is not None:
            # Length-filter FIRST, so the holdout split, the max_programs
            # subsample, indexing and every per-episode randomisation below run
            # over the kept programs only. Long targets are both hard to learn
            # and, past ``--max-program-tokens`` (80 by default), impossible to
            # emit at eval time.
            n_before = len(self.programs)
            self.programs = [
                e for e in self.programs
                if self._within_length(e["program"], max_program_length)
            ]
            n_after = len(self.programs)
            assert n_after > 0, (
                f"max_program_length={max_program_length} filtered out every "
                f"program in {corpus_files}"
            )
            print(f"Length filter (<= {max_program_length} tokens): kept "
                  f"{n_after:,} / {n_before:,} programs "
                  f"({n_after / n_before:.2%}); dropped {n_before - n_after:,}")

        self._custom_sampler = io_sampler is not None
        self.io_sampler = io_sampler or RuleIOSampler(num_io_pairs=n_io_per_program)

        # A program's I/O pool is seeded on its position, so filtering would
        # otherwise re-seed every survivor and hand it a *different* pool than
        # the one it was selected on. Carrying the pre-filter index through the
        # split and subsample below keeps each pool exactly what the filter saw
        # — and means the filter only has to remember one int per program
        # instead of the pool itself (~8 GB at corpus scale).
        self._pool_seed_idx: np.ndarray | None = None
        if min_pairs is not None:
            self._pool_seed_idx = self._filter_by_io_pool(
                min_pairs, io_workers,
                cache_key=self._io_pool_cache_key(
                    corpus_files, type_filter, max_program_length, min_pairs)
                if io_pool_cache else None,
            )

        if holdout:
            # Deterministic held-out split: two datasets constructed over the
            # same corpus_files with the same (holdout, split_seed) but
            # different ``split`` partition the programs disjointly. Applied
            # before the max_programs cap so the cap never eats the val set.
            assert split in ("train", "val"), f"split must be train|val, got {split!r}"
            assert holdout < len(self.programs), (
                f"holdout={holdout} >= corpus size {len(self.programs)}"
            )
            order = list(range(len(self.programs)))
            random.Random(split_seed).shuffle(order)
            keep = order[:holdout] if split == "val" else order[holdout:]
            keep.sort()  # preserve enumeration order within the split
            self.programs = [self.programs[i] for i in keep]
            if self._pool_seed_idx is not None:
                self._pool_seed_idx = self._pool_seed_idx[keep]
            print(f"Holdout split '{split}': {len(self.programs):,} programs "
                  f"(holdout={holdout:,}, split_seed={split_seed})")

        if max_programs is not None and len(self.programs) > max_programs:
            # Random subsample (not slice) since the corpus is likely stored in
            # enumeration order — taking the first N would skew toward small programs.
            import random as _random
            n_before = len(self.programs)
            # Permute indices rather than the list itself so the pool seeds can
            # follow; same RNG stream, so the selected subset is unchanged.
            order = list(range(n_before))
            _random.Random(seed).shuffle(order)
            order = order[:max_programs]
            self.programs = [self.programs[i] for i in order]
            if self._pool_seed_idx is not None:
                self._pool_seed_idx = self._pool_seed_idx[order]
            print(f"Subsampled corpus: {len(self.programs):,} / {n_before:,} programs (cap={max_programs:,}, seed={seed})")

        # Re-export the special-token ids that downstream code reads off the
        # dataset directly (e.g. ``train.py`` reads ``dataset.start`` /
        # ``dataset.end``). These all live on the ``ProgramIO`` now.
        self.pad = self.io.pad
        self.to = self.io.to
        self.defined_as = self.io.defined_as
        self.newline = self.io.newline
        self.sep = self.io.sep
        self.start = self.io.start
        self.end = self.io.end

        self._io_cache: dict[int, list[tuple[list[int], list[int]]]] = {}
        self._prog_idx_redirect: dict[int, int] = {}

        # easy-symbol-shuffling curriculum knob: number of grammar functions
        # to permute per episode. ``None`` means "all of them" (i.e. behaves
        # exactly like ``symbol-shuffling``). The training loop is expected
        # to mutate this between epochs to ramp difficulty.
        self.n_permuted: int | None = None

    def _within_length(self, program_str: str, max_len: int) -> bool:
        """Whether ``program_str`` tokenises to at most ``max_len`` tokens.

        Every token contributes at least one character, so a string no longer
        than the cap is under it without paying for the lexer — which matters
        when the check runs over a multi-million-program corpus at startup.
        """
        if len(program_str) <= max_len:
            return True
        return len(self.tokeniser.tokenise_program(program_str)) <= max_len

    @property
    def max_n_io_shown(self) -> int:
        return self.n_io_per_program

    @property
    def n_io_views(self) -> int:
        return self.max_n_io_shown - self.min_n_io_shown + 1

    def __len__(self) -> int:
        return len(self.programs) * self.n_io_views

    def _get_io_pairs(self, prog_idx: int) -> list[tuple[list[int], list[int]]]:
        """Sample (and cache) the I/O pool for a given program.

        Seeded on the program's pre-filter position when the I/O filter ran, so
        a program keeps the pool its pool size was judged on no matter how the
        corpus was later split or subsampled.
        """
        if prog_idx not in self._io_cache:
            seed_idx = (int(self._pool_seed_idx[prog_idx])
                        if self._pool_seed_idx is not None else prog_idx)
            rng = random.Random(self.seed * 1000003 + seed_idx)
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

    # ------------------------------------------------------------------
    # I/O pool filter
    # ------------------------------------------------------------------
    def _io_pool_cache_key(self, corpus_files, type_filter, max_program_length,
                           min_pairs) -> str:
        """Fingerprint of everything that determines the filter's verdict.

        Corpus identity (path, size, mtime) plus every setting that changes
        which programs are sampled or with what seed. Deliberately excludes
        the holdout split, ``max_programs`` and ``split_seed``: those run
        *after* the filter, so runs that differ only in them share a cache
        entry.
        """
        parts = [
            f"{Path(f).resolve()}:{Path(f).stat().st_size}:{Path(f).stat().st_mtime_ns}"
            for f in corpus_files
        ]
        parts += [f"type={type_filter}", f"maxlen={max_program_length}",
                  f"seed={self.seed}", f"n_io={self.n_io_per_program}",
                  f"min_pairs={min_pairs}"]
        return hashlib.blake2b("|".join(parts).encode(), digest_size=16).hexdigest()

    @staticmethod
    def _io_pool_cache_dir() -> Path:
        return Path(os.environ.get("MLMP_CACHE_DIR",
                                   Path.home() / ".cache" / "mlmp")) / "io_pools"

    def _filter_by_io_pool(self, min_pairs: int, workers: int | None = None,
                           cache_key: str | None = None) -> np.ndarray:
        """Drop programs whose I/O pool has fewer than ``min_pairs`` pairs.

        Returns the surviving programs' pre-filter indices (which is what
        ``_get_io_pairs`` seeds on).

        ``RuleIOSampler.sample`` returns *up to* ``n_io_per_program`` pairs —
        fewer when the program fails to compile, raises on candidate inputs, or
        can't produce enough distinct outputs. Two thresholds matter:

        ``min_pairs=1``
            An empty pool gives a 0-length encoder source in in-weight mode,
            which crashes dense-path RoPE inside ``ProgramIO.decode_one``.
        ``min_pairs=n_io_per_program``
            Every program can then fill every view, so ``n_io_shown`` really is
            the number of pairs shown. Below it, a program with a short pool is
            silently shown fewer pairs than its view asks for, which blurs the
            per-``n_io_shown`` bins in validation.

        Deciding this means executing every program on up to
        ``RuleIOSampler.num_candidates`` inputs — ~0.8 ms each, so ~75 minutes
        for corpus-a's 5.6M programs. Hence the two optimisations: the pass
        runs across processes, and its verdict is cached on disk under
        ``cache_key`` so resumes and reruns skip it entirely. Only pool
        *sizes* cross the process boundary; the pools themselves are re-drawn
        lazily during training from the same seeds, which keeps both the IPC
        and the resident memory O(1) per program.

        Runs before the split and subsample, so a holdout of N really yields N.
        """
        n_before = len(self.programs)
        cache_path = (self._io_pool_cache_dir() / f"{cache_key}.npy"
                      if cache_key else None)
        kept = None
        if cache_path is not None and cache_path.exists():
            try:
                kept = np.load(cache_path)
                print(f"I/O pool filter (>= {min_pairs} pairs): reusing cached "
                      f"verdict for {len(kept):,} / {n_before:,} programs "
                      f"({cache_path})")
            except Exception as e:   # a truncated/corrupt cache must not be fatal
                print(f"Ignoring unreadable I/O pool cache {cache_path}: {e}")
                kept = None

        if kept is None:
            sizes = self._io_pool_sizes(workers)
            kept = np.flatnonzero(sizes >= min_pairs).astype(np.int64)
            n_short = int(((sizes > 0) & (sizes < min_pairs)).sum())
            print(
                f"I/O pool filter (>= {min_pairs} pairs): kept {len(kept):,} / "
                f"{n_before:,} programs ({100.0 * len(kept) / max(1, n_before):.2f}%); "
                f"dropped {n_before - len(kept):,} ({n_short:,} short pools, "
                f"{n_before - len(kept) - n_short:,} empty)"
            )
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cache_path.with_name(cache_path.name + ".tmp")
                    with open(tmp, "wb") as fh:   # a path would gain another .npy
                        np.save(fh, kept)
                    os.replace(tmp, cache_path)
                    print(f"Cached the verdict to {cache_path}")
                except Exception as e:  # a read-only cache dir is not fatal
                    print(f"Could not write I/O pool cache {cache_path}: {e}")

        self.programs = [self.programs[i] for i in kept]
        assert self.programs, (
            f"min_io_pairs={min_pairs} filtered out every program in "
            f"{self.corpus_files}"
        )
        return kept

    @staticmethod
    def _spawn_without_importable_main() -> bool:
        import multiprocessing as mp
        if mp.get_start_method(allow_none=True) not in (None, "spawn"):
            return False
        import __main__
        return getattr(__main__, "__file__", None) is None

    def _io_pool_sizes(self, workers: int | None) -> np.ndarray:
        """Pool size for every program, across ``workers`` processes.

        Falls back to one process for a custom sampler (which the workers
        can't rebuild) or a small corpus (where the fork cost dominates).
        """
        programs = [p["program"] for p in self.programs]
        seeds = [self.seed * 1000003 + i for i in range(len(programs))]
        if workers is None:
            # Starting workers costs more than it saves on a small corpus, and
            # far more under 'spawn' (macOS/Windows), where each worker boots a
            # fresh interpreter and re-imports torch — measured at ~2-4 s each,
            # which loses to a serial pass over anything under ~50k programs.
            # 'fork' (the Linux default, i.e. the cluster) is near-free.
            import multiprocessing as mp
            per_worker = 2_000 if mp.get_start_method(allow_none=True) == "fork" else 50_000
            workers = min(os.cpu_count() or 1, len(programs) // per_worker + 1)
        workers = max(1, workers)
        desc = f"Sampling I/O pools ({workers} proc)" if workers > 1 else "Sampling I/O pools"

        if workers > 1 and self._spawn_without_importable_main():
            # spawn (macOS/Windows default) re-imports __main__ in each worker.
            # From a REPL, a notebook or `python - <<EOF` there is nothing to
            # re-import, and the pool hangs on failing bootstraps rather than
            # raising — so decide up front instead.
            print("Parallel I/O sampling needs an importable __main__ under the "
                  "'spawn' start method; falling back to one process.")
            workers = 1

        if workers == 1 or self._custom_sampler:
            return np.fromiter(
                (len(self.io_sampler.sample(p, random.Random(sd)))
                 for p, sd in tqdm(list(zip(programs, seeds)), desc=desc)),
                dtype=np.int32, count=len(programs))

        from multiprocessing import Pool
        with Pool(workers, initializer=_init_pool_worker,
                  initargs=(self.n_io_per_program,)) as pool:
            sizes = list(tqdm(
                pool.imap(_pool_size_worker, zip(programs, seeds), chunksize=512),
                total=len(programs), desc=desc))
        return np.asarray(sizes, dtype=np.int32)

    # -------- delegates to ``self.io`` (see src/data/program_io.py) --------
    # Everything project-specific about how programs and I/O pairs become
    # tokens lives on ``ProgramIO``. These wrappers exist only because external
    # callers (train.py, the dataset CLI demo, downstream scripts) still reach
    # through ``ProgramDataset`` for them.

    def _episode_rng(self, idx: int) -> random.Random:
        """Per-episode RNG, deterministic in ``(self.seed, idx)``. Distinct
        from the I/O sampler's seed scheme so the I/O pool and the symbol
        permutation don't share a stream. Lives here because the keying scheme
        is dataset-iteration-specific (``idx``)."""
        return random.Random(self.seed * 1000037 + idx * 7919 + 13)

    def _sample_name_map(self, rng: random.Random) -> dict[str, str]:
        return self.io.sample_name_map(rng)

    def _sample_partial_name_map(self, rng: random.Random, k: int) -> dict[str, str]:
        return self.io.sample_partial_name_map(rng, k)

    def tokenise_program_item(
        self,
        program_str: str,
        io_pairs: list[tuple[list[int], list[int]]],
        name_map: dict[str, str] | None = None,
    ) -> tuple[list[int], list[int]]:
        return self.io.tokenise_program_item(program_str, io_pairs, name_map)

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
