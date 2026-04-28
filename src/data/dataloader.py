import json
import random
from pathlib import Path
from torch.utils.data import Dataset

from .tokeniser import (
    Tokeniser,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    TO_TOKEN,
    SEP_TOKEN,
    NEWLINE_TOKEN,
)
from .io_sampler import RuleIOSampler


class ProgramDataset(Dataset):
    """Corpus-A program dataset.

    Loads ``list[int] -> list[int]`` programs from one or more corpus JSON files
    (each entry ``{"program": str, "type": str, "size": int}``) and, for each
    item, samples I/O pairs on the fly via ``RuleIOSampler``.

    Each program is seen ``max_n_io_shown`` times across the dataset, with
    ``n_io_shown`` ranging from ``1..max_n_io_shown``. The same program always
    samples the same I/O pool (seed = ``base_seed * 1000003 + prog_idx``); the
    n-th view simply takes the first ``n`` pairs of that fixed pool.

    Sequence layout (matches the original episode format minus the support set):
        [io_1.input] → [io_1.output] \n
        ...
        [io_n.input] → [io_n.output] \n
        <start> <program tokens> <end>
    Loss mask is 0 over the I/O context and 1 over the program tokens
    (predicting the program from its I/O examples).
    """

    def __init__(
        self,
        corpus_files: Path | list[Path],
        seed: int = 0,
        n_io_per_program: int = 11,
        type_filter: str | None = "list[int]",
        io_sampler: RuleIOSampler | None = None,
    ):
        self.tokeniser = Tokeniser()
        self.seed = seed
        self.n_io_per_program = n_io_per_program

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

        self.io_sampler = io_sampler or RuleIOSampler(num_io_pairs=n_io_per_program)

        self.pad = self.tokeniser.vocab.stoi[PAD_TOKEN]
        self.to = self.tokeniser.vocab.stoi[TO_TOKEN]
        self.newline = self.tokeniser.vocab.stoi[NEWLINE_TOKEN]
        self.sep = self.tokeniser.vocab.stoi[SEP_TOKEN]
        self.start = self.tokeniser.vocab.stoi[START_TOKEN]
        self.end = self.tokeniser.vocab.stoi[END_TOKEN]

        self._io_cache: dict[int, list[tuple[list[int], list[int]]]] = {}

    @property
    def max_n_io_shown(self) -> int:
        return self.n_io_per_program

    def __len__(self) -> int:
        return len(self.programs) * self.max_n_io_shown

    def _get_io_pairs(self, prog_idx: int) -> list[tuple[list[int], list[int]]]:
        """Sample (and cache) the I/O pool for a given program."""
        if prog_idx not in self._io_cache:
            rng = random.Random(self.seed * 1000003 + prog_idx)
            self._io_cache[prog_idx] = self.io_sampler.sample(
                self.programs[prog_idx]["program"], rng
            )
        return self._io_cache[prog_idx]

    def tokenise_program_item(
        self,
        program_str: str,
        io_pairs: list[tuple[list[int], list[int]]],
    ) -> tuple[list[int], list[int]]:
        x: list[int] = []
        for inp, out in io_pairs:
            x.extend(self.tokeniser.tokenise_list(inp) + [self.to])
            x.extend(self.tokeniser.tokenise_list(out) + [self.newline])
        y = [self.start] + self.tokeniser.tokenise_program(program_str) + [self.end]
        return x, y

    def __getitem__(self, idx: int, include_program: bool = False):
        prog_idx = idx // self.max_n_io_shown
        n_io_shown = idx % self.max_n_io_shown + 1

        program = self.programs[prog_idx]
        io_pairs = self._get_io_pairs(prog_idx)[:n_io_shown]

        x, y = self.tokenise_program_item(program["program"], io_pairs)
        # loss mask has length seq_len - 1 (no prediction at first token);
        # 1 over the predictions of program tokens following <start>.
        loss_mask = [0] * len(x) + [1] * (len(y) - 1)

        if include_program:
            return x + y, loss_mask, {**program, "n_io_shown": n_io_shown, "io_pairs": io_pairs}
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

            dataset = ProgramDataset(corpus_files=corpus_files, seed=seed)
            print(f"\nLoaded {len(dataset.programs):,} programs"
                  f" -> {len(dataset):,} items (max_n_io_shown={dataset.max_n_io_shown})")

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
                print(f"  prog_idx   : {idx // dataset.max_n_io_shown}")
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
