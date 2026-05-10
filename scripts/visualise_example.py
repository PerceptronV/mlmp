#!/usr/bin/env python3
"""Visualise a single training example by index.

Builds the same dataset object that ``train.py`` consumes, asks it for one
item via ``__getitem__(idx, include_program=True)``, and prints both the
tokenised (token-id) and untokenised (detokenised string) views of the
encoder *input* and decoder *target* halves.

Two dataset families are supported, gated by ``--dataset``:

    program      → src.data.dataloader.ProgramDataset
                   --mode {in-weight, symbol-shuffling, easy-symbol-shuffling}
                   --corpus FILE [FILE ...]   (corpus-A JSON files)

    inverse-mlc  → src.data.inverse_mlc_dataloader.InverseMLCDataset
                   --mode {train, val}
                   --episode-type {algebraic, algebraic_noise,
                                   algebraic+biases, retrieve}

Examples:
    python scripts/visualise_example.py \\
        --dataset program --mode symbol-shuffling \\
        --corpus datasets/corpus-a/rl_corpus.json --idx 0

    python scripts/visualise_example.py \\
        --dataset inverse-mlc --mode val --episode-type algebraic --idx 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataloader import ProgramDataset, TRAINING_MODES
from src.data.inverse_mlc_dataloader import (
    INVERSE_MLC_EPISODE_TYPES,
    InverseMLCDataset,
)


GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _build_dataset(args):
    if args.dataset == "program":
        assert args.corpus, "--corpus is required for --dataset program"
        # Accept both space-separated (``--corpus a.json b.json``) and
        # comma-separated (``--corpus a.json,b.json``) — the latter matches
        # the dataloader's own interactive prompt.
        flat: list[str] = []
        for chunk in args.corpus:
            flat.extend(p.strip() for p in chunk.split(",") if p.strip())
        ds = ProgramDataset(
            corpus_files=[Path(p) for p in flat],
            seed=args.seed,
            mode=args.mode,
            max_programs=args.max_programs,
        )
        if args.n_permuted is not None:
            ds.n_permuted = args.n_permuted
        return ds
    if args.dataset == "inverse-mlc":
        return InverseMLCDataset(mode=args.mode, episode_type=args.episode_type)
    raise ValueError(f"Unknown --dataset {args.dataset!r}")


def _split_seq(seq: list[int], loss_mask: list[int]) -> tuple[list[int], list[int]]:
    """``__getitem__`` returns ``seq = x + y`` and
    ``loss_mask = [0]*len(x) + [1]*(len(y)-1)``. Recover ``x`` and ``y`` from
    the mask so we don't reach into dataset internals."""
    n_x = loss_mask.count(0)
    return seq[:n_x], seq[n_x:]


def _detokenise_ids(dataset, ids: list[int]) -> list[str]:
    return [dataset.tokeniser.vocab.itos[t] for t in ids]


def _print_header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 80}{RESET}")
    print(f"{BOLD}{CYAN}{text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}")


def _print_section(title: str, body: str) -> None:
    print(f"\n{DIM}--- {title} ---{RESET}")
    print(body)


def _kv(label: str, value) -> None:
    print(f"{label}: {value}")


def _format_ids(ids: list[int]) -> str:
    return " ".join(str(t) for t in ids)


def _format_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def _print_one(dataset, idx: int) -> None:
    seq, loss_mask, info = dataset.__getitem__(idx, include_program=True)
    x_ids, y_ids = _split_seq(seq, loss_mask)

    _print_header(f"idx={idx}    (dataset size {len(dataset)})")

    # Per-dataset raw view ----------------------------------------------
    if isinstance(dataset, ProgramDataset):
        _kv("prog_idx", idx // dataset.n_io_views)
        _kv("n_io_shown", info["n_io_shown"])
        _kv("type / size", f"{info.get('type')} / {info.get('size')}")
        _kv("program", info["program"])
        if info.get("name_map"):
            print("name_map:")
            for orig, mapped in sorted(info["name_map"].items()):
                print(f"{orig} → {mapped}")
        print("io_pairs:")
        for inp, out in info["io_pairs"]:
            print(f"{inp} → {out}")
    elif isinstance(dataset, InverseMLCDataset):
        _kv("prog_idx", idx // dataset.n_io_views)
        _kv("q_idx", info["q_idx"])
        _kv("file", info["file"])
        _kv("xq", info["xq"])
        _kv("yq", info["yq"])
        print("grammar_str:")
        print(info["grammar_str"])

    # Encoder input -----------------------------------------------------
    _print_section(
        f"INPUT (encoder src) — token ids, len {len(x_ids)}",
        _format_ids(x_ids),
    )
    _print_section(
        "INPUT (encoder src) — detokenised",
        _format_tokens(_detokenise_ids(dataset, x_ids)),
    )

    # Decoder target ----------------------------------------------------
    _print_section(
        f"TARGET (decoder tgt) — token ids, len {len(y_ids)}",
        _format_ids(y_ids),
    )
    _print_section(
        "TARGET (decoder tgt) — detokenised",
        _format_tokens(_detokenise_ids(dataset, y_ids)),
    )

    # Combined view with loss mask --------------------------------------
    aligned_mask = [0] + list(loss_mask)  # align with ``seq`` (no prediction at first tok)
    coloured = [
        f"{GREEN}{tok}{RESET}" if m else tok
        for tok, m in zip(_detokenise_ids(dataset, seq), aligned_mask)
    ]
    _print_section(
        f"FULL SEQUENCE — green = loss-masked (predicted) target tokens, "
        f"len {len(seq)}, mask sum {sum(loss_mask)}",
        _format_tokens(coloured),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=("program", "inverse-mlc"), default="program",
    )
    parser.add_argument(
        "--mode", default="in-weight",
        help=(
            f"For --dataset program: one of {TRAINING_MODES}. "
            f"For --dataset inverse-mlc: 'train' or 'val'."
        ),
    )
    parser.add_argument(
        "--corpus", nargs="+", default=None,
        help="(program only) Corpus JSON file(s).",
    )
    parser.add_argument(
        "--episode-type", choices=INVERSE_MLC_EPISODE_TYPES, default="algebraic",
        help="(inverse-mlc only) Which underlying MLC dataset to build.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-programs", type=int, default=None,
        help="(program only) Cap on number of programs loaded.",
    )
    parser.add_argument(
        "--n-permuted", type=int, default=None,
        help="(program + easy-symbol-shuffling only) K, the number of grammar "
             "function names permuted per episode. None ≡ all (i.e. equivalent "
             "to symbol-shuffling).",
    )
    parser.add_argument(
        "--idx", nargs="*", type=int, default=None,
        help="One or more indices to print. If omitted, drops into an "
             "interactive prompt.",
    )
    args = parser.parse_args()

    dataset = _build_dataset(args)

    n_views = getattr(dataset, "n_io_views", 1)
    print(
        f"{YELLOW}Loaded {len(dataset.programs):,} programs / episodes "
        f"-> {len(dataset):,} items (n_io_views={n_views}){RESET}"
    )

    if args.idx is not None:
        for i in args.idx:
            if not 0 <= i < len(dataset):
                print(f"idx {i} out of range [0, {len(dataset) - 1}] — skipping")
                continue
            _print_one(dataset, i)
        return

    while True:
        try:
            raw = input("\nidx (blank or -1 to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw in ("", "-1"):
            break
        try:
            i = int(raw)
        except ValueError:
            print("not an integer")
            continue
        if not 0 <= i < len(dataset):
            print(f"out of range [0, {len(dataset) - 1}]")
            continue
        _print_one(dataset, i)


if __name__ == "__main__":
    main()
