#!/usr/bin/env python3
"""Build a Rule-et-al validation split with semantic dedupe.

Workflow:
  1. Read programs from ``src/data/rule/functions.txt`` (one per line).
  2. Compute each program's fingerprint on ``DEFAULT_TEST_SUITE``; drop
     programs that fail to compile, evaluate to FAIL on every input, or
     don't return ``list[int]`` (matching ``ProgramDataset``'s default
     type filter and ``RuleIOSampler``'s I/O contract).
  3. Write ``--val-out`` in corpus format
     ``[{"program": str, "type": "list[int]", "size": int}]``.
  4. For each ``--train-corpus``, fingerprint every program in parallel,
     drop any whose fingerprint matches the Rule set, and write the
     survivors to a sibling ``<stem>_no_rule.json``.

Run:
    python scripts/build_rule_split.py
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from src.lang.ast_nodes import LambdaNode
from src.lang.compiler import JITCompiler
from src.lang.enumeration.fingerprint import FAIL, Fingerprint, compute_fingerprint
from src.lang.enumeration.test_suite import DEFAULT_TEST_SUITE
from src.lang.grammar import DefaultGrammar
from src.lang.parser import parse
from src.lang.utils import program_size


def _is_list_int_fingerprint(fp: Fingerprint) -> bool:
    """True iff every non-FAIL result is a tuple of plain ints (i.e., list[int])."""
    saw_any = False
    for v in fp.values:
        if v is FAIL:
            continue
        saw_any = True
        if not isinstance(v, tuple):
            return False
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return False
    return saw_any


# Per-worker JIT (avoids pickling and re-creating the compiler per call).
_JIT: JITCompiler | None = None


def _init_worker():
    global _JIT
    _JIT = JITCompiler(DefaultGrammar)


def _fingerprint_program(program_str: str) -> Fingerprint | None:
    try:
        ast = parse(program_str)
    except Exception:
        return None
    return compute_fingerprint(ast, DEFAULT_TEST_SUITE, _JIT)


def fingerprint_rule_programs(rule_path: Path) -> tuple[set[Fingerprint], list[dict]]:
    seen: set[str] = set()
    programs: list[str] = []
    with open(rule_path) as f:
        for line in f:
            s = line.strip()
            if s and s not in seen:
                seen.add(s)
                programs.append(s)
    print(f"Rule canonical programs: {len(programs)} unique")

    jit = JITCompiler(DefaultGrammar)
    rule_fps: set[Fingerprint] = set()
    val_entries: list[dict] = []
    bad_compile = bad_type = all_fail = 0

    for s in programs:
        try:
            ast = parse(s)
        except Exception:
            bad_compile += 1
            continue
        if not isinstance(ast, LambdaNode):
            bad_type += 1
            continue
        fp = compute_fingerprint(ast, DEFAULT_TEST_SUITE, jit)
        if fp is None:
            bad_compile += 1
            continue
        if all(v is FAIL for v in fp.values):
            all_fail += 1
            continue
        if not _is_list_int_fingerprint(fp):
            bad_type += 1
            continue
        rule_fps.add(fp)
        val_entries.append({
            "program": s,
            "type": "list[int]",
            "size": program_size(ast),
        })

    print(f"  -> {len(val_entries)} val entries; {len(rule_fps)} unique fingerprints")
    print(f"  skipped: {bad_compile} compile, {bad_type} non-list[int], {all_fail} all-FAIL")
    return rule_fps, val_entries


def filter_corpus(
    corpus_path: Path,
    rule_fps: set[Fingerprint],
    output_path: Path,
    workers: int,
) -> None:
    print(f"\nFiltering {corpus_path}")
    with open(corpus_path) as f:
        entries = json.load(f)
    print(f"  loaded {len(entries):,} programs")

    programs = [e["program"] for e in entries]

    if workers <= 1:
        _init_worker()
        fps = [_fingerprint_program(p) for p in tqdm(programs, desc="fingerprint")]
    else:
        with Pool(workers, initializer=_init_worker) as pool:
            fps = list(tqdm(
                pool.imap(_fingerprint_program, programs, chunksize=512),
                total=len(programs),
                desc=f"fingerprint ({workers} workers)",
            ))

    survivors = [e for e, fp in zip(entries, fps) if fp is not None and fp not in rule_fps]
    n_drop_match = sum(1 for fp in fps if fp is not None and fp in rule_fps)
    n_drop_compile = sum(1 for fp in fps if fp is None)
    print(f"  dropped {n_drop_match:,} Rule-matching, {n_drop_compile:,} compile-fail")
    print(f"  survivors: {len(survivors):,} ({len(survivors)/len(entries):.2%})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(survivors, f)
    print(f"  wrote -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-file", type=Path, default=Path("src/data/rule/functions.txt"))
    parser.add_argument("--val-out", type=Path, default=Path("datasets/rule_val.json"))
    parser.add_argument(
        "--train-corpus", type=Path, action="append", default=None,
        help="Train corpus(es) to filter (repeat). "
             "Default: rl_corpus.json + enum_corpus.json under datasets/corpus-a/",
    )
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    if args.train_corpus is None:
        args.train_corpus = [
            Path("datasets/corpus-a/rl_corpus.json"),
            Path("datasets/corpus-a/enum_corpus.json"),
        ]

    rule_fps, val_entries = fingerprint_rule_programs(args.rule_file)

    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.val_out, "w") as f:
        json.dump(val_entries, f, indent=2)
    print(f"Wrote {len(val_entries)} val entries -> {args.val_out}")

    for corpus_path in args.train_corpus:
        out_path = corpus_path.with_name(corpus_path.stem + "_no_rule.json")
        filter_corpus(corpus_path, rule_fps, out_path, args.workers)


if __name__ == "__main__":
    main()
