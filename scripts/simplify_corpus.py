#!/usr/bin/env python3
"""Apply equality-saturation simplification to a corpus JSON.

Usage:
    python scripts/simplify_corpus.py \\
        --input  datasets/corpus-a/rl_corpus_no_rule.json \\
        --output datasets/corpus-a/rl_corpus_no_rule.simplified.json

For every record ``{"program", "type", "size"}`` in the input, the
program is parsed, simplified via :func:`src.lang.simplify.simplify`,
and the simplified S-expression is written out alongside the originals
plus per-record ``original_size`` / ``simplified_size`` /
``reduction_ratio`` fields. Records whose simplification fails the
Φ-preservation check are kept *unchanged*; the failure rate is logged
and the run aborts if it exceeds ``--max-phi-fail-rate``.

Records that fail to parse are also kept unchanged (the corpus is
trusted to be parseable, but we don't want a single bad string to kill
a multi-million-program job).

The simplifier validates its rule set once at import; per-worker
processes share the validated rule list because the validation is
deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from src.lang.parser import parse
from src.lang.simplify import SimplifyError, simplify
from src.lang.utils import program_size


# Per-worker import: importing the simplify module triggers rule
# validation. We gate it behind an init function so each worker pays
# the cost once.
def _init_worker() -> None:
    # Suppress the validator's "rules rejected" warning in worker
    # subprocesses; the main process already printed it on import.
    warnings.filterwarnings("ignore", category=UserWarning, module="src.lang.simplify")
    # Force the lazy imports to happen now.
    import src.lang.simplify  # noqa: F401


def _simplify_one(program_str: str) -> tuple[str | None, str, int, int, str | None]:
    """Return (simplified_str_or_None, original_str, orig_size, new_size, error).

    ``simplified_str_or_None`` is the simplified program if simplification
    succeeded and produced something; ``None`` means we should keep the
    original. ``error`` is None on success, otherwise a short tag.
    """
    try:
        ast = parse(program_str)
    except Exception:
        return None, program_str, 0, 0, "parse_fail"

    orig_size = program_size(ast)

    try:
        out = simplify(ast)
    except SimplifyError:
        return None, program_str, orig_size, orig_size, "phi_fail"
    except Exception as e:  # defensive: any internal error -> keep original
        return None, program_str, orig_size, orig_size, f"internal:{type(e).__name__}"

    new_size = program_size(out)
    if new_size >= orig_size:
        return None, program_str, orig_size, orig_size, None
    return str(out), program_str, orig_size, new_size, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/corpus-a/rl_corpus_no_rule.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <input_stem>.simplified.json",
    )
    p.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N records (for dry-run / sanity check).",
    )
    p.add_argument(
        "--max-phi-fail-rate", type=float, default=0.001,
        help="Abort if more than this fraction of records fail Φ-preservation.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Don't write output; print stats only.",
    )
    args = p.parse_args()

    if args.output is None:
        args.output = args.input.with_name(args.input.stem + ".simplified.json")

    print(f"Loading {args.input} ...", flush=True)
    with open(args.input) as f:
        entries = json.load(f)
    print(f"  {len(entries):,} records", flush=True)
    if args.limit:
        entries = entries[: args.limit]
        print(f"  (limited to {len(entries):,} records)", flush=True)

    programs = [e["program"] for e in entries]

    t0 = time.monotonic()
    if args.workers <= 1:
        _init_worker()
        results = [
            _simplify_one(s)
            for s in tqdm(programs, desc="simplify (1 worker)")
        ]
    else:
        with Pool(args.workers, initializer=_init_worker) as pool:
            results = list(tqdm(
                pool.imap(_simplify_one, programs, chunksize=512),
                total=len(programs),
                desc=f"simplify ({args.workers} workers)",
            ))
    elapsed = time.monotonic() - t0

    # Stats.
    n_total = len(results)
    n_simplified = sum(1 for r in results if r[0] is not None)
    n_unchanged = n_total - n_simplified
    n_phi_fail = sum(1 for r in results if r[4] == "phi_fail")
    n_parse_fail = sum(1 for r in results if r[4] == "parse_fail")
    n_internal = sum(
        1 for r in results
        if r[4] is not None and r[4].startswith("internal:")
    )

    sum_orig = sum(r[2] for r in results if r[4] is None or r[4] == "phi_fail")
    sum_new = sum(r[3] for r in results if r[4] is None or r[4] == "phi_fail")
    mean_ratio = (sum_orig / sum_new) if sum_new > 0 else 1.0

    print()
    print("=" * 60)
    print(f"  total              : {n_total:,}")
    print(f"  simplified         : {n_simplified:,} ({n_simplified/n_total:.1%})")
    print(f"  unchanged          : {n_unchanged:,}")
    print(f"  phi-mismatch       : {n_phi_fail} ({n_phi_fail/n_total:.4%})")
    print(f"  parse-fail         : {n_parse_fail}")
    print(f"  internal-error     : {n_internal}")
    print(f"  total size in -> out: {sum_orig:,} -> {sum_new:,}")
    if sum_orig:
        print(f"  size reduction     : {(1 - sum_new/sum_orig):.2%}")
    print(f"  elapsed            : {elapsed:.1f}s ({elapsed/n_total*1000:.2f} ms/record)")
    print("=" * 60)

    if n_phi_fail / max(1, n_total) > args.max_phi_fail_rate:
        print(
            f"\nERROR: Φ-mismatch rate {n_phi_fail/n_total:.4%} exceeds "
            f"--max-phi-fail-rate ({args.max_phi_fail_rate:.4%}); aborting "
            f"without writing output. The default rule set may need an "
            f"unsound rule removed.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.dry_run:
        print("\n(dry-run: not writing output)")
        return

    print(f"\nWriting {args.output} ...", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_entries: list[dict] = []
    for entry, (simp_str, orig_str, orig_size, new_size, _) in zip(entries, results):
        record = dict(entry)
        record["original_program"] = orig_str
        record["original_size"] = orig_size
        if simp_str is not None:
            record["program"] = simp_str
            record["size"] = new_size
            record["reduction_ratio"] = orig_size / max(1, new_size)
        else:
            record["reduction_ratio"] = 1.0
        out_entries.append(record)

    with open(args.output, "w") as f:
        json.dump(out_entries, f)
    print(f"  wrote {len(out_entries):,} records -> {args.output}")


if __name__ == "__main__":
    main()
