"""Stage 2: Pareto selection and qualitative markdown reports."""

import json
import os
from collections import defaultdict

from ..enumeration.enumerator import BottomUpEnumerator
from ..enumeration.filters import passes_quality_filter
from ..enumeration.fingerprint import FAIL
from ..enumeration.test_suite import DEFAULT_TEST_SUITE
from .pool import pool_grammar
from .scoring import PARETO_DIMS, PARETO_DIMS_TARGETED


def pareto_front(
    vectors: dict[str, dict],
    dims: tuple[str, ...] = PARETO_DIMS,
) -> list[str]:
    """Keys not strictly dominated on ``dims``, by descending n_distinct."""
    def dominates(a: dict, b: dict) -> bool:
        return (
            all(a[d] >= b[d] for d in dims)
            and any(a[d] > b[d] for d in dims)
        )

    front = [
        k for k in vectors
        if not any(dominates(vectors[o], vectors[k]) for o in vectors if o != k)
    ]
    return sorted(front, key=lambda k: -vectors[k]['n_distinct'])


def _format_value(v) -> str:
    if v is FAIL:
        return "FAIL"
    if isinstance(v, tuple):
        return "[" + ", ".join(_format_value(x) for x in v) + "]"
    return str(v)


def _finalist_report(
    key: str, vec: dict, max_size: int,
    samples_per_size: int, n_hardest: int,
    dims: tuple[str, ...] = PARETO_DIMS,
    target_type: str | None = None,
) -> str:
    enum = BottomUpEnumerator(grammar=pool_grammar(tuple(key.split(' '))),
                              max_size=max_size)
    bank = enum.enumerate()
    quality = [
        p for p in bank.all_programs()
        if p.fingerprint is not None and passes_quality_filter(p.fingerprint)
        and (target_type is None or str(p.type) == target_type)
    ]

    lines = [f"# Subset: `{key}`", ""]
    if target_type is not None:
        lines += [f"Restricted to output type `{target_type}`.", ""]

    lines += ["## Score vector", ""]
    for dim in dims:
        lines.append(f"- **{dim}**: {vec[dim]}")
    lines.append("")

    lines += ["## Yield curve", "", "| size k | N(k) |", "|---|---|"]
    for k, n in enumerate(vec['curve'], start=1):
        lines.append(f"| {k} | {n} |")
    lines.append("")

    lines += ["## Sample programs", ""]
    by_size = defaultdict(list)
    for p in quality:
        by_size[p.size].append(p)
    for size in sorted(by_size):
        lines.append(f"### Size {size}")
        lines.append("")
        for p in by_size[size][:samples_per_size]:
            lines.append(f"- `{p.ast.pretty_print(0, True)}` : `{p.type}`")
        lines.append("")

    lines += ["## Hardest behaviors", "",
              "Behaviors with the largest minimal program size.", ""]
    hardest = sorted(quality, key=lambda p: -p.size)[:n_hardest]
    for p in hardest:
        lines.append(f"### `{p.ast.pretty_print(0, True)}` (size {p.size})")
        lines.append("")
        n_shown = min(len(DEFAULT_TEST_SUITE), len(p.fingerprint.values))
        for inp, out in zip(DEFAULT_TEST_SUITE[:n_shown], p.fingerprint.values):
            lines.append(f"- `{inp}` → `{_format_value(out)}`")
        lines.append("")

    return "\n".join(lines)


def write_reports(
    out_dir: str,
    max_finalists: int = 10,
    samples_per_size: int = 5,
    n_hardest: int = 10,
) -> list[str]:
    with open(os.path.join(out_dir, 'stage1.json')) as f:
        stage1 = json.load(f)
    with open(os.path.join(out_dir, 'stage0.json')) as f:
        target_type = json.load(f).get('target_type')
    dims = PARETO_DIMS if target_type is None else PARETO_DIMS_TARGETED

    ok = {k: v for k, v in stage1.items() if v['status'] == 'ok'}
    vectors = {k: v['score'] for k, v in ok.items()}
    finalists = pareto_front(vectors, dims)[:max_finalists]

    reports_dir = os.path.join(out_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    title = "# Subset search — Pareto front"
    if target_type is not None:
        title += f" (output type `{target_type}`)"
    summary = [title, "",
               "| subset | " + " | ".join(dims) + " |",
               "|" + "---|" * (len(dims) + 1)]
    for key in finalists:
        vec = vectors[key]
        row = " | ".join(
            f"{vec[d]:.4g}" if isinstance(vec[d], float) else str(vec[d])
            for d in dims
        )
        summary.append(f"| `{key}` | {row} |")
        report = _finalist_report(
            key, vec, ok[key]['max_size'], samples_per_size, n_hardest,
            dims=dims, target_type=target_type,
        )
        path = os.path.join(reports_dir, f"{key.replace(' ', '_')}.md")
        with open(path, 'w') as f:
            f.write(report)
    summary.append("")
    summary.append(f"Full Stage 1 results: {len(vectors)} subsets evaluated; "
                   f"{len(finalists)} on the (capped) Pareto front.")

    with open(os.path.join(reports_dir, 'summary.md'), 'w') as f:
        f.write("\n".join(summary))
    return finalists
