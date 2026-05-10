"""MPLMethod / MPLBestMethod adapters for Rule's MPL CSVs.

Also exposes the meta-primitive vocabulary and parsing helpers used by the
probing analysis (see ``.claude/plans/probing.md``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import ClassVar

import numpy as np

from .csv_method import CSVMethod

logger = logging.getLogger(__name__)


# Paper meta-primitive → logged token(s). Values are tuples so we can re-map
# if Rule's logging conventions change. Excluded from labels (search-control
# tokens, not meta-primitives): SampleAtom, SampleRule, Stop, RegenerateRule,
# RegenerateThisRule, RegenerateThisPart, RegenerateThisPlace, Generalize.
META_PRIMITIVE_VOCAB: dict[str, tuple[str, ...]] = {
    "MemorizeAll": ("MemorizeAll",),
    "Memorize": ("MemorizeDatum",),
    "AntiUnify": ("AntiUnify",),
    "Recurse": ("Recurse",),
    "Variable": ("Variablize",),
    "Compose": ("Compose",),
    "Delete": ("DeleteRule",),
}


_TOKEN_PREFIX = re.compile(r"^([A-Za-z_]+)")


def _parse_metaprogram(s: str) -> list[str]:
    """Split a metaprogram string on ``.`` and extract the alphabetic prefix
    of each part. ``"MemorizeAll.SampleRule.SampleAtom(Some(...))"`` →
    ``["MemorizeAll", "SampleRule", "SampleAtom"]``.
    """
    if not s:
        return []
    out: list[str] = []
    # The metaprogram syntax has nested ``Token(arg.arg)`` parts where arg may
    # itself contain dots. Split on top-level dots only.
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == "." and depth == 0:
            piece = "".join(buf)
            buf = []
            m = _TOKEN_PREFIX.match(piece.strip())
            if m:
                out.append(m.group(1))
        else:
            buf.append(ch)
    if buf:
        piece = "".join(buf)
        m = _TOKEN_PREFIX.match(piece.strip())
        if m:
            out.append(m.group(1))
    return out


def _multi_hot(tokens: list[str], vocab: dict[str, tuple[str, ...]]) -> np.ndarray:
    """Map a token list → a multi-hot label vector ordered by ``vocab.keys()``."""
    present = set(tokens)
    return np.array(
        [int(any(tok in present for tok in vocab[primitive])) for primitive in vocab],
        dtype=np.int8,
    )


class _MPLLabelMixin:
    """Shared metaprogram-extraction helpers for MPLMethod / MPLBestMethod.

    Both back the same column schema for the bits we care about — the
    ``metaprogram`` column is a parseable dot-separated sequence of tokens —
    so the helpers live here rather than being duplicated.
    """

    # _buckets is set up by CSVMethod.__post_init__ on subclasses.
    _buckets: dict[tuple, list[dict]]

    def acquired_tasks(
        self,
        rule_acq_dir: Path | None = None,
        *,
        method_name: str = "mpl",
    ) -> set[str]:
        """Task IDs where ≥1 (function, order) row in
        ``rule_acquisition/acquired.parquet`` (filtered to ``method == method_name``)
        has ``acquired_on ≤ 11``. Reads the parquet rather than recomputing so
        the strict-acquisition criterion lives in one place
        (``RuleAcquisitionAnalysis``, main plan §7.1).
        """
        import pandas as pd  # type: ignore[import-untyped]

        if rule_acq_dir is None:
            raise ValueError(
                "acquired_tasks: rule_acq_dir is required. Run rule_acquisition "
                "first; ProbingAnalysis derives the path from its output dir."
            )
        path = Path(rule_acq_dir) / "acquired.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run a rule_acquisition analysis with "
                f"a method named {method_name!r} before probing."
            )
        df = pd.read_parquet(path)
        sub = df[df["method"] == method_name]
        if sub.empty:
            raise ValueError(
                f"acquired.parquet has no rows for method={method_name!r}; "
                f"available: {sorted(df['method'].unique().tolist())}"
            )
        acquired = (
            sub[sub["acquired_on"] <= 11]["function"].astype(str).unique().tolist()
        )
        return set(acquired)

    def metaprimitives_for(
        self,
        task_id: str,
        *,
        trial: int = 11,
        vocab: dict[str, tuple[str, ...]] | None = None,
    ) -> np.ndarray:
        """Multi-hot label matrix of shape ``(n_replicates, len(vocab))`` for
        the given task. Each row is one ``(run, order)`` replicate's best
        metaprogram at the given trial (best = max ``lposterior`` per cell);
        entry ``[r, p] = 1`` iff any token in ``vocab[p]`` appears in that
        metaprogram. Returns ``(0, len(vocab))`` if no rows exist.
        """
        v = vocab if vocab is not None else META_PRIMITIVE_VOCAB
        n_p = len(v)
        rows_out: list[np.ndarray] = []
        # CSVMethod buckets are keyed (task_id, order, trial); each bucket
        # contains rows for all runs and possibly multiple chain steps. Pick
        # the row with max lposterior per (run, order), parse its metaprogram.
        for (tid, order, tr), rows in self._buckets.items():
            if tid != task_id or int(tr) != int(trial):
                continue
            by_run: dict[str, dict] = {}
            for r in rows:
                run = str(r.get("run", "1"))
                lp = _safe_float(r.get("lposterior"))
                cur = by_run.get(run)
                if cur is None or _safe_float(cur.get("lposterior")) < lp:
                    by_run[run] = r
            for r in by_run.values():
                tokens = _parse_metaprogram(r.get("metaprogram", ""))
                rows_out.append(_multi_hot(tokens, v))
        if not rows_out:
            return np.zeros((0, n_p), dtype=np.int8)
        return np.stack(rows_out, axis=0)


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("-inf")


class MPLMethod(_MPLLabelMixin, CSVMethod):
    csv_filename: ClassVar[str] = "mpl.csv"
    response_col: ClassVar[str] = "response"
    program_col: ClassVar[str] = "program"
    correct_col: ClassVar[str] = "accuracy"
    effort_cols: ClassVar[tuple[str, ...]] = ("time", "count", "lposterior")


class MPLBestMethod(_MPLLabelMixin, CSVMethod):
    """``mpl_best.csv`` doesn't carry response / accuracy — only the highest-
    posterior program per cell. We treat ``program`` as the prediction, leave
    ``response`` empty, and let consumers compile-and-execute when they need to
    score it. This is enough for clustering / failure modes that only need
    accuracy markers, **provided** the caller supplies a ``response`` column or
    routes through MPLMethod for prediction-level work.
    """

    csv_filename: ClassVar[str] = "mpl_best.csv"
    response_col: ClassVar[str] = "output"  # the gold output, since there is no model response
    program_col: ClassVar[str] = "program"
    correct_col: ClassVar[str | None] = None  # not present in mpl_best
    effort_cols: ClassVar[tuple[str, ...]] = ("time", "count", "lposterior")
