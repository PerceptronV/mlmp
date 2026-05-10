"""Per-method on-disk cache for predictions and embeddings.

Keyed by ``(method.name, task_id, order, trial)``. Predictions land in a
parquet under ``cache/<method>/predictions.parquet``; embeddings in an npz at
``cache/<method>/embeddings.npz`` keyed by ``(task_id, order)`` (see §10).

Why a cache: CSV methods are basically free and we still cache for uniformity,
but the real motivation is that transformer eval is expensive (greedy decode
+ JIT compile + execute per trial). The cache keeps re-runs and downstream
analyses fast.

Cache key fingerprint is ``method.cache_fingerprint()`` — implemented per-method,
mismatch on disk → invalidate.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from .methods.base import Method, Prediction
    from .task import Trial

logger = logging.getLogger(__name__)

_BATCH_FLUSH = 256


@dataclass
class _MethodCache:
    method_name: str
    fingerprint: str
    pred_path: Path
    emb_path: Path
    predictions: dict[tuple[str, int, int], "Prediction"]
    embeddings: dict[tuple[str, int], np.ndarray]
    pending_pred: int = 0
    pending_emb: int = 0


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._caches: dict[str, _MethodCache] = {}

    def _ensure(self, method: "Method") -> _MethodCache:
        if method.name in self._caches:
            return self._caches[method.name]
        mdir = self.root / method.name
        mdir.mkdir(parents=True, exist_ok=True)
        pred_path = mdir / "predictions.parquet"
        emb_path = mdir / "embeddings.npz"
        fp_path = mdir / "fingerprint.json"
        fingerprint = _fingerprint_for(method)
        # Invalidate stale caches whose fingerprint doesn't match the method's
        # current configuration (different ckpt, different filters, etc.).
        if fp_path.exists():
            try:
                stored = json.loads(fp_path.read_text())
            except Exception:
                stored = {}
            if stored.get("fingerprint") != fingerprint:
                logger.info("[%s] cache fingerprint mismatch — invalidating", method.name)
                pred_path.unlink(missing_ok=True)
                emb_path.unlink(missing_ok=True)
        fp_path.write_text(json.dumps({"fingerprint": fingerprint}, indent=2))

        predictions: dict[tuple[str, int, int], "Prediction"] = {}
        if pred_path.exists():
            df = pd.read_parquet(pred_path)
            for row in df.itertuples(index=False):
                from .methods.base import Prediction  # local import to avoid cycle
                response = json.loads(row.response_json) if row.response_json else None
                effort = json.loads(row.effort_json) if row.effort_json else None
                predictions[(row.task_id, int(row.order), int(row.trial))] = Prediction(
                    response=response,
                    program=row.program if row.program else None,
                    correct=bool(row.correct),
                    effort=effort,
                )
        embeddings: dict[tuple[str, int], np.ndarray] = {}
        if emb_path.exists():
            arr = np.load(emb_path)
            for k in arr.files:
                # Keys are stored as "<task_id>__<order>"
                tid, order_s = k.split("__")
                embeddings[(tid, int(order_s))] = arr[k]

        cache = _MethodCache(
            method_name=method.name,
            fingerprint=fingerprint,
            pred_path=pred_path,
            emb_path=emb_path,
            predictions=predictions,
            embeddings=embeddings,
        )
        self._caches[method.name] = cache
        return cache

    def get_or_compute(
        self,
        method: "Method",
        trial: "Trial",
        fn: Callable[["Trial"], "Prediction"],
    ) -> "Prediction":
        c = self._ensure(method)
        key = (trial.task_id, trial.order, trial.trial)
        if key in c.predictions:
            return c.predictions[key]
        pred = fn(trial)
        c.predictions[key] = pred
        c.pending_pred += 1
        if c.pending_pred >= _BATCH_FLUSH:
            self._flush_predictions(c)
        return pred

    def get_or_compute_embedding(
        self,
        method: "Method",
        task_id: str,
        order: int,
        fn: Callable[[], np.ndarray],
    ) -> np.ndarray:
        c = self._ensure(method)
        key = (task_id, order)
        if key in c.embeddings:
            return c.embeddings[key]
        v = fn()
        c.embeddings[key] = v
        c.pending_emb += 1
        if c.pending_emb >= _BATCH_FLUSH:
            self._flush_embeddings(c)
        return v

    def flush(self) -> None:
        for c in self._caches.values():
            self._flush_predictions(c)
            self._flush_embeddings(c)

    def _flush_predictions(self, c: _MethodCache) -> None:
        if c.pending_pred == 0 and c.pred_path.exists():
            return
        if not c.predictions:
            return
        rows = []
        for (tid, order, trial), p in c.predictions.items():
            rows.append({
                "task_id": tid,
                "order": order,
                "trial": trial,
                "response_json": json.dumps(p.response) if p.response is not None else "",
                "program": p.program or "",
                "correct": bool(p.correct),
                "effort_json": json.dumps(p.effort) if p.effort is not None else "",
            })
        df = pd.DataFrame(rows)
        df.to_parquet(c.pred_path, index=False)
        c.pending_pred = 0

    def _flush_embeddings(self, c: _MethodCache) -> None:
        if c.pending_emb == 0 and c.emb_path.exists():
            return
        if not c.embeddings:
            return
        np.savez(
            c.emb_path,
            **{f"{tid}__{order}": v for (tid, order), v in c.embeddings.items()},
        )
        c.pending_emb = 0


def _fingerprint_for(method) -> str:
    """Best-effort cache fingerprint. Methods may override
    ``method.cache_fingerprint()`` to include eg. ckpt mtime.
    """
    if hasattr(method, "cache_fingerprint"):
        return str(method.cache_fingerprint())
    parts = [method.name, method.__class__.__name__]
    if hasattr(method, "filters"):
        parts.append(json.dumps(getattr(method, "filters", {}), sort_keys=True))
    if hasattr(method, "csv_filename"):
        parts.append(method.csv_filename)
    return "::".join(parts)
