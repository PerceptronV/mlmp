"""Stage orchestration for the primitive subset search.

Stage 0: enumerate the full pool once, proxy-score every subset combinatorially.
Stage 1: exact deep enumeration of the top-N subsets, in worker processes with
a per-subset wall-clock timeout (SIGALRM), results persisted incrementally.
"""

import json
import os
import signal
from concurrent.futures import ProcessPoolExecutor

from ..enumeration.enumerator import BottomUpEnumerator
from .pool import POOL_NAMES, pool_grammar, iter_subsets, subset_key
from .support import support_buckets, proxy_score
from .scoring import score_vector


class SubsetTimeout(Exception):
    """Raised inside a worker when the per-subset wall clock expires."""


def _on_alarm(signum, frame):
    raise SubsetTimeout()


def run_stage0(out_dir, max_size=7, sizes=(5, 6), pool_names=None):
    """One full-pool enumeration; proxy-score every candidate subset."""
    os.makedirs(out_dir, exist_ok=True)
    names = tuple(pool_names) if pool_names is not None else POOL_NAMES

    enum = BottomUpEnumerator(grammar=pool_grammar(names), max_size=max_size)
    bank = enum.enumerate()
    buckets = support_buckets(bank, frozenset(names))
    # A support larger than the biggest subset can never be contained in one.
    buckets = {s: n for s, n in buckets.items() if len(s) <= max(sizes)}

    scores = sorted(
        (
            (subset_key(s), proxy_score(s, buckets))
            for s in iter_subsets(sizes=sizes, pool_names=names)
        ),
        key=lambda kv: (-kv[1], kv[0]),
    )
    with open(os.path.join(out_dir, 'stage0.json'), 'w') as f:
        json.dump({'max_size': max_size, 'pool': list(names), 'scores': scores}, f)
    return scores


def _stage1_worker(task):
    """Deep-enumerate one subset under a SIGALRM wall-clock guard."""
    key, max_size, timeout_s = task
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_s)
    try:
        grammar = pool_grammar(tuple(key.split(' ')))
        enum = BottomUpEnumerator(grammar=grammar, max_size=max_size)
        bank = enum.enumerate()
        vec = score_vector(bank, enum.attempts, max_size)
        return key, {'status': 'ok', 'score': vec, 'max_size': max_size}
    except SubsetTimeout:
        return key, {'status': 'timeout', 'max_size': max_size}
    except Exception as e:  # never let one subset kill the sweep
        return key, {'status': 'error', 'error': repr(e), 'max_size': max_size}
    finally:
        signal.alarm(0)


def run_stage1(out_dir, top_n=200, max_size=10, timeout_s=600, workers=None):
    """Exact deep evaluation of the Stage 0 top-N. Incremental + resumable."""
    with open(os.path.join(out_dir, 'stage0.json')) as f:
        stage0 = json.load(f)

    results_path = os.path.join(out_dir, 'stage1.json')
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    todo = [
        (key, max_size, timeout_s)
        for key, _score in stage0['scores'][:top_n]
        if key not in results
    ]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for key, res in ex.map(_stage1_worker, todo):
                results[key] = res
                with open(results_path, 'w') as f:
                    json.dump(results, f)

    return {
        key: results[key]
        for key, _score in stage0['scores'][:top_n]
        if key in results
    }
