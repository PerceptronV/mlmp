"""Stage orchestration for the primitive subset search.

Stage 0: enumerate the full pool once, proxy-score every subset combinatorially.
Stage 1: exact deep enumeration of the top-N subsets, in worker processes with
a per-subset wall-clock timeout (SIGALRM), results persisted incrementally.
"""

import json
import os
import signal
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

from math import comb

from ..enumeration.enumerator import BottomUpEnumerator
from .pool import POOL_NAMES, pool_grammar, subset_key
from .support import support_buckets, proxy_scores
from .scoring import score_vector

# Above this many subsets of one size we record only those with a nonzero
# proxy score. Zero-score subsets can never reach Stage 1 (which takes the
# top N), and materialising e.g. all C(57,6) = 36M of them is neither useful
# nor writable.
ZERO_FILL_LIMIT = 2_000_000


class SubsetTimeout(BaseException):
    """Raised inside a worker when the per-subset wall clock expires.

    Subclasses BaseException (like KeyboardInterrupt) so it escapes the
    broad ``except Exception`` handlers in the enumeration/fingerprint
    evaluation loops — otherwise a hung evaluation would swallow the
    alarm and the timeout would never fire.
    """


def _on_alarm(signum, frame):
    raise SubsetTimeout()


def run_stage0(out_dir, max_size=7, sizes=(5, 6), pool_names=None,
               target_type=None):
    """One full-pool enumeration; proxy-score every candidate subset.

    ``target_type`` (e.g. 'list[int]') restricts the search to behaviors of
    that output type. It is recorded in stage0.json and inherited by the
    later stages, so one output directory is always internally consistent.

    ``pool_names`` defaults to the curated :data:`POOL_NAMES`; pass
    :data:`pool.ALL_NAMES` (CLI ``--pool all``) to search every DefaultGrammar
    primitive. Note the proxy gets looser as the pool grows: it counts a
    behavior for a subset only if the *full-pool* enumeration's witness for
    that behavior happens to fit inside the subset, and a richer pool means
    more behaviors are first discovered via a witness using primitives outside
    the subset. Stage 1 (exact per-subset enumeration) is the correction.
    """
    os.makedirs(out_dir, exist_ok=True)
    names = tuple(pool_names) if pool_names is not None else POOL_NAMES

    enum = BottomUpEnumerator(grammar=pool_grammar(names), max_size=max_size)
    bank = enum.enumerate()
    buckets = support_buckets(bank, frozenset(names), target_type=target_type)
    # A support larger than the biggest subset can never be contained in one.
    buckets = {s: n for s, n in buckets.items() if len(s) <= max(sizes)}

    scores: list[tuple[str, int]] = []
    for k in sizes:
        reached = proxy_scores(buckets, names, k)
        total = comb(len(names), k)
        if total <= ZERO_FILL_LIMIT:
            for combo in combinations(sorted(names), k):
                reached.setdefault(combo, 0)
        elif len(reached) < total:
            print(f"stage0: {total:,} subsets of size {k}; recording the "
                  f"{len(reached):,} with a nonzero proxy score")
        scores.extend((subset_key(frozenset(s)), n) for s, n in reached.items())
    scores.sort(key=lambda kv: (-kv[1], kv[0]))
    with open(os.path.join(out_dir, 'stage0.json'), 'w') as f:
        json.dump({'max_size': max_size, 'pool': list(names),
                   'target_type': target_type, 'scores': scores}, f)
    return scores


def _stage1_worker(task):
    """Deep-enumerate one subset under a SIGALRM wall-clock guard."""
    key, max_size, timeout_s, target_type = task
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_s)
    try:
        grammar = pool_grammar(tuple(key.split(' ')))
        enum = BottomUpEnumerator(grammar=grammar, max_size=max_size)
        bank = enum.enumerate()
        vec = score_vector(bank, enum.attempts, max_size,
                           target_type=target_type)
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

    target_type = stage0.get('target_type')
    todo = [
        (key, max_size, timeout_s, target_type)
        for key, _score in stage0['scores'][:top_n]
        if key not in results
    ]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_stage1_worker, task): task[0] for task in todo
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    key, res = fut.result()
                except (Exception, SubsetTimeout) as e:
                    # Parent-side failure (e.g. BrokenProcessPool, escaped
                    # alarm): an infrastructure problem, not a per-subset
                    # verdict. Leave the key unwritten so a resume retries it.
                    print(f"stage1: {key} failed in parent, will retry on resume: {e!r}")
                    continue
                results[key] = res
                with open(results_path + '.tmp', 'w') as f:
                    json.dump(results, f)
                os.replace(results_path + '.tmp', results_path)

    return {
        key: results[key]
        for key, _score in stage0['scores'][:top_n]
        if key in results
    }
