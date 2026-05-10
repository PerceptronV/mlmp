"""Sample I/O pairs for ``list[int] -> list[int]`` programs.

Methodology (after Rule's meta-program learner): draw a pool of candidate
inputs, execute the program on each, and select a diverse subset by
length-bucketed stratified sampling preferring unique outputs and non-identity
pairs.

All output integers are reduced mod ``mod`` (default 100) so I/O pairs match
the project-wide modular-int convention used downstream by the tokeniser.
"""

from __future__ import annotations

import random
import signal
from contextlib import contextmanager
from typing import Callable

from ..lang.compiler import JITCompiler
from ..lang.grammar import Grammar, DefaultGrammar
from ..lang.parser import parse


@contextmanager
def _alarm(seconds: float):
    """SIGALRM-bounded context. Interrupts a runaway pure-Python call after
    ``seconds``. POSIX, main-thread of the (possibly forked DataLoader worker)
    process only — that holds for our use. Same shape as the helper in
    ``train.py``; duplicated here to avoid a cross-package import."""
    def _handler(_signum, _frame):
        raise TimeoutError(f"call exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class RuleIOSampler:
    """Sample I/O pairs for a unary ``list[int] -> list[int]`` program."""

    def __init__(
        self,
        grammar: Grammar = DefaultGrammar,
        num_io_pairs: int = 11,
        num_candidates: int = 100,
        min_list_len: int = 0,
        max_list_len: int = 15,
        min_elem: int = 0,
        max_elem: int = 100,
        mod: int = 100,
        exec_timeout: float = 0.05,
    ):
        self.jit = JITCompiler(grammar)
        self.num_io_pairs = num_io_pairs
        self.num_candidates = num_candidates
        self.min_list_len = min_list_len
        self.max_list_len = max_list_len
        self.min_elem = min_elem
        self.max_elem = max_elem
        self.mod = mod
        self.exec_timeout = exec_timeout
        self._fn_cache: dict[str, Callable | None] = {}

    def _compile(self, program_str: str) -> Callable | None:
        if program_str not in self._fn_cache:
            try:
                fn, _ = self.jit.compile(parse(program_str))
                self._fn_cache[program_str] = fn
            except Exception:
                self._fn_cache[program_str] = None
        return self._fn_cache[program_str]

    def sample(self, program_str: str, rng: random.Random) -> list[tuple[list[int], list[int]]]:
        """Return up to ``num_io_pairs`` I/O pairs for ``program_str``.

        ``rng`` controls input sampling so a given (program, RNG state) is reproducible.
        Returns fewer than ``num_io_pairs`` pairs (possibly zero) if the program fails
        on too many candidates.
        """
        fn = self._compile(program_str)
        if fn is None:
            return []

        pairs: list[tuple[list[int], list[int]]] = []
        for _ in range(self.num_candidates):
            length = rng.randint(self.min_list_len, self.max_list_len)
            inp = [rng.randint(self.min_elem, self.max_elem) for _ in range(length)]
            try:
                # Bound each execution: a non-terminating program would otherwise
                # hang a DataLoader worker, blocking the main process indefinitely
                # on the next batch fetch (observed as 0% GPU util + non-moving
                # tqdm). TimeoutError is caught alongside other exceptions.
                with _alarm(self.exec_timeout):
                    out = fn(inp)
            except Exception:
                continue
            if not isinstance(out, list) or len(out) > self.max_list_len:
                continue
            if not all(isinstance(x, int) for x in out):
                continue
            pairs.append((inp, [x % self.mod for x in out]))

        return self._select(pairs, self.num_io_pairs, rng)

    def _select(
        self,
        pairs: list[tuple[list[int], list[int]]],
        n: int,
        rng: random.Random,
    ) -> list[tuple[list[int], list[int]]]:
        """Stratify by input-length quartile, prefer unique non-identity outputs."""
        if len(pairs) <= n:
            return pairs

        bucket_size = max(1, (self.max_list_len + 1) // 4)
        buckets: list[list[tuple[list[int], list[int]]]] = [[] for _ in range(4)]
        for p in pairs:
            buckets[min(len(p[0]) // bucket_size, 3)].append(p)

        seen: set[tuple[int, ...]] = set()
        def priority(p):
            inp, out = p
            return (int(tuple(out) not in seen), int(inp != out), rng.random())

        per_bucket, remainder = divmod(n, 4)
        selected: list[tuple[list[int], list[int]]] = []
        for i, bucket in enumerate(buckets):
            target = per_bucket + (1 if i < remainder else 0)
            for p in sorted(bucket, key=priority, reverse=True)[:target]:
                selected.append(p)
                seen.add(tuple(p[1]))

        if len(selected) < n:
            chosen = {id(p) for p in selected}
            leftover = [p for p in pairs if id(p) not in chosen]
            for p in sorted(leftover, key=priority, reverse=True):
                if len(selected) >= n:
                    break
                selected.append(p)
                seen.add(tuple(p[1]))

        return selected
