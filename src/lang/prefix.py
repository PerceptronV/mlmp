"""Legal-prefix automaton: which tokens can still complete a parseable program.

Mirrors :mod:`src.lang.parser`'s accepted language incrementally, but instead
of raising on a bad token it enumerates the tokens that *could* come next. Feed
the choice back with :meth:`PrefixState.advance` and the state machine tracks
the rest.

Used for constrained decoding (see ``ProgramIO.decode_one`` /
``decode_batch``): masking the decoder's logits to :meth:`PrefixState.legal`
makes a malformed generation impossible rather than merely unlikely. The
guarantees are syntactic and scope-level —

  * the emitted string parses,
  * every function is applied to exactly its grammar arity,
  * every variable reference is bound by an enclosing lambda,
  * ``#`` (a comment marker, which would swallow the rest of the line) is
    never emitted,

— but *not* type-level: ``(+ [1 2] 3)`` is well-formed, in scope, correctly
applied, and still ill-typed. Catching that needs the argument types in
``Grammar.functions[name]['arg_types']``, which this module deliberately
leaves alone.

The automaton is stricter than the parser in three places, each matching how
the corpora are actually written rather than the full grammar the parser would
tolerate: a program must be a lambda, lambda parameters must be bracketed
(``(λ (_p0) ...)``, not ``(λ _p0 ...)``), and an application's head must be a
function name, ``λ``, ``if``, or a nested s-expression. ``tests/test_prefix.py``
replays the corpus through it to prove none of that rejects real programs.
"""

from __future__ import annotations

from typing import Iterable

from .grammar import DefaultGrammar, Grammar
from .type_utils import get_args, get_origin

# Vocabulary categories the decoder can emit (see src/data/tokeniser.py).
DEFAULT_INT_MAX = 99
DEFAULT_N_VARS = 26

BOOLEANS = frozenset({'true', 'false'})
LPAREN, RPAREN, LBRACKET, RBRACKET, LAMBDA, IF = '(', ')', '[', ']', 'λ', 'if'


def _max_callable_arity(grammar: Grammar) -> int:
    """Widest lambda the grammar can consume, e.g. 2 for ``fold``'s binary
    operator. Bounds how many parameters an inner lambda may bind."""
    widest = 1
    for spec in grammar.functions.values():
        for arg_type in spec['arg_types']:
            if get_origin(arg_type) is not None and callable(get_origin(arg_type)):
                args = get_args(arg_type)
                if args and isinstance(args[0], list):
                    widest = max(widest, len(args[0]))
    return widest


class PrefixError(ValueError):
    """Raised when :meth:`PrefixState.advance` is given an illegal token."""


class _Frame:
    """One open construct. ``kind`` drives both ``legal`` and ``advance``:

    ``head``    — after ``(``, waiting for the head of the s-expression
    ``lparams`` — after ``λ``, waiting for the ``(`` that opens the parameter
                  list
    ``args``    — after the head; ``remaining`` counts arguments still owed
                  (``None`` = unknown arity, i.e. a computed head, so any
                  number of arguments is accepted)
    ``params``  — inside a lambda's ``( ... )`` parameter list
    ``body``    — a lambda's single body expression (then the closing paren)
    ``if``      — an ``if`` form's three expressions
    ``list``    — inside a ``[ ... ]`` literal, which takes any number of
                  elements
    """

    __slots__ = ('kind', 'remaining', 'names', 'done')

    def __init__(self, kind: str, remaining: int | None = None):
        self.kind = kind
        self.remaining = remaining
        self.names: list[str] = []   # params collected, for `params`/`body`
        self.done = False            # body seen, for `body`


class PrefixState:
    """Incremental legality checker for one program being decoded.

    ``var_names`` is the pool of lambda-parameter names the vocabulary can
    express (``_p0.._p25`` by default); a parameter may be introduced only if
    it isn't already in scope, and referenced only while it is.

    ``name_map`` is the episode's ``{orig: mapped}`` symbol-shuffling
    permutation (see ``ProgramIO.sample_name_map``). The decoder emits *mapped*
    names, so arity has to be resolved through its inverse — without this, a
    shuffled run would enforce the wrong argument counts.
    """

    def __init__(
        self,
        grammar: Grammar = DefaultGrammar,
        int_max: int = DEFAULT_INT_MAX,
        var_names: Iterable[str] | None = None,
        require_lambda: bool = True,
        name_map: dict[str, str] | None = None,
        max_lambda_params: int | None = None,
    ):
        self.grammar = grammar
        self._mapped_to_orig = {v: k for k, v in (name_map or {}).items()}
        self.max_lambda_params = (max_lambda_params if max_lambda_params is not None
                                  else _max_callable_arity(grammar))
        self._min_arity = min((len(spec['arg_types'])
                               for spec in grammar.functions.values()), default=1)
        self.fn_names = frozenset(grammar.functions)
        self.numbers = frozenset(str(i) for i in range(int_max + 1))
        self.var_names = tuple(var_names) if var_names is not None else tuple(
            f'_p{i}' for i in range(DEFAULT_N_VARS)
        )
        self.require_lambda = require_lambda

        self._stack: list[_Frame] = []
        self._scopes: list[list[str]] = []
        self._complete = False
        self._started = False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def complete(self) -> bool:
        """True once a whole top-level expression has been emitted, i.e. when
        ``<end>`` is the only sensible continuation."""
        return self._complete

    @property
    def bound(self) -> frozenset[str]:
        return frozenset(v for scope in self._scopes for v in scope)

    def _expr_start(self) -> frozenset[str]:
        """Tokens that can begin an expression here."""
        return (frozenset({LPAREN, LBRACKET}) | self.numbers | BOOLEANS
                | self.bound | self.fn_names)

    def min_completion_length(self) -> int:
        """How many tokens are still needed to finish the program.

        Each pending expression is costed as one token (the cheapest expression
        is an atom) and each open construct as its closing token. Never an
        under-estimate — an expression already in progress is counted once by
        its own frame and again by its parent — so a decoder that starts
        closing once its budget reaches this bound always finishes in time.
        Constrained decoding stops a program being *ill-formed*; this is what
        stops it being *cut off*, which is just as malformed.
        """
        if self._complete:
            return 0
        if not self._stack:
            # '(' 'λ' '(' param ')' body ')' — the shortest program there is.
            return 7 if self.require_lambda else 1
        total = 0
        for frame in self._stack:
            if frame.kind == 'head':
                total += 2 + self._min_arity   # fn name + its arguments + ')'
            elif frame.kind == 'args':
                total += (frame.remaining or 0) + 1
            elif frame.kind == 'lparams':
                total += 5          # '(' param ')' body ')'
            elif frame.kind == 'params':
                total += (1 if not frame.names else 0) + 3   # [param] ')' body ')'
            elif frame.kind == 'body':
                total += 1 if frame.done else 2
            elif frame.kind == 'if':
                total += frame.remaining + 1
            elif frame.kind == 'list':
                total += 1
        return total

    def legal(self, closing_only: bool = False) -> frozenset[str]:
        """The tokens that keep the program parseable. Empty once
        :attr:`complete` — only ``<end>`` may follow, which the decoder owns.

        With ``closing_only``, tokens that open a new construct (``(``, ``[``)
        are dropped whenever something else is available, so a decoder running
        out of budget can still close cleanly instead of being truncated.
        """
        tokens = self._legal()
        if closing_only:
            narrowed = tokens - {LPAREN, LBRACKET}
            if narrowed:
                return narrowed
        return tokens

    def _legal(self) -> frozenset[str]:
        if self._complete:
            return frozenset()
        if not self._stack:
            # Top level: a program is a lambda, so it opens with '('.
            return frozenset({LPAREN}) if self.require_lambda else self._expr_start()

        top = self._stack[-1]
        if top.kind == 'head':
            heads = frozenset({LAMBDA, IF, LPAREN}) | self.fn_names
            # Only the outermost s-expression is forced to be a lambda.
            return frozenset({LAMBDA}) if (self.require_lambda and len(self._stack) == 1) else heads
        if top.kind == 'args':
            if top.remaining is None:
                return self._expr_start() | {RPAREN}
            return self._expr_start() if top.remaining > 0 else frozenset({RPAREN})
        if top.kind == 'lparams':
            return frozenset({LPAREN})
        if top.kind == 'params':
            # A program is list[int] -> list[int], so the outermost lambda binds
            # exactly one parameter; inner lambdas are bounded by the widest
            # callable the grammar takes (2, for a fold's binary operator).
            cap = 1 if (self.require_lambda and len(self._stack) == 1) else self.max_lambda_params
            free = (frozenset(v for v in self.var_names if v not in self.bound)
                    if len(top.names) < cap else frozenset())
            return free | (frozenset({RPAREN}) if top.names else frozenset())
        if top.kind == 'body':
            return frozenset({RPAREN}) if top.done else self._expr_start()
        if top.kind == 'if':
            return self._expr_start() if top.remaining else frozenset({RPAREN})
        if top.kind == 'list':
            return self._expr_start() | {RBRACKET}
        raise AssertionError(f"unknown frame kind {top.kind!r}")

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------
    def advance(self, token: str) -> None:
        """Consume ``token``, which must be in :meth:`legal`."""
        if token not in self.legal():
            raise PrefixError(f"illegal token {token!r}; legal here: "
                              f"{sorted(self.legal())[:12]}...")
        self._started = True

        if token == LPAREN:
            top = self._stack[-1] if self._stack else None
            if top is not None and top.kind == 'lparams':
                top.kind = 'params'   # this paren opens the parameter list
                return
            self._stack.append(_Frame('head'))
            return
        if token == LBRACKET:
            self._stack.append(_Frame('list'))
            return
        if token == RBRACKET:
            self._stack.pop()
            self._expr_done()
            return
        if token == RPAREN:
            frame = self._stack[-1]
            if frame.kind == 'params':
                # Parameter list closed: its names scope over the body.
                self._scopes.append(list(frame.names))
                frame.kind = 'body'
                return
            self._stack.pop()
            if frame.kind == 'body':
                self._scopes.pop()
            self._expr_done()
            return

        top = self._stack[-1] if self._stack else None
        if top is not None and top.kind == 'head':
            if token == LAMBDA:
                top.kind = 'lparams'
                return
            if token == IF:
                top.kind = 'if'
                top.remaining = 3
                return
            # A function name in head position fixes the application's arity.
            orig = self._mapped_to_orig.get(token, token)
            spec = self.grammar.functions.get(self.grammar.name_map.get(orig, orig))
            top.kind = 'args'
            top.remaining = len(spec['arg_types']) if spec else None
            return
        if top is not None and top.kind == 'params':
            top.names.append(token)
            return

        # Anything else here is a complete atom: number, boolean, variable, or
        # a bare function name passed as a value (e.g. `(fold + 0 xs)`).
        self._expr_done()

    def _expr_done(self) -> None:
        """Record that one complete expression just closed."""
        if not self._stack:
            self._complete = True
            return
        top = self._stack[-1]
        if top.kind == 'args' and top.remaining is not None:
            top.remaining -= 1
        elif top.kind == 'if':
            top.remaining -= 1
        elif top.kind == 'body':
            top.done = True
        elif top.kind == 'head':
            # A computed head, e.g. `((λ (_p0) _p0) xs)`: arity unknown.
            top.kind = 'args'
            top.remaining = None
        # 'list' takes any number of elements, so nothing to record.
