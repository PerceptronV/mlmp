"""Pattern AST and recursive e-matcher.

A :class:`Pattern` is either a :class:`PVar` (binds to an e-class) or a
:class:`PNode` (matches an e-node with a specified ``Op`` and optional
payload predicate). E-matching searches every e-node in a target
e-class for a structural match, recursively matching child patterns
against child e-classes, and yielding every consistent substitution.

The matcher is intentionally simple: no compiled patterns, no
indexing. For ~25 §6.2 rules and programs ≤ ~100 nodes, this is fast
enough; the bottleneck is saturation iterations, not per-match cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Union

from .egraph import EClassId, EGraph
from .encode import ENode, Op


@dataclass(frozen=True)
class PVar:
    name: str


@dataclass(frozen=True)
class PNode:
    op: Op
    children: tuple
    # Either a fixed payload tuple to match exactly, a predicate
    # ``Callable[[tuple], bool]`` to test the e-node's payload, or
    # None to match any payload.
    payload: object = None


Pattern = Union[PVar, PNode]
Subst = dict[str, EClassId]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def pvar(name: str) -> PVar:
    return PVar(name)


def pnum(value: int | None = None) -> PNode:
    if value is None:
        return PNode(Op.NUM, (), None)
    return PNode(Op.NUM, (), (value,))


def pbool(value: bool | None = None) -> PNode:
    if value is None:
        return PNode(Op.BOOL, (), None)
    return PNode(Op.BOOL, (), (value,))


def pvariable(name: str | None = None) -> PNode:
    """Match a :class:`VariableNode` by name (or any name if ``None``)."""
    if name is None:
        return PNode(Op.VAR, (), None)
    return PNode(Op.VAR, (), (name,))


def papp(fn_name: str, *args: Pattern) -> PNode:
    """Match an ``ApplicationNode`` whose function is the variable ``fn_name``."""
    return PNode(Op.APP, tuple(args), (fn_name,))


def plam(params: int | tuple[str, ...], body: Pattern) -> PNode:
    """Match a lambda.

    If ``params`` is an int, match any lambda with that arity.
    If ``params`` is a tuple of names, match a lambda with exactly those
    parameter names (used for rules that target ``(λ x. x)``-shaped
    bodies, where the parameter name is referenced inside ``body``).
    """
    if isinstance(params, int):
        n_params = params

        def pred(payload: tuple) -> bool:
            (param_tuple,) = payload
            return len(param_tuple) == n_params

        return PNode(Op.LAM, (body,), pred)
    return PNode(Op.LAM, (body,), (tuple(params),))


def pif(c: Pattern, t: Pattern, e: Pattern) -> PNode:
    return PNode(Op.IF, (c, t, e), ())


def plist(*elems: Pattern) -> PNode:
    return PNode(Op.LIST, tuple(elems), ())


def pempty_list() -> PNode:
    return PNode(Op.LIST, (), ())


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

def _payload_matches(spec: object, payload: tuple) -> bool:
    if spec is None:
        return True
    if callable(spec):
        return bool(spec(payload))
    return spec == payload


def _merge(a: Subst, b: Subst, eg: EGraph) -> Subst | None:
    """Combine two substitutions, returning ``None`` on inconsistency."""
    if not a:
        return b
    if not b:
        return a
    out = dict(a)
    for k, v in b.items():
        if k in out:
            if eg.find(out[k]) != eg.find(v):
                return None
        else:
            out[k] = v
    return out


def ematch(p: Pattern, c: EClassId, eg: EGraph) -> Iterator[Subst]:
    """Yield every substitution under which ``p`` matches some e-node in ``c``."""
    c = eg.find(c)
    if isinstance(p, PVar):
        yield {p.name: c}
        return
    for n in eg.nodes_in(c):
        if n.op is not p.op:
            continue
        if not _payload_matches(p.payload, n.payload):
            continue
        if len(n.children) != len(p.children):
            continue
        yield from _match_children(p.children, n.children, eg)


def _match_children(
    ps: tuple,
    cs: tuple,
    eg: EGraph,
) -> Iterator[Subst]:
    if not ps:
        yield {}
        return
    head_p, *tail_ps = ps
    head_c, *tail_cs = cs
    for sub_head in ematch(head_p, head_c, eg):
        for sub_tail in _match_children(tuple(tail_ps), tuple(tail_cs), eg):
            merged = _merge(sub_head, sub_tail, eg)
            if merged is not None:
                yield merged


# ---------------------------------------------------------------------------
# Helpers used by rules
# ---------------------------------------------------------------------------

def find_enode(c: EClassId, op: Op, eg: EGraph) -> ENode | None:
    """Return any e-node in class ``c`` with the given ``op``, or ``None``."""
    c = eg.find(c)
    for n in eg.nodes_in(c):
        if n.op is op:
            return n
    return None


def get_literal_int(c: EClassId, eg: EGraph) -> int | None:
    n = find_enode(c, Op.NUM, eg)
    return None if n is None else int(n.payload[0])


def get_literal_bool(c: EClassId, eg: EGraph) -> bool | None:
    n = find_enode(c, Op.BOOL, eg)
    return None if n is None else bool(n.payload[0])


def get_var_name(c: EClassId, eg: EGraph) -> str | None:
    n = find_enode(c, Op.VAR, eg)
    return None if n is None else str(n.payload[0])


def is_empty_list(c: EClassId, eg: EGraph) -> bool:
    c = eg.find(c)
    for n in eg.nodes_in(c):
        if n.op is Op.LIST and not n.children:
            return True
    return False
