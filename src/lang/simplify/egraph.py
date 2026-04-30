"""E-graph data structure: e-classes, e-nodes, hashcons, congruence rebuild.

The implementation follows the egg-style design: unions are O(α(n)) plus
a deferred congruence-repair pass triggered by :meth:`EGraph.rebuild`.
Saturation calls ``rebuild`` once per iteration after applying a batch
of unions; this avoids cascading-rebuild costs inside e-matching.

E-nodes are flat ``(op, children, payload)`` tuples — see
:mod:`src.lang.simplify.encode` for how the heterogeneous AST shapes
project onto this form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .encode import ENode, Op


EClassId = int


class UnionFind:
    """Union-find with path compression and union-by-size."""

    __slots__ = ("_parent", "_size")

    def __init__(self) -> None:
        self._parent: list[int] = []
        self._size: list[int] = []

    def make_set(self) -> int:
        i = len(self._parent)
        self._parent.append(i)
        self._size.append(1)
        return i

    def find(self, x: int) -> int:
        p = self._parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        return ra


@dataclass
class EClass:
    """A set of e-nodes deemed equivalent under the e-graph's congruence."""

    id: EClassId
    nodes: list[ENode] = field(default_factory=list)
    # Parents: e-nodes (in their canonical-at-insert form) that reference
    # this class as a child, alongside the e-class id those parents
    # currently inhabit. Used by :meth:`EGraph.rebuild` to propagate
    # unions through enclosing contexts.
    parents: list[tuple[ENode, EClassId]] = field(default_factory=list)


class EGraph:
    """Hashconsed e-graph with deferred congruence rebuild.

    Usage::

        eg = EGraph()
        c = eg.add(ENode(Op.NUM, (), (5,)))
        eg.union(c, eg.add(ENode(Op.NUM, (), (5,))))   # no-op (hashcons)
        eg.rebuild()
    """

    def __init__(self) -> None:
        self._uf = UnionFind()
        self._classes: dict[EClassId, EClass] = {}
        self._hashcons: dict[ENode, EClassId] = {}
        # Pending list of (possibly stale) e-class ids whose parent links
        # need to be re-canonicalised on the next ``rebuild``.
        self._pending: list[EClassId] = []

    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------

    def find(self, c: EClassId) -> EClassId:
        return self._uf.find(c)

    def _canon(self, n: ENode) -> ENode:
        if not n.children:
            return n
        canon_children = tuple(self._uf.find(ch) for ch in n.children)
        if canon_children == n.children:
            return n
        return ENode(n.op, canon_children, n.payload)

    def add(self, node: ENode) -> EClassId:
        """Insert ``node`` (canonicalising its children) and return its e-class id."""
        canon = self._canon(node)
        existing = self._hashcons.get(canon)
        if existing is not None:
            return self._uf.find(existing)
        new_id = self._uf.make_set()
        cls = EClass(id=new_id, nodes=[canon])
        self._classes[new_id] = cls
        self._hashcons[canon] = new_id
        # Register this e-node as a parent of each child class.
        for ch in canon.children:
            ch_root = self._uf.find(ch)
            self._classes[ch_root].parents.append((canon, new_id))
        return new_id

    def union(self, a: EClassId, b: EClassId) -> EClassId:
        """Merge the e-classes of ``a`` and ``b``.

        Does not eagerly resolve congruence — the caller must invoke
        :meth:`rebuild` to repair parent contexts.
        """
        ra, rb = self._uf.find(a), self._uf.find(b)
        if ra == rb:
            return ra
        new_root = self._uf.union(ra, rb)
        old = rb if new_root == ra else ra
        # Migrate nodes and parent links from the absorbed class.
        absorbed = self._classes.pop(old)
        survivor = self._classes[new_root]
        survivor.nodes.extend(absorbed.nodes)
        survivor.parents.extend(absorbed.parents)
        self._pending.append(new_root)
        return new_root

    def rebuild(self) -> int:
        """Restore the congruence invariant after a batch of unions.

        Returns the number of unions performed during rebuild.
        """
        n_unions = 0
        while self._pending:
            c = self._uf.find(self._pending.pop())
            cls = self._classes[c]
            new_parents: list[tuple[ENode, EClassId]] = []
            seen_in_class: dict[ENode, EClassId] = {}
            for pnode, pclass in cls.parents:
                # Strip the stale entry from the hashcons; re-insert
                # under the canonical form.
                self._hashcons.pop(pnode, None)
                canon = self._canon(pnode)
                pclass_root = self._uf.find(pclass)
                existing = self._hashcons.get(canon)
                if existing is None:
                    self._hashcons[canon] = pclass_root
                    new_parents.append((canon, pclass_root))
                    # Track within-class duplicates separately so two
                    # parents that canonicalise the same way still
                    # collapse below.
                    seen_in_class[canon] = pclass_root
                elif self._uf.find(existing) != pclass_root:
                    merged = self.union(existing, pclass_root)
                    self._hashcons[canon] = merged
                    n_unions += 1
                    new_parents.append((canon, merged))
                else:
                    new_parents.append((canon, pclass_root))
            cls.parents = new_parents
            # Also de-duplicate within-class nodes that became congruent.
            unique_nodes: dict[ENode, None] = {}
            for n in cls.nodes:
                unique_nodes[self._canon(n)] = None
            cls.nodes = list(unique_nodes.keys())
        return n_unions

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def classes(self) -> Iterator[EClass]:
        # Snapshot to allow callers to mutate during iteration safely.
        return iter(list(self._classes.values()))

    def nodes_in(self, c: EClassId) -> Iterable[ENode]:
        return self._classes[self._uf.find(c)].nodes

    def num_enodes(self) -> int:
        return sum(len(c.nodes) for c in self._classes.values())

    def num_classes(self) -> int:
        return len(self._classes)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def dump(self) -> str:
        lines = []
        for cid in sorted(self._classes):
            cls = self._classes[cid]
            lines.append(f"C{cid}:")
            for n in cls.nodes:
                pretty = f"  {n.op.name}{n.payload if n.payload else ''}"
                if n.children:
                    pretty += f" [{', '.join(f'C{self._uf.find(c)}' for c in n.children)}]"
                lines.append(pretty)
        return "\n".join(lines)
