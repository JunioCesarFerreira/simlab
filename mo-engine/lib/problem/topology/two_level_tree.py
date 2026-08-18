"""Two-level rooted-forest structure (2LETT-style) for tree-encoded P2.

## What this is

The classical linear encoding of a rooted forest (Delbem et al., *node-depth
encoding*) stores a forest as the **preorder walk** of its trees, each entry a
``(node, depth)`` pair:

    [(sigma,0) (a,1) (b,2) (c,2) (d,1) | (e,0) (f,1)]
     |<------ tree rooted at sigma ----->| |<-fragment->|

The property that makes it useful is that **a subtree occupies a contiguous
range**: the subtree of the entry at position ``i`` runs until the first later
entry whose depth is ``<= depth(i)``.  Moving a subtree is therefore a *splice*
of one range, which is exactly what an evolutionary operator needs.

A flat array makes that splice ``O(n)``.  The **two-level** structure — the
same idea as the two-level doubly-linked list used for TSP tours — cuts the
sequence into ``~sqrt(n)`` contiguous **segments**, each holding ``~sqrt(n)``
entries, so a splice rewrites at most the two boundary segments and relinks
whole segments in between.

Implemented here, with the costs the structure is chosen for:

| operation | cost |
|---|---|
| ``precedes(a, b)`` (sequence order) | ``O(1)`` |
| ``sequence_index(v)`` | ``O(1)`` |
| ``cut_subtree(v)`` / ``link(v, p)`` of a block of ``m`` nodes | ``O(m + sqrt(n))`` |
| ``rebalance`` | ``O(n)``, amortised ``O(sqrt(n))`` per operation |

## What this deliberately is not

The TSP form of the structure carries a **reversal bit** per segment so a
2-opt move can flip a whole sub-path in ``O(1)``.  Rooted trees have a fixed
orientation (every edge points at the parent), so there is nothing to reverse
and no reversal bit is kept.  Re-rooting a fragment — the one operation that
does invert a path — is done explicitly by
:meth:`TwoLevelTree.reroot_component` in ``O(depth)``.

Everything else follows :class:`RootedTreeBackend`, so this class and
:class:`ParentArrayTree` are interchangeable: the descriptor extractor, the
repair pass and the routing knowledge consume either.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from .rooted_tree import SINK_NODE

# A segment is rebuilt from scratch once it grows past this multiple of the
# target size; the whole index is rebuilt once the segment count does.
_REBALANCE_FACTOR = 3


@dataclass(slots=True)
class _Segment:
    """One contiguous chunk of the preorder sequence."""

    nodes: list[int] = field(default_factory=list)
    rank: int = 0       # position of this segment in the sequence
    start: int = 0      # number of entries before this segment

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.nodes)


class TwoLevelTree:
    """Rooted forest stored as a segmented preorder sequence.

    Component 0 is the tree rooted at ``root``; any further component is a
    *detached fragment*, produced by :meth:`cut_subtree` and waiting to be
    re-linked.  Detached nodes therefore behave exactly as they do in
    :class:`ParentArrayTree`, which keeps both backends substitutable.
    """

    __slots__ = ("_root", "_parent", "_children", "_depth", "_segments", "_seg_of", "_pos_of")

    def __init__(self, root: int = SINK_NODE) -> None:
        self._root = root
        self._parent: dict[int, Optional[int]] = {root: None}
        self._children: dict[int, list[int]] = {root: []}
        self._depth: dict[int, int] = {root: 0}
        first = _Segment(nodes=[root])
        self._segments: list[_Segment] = [first]
        self._seg_of: dict[int, _Segment] = {root: first}
        self._pos_of: dict[int, int] = {root: 0}
        self._reindex()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_parent_map(
        cls,
        parent_map: dict[int, Optional[int]],
        root: int = SINK_NODE,
        detached: Iterable[int] = (),
    ) -> "TwoLevelTree":
        """Build from a ``node -> parent`` map (root excluded from the map).

        Handles a *forest*, not just a tree: a node whose parent chain never
        reaches ``root`` is rebuilt as a detached component, so the map
        produced by :meth:`parent_map` round-trips exactly.
        """
        tree = cls(root)
        nodes = sorted(set(parent_map) | set(detached))
        for node in nodes:
            tree.add_node(node, None)

        def _rank(node: int) -> int:
            """Hops from ``node`` up to its component root, for link ordering."""
            distance = 0
            seen: set[int] = set()
            cur = node
            while cur not in seen:
                seen.add(cur)
                target = parent_map.get(cur)
                if target is None:
                    return distance
                cur = target
                distance += 1
            return distance  # cycle: linking will reject it below

        for node in sorted(nodes, key=_rank):
            target = parent_map.get(node)
            if target is not None:
                tree.link(node, target)
        return tree

    def add_node(self, node: int, parent: Optional[int] = None) -> None:
        """Register ``node``; ``parent=None`` starts a new detached component."""
        if node == self._root:
            raise ValueError(f"Cannot re-add the root node {node!r}.")
        if node in self._parent:
            if parent is not None:
                self.link(node, parent)
            return
        self._parent[node] = None
        self._children[node] = []
        self._depth[node] = 0
        self._append_block([node])
        if parent is not None:
            self.link(node, parent)

    # ------------------------------------------------------------------
    # RootedTreeBackend - queries
    # ------------------------------------------------------------------
    @property
    def root(self) -> int:
        return self._root

    def nodes(self) -> tuple[int, ...]:
        """Every node, in preorder (root first)."""
        return tuple(node for segment in self._segments for node in segment.nodes)

    def __contains__(self, node: int) -> bool:
        return node in self._parent

    def __len__(self) -> int:
        return len(self._parent)

    def __iter__(self) -> Iterator[int]:  # pragma: no cover - convenience
        return iter(self.nodes())

    def parent(self, node: int) -> Optional[int]:
        self._assert_known(node)
        return self._parent[node]

    def children(self, node: int) -> tuple[int, ...]:
        self._assert_known(node)
        return tuple(self._children[node])

    def depth(self, node: int) -> int:
        """Hop distance to the root; ``-1`` for a detached node."""
        self._assert_known(node)
        return self._depth[node] if self._in_root_component(node) else -1

    def path_to_root(self, node: int) -> list[int]:
        self._assert_known(node)
        path: list[int] = []
        seen: set[int] = set()
        cur: Optional[int] = node
        while cur is not None:
            if cur in seen:  # defensive: a cycle means the structure is corrupt
                return []
            seen.add(cur)
            path.append(cur)
            if cur == self._root:
                return path
            cur = self._parent[cur]
        return []

    def is_connected_to_root(self, node: int) -> bool:
        return self._in_root_component(node)

    def subtree_nodes(self, node: int) -> list[int]:
        """``node`` plus every descendant, in preorder — a contiguous range."""
        self._assert_known(node)
        base = self._depth[node]
        block = [node]
        for candidate in self._iter_from(node, skip_first=True):
            if self._depth[candidate] <= base:
                break
            block.append(candidate)
        return block

    def attached_nodes(self) -> list[int]:
        """Every node of the root component, in preorder."""
        out: list[int] = []
        for node in self._iter_from(self._root):
            if node != self._root and self._depth[node] == 0:
                break  # start of the next component
            out.append(node)
        return out

    def detached_nodes(self) -> list[int]:
        attached = set(self.attached_nodes())
        return [node for node in self.nodes() if node not in attached]

    def leaves(self) -> list[int]:
        return [node for node in self.attached_nodes() if not self._children[node]]

    # ------------------------------------------------------------------
    # Two-level index
    # ------------------------------------------------------------------
    def sequence_index(self, node: int) -> int:
        """Position of ``node`` in the preorder sequence — ``O(1)``."""
        self._assert_known(node)
        return self._seg_of[node].start + self._pos_of[node]

    def precedes(self, a: int, b: int) -> bool:
        """Whether ``a`` comes before ``b`` in the sequence — ``O(1)``."""
        self._assert_known(a)
        self._assert_known(b)
        sa, sb = self._seg_of[a], self._seg_of[b]
        return (sa.rank, self._pos_of[a]) < (sb.rank, self._pos_of[b])

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def segment_sizes(self) -> list[int]:
        return [len(segment.nodes) for segment in self._segments]

    def _target_segment_size(self) -> int:
        return max(1, int(math.isqrt(max(1, len(self._parent)))))

    def _reindex(self) -> None:
        """Refresh segment ranks and prefix counts — ``O(sqrt(n))``."""
        start = 0
        for rank, segment in enumerate(self._segments):
            segment.rank = rank
            segment.start = start
            start += len(segment.nodes)

    def _refresh(self, segment: _Segment) -> None:
        """Re-point the nodes of one segment at it — ``O(sqrt(n))``."""
        for position, node in enumerate(segment.nodes):
            self._seg_of[node] = segment
            self._pos_of[node] = position

    def rebalance(self) -> None:
        """Re-split the sequence into ``~sqrt(n)`` equal segments."""
        flat = [node for segment in self._segments for node in segment.nodes]
        size = self._target_segment_size()
        self._segments = [
            _Segment(nodes=flat[i:i + size]) for i in range(0, len(flat), size)
        ] or [_Segment(nodes=[])]
        for segment in self._segments:
            self._refresh(segment)
        self._reindex()

    def _maybe_rebalance(self) -> None:
        target = self._target_segment_size()
        if len(self._segments) > _REBALANCE_FACTOR * target or any(
            len(segment.nodes) > _REBALANCE_FACTOR * target for segment in self._segments
        ):
            self.rebalance()

    # ------------------------------------------------------------------
    # Sequence splicing
    # ------------------------------------------------------------------
    def _iter_from(self, node: int, skip_first: bool = False) -> Iterator[int]:
        """Walk the sequence forward from ``node`` — ``O(1)`` per step."""
        segment = self._seg_of[node]
        position = self._pos_of[node] + (1 if skip_first else 0)
        rank = segment.rank
        while rank < len(self._segments):
            current = self._segments[rank]
            while position < len(current.nodes):
                yield current.nodes[position]
                position += 1
            rank += 1
            position = 0

    def _extract_block(self, block: list[int]) -> None:
        """Remove ``len(block)`` consecutive entries starting at ``block[0]``."""
        head = self._seg_of[block[0]]
        position = self._pos_of[block[0]]
        remaining = len(block)
        rank = head.rank
        touched: list[_Segment] = []
        while remaining > 0 and rank < len(self._segments):
            segment = self._segments[rank]
            take = min(remaining, len(segment.nodes) - position)
            del segment.nodes[position:position + take]
            remaining -= take
            touched.append(segment)
            position = 0
            rank += 1
        self._segments = [s for s in self._segments if s.nodes]
        if not self._segments:
            self._segments = [_Segment(nodes=[])]
        for segment in touched:
            if segment.nodes:
                self._refresh(segment)
        self._reindex()

    def _append_block(self, block: list[int]) -> None:
        """Append entries at the very end of the sequence (a new component)."""
        last = self._segments[-1]
        if len(last.nodes) + len(block) <= _REBALANCE_FACTOR * self._target_segment_size():
            last.nodes.extend(block)
            self._refresh(last)
        else:
            segment = _Segment(nodes=list(block))
            self._segments.append(segment)
            self._refresh(segment)
        self._reindex()
        self._maybe_rebalance()

    def _insert_block_after(self, anchor: int, block: list[int]) -> None:
        """Insert ``block`` immediately after ``anchor`` in the sequence."""
        segment = self._seg_of[anchor]
        position = self._pos_of[anchor] + 1
        segment.nodes[position:position] = block
        self._refresh(segment)
        self._reindex()
        self._maybe_rebalance()

    # ------------------------------------------------------------------
    # RootedTreeBackend - mutation
    # ------------------------------------------------------------------
    def cut_subtree(self, node: int) -> list[int]:
        """Detach ``node``'s subtree into its own component; returns its nodes."""
        self._assert_known(node)
        if node == self._root:
            raise ValueError("The root cannot be cut.")

        block = self.subtree_nodes(node)
        old_parent = self._parent[node]
        if old_parent is not None:
            self._children[old_parent].remove(node)
        self._parent[node] = None

        shift = self._depth[node]
        if shift:
            for member in block:
                self._depth[member] -= shift

        self._extract_block(block)
        self._append_block(block)
        return block

    def link(self, node: int, new_parent: int) -> None:
        """Attach ``node``'s component under ``new_parent``."""
        self._assert_known(node)
        self._assert_known(new_parent)
        if node == self._root:
            raise ValueError("The root cannot be linked to a parent.")
        if new_parent == node:
            raise ValueError(f"Node {node!r} cannot be its own parent.")

        block = self.subtree_nodes(node)
        if new_parent in block:
            raise ValueError(
                f"Linking {node!r} under {new_parent!r} would create a cycle."
            )

        if self._parent[node] is not None:
            self._children[self._parent[node]].remove(node)
            self._parent[node] = None
            shift = self._depth[node]
            if shift:
                for member in block:
                    self._depth[member] -= shift
            self._extract_block(block)
        else:
            self._extract_block(block)

        shift = self._depth[new_parent] + 1
        for member in block:
            self._depth[member] += shift
        self._parent[node] = new_parent
        self._children[new_parent].append(node)
        self._insert_block_after(new_parent, block)

    def reroot_component(self, new_root: int) -> None:
        """Re-root the *detached* component containing ``new_root``.

        Inverts the parent chain from ``new_root`` up to its component root —
        the one operation a rooted forest cannot express as a pure splice, and
        the reason the TSP reversal bit has no analogue here.  ``O(depth)``.
        """
        self._assert_known(new_root)
        if self._in_root_component(new_root):
            raise ValueError("Only a detached component can be re-rooted.")

        chain: list[int] = []
        cur: Optional[int] = new_root
        while cur is not None:
            chain.append(cur)
            cur = self._parent[cur]

        old_root = chain[-1]
        if old_root == new_root:
            return

        # Snapshot the component while the sequence still matches the tree:
        # inverting the chain invalidates both the depths and the preorder.
        block = self.subtree_nodes(old_root)
        self._extract_block(block)

        for child, parent in zip(chain, chain[1:]):
            self._children[parent].remove(child)
            self._parent[parent] = child
            self._children[child].append(parent)
        self._parent[new_root] = None

        self._rebuild_component(new_root)

    def _rebuild_component(self, component_root: int) -> None:
        """Re-emit a component's preorder block after a structural change."""
        order: list[int] = []
        stack = [component_root]
        while stack:
            node = stack.pop()
            order.append(node)
            stack.extend(reversed(self._children[node]))

        for node in order:
            parent = self._parent[node]
            self._depth[node] = 0 if parent is None else self._depth[parent] + 1
        self._append_block(order)

    # ------------------------------------------------------------------
    def _in_root_component(self, node: int) -> bool:
        seen: set[int] = set()
        cur: Optional[int] = node
        while cur is not None:
            if cur in seen:
                return False
            seen.add(cur)
            if cur == self._root:
                return True
            cur = self._parent[cur]
        return False

    def _assert_known(self, node: int) -> None:
        if node not in self._parent:
            raise KeyError(f"Unknown tree node: {node!r}")

    def parent_map(self) -> dict[int, Optional[int]]:
        """``node -> parent`` for every non-root node (``None`` = detached)."""
        return {n: p for n, p in self._parent.items() if n != self._root}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TwoLevelTree(root={self._root}, nodes={len(self._parent)}, "
            f"segments={len(self._segments)}, detached={len(self.detached_nodes())})"
        )
