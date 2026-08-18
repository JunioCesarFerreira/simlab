"""Sink-rooted tree abstraction for the topology-aware P2 variant.

The tree is an *auxiliary structure derived from the binary chromosome*, not a
replacement for it, and it is deliberately **not** an RPL DODAG: it is the
structural connectivity skeleton the repair/descriptor machinery reasons about.

``RootedTreeBackend`` is the contract the rest of the heuristic programs
against.  ``ParentArrayTree`` is the straightforward implementation (parent
array + children lists); :class:`~lib.problem.topology.two_level_tree.TwoLevelTree`
is the segmented-preorder one, whose splices stay ``O(sqrt(n))``.  The two are
interchangeable: the descriptor extractor, the repair pass, the routing
knowledge and the tree operators all program against this Protocol only.
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

# Canonical identifier of the sink inside every rooted tree.  Relay nodes use
# their candidate index (>= 0), so a negative id can never collide with one.
SINK_NODE: int = -1


@runtime_checkable
class RootedTreeBackend(Protocol):
    """Minimal rooted-tree interface required by the topology heuristic."""

    @property
    def root(self) -> int:
        """Identifier of the root (the sink)."""
        ...

    def nodes(self) -> tuple[int, ...]:
        """All node ids held by the tree, root included, in insertion order."""
        ...

    def parent(self, node: int) -> int | None:
        """Parent of ``node``; ``None`` for the root and for detached nodes."""
        ...

    def children(self, node: int) -> tuple[int, ...]:
        """Direct children of ``node`` (deterministic order)."""
        ...

    def depth(self, node: int) -> int:
        """Hop distance from ``node`` to the root; ``-1`` when disconnected."""
        ...

    def path_to_root(self, node: int) -> list[int]:
        """``[node, ..., root]``; empty list when ``node`` is disconnected."""
        ...

    def subtree_nodes(self, node: int) -> list[int]:
        """``node`` plus every descendant, in BFS order."""
        ...

    def cut_subtree(self, node: int) -> list[int]:
        """Detach ``node`` from its parent and return the detached subtree."""
        ...

    def link(self, node: int, new_parent: int) -> None:
        """(Re)attach ``node`` under ``new_parent``."""
        ...

    def is_connected_to_root(self, node: int) -> bool:
        """Whether ``node`` still has a path to the root."""
        ...

    def reroot_component(self, new_root: int) -> None:
        """Re-root the detached component containing ``new_root``."""
        ...


class ParentArrayTree:
    """Rooted tree stored as a parent map plus children lists.

    Every node known to the tree is either *attached* (its parent chain reaches
    the root) or *detached* (``parent is None`` and not the root).  Detached
    nodes stay in the structure so a repair pass can re-link them.
    """

    __slots__ = ("_root", "_parent", "_children", "_order")

    def __init__(self, root: int = SINK_NODE) -> None:
        self._root = root
        self._parent: dict[int, int | None] = {root: None}
        self._children: dict[int, list[int]] = {root: []}
        self._order: list[int] = [root]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_parent_map(
        cls,
        parent_map: dict[int, int | None],
        root: int = SINK_NODE,
        detached: Iterable[int] = (),
    ) -> "ParentArrayTree":
        """Build a tree from a ``node -> parent`` map (root excluded)."""
        tree = cls(root)
        for node in sorted(parent_map):
            tree.add_node(node, None)
        for node in sorted(detached):
            tree.add_node(node, None)
        for node in sorted(parent_map):
            target = parent_map[node]
            if target is not None:
                tree.link(node, target)
        return tree

    def add_node(self, node: int, parent: int | None = None) -> None:
        """Register ``node`` under ``parent`` (``None`` leaves it detached)."""
        if node == self._root:
            raise ValueError(f"Cannot re-add the root node {node!r}.")
        if node not in self._parent:
            self._parent[node] = None
            self._children[node] = []
            self._order.append(node)
        if parent is not None:
            self.link(node, parent)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def root(self) -> int:
        return self._root

    def nodes(self) -> tuple[int, ...]:
        return tuple(self._order)

    def __contains__(self, node: int) -> bool:
        return node in self._parent

    def __len__(self) -> int:
        return len(self._order)

    def parent(self, node: int) -> int | None:
        self._assert_known(node)
        return self._parent[node]

    def children(self, node: int) -> tuple[int, ...]:
        self._assert_known(node)
        return tuple(self._children[node])

    def path_to_root(self, node: int) -> list[int]:
        self._assert_known(node)
        path: list[int] = []
        seen: set[int] = set()
        cur: int | None = node
        while cur is not None:
            if cur in seen:  # defensive: a cycle means the structure is corrupt
                return []
            seen.add(cur)
            path.append(cur)
            if cur == self._root:
                return path
            cur = self._parent[cur]
        return []

    def depth(self, node: int) -> int:
        path = self.path_to_root(node)
        return len(path) - 1 if path else -1

    def is_connected_to_root(self, node: int) -> bool:
        return bool(self.path_to_root(node))

    def subtree_nodes(self, node: int) -> list[int]:
        self._assert_known(node)
        out: list[int] = []
        queue: list[int] = [node]
        while queue:
            cur = queue.pop(0)
            out.append(cur)
            queue.extend(self._children[cur])
        return out

    def attached_nodes(self) -> list[int]:
        """Every node (root included) that still reaches the root."""
        return [n for n in self._order if self.is_connected_to_root(n)]

    def detached_nodes(self) -> list[int]:
        """Every non-root node with no path to the root."""
        return [
            n for n in self._order
            if n != self._root and not self.is_connected_to_root(n)
        ]

    def leaves(self) -> list[int]:
        """Attached nodes without children."""
        return [n for n in self.attached_nodes() if not self._children[n]]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def cut_subtree(self, node: int) -> list[int]:
        """Detach ``node`` from its parent and return the detached subtree."""
        self._assert_known(node)
        if node == self._root:
            raise ValueError("The root cannot be cut.")
        detached = self.subtree_nodes(node)
        parent = self._parent[node]
        if parent is not None:
            self._children[parent].remove(node)
        self._parent[node] = None
        return detached

    def reroot_component(self, new_root: int) -> None:
        """Re-root the *detached* component containing ``new_root``.

        Inverts the parent chain from ``new_root`` up to its component root,
        which is what lets the Change-Ancestor operator re-hang a fragment by
        a different node.  Every inverted edge already existed, so a caller
        working under a distance constraint keeps satisfying it.  ``O(depth)``.
        """
        self._assert_known(new_root)
        if self.is_connected_to_root(new_root):
            raise ValueError("Only a detached component can be re-rooted.")

        chain: list[int] = []
        cur: int | None = new_root
        while cur is not None:
            chain.append(cur)
            cur = self._parent[cur]
        if len(chain) < 2:
            return

        for child, parent in zip(chain, chain[1:]):
            self._children[parent].remove(child)
            self._parent[parent] = child
            self._children[child].append(parent)
        self._parent[new_root] = None

    def link(self, node: int, new_parent: int) -> None:
        """Attach ``node`` under ``new_parent``, rejecting cycles."""
        self._assert_known(node)
        self._assert_known(new_parent)
        if node == self._root:
            raise ValueError("The root cannot be linked to a parent.")
        if new_parent == node:
            raise ValueError(f"Node {node!r} cannot be its own parent.")
        if new_parent in self.subtree_nodes(node):
            raise ValueError(
                f"Linking {node!r} under {new_parent!r} would create a cycle."
            )
        old_parent = self._parent[node]
        if old_parent is not None:
            self._children[old_parent].remove(node)
        self._parent[node] = new_parent
        self._children[new_parent].append(node)

    # ------------------------------------------------------------------
    def _assert_known(self, node: int) -> None:
        if node not in self._parent:
            raise KeyError(f"Unknown tree node: {node!r}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ParentArrayTree(root={self._root}, nodes={len(self._order)}, "
            f"detached={len(self.detached_nodes())})"
        )
