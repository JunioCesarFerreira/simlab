"""Genetic operators closed over feasible sink-rooted trees.

These are the operators that make the repair pass unnecessary.  Where the
mask-encoded P2 lets crossover and mutation produce a disconnected chromosome
and then fixes it afterwards, here **every move is feasibility-preserving by
construction**:

* a node is only ever linked under a parent it can actually hear
  (``d(child, parent) <= R_com``), so no edge can violate the radio model;
* a node is only ever linked under a parent that is itself attached to the
  sink, so no move can orphan anything;
* removing relays is a subtree *cut*, and cutting a subtree from a tree always
  leaves a tree.

The operators are the node-depth-encoding pair from the forest-EA literature —
**PAO** (preserve ancestor) and **CAO** (change ancestor) — plus the two moves
P2 additionally needs because its relay set is not fixed: ``grow`` activates a
candidate, ``prune`` deactivates a subtree.  Deactivated relays stay in the
forest as detached fragments, so activation and deactivation are the same
cut/link primitives the structure already provides.

Everything is written against :class:`RootedTreeBackend`, so it runs on
:class:`TwoLevelTree` or :class:`ParentArrayTree` unchanged.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from random import Random
from typing import Callable, Optional, Sequence

from .rooted_tree import SINK_NODE
from .scenario import ScenarioTopology
from .two_level_tree import TwoLevelTree

log = logging.getLogger(__name__)

#: Sentinel stored in a serialised parent array for a candidate outside the
#: tree. Distinct from ``SINK_NODE`` (-1), which means "child of the sink", so
#: an inactive relay and a first-hop relay can never be confused on restore.
INACTIVE: int = -2

TreeFactory = Callable[[int], "TwoLevelTree"]


@dataclass(frozen=True, slots=True)
class CoverageModel:
    """Trajectory-coverage view the growth operators optimise against.

    Kept as a small value object so the operators never import the P2 adapter:
    ``cover_bits[j]`` is the bitset of sampled trajectory points candidate ``j``
    covers, and ``n_points`` is how many there are in total.
    """

    cover_bits: Sequence[int]
    n_points: int
    threshold: float = 0.0

    def covered_bits(self, active: Sequence[int]) -> int:
        bits = 0
        for node in active:
            bits |= self.cover_bits[node]
        return bits

    def score(self, active: Sequence[int]) -> float:
        """Coverage percentage of an active set, in ``[0, 100]``."""
        if self.n_points <= 0:
            return 100.0
        return 100.0 * self.covered_bits(active).bit_count() / self.n_points

    def required_points(self) -> int:
        """How many sampled points must be covered to satisfy the threshold."""
        return math.ceil(self.threshold / 100.0 * self.n_points) if self.n_points else 0


def mask_from_tree(tree, n_candidates: int) -> list[int]:
    """Binary mask of the relays currently attached to the sink.

    Detached fragments are *inactive by definition*, which is what lets the
    operators express "deactivate a relay" as a subtree cut.
    """
    mask = [0] * n_candidates
    for node in tree.attached_nodes():
        if node != SINK_NODE:
            mask[node] = 1
    return mask


class TreeOperators:
    """Feasibility-preserving variation operators for tree-encoded P2."""

    def __init__(
        self,
        scenario: ScenarioTopology,
        rng: Random,
        coverage: Optional[CoverageModel] = None,
        tree_factory: Optional[TreeFactory] = None,
    ) -> None:
        self.scenario = scenario
        self.rng = rng
        self.coverage = coverage
        self._tree_factory: TreeFactory = tree_factory or (lambda root: TwoLevelTree(root))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def new_tree(self):
        return self._tree_factory(SINK_NODE)

    def tree_from_mask(self, mask: Sequence[int], parents: Optional[Sequence[int]] = None):
        """Rebuild a tree from a mask, honouring a stored parent array.

        ``parents[i]`` is the parent of candidate ``i`` (``SINK_NODE`` for a
        child of the sink).  When the array is missing or inconsistent — a
        chromosome restored from an older document, or one produced by the
        mask-encoded variant — the canonical shortest-path tree is rebuilt
        instead, so the two encodings stay interoperable.
        """
        active = [i for i, bit in enumerate(mask) if bit]
        if parents is not None and self._parents_are_valid(active, parents):
            tree = self.new_tree()
            for node in active:
                tree.add_node(node, None)
            for node in sorted(active, key=lambda n: self._parent_depth(n, parents)):
                tree.link(node, parents[node])
            return tree

        from .builder import build_sink_rooted_tree

        canonical = build_sink_rooted_tree(self.scenario, mask)
        tree = self.new_tree()
        for node in active:
            tree.add_node(node, None)
        for node in sorted(active, key=lambda n: canonical.depth(n) if canonical.depth(n) >= 0 else 10**9):
            parent = canonical.parent(node)
            if parent is not None:
                tree.link(node, parent)
        return tree

    def _parents_are_valid(self, active: Sequence[int], parents: Sequence[int]) -> bool:
        """Whether ``parents`` really encodes a sink-rooted tree over ``active``."""
        if len(parents) != self.scenario.n_candidates:
            return False
        active_set = set(active)
        for node in active:
            parent = parents[node]
            if parent == SINK_NODE:
                if node not in self.scenario.sink_neighbours:
                    return False
                continue
            if parent not in active_set or parent not in self.scenario.adjacency[node]:
                return False
        # every active node must reach the sink without cycling
        for node in active:
            seen: set[int] = set()
            cur = node
            while cur != SINK_NODE:
                if cur in seen:
                    return False
                seen.add(cur)
                cur = parents[cur]
        return True

    def _parent_depth(self, node: int, parents: Sequence[int]) -> int:
        depth = 0
        cur = node
        while cur != SINK_NODE and depth <= len(parents):
            cur = parents[cur]
            depth += 1
        return depth

    def parent_array(self, tree, n_candidates: int) -> list[int]:
        """Serialisable parent array; ``-2`` marks an inactive candidate."""
        parents = [INACTIVE] * n_candidates
        for node in tree.attached_nodes():
            if node == SINK_NODE:
                continue
            parent = tree.parent(node)
            parents[node] = SINK_NODE if parent is None else parent
        return parents

    # ------------------------------------------------------------------
    # Admissibility
    # ------------------------------------------------------------------
    def admissible_parents(self, tree, node: int) -> list[int]:
        """Attached nodes (sink included) that ``node`` can legally hang from."""
        out: list[int] = []
        if node in self.scenario.sink_neighbours:
            out.append(SINK_NODE)
        for neighbour in self.scenario.adjacency[node]:
            if neighbour in tree and tree.is_connected_to_root(neighbour):
                out.append(neighbour)
        return out

    def frontier(self, tree) -> list[int]:
        """Inactive candidates that could be activated by a single link."""
        attached = set(tree.attached_nodes())
        out: list[int] = []
        for candidate in range(self.scenario.n_candidates):
            if candidate in attached:
                continue
            if candidate in self.scenario.sink_neighbours or any(
                neighbour in attached for neighbour in self.scenario.adjacency[candidate]
            ):
                out.append(candidate)
        return out

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------
    def grow(self, tree, count: int = 1) -> int:
        """Activate up to ``count`` candidates as new leaves. Always valid."""
        added = 0
        for _ in range(count):
            options = self.frontier(tree)
            if not options:
                break
            node = self.rng.choice(options)
            parents = self.admissible_parents(tree, node)
            if not parents:
                continue
            if node not in tree:
                tree.add_node(node, None)
            tree.link(node, self.rng.choice(parents))
            added += 1
        return added

    def prune(self, tree, count: int = 1) -> int:
        """Deactivate up to ``count`` subtrees by cutting them off the sink."""
        removed = 0
        for _ in range(count):
            options = [n for n in tree.attached_nodes() if n != SINK_NODE]
            if not options:
                break
            tree.cut_subtree(self.rng.choice(options))
            removed += 1
        return removed

    def pao(self, tree) -> bool:
        """Preserve Ancestor Operator: re-hang a subtree under a new parent.

        The subtree keeps its internal orientation; only the edge joining it to
        the rest of the tree changes.
        """
        movable = [n for n in tree.attached_nodes() if n != SINK_NODE]
        self.rng.shuffle(movable)
        for node in movable:
            original = tree.parent(node)
            tree.cut_subtree(node)
            options = [p for p in self.admissible_parents(tree, node) if p != original]
            if options:
                tree.link(node, self.rng.choice(options))
                return True
            tree.link(node, original)  # nothing better available: restore
        return False

    def cao(self, tree) -> bool:
        """Change Ancestor Operator: re-root a subtree, then re-hang it.

        The path from the new root up to the old subtree root is inverted —
        every inverted edge already existed, so the radio model still holds.
        """
        movable = [n for n in tree.attached_nodes() if n != SINK_NODE]
        self.rng.shuffle(movable)
        for node in movable:
            original = tree.parent(node)
            fragment = tree.cut_subtree(node)
            if len(fragment) < 2:
                tree.link(node, original)
                continue

            candidates = [n for n in fragment if n != node]
            self.rng.shuffle(candidates)
            for new_root in candidates:
                tree.reroot_component(new_root)
                options = self.admissible_parents(tree, new_root)
                if options:
                    tree.link(new_root, self.rng.choice(options))
                    return True
                tree.reroot_component(node)  # undo and try the next one
            tree.link(node, original)
        return False

    def transplant(self, tree, donor) -> bool:
        """Graft a whole subtree of ``donor`` into ``tree``.

        The donor block keeps its internal structure — that is what makes this
        a *structural* crossover rather than a random bit mix — and it is only
        attached where the radio model allows, so the child needs no repair.
        """
        donor_nodes = [n for n in donor.attached_nodes() if n != SINK_NODE]
        if not donor_nodes:
            return False

        block_root = self.rng.choice(donor_nodes)
        block = donor.subtree_nodes(block_root)  # preorder: parents precede children
        donor_parent = {n: donor.parent(n) for n in block}

        # Detach anything the block will replace, so no node is claimed twice.
        for node in block:
            if node in tree and tree.is_connected_to_root(node):
                tree.cut_subtree(node)

        options = self.admissible_parents(tree, block_root)
        if not options:
            return False

        for node in block:
            if node not in tree:
                tree.add_node(node, None)
        tree.link(block_root, self.rng.choice(options))
        for node in block[1:]:
            tree.link(node, donor_parent[node])
        return True

    # ------------------------------------------------------------------
    # Coverage-driven growth (replaces the greedy coverage *repair*)
    # ------------------------------------------------------------------
    def grow_to_coverage(self, tree, budget: int = 8) -> int:
        """Extend the tree until the coverage threshold is met.

        Same greedy set-cover as the mask-encoded variant, but restricted to
        the *admissible frontier*: each activation is a legal link, so unlike
        ``greedy_coverage_repair_mask`` it can never require a follow-up
        connectivity repair.
        """
        model = self.coverage
        if model is None or model.n_points <= 0 or budget <= 0:
            return 0

        active = [n for n in tree.attached_nodes() if n != SINK_NODE]
        covered = model.covered_bits(active)
        required = model.required_points()
        added = 0

        for _ in range(budget):
            if covered.bit_count() >= required:
                break
            best_node = -1
            best_gain = 0
            best_parents: list[int] = []
            for node in self.frontier(tree):
                gain = (model.cover_bits[node] & ~covered).bit_count()
                if gain > best_gain:
                    parents = self.admissible_parents(tree, node)
                    if parents:
                        best_gain, best_node, best_parents = gain, node, parents
            if best_node < 0:
                break
            if best_node not in tree:
                tree.add_node(best_node, None)
            tree.link(best_node, self.rng.choice(best_parents))
            covered |= model.cover_bits[best_node]
            added += 1
        return added

    # ------------------------------------------------------------------
    def random_tree(self, max_relays: Optional[int] = None):
        """Grow a random feasible tree from the sink outwards."""
        tree = self.new_tree()
        limit = max_relays or self.scenario.n_candidates
        target = self.rng.randint(1, max(1, limit))
        self.grow(tree, count=target)
        self.grow_to_coverage(tree, budget=self.scenario.n_candidates)
        return tree
