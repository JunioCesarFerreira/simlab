"""The two-level rooted-forest structure.

Two things are pinned here: that the segmented preorder sequence stays a
*correct* forest under arbitrary cut/link sequences, and that it stays
*balanced*, since the whole reason to prefer it over a flat parent array is
the O(sqrt(n)) splice.
"""
import math
import random

import pytest

from lib.problem.topology import SINK_NODE, ParentArrayTree, RootedTreeBackend, TwoLevelTree


def _chain(cls):
    #   sink -> 0 -> 1 -> 2      sink -> 3
    tree = cls(SINK_NODE)
    tree.add_node(0, SINK_NODE)
    tree.add_node(1, 0)
    tree.add_node(2, 1)
    tree.add_node(3, SINK_NODE)
    return tree


BACKENDS = [ParentArrayTree, TwoLevelTree]


# ---------------------------------------------------------------------------
# Backend parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", BACKENDS)
class TestBackendParity:
    def test_satisfies_the_protocol(self, cls):
        assert isinstance(cls(SINK_NODE), RootedTreeBackend)

    def test_depth_and_path(self, cls):
        tree = _chain(cls)
        assert tree.depth(SINK_NODE) == 0
        assert tree.depth(2) == 3
        assert tree.path_to_root(2) == [2, 1, 0, SINK_NODE]
        assert tree.children(0) == (1,)
        assert set(tree.leaves()) == {2, 3}

    def test_cut_detaches_only_the_expected_subtree(self, cls):
        tree = _chain(cls)
        detached = tree.cut_subtree(1)

        assert set(detached) == {1, 2}
        assert not tree.is_connected_to_root(1)
        assert not tree.is_connected_to_root(2)
        assert tree.is_connected_to_root(0)
        assert tree.is_connected_to_root(3)
        assert set(tree.detached_nodes()) == {1, 2}
        assert tree.depth(1) == -1

    def test_link_reconnects_the_subtree(self, cls):
        tree = _chain(cls)
        tree.cut_subtree(1)
        tree.link(1, 3)

        assert tree.parent(1) == 3
        assert tree.depth(1) == 2
        assert tree.depth(2) == 3
        assert tree.detached_nodes() == []

    def test_link_rejects_cycles(self, cls):
        tree = _chain(cls)
        with pytest.raises(ValueError):
            tree.link(0, 2)

    def test_root_cannot_be_cut(self, cls):
        with pytest.raises(ValueError):
            _chain(cls).cut_subtree(SINK_NODE)

    def test_unknown_node_is_rejected(self, cls):
        with pytest.raises(KeyError):
            _chain(cls).parent(99)


def test_backends_agree_under_a_random_operation_sequence():
    """Differential test: the two backends must be indistinguishable."""
    rng = random.Random(20240607)
    flat = ParentArrayTree(SINK_NODE)
    twol = TwoLevelTree(SINK_NODE)

    for node in range(1, 40):
        parent = rng.choice([SINK_NODE, *range(1, node)]) if node > 1 else SINK_NODE
        flat.add_node(node, parent)
        twol.add_node(node, parent)

    for _ in range(300):
        candidates = [n for n in flat.nodes() if n != SINK_NODE]
        node = rng.choice(candidates)
        if rng.random() < 0.5:
            flat.cut_subtree(node)
            twol.cut_subtree(node)
        else:
            forbidden = set(flat.subtree_nodes(node))
            targets = [n for n in flat.nodes() if n not in forbidden]
            if not targets:
                continue
            target = rng.choice(targets)
            flat.link(node, target)
            twol.link(node, target)

        assert {n: flat.parent(n) for n in flat.nodes()} == {n: twol.parent(n) for n in twol.nodes()}
        assert set(flat.attached_nodes()) == set(twol.attached_nodes())
        assert set(flat.detached_nodes()) == set(twol.detached_nodes())
        assert {n: flat.depth(n) for n in flat.nodes()} == {n: twol.depth(n) for n in twol.nodes()}


# ---------------------------------------------------------------------------
# Two-level specifics
# ---------------------------------------------------------------------------
class TestSequenceInvariants:
    def _random_tree(self, n: int, seed: int) -> TwoLevelTree:
        rng = random.Random(seed)
        tree = TwoLevelTree(SINK_NODE)
        for node in range(n):
            parent = rng.choice([SINK_NODE, *range(node)]) if node else SINK_NODE
            tree.add_node(node, parent)
        return tree

    def test_a_subtree_is_a_contiguous_range(self, ):
        tree = self._random_tree(60, seed=3)
        order = tree.nodes()
        position = {node: i for i, node in enumerate(order)}

        for node in order:
            block = tree.subtree_nodes(node)
            indices = sorted(position[n] for n in block)
            assert indices == list(range(indices[0], indices[0] + len(block))), (
                f"subtree of {node} is not contiguous in the preorder sequence"
            )

    def test_preorder_places_every_parent_before_its_children(self):
        tree = self._random_tree(60, seed=11)
        position = {node: i for i, node in enumerate(tree.nodes())}
        for node in tree.attached_nodes():
            for child in tree.children(node):
                assert position[node] < position[child]

    def test_sequence_index_and_precedes_are_consistent(self):
        tree = self._random_tree(50, seed=5)
        order = tree.nodes()
        for i, node in enumerate(order):
            assert tree.sequence_index(node) == i
        for a, b in zip(order, order[1:]):
            assert tree.precedes(a, b)
            assert not tree.precedes(b, a)

    def test_segments_stay_around_sqrt_n(self):
        rng = random.Random(99)
        tree = self._random_tree(200, seed=7)
        for _ in range(200):
            node = rng.choice([n for n in tree.nodes() if n != SINK_NODE])
            if rng.random() < 0.5:
                tree.cut_subtree(node)
            else:
                forbidden = set(tree.subtree_nodes(node))
                targets = [n for n in tree.nodes() if n not in forbidden]
                if targets:
                    tree.link(node, rng.choice(targets))

        target = math.isqrt(len(tree))
        assert tree.segment_count <= 3 * target + 1
        assert max(tree.segment_sizes) <= 3 * target
        assert sum(tree.segment_sizes) == len(tree)

    def test_rebalance_preserves_the_sequence(self):
        tree = self._random_tree(120, seed=13)
        before = tree.nodes()
        tree.rebalance()
        assert tree.nodes() == before
        assert sum(tree.segment_sizes) == len(tree)
        assert all(tree.sequence_index(n) == i for i, n in enumerate(tree.nodes()))


class TestRerooting:
    def _fragment(self) -> TwoLevelTree:
        #   sink -> 0 -> 1 -> 2 -> 3, then detach the 1..3 fragment
        tree = TwoLevelTree(SINK_NODE)
        tree.add_node(0, SINK_NODE)
        tree.add_node(1, 0)
        tree.add_node(2, 1)
        tree.add_node(3, 2)
        tree.cut_subtree(1)
        return tree

    def test_reroot_inverts_exactly_the_path(self):
        tree = self._fragment()
        tree.reroot_component(3)

        assert tree.parent(3) is None
        assert tree.parent(2) == 3
        assert tree.parent(1) == 2
        assert set(tree.detached_nodes()) == {1, 2, 3}
        assert tree.attached_nodes() == [SINK_NODE, 0]

    def test_rerooted_fragment_can_be_relinked(self):
        tree = self._fragment()
        tree.reroot_component(3)
        tree.link(3, 0)

        assert tree.detached_nodes() == []
        assert tree.depth(3) == 2
        assert tree.depth(1) == 4
        assert tree.path_to_root(1) == [1, 2, 3, 0, SINK_NODE]

    def test_reroot_is_an_involution_on_the_original_root(self):
        tree = self._fragment()
        before = {n: tree.parent(n) for n in tree.nodes()}
        tree.reroot_component(3)
        tree.reroot_component(1)
        assert {n: tree.parent(n) for n in tree.nodes()} == before

    def test_attached_component_cannot_be_rerooted(self):
        tree = _chain(TwoLevelTree)
        with pytest.raises(ValueError):
            tree.reroot_component(2)


def test_from_parent_map_round_trips():
    tree = _chain(TwoLevelTree)
    tree.cut_subtree(1)
    rebuilt = TwoLevelTree.from_parent_map(tree.parent_map(), SINK_NODE)

    assert {n: rebuilt.parent(n) for n in rebuilt.nodes()} == {n: tree.parent(n) for n in tree.nodes()}
    assert set(rebuilt.detached_nodes()) == set(tree.detached_nodes())


def test_deep_chain_does_not_recurse():
    """A degenerate 1-child-per-node tree must not blow the interpreter stack."""
    parent_map = {0: SINK_NODE}
    for node in range(1, 2000):
        parent_map[node] = node - 1
    tree = TwoLevelTree.from_parent_map(parent_map, SINK_NODE)

    assert tree.depth(1999) == 2000
    assert len(tree.subtree_nodes(0)) == 2000
