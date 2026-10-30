"""MST is an alternative structural skeleton, not a routing-performance model."""
from itertools import combinations
import random

import pytest

from lib.problem.topology import (
    SINK_NODE, ScenarioTopology, build_sink_rooted_mst,
    build_sink_rooted_tree, edge_cost,
)


def _scenario(candidates, radius=10.0):
    return ScenarioTopology(candidates, (0.0, 0.0), radius, [])


def _cost(scenario, tree):
    return sum(
        edge_cost(scenario, tree.parent(n), n)
        for n in tree.attached_nodes() if n != SINK_NODE
    )


def test_mst_can_reduce_total_edge_cost_while_increasing_sink_path_cost():
    scenario = _scenario([(3.0, 0.0), (2.0, 3.0)], radius=4.0)
    shortest = build_sink_rooted_tree(scenario, [1, 1])
    mst = build_sink_rooted_mst(scenario, [1, 1])
    assert shortest.parent(1) == SINK_NODE
    assert mst.parent(1) == 0
    assert _cost(scenario, mst) < _cost(scenario, shortest)
    assert mst.depth(1) > shortest.depth(1)
    assert (
        edge_cost(scenario, SINK_NODE, 0) + edge_cost(scenario, 0, 1)
        > edge_cost(scenario, SINK_NODE, 1)
    )


def test_unreachable_nodes_stay_visible_and_inactive_bridges_are_not_added():
    scenario = _scenario([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)], radius=1.1)
    mask = [1, 0, 1]
    mst = build_sink_rooted_mst(scenario, mask)
    assert mst.attached_nodes() == [SINK_NODE, 0]
    assert mst.detached_nodes() == [2]
    assert 1 not in mst
    assert mask == [1, 0, 1]


def test_empty_mask_and_zero_cost_ties_are_deterministic_and_acyclic():
    scenario = _scenario([(0.0, 0.0)] * 4)
    assert build_sink_rooted_mst(scenario, [0] * 4).nodes() == (SINK_NODE,)
    first = build_sink_rooted_mst(scenario, [1] * 4)
    second = build_sink_rooted_mst(scenario, [1] * 4)
    assert first.detached_nodes() == []
    for node in range(4):
        assert first.parent(node) == second.parent(node)
        assert first.path_to_root(node)[-1] == SINK_NODE


def test_mask_length_is_checked():
    with pytest.raises(ValueError, match="Mask length"):
        build_sink_rooted_mst(_scenario([(1.0, 0.0)]), [])


@pytest.mark.parametrize("seed", range(6))
def test_total_weight_matches_exhaustive_spanning_tree_search(seed):
    # Independent optimality oracle: enumerate every 4-edge tree on 5 nodes.
    rng = random.Random(seed)
    scenario = _scenario([(rng.random(), rng.random()) for _ in range(4)])
    nodes = [SINK_NODE, 0, 1, 2, 3]
    optimum = float("inf")
    for edges in combinations(list(combinations(nodes, 2)), len(nodes) - 1):
        reached = {SINK_NODE}
        for _ in nodes:
            for a, b in edges:
                if a in reached or b in reached:
                    reached.update((a, b))
        if len(reached) == len(nodes):
            optimum = min(optimum, sum(edge_cost(scenario, a, b) for a, b in edges))

    mst = build_sink_rooted_mst(scenario, [1] * 4)
    assert mst.detached_nodes() == []
    assert _cost(scenario, mst) == pytest.approx(optimum)
