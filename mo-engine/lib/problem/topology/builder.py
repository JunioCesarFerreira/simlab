"""Construction of the sink-rooted structural tree of a P2 individual.

The tree spans the *active* relays of a chromosome plus the sink.  When a node
has several admissible parents, the cheapest one under a configurable
structural cost wins:

    c(i, j) = w_d * d_norm(i, j) ** 2  +  w_r * (1 - r_ij)

With the default weights (``w_r = 0``) this degenerates to the plain squared
distance required by the design notes, so the builder works with pure
topological information.  Once routing statistics exist, raising
``routing_importance_weight`` pulls the tree towards historically important
links without ever making the builder *depend* on that history.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional, Sequence

from .rooted_tree import SINK_NODE, ParentArrayTree
from .routing import RoutingKnowledge
from .scenario import ScenarioTopology


@dataclass(frozen=True, slots=True)
class TreeCostWeights:
    """Weights of the structural parent-selection cost."""

    distance_weight: float = 1.0
    routing_importance_weight: float = 0.0


DEFAULT_TREE_COST = TreeCostWeights()


def edge_cost(
    scenario: ScenarioTopology,
    parent: int,
    child: int,
    weights: TreeCostWeights = DEFAULT_TREE_COST,
    routing: Optional[RoutingKnowledge] = None,
) -> float:
    """Structural cost of using ``parent`` as the parent of ``child``."""
    d = scenario.normalized_distance(child, parent)
    cost = weights.distance_weight * d * d
    if weights.routing_importance_weight and routing is not None:
        r = routing.importance(child, parent)
        cost += weights.routing_importance_weight * (1.0 - r)
    return cost


def build_sink_rooted_tree(
    scenario: ScenarioTopology,
    mask: Sequence[int],
    weights: TreeCostWeights = DEFAULT_TREE_COST,
    routing: Optional[RoutingKnowledge] = None,
) -> ParentArrayTree:
    """Return the minimum-cost sink-rooted tree over the active relays.

    Relays that cannot reach the sink through other *active* relays are kept in
    the tree as **detached** nodes, which is exactly the signal the repair pass
    consumes.  The construction is a Dijkstra shortest-path tree with
    deterministic tie-breaking on the candidate index, so the same
    ``(scenario, mask)`` always yields the same tree.
    """
    active = [i for i, bit in enumerate(mask) if bit]
    tree = ParentArrayTree(SINK_NODE)
    for node in active:
        tree.add_node(node, None)
    if not active:
        return tree

    active_set = set(active)
    dist: dict[int, float] = {}
    best_parent: dict[int, int] = {}
    heap: list[tuple[float, int, int]] = []

    for node in sorted(scenario.sink_neighbours & active_set):
        c = edge_cost(scenario, SINK_NODE, node, weights, routing)
        dist[node] = c
        best_parent[node] = SINK_NODE
        heapq.heappush(heap, (c, node, SINK_NODE))

    settled: set[int] = set()
    while heap:
        cost, node, parent = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        best_parent[node] = parent
        for nb in scenario.adjacency[node]:
            if nb not in active_set or nb in settled:
                continue
            c = cost + edge_cost(scenario, node, nb, weights, routing)
            if c < dist.get(nb, float("inf")):
                dist[nb] = c
                heapq.heappush(heap, (c, nb, node))

    # Link in increasing depth order so every parent is attached first.
    for node in sorted(settled, key=lambda n: (dist[n], n)):
        tree.link(node, best_parent[node])
    return tree


def active_components(
    scenario: ScenarioTopology, mask: Sequence[int], nodes: Sequence[int]
) -> list[list[int]]:
    """Connected components of ``nodes`` inside the active sub-graph."""
    remaining = {n for n in nodes if mask[n]}
    components: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.discard(seed)
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in scenario.adjacency[cur]:
                if nb in remaining:
                    remaining.discard(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    return components
