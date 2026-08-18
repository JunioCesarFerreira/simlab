"""Structure-driven connectivity repair for the topology-aware P2 variant.

Where the base P2 adapter repairs a chromosome with a purely global BFS
(``lib.util.connectivity.repair_connectivity_to_sink``), this pass works from
the sink-rooted tree:

1. build the tree and read off which relays lost their path to the sink;
2. consider only candidates able to form a valid bridge (each activated
   candidate is within ``R_com`` of its predecessor by construction);
3. order those candidates by the heuristic score

       H(u, v) = w1 * I_v + w2 * (1 - d_norm(u, v)) + w3 * Q_v - w4 * C_v

   turned into a strictly positive Dijkstra cost, so the cheapest bridge is the
   one that activates the fewest / most valuable relays;
4. activate that bridge and re-attach the subtree to the main tree.

Without routing history ``I_v = 0`` and the score degrades gracefully to pure
topology.  Every pass is bounded by ``max_iterations``; components that cannot
be bridged at all are deactivated rather than left dangling, so the returned
mask always satisfies the P2 invariant "every active relay reaches the sink".
"""
from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from .builder import (
    DEFAULT_TREE_COST,
    TreeCostWeights,
    active_components,
    build_sink_rooted_tree,
)
from .rooted_tree import SINK_NODE, ParentArrayTree
from .routing import RoutingKnowledge
from .scenario import ScenarioTopology

log = logging.getLogger(__name__)

# Cost of traversing an already-active relay: strictly positive (Dijkstra
# requires non-negative edges and benefits from a total order) but negligible
# next to the cost of activating a new one.
_TRAVERSAL_COST = 1e-9


@dataclass(frozen=True, slots=True)
class RepairWeights:
    """Weights of the bridging score ``H(u, v)``."""

    routing_importance_weight: float = 1.0   # w1 - I_v
    distance_weight: float = 1.0             # w2 - 1 - d_norm(u, v)
    structural_quality_weight: float = 1.0   # w3 - Q_v
    relay_cost_weight: float = 1.0           # w4 - C_v (uniform activation cost)

    @property
    def score_span(self) -> float:
        """Upper bound of ``H`` used to turn the score into a positive cost."""
        return (
            self.routing_importance_weight
            + self.distance_weight
            + self.structural_quality_weight
        )


DEFAULT_REPAIR_WEIGHTS = RepairWeights()


@dataclass(slots=True)
class RepairResult:
    """Outcome of one structural repair pass."""

    mask: list[int]
    tree: ParentArrayTree
    feasible: bool
    activated: tuple[int, ...] = ()
    deactivated: tuple[int, ...] = ()
    iterations: int = 0
    reason: Optional[str] = None

    @property
    def changed(self) -> bool:
        return bool(self.activated or self.deactivated)


class TopologyRepair:
    """Sink-oriented structural repair of a P2 binary mask."""

    def __init__(
        self,
        scenario: ScenarioTopology,
        weights: RepairWeights = DEFAULT_REPAIR_WEIGHTS,
        tree_cost: TreeCostWeights = DEFAULT_TREE_COST,
        routing: Optional[RoutingKnowledge] = None,
        max_iterations: int = 32,
    ) -> None:
        self.scenario = scenario
        self.weights = weights
        self.tree_cost = tree_cost
        self.routing = routing
        self.max_iterations = max(1, int(max_iterations))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_tree(self, mask: Sequence[int]) -> ParentArrayTree:
        """Sink-rooted tree of ``mask`` (no repair)."""
        return build_sink_rooted_tree(self.scenario, mask, self.tree_cost, self.routing)

    def repair(self, mask: Sequence[int]) -> RepairResult:
        """Reconnect every active relay of ``mask`` to the sink."""
        out = [int(bit) for bit in mask]
        activated: list[int] = []
        deactivated: list[int] = []
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):
            tree = self.build_tree(out)
            detached = tree.detached_nodes()
            if not detached:
                return RepairResult(
                    mask=out,
                    tree=tree,
                    feasible=True,
                    activated=tuple(activated),
                    deactivated=tuple(deactivated),
                    iterations=iterations,
                    reason="dropped_unreachable" if deactivated else None,
                )

            attached = [n for n in tree.attached_nodes() if n != SINK_NODE]
            bridge = self._cheapest_bridge(out, attached, set(detached))
            if bridge is None:
                # No admissible bridge exists for the closest component: the
                # relays are physically unreachable from the sink, so keeping
                # them active would violate the P2 connectivity invariant.
                dropped = self._drop_unreachable(out, detached)
                if not dropped:
                    return RepairResult(
                        mask=out,
                        tree=tree,
                        feasible=False,
                        activated=tuple(activated),
                        deactivated=tuple(deactivated),
                        iterations=iterations,
                        reason="unreachable_component",
                    )
                deactivated.extend(dropped)
                continue

            newly_active = [node for node in bridge if not out[node]]
            for node in newly_active:
                out[node] = 1
            activated.extend(newly_active)

        tree = self.build_tree(out)
        log.warning(
            "[P2-topology] Repair budget exhausted after %d iterations; "
            "%d relay(s) still detached.",
            self.max_iterations,
            len(tree.detached_nodes()),
        )
        return RepairResult(
            mask=out,
            tree=tree,
            feasible=not tree.detached_nodes(),
            activated=tuple(activated),
            deactivated=tuple(deactivated),
            iterations=iterations,
            reason="iteration_limit",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _bridge_score(self, parent: int, node: int, importance: dict[int, float]) -> float:
        """``H(u, v)`` — higher means a better bridging candidate."""
        w = self.weights
        i_v = importance.get(node, 0.0)
        d = self.scenario.normalized_distance(node, parent)
        q = self.scenario.structural_quality(node)
        return (
            w.routing_importance_weight * i_v
            + w.distance_weight * (1.0 - d)
            + w.structural_quality_weight * q
            - w.relay_cost_weight * 1.0
        )

    def _activation_cost(self, parent: int, node: int, importance: dict[int, float]) -> float:
        """Strictly positive Dijkstra cost derived from ``H``."""
        return self.weights.score_span - self._bridge_score(parent, node, importance) + _TRAVERSAL_COST

    def _cheapest_bridge(
        self,
        mask: Sequence[int],
        attached: Sequence[int],
        detached: set[int],
    ) -> Optional[list[int]]:
        """Cheapest path from the sink component to any detached relay.

        The *whole* sink component (the sink plus every relay already attached
        to it) is a single zero-distance source, so the search returns the
        cheapest bridge from anywhere on the existing tree.  Returns the node
        sequence of the bridge — source excluded, target included — or ``None``
        when no detached relay is reachable through the candidate graph.
        """
        importance = self.routing.node_importance_map() if self.routing is not None else {}
        scenario = self.scenario

        dist: dict[int, float] = {}
        prev: dict[int, Optional[int]] = {}
        heap: list[tuple[float, int]] = []

        sources: set[int] = {SINK_NODE, *attached}
        for src in sorted(sources):
            dist[src] = 0.0
            prev[src] = None
            heapq.heappush(heap, (0.0, src))

        settled: set[int] = set()
        target: Optional[int] = None
        while heap:
            cost, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled.add(node)
            if node in detached:
                target = node
                break
            for nb in scenario.neighbours(node):
                if nb in settled or nb in sources:
                    continue
                step = _TRAVERSAL_COST if mask[nb] else self._activation_cost(node, nb, importance)
                candidate_cost = cost + step
                if candidate_cost < dist.get(nb, float("inf")):
                    dist[nb] = candidate_cost
                    prev[nb] = node
                    heapq.heappush(heap, (candidate_cost, nb))

        if target is None:
            return None

        chain: list[int] = []
        cur: Optional[int] = target
        limit = len(mask) + 2
        while cur is not None:
            chain.append(cur)
            if len(chain) > limit:  # defensive: a corrupt predecessor chain
                log.error("[P2-topology] Bridge reconstruction exceeded %d hops.", limit)
                return None
            cur = prev.get(cur)
        chain.reverse()
        return chain[1:]  # drop the source: it is already connected

    def _drop_unreachable(self, mask: list[int], detached: Sequence[int]) -> list[int]:
        """Deactivate the whole active component of the first detached relay."""
        components = active_components(self.scenario, mask, detached)
        if not components:
            return []
        component = components[0]
        for node in component:
            mask[node] = 0
        log.debug(
            "[P2-topology] Dropped unreachable component %s (no admissible bridge).",
            component,
        )
        return component
