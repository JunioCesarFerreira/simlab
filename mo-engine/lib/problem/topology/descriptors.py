"""Cheap structural descriptors ``phi(x)`` of a topology-aware P2 individual.

The descriptors are the only thing the objective estimator ever sees, so they
must be (a) cheap — no simulation, no heavy geometry, everything reads the
pre-computed :class:`ScenarioTopology` caches — and (b) **deterministic for a
given ``scenario + chromosome``**.

That determinism requirement is why the vector fed to the estimator holds only
the *structural* block.  The *historical* block (routing importance, distance
to the nearest evaluated individual) is computed and persisted alongside it —
it drives the novelty metric and the repair heuristic, and it is available to
future estimators — but it is intentionally kept out of the regression input:
its value for a fixed chromosome drifts as the knowledge base grows, which
would silently invalidate every descriptor stored in earlier generations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .rooted_tree import SINK_NODE, ParentArrayTree
from .routing import RoutingKnowledge
from .scenario import ScenarioTopology

#: Names of the structural descriptors, in the exact order of ``vector()``.
STRUCTURAL_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "active_relays",
    "relay_ratio",
    "number_of_edges",
    "mean_degree",
    "min_degree",
    "max_degree",
    "connected_components",
    "sink_reachability_ratio",
    "mean_tree_depth",
    "max_tree_depth",
    "tree_leaves",
    "tree_branching_factor",
    "mean_distance_to_sink",
    "max_distance_to_sink",
    "mean_hop_count",
    "max_hop_count",
    "trajectory_coverage_ratio",
    "minimum_temporal_connectivity",
    "mean_temporal_connectivity",
    "critical_time_slices",
)

#: Names of the history-dependent descriptors (never part of ``vector()``).
HISTORICAL_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "routing_importance_sum",
    "routing_importance_mean",
    "routing_importance_max",
    "nearest_evaluated_hamming_distance",
    "nearest_evaluated_descriptor_distance",
)


@dataclass(frozen=True, slots=True)
class TopologyDescriptors:
    """Structural + historical description of one individual."""

    structural: dict[str, float]
    historical: dict[str, float] = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        """Estimator input: the structural block, in canonical order."""
        return np.asarray(
            [self.structural[name] for name in STRUCTURAL_DESCRIPTOR_NAMES],
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        """Flat, persistence-friendly view of every descriptor."""
        out = dict(self.structural)
        out.update(self.historical)
        return out

    def with_historical(self, **values: float) -> "TopologyDescriptors":
        merged = dict(self.historical)
        merged.update({k: float(v) for k, v in values.items()})
        return TopologyDescriptors(structural=dict(self.structural), historical=merged)


class TopologyDescriptorExtractor:
    """Computes ``phi(x)`` from a binary mask and its sink-rooted tree."""

    def __init__(
        self,
        scenario: ScenarioTopology,
        routing: Optional[RoutingKnowledge] = None,
    ) -> None:
        self.scenario = scenario
        self.routing = routing

    # ------------------------------------------------------------------
    def extract(
        self,
        mask: Sequence[int],
        tree: Optional[ParentArrayTree] = None,
    ) -> TopologyDescriptors:
        """Return the descriptors of the individual encoded by ``mask``."""
        scenario = self.scenario
        active = [i for i, bit in enumerate(mask) if bit]
        n_active = len(active)
        n_candidates = max(1, scenario.n_candidates)

        edges, degrees = self._graph_stats(active)
        reachable, hops = self._reachability(active)
        components = self._component_count(active)

        structural: dict[str, float] = {
            "active_relays": float(n_active),
            "relay_ratio": n_active / n_candidates,
            "number_of_edges": float(edges),
            "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            "min_degree": float(min(degrees)) if degrees else 0.0,
            "max_degree": float(max(degrees)) if degrees else 0.0,
            "connected_components": float(components),
            "sink_reachability_ratio": (len(reachable) / n_active) if n_active else 1.0,
        }

        structural.update(self._tree_stats(tree, mask))
        structural.update(self._distance_stats(active))
        structural.update(
            {
                "mean_hop_count": float(np.mean(list(hops.values()))) if hops else 0.0,
                "max_hop_count": float(max(hops.values())) if hops else 0.0,
            }
        )
        structural.update(self._temporal_stats(reachable))

        return TopologyDescriptors(
            structural=structural,
            historical=self._historical_stats(active),
        )

    # ------------------------------------------------------------------
    # Structural blocks
    # ------------------------------------------------------------------
    def _graph_stats(self, active: Sequence[int]) -> tuple[int, list[int]]:
        """Edge count and per-node degrees of ``G_x`` (active relays + sink)."""
        scenario = self.scenario
        active_set = set(active)
        edges = 0
        degrees: list[int] = []
        for node in active:
            deg = sum(1 for nb in scenario.adjacency[node] if nb in active_set)
            if node in scenario.sink_neighbours:
                deg += 1
            degrees.append(deg)
            edges += deg
        sink_degree = len(scenario.sink_neighbours & active_set)
        degrees.append(sink_degree)
        edges += sink_degree
        return edges // 2, degrees

    def _reachability(self, active: Sequence[int]) -> tuple[set[int], dict[int, int]]:
        """BFS from the sink over ``G_x``: reachable relays and their hop count."""
        scenario = self.scenario
        active_set = set(active)
        hops: dict[int, int] = {}
        queue: deque[int] = deque()
        for node in sorted(scenario.sink_neighbours & active_set):
            hops[node] = 1
            queue.append(node)
        while queue:
            cur = queue.popleft()
            for nb in scenario.adjacency[cur]:
                if nb in active_set and nb not in hops:
                    hops[nb] = hops[cur] + 1
                    queue.append(nb)
        return set(hops), hops

    def _component_count(self, active: Sequence[int]) -> int:
        """Connected components of ``G_x``, sink included."""
        scenario = self.scenario
        remaining = set(active)
        components = 0
        # The sink's own component (empty when no active relay touches it).
        seeds = sorted(scenario.sink_neighbours & remaining)
        components += 1  # the sink always forms at least one component
        stack = list(seeds)
        remaining.difference_update(seeds)
        while stack:
            cur = stack.pop()
            for nb in scenario.adjacency[cur]:
                if nb in remaining:
                    remaining.discard(nb)
                    stack.append(nb)
        while remaining:
            seed = min(remaining)
            remaining.discard(seed)
            stack = [seed]
            components += 1
            while stack:
                cur = stack.pop()
                for nb in scenario.adjacency[cur]:
                    if nb in remaining:
                        remaining.discard(nb)
                        stack.append(nb)
        return components

    def _tree_stats(
        self, tree: Optional[ParentArrayTree], mask: Sequence[int]
    ) -> dict[str, float]:
        if tree is None:
            from .builder import build_sink_rooted_tree

            tree = build_sink_rooted_tree(self.scenario, mask)

        depths = [
            tree.depth(node)
            for node in tree.nodes()
            if node != SINK_NODE and tree.is_connected_to_root(node)
        ]
        leaves = [n for n in tree.leaves() if n != SINK_NODE]
        internal = [
            n
            for n in tree.attached_nodes()
            if tree.children(n)
        ]
        branching = (
            float(np.mean([len(tree.children(n)) for n in internal])) if internal else 0.0
        )
        return {
            "mean_tree_depth": float(np.mean(depths)) if depths else 0.0,
            "max_tree_depth": float(max(depths)) if depths else 0.0,
            "tree_leaves": float(len(leaves)),
            "tree_branching_factor": branching,
        }

    def _distance_stats(self, active: Sequence[int]) -> dict[str, float]:
        scenario = self.scenario
        if not active:
            return {"mean_distance_to_sink": 0.0, "max_distance_to_sink": 0.0}
        d = [scenario.distance_to_sink(i) / scenario.max_distance for i in active]
        return {
            "mean_distance_to_sink": float(np.mean(d)),
            "max_distance_to_sink": float(max(d)),
        }

    def _temporal_stats(self, reachable: set[int]) -> dict[str, float]:
        """Coverage of the mobile fleet by sink-connected relays, over time."""
        scenario = self.scenario
        slices = scenario.time_slices
        if not slices:
            return {
                "trajectory_coverage_ratio": 1.0,
                "minimum_temporal_connectivity": 1.0,
                "mean_temporal_connectivity": 1.0,
                "critical_time_slices": 0.0,
            }

        reachable_bits = 0
        for node in reachable:
            reachable_bits |= 1 << node

        per_slice: list[float] = []
        covered_pairs = 0
        total_pairs = 0
        for slice_ in slices:
            n_nodes = len(slice_.cover_bits)
            if n_nodes == 0:
                continue
            covered = 0
            for bits, sink_ok in zip(slice_.cover_bits, slice_.sink_covered):
                total_pairs += 1
                if sink_ok or (bits & reachable_bits):
                    covered += 1
                    covered_pairs += 1
            per_slice.append(covered / n_nodes)

        if not per_slice:
            return {
                "trajectory_coverage_ratio": 1.0,
                "minimum_temporal_connectivity": 1.0,
                "mean_temporal_connectivity": 1.0,
                "critical_time_slices": 0.0,
            }

        critical = sum(1 for v in per_slice if v < 1.0)
        return {
            "trajectory_coverage_ratio": covered_pairs / total_pairs if total_pairs else 1.0,
            "minimum_temporal_connectivity": float(min(per_slice)),
            "mean_temporal_connectivity": float(np.mean(per_slice)),
            "critical_time_slices": critical / len(per_slice),
        }

    # ------------------------------------------------------------------
    # Historical block
    # ------------------------------------------------------------------
    def _historical_stats(self, active: Sequence[int]) -> dict[str, float]:
        if self.routing is None or self.routing.observation_count == 0 or not active:
            return {
                "routing_importance_sum": 0.0,
                "routing_importance_mean": 0.0,
                "routing_importance_max": 0.0,
            }
        importance = self.routing.node_importance_map()
        values = [importance.get(node, 0.0) for node in active]
        return {
            "routing_importance_sum": float(sum(values)),
            "routing_importance_mean": float(np.mean(values)),
            "routing_importance_max": float(max(values)),
        }
