"""Unit tests for the topology-aware P2 variant.

Covers the structural machinery the adaptive strategy depends on: the rooted
tree, the structure-driven repair and the descriptor vector.  The original P2
adapter is deliberately untouched, so its own tests keep guarding it.
"""
import math
import random

import pytest

from lib.problem.chromosomes import ChromosomeP2
from lib.problem.p2_discrete_mobility import Problem2DiscreteMobilityAdapter
from lib.problem.p2_topology_aware import Problem2TopologyAwareAdapter
from lib.problem.resolve import build_adapter, build_test_adapter
from lib.problem.topology import (
    SINK_NODE,
    ParentArrayTree,
    RepairWeights,
    ScenarioTopology,
    TopologyRepair,
    build_sink_rooted_tree,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _grid_problem(min_coverage: float = 70.0) -> dict:
    """A 5x5 candidate grid, one straight trajectory, sink at the origin."""
    return {
        "name": "problem2_topology_aware",
        "region": [-100.0, -100.0, 100.0, 100.0],
        "sink": (0.0, 0.0),
        "candidates": [
            (float(x * 20 - 40), float(y * 20 - 40))
            for x in range(5) for y in range(5)
        ],
        "mobile_nodes": [
            {
                "path_segments": [("-40 + 80*t", "40")],
                "is_closed": False,
                "is_round_trip": True,
                "speed": 5.0,
                "time_step": 1.0,
            }
        ],
        "radius_of_reach": 30.0,
        "radius_of_inter": 60.0,
        "min_coverage_percentage": min_coverage,
    }


def _line_problem() -> dict:
    """Two disjoint candidate chains leaving the sink; nothing bridges them."""
    return {
        "name": "problem2_topology_aware",
        "region": [-200.0, -200.0, 200.0, 200.0],
        "sink": (0.0, 0.0),
        "candidates": [
            (8.0, 0.0), (16.0, 0.0), (24.0, 0.0), (32.0, 0.0),   # 0..3 (east chain)
            (0.0, 8.0), (0.0, 16.0), (0.0, 24.0), (0.0, 32.0),   # 4..7 (north chain)
            (150.0, 150.0),                                      # 8 (unreachable)
        ],
        "mobile_nodes": [
            {
                "path_segments": [("10", "10")],
                "is_closed": False,
                "is_round_trip": False,
                "speed": 1.0,
                "time_step": 1.0,
            }
        ],
        "radius_of_reach": 10.0,
        "radius_of_inter": 20.0,
        "min_coverage_percentage": 0.0,
    }


def _adapter(problem: dict, seed: int = 11) -> Problem2TopologyAwareAdapter:
    return build_adapter(problem, {"per_gene_prob": 0.1}, random.Random(seed))


# ---------------------------------------------------------------------------
# Registry / compatibility
# ---------------------------------------------------------------------------
def test_registry_resolves_both_p2_variants():
    classic = build_test_adapter({**_grid_problem(), "name": "problem2"})
    aware = build_test_adapter(_grid_problem())
    assert type(classic) is Problem2DiscreteMobilityAdapter
    assert type(aware) is Problem2TopologyAwareAdapter
    # The variant must not leak topology state into the original adapter.
    assert not hasattr(classic, "descriptors")


def test_chromosome_and_encoding_are_unchanged():
    aware = _adapter(_grid_problem())
    chromosome = ChromosomeP2(mac_protocol=0, mask=[1] + [0] * 24)
    encoded = aware.encode_simulation_input(chromosome)
    assert encoded["fixedMotes"][0]["name"] == "sink"
    assert len(encoded["fixedMotes"]) == 2
    assert isinstance(chromosome.mask, list)


# ---------------------------------------------------------------------------
# Rooted tree
# ---------------------------------------------------------------------------
class TestParentArrayTree:
    def _chain(self) -> ParentArrayTree:
        #   sink -> 0 -> 1 -> 2      sink -> 3
        tree = ParentArrayTree(SINK_NODE)
        tree.add_node(0, SINK_NODE)
        tree.add_node(1, 0)
        tree.add_node(2, 1)
        tree.add_node(3, SINK_NODE)
        return tree

    def test_depth_and_path(self):
        tree = self._chain()
        assert tree.depth(SINK_NODE) == 0
        assert tree.depth(2) == 3
        assert tree.path_to_root(2) == [2, 1, 0, SINK_NODE]
        assert tree.children(0) == (1,)
        assert set(tree.leaves()) == {2, 3}

    def test_cut_detaches_only_the_expected_subtree(self):
        tree = self._chain()
        detached = tree.cut_subtree(1)

        assert set(detached) == {1, 2}
        assert not tree.is_connected_to_root(1)
        assert not tree.is_connected_to_root(2)
        # Everything outside the cut subtree keeps its path to the sink.
        assert tree.is_connected_to_root(0)
        assert tree.is_connected_to_root(3)
        assert set(tree.detached_nodes()) == {1, 2}

    def test_link_reconnects_the_subtree(self):
        tree = self._chain()
        tree.cut_subtree(1)
        tree.link(1, 3)

        assert tree.is_connected_to_root(1)
        assert tree.is_connected_to_root(2)
        assert tree.parent(1) == 3
        assert tree.depth(2) == 3
        assert tree.detached_nodes() == []

    def test_link_rejects_cycles(self):
        tree = self._chain()
        with pytest.raises(ValueError):
            tree.link(0, 2)  # 2 is a descendant of 0

    def test_root_cannot_be_cut(self):
        with pytest.raises(ValueError):
            self._chain().cut_subtree(SINK_NODE)


# ---------------------------------------------------------------------------
# Scenario caches
# ---------------------------------------------------------------------------
def test_neighbourhood_cache_matches_direct_computation():
    problem = _grid_problem()
    scenario = _adapter(problem).scenario
    candidates = scenario.candidates
    radius = scenario.radius

    for i, p in enumerate(candidates):
        expected = tuple(
            j for j, q in enumerate(candidates)
            if j != i and math.dist(p, q) <= radius
        )
        assert scenario.adjacency[i] == expected
        assert scenario.degree[i] == len(expected) + (
            1 if math.dist(p, scenario.sink) <= radius else 0
        )

    expected_sink = {
        i for i, p in enumerate(candidates) if math.dist(p, scenario.sink) <= radius
    }
    assert set(scenario.sink_neighbours) == expected_sink


def test_scenario_fingerprint_separates_incompatible_scenarios():
    base = _adapter(_grid_problem())
    same = _adapter(_grid_problem())
    assert base.scenario_fingerprint() == same.scenario_fingerprint()

    wider = _grid_problem()
    wider["radius_of_reach"] = 45.0
    assert _adapter(wider).scenario_fingerprint() != base.scenario_fingerprint()

    # Context outside the geometry also changes the meaning of an objective.
    assert base.scenario_fingerprint({"objectives": ["latency"]}) != base.scenario_fingerprint()


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------
def test_feasible_individual_yields_a_tree_connected_to_the_sink():
    adapter = _adapter(_grid_problem())
    for chromosome in adapter.random_individual_generator(12):
        tree = adapter.build_tree(chromosome)
        active = {i for i, bit in enumerate(chromosome.mask) if bit}
        assert tree.detached_nodes() == []
        for node in active:
            assert tree.is_connected_to_root(node)
            assert tree.depth(node) >= 1
        assert set(tree.nodes()) == active | {SINK_NODE}


def test_tree_edges_respect_the_communication_radius():
    adapter = _adapter(_grid_problem())
    scenario = adapter.scenario
    chromosome = adapter.random_individual_generator(1)[0]
    tree = adapter.build_tree(chromosome)

    for node in tree.nodes():
        parent = tree.parent(node)
        if parent is None:
            continue
        distance = (
            scenario.distance_to_sink(node) if parent == SINK_NODE
            else scenario.distance(node, parent)
        )
        assert distance <= scenario.radius + 1e-9


# ---------------------------------------------------------------------------
# Structural repair
# ---------------------------------------------------------------------------
class TestTopologyRepair:
    def test_repair_reconnects_every_active_relay(self):
        adapter = _adapter(_line_problem())
        # Only the far end of the east chain is active: it needs a bridge.
        result = adapter.structural_repair([0, 0, 0, 1, 0, 0, 0, 0, 0])

        assert result.feasible
        assert result.tree.detached_nodes() == []
        assert result.mask[3] == 1
        assert set(result.activated) == {0, 1, 2}

    def test_repair_activates_the_minimum_reasonable_bridge(self):
        adapter = _adapter(_line_problem())
        result = adapter.structural_repair([0, 0, 0, 1, 0, 0, 0, 1, 0])

        assert result.feasible
        # Two chains, each needing exactly the relays between sink and endpoint.
        assert set(result.activated) == {0, 1, 2, 4, 5, 6}
        assert result.mask[8] == 0

    def test_repair_never_activates_an_unreachable_candidate(self):
        adapter = _adapter(_line_problem())
        scenario = adapter.scenario
        result = adapter.structural_repair([1, 1, 1, 1, 1, 1, 1, 1, 0])

        assert result.mask[8] == 0, "isolated candidate was activated"
        for node in result.activated:
            neighbours = set(scenario.adjacency[node]) | (
                {SINK_NODE} if node in scenario.sink_neighbours else set()
            )
            reachable = {n for n in neighbours if n == SINK_NODE or result.mask[n]}
            assert reachable, f"candidate {node} was activated with no active neighbour"

    def test_unbridgeable_component_is_dropped_and_reported(self):
        adapter = _adapter(_line_problem())
        result = adapter.structural_repair([0, 0, 0, 0, 0, 0, 0, 0, 1])

        assert result.mask[8] == 0
        assert result.deactivated == (8,)
        assert result.reason == "dropped_unreachable"
        assert result.tree.detached_nodes() == []

    def test_individual_with_orphan_relays_is_flagged_infeasible(self):
        adapter = _adapter(_line_problem())
        orphan = ChromosomeP2(mac_protocol=0, mask=[0] * 8 + [1])

        assert not adapter.is_structurally_feasible(orphan)
        penalty = adapter.penalty_objectives(orphan, 3)
        assert penalty is not None and len(penalty) == 3
        assert all(p > 1e9 for p in penalty), "structural penalty must outrank coverage penalty"

    def test_repair_budget_exhaustion_is_signalled(self):
        adapter = _adapter(_line_problem())
        repairer = TopologyRepair(
            scenario=adapter.scenario,
            weights=RepairWeights(),
            routing=adapter.routing_knowledge,
            max_iterations=1,
        )
        # Two independent detached chains need two repair iterations.
        result = repairer.repair([0, 0, 0, 1, 0, 0, 0, 1, 0])

        assert not result.feasible
        assert result.reason == "iteration_limit"

    def test_operators_always_return_connected_offspring(self):
        adapter = _adapter(_grid_problem())
        parents = adapter.random_individual_generator(2)

        for _ in range(15):
            children = adapter.crossover(parents)
            assert len(children) == 2
            for child in children:
                assert build_sink_rooted_tree(adapter.scenario, child.mask).detached_nodes() == []
            mutated = adapter.mutate(children[0])
            assert build_sink_rooted_tree(adapter.scenario, mutated.mask).detached_nodes() == []
            parents = children


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
class TestDescriptors:
    def test_same_chromosome_yields_identical_descriptors(self):
        adapter = _adapter(_grid_problem())
        chromosome = adapter.random_individual_generator(1)[0]

        first = adapter.descriptors(chromosome)
        second = adapter.descriptors(chromosome)
        assert first.structural == second.structural
        assert list(first.vector()) == list(second.vector())

        # ... and across adapter instances built from the same scenario.
        other = _adapter(_grid_problem(), seed=999)
        assert other.descriptors(chromosome).structural == first.structural

    def test_descriptor_vector_has_the_documented_layout(self):
        from lib.problem.topology import STRUCTURAL_DESCRIPTOR_NAMES

        adapter = _adapter(_grid_problem())
        descriptors = adapter.descriptors(adapter.random_individual_generator(1)[0])
        vector = descriptors.vector()

        assert len(vector) == len(STRUCTURAL_DESCRIPTOR_NAMES)
        assert all(math.isfinite(v) for v in vector)
        for i, name in enumerate(STRUCTURAL_DESCRIPTOR_NAMES):
            assert vector[i] == pytest.approx(descriptors.structural[name])

    def test_descriptors_react_to_the_deployed_topology(self):
        adapter = _adapter(_line_problem())
        small = adapter.descriptors(ChromosomeP2(mac_protocol=0, mask=[1, 0, 0, 0, 0, 0, 0, 0, 0]))
        large = adapter.descriptors(ChromosomeP2(mac_protocol=0, mask=[1, 1, 1, 1, 0, 0, 0, 0, 0]))

        assert large.structural["active_relays"] > small.structural["active_relays"]
        assert large.structural["max_tree_depth"] > small.structural["max_tree_depth"]
        assert small.structural["sink_reachability_ratio"] == 1.0

    def test_empty_mask_descriptors_are_well_defined(self):
        adapter = _adapter(_line_problem())
        descriptors = adapter.descriptors(ChromosomeP2(mac_protocol=0, mask=[0] * 9))

        assert descriptors.structural["active_relays"] == 0.0
        assert descriptors.structural["sink_reachability_ratio"] == 1.0
        assert all(math.isfinite(v) for v in descriptors.vector())

    def test_routing_history_only_moves_the_historical_block(self):
        adapter = _adapter(_grid_problem())
        chromosome = adapter.random_individual_generator(1)[0]
        before = adapter.descriptors(chromosome)

        adapter.observe_simulated(chromosome, key="genome-1")
        after = adapter.descriptors(chromosome)

        assert list(before.vector()) == list(after.vector())
        assert after.historical["routing_importance_max"] > 0.0
        assert before.historical["routing_importance_max"] == 0.0


# ---------------------------------------------------------------------------
# Scenario topology used standalone
# ---------------------------------------------------------------------------
def test_scenario_topology_can_be_used_without_an_adapter():
    scenario = ScenarioTopology(
        candidates=[(5.0, 0.0), (10.0, 0.0)],
        sink=(0.0, 0.0),
        radius=6.0,
        mobile_nodes=[],
    )
    assert scenario.sink_neighbours == frozenset({0})
    assert scenario.adjacency[0] == (1,)
    tree = build_sink_rooted_tree(scenario, [1, 1])
    assert tree.parent(0) == SINK_NODE
    assert tree.parent(1) == 0
