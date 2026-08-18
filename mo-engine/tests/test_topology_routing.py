"""Observed link importance R = (r_ij) and its optional Cooja/RPL source."""
import pytest

from lib.problem.topology import (
    SINK_NODE,
    ParentArrayTree,
    RoutingKnowledge,
    RoutingObservation,
    ScenarioTopology,
    TreeCostWeights,
    build_sink_rooted_tree,
    merge_observations,
    observation_from_dodag,
)


def _tree(edges: dict[int, int]) -> ParentArrayTree:
    return ParentArrayTree.from_parent_map(edges)


class TestFrequencyModel:
    def test_importance_is_zero_without_history(self):
        knowledge = RoutingKnowledge()
        assert knowledge.observation_count == 0
        assert knowledge.importance(0, 1) == 0.0
        assert knowledge.node_importance(0) == 0.0
        assert knowledge.node_importance_map() == {}

    def test_frequency_of_a_link_over_observations(self):
        knowledge = RoutingKnowledge()
        knowledge.observe_tree(_tree({0: SINK_NODE, 1: 0}), key="a")
        knowledge.observe_tree(_tree({0: SINK_NODE, 2: 0}), key="b")

        assert knowledge.observation_count == 2
        assert knowledge.importance(0, SINK_NODE) == pytest.approx(1.0)
        assert knowledge.importance(1, 0) == pytest.approx(0.5)
        assert knowledge.importance(3, 0) == 0.0

    def test_importance_is_symmetric(self):
        knowledge = RoutingKnowledge()
        knowledge.observe_tree(_tree({0: SINK_NODE, 1: 0}), key="a")
        assert knowledge.importance(1, 0) == knowledge.importance(0, 1)

    def test_ingestion_is_idempotent_on_the_key(self):
        knowledge = RoutingKnowledge()
        assert knowledge.observe_tree(_tree({0: SINK_NODE}), key="same") is True
        assert knowledge.observe_tree(_tree({0: SINK_NODE}), key="same") is False
        assert knowledge.observation_count == 1

    def test_cached_map_is_invalidated_by_new_observations(self):
        knowledge = RoutingKnowledge()
        knowledge.observe_tree(_tree({0: SINK_NODE}), key="a")
        first = dict(knowledge.node_importance_map())
        knowledge.observe_tree(_tree({0: SINK_NODE, 5: 0}), key="b")
        second = knowledge.node_importance_map()

        assert 5 not in first
        assert 5 in second

    def test_round_trips_through_a_plain_dict(self):
        knowledge = RoutingKnowledge("fingerprint-x")
        knowledge.observe_tree(_tree({0: SINK_NODE, 1: 0}), key="a")
        restored = RoutingKnowledge.from_dict(knowledge.to_dict())

        assert restored.scenario_fingerprint == "fingerprint-x"
        assert restored.observation_count == knowledge.observation_count
        assert restored.importance(1, 0) == knowledge.importance(1, 0)


class TestCoojaBridge:
    DODAG = {
        "tree": {
            "root": "fe80::1",
            "edges": {"fe80::3": "fe80::2", "fe80::2": "fe80::1"},
            "depth": {"fe80::2": 1, "fe80::3": 2},
        }
    }
    MAPPING = {"fe80::1": SINK_NODE, "fe80::2": 0, "fe80::3": 1}

    def test_absent_dodag_or_mapping_yields_nothing(self):
        assert observation_from_dodag(None, self.MAPPING) is None
        assert observation_from_dodag(self.DODAG, None) is None
        assert observation_from_dodag({"tree": {"edges": {}}}, self.MAPPING) is None

    def test_edges_are_translated_to_candidate_indices(self):
        observation = observation_from_dodag(self.DODAG, self.MAPPING)

        assert observation is not None
        assert observation.source == "rpl_dodag"
        assert set(observation.links) == {(1, 0), (0, SINK_NODE)}

    def test_unmappable_addresses_are_skipped_not_guessed(self):
        partial = {"fe80::1": SINK_NODE, "fe80::2": 0}
        observation = observation_from_dodag(self.DODAG, partial)

        assert observation is not None
        assert observation.links == ((0, SINK_NODE),)

    def test_rpl_observation_feeds_the_same_matrix_as_a_structural_one(self):
        knowledge = RoutingKnowledge()
        knowledge.observe(observation_from_dodag(self.DODAG, self.MAPPING), key="sim-1")
        knowledge.observe_tree(_tree({0: SINK_NODE, 1: 0}), key="genome-1")

        assert knowledge.observation_count == 2
        assert knowledge.importance(1, 0) == pytest.approx(1.0)

    def test_merging_multi_seed_observations(self):
        merged = merge_observations([
            RoutingObservation(links=((0, SINK_NODE), (1, 0))),
            RoutingObservation(links=((0, SINK_NODE), (2, 0))),
        ])
        assert set(merged.links) == {(0, SINK_NODE), (1, 0), (2, 0)}
        assert merged.weights[(0, SINK_NODE)] == pytest.approx(2.0)


class TestTreeCostUsesHistory:
    def _scenario(self) -> ScenarioTopology:
        # Two equidistant parents for candidate 2, so only history can break the tie.
        return ScenarioTopology(
            candidates=[(0.0, 10.0), (0.0, -10.0), (10.0, 0.0)],
            sink=(0.0, 0.0),
            radius=15.0,
            mobile_nodes=[],
        )

    def test_without_history_the_tie_breaks_deterministically(self):
        scenario = self._scenario()
        first = build_sink_rooted_tree(scenario, [1, 1, 1])
        second = build_sink_rooted_tree(scenario, [1, 1, 1])
        assert first.parent(2) == second.parent(2)

    def test_history_can_change_the_chosen_parent(self):
        scenario = self._scenario()
        weights = TreeCostWeights(distance_weight=1.0, routing_importance_weight=5.0)

        neutral = build_sink_rooted_tree(scenario, [1, 1, 1], weights, RoutingKnowledge())
        knowledge = RoutingKnowledge()
        for i in range(5):
            knowledge.observe_tree(_tree({1: SINK_NODE, 2: 1}), key=f"obs{i}")
        informed = build_sink_rooted_tree(scenario, [1, 1, 1], weights, knowledge)

        assert informed.parent(2) == 1
        assert informed.detached_nodes() == []
        assert neutral.detached_nodes() == []
