"""P4 runs from JSON, keeps its routes on the mobility graph, and penalises a
route that loses data.

The adapter refused the lists JSON and BSON give, appended the base after an
invalid move even where that move did not exist, and enforced no data
constraint at all.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from lib.problem.chromosomes import ChromosomeP4
from lib.problem.resolve import build_adapter, build_test_adapter
from lib.strategy.batch import BatchStrategy
from pylib.db.models.enums import EnumStatus
from tests.resume_harness import Run

# Motes 0 and 1 form one connected part, mote 2 is alone.
# Sojourn 0 is the base; 1 sees nothing; 2 sees mote 2.
LINE = {
    "name": "problem4",
    "region": [-10.0, -50.0, 250.0, 50.0],
    "radius_of_reach": 50.0,
    "radius_of_inter": 60.0,
    "nodes": [[10.0, 0.0], [20.0, 0.0], [200.0, 0.0]],
    "sink_base": [0.0, 0.0],
    "initial_energy": 100.0,
    "buffer_capacity": 100.0,
    "data_rate": 1.0,
    "speed": 10.0,
    "time_step": 1.0,
    "max_route_length": 6,
    "tau_bounds": [0.0, 50.0],
    "sojourns": [
        {"id": 0, "position": [0.0, 0.0], "adjacency": [1, 2], "visibleNodes": [0]},
        {"id": 1, "position": [100.0, 0.0], "adjacency": [0, 2], "visibleNodes": []},
        {"id": 2, "position": [190.0, 0.0], "adjacency": [0, 1], "visibleNodes": [2]},
    ],
}

# A one-way ring 0 → 1 → 2 → 3 → 0.
RING = {**LINE, "sojourns": [
    {"id": i, "position": [50.0 * i, 0.0], "adjacency": [(i + 1) % 4], "visibleNodes": []}
    for i in range(4)
]}


def adapter(problem=LINE, seed=1, **settings):
    a = build_adapter(problem, {"per_gene_prob": 0.5, **settings}, random.Random(seed))
    return a


def moves_exist(problem, route) -> bool:
    """Every move is an edge; [0, 0] is the route that never leaves the base."""
    if route == [0, 0]:
        return True
    adjacency = {s["id"]: set(s["adjacency"]) for s in problem["sojourns"]}
    return all(b in adjacency[a] for a, b in zip(route, route[1:]))


class TestInput:
    def test_the_lists_of_json_are_accepted(self):
        build_test_adapter(json.loads(json.dumps(LINE)))

    def test_tuples_still_are(self):
        problem = {**LINE, "nodes": [tuple(p) for p in LINE["nodes"]], "sink_base": (0.0, 0.0)}
        build_test_adapter(problem)

    def test_sojourn_ids_must_be_their_indices(self):
        sojourns = [dict(s) for s in LINE["sojourns"]]
        sojourns[1]["id"], sojourns[2]["id"] = 2, 1
        with pytest.raises(ValueError, match="position in the list"):
            build_test_adapter({**LINE, "sojourns": sojourns})

    @pytest.mark.parametrize("key, value", [
        ("max_route_length", 1), ("max_route_length", 3.0),
        ("tau_bounds", [5.0, 1.0]), ("tau_bounds", [-1.0, 1.0]),
        ("nodes", [[1.0]]), ("sink_base", "0,0"),
    ])
    def test_malformed_settings_are_refused(self, key, value):
        with pytest.raises((ValueError, TypeError)):
            build_test_adapter({**LINE, key: value})

    def test_the_bundled_example_is_valid(self):
        example = Path(__file__).resolve().parents[2] / "debug/requests/post-nsga3-experiment-p4.json"
        problem = json.loads(example.read_text())["parameters"]["problem"]
        a = adapter(problem)
        for chromosome in a.random_individual_generator(10):
            a.encode_simulation_input(chromosome)


class TestRoutes:
    def test_an_invalid_move_is_followed_by_existing_ones_back_to_the_base(self):
        a = adapter(RING)
        # 0 → 1 → 2 is valid, 2 → 1 is not: return over 2 → 3 → 0.
        assert a._repair_route([0, 1, 2, 1, 0]) == [0, 1, 2, 3, 0]

    def test_the_route_respects_its_length(self):
        # 0 1 2 3 0 has five stops: the walk is shortened until it fits. On
        # this one-way ring no round trip fits in four, so the sink stays home.
        assert adapter({**RING, "max_route_length": 4})._repair_route([0, 1, 2, 3, 0]) == [0, 0]
        assert adapter({**RING, "max_route_length": 5})._repair_route([0, 1, 2, 3, 0, 1]) == [0, 1, 2, 3, 0]

    @pytest.mark.parametrize("seed", range(10))
    def test_every_operator_keeps_the_moves_on_the_graph(self, seed):
        for problem in (LINE, RING):
            a = adapter(problem, seed=seed)
            population = a.random_individual_generator(6)
            children = [c for p in zip(population, population[1:]) for c in a.crossover(list(p))]
            children += [a.mutate(p) for p in population]
            for c in [*population, *children]:
                assert moves_exist(problem, c.route) and c.route[0] == c.route[-1] == 0
                assert len(c.sojourn_times) == len(c.route)
                assert len(c.route) <= problem["max_route_length"]


class TestCollection:
    def route(self, stops, dwells):
        return ChromosomeP4(mac_protocol=0, route=list(stops), sojourn_times=list(dwells))

    def test_a_route_that_collects_everything_in_time_is_feasible(self):
        # Base [0, 5], stop 2 at 24..29, back at 48: the longest gap is 43 s.
        assert adapter().penalty_objectives(self.route([0, 2, 0], [5, 5, 0]), 2) is None

    def test_a_mote_no_stop_collects_is_infeasible(self):
        a = adapter()
        assert a.collection_deficits(self.route([0, 1, 0], [5, 5, 0])) == (pytest.approx(1 / 3), 0.0)

    def test_passing_through_collects_nothing(self):
        a = adapter()
        assert a.collection_deficits(self.route([0, 2, 0], [0, 5, 0]))[0] == pytest.approx(2 / 3)

    def test_a_buffer_that_fills_between_rounds_is_infeasible(self):
        a = adapter({**LINE, "data_rate": 10.0})
        # 43 s × 10 = 430 against a buffer of 100: capped at a whole buffer.
        penalty = a.penalty_objectives(self.route([0, 2, 0], [5, 5, 0]), 3)
        assert penalty == [pytest.approx(1.5e9)] * 3

    def test_the_penalty_grows_with_what_is_lost(self):
        a = adapter()
        one = a.penalty_objectives(self.route([0, 1, 0], [5, 5, 0]), 1)[0]
        two = a.penalty_objectives(self.route([0, 2, 0], [0, 5, 0]), 1)[0]
        assert 1e9 < one < two <= 2e9


def test_a_p4_batch_runs_from_json_to_the_end():
    problem = json.loads(json.dumps(LINE))
    chromosomes = [{"mac_protocol": 0, "route": [0, 2, 0], "sojourn_times": [5, 5, 0]},
                   {"mac_protocol": 0, "route": [0, 1, 0], "sojourn_times": [5, 5, 0]}]
    run = Run("targets", strategy=BatchStrategy,
              parameters={"problem": problem, "chromosomes": chromosomes}).finish()

    states = sorted(i["completeness"]["state"] for i in run.db.individuals)
    assert run.db.experiment["status"] == EnumStatus.DONE
    assert states == ["complete", "infeasible"]
    simulated = {s["individual_id"] for s in run.db.simulations}
    assert len(simulated) == 1
