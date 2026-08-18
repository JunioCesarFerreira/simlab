"""
Regression test for P4's genetic-algorithm pipeline.

Until this fix, P4 was unusable through the actual GA loop:
  - ProblemP4.cast() never assigned `max_route_length`/`tau_bounds` from the
    input dict (see pylib/config/problems.py), so `_random_route()` and
    `_random_tau()` raised AttributeError on the very first individual.
  - `random_individual_generator`/`crossover`/`mutate` built/consumed
    chromosomes as raw `(route, tau)` tuples, incompatible with
    `ChromosomeP4` (a frozen dataclass with `mac_protocol`, `route`,
    `sojourn_times`) — `ChromosomeP4(chromosome=...)` is not a valid call,
    and dataclass instances aren't tuple-unpackable.
No prior test exercised this path — test_encode_p4.py only calls
`encode_simulation_input` on a manually constructed ChromosomeP4.
"""
import random

from lib.problem.resolve import build_test_adapter
from lib.problem.chromosomes import ChromosomeP4


def _p4_problem():
    return {
        "name": "problem4",
        "region": [-50.0, -50.0, 50.0, 50.0],
        "radius_of_reach": 20.0,
        "radius_of_inter": 25.0,
        "nodes": [(5.0, 5.0)],
        "sink_base": (0.0, 0.0),
        "initial_energy": 100.0,
        "buffer_capacity": 10.0,
        "data_rate": 1.0,
        "speed": 1.0,
        "time_step": 1.0,
        "max_route_length": 6,
        "tau_bounds": (0.0, 5.0),
        "sojourns": [
            {"id": 0, "position": (0.0, 0.0), "adjacency": [1, 2], "visibleNodes": []},
            {"id": 1, "position": (10.0, 0.0), "adjacency": [0, 2], "visibleNodes": []},
            {"id": 2, "position": (5.0, 10.0), "adjacency": [0, 1], "visibleNodes": []},
        ],
    }


def _adapter():
    adapter = build_test_adapter(_p4_problem())
    adapter.set_ga_operator_configs(
        random.Random(1), {"per_gene_prob": 0.1, "pm_tau": 0.5, "sigma_tau": 2.0}
    )
    return adapter


def test_p4_random_individual_generator_returns_chromosomes():
    adapter = _adapter()
    pop = adapter.random_individual_generator(10)

    assert len(pop) == 10
    for chrm in pop:
        assert isinstance(chrm, ChromosomeP4)
        assert chrm.mac_protocol in (0, 1)
        assert len(chrm.sojourn_times) == len(chrm.route)


def test_p4_crossover_returns_chromosomes():
    adapter = _adapter()
    parents = adapter.random_individual_generator(2)

    children = adapter.crossover(parents)

    assert len(children) == 2
    for child in children:
        assert isinstance(child, ChromosomeP4)
        assert child.mac_protocol in (0, 1)
        assert len(child.sojourn_times) == len(child.route)


def test_p4_mutate_returns_chromosome():
    adapter = _adapter()
    parent = adapter.random_individual_generator(1)[0]

    child = adapter.mutate(parent)

    assert isinstance(child, ChromosomeP4)
    assert child.mac_protocol in (0, 1)
    assert len(child.sojourn_times) == len(child.route)


def test_p4_encode_after_generation_does_not_raise():
    adapter = _adapter()
    ind = adapter.random_individual_generator(1)[0]

    sim = adapter.encode_simulation_input(ind)

    assert len(sim["mobileMotes"]) == 1
