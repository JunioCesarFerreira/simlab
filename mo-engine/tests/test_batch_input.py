"""The shape the API accepts for a batch is the shape the engine writes.

``pylib/batch_input.py`` cannot import the chromosomes, so this holds the two
together: whatever an operator makes can be sent back as a batch.
"""
from __future__ import annotations

import json
import random

import pytest

from lib.problem.chromosomes import chromosome_from_dict
from lib.problem.resolve import build_adapter
from lib.strategy.batch import BatchStrategy
from pylib.batch_input import validate_batch_input
from pylib.db.models.enums import EnumStatus
from tests.resume_harness import PROBLEMS, Run

# Tuples: the adapter of P4 does not take the lists that JSON gives.
ROUTE = {
    "name": "problem4",
    "region": [-100.0, -100.0, 100.0, 100.0],
    "radius_of_reach": 50.0,
    "radius_of_inter": 60.0,
    "nodes": [(10.0, 0.0), (20.0, 5.0)],
    "sink_base": (0.0, 0.0),
    "initial_energy": 100.0,
    "buffer_capacity": 10.0,
    "data_rate": 1.0,
    "speed": 5.0,
    "time_step": 1.0,
    "max_route_length": 6,
    "tau_bounds": [0.0, 4.0],
    "sojourns": [
        {"id": i, "position": (10.0 * i, 0.0), "visibleNodes": [0, 1],
         "adjacency": [j for j in range(4) if j != i]}
        for i in range(4)
    ],
}
EVERY = {**PROBLEMS, "route": ROUTE}


def as_sent(chromosome) -> dict:
    return json.loads(json.dumps(chromosome.to_dict()))


@pytest.mark.parametrize("problem", list(EVERY))
def test_whatever_an_operator_makes_can_be_sent_back(problem):
    config = EVERY[problem]
    adapter = build_adapter(config, {"per_gene_prob": 0.5}, random.Random(7))
    made = list(adapter.random_individual_generator(8))
    for _ in range(40):
        children = adapter.crossover(random.Random(len(made)).sample(made, 2))
        made.extend(adapter.mutate(child) for child in children)

    sent = [as_sent(chromosome) for chromosome in made]

    validate_batch_input(
        {"problem": config, "simulation": {}, "chromosomes": sent},
        {"csma": "a", "tsch": "b"},
    )
    for chromosome, document in zip(made, sent):
        assert chromosome_from_dict(config["name"], document).get_hash() == chromosome.get_hash()
        adapter.encode_simulation_input(chromosome_from_dict(config["name"], document))


@pytest.mark.parametrize("problem", list(PROBLEMS))
def test_what_is_accepted_runs_to_the_end(problem):
    adapter = build_adapter(PROBLEMS[problem], {}, random.Random(3))
    parameters = {"chromosomes": [as_sent(c) for c in adapter.random_individual_generator(4)]}
    validate_batch_input({"problem": PROBLEMS[problem], "simulation": {}, **parameters}, {"csma": "a"})

    run = Run(problem, strategy=BatchStrategy, parameters=parameters).finish()

    assert run.db.experiment["status"] == EnumStatus.DONE
