"""The request examples in debug/requests are experiments the engine can run.

Two of them were not: the P3 example declared objectives its conversion did
not produce, and read its seed from a key the engine ignores; the P4 example
had no motes and no stops.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from lib.problem.chromosomes import chromosome_from_dict
from lib.problem.resolve import build_adapter

REQUESTS = Path(__file__).resolve().parents[2] / "debug" / "requests"
EXAMPLES = sorted(REQUESTS.glob("post-*.json"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_every_objective_is_produced_by_the_conversion(path):
    body = json.loads(path.read_text())
    produced = {m["name"] for m in body["data_conversion_config"]["metrics"]}
    wanted = {o["metric_name"] for o in body["parameters"]["objectives"]}
    assert wanted <= produced, f"missing from the conversion: {sorted(wanted - produced)}"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_the_seeds_use_the_key_the_engine_reads(path):
    simulation = json.loads(path.read_text())["parameters"]["simulation"]
    assert "random_seed" not in simulation
    assert simulation.get("random_seeds") or simulation.get("random_seeds_count")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_the_problem_builds_and_its_individuals_encode(path):
    parameters = json.loads(path.read_text())["parameters"]
    problem = parameters["problem"]
    adapter = build_adapter(problem, parameters.get("algorithm") or {}, random.Random(1))
    if parameters["strategy"] == "batch":
        genomes = [chromosome_from_dict(problem["name"], c) for c in parameters["chromosomes"]]
    else:
        genomes = adapter.random_individual_generator(4)
    for genome in genomes:
        adapter.encode_simulation_input(genome)
