"""What the GUI exports for P4 runs in the engine.

The fixture is the one gui/simlab/src/services/__tests__/problemJsonP4.spec.ts
checks exportProblem and exportChromosome against.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from lib.problem.chromosomes import chromosome_from_dict
from lib.problem.resolve import build_adapter

FIXTURE = (Path(__file__).resolve().parents[2]
           / "gui/simlab/src/services/__tests__/fixtures/p4-export.json")


def test_the_exported_problem_and_chromosome_run():
    data = json.loads(FIXTURE.read_text())
    adapter = build_adapter(data["problem"], {}, random.Random(1))

    chromosome = chromosome_from_dict("problem4", data["chromosome"]["chromosome"])
    assert adapter.penalty_objectives(chromosome, 3) is None
    adapter.encode_simulation_input(chromosome)

    for genome in adapter.random_individual_generator(8):
        adapter.encode_simulation_input(genome)
