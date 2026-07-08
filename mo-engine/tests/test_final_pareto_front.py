"""Regression: the reported Pareto front must be the non-dominated set of the
FINAL population (parents ∪ last offspring), NOT the whole archive of every
genome ever evaluated.

The archive front accumulates near-front points from early, poorly-converged
generations, producing a thick/noisy front that does not match a standard
reference plot. This test locks the clean final-population behaviour for both
NSGA-II and NSGA-III.
"""
import pytest

from lib.problem.chromosomes import Chromosome
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.nsga2 import NSGA2LoopStrategy


class _FakeChromosome(Chromosome):
    def __init__(self, tag: str):
        self._tag = tag

    def to_dict(self):
        return {"tag": self._tag}

    def get_source_by_mac_protocol(self, options):
        return self, None

    def __eq__(self, other):
        return isinstance(other, _FakeChromosome) and self._tag == other._tag

    def __hash__(self):
        return hash(self._tag)


@pytest.mark.parametrize("cls", [NSGA3LoopStrategy, NSGA2LoopStrategy])
def test_final_front_uses_final_population_only(cls):
    strat = cls.__new__(cls)
    strat._objective_keys = ["f1", "f2"]
    strat._objective_goals = [1, 1]  # both minimize

    # Final population: two clean, non-dominated points.
    fp1 = _FakeChromosome("final-1")
    fp2 = _FakeChromosome("final-2")
    strat._parents = [fp1]
    strat._current_population = [fp2]

    # An archive-only genome that is non-dominated (extreme in f1) but NOT in the
    # final population — it must NOT appear in the reported front.
    archive_only = _FakeChromosome("archive-only")

    strat._map_genome_objectives = {
        fp1: [0.1, 0.9],
        fp2: [0.9, 0.1],
        archive_only: [0.05, 2.0],   # non-dominated vs the finals, but stale
    }

    front = strat._final_pareto_front()
    tags = {item["chromosome"]["tag"] for item in front}

    assert "archive-only" not in tags, "archive-only genome leaked into the reported front"
    assert tags == {"final-1", "final-2"}
    # objectives echoed in original space (all-min → unchanged), keyed by name
    by_tag = {item["chromosome"]["tag"]: item["objectives"] for item in front}
    assert by_tag["final-1"] == {"f1": 0.1, "f2": 0.9}
    assert by_tag["final-2"] == {"f1": 0.9, "f2": 0.1}


def test_final_front_empty_population_returns_empty():
    strat = NSGA3LoopStrategy.__new__(NSGA3LoopStrategy)
    strat._objective_keys = ["f1", "f2"]
    strat._objective_goals = [1, 1]
    strat._parents = []
    strat._current_population = []
    strat._map_genome_objectives = {}
    assert strat._final_pareto_front() == []
