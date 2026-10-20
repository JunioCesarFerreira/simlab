"""Regression: the reported Pareto front must be the non-dominated set of
P_final — the population environmental selection SELECTED — and nothing else.

Two ways the front gets polluted, both locked here for NSGA-II and NSGA-III:

  * the whole archive of every genome ever evaluated, whose non-dominated set
    accumulates near-front points from early, poorly-converged generations and
    produces a thick/noisy front that matches no reference plot;
  * ``parents ∪ last offspring``, up to 2·pop_size candidates that no selection
    ever ran on. ``_evolution`` runs the final environmental selection before
    the stop condition, so ``self._parents`` is already the selected P_final and
    the last offspring are folded into it.
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

    # P_final: the survivors of the last environmental selection.
    fp1 = _FakeChromosome("final-1")
    fp2 = _FakeChromosome("final-2")
    strat._parents = [fp1, fp2]

    # Last offspring still in _current_population. fp2 survived and is in both;
    # "rejected" lost the selection despite being non-dominated in the union,
    # and must NOT reappear in the reported front.
    rejected = _FakeChromosome("rejected")
    strat._current_population = [fp2, rejected]

    # An archive-only genome that is non-dominated (extreme in f1) but in
    # neither set — it must NOT appear in the reported front either.
    archive_only = _FakeChromosome("archive-only")

    strat._map_genome_objectives = {
        fp1: [0.1, 0.9],
        fp2: [0.9, 0.1],
        rejected: [0.5, 0.5],        # non-dominated vs the finals, but not selected
        archive_only: [0.05, 2.0],   # non-dominated vs the finals, but stale
    }

    front = strat._final_pareto_front()
    tags = {item["chromosome"]["tag"] for item in front}

    assert "archive-only" not in tags, "archive-only genome leaked into the reported front"
    assert "rejected" not in tags, "unselected offspring leaked into the reported front"
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
