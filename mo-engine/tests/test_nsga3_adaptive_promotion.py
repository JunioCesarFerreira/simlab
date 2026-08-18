"""Promotion and survivor filtering, exercised without MongoDB.

These two steps are what make the heuristic safe: an estimate that could steer
the search must be measured for real (promotion), and anything still estimated
must never survive into the next population.
"""
import random

import numpy as np
import pytest

from lib.adaptive import (
    AdaptiveEvaluationConfig,
    DecisionReason,
    DecisionRecord,
    EvaluationDecision,
)
from lib.nsga import generate_reference_points
from lib.problem.chromosomes import Chromosome
from lib.strategy.nsga3_adaptive import NSGA3AdaptiveSimulationStrategy


class _Genome(Chromosome):
    """Minimal chromosome stand-in with a stable hash."""

    def __init__(self, tag: str):
        self.tag = tag

    def to_dict(self):
        return {"tag": self.tag}

    def get_hash(self):
        return self.tag

    def get_source_by_mac_protocol(self, options):
        return self, None

    def __eq__(self, other):
        return isinstance(other, _Genome) and self.tag == other.tag

    def __hash__(self):
        return hash(self.tag)


def _strategy(pop_size: int = 4, require_simulated_survivors: bool = True):
    strategy = NSGA3AdaptiveSimulationStrategy.__new__(NSGA3AdaptiveSimulationStrategy)
    strategy._pop_size = pop_size
    strategy._objective_keys = ["f1", "f2"]
    strategy._objective_goals = [1, 1]
    strategy._ref_points = generate_reference_points(2, 4)
    strategy._ga_rng = random.Random(0)
    strategy._estimated_hashes = set()
    strategy._gen_decisions = {}
    strategy._map_genome_objectives = {}
    strategy._parents = []
    strategy._current_population = []
    strategy._adaptive_config = AdaptiveEvaluationConfig(
        require_simulated_survivors=require_simulated_survivors
    )
    return strategy


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
class TestPromotionSelection:
    def test_estimated_survivor_is_promoted(self):
        strategy = _strategy(pop_size=3)
        simulated = [_Genome(f"sim{i}") for i in range(3)]
        estimated = _Genome("est-good")

        strategy._parents = simulated
        strategy._current_population = [estimated]
        strategy._estimated_hashes = {"est-good"}
        strategy._map_genome_objectives = {
            simulated[0]: [5.0, 5.0],
            simulated[1]: [6.0, 6.0],
            simulated[2]: [7.0, 7.0],
            estimated: [1.0, 1.0],  # conservative bound, still the best point
        }

        promoted = strategy._select_promotions()

        assert [g.get_hash() for g in promoted] == ["est-good"]

    def test_clearly_dominated_estimate_is_not_promoted(self):
        strategy = _strategy(pop_size=2)
        simulated = [_Genome("sim0"), _Genome("sim1")]
        estimated = _Genome("est-bad")

        strategy._parents = simulated
        strategy._current_population = [estimated]
        strategy._estimated_hashes = {"est-bad"}
        strategy._map_genome_objectives = {
            simulated[0]: [1.0, 1.0],
            simulated[1]: [1.5, 1.5],
            estimated: [90.0, 90.0],
        }

        assert strategy._select_promotions() == []

    def test_nothing_to_promote_when_everything_was_simulated(self):
        strategy = _strategy()
        genomes = [_Genome(f"g{i}") for i in range(4)]
        strategy._parents = genomes[:2]
        strategy._current_population = genomes[2:]
        strategy._map_genome_objectives = {g: [float(i), float(4 - i)] for i, g in enumerate(genomes)}

        assert strategy._select_promotions() == []

    def test_first_front_membership_is_enough_to_be_promoted(self):
        # pop_size is small enough that niching alone could drop the estimate;
        # the first front is always included so it is promoted anyway.
        strategy = _strategy(pop_size=1)
        simulated = [_Genome("sim0"), _Genome("sim1")]
        estimated = _Genome("est-extreme")

        strategy._parents = simulated
        strategy._current_population = [estimated]
        strategy._estimated_hashes = {"est-extreme"}
        strategy._map_genome_objectives = {
            simulated[0]: [1.0, 9.0],
            simulated[1]: [9.0, 1.0],
            estimated: [4.0, 4.0],  # non-dominated, in front 0
        }

        assert [g.get_hash() for g in strategy._select_promotions()] == ["est-extreme"]

    def test_promotion_is_skipped_when_objectives_are_missing(self):
        strategy = _strategy()
        strategy._current_population = [_Genome("est")]
        strategy._estimated_hashes = {"est"}
        strategy._map_genome_objectives = {}

        assert strategy._select_promotions() == []


# ---------------------------------------------------------------------------
# Survivor filtering
# ---------------------------------------------------------------------------
class TestSurvivorFiltering:
    def _union(self, strategy, n_simulated: int, n_estimated: int):
        simulated = [_Genome(f"sim{i}") for i in range(n_simulated)]
        estimated = [_Genome(f"est{i}") for i in range(n_estimated)]
        strategy._estimated_hashes = {g.get_hash() for g in estimated}
        population = simulated + estimated
        # Estimated individuals are given the *best* objectives on purpose: if
        # the filter were missing they would dominate and survive.
        objectives = (
            [[10.0 + i, 10.0 - i] for i in range(n_simulated)]
            + [[0.0, 0.0] for _ in range(n_estimated)]
        )
        return population, objectives

    def test_estimated_individuals_never_survive(self):
        strategy = _strategy(pop_size=4)
        population, objectives = self._union(strategy, n_simulated=6, n_estimated=3)

        survivors = strategy._select_next_parents(population, objectives)

        assert len(survivors) == 4
        assert all(g.get_hash() not in strategy._estimated_hashes for g in survivors)

    def test_filter_is_skipped_when_it_would_starve_the_population(self):
        strategy = _strategy(pop_size=6)
        population, objectives = self._union(strategy, n_simulated=3, n_estimated=5)

        survivors = strategy._select_next_parents(population, objectives)

        assert len(survivors) == 6
        assert any(g.get_hash() in strategy._estimated_hashes for g in survivors)

    def test_relaxed_configuration_admits_estimated_survivors(self):
        strategy = _strategy(pop_size=2, require_simulated_survivors=False)
        population, objectives = self._union(strategy, n_simulated=6, n_estimated=2)

        survivors = strategy._select_next_parents(population, objectives)

        assert any(g.get_hash() in strategy._estimated_hashes for g in survivors)


# ---------------------------------------------------------------------------
# Ground-truth gate
# ---------------------------------------------------------------------------
class TestGroundTruthGate:
    def test_estimated_genome_is_not_ground_truth(self):
        strategy = _strategy()
        strategy._estimated_hashes = {"est"}

        assert strategy._is_ground_truth(_Genome("sim")) is True
        assert strategy._is_ground_truth(_Genome("est")) is False

    def test_final_front_drops_estimated_individuals(self):
        strategy = _strategy()
        good = _Genome("sim-good")
        estimated = _Genome("est-extreme")
        strategy._parents = [good]
        strategy._current_population = [estimated]
        strategy._estimated_hashes = {"est-extreme"}
        strategy._map_genome_objectives = {good: [1.0, 1.0], estimated: [0.0, 0.0]}

        front = strategy._final_pareto_front()

        assert [item["chromosome"]["tag"] for item in front] == ["sim-good"]
        # The population itself must be restored after the filtered read.
        assert strategy._current_population == [estimated]


# ---------------------------------------------------------------------------
# Chromosome bits helper
# ---------------------------------------------------------------------------
def test_chromosome_bits_reads_the_mask():
    from lib.problem.chromosomes import ChromosomeP2

    chromosome = ChromosomeP2(mac_protocol=0, mask=[1, 0, 1, 1])
    assert NSGA3AdaptiveSimulationStrategy._chromosome_bits(chromosome) == [1, 0, 1, 1]


def test_decision_priority_orders_the_budget():
    def _record(reason, promotion=False):
        record = DecisionRecord(
            individual_id="x",
            generation=0,
            decision=EvaluationDecision.SIMULATE,
            reason=reason,
        )
        record.promotion_selected = promotion
        return record

    priorities = [
        _record(DecisionReason.PROVISIONAL_SURVIVOR, promotion=True).priority,
        _record(DecisionReason.WARMUP).priority,
        _record(DecisionReason.POTENTIALLY_NONDOMINATED).priority,
        _record(DecisionReason.HIGH_UNCERTAINTY).priority,
        _record(DecisionReason.HIGH_NOVELTY).priority,
        _record(DecisionReason.AUDIT_SAMPLE).priority,
    ]
    assert priorities == sorted(priorities), "documented priority order is broken"
