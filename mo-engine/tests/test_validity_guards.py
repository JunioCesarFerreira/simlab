"""Guards on what reaches the algorithm and what is reported as a result.

Three defects found in the 2026-11-18 review, each silent:

1. **NaN reached the sort.** ``statistics`` returns NaN for an absent or empty
   column, the master publishes the simulation as ``Done``, the completeness
   count treated the NaN as present, and the objective extraction rejected
   only ``None``. A NaN vector is never dominated (every comparison is false):
   it entered front 0 and, through the ideal point, randomised NSGA-III
   niching.
2. **A front of penalties finished as "completed".** A P2 run whose every
   individual violated the coverage constraint ended ``Done`` with 63 front
   items at ~1e9 and nothing saying they were not a result.
3. **The MAC gene was false.** With one firmware source the adjusted
   chromosome was thrown away: the stored gene drifted ("TSCH" in a CSMA-only
   run) and two hashes could share one phenotype.
"""
import math
import random
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.problem.chromosomes import ChromosomeP0, ChromosomeP2
from lib.problem.mac_gene import (
    MacPinnedAdapter,
    pin_mac,
    pin_mac_gene,
    pinned_mac_protocol,
)
from lib.strategy.completeness import (
    GateResult,
    ReplicaPolicy,
    annotate,
    evaluation_of,
    finite_or_none,
    front_validity,
    front_warning,
    is_usable,
    is_valid_evaluation,
    objective_vector,
    refuse,
    without_simulations,
    STATE_INFEASIBLE,
)

GEN_ID = ObjectId("507f1f77bcf86cd799450101")

COMPLETE = {
    "expected": 1, "observed": 1, "state": "complete",
    "per_metric": {"latency": 1}, "thin_metrics": [],
}


# ── 1. Objective extraction refuses what is not a number ─────────────────────

class TestObjectiveVector:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "1.0", None, object()])
    def test_a_non_finite_or_non_numeric_metric_has_no_vector(self, bad):
        vector, reason = objective_vector({"latency": bad}, ["latency"], [1])
        assert vector is None
        assert "latency" in reason

    def test_a_missing_metric_has_no_vector(self):
        vector, reason = objective_vector({}, ["latency"], [1])
        assert vector is None and "missing" in reason

    def test_finite_metrics_are_converted_to_minimisation(self):
        vector, reason = objective_vector(
            {"latency": 2, "throughput": 5.0}, ["latency", "throughput"], [1, -1])
        assert vector == [2.0, -5.0] and reason is None

    def test_finite_or_none(self):
        assert finite_or_none(3) == 3.0
        assert finite_or_none(float("nan")) is None
        assert finite_or_none("3") is None


class TestARefusedVectorIsNotAMeasurement:
    def test_refused_is_not_usable_even_when_the_replicas_are_complete(self):
        record = refuse(COMPLETE, "a metric is missing or not a finite number")
        assert is_usable(record, ReplicaPolicy()) is False
        stored = annotate(record, usable=is_usable(record, ReplicaPolicy()),
                          policy=ReplicaPolicy())
        assert stored["objective_used"] is False
        assert stored["refused"]
        assert stored["state"] == "complete"   # the evidence is kept as it was

    def test_refusing_with_no_record_still_says_why(self):
        record = refuse(None, "no aggregated metrics")
        assert record["state"] == "unknown" and record["refused"]

    def test_validity_of_each_kind_of_record(self):
        assert is_valid_evaluation(None) is True
        assert is_valid_evaluation(annotate(COMPLETE, usable=True, policy=ReplicaPolicy()))
        assert not is_valid_evaluation(refuse(COMPLETE, "nan"))
        assert not is_valid_evaluation(
            without_simulations(STATE_INFEASIBLE, "violates a constraint"))
        assert not is_valid_evaluation({"state": "partial", "objective_used": False})


# ── 1b. The handlers of the four strategies ──────────────────────────────────

def _nsga(cls):
    strat = cls.__new__(cls)
    strat._current_population = []
    strat._genome_objectives_cache = {}
    strat._genome_completeness_cache = {}
    strat._generation_completeness = {}
    strat._evolution = MagicMock()
    return strat


def _random_search(cls):
    strat = _nsga(cls)
    strat._gen_index = 1
    strat._max_gen = 5
    strat._pop_size = 1
    strat._generation_enqueue = MagicMock()
    strat._problem_adapter = MagicMock()
    strat._problem_adapter.random_individual_generator.return_value = []
    return strat


def _batch(cls):
    strat = cls.__new__(cls)
    strat._evidence = {}
    strat.stop = MagicMock()
    return strat


def _strategies():
    from lib.strategy.batch import BatchStrategy
    from lib.strategy.nsga2 import NSGA2LoopStrategy
    from lib.strategy.nsga3 import NSGA3LoopStrategy
    from lib.strategy.random_search import RandomSearchStrategy
    return [
        (NSGA2LoopStrategy, _nsga),
        (NSGA3LoopStrategy, _nsga),
        (RandomSearchStrategy, _random_search),
        (BatchStrategy, _batch),
    ]


def _build(cls, factory, metrics, completeness):
    strat = factory(cls)
    strat._stop_flag = False
    strat._generation_id = GEN_ID
    strat._exp_id = ObjectId()
    strat._objective_keys = ["latency"]
    strat._objective_goals = [1]
    strat._aggregators = ["mean"]
    strat._ga_rng = random.Random(1)
    strat._map_genome_objectives = {}
    strat._replica_policy = ReplicaPolicy()
    strat._replica_gate = MagicMock()
    strat._replica_gate.evaluate.return_value = GateResult(
        proceed=True, completeness=completeness)

    mongo = MagicMock()
    mongo.experiment_repo.is_cancelled.return_value = False
    mongo.generation_repo.get_simulations_metrics_by_individual.return_value = metrics
    strat.mongo = mongo

    genome = MagicMock()
    genome.get_hash.return_value = "a"
    genome.to_dict.return_value = {"x": [0.0]}
    if hasattr(strat, "_current_population"):
        strat._current_population = [genome]
    else:
        strat._chromosomes = [genome]
    return strat, mongo, genome


@pytest.mark.parametrize("cls,factory", _strategies(), ids=lambda v: getattr(v, "__name__", ""))
class TestTheHandlersRefuseNaN:
    def test_a_nan_metric_gets_the_worst_vector_and_is_marked_refused(self, cls, factory):
        import importlib
        worst = importlib.import_module(cls.__module__).WORST_OBJECTIVE
        strat, mongo, genome = _build(
            cls, factory, {"a": {"latency": float("nan")}}, {"a": dict(COMPLETE)})

        strat._handle_generation_done(GEN_ID)

        assert strat._map_genome_objectives[genome] == [worst]
        stored = mongo.individual_repo.update_evaluation.call_args.args[3]
        assert stored["objective_used"] is False
        assert "finite" in stored["refused"]

    def test_no_metrics_at_all_is_marked_refused_too(self, cls, factory):
        strat, mongo, genome = _build(cls, factory, {}, {"a": dict(COMPLETE)})

        strat._handle_generation_done(GEN_ID)

        stored = mongo.individual_repo.update_evaluation.call_args.args[3]
        assert stored["objective_used"] is False and stored["refused"]

    def test_a_finite_metric_is_still_a_measurement(self, cls, factory):
        strat, mongo, genome = _build(
            cls, factory, {"a": {"latency": 2.5}}, {"a": dict(COMPLETE)})

        strat._handle_generation_done(GEN_ID)

        assert strat._map_genome_objectives[genome] == [2.5]
        stored = mongo.individual_repo.update_evaluation.call_args.args[3]
        assert stored["objective_used"] is True and "refused" not in stored


# ── 2. A front made of penalties is not reported as a result ─────────────────

class TestFrontValidity:
    def _item(self, valid, state="complete"):
        return {"objectives": {}, "chromosome": {},
                "evaluation": {"state": state, "objective_used": valid, "valid": valid}}

    def test_counts(self):
        items = [self._item(True), self._item(False, STATE_INFEASIBLE),
                 self._item(False, "partial")]
        assert front_validity(items) == {
            "items": 3, "valid": 1, "infeasible": 1, "refused": 1}

    def test_an_all_penalty_front_is_flagged(self):
        validity = front_validity([self._item(False, STATE_INFEASIBLE)] * 3)
        assert "WITHOUT a valid solution" in front_warning(validity)

    def test_a_clean_front_has_no_warning(self):
        assert front_warning(front_validity([self._item(True)] * 2)) is None
        assert front_warning(front_validity([])) is None

    def test_legacy_items_without_evaluation_count_as_valid(self):
        assert front_validity([{"objectives": {}}])["valid"] == 1

    def test_evaluation_of_a_penalty(self):
        record = without_simulations(STATE_INFEASIBLE, "violates a constraint")
        assert evaluation_of(record) == {
            "state": STATE_INFEASIBLE, "objective_used": True, "valid": False}


@pytest.mark.parametrize("cls,factory", _strategies(), ids=lambda v: getattr(v, "__name__", ""))
def test_finalize_writes_the_validity_and_the_warning(cls, factory):
    strat = factory(cls)
    strat._exp_id = ObjectId()
    strat.mongo = MagicMock()
    strat.stop = MagicMock()
    item = {"chromosome": {}, "objectives": {"latency": 1.01e9},
            "evaluation": evaluation_of(
                without_simulations(STATE_INFEASIBLE, "violates a constraint"))}

    strat._finalize_experiment(pareto_front=[item])

    fields = strat.mongo.experiment_repo.update.call_args.args[1]
    assert fields["status"] == "Done"
    assert fields["pareto_front_validity"] == {
        "items": 1, "valid": 0, "infeasible": 1, "refused": 0}
    assert "WITHOUT a valid solution" in fields["system_message"]


def test_the_nsga3_front_items_carry_their_evidence():
    from lib.strategy.nsga3 import NSGA3LoopStrategy

    strat = NSGA3LoopStrategy.__new__(NSGA3LoopStrategy)
    strat._objective_keys = ["f1", "f2"]
    strat._objective_goals = [1, 1]
    feasible, penalised = MagicMock(), MagicMock()
    feasible.get_hash.return_value, penalised.get_hash.return_value = "ok", "pen"
    feasible.to_dict.return_value, penalised.to_dict.return_value = {}, {}
    strat._parents = [feasible, penalised]
    strat._map_genome_objectives = {feasible: [1.0, 2.0], penalised: [1e9, 1e9]}
    strat._genome_completeness_cache = {
        "ok": annotate(COMPLETE, usable=True, policy=ReplicaPolicy()),
        "pen": without_simulations(STATE_INFEASIBLE, "violates a constraint"),
    }

    front = strat._final_pareto_front()

    assert len(front) == 1
    assert front[0]["evaluation"]["valid"] is True


# ── 3. The MAC gene is pinned where chromosomes are made ─────────────────────

class _FakeAdapter:
    """Produces chromosomes with the MAC gene set to TSCH, as a drift would."""

    def __init__(self, masks):
        self._masks = list(masks)
        self.radius_of_reach = 50.0

    def random_individual_generator(self, size):
        out = [ChromosomeP2(mac_protocol=1, mask=self._masks.pop(0)) for _ in range(size)]
        return out

    def crossover(self, parents):
        return (ChromosomeP2(mac_protocol=1, mask=[1, 0]),
                ChromosomeP2(mac_protocol=0, mask=[0, 1]))

    def mutate(self, chromosome):
        return ChromosomeP2(mac_protocol=1 - chromosome.mac_protocol, mask=chromosome.mask)


class TestMacGene:
    def test_the_only_source_decides(self):
        assert pinned_mac_protocol({"csma": "id"}) == 0
        assert pinned_mac_protocol({"tsch": "id"}) == 1
        assert pinned_mac_protocol({"csma": "a", "tsch": "b"}) is None
        assert pinned_mac_protocol({}) is None

    def test_two_sources_leave_the_adapter_alone(self):
        inner = _FakeAdapter([])
        assert pin_mac_gene(inner, {"csma": "a", "tsch": "b"}) is inner

    def test_every_operator_returns_the_pinned_gene(self):
        adapter = pin_mac_gene(_FakeAdapter([[1, 1], [0, 1]]), {"csma": "id"})
        assert isinstance(adapter, MacPinnedAdapter)
        population = adapter.random_individual_generator(2)
        assert {c.mac_protocol for c in population} == {0}
        children = adapter.crossover(population)
        assert isinstance(children, tuple)
        assert {c.mac_protocol for c in children} == {0}
        assert adapter.mutate(population[0]).mac_protocol == 0

    def test_delegates_everything_else(self):
        adapter = pin_mac_gene(_FakeAdapter([]), {"csma": "id"})
        assert adapter.radius_of_reach == 50.0

    def test_individuals_that_differed_only_in_the_gene_are_not_duplicated(self):
        # The second and third masks are equal: before pinning they could
        # differ in the MAC bit only. The generator tops the population up.
        inner = _FakeAdapter([[1, 0], [1, 1], [1, 1], [0, 0]])
        population = pin_mac_gene(inner, {"csma": "id"}).random_individual_generator(3)
        masks = [c.mask for c in population]
        assert len(masks) == 3 and len({tuple(m) for m in masks}) == 3

    def test_a_space_too_small_still_returns_the_size_asked_for(self):
        class Tiny(_FakeAdapter):
            def random_individual_generator(self, size):
                return [ChromosomeP2(mac_protocol=1, mask=[1]) for _ in range(size)]
        population = pin_mac_gene(Tiny([]), {"csma": "id"}).random_individual_generator(4)
        assert len(population) == 4

    def test_a_chromosome_without_the_gene_is_untouched(self):
        p0 = ChromosomeP0(x=[0.1, 0.2])
        assert pin_mac(p0, 0) is p0

    def test_the_stored_gene_is_the_one_that_runs(self):
        chromosome = ChromosomeP2(mac_protocol=1, mask=[1, 0])
        pinned = pin_mac(chromosome, 0)
        assert pinned.mac_protocol == 0
        adjusted, source = pinned.get_source_by_mac_protocol({"csma": "src"})
        assert adjusted is pinned and source == "src"
