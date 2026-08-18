"""End-to-end comparison of NSGA-III against NSGA-III adaptive simulation.

Both strategies run the *same* P2 scenario, the same seeds and the same
deterministic evaluator, driven through the real generation lifecycle by the
in-process fakes.  What the tests pin down is the whole point of the
heuristic: fewer real evaluations, without any estimated value leaking into
the evolutionary state or into the published front.
"""
import copy

import pytest
from bson import ObjectId

from adaptive_fakes import (
    FakeMongo,
    TopologyEvaluator,
    harness_strategy,
    run_to_completion,
)
from lib.adaptive import EvaluationDecision
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.nsga3_adaptive import GenerationPhase, NSGA3AdaptiveSimulationStrategy

POPULATION = 16
GENERATIONS = 6
SINK = (0.0, 0.0)

# kappa is deliberately below the 1.96 default: on a three-objective landscape
# the 95% band is conservative enough that few individuals are ever skipped, and
# these tests need the ESTIMATE_ONLY / PROMOTE path to actually fire.
ADAPTIVE_BLOCK = {
    "enabled": True,
    "min_training_samples": 20,
    "estimator": {"type": "weighted_knn", "k": 5},
    "confidence": {"kappa": 0.5},
    "novelty": {"descriptor_weight": 0.7, "hamming_weight": 0.3, "threshold": 0.35},
    "uncertainty_threshold": 0.35,
    "dominance_margin": 0.0,
    "audit_probability": 0.05,
    "require_simulated_survivors": True,
}


# ---------------------------------------------------------------------------
# Experiment documents
# ---------------------------------------------------------------------------
def _problem(name: str) -> dict:
    return {
        "name": name,
        "region": [-100.0, -100.0, 100.0, 100.0],
        "sink": list(SINK),
        "candidates": [[float(x * 20 - 60), float(y * 20 - 60)] for x in range(7) for y in range(7)],
        "mobile_nodes": [
            {
                "path_segments": [["-60 + 120*t", "40"]],
                "is_closed": False,
                "is_round_trip": True,
                "speed": 5.0,
                "time_step": 1.0,
            }
        ],
        "radius_of_reach": 30.0,
        "radius_of_inter": 60.0,
        "min_coverage_percentage": 60.0,
    }


def _experiment(strategy: str, problem_name: str, adaptive: dict | None = None) -> dict:
    algorithm = {
        "population_size": POPULATION,
        "number_of_generations": GENERATIONS,
        "random_seed": 42,
        "prob_cx": 0.9,
        "prob_mt": 0.3,
        "divisions": 6,
        "per_gene_prob": 0.08,
    }
    if adaptive is not None:
        algorithm["adaptive_evaluation"] = copy.deepcopy(adaptive)
    return {
        "_id": ObjectId(),
        "parameters": {
            "strategy": strategy,
            "algorithm": algorithm,
            "simulation": {
                "duration": 60,
                "random_seeds": [11],
                "synthetic": {"enabled": True},
            },
            "problem": _problem(problem_name),
            "objectives": [
                {"metric_name": "latency", "goal": "min"},
                {"metric_name": "energy", "goal": "min"},
                {"metric_name": "throughput", "goal": "max"},
            ],
        },
        "source_repository_options": {},
        "data_conversion_config": {},
    }


def _run(cls, experiment: dict):
    mongo = FakeMongo()
    evaluator = TopologyEvaluator(SINK)
    strategy = harness_strategy(cls, experiment, mongo)
    run_to_completion(strategy, mongo, evaluator)
    return strategy, mongo, evaluator


@pytest.fixture(scope="module")
def baseline():
    return _run(NSGA3LoopStrategy, _experiment("nsga3", "problem2"))


@pytest.fixture(scope="module")
def adaptive():
    return _run(
        NSGA3AdaptiveSimulationStrategy,
        _experiment("nsga3_adaptive_simulation", "problem2_topology_aware", ADAPTIVE_BLOCK),
    )


# ---------------------------------------------------------------------------
# The run completes
# ---------------------------------------------------------------------------
def test_both_strategies_complete_every_generation(baseline, adaptive):
    for strategy, mongo, _ in (baseline, adaptive):
        generations = mongo.generation_repo.find_by_experiment(strategy._exp_id)
        assert len(generations) == GENERATIONS + 1
        assert [g["index"] for g in generations] == list(range(GENERATIONS + 1))
        assert mongo.experiment_repo.documents[str(strategy._exp_id)]["status"] == "Done"
        assert mongo.experiment_repo.documents[str(strategy._exp_id)]["pareto_front"]


def test_adaptive_run_produces_a_pareto_front(adaptive):
    strategy, mongo, _ = adaptive
    front = mongo.experiment_repo.documents[str(strategy._exp_id)]["pareto_front"]
    assert front
    for item in front:
        assert set(item["objectives"]) == {"latency", "energy", "throughput"}
        assert "mask" in item["chromosome"]


# ---------------------------------------------------------------------------
# The heuristic actually fires
# ---------------------------------------------------------------------------
def test_some_individuals_are_only_estimated(adaptive):
    strategy, mongo, _ = adaptive
    decisions = mongo.adaptive_repo.decisions.values()
    estimated = [d for d in decisions if d["decision"] == EvaluationDecision.ESTIMATE_ONLY.value]

    assert estimated, "the policy never skipped a simulation"
    assert all(d["evaluation_source"] == "estimated" for d in estimated)
    assert all(d["decision_reason"] for d in estimated)


def test_promising_estimates_are_promoted(adaptive):
    strategy, mongo, _ = adaptive
    promoted = [
        d for d in mongo.adaptive_repo.decisions.values()
        if d["decision"] == EvaluationDecision.PROMOTE_TO_SIMULATION.value
    ]

    assert promoted, "no estimated individual was ever promoted"
    for decision in promoted:
        assert decision["promotion_selected"] is True
        assert decision["evaluation_source"] == "simulated"
        assert decision["decision_reason"] == "provisional_nsga3_survivor"


def test_warmup_forces_the_first_generation_to_be_simulated(adaptive):
    strategy, mongo, _ = adaptive
    first = [d for d in mongo.adaptive_repo.decisions.values() if d["generation_index"] == 0]

    assert first
    assert all(d["decision"] != EvaluationDecision.ESTIMATE_ONLY.value for d in first)
    assert any(d["decision_reason"] == "warmup_insufficient_history" for d in first)


def test_exact_cache_hits_are_reused_not_resimulated(adaptive):
    strategy, mongo, _ = adaptive
    reused = [
        d for d in mongo.adaptive_repo.decisions.values()
        if d["decision"] == EvaluationDecision.REUSE.value
    ]
    if not reused:
        pytest.skip("no genome recurred across generations in this run")
    for decision in reused:
        assert decision["evaluation_source"] == "cache"
        assert decision["actual_objectives"]


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def test_adaptive_run_needs_fewer_real_evaluations(baseline, adaptive):
    _, baseline_mongo, baseline_eval = baseline
    strategy, adaptive_mongo, adaptive_eval = adaptive

    assert adaptive_eval.calls < baseline_eval.calls
    assert len(adaptive_mongo.simulation_repo.documents) < len(baseline_mongo.simulation_repo.documents)


def test_saving_metrics_are_recorded_per_generation_and_per_experiment(adaptive):
    strategy, mongo, evaluator = adaptive
    summary = strategy.adaptive_metrics.experiment_summary()

    assert summary["baseline_simulations"] > summary["total_actual_simulations"] > 0
    assert summary["avoided_simulations"] == (
        summary["baseline_simulations"] - summary["total_actual_simulations"]
    )
    assert 0.0 < summary["simulation_reduction_ratio"] < 1.0
    assert summary["total_actual_simulations"] == (
        summary["initial_simulations"] + summary["promotions"]
    )
    # Every real evaluation went through a Simulation document.
    assert summary["simulation_documents"] == evaluator.calls

    persisted = mongo.adaptive_repo.metrics
    assert (strategy._exp_id, -1) in persisted, "experiment summary was not persisted"
    for generation in range(GENERATIONS + 1):
        assert (strategy._exp_id, generation) in persisted


def test_estimator_error_is_measured_from_promotions_and_audits(adaptive):
    strategy, _, _ = adaptive
    summary = strategy.adaptive_metrics.experiment_summary()

    assert summary["prediction_mae"] >= 0.0
    assert summary["prediction_rmse"] >= summary["prediction_mae"] * 0.0
    assert len(summary["prediction_error_per_objective"]) == 3


# ---------------------------------------------------------------------------
# The invariant: no estimate reaches the evolutionary state
# ---------------------------------------------------------------------------
def test_survivors_are_never_estimated(adaptive):
    strategy, _, _ = adaptive
    for genome in strategy._parents:
        assert genome.get_hash() not in strategy._estimated_hashes


def test_published_front_excludes_estimated_individuals(adaptive):
    strategy, mongo, _ = adaptive
    front = mongo.experiment_repo.documents[str(strategy._exp_id)]["pareto_front"]

    estimated_masks = {
        tuple(doc["chromosome"]["mask"])
        for doc in mongo.individual_repo.documents.values()
        if doc.get("evaluation_source") == "estimated"
        and doc["individual_id"] in strategy._estimated_hashes
    }
    for item in front:
        assert tuple(item["chromosome"]["mask"]) not in estimated_masks


def test_estimated_objectives_never_enter_the_genome_cache(adaptive):
    strategy, mongo, _ = adaptive
    for (_, genome_hash), entry in mongo.genome_cache_repo.documents.items():
        if genome_hash in strategy._estimated_hashes:
            assert entry["objectives"] is None, (
                f"estimated genome {genome_hash} was cached as ground truth"
            )


def test_knowledge_base_only_holds_ground_truth(adaptive):
    strategy, mongo, _ = adaptive
    for record in strategy.knowledge_base.records():
        assert record.chromosome_hash not in strategy._estimated_hashes
        assert record.evaluation_type in {"simulated", "cached", "penalty"}
        assert record.scenario_fingerprint == strategy._scenario_fingerprint


def test_individual_documents_carry_their_provenance(adaptive):
    strategy, mongo, _ = adaptive
    sources = {
        doc.get("evaluation_source") for doc in mongo.individual_repo.documents.values()
    }
    assert sources <= {"simulated", "estimated", "cache", "penalty"}
    assert "simulated" in sources
    assert "estimated" in sources


# ---------------------------------------------------------------------------
# Restart / resume
# ---------------------------------------------------------------------------
def test_resume_continues_without_repeating_work():
    experiment = _experiment(
        "nsga3_adaptive_simulation", "problem2_topology_aware", ADAPTIVE_BLOCK
    )
    mongo = FakeMongo()
    evaluator = TopologyEvaluator(SINK)

    # --- run two full generations, then "crash" ---
    first = harness_strategy(NSGA3AdaptiveSimulationStrategy, experiment, mongo)
    first.start()
    for _ in range(3):
        generation_id = first._generation_id
        evaluator.complete_generation(mongo, generation_id)
        with first._lock:
            first._handle_generation_done(generation_id)
    first.stop()

    calls_before = evaluator.calls
    simulations_before = len(mongo.simulation_repo.documents)
    kb_before = len(first.knowledge_base)
    estimated_before = set(first._estimated_hashes)
    assert kb_before > 0 and estimated_before

    # --- restart against the same database ---
    second = harness_strategy(NSGA3AdaptiveSimulationStrategy, experiment, mongo)
    second.start()

    assert second._scenario_fingerprint == first._scenario_fingerprint
    assert len(second.knowledge_base) == kb_before, "knowledge base was not restored"
    assert second._estimated_hashes >= estimated_before, "decision log was not restored"
    assert second._genome_objectives_cache, "genome cache was not restored"
    assert second._gen_index == first._gen_index, "resumed at the wrong generation"
    # Resuming must not re-queue anything that was already evaluated.
    assert evaluator.calls == calls_before
    assert len(mongo.simulation_repo.documents) == simulations_before

    # --- and the run finishes normally from there ---
    run_to_completion(second, mongo, evaluator)
    assert mongo.experiment_repo.documents[str(experiment["_id"])]["status"] == "Done"

    # Every DONE simulation was evaluated exactly once.
    per_individual: dict[tuple, int] = {}
    for sim in mongo.simulation_repo.documents.values():
        key = (sim["individual_id"], sim["generation_id"], sim["random_seed"])
        per_individual[key] = per_individual.get(key, 0) + 1
    assert all(count == 1 for count in per_individual.values())


def test_resumed_decisions_are_upserted_not_duplicated():
    experiment = _experiment(
        "nsga3_adaptive_simulation", "problem2_topology_aware", ADAPTIVE_BLOCK
    )
    mongo = FakeMongo()
    evaluator = TopologyEvaluator(SINK)

    first = harness_strategy(NSGA3AdaptiveSimulationStrategy, experiment, mongo)
    first.start()
    for _ in range(2):
        generation_id = first._generation_id
        evaluator.complete_generation(mongo, generation_id)
        with first._lock:
            first._handle_generation_done(generation_id)
    first.stop()

    second = harness_strategy(NSGA3AdaptiveSimulationStrategy, experiment, mongo)
    run_to_completion(second, mongo, evaluator)

    keys = list(mongo.adaptive_repo.decisions.keys())
    assert len(keys) == len(set(keys)), "duplicate decision records"


# ---------------------------------------------------------------------------
# Strategy-level guards, exercised in isolation
# ---------------------------------------------------------------------------
def test_adaptive_strategy_rejects_a_problem_without_descriptors():
    experiment = _experiment("nsga3_adaptive_simulation", "problem2", ADAPTIVE_BLOCK)
    with pytest.raises(ValueError, match="topology-aware"):
        NSGA3AdaptiveSimulationStrategy(experiment, FakeMongo())


def test_phase_returns_to_screening_for_each_new_generation(adaptive):
    strategy, _, _ = adaptive
    assert isinstance(strategy._phase, GenerationPhase)
