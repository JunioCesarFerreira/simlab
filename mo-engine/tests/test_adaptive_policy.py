"""Knowledge base + decision policy.

Every branch of the policy is pinned here, because the whole scientific claim
rests on *which* individuals get a real simulation:

    warm-up            -> SIMULATE
    exact cache        -> REUSE
    high novelty       -> SIMULATE
    high uncertainty   -> SIMULATE
    optimistic dominated -> ESTIMATE_ONLY
    audit draw         -> SIMULATE
"""
import random

import numpy as np
import pytest

from lib.adaptive import (
    AdaptiveEvaluationConfig,
    AdaptiveEvaluationPolicy,
    DecisionReason,
    EvaluationDecision,
    EvaluationKnowledgeBase,
    EvaluationRecord,
    SimulationBudget,
)

FINGERPRINT = "scenario-a"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _record(index: int, descriptor: list[float], objectives: list[float], fingerprint=FINGERPRINT):
    return EvaluationRecord(
        scenario_fingerprint=fingerprint,
        chromosome_hash=f"h{index}",
        chromosome={"mac_protocol": 0, "mask": [(index >> b) & 1 for b in range(8)]},
        descriptors={"d0": descriptor[0]},
        descriptor_vector=tuple(descriptor),
        objectives=tuple(objectives),
    )


def _populated_kb(n: int = 30, spread: float = 0.0) -> EvaluationKnowledgeBase:
    """A base whose objectives are a smooth function of the descriptors."""
    kb = EvaluationKnowledgeBase(FINGERPRINT)
    for i in range(n):
        t = i / max(1, n - 1)
        kb.add(_record(i, [t, 1.0 - t], [10.0 + 10.0 * t + spread * (i % 2), 20.0 - 5.0 * t]))
    return kb


def _config(**overrides) -> AdaptiveEvaluationConfig:
    base = dict(
        min_training_samples=10,
        estimator_k=3,
        kappa=1.0,
        novelty_threshold=0.90,
        uncertainty_threshold=0.90,
        dominance_margin=0.0,
        audit_probability=0.0,
    )
    base.update(overrides)
    return AdaptiveEvaluationConfig(**base)


def _policy(kb, seed: int = 5, **overrides) -> AdaptiveEvaluationPolicy:
    return AdaptiveEvaluationPolicy(_config(**overrides), kb, random.Random(seed))


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
class TestKnowledgeBase:
    def test_rejects_records_from_another_scenario(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        assert kb.add(_record(0, [0.0, 0.0], [1.0, 1.0]))
        assert not kb.add(_record(1, [0.0, 0.0], [1.0, 1.0], fingerprint="scenario-b"))
        assert len(kb) == 1

    def test_is_idempotent_on_the_chromosome_hash(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        assert kb.add(_record(0, [0.0, 0.0], [1.0, 1.0]))
        assert not kb.add(_record(0, [9.9, 9.9], [7.0, 7.0]))

    def test_penalty_records_are_kept_but_never_trained_on(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        kb.add(_record(0, [0.0, 0.0], [1.0, 1.0]))
        kb.add(_record(1, [1.0, 1.0], [1e9, 1e9]))  # infeasibility penalty

        assert len(kb) == 2
        assert kb.training_size == 1
        X, Y = kb.training_arrays()
        assert X.shape[0] == 1
        assert not any(v > 1e8 for row in Y for v in row)

    def test_known_front_is_the_non_dominated_measured_set(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        kb.add(_record(0, [0.0, 0.0], [1.0, 9.0]))
        kb.add(_record(1, [1.0, 0.0], [9.0, 1.0]))
        kb.add(_record(2, [0.5, 0.5], [5.0, 5.0]))
        kb.add(_record(3, [0.6, 0.6], [6.0, 6.0]))  # dominated by record 2

        front = kb.known_front()
        assert (6.0, 6.0) not in front
        assert len(front) == 3

    def test_novelty_of_an_empty_base_is_maximal(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        assert kb.nearest_hamming([0, 1, 0]) == 1.0
        assert kb.nearest_descriptor_distance([0.0, 0.0]) == 1.0

    def test_hamming_distance_is_normalised(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        kb.add(_record(0, [0.0, 0.0], [1.0, 1.0]))  # mask of index 0 -> all zeros
        assert kb.nearest_hamming([0] * 8) == 0.0
        assert kb.nearest_hamming([1, 1, 0, 0, 0, 0, 0, 0]) == pytest.approx(0.25)

    def test_rebuild_from_genome_cache(self):
        class _Descriptors:
            def __init__(self, mask):
                self._n = sum(mask)

            def vector(self):
                return np.asarray([float(self._n)])

            def as_dict(self):
                return {"active_relays": float(self._n)}

        entries = [
            {"genome_hash": "a", "chromosome": {"mask": [1, 0, 1]}, "objectives": [1.0, 2.0]},
            {"genome_hash": "b", "chromosome": {"mask": [1, 1, 1]}, "objectives": [3.0, 4.0]},
            {"genome_hash": "c", "chromosome": {"mask": [0, 0, 0]}, "objectives": None},  # unfinished
        ]
        kb = EvaluationKnowledgeBase.from_genome_cache(
            FINGERPRINT, entries, lambda chrom: _Descriptors(chrom["mask"])
        )

        assert len(kb) == 2
        assert kb.get("a").objectives == (1.0, 2.0)
        assert kb.get("c") is None


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------
class TestDecisionPolicy:
    def test_warmup_forces_simulation(self):
        kb = _populated_kb(n=3)
        policy = _policy(kb, min_training_samples=10)

        decision = policy.decide("x", 0, [0.5, 0.5], [0, 1, 0, 1])

        assert decision.decision == EvaluationDecision.SIMULATE
        assert decision.reason == DecisionReason.WARMUP
        assert not policy.is_warm

    def test_disabled_heuristic_always_simulates(self):
        policy = _policy(_populated_kb(), enabled=False)
        decision = policy.decide("x", 5, [0.5, 0.5], [0, 1])
        assert decision.decision == EvaluationDecision.SIMULATE

    def test_high_novelty_forces_simulation(self):
        kb = _populated_kb()
        policy = _policy(kb, novelty_threshold=0.01)

        decision = policy.decide("x", 5, [0.5, 0.5], [1] * 8)

        assert decision.decision == EvaluationDecision.SIMULATE
        assert decision.reason == DecisionReason.HIGH_NOVELTY
        assert decision.novelty is not None and decision.novelty > 0.01

    def test_high_uncertainty_forces_simulation(self):
        # Alternating objectives make neighbouring descriptors disagree.
        kb = _populated_kb(n=30, spread=8.0)
        policy = _policy(kb, uncertainty_threshold=0.01, novelty_threshold=0.99)

        decision = policy.decide("x", 5, [0.5, 0.5], [0] * 8)

        assert decision.decision == EvaluationDecision.SIMULATE
        assert decision.reason == DecisionReason.HIGH_UNCERTAINTY

    def test_optimistically_dominated_individual_is_only_estimated(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        # A tight cluster of excellent solutions ...
        for i in range(12):
            kb.add(_record(i, [0.0 + 0.001 * i, 0.0], [1.0, 1.0]))
        # ... and a cluster of clearly worse ones near the query point.
        for i in range(12, 24):
            kb.add(_record(i, [1.0 - 0.001 * i, 1.0], [50.0, 50.0]))

        policy = _policy(kb, novelty_threshold=0.99, uncertainty_threshold=0.99)
        decision = policy.decide("x", 5, [0.99, 1.0], [1] * 8)

        assert decision.decision == EvaluationDecision.ESTIMATE_ONLY
        assert decision.reason == DecisionReason.OPTIMISTIC_DOMINATED
        assert decision.dominance_result is True
        assert decision.conservative_objectives is not None
        assert decision.optimistic_objectives is not None
        # U(x) is never better than L(x): the band brackets the prediction.
        assert all(
            u >= m >= l for u, m, l in zip(
                decision.conservative_objectives,
                decision.predicted_objectives,
                decision.optimistic_objectives,
            )
        )

    def test_potentially_nondominated_individual_is_simulated(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        for i in range(12):
            kb.add(_record(i, [0.0 + 0.001 * i, 0.0], [50.0, 50.0]))
        for i in range(12, 24):
            kb.add(_record(i, [1.0 - 0.001 * i, 1.0], [1.0, 1.0]))

        policy = _policy(kb, novelty_threshold=0.99, uncertainty_threshold=0.99)
        decision = policy.decide("x", 5, [0.99, 1.0], [1] * 8)

        assert decision.decision == EvaluationDecision.SIMULATE
        assert decision.reason == DecisionReason.POTENTIALLY_NONDOMINATED
        assert decision.dominance_result is False

    def test_audit_sampling_rescues_a_skipped_individual(self):
        kb = EvaluationKnowledgeBase(FINGERPRINT)
        for i in range(12):
            kb.add(_record(i, [0.0 + 0.001 * i, 0.0], [1.0, 1.0]))
        for i in range(12, 24):
            kb.add(_record(i, [1.0 - 0.001 * i, 1.0], [50.0, 50.0]))

        skipping = _policy(kb, novelty_threshold=0.99, uncertainty_threshold=0.99)
        assert skipping.decide("x", 5, [0.99, 1.0], [1] * 8).decision == EvaluationDecision.ESTIMATE_ONLY

        auditing = _policy(
            kb, novelty_threshold=0.99, uncertainty_threshold=0.99, audit_probability=1.0
        )
        decision = auditing.decide("x", 5, [0.99, 1.0], [1] * 8)

        assert decision.decision == EvaluationDecision.SIMULATE
        assert decision.reason == DecisionReason.AUDIT_SAMPLE
        assert decision.audit_selected is True

    def test_audit_draws_are_reproducible_from_the_seed(self):
        kb = _populated_kb()
        left = [
            _policy(kb, seed=99, novelty_threshold=0.99, uncertainty_threshold=0.99,
                    audit_probability=0.5).decide(f"x{i}", 5, [0.5, 0.5], [0] * 8).reason
            for i in range(20)
        ]
        right = [
            _policy(kb, seed=99, novelty_threshold=0.99, uncertainty_threshold=0.99,
                    audit_probability=0.5).decide(f"x{i}", 5, [0.5, 0.5], [0] * 8).reason
            for i in range(20)
        ]
        assert left == right

    def test_decision_record_is_serialisable_and_loggable(self):
        policy = _policy(_populated_kb())
        decision = policy.decide("abc123def456", 4, [0.5, 0.5], [0, 1, 0, 1])
        payload = decision.to_dict()

        assert payload["individual_id"] == "abc123def456"
        assert payload["generation"] == 4
        assert payload["decision"] in {d.value for d in EvaluationDecision}
        assert "[adaptive-eval]" in decision.log_line()
        assert "decision=" in decision.log_line()


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
class TestSimulationBudget:
    def _decisions(self, policy, n: int):
        return [policy.decide(f"x{i}", 3, [i / n, 1 - i / n], [0] * 8) for i in range(n)]

    def test_budget_disabled_keeps_every_decision(self):
        policy = _policy(_populated_kb())
        decisions = self._decisions(policy, 12)
        assert policy.apply_budget(decisions) == decisions

    def test_budget_demotes_the_lowest_priority_simulations(self):
        kb = _populated_kb()
        policy = _policy(
            kb,
            novelty_threshold=0.99,
            uncertainty_threshold=0.99,
            budget=SimulationBudget(enabled=True, min_per_generation=2,
                                    max_per_generation=4, promotion_reserve=1),
        )
        decisions = self._decisions(policy, 12)
        for d in decisions:  # make every one demotable
            d.decision = EvaluationDecision.SIMULATE
            d.conservative_objectives = [1.0, 1.0]
            d.reason = DecisionReason.POTENTIALLY_NONDOMINATED

        policy.apply_budget(decisions)
        simulated = [d for d in decisions if d.decision == EvaluationDecision.SIMULATE]

        assert len(simulated) == 3  # max 4 minus the promotion reserve
        demoted = [d for d in decisions if d.decision == EvaluationDecision.ESTIMATE_ONLY]
        assert all(d.reason == DecisionReason.BUDGET_EXHAUSTED for d in demoted)

    def test_budget_never_demotes_an_individual_without_an_estimate(self):
        kb = _populated_kb(n=3)  # warm-up: no predictions available
        policy = _policy(
            kb,
            min_training_samples=100,
            budget=SimulationBudget(enabled=True, min_per_generation=1,
                                    max_per_generation=2, promotion_reserve=1),
        )
        decisions = self._decisions(policy, 10)
        policy.apply_budget(decisions)

        assert all(d.decision == EvaluationDecision.SIMULATE for d in decisions)


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------
class TestConfiguration:
    def test_defaults_are_documented_values(self):
        cfg = AdaptiveEvaluationConfig.from_mapping(None)
        assert cfg.enabled is True
        assert cfg.min_training_samples == 40
        assert cfg.kappa == pytest.approx(1.96)
        assert cfg.require_simulated_survivors is True
        assert cfg.budget.enabled is False

    def test_nested_blocks_are_parsed(self):
        cfg = AdaptiveEvaluationConfig.from_mapping({
            "enabled": True,
            "min_training_samples": 50,
            "estimator": {"type": "weighted_knn", "k": 9, "epsilon": 1e-6},
            "confidence": {"kappa": 2.5},
            "novelty": {"descriptor_weight": 0.6, "hamming_weight": 0.4, "threshold": 0.33},
            "uncertainty_threshold": 0.15,
            "dominance_margin": 0.05,
            "audit_probability": 0.2,
            "simulation_budget": {
                "enabled": True, "min_per_generation": 3,
                "max_per_generation": 11, "promotion_reserve": 4,
            },
            "require_simulated_survivors": False,
        })

        assert cfg.estimator_k == 9
        assert cfg.kappa == pytest.approx(2.5)
        assert cfg.novelty_threshold == pytest.approx(0.33)
        assert cfg.dominance_margin == pytest.approx(0.05)
        assert cfg.require_simulated_survivors is False
        assert cfg.budget.screening_cap == 7

    def test_unknown_estimator_is_rejected(self):
        from lib.adaptive import build_estimator

        with pytest.raises(ValueError):
            build_estimator(AdaptiveEvaluationConfig(estimator_type="deep_magic"))
