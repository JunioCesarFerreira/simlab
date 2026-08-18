"""Adaptive evaluation policy: who gets simulated, who only gets estimated.

The policy is a pure function of (descriptors, chromosome, knowledge base,
configuration, RNG).  It owns no MongoDB state and no evolutionary state — the
strategy orchestrates, the policy decides — which makes every decision
reproducible from the experiment seed and auditable after the fact.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .dominance import dominated_by_any, lower_bound, upper_bound
from .estimator import ObjectiveEstimator, WeightedKNNEstimator
from .knowledge_base import EvaluationKnowledgeBase

log = logging.getLogger(__name__)


class EvaluationDecision(str, Enum):
    """What the strategy must do with an individual."""

    REUSE = "REUSE"                                  # exact cache hit, no simulation
    SIMULATE = "SIMULATE"                            # full simulation, all seeds
    SIMULATE_REDUCED = "SIMULATE_REDUCED"            # reserved: fewer seeds (not emitted yet)
    ESTIMATE_ONLY = "ESTIMATE_ONLY"                  # provisional, estimator-provided
    PROMOTE_TO_SIMULATION = "PROMOTE_TO_SIMULATION"  # estimated individual that mattered


class DecisionReason(str, Enum):
    """Why the policy took a decision (persisted for the thesis experiments)."""

    EXACT_CACHE_HIT = "exact_cache_hit"
    WARMUP = "warmup_insufficient_history"
    NO_PREDICTION = "estimator_unavailable"
    HIGH_NOVELTY = "high_novelty"
    HIGH_UNCERTAINTY = "high_uncertainty"
    POTENTIALLY_NONDOMINATED = "optimistic_bound_not_dominated"
    OPTIMISTIC_DOMINATED = "optimistic_bound_dominated"
    AUDIT_SAMPLE = "audit_sample"
    BUDGET_EXHAUSTED = "simulation_budget_exhausted"
    PROVISIONAL_SURVIVOR = "provisional_nsga3_survivor"
    PROVISIONAL_FRONT = "provisional_nsga3_first_front"


# Priority classes used when a per-generation simulation budget is active.
# Lower value == simulated first (design note 24: promotion, near-front,
# uncertainty, novelty, audit).
_PRIORITY: dict[DecisionReason, int] = {
    DecisionReason.WARMUP: 0,
    DecisionReason.NO_PREDICTION: 0,
    DecisionReason.POTENTIALLY_NONDOMINATED: 1,
    DecisionReason.HIGH_UNCERTAINTY: 2,
    DecisionReason.HIGH_NOVELTY: 3,
    DecisionReason.AUDIT_SAMPLE: 4,
}


@dataclass(frozen=True, slots=True)
class SimulationBudget:
    enabled: bool = False
    min_per_generation: int = 5
    max_per_generation: int = 20
    promotion_reserve: int = 5

    @property
    def screening_cap(self) -> int:
        """Slots available before the promotion reserve is set aside."""
        return max(self.min_per_generation, self.max_per_generation - self.promotion_reserve)


@dataclass(frozen=True, slots=True)
class AdaptiveEvaluationConfig:
    """Everything the heuristic exposes to the experiment document."""

    enabled: bool = True
    min_training_samples: int = 40
    estimator_type: str = "weighted_knn"
    estimator_k: int = 7
    estimator_epsilon: float = 1e-9
    kappa: float = 1.96
    novelty_descriptor_weight: float = 0.7
    novelty_hamming_weight: float = 0.3
    novelty_threshold: float = 0.40
    uncertainty_threshold: float = 0.25
    dominance_margin: float = 0.02
    audit_probability: float = 0.05
    require_simulated_survivors: bool = True
    budget: SimulationBudget = field(default_factory=SimulationBudget)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "AdaptiveEvaluationConfig":
        """Build from the ``algorithm.adaptive_evaluation`` block."""
        cfg = dict(data or {})
        estimator = dict(cfg.get("estimator") or {})
        confidence = dict(cfg.get("confidence") or {})
        novelty = dict(cfg.get("novelty") or {})
        budget = dict(cfg.get("simulation_budget") or {})
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            min_training_samples=int(cfg.get("min_training_samples", 40)),
            estimator_type=str(estimator.get("type", "weighted_knn")),
            estimator_k=int(estimator.get("k", 7)),
            estimator_epsilon=float(estimator.get("epsilon", 1e-9)),
            kappa=float(confidence.get("kappa", 1.96)),
            novelty_descriptor_weight=float(novelty.get("descriptor_weight", 0.7)),
            novelty_hamming_weight=float(novelty.get("hamming_weight", 0.3)),
            novelty_threshold=float(novelty.get("threshold", 0.40)),
            uncertainty_threshold=float(cfg.get("uncertainty_threshold", 0.25)),
            dominance_margin=float(cfg.get("dominance_margin", 0.02)),
            audit_probability=float(cfg.get("audit_probability", 0.05)),
            require_simulated_survivors=bool(cfg.get("require_simulated_survivors", True)),
            budget=SimulationBudget(
                enabled=bool(budget.get("enabled", False)),
                min_per_generation=int(budget.get("min_per_generation", 5)),
                max_per_generation=int(budget.get("max_per_generation", 20)),
                promotion_reserve=int(budget.get("promotion_reserve", 5)),
            ),
        )


@dataclass(slots=True)
class DecisionRecord:
    """Full, persistable trace of one evaluation decision."""

    individual_id: str
    generation: int
    decision: EvaluationDecision
    reason: DecisionReason
    predicted_objectives: Optional[list[float]] = None
    uncertainty: Optional[list[float]] = None
    optimistic_objectives: Optional[list[float]] = None
    conservative_objectives: Optional[list[float]] = None
    novelty: Optional[float] = None
    nearest_neighbor_distance: Optional[float] = None
    nearest_hamming_distance: Optional[float] = None
    normalized_uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    dominance_result: Optional[bool] = None
    audit_selected: bool = False
    promotion_selected: bool = False
    actual_objectives: Optional[list[float]] = None
    evaluation_source: str = "pending"
    descriptors: dict[str, float] = field(default_factory=dict)

    @property
    def priority(self) -> int:
        if self.promotion_selected:
            return -1
        return _PRIORITY.get(self.reason, 5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "individual_id": self.individual_id,
            "generation": self.generation,
            "decision": self.decision.value,
            "decision_reason": self.reason.value,
            "predicted_objectives": self.predicted_objectives,
            "uncertainty": self.uncertainty,
            "optimistic_objectives": self.optimistic_objectives,
            "conservative_objectives": self.conservative_objectives,
            "novelty": self.novelty,
            "nearest_neighbor_distance": self.nearest_neighbor_distance,
            "nearest_hamming_distance": self.nearest_hamming_distance,
            "normalized_uncertainty": self.normalized_uncertainty,
            "confidence": self.confidence,
            "dominance_result": self.dominance_result,
            "audit_selected": self.audit_selected,
            "promotion_selected": self.promotion_selected,
            "actual_objectives": self.actual_objectives,
            "evaluation_source": self.evaluation_source,
            "descriptors": self.descriptors,
        }

    def log_line(self) -> str:
        parts = [
            f"generation={self.generation}",
            f"individual={self.individual_id[:12]}",
            f"decision={self.decision.value}",
            f"reason={self.reason.value}",
        ]
        if self.nearest_neighbor_distance is not None:
            parts.append(f"nearest_distance={self.nearest_neighbor_distance:.4f}")
        if self.novelty is not None:
            parts.append(f"novelty={self.novelty:.4f}")
        return "[adaptive-eval] " + " ".join(parts)


def build_estimator(config: AdaptiveEvaluationConfig) -> ObjectiveEstimator:
    """Instantiate the estimator named by the configuration."""
    if config.estimator_type == "weighted_knn":
        return WeightedKNNEstimator(k=config.estimator_k, epsilon=config.estimator_epsilon)
    raise ValueError(
        f"Unknown estimator type {config.estimator_type!r} (known: 'weighted_knn')."
    )


class AdaptiveEvaluationPolicy:
    """Decides, per individual, whether a real simulation is worth its cost."""

    def __init__(
        self,
        config: AdaptiveEvaluationConfig,
        knowledge_base: EvaluationKnowledgeBase,
        rng: Random,
        estimator: Optional[ObjectiveEstimator] = None,
    ) -> None:
        self.config = config
        self.kb = knowledge_base
        self.rng = rng
        self.estimator: ObjectiveEstimator = estimator or build_estimator(config)
        self._fitted_size = -1

    # ------------------------------------------------------------------
    def refit(self, force: bool = False) -> None:
        """Retrain the estimator when the knowledge base has grown."""
        size = self.kb.training_size
        if not force and size == self._fitted_size:
            return
        X, Y = self.kb.training_arrays()
        self.estimator.fit(X, Y)
        self._fitted_size = size
        log.debug("[adaptive-eval] Estimator refit on %d samples.", size)

    @property
    def is_warm(self) -> bool:
        """Whether enough ground truth exists to trust an estimate."""
        return self.kb.training_size >= self.config.min_training_samples

    # ------------------------------------------------------------------
    def decide(
        self,
        individual_id: str,
        generation: int,
        descriptor_vector: Sequence[float],
        mask: Sequence[int],
        descriptors: Optional[Mapping[str, float]] = None,
    ) -> DecisionRecord:
        """Phase-A decision for one new individual."""
        base = dict(descriptors or {})

        if not self.config.enabled:
            return DecisionRecord(
                individual_id=individual_id,
                generation=generation,
                decision=EvaluationDecision.SIMULATE,
                reason=DecisionReason.WARMUP,
                descriptors=base,
            )

        novelty, d_phi, d_ham = self._novelty(descriptor_vector, mask)
        base = dict(base)
        base.setdefault("nearest_evaluated_descriptor_distance", d_phi)
        base.setdefault("nearest_evaluated_hamming_distance", d_ham)

        if not self.is_warm:
            return DecisionRecord(
                individual_id=individual_id,
                generation=generation,
                decision=EvaluationDecision.SIMULATE,
                reason=DecisionReason.WARMUP,
                novelty=novelty,
                nearest_neighbor_distance=d_phi,
                nearest_hamming_distance=d_ham,
                descriptors=base,
            )

        self.refit()
        prediction = self.estimator.predict(descriptor_vector)
        if prediction is None:
            return DecisionRecord(
                individual_id=individual_id,
                generation=generation,
                decision=EvaluationDecision.SIMULATE,
                reason=DecisionReason.NO_PREDICTION,
                novelty=novelty,
                nearest_neighbor_distance=d_phi,
                nearest_hamming_distance=d_ham,
                descriptors=base,
            )

        ranges = self.kb.objective_ranges()
        lower = lower_bound(prediction.mean, prediction.uncertainty, self.config.kappa)
        upper = upper_bound(prediction.mean, prediction.uncertainty, self.config.kappa)
        norm_sigma = float(np.mean(np.asarray(prediction.uncertainty, dtype=float) / ranges))
        margin = self.config.dominance_margin * ranges
        dominated = dominated_by_any(lower, self.kb.known_front(), margin)

        record = DecisionRecord(
            individual_id=individual_id,
            generation=generation,
            decision=EvaluationDecision.SIMULATE,
            reason=DecisionReason.POTENTIALLY_NONDOMINATED,
            predicted_objectives=[float(v) for v in prediction.mean],
            uncertainty=[float(v) for v in prediction.uncertainty],
            optimistic_objectives=[float(v) for v in lower],
            conservative_objectives=[float(v) for v in upper],
            novelty=novelty,
            nearest_neighbor_distance=d_phi,
            nearest_hamming_distance=d_ham,
            normalized_uncertainty=norm_sigma,
            confidence=prediction.confidence,
            dominance_result=dominated,
            descriptors=base,
        )

        # Exploration first: an unfamiliar or badly-resolved region is
        # simulated regardless of what the (untrustworthy) estimate says.
        if novelty > self.config.novelty_threshold:
            record.reason = DecisionReason.HIGH_NOVELTY
            return record
        if norm_sigma > self.config.uncertainty_threshold:
            record.reason = DecisionReason.HIGH_UNCERTAINTY
            return record

        if not dominated:
            record.reason = DecisionReason.POTENTIALLY_NONDOMINATED
            return record

        # Even the optimistic bound is clearly dominated: skip the simulation,
        # unless this individual is drawn for the audit sample.
        if self.rng.random() < self.config.audit_probability:
            record.decision = EvaluationDecision.SIMULATE
            record.reason = DecisionReason.AUDIT_SAMPLE
            record.audit_selected = True
            return record

        record.decision = EvaluationDecision.ESTIMATE_ONLY
        record.reason = DecisionReason.OPTIMISTIC_DOMINATED
        return record

    # ------------------------------------------------------------------
    def apply_budget(self, records: Sequence[DecisionRecord]) -> list[DecisionRecord]:
        """Demote the lowest-priority ``SIMULATE`` decisions past the budget.

        Individuals with no usable estimate (warm-up, estimator unavailable)
        are never demoted: there would be nothing to fall back on.
        """
        budget = self.config.budget
        if not budget.enabled:
            return list(records)

        simulate = [r for r in records if r.decision == EvaluationDecision.SIMULATE]
        demotable = [r for r in simulate if r.conservative_objectives is not None]
        keep = budget.screening_cap
        if len(simulate) <= keep or not demotable:
            return list(records)

        ordered = sorted(demotable, key=lambda r: (r.priority, r.individual_id))
        surplus = len(simulate) - keep
        for record in reversed(ordered):
            if surplus <= 0:
                break
            record.decision = EvaluationDecision.ESTIMATE_ONLY
            record.reason = DecisionReason.BUDGET_EXHAUSTED
            record.audit_selected = False
            surplus -= 1
        return list(records)

    # ------------------------------------------------------------------
    def _novelty(
        self, descriptor_vector: Sequence[float], mask: Sequence[int]
    ) -> tuple[float, float, float]:
        """``N(x) = lambda d_phi + (1 - lambda) d_H`` plus its two components."""
        d_phi = self.kb.nearest_descriptor_distance(descriptor_vector)
        d_ham = self.kb.nearest_hamming(mask)
        wd = self.config.novelty_descriptor_weight
        wh = self.config.novelty_hamming_weight
        total = wd + wh
        if total <= 0:
            return 0.0, d_phi, d_ham
        return (wd * d_phi + wh * d_ham) / total, d_phi, d_ham
