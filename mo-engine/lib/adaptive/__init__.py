"""Adaptive simulation-budget allocation for simulation-based optimization.

The package is problem-agnostic: it consumes descriptor vectors and objective
vectors in minimization space, and knows nothing about P2, Cooja or MongoDB.

    dominance      -- min/max conversion and margin-aware dominance tests
    estimator      -- replaceable objective estimator (weighted k-NN today)
    knowledge_base -- the historical base of really-evaluated individuals
    policy         -- SIMULATE / ESTIMATE_ONLY / REUSE decision logic
    metrics        -- how much simulation cost was actually saved
"""
from .dominance import (
    dominated_by_any,
    dominates,
    dominates_clearly,
    goal_signs,
    lower_bound,
    non_dominated,
    objective_ranges,
    to_minimization,
    to_original,
    upper_bound,
)
from .estimator import (
    EXACT_MATCH_TOLERANCE,
    ObjectiveEstimator,
    ObjectivePrediction,
    WeightedKNNEstimator,
)
from .knowledge_base import (
    SENTINEL_THRESHOLD,
    EvaluationKnowledgeBase,
    EvaluationRecord,
)
from .metrics import (
    AdaptiveMetricsTracker,
    GenerationAdaptiveMetrics,
    PredictionSample,
)
from .policy import (
    AdaptiveEvaluationConfig,
    AdaptiveEvaluationPolicy,
    DecisionReason,
    DecisionRecord,
    EvaluationDecision,
    SimulationBudget,
    build_estimator,
)

__all__ = [
    "AdaptiveEvaluationConfig",
    "AdaptiveEvaluationPolicy",
    "AdaptiveMetricsTracker",
    "DecisionReason",
    "DecisionRecord",
    "EXACT_MATCH_TOLERANCE",
    "EvaluationDecision",
    "EvaluationKnowledgeBase",
    "EvaluationRecord",
    "GenerationAdaptiveMetrics",
    "ObjectiveEstimator",
    "ObjectivePrediction",
    "PredictionSample",
    "SENTINEL_THRESHOLD",
    "SimulationBudget",
    "WeightedKNNEstimator",
    "build_estimator",
    "dominated_by_any",
    "dominates",
    "dominates_clearly",
    "goal_signs",
    "lower_bound",
    "non_dominated",
    "objective_ranges",
    "to_minimization",
    "to_original",
    "upper_bound",
]
