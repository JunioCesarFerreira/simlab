from typing import Any, NotRequired, Optional, TypedDict
from datetime import datetime
from bson import ObjectId


class AdaptiveEvaluation(TypedDict):
    """One reproducible decision of the adaptive-simulation heuristic.

    Written by ``NSGA3AdaptiveSimulationStrategy`` for every individual it
    screens, so an experiment can be replayed and audited after the fact:
    which individuals were really simulated, which were only estimated, why,
    and — for the promoted/audited ones — how far the estimate actually was.

    Optional fields stay ``None`` when the corresponding stage did not run
    (e.g. no prediction during the warm-up phase).
    """

    experiment_id: ObjectId
    scenario_fingerprint: str
    generation_index: int
    individual_id: str                       # chromosome hash
    decision: str                            # EvaluationDecision value
    decision_reason: str                     # DecisionReason value
    evaluation_source: str                   # "simulated" | "estimated" | "cache" | "penalty"
    created_at: datetime
    predicted_objectives: Optional[list[float]]
    uncertainty: Optional[list[float]]
    optimistic_objectives: Optional[list[float]]
    conservative_objectives: Optional[list[float]]
    actual_objectives: Optional[list[float]]
    novelty: Optional[float]
    nearest_neighbor_distance: Optional[float]
    nearest_hamming_distance: Optional[float]
    normalized_uncertainty: Optional[float]
    confidence: Optional[float]
    dominance_result: Optional[bool]
    audit_selected: bool
    promotion_selected: bool
    descriptors: NotRequired[dict[str, float]]


class AdaptiveMetrics(TypedDict):
    """Per-generation (and per-experiment) cost accounting of the heuristic."""

    experiment_id: ObjectId
    scenario_fingerprint: str
    generation_index: int                    # -1 for the experiment-level summary
    created_at: datetime
    metrics: dict[str, Any]
