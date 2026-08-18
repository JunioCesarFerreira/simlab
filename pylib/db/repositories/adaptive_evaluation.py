import logging
from datetime import datetime
from typing import Any, Optional
from bson import ObjectId

from pylib.db.models.adaptive_evaluation import AdaptiveEvaluation, AdaptiveMetrics
from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)

# Sentinel generation index of the experiment-level metrics summary.
EXPERIMENT_SUMMARY_INDEX = -1


class AdaptiveEvaluationRepository:
    """Persists the decision trace and the cost metrics of adaptive strategies.

    Two small collections, both keyed so that a restart can neither duplicate a
    decision nor lose one:

    * ``adaptive_evaluations`` — one document per (experiment, generation,
      individual); upserted, so replaying a generation after a crash overwrites
      instead of appending.
    * ``adaptive_metrics`` — one document per (experiment, generation) plus one
      experiment-level summary at index ``-1``.

    Nothing else in SimLab reads these collections: they exist purely for the
    reproducibility and cost-comparison experiments.
    """

    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["adaptive_evaluations"].create_index(
                [("experiment_id", 1), ("generation_index", 1), ("individual_id", 1)],
                unique=True,
                name="idx_adaptive_eval_exp_gen_ind",
            )
            db["adaptive_evaluations"].create_index(
                [("experiment_id", 1), ("decision", 1)],
                name="idx_adaptive_eval_exp_decision",
            )
            db["adaptive_metrics"].create_index(
                [("experiment_id", 1), ("generation_index", 1)],
                unique=True,
                name="idx_adaptive_metrics_exp_gen",
            )

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------
    def upsert_decision(
        self,
        experiment_id: ObjectId,
        scenario_fingerprint: str,
        generation_index: int,
        decision: dict[str, Any],
    ) -> None:
        """Insert or replace the decision of one individual in one generation."""
        doc: AdaptiveEvaluation = {
            "experiment_id": experiment_id,
            "scenario_fingerprint": scenario_fingerprint,
            "generation_index": int(generation_index),
            "created_at": datetime.now(),
            **decision,  # type: ignore[misc]
        }
        with self.connection.connect() as db:
            db["adaptive_evaluations"].update_one(
                {
                    "experiment_id": experiment_id,
                    "generation_index": int(generation_index),
                    "individual_id": decision["individual_id"],
                },
                {"$set": doc},
                upsert=True,
            )

    def update_actual_objectives(
        self,
        experiment_id: ObjectId,
        generation_index: int,
        individual_id: str,
        actual_objectives: list[float],
        evaluation_source: str = "simulated",
    ) -> bool:
        """Attach the ground truth to a decision once the simulation finished."""
        with self.connection.connect() as db:
            result = db["adaptive_evaluations"].update_one(
                {
                    "experiment_id": experiment_id,
                    "generation_index": int(generation_index),
                    "individual_id": individual_id,
                },
                {"$set": {
                    "actual_objectives": actual_objectives,
                    "evaluation_source": evaluation_source,
                }},
            )
            return result.modified_count > 0

    def find_by_experiment(
        self,
        experiment_id: ObjectId,
        generation_index: Optional[int] = None,
    ) -> list[dict]:
        query: dict[str, Any] = {"experiment_id": experiment_id}
        if generation_index is not None:
            query["generation_index"] = int(generation_index)
        with self.connection.connect() as db:
            return list(db["adaptive_evaluations"].find(query, sort=[("generation_index", 1)]))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def upsert_metrics(
        self,
        experiment_id: ObjectId,
        scenario_fingerprint: str,
        generation_index: int,
        metrics: dict[str, Any],
    ) -> None:
        doc: AdaptiveMetrics = {
            "experiment_id": experiment_id,
            "scenario_fingerprint": scenario_fingerprint,
            "generation_index": int(generation_index),
            "created_at": datetime.now(),
            "metrics": metrics,
        }
        with self.connection.connect() as db:
            db["adaptive_metrics"].update_one(
                {"experiment_id": experiment_id, "generation_index": int(generation_index)},
                {"$set": doc},
                upsert=True,
            )

    def find_metrics(self, experiment_id: ObjectId) -> list[dict]:
        with self.connection.connect() as db:
            return list(db["adaptive_metrics"].find(
                {"experiment_id": experiment_id},
                sort=[("generation_index", 1)],
            ))

    # ------------------------------------------------------------------
    def delete_by_experiment(self, experiment_id: ObjectId) -> int:
        with self.connection.connect() as db:
            a = db["adaptive_evaluations"].delete_many({"experiment_id": experiment_id})
            b = db["adaptive_metrics"].delete_many({"experiment_id": experiment_id})
            return a.deleted_count + b.deleted_count
