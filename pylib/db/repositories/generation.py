import logging
from datetime import datetime
from typing import Callable
from bson import ObjectId, errors

from pylib.db.models.generation import Generation
from pylib.db.models.enums import EnumStatus
from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)


class GenerationRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["generations"].create_index([("experiment_id", 1)], name="idx_generations_experiment_id")
            db["generations"].create_index([("experiment_id", 1), ("index", 1)], unique=True, name="idx_generations_experiment_index")
            db["generations"].create_index([("status", 1)], name="idx_generations_status")

    def insert(self, gen: Generation) -> ObjectId:
        with self.connection.connect() as db:
            return db["generations"].insert_one(gen).inserted_id

    def get(self, generation_id: str) -> Generation:
        try:
            oid = ObjectId(generation_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", generation_id)
            return None
        with self.connection.connect() as db:
            return db["generations"].find_one({"_id": oid})

    def find_by_experiment(self, experiment_id: ObjectId) -> list[Generation]:
        with self.connection.connect() as db:
            return list(db["generations"].find(
                {"experiment_id": experiment_id},
                sort=[("index", 1)]
            ))

    def find_by_status(self, status: EnumStatus) -> list[Generation]:
        with self.connection.connect() as db:
            return list(db["generations"].find({"status": status}))

    def find_pending(self) -> list[Generation]:
        return self.find_by_status(EnumStatus.WAITING)

    def update(self, generation_id: ObjectId, updates: dict) -> bool:
        with self.connection.connect() as db:
            result = db["generations"].update_one({"_id": generation_id}, {"$set": updates})
            return result.modified_count > 0

    def mark_waiting(self, generation_id: ObjectId):
        self.update(generation_id, {"status": EnumStatus.WAITING})

    def mark_running(self, generation_id: ObjectId):
        self.update(generation_id, {
            "status": EnumStatus.RUNNING,
            "start_time": datetime.now()
        })

    def mark_done(self, generation_id: ObjectId):
        self.update(generation_id, {
            "status": EnumStatus.DONE,
            "end_time": datetime.now()
        })

    def mark_error(self, generation_id: ObjectId):
        self.update(generation_id, {
            "status": EnumStatus.ERROR,
            "end_time": datetime.now()
        })

    def all_simulations_done(self, generation_id: ObjectId) -> bool:
        """Returns True if every simulation in this generation has status DONE."""
        with self.connection.connect() as db:
            total = db["simulations"].count_documents({"generation_id": generation_id})
            if total == 0:
                log.warning("Generation %s has no simulations; treating as done.", generation_id)
                return True
            done = db["simulations"].count_documents({
                "generation_id": generation_id,
                "status": EnumStatus.DONE
            })
            return done == total

    def any_simulation_active(self, generation_id: ObjectId) -> bool:
        """Returns True if any simulation is still WAITING or RUNNING."""
        with self.connection.connect() as db:
            count = db["simulations"].count_documents({
                "generation_id": generation_id,
                "status": {"$in": [EnumStatus.WAITING, EnumStatus.RUNNING]}
            })
            return count > 0

    def get_simulations_metrics_by_individual(
        self,
        generation_id: ObjectId | str,
        metrics: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Returns a map: individual_id_hash → {metric_name → averaged value}
        Averages over all DONE simulations belonging to the same individual
        (multiple seeds per individual).
        """
        if isinstance(generation_id, str):
            try:
                generation_id = ObjectId(generation_id)
            except Exception:
                raise ValueError(f"Invalid generation_id: {generation_id!r}")

        with self.connection.connect() as db:
            sims = list(db["simulations"].find(
                {"generation_id": generation_id, "status": EnumStatus.DONE},
                {"_id": 0, "individual_id": 1, "network_metrics": 1}
            ))

        # Accumulate per individual
        accumulated: dict[str, dict[str, list[float]]] = {}
        for sim in sims:
            ind_id = sim.get("individual_id", "")
            metrics_map = sim.get("network_metrics") or {}
            if ind_id not in accumulated:
                accumulated[ind_id] = {}
            for m in metrics:
                if m in metrics_map:
                    accumulated[ind_id].setdefault(m, []).append(float(metrics_map[m]))

        # Average
        result: dict[str, dict[str, float]] = {}
        for ind_id, metrics_vals in accumulated.items():
            result[ind_id] = {m: sum(v) / len(v) for m, v in metrics_vals.items() if v}
        return result

    def watch_status_waiting(self, on_change: Callable[[dict], None]):
        log.info("[GenerationRepository] Watching for WAITING generations...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": EnumStatus.WAITING
                }
            }
        ]
        self.connection.watch_collection(
            "generations",
            pipeline,
            on_change,
            full_document="updateLookup"
        )

    def watch_status_terminal(self, on_change: Callable[[dict], None]):
        """Watches for generations reaching any terminal state (DONE or ERROR)."""
        log.info("[GenerationRepository] Watching for terminal generations (DONE or ERROR)...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": {"$in": [EnumStatus.DONE, EnumStatus.ERROR]}
                }
            }
        ]
        self.connection.watch_collection(
            "generations",
            pipeline,
            on_change,
            full_document="updateLookup"
        )
