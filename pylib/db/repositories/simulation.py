import logging
from datetime import datetime
from typing import Optional, Callable
from bson import ObjectId, errors

from pylib.db.models.simulation import Simulation
from pylib.db.models.enums import EnumStatus
from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)


class SimulationRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["simulations"].create_index([("status", 1)], name="idx_simulations_status")
            db["simulations"].create_index([("experiment_id", 1)], name="idx_simulations_experiment_id")
            db["simulations"].create_index([("generation_id", 1)], name="idx_simulations_generation_id")

    def insert(self, simulation: Simulation) -> ObjectId:
        with self.connection.connect() as db:
            return db["simulations"].insert_one(simulation).inserted_id

    def get(self, simulation_id: str) -> Simulation:
        try:
            oid = ObjectId(simulation_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", simulation_id)
            return None
        with self.connection.connect() as db:
            return db["simulations"].find_one({"_id": oid})

    def find_pending(self) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find({"status": EnumStatus.WAITING}))

    def find_running_before(self, cutoff: datetime) -> list[Simulation]:
        """Returns simulations stuck in RUNNING with start_time older than cutoff."""
        with self.connection.connect() as db:
            return list(db["simulations"].find({
                "status": EnumStatus.RUNNING,
                "start_time": {"$lt": cutoff}
            }))

    def find_pending_by(self, parent: str, object_id: ObjectId) -> list[Simulation]:
        if parent not in ["experiment_id", "generation_id"]:
            log.error("Invalid parent field: %s", parent)
            return []
        with self.connection.connect() as db:
            return list(db["simulations"].find({
                "status": EnumStatus.WAITING,
                parent: object_id
            }))

    def find_by_individual(self, individual_id: str) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find({"individual_id": individual_id}))

    def find_by_status(self, status: str) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find({"status": status}))

    def update(self, sim_id: ObjectId, updates: dict) -> bool:
        with self.connection.connect() as db:
            result = db["simulations"].update_one({"_id": sim_id}, {"$set": updates})
            return result.modified_count > 0

    def update_status(self, sim_id: str, status: str):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": ObjectId(sim_id)},
                {"$set": {"status": status}}
            )

    def mark_running(self, sim_id: ObjectId):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "status": EnumStatus.RUNNING,
                    "start_time": datetime.now(),
                }}
            )

    def mark_done(
        self,
        sim_id: ObjectId,
        log_id: Optional[ObjectId],
        csv_id: Optional[ObjectId],
        network_metrics: dict[str, float]
    ):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "status": EnumStatus.DONE,
                    "end_time": datetime.now(),
                    "log_cooja_id": log_id,
                    "csv_log_id": csv_id,
                    "network_metrics": network_metrics,
                }}
            )

    def mark_error(self, sim_id: ObjectId, system_message: str):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "status": EnumStatus.ERROR,
                    "end_time": datetime.now(),
                    "system_message": system_message
                }}
            )

    def delete_by_id(self, simulation_id: str) -> bool:
        try:
            oid = ObjectId(simulation_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return False
        with self.connection.connect() as db:
            result = db["simulations"].delete_one({"_id": oid})
            return result.deleted_count > 0

    def watch_status_waiting(self, on_change: Callable[[dict], None]) -> None:
        """Watches the simulations collection for documents entering WAITING status."""
        log.info("[SimulationRepository] Watching for WAITING simulations...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": EnumStatus.WAITING
                }
            }
        ]
        self.connection.watch_collection(
            "simulations",
            pipeline,
            on_change,
            full_document="updateLookup"
        )

    def watch_status_terminal(self, on_change: Callable[[dict], None]) -> None:
        """Watches the simulations collection for documents entering DONE or ERROR status."""
        log.info("[SimulationRepository] Watching for terminal simulations (DONE or ERROR)...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["update", "replace"]},
                    "fullDocument.status": {"$in": [EnumStatus.DONE, EnumStatus.ERROR]}
                }
            }
        ]
        self.connection.watch_collection(
            "simulations",
            pipeline,
            on_change,
            full_document="updateLookup"
        )
