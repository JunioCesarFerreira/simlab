import logging
from datetime import datetime
from typing import Any, Callable, Optional
from bson import ObjectId, errors

from pylib.db.connection import MongoDBConnection
from pylib.db.models.enums import EnumStatus
from pylib.db.models.milp_sweep import MilpSweep

log = logging.getLogger(__name__)

# Listing projection: solutions and dedup state can grow to thousands of
# entries; list views never need them.
_LIST_PROJECTION = {"solutions": 0, "checkpoint.genotypes": 0}


class MilpSweepRepository:
    COLLECTION = "milp_sweeps"

    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def insert(self, sweep: MilpSweep) -> ObjectId:
        with self.connection.connect() as db:
            return db[self.COLLECTION].insert_one(sweep).inserted_id

    def find_all(self) -> list[dict[str, Any]]:
        with self.connection.connect() as db:
            return list(
                db[self.COLLECTION]
                .find({}, _LIST_PROJECTION)
                .sort("created_time", -1)
            )

    def find_pending(self) -> list[dict[str, Any]]:
        """Sweeps waiting to be picked up (startup recovery of the engine)."""
        with self.connection.connect() as db:
            return list(db[self.COLLECTION].find({"status": EnumStatus.WAITING}))

    def find_running(self) -> list[dict[str, Any]]:
        """Sweeps left Running by a crashed engine. Safe to resume thanks to
        the per-combination checkpoint (single-engine deployment assumption)."""
        with self.connection.connect() as db:
            return list(db[self.COLLECTION].find({"status": EnumStatus.RUNNING}))

    def get(self, sweep_id: str) -> Optional[dict[str, Any]]:
        oid = self._to_oid(sweep_id)
        if oid is None:
            return None
        with self.connection.connect() as db:
            return db[self.COLLECTION].find_one({"_id": oid})

    def update(self, sweep_id: str, updates: dict) -> bool:
        oid = self._to_oid(sweep_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            result = db[self.COLLECTION].update_one({"_id": oid}, {"$set": updates})
            return result.modified_count > 0

    def update_status(self, sweep_id: str, status: str, message: Optional[str] = None) -> bool:
        updates: dict[str, Any] = {"status": status}
        if message is not None:
            updates["system_message"] = message
        return self.update(sweep_id, updates)

    def update_starting(self, sweep_id: str) -> bool:
        """Atomically claim a Waiting sweep (Waiting -> Running)."""
        oid = self._to_oid(sweep_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            result = db[self.COLLECTION].update_one(
                {"_id": oid, "status": EnumStatus.WAITING},
                {"$set": {"status": EnumStatus.RUNNING, "start_time": datetime.now()}},
            )
            return result.modified_count > 0

    def append_solution(
        self,
        sweep_id: str,
        record: dict[str, Any],
        progress: dict[str, Any],
        checkpoint: dict[str, Any],
    ) -> bool:
        """Persist one grid-combination outcome plus updated progress/checkpoint
        in a single atomic write, so an interrupted sweep can always resume."""
        oid = self._to_oid(sweep_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            result = db[self.COLLECTION].update_one(
                {"_id": oid},
                {
                    "$push": {"solutions": record},
                    "$set": {"progress": progress, "checkpoint": checkpoint},
                },
            )
            return result.modified_count > 0

    def request_cancel(self, sweep_id: str) -> bool:
        """Cooperative cancellation: only Waiting/Running sweeps can be cancelled."""
        oid = self._to_oid(sweep_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            result = db[self.COLLECTION].update_one(
                {"_id": oid, "status": {"$in": [EnumStatus.WAITING, EnumStatus.RUNNING]}},
                {"$set": {"cancel_requested": True}},
            )
            return result.modified_count > 0

    def is_cancel_requested(self, sweep_id: str) -> bool:
        oid = self._to_oid(sweep_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            doc = db[self.COLLECTION].find_one({"_id": oid}, {"cancel_requested": 1})
            return bool(doc and doc.get("cancel_requested"))

    def finish(
        self,
        sweep_id: str,
        status: str,
        experiment_id: Optional[ObjectId] = None,
        message: Optional[str] = None,
    ) -> bool:
        updates: dict[str, Any] = {"status": status, "end_time": datetime.now()}
        if experiment_id is not None:
            updates["experiment_id"] = experiment_id
        if message is not None:
            updates["system_message"] = message
        return self.update(sweep_id, updates)

    def delete(self, sweep_id: str) -> bool:
        oid = self._to_oid(sweep_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            result = db[self.COLLECTION].delete_one({"_id": oid})
            return result.deleted_count == 1

    def watch_status_waiting(self, on_change: Callable[[dict], None]):
        """Blocks watching for sweeps entering Waiting status (insert or update)."""
        log.info("[MilpSweepRepository] Waiting for new MILP sweeps...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": EnumStatus.WAITING,
                }
            }
        ]
        self.connection.watch_collection(
            self.COLLECTION,
            pipeline,
            on_change,
            full_document="updateLookup",
        )

    # ---- engine status (singleton document published by the milp-engine) ----

    STATUS_COLLECTION = "milp_engine_status"

    def get_engine_status(self) -> Optional[dict[str, Any]]:
        with self.connection.connect() as db:
            return db[self.STATUS_COLLECTION].find_one({"_id": "milp-engine"})

    def publish_engine_status(self, status: dict[str, Any]) -> None:
        status = dict(status)
        status["updated_time"] = datetime.now()
        with self.connection.connect() as db:
            db[self.STATUS_COLLECTION].replace_one(
                {"_id": "milp-engine"}, {"_id": "milp-engine", **status}, upsert=True
            )

    @staticmethod
    def _to_oid(sweep_id: str) -> Optional[ObjectId]:
        try:
            return ObjectId(sweep_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", sweep_id)
            return None
