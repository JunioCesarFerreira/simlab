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

    def find_by_individual(
        self, individual_id: str, experiment_id: Optional[ObjectId] = None
    ) -> list[Simulation]:
        """Simulations of one individual, scoped to an experiment.

        ``individual_id`` is a chromosome hash, not an identity: the same
        chromosome recurs across experiments, and its simulations accumulate
        under the same hash. Without the experiment, this returns every run of
        that chromosome ever made — observed in practice as an individual's
        panel listing nine simulations when two belonged to the experiment
        being looked at.

        The pair (experiment, individual) is the identity that means something,
        so ``experiment_id`` should always be supplied. It stays optional only
        so an existing caller cannot break silently.
        """
        query: dict = {"individual_id": individual_id}
        if experiment_id is not None:
            query["experiment_id"] = experiment_id
        with self.connection.connect() as db:
            return list(db["simulations"].find(query))

    def find_ids_grouped_by_individual(
        self, experiment_id: ObjectId
    ) -> dict[str, list[str]]:
        """Map individual_id (chromosome hash) -> list of simulation _id strings."""
        with self.connection.connect() as db:
            cursor = db["simulations"].find(
                {"experiment_id": experiment_id},
                {"_id": 1, "individual_id": 1},
            )
            out: dict[str, list[str]] = {}
            for doc in cursor:
                ind_id = doc.get("individual_id")
                if not ind_id:
                    continue
                out.setdefault(ind_id, []).append(str(doc["_id"]))
            return out

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
        network_metrics: dict[str, float],
        dodag: Optional[dict] = None,
    ):
        fields: dict = {
            "status": EnumStatus.DONE,
            "end_time": datetime.now(),
            "log_cooja_id": log_id,
            "csv_log_id": csv_id,
            "network_metrics": network_metrics,
        }
        if dodag is not None:
            fields["dodag"] = dodag
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": fields}
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

    def set_assignment(
        self,
        sim_id: ObjectId,
        *,
        host_id: ObjectId,
        host_name: str,
        resource_id: ObjectId,
        resource_name: str,
        attempt_id: str,
        container_id: Optional[str] = None,
    ) -> bool:
        """Record where this simulation was placed, and under which attempt.

        Written at reservation time, before anything runs, so a master-node
        that restarts mid-simulation can find its way back to the container
        instead of guessing. The names are denormalized on purpose: a finished
        simulation must still say where it ran after the host registration is
        edited or removed, which is what keeps the chain experiment → generation
        → individual → simulation → host → container readable later.
        """
        with self.connection.connect() as db:
            res = db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "assigned_host_id": host_id,
                    "assigned_host_name": host_name,
                    "assigned_resource_id": resource_id,
                    "assigned_resource_name": resource_name,
                    "assigned_container_id": container_id,
                    "attempt_id": attempt_id,
                    "assigned_at": datetime.now(),
                }},
            )
            return res.matched_count > 0

    def set_remote_pid(self, sim_id: ObjectId, attempt_id: str, remote_pid: int) -> bool:
        """Record the detached Cooja process, for this attempt only.

        The PID is what lets a reconnecting master-node ask "is it still
        running?" instead of assuming a dropped channel means a dead run. The
        attempt guard keeps a stale thread from overwriting a newer placement.
        """
        with self.connection.connect() as db:
            res = db["simulations"].update_one(
                {"_id": sim_id, "attempt_id": attempt_id},
                {"$set": {"remote_pid": remote_pid}},
            )
            return res.modified_count > 0

    def find_assigned_to_resource(self, resource_id: ObjectId) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find({"assigned_resource_id": resource_id}))

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
