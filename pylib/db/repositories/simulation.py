import logging
from datetime import datetime
from typing import Any, Callable, Optional
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

    def _attempt_filter(
        self, sim_id: ObjectId, attempt_id: Optional[str]
    ) -> dict[str, Any]:
        """The filter for a write that belongs to one attempt.

        ``attempt_id`` is required of every caller and may be ``None``, which
        means "unconditional — there is no attempt, or the one that existed is
        known to be gone". Recovery and the synthetic path pass ``None``
        deliberately; everything that runs on a container passes the attempt
        it holds.

        The guard matters because an attempt can lose its container without
        knowing: a recovery pass or the stale-reservation sweep may have given
        it to someone else. Without the filter, the thread that lost it still
        publishes its result over the one that took over — two attempts
        writing the same document, last one wins, and nothing says so.
        """
        if attempt_id is None:
            return {"_id": sim_id}
        return {"_id": sim_id, "attempt_id": attempt_id}

    def mark_running(
        self, sim_id: ObjectId, *, attempt_id: Optional[str]
    ) -> bool:
        with self.connection.connect() as db:
            res = db["simulations"].update_one(
                self._attempt_filter(sim_id, attempt_id),
                {"$set": {
                    "status": EnumStatus.RUNNING,
                    "start_time": datetime.now(),
                }}
            )
            return res.matched_count > 0

    def mark_done(
        self,
        sim_id: ObjectId,
        log_id: Optional[ObjectId],
        csv_id: Optional[ObjectId],
        network_metrics: dict[str, float],
        dodag: Optional[dict] = None,
        *,
        attempt_id: Optional[str],
    ) -> bool:
        """Publish a result, for the attempt that is still the owner.

        Returns false when the write matched nothing, which means this attempt
        no longer owns the simulation. The caller must not treat that as
        success: the result belongs to a placement that was taken over.
        """
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
            res = db["simulations"].update_one(
                self._attempt_filter(sim_id, attempt_id),
                {"$set": fields}
            )
            return res.matched_count > 0

    def mark_error(
        self, sim_id: ObjectId, system_message: str, *, attempt_id: Optional[str]
    ) -> bool:
        with self.connection.connect() as db:
            res = db["simulations"].update_one(
                self._attempt_filter(sim_id, attempt_id),
                {"$set": {
                    "status": EnumStatus.ERROR,
                    "end_time": datetime.now(),
                    "system_message": system_message
                }}
            )
            return res.matched_count > 0

    def requeue(self, sim_id: ObjectId, attempt_id: str, reason: str) -> bool:
        """Put a simulation back in the queue, losing its placement.

        For the one case where a failure is not the simulation's fault: the
        attempt discovered mid-run that it no longer owns the container. The
        work is sound and worth doing again, so the simulation returns to
        ``Waiting`` and the dispatcher places it afresh — and the change
        stream picks it up without anyone having to remember to resubmit.

        Guarded by the attempt, so the thread that already lost the race
        cannot drag a newer placement back into the queue.
        """
        with self.connection.connect() as db:
            res = db["simulations"].update_one(
                {"_id": sim_id, "attempt_id": attempt_id},
                {"$set": {
                    "status": EnumStatus.WAITING,
                    "system_message": reason,
                    "start_time": None,
                    "end_time": None,
                    "assigned_host_id": None,
                    "assigned_host_name": None,
                    "assigned_resource_id": None,
                    "assigned_resource_name": None,
                    "assigned_container_id": None,
                    "attempt_id": None,
                    "remote_pid": None,
                }},
            )
            return res.modified_count > 0

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
