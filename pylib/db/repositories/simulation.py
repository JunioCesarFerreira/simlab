import logging
from datetime import datetime
from typing import Any, Callable, Optional
from bson import ObjectId, errors
from pymongo import ReturnDocument

from pylib.db.models.simulation import Simulation
from pylib.db.models.enums import EnumStatus
from pylib.db.connection import MongoDBConnection
from pylib.timeutil import utc_now

log = logging.getLogger(__name__)


def _mean_sec(duration_ms: float, runs: int) -> Optional[float]:
    """A mean duration in seconds, or ``None`` when nothing was timed."""
    return round(duration_ms / runs / 1000.0, 1) if runs else None


class SimulationRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["simulations"].create_index([("status", 1)], name="idx_simulations_status")
            db["simulations"].create_index([("experiment_id", 1)], name="idx_simulations_experiment_id")
            db["simulations"].create_index([("generation_id", 1)], name="idx_simulations_generation_id")
            db["simulations"].create_index(
                [("generation_id", 1), ("individual_id", 1),
                 ("random_seed", 1), ("retry_round", 1)],
                name="idx_simulations_retry_identity", unique=True,
                partialFilterExpression={"retry_round": {"$exists": True}},
            )

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
                    "start_time": utc_now(),
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
            "end_time": utc_now(),
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
        self,
        sim_id: ObjectId,
        system_message: str,
        *,
        attempt_id: Optional[str],
        log_id: Optional[ObjectId] = None,
        csv_id: Optional[ObjectId] = None,
        partial_metrics: Optional[dict[str, float]] = None,
    ) -> bool:
        """Record a failure — never over a result that was already published.

        The ``$ne: Done`` guard is an invariant, not a precaution. A
        simulation that reached ``Done`` has had its metrics read by the
        optimiser; turning it into ``Error`` afterwards changes an experiment's
        outcome retroactively. It happened: ``_publish`` wrote ``mark_done``
        and then closed the generation, and a failure of that *second*,
        administrative write fell through to the generic handler, which marked
        the finished simulation in error (R24).

        The guard belongs here rather than only at the call site, because the
        call site is exactly what got it wrong, and because the attempt filter
        alone matches a simulation this attempt itself has just completed.

        ``log_id``/``csv_id``/``partial_metrics`` carry the artifacts of a run
        that produced bytes without a confirmed completion. They are recorded
        so nothing is lost, and ``partial_metrics`` is deliberately *not*
        ``network_metrics``: the aggregation that feeds the objectives reads
        the latter, and an unconfirmed run must not enter a mean.
        """
        fields: dict[str, Any] = {
            "status": EnumStatus.ERROR,
            "end_time": utc_now(),
            "system_message": system_message,
        }
        if log_id is not None:
            fields["log_cooja_id"] = log_id
        if csv_id is not None:
            fields["csv_log_id"] = csv_id
        if partial_metrics is not None:
            fields["partial_metrics"] = partial_metrics
            fields["partial"] = True
        condition = self._attempt_filter(sim_id, attempt_id)
        condition["status"] = {"$ne": EnumStatus.DONE}
        with self.connection.connect() as db:
            res = db["simulations"].update_one(condition, {"$set": fields})
            return res.matched_count > 0

    def expire_attempt(
        self, sim_id: ObjectId, attempt_id: str, message: str
    ) -> bool:
        """End an attempt and **revoke** it, in one write.

        Marking the simulation in error is not enough on its own: the guards
        on the other writes compare ``_id`` and ``attempt_id`` only, so a
        thread still holding the expired attempt could follow an error with
        ``mark_running`` and then ``mark_done`` — all three matching, the
        document ending ``Done`` after having been expired. Clearing
        ``attempt_id`` here is what makes those later writes match nothing.

        Used by the stale-reservation sweep, which is the one place that ends
        an attempt the owner may still believe in.
        """
        with self.connection.connect() as db:
            res = db["simulations"].update_one(
                {"_id": sim_id, "attempt_id": attempt_id},
                {"$set": {
                    "status": EnumStatus.ERROR,
                    "end_time": utc_now(),
                    "system_message": message,
                    "attempt_id": None,
                }},
            )
            return res.modified_count > 0

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
                    "assigned_at": utc_now(),
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

    #: Fields a repeat must *not* inherit: everything about the attempt that
    #: failed, and everything about its result.
    _RETRY_DROPS = (
        "_id", "status", "start_time", "end_time", "system_message",
        "log_cooja_id", "csv_log_id", "network_metrics", "partial_metrics",
        "partial", "dodag", "attempt_id", "remote_pid", "assigned_at",
        "assigned_host_id", "assigned_host_name", "assigned_resource_id",
        "assigned_resource_name", "assigned_container_id", "unfetched_result",
        "retry_round", "recovered_result",
    )

    def clone_for_retry(
        self,
        sim_id: ObjectId,
        *,
        reason: str = "",
        retry_round: Optional[int] = None,
    ) -> Optional[ObjectId]:
        """Queue the same replica again: same individual, same seed.

        A failed replica leaves a hole in an evaluation, and the only honest
        ways to fill it are to run that seed again or to say the evaluation is
        incomplete. This is the first.

        A **new document** rather than a reset of the old one, and that is the
        point: the failure stays on the record with its message, and the
        repeat is a separate run with its own placement and its own outcome.
        Resetting would erase the evidence that a replica had to be repeated,
        which is exactly what the completeness record exists to show.

        Only ``Done`` documents are aggregated and completeness counts
        **seeds**, so the pair cannot contribute twice. Refuses to repeat a
        simulation that already has a result.

        **Idempotent per round.** ``retry_round`` is the repetition this clone
        belongs to, and a second call for the same
        ``(generation, individual, seed, round)`` returns the existing id. A
        unique index also excludes two concurrent inserts. That is what
        makes the whole repetition recoverable: a pass that claimed a round
        and then failed part-way can be run again without duplicating the
        clones it did manage to create.
        """
        with self.connection.connect() as db:
            original = db["simulations"].find_one({"_id": sim_id})
            if original is None:
                log.error("Cannot repeat simulation %s: it no longer exists", sim_id)
                return None
            if original.get("status") == EnumStatus.DONE:
                log.info("Simulation %s already has a result; not repeating it", sim_id)
                return None
            if retry_round is not None:
                identity = {
                    "generation_id": original.get("generation_id"),
                    "individual_id": original.get("individual_id"),
                    "random_seed": original.get("random_seed"),
                    "retry_round": int(retry_round),
                }
            clone = {k: v for k, v in original.items() if k not in self._RETRY_DROPS}
            clone["status"] = EnumStatus.WAITING
            clone["retry_of"] = sim_id
            clone["retry_reason"] = reason
            clone["created_time"] = utc_now()
            if retry_round is not None:
                clone["retry_round"] = int(retry_round)
                doc = db["simulations"].find_one_and_update(
                    identity, {"$setOnInsert": clone}, upsert=True,
                    return_document=ReturnDocument.AFTER, projection={"_id": 1},
                )
                return doc["_id"]
            return db["simulations"].insert_one(clone).inserted_id

    def find_terminal_by_generation(
        self, generation_id: ObjectId
    ) -> list[Simulation]:
        """Simulations of one generation that ended without a result.

        What a repeat decision needs: the failed replicas, with their seeds
        and the identity of what they were evaluating.
        """
        with self.connection.connect() as db:
            return list(db["simulations"].find({
                "generation_id": generation_id,
                "status": {"$in": [EnumStatus.ERROR, EnumStatus.CANCELLED]},
            }))

    def placement_distribution(self, experiment_id: ObjectId) -> dict:

        """How an experiment's work was spread over the containers that ran it.

        The placement is denormalized onto each simulation at reservation
        time, which is what makes this answerable at all: a worker can be
        renamed, re-registered or removed after the run, and the finished
        simulation still says where it executed.

        Counted **per simulation document**, not per seed. A replica that was
        repeated ran twice, on two placements, and both occupied a container —
        so for "how much work did this container do", both count. (The
        *evaluation* side deduplicates by seed; that is a different question,
        answered by ``replica_completeness``.)

        Simulations that were never placed are reported separately rather than
        dropped. A run that is still waiting, or that failed before a
        reservation, belongs to no container, and silently excluding it would
        make the percentages describe a smaller experiment than the one that
        exists.

        ``mean_duration_sec`` is the mean of ``end_time - start_time`` over the
        simulations that finished **Done**: from ``mark_running`` to the
        published result, so it includes the collection but not the upload. A
        failed run is left out because it stops at an arbitrary point — a
        timeout reads as the slowest worker, a crash at launch as the fastest.
        ``None`` when no finished run has both times. The host's mean is
        weighted by run, not a mean of its workers' means.
        """
        # Typed, so a record whose times are missing or not dates is skipped
        # instead of failing the whole aggregation.
        timed = {"$and": [
            {"$eq": ["$status", EnumStatus.DONE]},
            {"$eq": [{"$type": "$start_time"}, "date"]},
            {"$eq": [{"$type": "$end_time"}, "date"]},
        ]}
        with self.connection.connect() as db:
            rows = list(db["simulations"].aggregate([
                {"$match": {"experiment_id": experiment_id}},
                {"$group": {
                    "_id": {
                        "host_id": "$assigned_host_id",
                        "host": "$assigned_host_name",
                        "resource_id": "$assigned_resource_id",
                        "resource": "$assigned_resource_name",
                    },
                    "total": {"$sum": 1},
                    "done": {"$sum": {
                        "$cond": [{"$eq": ["$status", EnumStatus.DONE]}, 1, 0]}},
                    "failed": {"$sum": {"$cond": [
                        {"$in": ["$status", [EnumStatus.ERROR,
                                             EnumStatus.CANCELLED]]}, 1, 0]}},
                    "active": {"$sum": {"$cond": [
                        {"$in": ["$status", [EnumStatus.WAITING,
                                             EnumStatus.RUNNING]]}, 1, 0]}},
                    "timed": {"$sum": {"$cond": [timed, 1, 0]}},
                    "duration_ms": {"$sum": {"$cond": [
                        timed, {"$subtract": ["$end_time", "$start_time"]}, 0]}},
                }},
            ]))

        unplaced = 0
        hosts: dict[str, dict] = {}
        placed = 0
        for row in rows:
            key = row["_id"]
            resource_name = key.get("resource")
            if not resource_name:
                unplaced += row["total"]
                continue
            placed += row["total"]
            # The name is the label; the id is the identity, because two hosts
            # may legitimately carry a worker of the same name.
            host_key = str(key.get("host_id") or key.get("host") or "?")
            host = hosts.setdefault(host_key, {
                "host_id": str(key["host_id"]) if key.get("host_id") else None,
                "host_name": key.get("host") or "(unregistered host)",
                "total": 0, "done": 0, "failed": 0, "active": 0,
                "timed": 0, "duration_ms": 0,
                "workers": [],
            })
            timed_runs = int(row.get("timed") or 0)
            duration_ms = float(row.get("duration_ms") or 0)
            host["total"] += row["total"]
            host["done"] += row["done"]
            host["failed"] += row["failed"]
            host["active"] += row["active"]
            host["timed"] += timed_runs
            host["duration_ms"] += duration_ms
            host["workers"].append({
                "resource_id": (str(key["resource_id"])
                                if key.get("resource_id") else None),
                "resource_name": resource_name,
                "total": row["total"], "done": row["done"],
                "failed": row["failed"], "active": row["active"],
                "mean_duration_sec": _mean_sec(duration_ms, timed_runs),
            })

        def share(count: int) -> float:
            return round(100.0 * count / placed, 2) if placed else 0.0

        for host in hosts.values():
            host["share"] = share(host["total"])
            host["mean_duration_sec"] = _mean_sec(
                host.pop("duration_ms"), host.pop("timed"))
            host["workers"].sort(key=lambda w: (-w["total"], w["resource_name"]))
            for worker in host["workers"]:
                worker["share"] = share(worker["total"])

        return {
            "placed": placed,
            "unplaced": unplaced,
            "total": placed + unplaced,
            "hosts": sorted(hosts.values(),
                            key=lambda h: (-h["total"], h["host_name"])),
        }

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
