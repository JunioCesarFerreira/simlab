"""Persistence for the distributed execution infrastructure.

The reservation primitives live here rather than in the scheduler on purpose:
the invariants that two simulations never share a container, and that a health
check never steals a container from a running simulation, are enforced by the
query filters themselves. No caller can get them wrong by forgetting a check.

Three repositories:

* ``HostRepository`` — registrations, check/discovery requests, and the
  inventory snapshot the scheduler reads.
* ``CoojaResourceRepository`` — discovered containers, and the atomic
  reserve/start/release cycle.
* ``InfraCryptoRepository`` — the published public key, so the browser can seal
  a credential against the key the master-node actually holds.
"""
import logging
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from bson import ObjectId, errors
from pymongo import ReturnDocument

from pylib.crypto.envelope import SecretEnvelope
from pylib.db.connection import MongoDBConnection
from pylib.db.models.infra import (
    CRYPTO_COLLECTION,
    HOSTS_COLLECTION,
    OCCUPIED_STATUSES,
    RESOURCES_COLLECTION,
    SCHEDULABLE_HOST_STATUSES,
    ContainerInfo,
    CoojaResource,
    EnumContainerState,
    EnumHostStatus,
    EnumHostType,
    EnumResourceStatus,
    Host,
    ResourceProbe,
)

log = logging.getLogger(__name__)

_OCCUPIED = [s.value for s in OCCUPIED_STATUSES]
_SCHEDULABLE = [s.value for s in SCHEDULABLE_HOST_STATUSES]

# Statuses a health check or a discovery pass may overwrite. The occupied ones
# are absent by design: losing contact with a host says nothing about whether
# the simulation on it is still running, so its reservation stands until the
# reconciliation layer decides otherwise.
_REASSIGNABLE = [
    EnumResourceStatus.AVAILABLE.value,
    EnumResourceStatus.OFFLINE.value,
    EnumResourceStatus.ERROR.value,
    EnumResourceStatus.DISABLED.value,
]


def released_status_expression(*, container_state_counts: bool) -> dict[str, Any]:
    """Where a released container lands, as an aggregation expression.

    Only a *negative observation* takes a container out of the pool:

    * disabled → ``disabled``, so it drains out;
    * Docker looked and it is not running → ``offline``;
    * the probe ran and failed → ``offline``;
    * otherwise, including nothing observed yet → ``available``.

    That last clause is load-bearing. Requiring positive evidence instead sent
    every freshly seeded container to ``offline`` on its first release, because
    discovery has not described it yet — draining the pool by one container per
    completed simulation. The same mistake made restart recovery hand back a
    pool of zero.

    ``container_state_counts`` says whether Docker's observation applies to
    this worker at all, and is **required**: the caller holds the host record,
    which this collection cannot see, and the one path that silently assumed
    "yes" is what sent two hand-configured remote workers out of the pool on
    every completed simulation. Pass
    ``not operator_configured(host, resource)``.

    Shared by the normal release and the recovery release so the rule exists
    once.
    """
    branches: list[dict[str, Any]] = [
        {
            "case": {"$ne": ["$enabled", True]},
            "then": EnumResourceStatus.DISABLED.value,
        },
    ]
    if container_state_counts:
        branches.append({
            # Docker looked, and it is not running.
            "case": {"$and": [
                {"$ne": [{"$ifNull": ["$container.checked_at", None]}, None]},
                {"$ne": ["$container.state", EnumContainerState.RUNNING.value]},
            ]},
            "then": EnumResourceStatus.OFFLINE.value,
        })
    branches.append({
        # The probe ran, and it failed.
        "case": {"$and": [
            {"$ne": [{"$ifNull": ["$probe.checked_at", None]}, None]},
            {"$ne": ["$probe.ssh_ok", True]},
        ]},
        "then": EnumResourceStatus.OFFLINE.value,
    })
    return {
        "$switch": {
            "branches": branches,
            "default": EnumResourceStatus.AVAILABLE.value,
        }
    }


def _oid(value: Any) -> Optional[ObjectId]:
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(value)
    except (errors.InvalidId, TypeError):
        log.error("Invalid ObjectId: %r", value)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Hosts
# ═══════════════════════════════════════════════════════════════════════════

class HostRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db[HOSTS_COLLECTION].create_index(
                [("name", 1)], unique=True, name="idx_hosts_name")
            db[HOSTS_COLLECTION].create_index(
                [("status", 1)], name="idx_hosts_status")
            db[HOSTS_COLLECTION].create_index(
                [("enabled", 1)], name="idx_hosts_enabled")

    # ── CRUD ───────────────────────────────────────────────────────────────

    def insert(self, host: Host) -> ObjectId:
        now = datetime.now()
        doc = {**host}
        doc.setdefault("created_time", now)
        doc["updated_time"] = now
        with self.connection.connect() as db:
            return db[HOSTS_COLLECTION].insert_one(doc).inserted_id

    def get(self, host_id: Any) -> Optional[Host]:
        oid = _oid(host_id)
        if oid is None:
            return None
        with self.connection.connect() as db:
            return db[HOSTS_COLLECTION].find_one({"_id": oid})

    def get_by_name(self, name: str) -> Optional[Host]:
        with self.connection.connect() as db:
            return db[HOSTS_COLLECTION].find_one({"name": name})

    def find_all(self) -> list[Host]:
        with self.connection.connect() as db:
            return list(db[HOSTS_COLLECTION].find(sort=[("name", 1)]))

    def find_enabled(self) -> list[Host]:
        with self.connection.connect() as db:
            return list(db[HOSTS_COLLECTION].find({"enabled": True}, sort=[("name", 1)]))

    def update(self, host_id: Any, updates: dict) -> bool:
        """Partial update. Refuses the fields the master-node alone may set.

        Status, check bookkeeping and the discovery results are observations,
        not settings: letting an API client write them would make the GUI show
        a state nobody measured.
        """
        oid = _oid(host_id)
        if oid is None:
            return False
        protected = {
            "status", "last_check", "last_check_message", "consecutive_failures",
            "created_time", "_id",
        }
        payload = {k: v for k, v in updates.items() if k not in protected}
        if not payload:
            return False
        payload["updated_time"] = datetime.now()
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one({"_id": oid}, {"$set": payload})
            return res.matched_count > 0

    def set_enabled(self, host_id: Any, enabled: bool) -> bool:
        """Take a host in or out of the pool.

        Disabling only stops new placements; running simulations are left to
        finish, which is what drains the host. The resources follow when they
        are released, not now.
        """
        oid = _oid(host_id)
        if oid is None:
            return False
        updates: dict[str, Any] = {"enabled": enabled, "updated_time": datetime.now()}
        if not enabled:
            updates["status"] = EnumHostStatus.DISABLED.value
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one({"_id": oid}, {"$set": updates})
            return res.modified_count > 0

    def delete(self, host_id: Any) -> bool:
        """Remove a host and its resources. Callers must check occupancy first."""
        oid = _oid(host_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            db[RESOURCES_COLLECTION].delete_many({"host_id": oid})
            res = db[HOSTS_COLLECTION].delete_one({"_id": oid})
            return res.deleted_count > 0

    # ── bootstrap ──────────────────────────────────────────────────────────

    def upsert_bootstrap(self, host: Host) -> ObjectId:
        """Create the host described by the environment, or leave it alone.

        Runs on every master-node start, so it must add what is missing without
        undoing administrative edits: ``enabled``, ``access`` and
        ``container_access`` are set only on insert. That is what keeps
        ``docker compose up`` working with no registration step while still
        letting an operator reconfigure the local host afterwards.
        """
        now = datetime.now()
        name = host["name"]
        on_insert = {k: v for k, v in host.items() if k != "name"}
        on_insert.setdefault("status", EnumHostStatus.OFFLINE.value)
        on_insert.setdefault("consecutive_failures", 0)
        on_insert["created_time"] = now
        with self.connection.connect() as db:
            doc = db[HOSTS_COLLECTION].find_one_and_update(
                {"name": name},
                {"$setOnInsert": on_insert, "$set": {"updated_time": now}},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            return doc["_id"]

    # ── health bookkeeping (master-node only) ──────────────────────────────

    def mark_check_result(
        self,
        host_id: Any,
        *,
        status: EnumHostStatus,
        message: str = "",
        reset_failures: bool = False,
    ) -> bool:
        """Record the outcome of a reachability check."""
        oid = _oid(host_id)
        if oid is None:
            return False
        now = datetime.now()
        update: dict[str, Any] = {
            "$set": {
                "status": status.value,
                "last_check": now,
                "last_check_message": message,
                "updated_time": now,
            }
        }
        if reset_failures:
            update["$set"]["consecutive_failures"] = 0
        else:
            update["$inc"] = {"consecutive_failures": 1}
        with self.connection.connect() as db:
            # A disabled host keeps its DISABLED status: an administrative
            # decision outranks an observation.
            res = db[HOSTS_COLLECTION].update_one(
                {"_id": oid, "enabled": True}, update)
            return res.matched_count > 0

    # ── requests from the API (D4/D6) ──────────────────────────────────────

    def request_check(self, host_id: Any) -> bool:
        """Ask the master-node to probe this host.

        The API cannot do it itself: it has no route to the container network
        and no private key to open the credential. So it records the request
        and the master-node, watching this collection, carries it out.
        """
        return self._set_request(host_id, "check_requested_at")

    def request_discovery(self, host_id: Any) -> bool:
        return self._set_request(host_id, "discovery_requested_at")

    def _set_request(self, host_id: Any, field: str) -> bool:
        oid = _oid(host_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one(
                {"_id": oid}, {"$set": {field: datetime.now()}})
            return res.matched_count > 0

    def clear_request(self, host_id: Any, field: str, requested_at: datetime) -> bool:
        """Clear a request, unless a newer one arrived while it was running."""
        oid = _oid(host_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one(
                {"_id": oid, field: requested_at}, {"$set": {field: None}})
            return res.modified_count > 0

    def record_discovery(
        self, host_id: Any, *, count: int, message: str = ""
    ) -> bool:
        oid = _oid(host_id)
        if oid is None:
            return False
        now = datetime.now()
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one(
                {"_id": oid},
                {"$set": {
                    "discovery.last_discovery": now,
                    "discovery.discovered_count": count,
                    "discovery.last_discovery_message": message,
                    "updated_time": now,
                }},
            )
            return res.matched_count > 0

    # ── inventory ──────────────────────────────────────────────────────────

    def inventory_snapshot(self) -> list[dict[str, Any]]:
        """Per-host occupancy, in one round trip.

        The scheduler's placement decision and the GUI's overview read the same
        document, so what an operator sees is what the scheduler used.

        Counts are honest — ``free`` is the number of free containers whether
        or not the host can take work — and ``schedulable`` carries the
        eligibility separately. Zeroing ``free`` for an offline host would make
        the GUI under-report capacity that is merely unreachable.
        """
        available = EnumResourceStatus.AVAILABLE.value

        def count_where(cond: dict) -> dict:
            return {"$size": {"$filter": {"input": "$_res", "as": "r", "cond": cond}}}

        pipeline: list[dict[str, Any]] = [
            {
                "$lookup": {
                    "from": RESOURCES_COLLECTION,
                    "localField": "_id",
                    "foreignField": "host_id",
                    "as": "_res",
                    "pipeline": [{"$project": {"status": 1, "enabled": 1}}],
                }
            },
            {
                "$set": {
                    "resource_total": {"$size": "$_res"},
                    "free": count_where({
                        "$and": [
                            {"$eq": ["$$r.enabled", True]},
                            {"$eq": ["$$r.status", available]},
                        ]
                    }),
                    "active": count_where({"$in": ["$$r.status", _OCCUPIED]}),
                    "unavailable": count_where({
                        "$in": ["$$r.status", [
                            EnumResourceStatus.OFFLINE.value,
                            EnumResourceStatus.ERROR.value,
                        ]]
                    }),
                    "disabled": count_where({
                        "$or": [
                            {"$eq": ["$$r.enabled", False]},
                            {"$eq": ["$$r.status", EnumResourceStatus.DISABLED.value]},
                        ]
                    }),
                    "schedulable": {
                        "$and": [
                            {"$eq": ["$enabled", True]},
                            {"$in": ["$status", _SCHEDULABLE]},
                        ]
                    },
                }
            },
            {"$unset": "_res"},
            {"$sort": {"name": 1}},
        ]
        with self.connection.connect() as db:
            return list(db[HOSTS_COLLECTION].aggregate(pipeline))

    def watch(self, on_change: Callable[[dict], None]) -> None:
        """Watch host registrations, so enabling a host takes effect at once.

        One of the wake-ups for a dispatcher blocked on capacity — and what
        makes adding or removing a host not require a restart.
        """
        self.connection.watch_collection(
            HOSTS_COLLECTION,
            [{"$match": {"operationType": {"$in": ["insert", "update", "replace", "delete"]}}}],
            on_change,
            full_document="updateLookup",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cooja resources
# ═══════════════════════════════════════════════════════════════════════════

class CoojaResourceRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            # Unique per host: discovery keys on (host_id, name), so a repeated
            # pass can never duplicate a container into two resources.
            db[RESOURCES_COLLECTION].create_index(
                [("host_id", 1), ("name", 1)], unique=True,
                name="idx_resources_host_name")
            db[RESOURCES_COLLECTION].create_index(
                [("status", 1)], name="idx_resources_status")
            db[RESOURCES_COLLECTION].create_index(
                [("current_simulation_id", 1)], name="idx_resources_simulation")

    # ── reads ──────────────────────────────────────────────────────────────

    def get(self, resource_id: Any) -> Optional[CoojaResource]:
        oid = _oid(resource_id)
        if oid is None:
            return None
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].find_one({"_id": oid})

    def find_all(self) -> list[CoojaResource]:
        with self.connection.connect() as db:
            return list(db[RESOURCES_COLLECTION].find(sort=[("host_id", 1), ("name", 1)]))

    def find_by_host(self, host_id: Any) -> list[CoojaResource]:
        oid = _oid(host_id)
        if oid is None:
            return []
        with self.connection.connect() as db:
            return list(db[RESOURCES_COLLECTION].find({"host_id": oid}, sort=[("name", 1)]))

    def find_free_by_host(self, host_id: Any) -> list[CoojaResource]:
        """Candidates for placement on one host, oldest check first.

        Ordering by ``last_check`` spreads work across a host's containers
        instead of hammering whichever one sorts first.
        """
        oid = _oid(host_id)
        if oid is None:
            return []
        with self.connection.connect() as db:
            return list(db[RESOURCES_COLLECTION].find(
                {
                    "host_id": oid,
                    "enabled": True,
                    "status": EnumResourceStatus.AVAILABLE.value,
                },
                sort=[("last_check", 1), ("name", 1)],
            ))

    def find_occupied(self) -> list[CoojaResource]:
        """Resources holding a simulation — the input to startup reconciliation."""
        with self.connection.connect() as db:
            return list(db[RESOURCES_COLLECTION].find({"status": {"$in": _OCCUPIED}}))

    def find_by_simulation(self, simulation_id: Any) -> Optional[CoojaResource]:
        oid = _oid(simulation_id)
        if oid is None:
            return None
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].find_one({"current_simulation_id": oid})

    def find_stale_reservations(self, cutoff: datetime) -> list[CoojaResource]:
        """Reservations that never became a running simulation.

        Covers only the window between claiming a container and starting work,
        so its timeout is short and unrelated to how long a simulation may run.
        """
        with self.connection.connect() as db:
            return list(db[RESOURCES_COLLECTION].find({
                "status": EnumResourceStatus.RESERVED.value,
                "reserved_at": {"$lt": cutoff},
            }))

    def count_active_by_host(self) -> dict[ObjectId, int]:
        """Active placements per host, for the load-balancing decision."""
        pipeline = [
            {"$match": {"status": {"$in": _OCCUPIED}}},
            {"$group": {"_id": "$host_id", "n": {"$sum": 1}}},
        ]
        with self.connection.connect() as db:
            return {d["_id"]: d["n"] for d in db[RESOURCES_COLLECTION].aggregate(pipeline)}

    # ── the reservation cycle ──────────────────────────────────────────────

    def reserve(
        self, resource_id: Any, simulation_id: ObjectId, attempt_id: str
    ) -> Optional[CoojaResource]:
        """Claim a container for a simulation, atomically.

        Returns the updated document, or ``None`` if someone else got there
        first. The filter is the whole mechanism: ``available`` is a state only
        a free, enabled, healthy container is in, and a single document update
        is indivisible — so two dispatchers racing for the last container
        cannot both win.
        """
        oid = _oid(resource_id)
        if oid is None:
            return None
        now = datetime.now()
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].find_one_and_update(
                {
                    "_id": oid,
                    "enabled": True,
                    "status": EnumResourceStatus.AVAILABLE.value,
                },
                {"$set": {
                    "status": EnumResourceStatus.RESERVED.value,
                    "current_simulation_id": simulation_id,
                    "attempt_id": attempt_id,
                    "reserved_at": now,
                    "started_at": None,
                    "remote_pid": None,
                    "updated_time": now,
                }},
                return_document=ReturnDocument.AFTER,
            )

    def mark_busy(
        self,
        resource_id: Any,
        simulation_id: ObjectId,
        attempt_id: str,
        remote_pid: Optional[int] = None,
    ) -> bool:
        """Promote a reservation to a running simulation, recording its PID.

        Guarded by ``attempt_id``: a thread from an earlier attempt cannot
        report a start for a container that has since been handed on.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = datetime.now()
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                {
                    "_id": oid,
                    "status": EnumResourceStatus.RESERVED.value,
                    "current_simulation_id": simulation_id,
                    "attempt_id": attempt_id,
                },
                {"$set": {
                    "status": EnumResourceStatus.BUSY.value,
                    "remote_pid": remote_pid,
                    "started_at": now,
                    "updated_time": now,
                }},
            )
            return res.modified_count > 0

    def release(
        self,
        resource_id: Any,
        simulation_id: ObjectId,
        attempt_id: str,
        *,
        container_state_counts: bool,
    ) -> Optional[CoojaResource]:
        """Hand a container back, if this attempt is still the one holding it.

        The guard is the point: a late thread whose simulation has already been
        recovered and re-placed must not free a container that is now running
        something else.

        Where it lands is derived from what was actually *observed*, and only a
        negative observation takes a container out of the pool:

        * disabled while busy → ``disabled``, so it drains out;
        * Docker looked and the container is not running → ``offline``;
        * the probe ran and failed → ``offline``;
        * otherwise, including nothing observed yet → ``available``.

        The last clause is the important one. Requiring positive evidence
        instead would send every freshly seeded container to ``offline`` on its
        first release, because discovery has not described it yet — draining
        the pool by one container per completed simulation.

        ``container_state_counts`` is ``not operator_configured(host,
        resource)`` — see :func:`released_status_expression`.
        """
        oid = _oid(resource_id)
        if oid is None:
            return None
        now = datetime.now()
        pipeline = [{"$set": {
            "status": released_status_expression(
                container_state_counts=container_state_counts),
            "current_simulation_id": None,
            "attempt_id": None,
            "remote_pid": None,
            "reserved_at": None,
            "started_at": None,
            "updated_time": now,
        }}]
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].find_one_and_update(
                {
                    "_id": oid,
                    "current_simulation_id": simulation_id,
                    "attempt_id": attempt_id,
                },
                pipeline,
                return_document=ReturnDocument.AFTER,
            )

    def force_release(
        self,
        resource_id: Any,
        message: str = "",
        *,
        container_state_counts: bool,
        expect_attempt_id: Optional[str] = None,
        reserved_before: Optional[datetime] = None,
    ) -> bool:
        """Free a container outside the normal completion path.

        Used by startup recovery and the stale-reservation sweep. Normal
        completion goes through :meth:`release`, which is guarded by the
        attempt that holds the container.

        ``expect_attempt_id`` and ``reserved_before`` let a caller that *read*
        the reservation insist on releasing the same one it saw. The sweep
        needs that: between its query and this write the runner may have
        promoted the reservation to ``busy`` and started the JVM, and freeing
        it then would hand a container that is running something to the next
        placement. Startup recovery passes neither, because there the
        attempt is known to be gone — the process that held it died.

        Lands the container by the same rule as a normal release, not at
        ``offline``: when a master-node restarts, the container is fine and it
        is the process that died. Forcing ``offline`` here handed recovery a
        pool of zero and left the dispatcher with nowhere to place work.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = datetime.now()
        pipeline = [{"$set": {
            "status": released_status_expression(
                container_state_counts=container_state_counts),
            "current_simulation_id": None,
            "attempt_id": None,
            "remote_pid": None,
            "reserved_at": None,
            "started_at": None,
            "last_check_message": message,
            "updated_time": now,
        }}]
        condition: dict[str, Any] = {"_id": oid}
        if expect_attempt_id is not None:
            condition["attempt_id"] = expect_attempt_id
            # Still waiting to start. A reservation that got as far as `busy`
            # is governed by the simulation timeout, not by this.
            condition["status"] = EnumResourceStatus.RESERVED.value
        if reserved_before is not None:
            condition["reserved_at"] = {"$lt": reserved_before}
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(condition, pipeline)
            return res.modified_count > 0

    # ── health and discovery (never touching an occupied resource) ─────────

    def mark_available(self, resource_id: Any, message: str = "") -> bool:
        """Put a container into the pool after both check levels passed."""
        return self._set_status(resource_id, EnumResourceStatus.AVAILABLE, message)

    def mark_offline(self, resource_id: Any, message: str = "") -> bool:
        return self._set_status(resource_id, EnumResourceStatus.OFFLINE, message)

    def mark_error(self, resource_id: Any, message: str = "") -> bool:
        return self._set_status(resource_id, EnumResourceStatus.ERROR, message)

    def _set_status(
        self, resource_id: Any, status: EnumResourceStatus, message: str
    ) -> bool:
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = datetime.now()
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                # Reserved and busy are absent from _REASSIGNABLE: a failing
                # check must never evict a simulation that may still be running.
                {"_id": oid, "status": {"$in": _REASSIGNABLE}},
                {"$set": {
                    "status": status.value,
                    "last_check": now,
                    "last_check_message": message,
                    "updated_time": now,
                }},
            )
            return res.modified_count > 0

    def set_enabled(self, resource_id: Any, enabled: bool) -> bool:
        """Administrative override. Survives every discovery pass.

        Disabling an occupied container lets it finish and leaves the pool when
        released, which is why the status is not forced here.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = datetime.now()
        updates: dict[str, Any] = {"enabled": enabled, "updated_time": now}
        with self.connection.connect() as db:
            if not enabled:
                db[RESOURCES_COLLECTION].update_one(
                    {"_id": oid, "status": {"$in": _REASSIGNABLE}},
                    {"$set": {"status": EnumResourceStatus.DISABLED.value}},
                )
            res = db[RESOURCES_COLLECTION].update_one({"_id": oid}, {"$set": updates})
            return res.matched_count > 0

    def update_probe(self, resource_id: Any, probe: ResourceProbe) -> bool:
        """Record functional readiness without deciding the status."""
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = datetime.now()
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                {"_id": oid},
                {"$set": {"probe": {**probe, "checked_at": now}, "updated_time": now}},
            )
            return res.matched_count > 0

    def sync_discovered(
        self,
        host_id: ObjectId,
        containers: Iterable[ContainerInfo],
        *,
        remote_dir: str,
        endpoints: Optional[dict[str, dict[str, Any]]] = None,
    ) -> dict[str, list[str]]:
        """Reconcile this host's resources with what Docker reported.

        Call only with a *successful* inspection: a container absent from the
        list is taken to be really gone, not merely unobserved.

        What it does and deliberately does not do:

        * new container with the expected label → resource created, starting as
          ``offline`` until the SSH probe passes. Nothing enters the pool on
          Docker's word alone.
        * known container → container facts and endpoint refreshed; ``enabled``
          and ``status`` untouched, so neither an administrative decision nor a
          running simulation is overwritten.
        * vanished container → marked ``missing`` and dropped from the pool,
          but **never deleted**: the resource is what ties finished simulations
          to where they ran.
        * vanished *while occupied* → reported back in ``missing_occupied``.
          Whether that means the simulation died is a policy question for the
          reconciliation layer, not something to decide in a query.

        Returns the names affected, keyed by ``created``/``updated``/
        ``missing``/``missing_occupied``.
        """
        now = datetime.now()
        endpoints = endpoints or {}
        seen: list[str] = []
        created: list[str] = []
        updated: list[str] = []

        with self.connection.connect() as db:
            coll = db[RESOURCES_COLLECTION]
            for container in containers:
                name = container.get("name")
                if not name:
                    continue
                seen.append(name)
                set_fields: dict[str, Any] = {
                    "container": {**container, "checked_at": now},
                    "last_check": now,
                    "updated_time": now,
                }
                endpoint = endpoints.get(name)
                set_on_insert: dict[str, Any] = {
                    "enabled": True,
                    # Discovered, not yet proven: the probe decides.
                    "status": EnumResourceStatus.OFFLINE.value,
                    "remote_dir": remote_dir,
                    "current_simulation_id": None,
                    "attempt_id": None,
                    "remote_pid": None,
                    "origin": "discovered",
                    "created_time": now,
                }
                if endpoint:
                    set_on_insert["endpoint"] = endpoint
                result = coll.update_one(
                    {"host_id": host_id, "name": name},
                    {"$set": set_fields, "$setOnInsert": set_on_insert},
                    upsert=True,
                )
                if endpoint and result.upserted_id is None:
                    # Refreshing the address of a worker that already exists
                    # is only discovery's business where discovery owns it. A
                    # worker whose address a person typed keeps it: the
                    # operator is the authority on how it is reached, and
                    # replacing that on the next inspection would silently
                    # undo a deliberate setting — the same division of
                    # authority the health cycle and the release apply.
                    coll.update_one(
                        {"host_id": host_id, "name": name,
                         "origin": {"$ne": "manual"}},
                        {"$set": {"endpoint": endpoint}},
                    )
                (created if result.upserted_id else updated).append(name)

            # A manually registered worker is invisible to Docker by
            # definition: its host may not even expose a daemon. Treating it
            # as vanished would take a working worker out of the pool on every
            # discovery pass.
            gone = list(coll.find({
                "host_id": host_id,
                "name": {"$nin": seen},
                "origin": {"$ne": "manual"},
            }))
            missing = [d["name"] for d in gone]
            missing_occupied = [
                d["name"] for d in gone if d.get("status") in _OCCUPIED
            ]
            if missing:
                coll.update_many(
                    {"host_id": host_id, "name": {"$in": missing},
                     "origin": {"$ne": "manual"}},
                    {"$set": {
                        "container.state": EnumContainerState.MISSING.value,
                        "container.checked_at": now,
                        "last_check": now,
                        "last_check_message": "container not reported by Docker",
                        "updated_time": now,
                    }},
                )
                # Only the unoccupied ones leave the pool here; an occupied one
                # keeps its reservation for the reconciliation layer.
                coll.update_many(
                    {
                        "host_id": host_id,
                        "name": {"$in": missing},
                        "status": {"$in": _REASSIGNABLE},
                    },
                    {"$set": {"status": EnumResourceStatus.OFFLINE.value}},
                )

        return {
            "created": created,
            "updated": updated,
            "missing": missing,
            "missing_occupied": missing_occupied,
        }

    def insert_manual(self, resource: CoojaResource) -> ObjectId:
        """Register one worker with its own connection parameters.

        The counterpart to discovery, for a host whose SSH user cannot reach
        its Docker daemon — a shared machine — or wherever stating the
        addresses is simply clearer than deriving them.

        Starts ``offline`` like a discovered one: being registered is not
        evidence that it works, and the SSH probe is what puts it in the pool.
        """
        now = datetime.now()
        doc = {
            **resource,
            "origin": "manual",
            "status": EnumResourceStatus.OFFLINE.value,
            "current_simulation_id": None,
            "attempt_id": None,
            "remote_pid": None,
            "created_time": now,
            "updated_time": now,
        }
        doc.setdefault("enabled", True)
        doc.setdefault("probe", {"ssh_ok": False, "checked_at": None})
        # No Docker answer exists for a manual worker, and none is expected:
        # its host may not even expose a daemon this user can reach. The state
        # is left empty rather than "missing", which would claim it was looked
        # for and not found — and reads as a failure in the interface.
        doc.setdefault("container", {
            "id": "", "name": doc.get("name", ""), "image": "",
            "state": "", "labels": {}, "checked_at": None,
        })
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].insert_one(doc).inserted_id

    def update_manual(self, resource_id: Any, updates: dict) -> bool:
        """Change a worker's own settings.

        Refuses the fields that are measured or owned by the scheduler, for
        the same reason the host update does: they are observations, and a
        client writing them would make the GUI show a state nobody measured.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        protected = {
            "_id", "host_id", "status", "container", "probe", "origin",
            "current_simulation_id", "attempt_id", "remote_pid",
            "reserved_at", "started_at", "last_check", "last_check_message",
            "created_time",
        }
        payload = {k: v for k, v in updates.items() if k not in protected}
        if not payload:
            return False
        payload["updated_time"] = datetime.now()
        if "connection" in payload or "endpoint" in payload:
            # Configuring how a worker is reached takes ownership of it, even
            # if it first appeared through discovery. `origin` then means "who
            # decides this worker's connection", which is the distinction the
            # health cycle and the discovery sync both need — without it, the
            # next discovery pass marks a hand-configured worker as missing
            # and the probe never runs.
            payload["origin"] = "manual"
            # The stored probe measured the *old* address, so it no longer
            # says anything about this worker: keeping it would show a result
            # for a path that no longer exists, and let the health cycle skip
            # the one container that most needs measuring. Cleared rather than
            # falsified — the next cycle fills it in, within one interval.
            payload["probe"] = {}
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one({"_id": oid}, {"$set": payload})
            return res.matched_count > 0

    def delete(self, resource_id: Any) -> bool:
        """Remove one worker. Callers must check it is not holding anything."""
        oid = _oid(resource_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].delete_one({"_id": oid})
            return res.deleted_count > 0

    def find_by_host_and_name(self, host_id: Any, name: str) -> Optional[CoojaResource]:
        oid = _oid(host_id)
        if oid is None:
            return None
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].find_one({"host_id": oid, "name": name})

    def insert_seeded(self, resource: CoojaResource) -> bool:
        """Insert a seeded resource, or leave the existing one untouched.

        ``$setOnInsert`` only: the startup seed must never overwrite a resource
        that discovery has since described, that an operator disabled, or that
        is holding a simulation right now. Returns whether it inserted.

        Atomic by construction, so two master-nodes starting at once cannot
        both create the same container — the unique (host_id, name) index
        decides, and the loser simply makes no change.
        """
        document = {k: v for k, v in resource.items() if k not in ("host_id", "name")}
        with self.connection.connect() as db:
            result = db[RESOURCES_COLLECTION].update_one(
                {"host_id": resource["host_id"], "name": resource["name"]},
                {"$setOnInsert": document},
                upsert=True,
            )
            return result.upserted_id is not None

    def delete_by_host(self, host_id: Any) -> int:
        oid = _oid(host_id)
        if oid is None:
            return 0
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].delete_many({"host_id": oid}).deleted_count

    def watch(self, on_change: Callable[[dict], None]) -> None:
        """Watch resources, so a release wakes a dispatcher waiting for capacity."""
        self.connection.watch_collection(
            RESOURCES_COLLECTION,
            [{"$match": {"operationType": {"$in": ["insert", "update", "replace", "delete"]}}}],
            on_change,
            full_document="updateLookup",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Published crypto material
# ═══════════════════════════════════════════════════════════════════════════

class InfraCryptoRepository:
    """The public half of the master-node's keystore.

    Written by the master-node at startup, read by the API so the browser can
    seal a credential against the key that can actually open it. Nothing
    private is ever stored here — the private key stays in a mounted file.
    """

    DOC_ID = "active"

    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def get_active(self) -> Optional[dict[str, Any]]:
        with self.connection.connect() as db:
            return db[CRYPTO_COLLECTION].find_one({"_id": self.DOC_ID})

    def publish(
        self,
        *,
        alg: str,
        key_id: str,
        public_key: str,
        fingerprint: str,
        retired_key_ids: Optional[list[str]] = None,
    ) -> None:
        now = datetime.now()
        with self.connection.connect() as db:
            db[CRYPTO_COLLECTION].update_one(
                {"_id": self.DOC_ID},
                {
                    "$set": {
                        "alg": alg,
                        "key_id": key_id,
                        "public_key": public_key,
                        "fingerprint": fingerprint,
                        "retired_key_ids": retired_key_ids or [],
                        "updated_at": now,
                    },
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )


def host_access_secret(host: Host) -> Optional[SecretEnvelope]:
    """The hop-1 credential envelope, or ``None`` for a local host."""
    access = host.get("access") or {}
    return access.get("secret")


def container_access_secret(host: Host) -> Optional[SecretEnvelope]:
    """The hop-2 credential envelope."""
    return (host.get("container_access") or {}).get("secret")
