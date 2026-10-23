"""Persistence for the distributed execution infrastructure.

The reservation primitives live here rather than in the scheduler on purpose:
the invariants that two simulations never share a container, and that a health
check never steals a container from a running simulation, are enforced by the
query filters themselves. No caller can get them wrong by forgetting a check.

Four repositories:

* ``HostRepository`` — registrations, check/discovery requests, and the
  inventory snapshot the scheduler reads.
* ``CoojaResourceRepository`` — discovered containers, and the atomic
  reserve/start/release cycle.
* ``InfraCryptoRepository`` — the published public key, so the browser can seal
  a credential against the key the master-node actually holds.
* ``MasterStatusRepository`` — the master-node's own heartbeat and the state of
  its DNS resolver, which no host document can carry.
"""
import logging
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from bson import ObjectId, errors
from pymongo import ReturnDocument
from pymongo.errors import OperationFailure

from pylib.crypto.envelope import SecretEnvelope
from pylib.db.connection import MongoDBConnection
from pylib.timeutil import utc_now
from pylib.db.models.infra import (
    CRYPTO_COLLECTION,
    MASTER_STATUS_COLLECTION,
    target_fingerprint,
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


class _Abort(Exception):
    """Roll the transaction back and report what stopped it.

    Raised rather than returned because the transaction's context manager
    aborts on an exception and commits on a normal return. A composite
    operation that decides *not* to go ahead has to leave by this door, or the
    writes it already made are committed.
    """

    def __init__(self, deleted: bool, blocking: list[str]):
        super().__init__("transaction aborted")
        self.deleted = deleted
        self.blocking = blocking


class RemovalConflicted(RuntimeError):
    """A registration or a discovery committed while the removal was running.

    Distinct from "the host is not there" and from "a worker is busy": nothing
    was removed, nothing is wrong, and trying again is the correct response.
    Collapsing it into the not-found branch answered 404 for a host that still
    exists.
    """


class HostWithdrawing(RuntimeError):
    """A worker was offered to a host that is being removed.

    Raised rather than returned, because every caller of the creation paths
    treats a falsy answer as "nothing to do" and this needs to reach the API
    as a refusal the operator can read.
    """


#: How fast the occupancy estimate forgets. At 0.2 a worker's mean is
#: effectively the last ~10 runs: slow enough that one unlucky simulation does
#: not move it, fast enough to follow a machine that got busier.
DURATION_ALPHA = 0.2

_OCCUPIED = [s.value for s in OCCUPIED_STATUSES]
_SCHEDULABLE = [s.value for s in SCHEDULABLE_HOST_STATUSES]

# Statuses a health check or a discovery pass may overwrite. The occupied ones
# are absent by design: losing contact with a host says nothing about whether
# the simulation on it is still running, so its reservation stands until the
# reconciliation layer decides otherwise.
#: ``enabled`` as a boolean, whatever the document holds. A registration
#: written before the flag existed has no field, and a missing value compares
#: equal to neither ``True`` nor ``False`` — so a worker would fall out of
#: every bucket of the inventory partition. Absent means *in the pool*, which
#: is what those documents meant.
_ENABLED = {"$ne": [{"$ifNull": ["$$r.enabled", True]}, False]}

_REASSIGNABLE = [
    EnumResourceStatus.AVAILABLE.value,
    EnumResourceStatus.OFFLINE.value,
    EnumResourceStatus.ERROR.value,
    EnumResourceStatus.DISABLED.value,
]


def released_status_expression(
    *, container_state_counts: bool, config_version: Optional[int] = None
) -> dict[str, Any]:
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

    ``config_version`` is the host's current one. Given, a worker whose probe
    measured an **older** configuration lands ``offline`` instead of
    ``available``: an access edit takes the idle workers out of the pool, but
    it deliberately leaves a busy one alone, and that worker used to come back
    as available on the strength of a measurement of how it was reached
    before.

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
    if config_version is not None:
        branches.append({
            # The host's access settings changed while this worker was busy.
            # Its probe describes the previous configuration, so letting the
            # release decide availability from it would put the worker back in
            # the pool on evidence about something else — the one path an
            # access edit could not reach, because an occupied worker is left
            # alone on purpose.
            "case": {"$ne": [
                {"$ifNull": ["$probe.config_version", 0]}, config_version]},
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
        now = utc_now()
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
        """Hosts the health cycle and discovery may act on.

        A host being withdrawn is excluded: discovery would re-create the very
        workers the removal is deleting, and a probe would put them back in
        the pool. Runs already on it are unaffected — the runner holds its own
        connection and does not come through here.
        """
        with self.connection.connect() as db:
            return list(db[HOSTS_COLLECTION].find(
                {"enabled": True, "withdrawn_at": None}, sort=[("name", 1)]))

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
            "backoff_until", "resolved_address", "created_time", "_id",
        }
        payload = {k: v for k, v in updates.items() if k not in protected}
        if not payload:
            return False
        payload["updated_time"] = utc_now()
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one({"_id": oid}, {"$set": payload})
            return res.matched_count > 0

    def bump_config_version(self, host_id: Any) -> int:
        """Record that the way this host's workers are reached has changed.

        R22. Clearing a probe is a *point* invalidation, and the two things it
        does not cover are the two that matter:

        * a probe that **started before** the edit can finish after the clear
          and write a positive result describing the old configuration;
        * the worker's ``status`` is untouched, so it stays in the pool and a
          placement can use a configuration nobody has measured.

        A version makes both decidable. The measurement carries the version it
        was taken under, and a reader can see that it describes something
        else. Returns the new value.
        """
        oid = _oid(host_id)
        if oid is None:
            return 0
        with self.connection.connect() as db:
            # The backoff goes with it: it was earned by the configuration
            # that just changed, and the operator who changed it is waiting to
            # see whether the new one works.
            doc = db[HOSTS_COLLECTION].find_one_and_update(
                {"_id": oid},
                {"$inc": {"config_version": 1},
                 "$set": {"updated_time": utc_now(), "backoff_until": None}},
                return_document=ReturnDocument.AFTER,
                projection={"config_version": 1},
            )
            return int((doc or {}).get("config_version") or 0)

    def pin_host_key(self, host_id: Any, fingerprint: str) -> bool:
        """Record a host key learned on first contact, and only then.

        Conditional on the field being unset, so a second first-contact —
        from another code path, or a reconnection that built its policy
        without the stored pin — cannot overwrite what was already trusted.
        Trust on first use is only a guarantee if the *first* use is the one
        that sticks.

        Returns whether this call was the one that pinned it.
        """
        oid = _oid(host_id)
        if oid is None or not fingerprint:
            return False
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one(
                {
                    "_id": oid,
                    "$or": [
                        {"access.host_key_fingerprint": None},
                        {"access.host_key_fingerprint": {"$exists": False}},
                    ],
                },
                {"$set": {
                    "access.host_key_fingerprint": fingerprint,
                    "updated_time": utc_now(),
                }},
            )
            return res.modified_count > 0

    def set_enabled(self, host_id: Any, enabled: bool) -> bool:
        """Take a host in or out of the pool.

        Disabling only stops new placements; running simulations are left to
        finish, which is what drains the host. The resources follow when they
        are released, not now.

        Re-enabling moves the status off ``disabled``, for the same reason as
        the worker's: ``disabled`` is not a second copy of the flag, and a
        host switched back on stayed unschedulable *and labelled disabled*
        until the next health cycle contradicted its own toggle. ``offline``
        says what is true — in the pool, not yet measured.
        """
        oid = _oid(host_id)
        if oid is None:
            return False
        updates: dict[str, Any] = {"enabled": enabled, "updated_time": utc_now()}
        if not enabled:
            updates["status"] = EnumHostStatus.DISABLED.value
        with self.connection.connect() as db:
            if enabled:
                # Measured at the next cycle, not after whatever backoff it
                # had accumulated before it was switched off.
                updates["backoff_until"] = None
                db[HOSTS_COLLECTION].update_one(
                    {"_id": oid, "status": EnumHostStatus.DISABLED.value},
                    {"$set": {
                        "status": EnumHostStatus.OFFLINE.value,
                        "last_check_message":
                            "back in the pool; waiting to be measured",
                    }},
                )
            res = db[HOSTS_COLLECTION].update_one({"_id": oid}, {"$set": updates})
            return res.modified_count > 0

    def begin_withdrawal(self, host_id: Any) -> bool:
        """Mark a host as being taken out, before anything is deleted.

        The first half of the two-phase removal. Deleting the workers and then
        the host is two or more writes, and between them a discovery pass or a
        registration that had *already read* this host could insert a new
        worker — which then outlived the host it belonged to. Reproduced
        against the real collection: the host was removed and one worker was
        left orphaned (R09).

        So a withdrawal is a state, not an instant. While it is set:

        * :meth:`find_enabled` skips the host, so neither the health cycle nor
          discovery touches it;
        * :meth:`inventory_snapshot` reports it unschedulable, so nothing new
          is placed on it;
        * resource creation refuses, which is what closes the orphan window.

        Runs already on the host are **not** disturbed: they keep their
        reservations and finish, and the deletion refuses until they have.
        """
        oid = _oid(host_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one(
                {"_id": oid}, {"$set": {
                    "withdrawn_at": utc_now(),
                    "updated_time": utc_now(),
                }})
            return res.matched_count > 0

    def cancel_withdrawal(self, host_id: Any) -> bool:
        """Put a host back in service after a refused removal.

        A withdrawal that cannot complete must not leave the host silently
        unschedulable: that would take its capacity out of the pool with
        nothing in the interface saying why.
        """
        oid = _oid(host_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one(
                {"_id": oid},
                {"$unset": {"withdrawn_at": ""},
                 "$set": {"updated_time": utc_now()}})
            return res.matched_count > 0

    def is_withdrawing(self, host_id: Any) -> bool:
        oid = _oid(host_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            return db[HOSTS_COLLECTION].count_documents(
                {"_id": oid, "withdrawn_at": {"$ne": None}}, limit=1) > 0

    def delete_if_drained(self, host_id: Any) -> tuple[bool, list[str]]:
        """Remove a host and its workers, if none of them is holding anything.

        Returns ``(deleted, blocking)``: the names that stopped it, empty when
        it went through.

        Three phases, and each one exists because of a way the simpler
        versions failed:

        1. **Withdraw** the host, so nothing new can be created on it or
           placed on it. Without this, a creation that had already read the
           host inserted a worker after the list below was taken, and the
           worker outlived its host.
        2. **Delete each worker conditionally.** Counting the busy ones first
           and then deleting everything left a window the scheduler could
           reserve in, and removed the host along with the record of a live
           run.
        3. **Delete the host only if nothing is left**, re-counting rather
           than trusting step 2. A worker that appears between them keeps the
           host alive and is reported.

        Steps 2 and 3 share a transaction: a refusal rolls back every worker
        deletion and returns the host to service. A retained run/result also
        blocks removal, even when quarantine cleared the active reservation.
        ``deleted`` is true only when no worker remains.
        """
        oid = _oid(host_id)
        if oid is None:
            return False, []
        if not self.begin_withdrawal(oid):
            return False, []
        try:
            return self._remove_withdrawn(oid)
        except _Abort as abort:
            self.cancel_withdrawal(oid)
            return abort.deleted, abort.blocking
        except OperationFailure as error:
            # A write conflict means a registration or a discovery committed
            # against this host while the removal was running. The withdrawal
            # is given back and the caller is told to try again — which is the
            # correct outcome: the alternative was removing the host and
            # discovering the new worker afterwards.
            log.warning(
                "The removal of host %s conflicted with a concurrent "
                "registration: %s", oid, error,
            )
            self.cancel_withdrawal(oid)
            raise RemovalConflicted(
                "a worker was registered on this host while it was being "
                "removed; nothing was removed"
            ) from error
        except Exception:
            # Anything else leaves the host withdrawn, which is the one state
            # a half-finished removal must not be left in silently.
            log.exception("The removal of host %s failed part-way", oid)
            self.cancel_withdrawal(oid)
            raise

    def _remove_withdrawn(self, oid: ObjectId) -> tuple[bool, list[str]]:
        """Phases two and three, atomically.

        One transaction, and it writes the **host document** as well as the
        workers. That is not incidental: a transaction gives snapshot
        isolation, which detects a conflict on a document two transactions
        both write but does *not* see a document inserted outside its
        snapshot. Reading the host would therefore not exclude a concurrent
        registration; writing it does, because every creation path bumps
        ``resource_epoch`` on the host inside its own transaction. The two
        conflict and one aborts.
        """
        with self.connection.transaction() as (db, session):
            hosts = db[HOSTS_COLLECTION]
            resources = db[RESOURCES_COLLECTION]

            # Writing the host is what makes a concurrent registration
            # conflict with this. It has to happen inside the transaction and
            # before the counting.
            hosts.update_one(
                {"_id": oid}, {"$inc": {"resource_epoch": 1}}, session=session)

            blocking: list[str] = []
            for resource in resources.find(
                {"host_id": oid}, {"_id": 1, "name": 1}, session=session
            ):
                gone = resources.delete_one({
                    "_id": resource["_id"],
                    "status": {"$nin": _OCCUPIED},
                    "current_simulation_id": None,
                    "held_for": None,
                }, session=session).deleted_count
                if not gone:
                    blocking.append(str(resource.get("name", "?")))

            if not blocking:
                # Re-counted inside the same transaction: the loop above only
                # knows about the workers it listed.
                blocking = [
                    str(r.get("name", "?"))
                    for r in resources.find(
                        {"host_id": oid}, {"name": 1}, session=session)
                ]
            if blocking:
                # Raised, not returned: the transaction's context manager
                # aborts on an exception, so the workers deleted above come
                # back. That is the difference the transaction makes — the
                # refusal is no longer partial.
                raise _Abort(False, blocking)

            if hosts.delete_one(
                {"_id": oid}, session=session
            ).deleted_count == 0:
                raise _Abort(False, [])
            return True, []

    # ── bootstrap ──────────────────────────────────────────────────────────

    def upsert_bootstrap(self, host: Host) -> ObjectId:
        """Create the host described by the environment, or leave it alone.

        Runs on every master-node start, so it must add what is missing without
        undoing administrative edits: ``enabled``, ``access`` and
        ``container_access`` are set only on insert. That is what keeps
        ``docker compose up`` working with no registration step while still
        letting an operator reconfigure the local host afterwards.
        """
        now = utc_now()
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
        backoff_until: Optional[datetime] = None,
    ) -> bool:
        """Record the outcome of a reachability check.

        ``backoff_until`` is written every time — ``None`` clears it — so a
        host that recovers is checked at the normal pace again at once.
        """
        oid = _oid(host_id)
        if oid is None:
            return False
        now = utc_now()
        update: dict[str, Any] = {
            "$set": {
                "status": status.value,
                "last_check": now,
                "last_check_message": message,
                "backoff_until": backoff_until,
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

    def mark_check_inconclusive(self, host_id: Any, message: str) -> bool:
        """Record a check that learned nothing about the host.

        For a failure on the master-node's side — its DNS resolver not
        answering. The status, the failure count and the backoff stay as they
        were: counting it against the host took every host reached by name
        offline at once, for a fault none of them had. ``last_check`` still
        moves, because the check did run.
        """
        oid = _oid(host_id)
        if oid is None:
            return False
        now = utc_now()
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one(
                {"_id": oid, "enabled": True},
                {"$set": {"last_check": now, "last_check_message": message}},
            )
            return res.matched_count > 0

    def record_resolved_address(self, host_id: Any, name: str, ip: str) -> bool:
        """Keep the address a connection to ``name`` last succeeded on.

        Written only when it changed, so a host that keeps its address costs
        no write per cycle. Read back at startup: an address learned before a
        restart is what lets the master-node reach a pinned host while its
        DNS is down.
        """
        oid = _oid(host_id)
        if oid is None or not name or not ip:
            return False
        with self.connection.connect() as db:
            res = db[HOSTS_COLLECTION].update_one(
                {
                    "_id": oid,
                    "$or": [
                        {"resolved_address.ip": {"$ne": ip}},
                        {"resolved_address.name": {"$ne": name}},
                    ],
                },
                {"$set": {"resolved_address": {
                    "name": name, "ip": ip, "since": utc_now()}}},
            )
            return res.modified_count > 0

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
                {"_id": oid}, {"$set": {field: utc_now()}})
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
        now = utc_now()
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
                    "pipeline": [{"$project": {
                        "status": 1, "enabled": 1, "mean_duration_sec": 1}}],
                }
            },
            {
                "$set": {
                    "resource_total": {"$size": "$_res"},
                    "free": count_where({
                        "$and": [
                            {"$eq": [_ENABLED, True]},
                            {"$eq": ["$$r.status", available]},
                        ]
                    }),
                    # The four below are a *partition* of resource_total, in
                    # this precedence: doing work, administratively out,
                    # enabled but not usable, free. They used to overlap — a
                    # container that was busy and disabled for draining
                    # counted in both, so one worker appeared twice and the
                    # bar's denominator exceeded the inventory.
                    #
                    # The administrative bucket follows **`enabled`**, which
                    # is the operator's intent, and not the status. Keying it
                    # on `status == disabled` as well put a worker that had
                    # just been switched back **on** into the "disabled"
                    # segment: the status only leaves that value when the next
                    # health cycle re-measures the container, so the bar
                    # contradicted the toggle next to it for as long as a
                    # cycle. The two are different facts — one is a decision,
                    # the other a measurement — and only the decision belongs
                    # in this bucket.
                    #
                    # `unavailable` is therefore the **complement** rather
                    # than a list of statuses: enabled, not occupied, not
                    # available. Written as a list, a status nobody
                    # anticipated — or the one above — falls into no bucket at
                    # all and the segments stop adding up to the inventory.
                    "active": count_where({"$in": ["$$r.status", _OCCUPIED]}),
                    "disabled": count_where({
                        "$and": [
                            {"$not": {"$in": ["$$r.status", _OCCUPIED]}},
                            {"$eq": [_ENABLED, False]},
                        ]
                    }),
                    "unavailable": count_where({
                        "$and": [
                            {"$not": {"$in": ["$$r.status", _OCCUPIED]}},
                            {"$eq": [_ENABLED, True]},
                            {"$ne": ["$$r.status", available]},
                        ]
                    }),
                    # Subsets, reported rather than summed. Draining is an
                    # attribute of a worker that is *in use*, and a
                    # reservation that has not started yet is not a running
                    # simulation — a card labelled "Running" that counted both
                    # was saying something untrue.
                    "draining": count_where({
                        "$and": [
                            {"$in": ["$$r.status", _OCCUPIED]},
                            {"$eq": [_ENABLED, False]},
                        ]
                    }),
                    "running": count_where({
                        "$eq": ["$$r.status", EnumResourceStatus.BUSY.value]
                    }),
                    "reserved": count_where({
                        "$eq": ["$$r.status", EnumResourceStatus.RESERVED.value]
                    }),
                    # What the *next* placement on this host would cost: the
                    # estimate of the container `find_free_by_host` would
                    # pick. A host mean would be wrong here — the slow
                    # containers of a host whose fast one is free do not
                    # affect the job being placed now.
                    #
                    # Null when no free container has been measured yet, which
                    # a speed-aware policy must read as "unknown", not as
                    # "infinitely slow".
                    "fastest_free_sec": {"$min": {"$map": {
                        "input": {"$filter": {
                            "input": "$_res", "as": "r",
                            "cond": {"$and": [
                                {"$eq": ["$$r.enabled", True]},
                                {"$eq": ["$$r.status", available]},
                                {"$ne": [
                                    {"$ifNull": ["$$r.mean_duration_sec", None]},
                                    None]},
                            ]},
                        }},
                        "as": "r", "in": "$$r.mean_duration_sec",
                    }}},
                    "schedulable": {
                        "$and": [
                            {"$eq": ["$enabled", True]},
                            {"$in": ["$status", _SCHEDULABLE]},
                            # A host being withdrawn takes no new work. Its
                            # `free` count stays honest — the containers are
                            # there and the interface should say so — but
                            # nothing is placed on them.
                            {"$eq": [
                                {"$ifNull": ["$withdrawn_at", None]}, None]},
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
            self._ensure_target_index(db)

    @staticmethod
    def _ensure_target_index(db: Any) -> None:
        """One registration per physical target on a host, enforced by the index.

        ``conflicting_target`` reads and then the caller inserts, which two
        requests can interleave through. For the realistic case — two names
        for one container on one host — the database can decide instead, and a
        unique index is the only thing that actually serialises it.

        Partial, on ``endpoint.address``: a freshly seeded worker has no
        endpoint until discovery describes it, and every one of those would
        otherwise collide on a missing field. Cross-host aliases are not
        covered, because the host's own address is part of the identity and is
        not in this collection; those stay with the read-then-insert check and
        are documented as such.

        A failure is logged rather than raised. An existing deployment may
        already hold a duplicate, and refusing to start over an index is worse
        than starting without it.
        """
        try:
            db[RESOURCES_COLLECTION].create_index(
                [("host_id", 1), ("endpoint.address", 1),
                 ("endpoint.port", 1), ("remote_dir", 1)],
                unique=True,
                name="idx_resources_target",
                partialFilterExpression={"endpoint.address": {"$type": "string"}},
            )
        except Exception as error:
            log.warning(
                "Could not create the unique target index; two registrations "
                "for one worker would only be caught by the API check: %s",
                error,
            )

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

    def find_free_by_host(
        self, host_id: Any, *, prefer_fastest: bool = False
    ) -> list[CoojaResource]:
        """Candidates for placement on one host.

        Two orderings, for two different scheduling policies:

        * by ``last_check`` (the default) spreads work across a host's
          containers instead of hammering whichever one sorts first;
        * by ``mean_duration_sec`` puts the container that *finishes* soonest
          first. Under saturation this changes nothing — every container is
          busy and the list has one entry — and when there is slack it is the
          whole point: an idle fast container should take the work before an
          idle slow one.

        A container with no measurement sorts **first** either way. It is the
        optimistic reading, and it is self-limiting: one run gives it a real
        number. The pessimistic one would leave a newly registered worker
        last for as long as the pool has slack, which is exactly when it
        would otherwise be getting its first samples.
        """
        oid = _oid(host_id)
        if oid is None:
            return []
        # MongoDB sorts missing before any value ascending, which is the
        # optimistic placement the docstring promises — stated here because it
        # is a property of the database, not of this code.
        sort = ([("mean_duration_sec", 1), ("name", 1)] if prefer_fastest
                else [("last_check", 1), ("name", 1)])
        with self.connection.connect() as db:
            return list(db[RESOURCES_COLLECTION].find(
                {
                    "host_id": oid,
                    "enabled": True,
                    "status": EnumResourceStatus.AVAILABLE.value,
                },
                sort=sort,
            ))

    def record_duration(self, resource_id: Any, seconds: float) -> Optional[float]:
        """Fold one completed run into this worker's occupancy estimate.

        How long a container is *held* — connection, upload, run, collection —
        is what decides throughput, so that is what is measured, rather than
        the simulation's own ``start_time``/``end_time``.

        An exponentially weighted mean, not a lifetime average: a host's speed
        changes with the load on the machine it shares, and an average over
        every run it ever did would keep reporting last month's. ``ALPHA``
        below sets how fast it forgets.

        What this estimate **cannot** separate is a slow worker from a worker
        that happened to receive harder simulations: duration varies more
        between individuals than between machines. It converges anyway,
        because placement is uncorrelated with difficulty and the differences
        worth acting on are large — measured on a three-host pool, the same
        genome with the same seed took 1.22–1.25x longer on the remote hosts,
        against a residual noise of a few per cent after a dozen samples.

        Only a run that *succeeded* is recorded. A failure that aborted in
        five seconds would otherwise make the worst worker look like the best.

        Returns the new mean, or ``None`` when nothing was written.
        """
        oid = _oid(resource_id)
        if oid is None or not seconds or seconds <= 0:
            return None
        with self.connection.connect() as db:
            current = db[RESOURCES_COLLECTION].find_one(
                {"_id": oid}, {"mean_duration_sec": 1, "duration_samples": 1})
            if current is None:
                return None
            previous = current.get("mean_duration_sec")
            updated = (float(seconds) if previous is None
                       else (1 - DURATION_ALPHA) * float(previous)
                       + DURATION_ALPHA * float(seconds))
            db[RESOURCES_COLLECTION].update_one({"_id": oid}, {"$set": {
                "mean_duration_sec": round(updated, 3),
                "duration_samples": int(current.get("duration_samples") or 0) + 1,
                "duration_measured_at": utc_now(),
            }})
            return updated

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
        self,
        resource_id: Any,
        simulation_id: ObjectId,
        attempt_id: str,
        *,
        expect_config_version: Optional[int] = None,
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
        now = utc_now()
        condition: dict[str, Any] = {
            "_id": oid,
            "enabled": True,
            "status": EnumResourceStatus.AVAILABLE.value,
            "held_for": None,
        }
        if expect_config_version is not None:
            # The measurement has to be about the configuration in force. A
            # probe that started before an access edit and published after it
            # could leave a worker `available` on the strength of a path that
            # no longer exists; the status filter alone cannot see that, and
            # the window between the two is a placement.
            #
            # Through `$ifNull`, not an equality on the field: a probe written
            # before this existed has no `config_version` at all, and
            # `{"probe.config_version": 0}` does not match a missing field. On
            # an existing deployment that would have made every worker
            # unschedulable until it was measured again — and `probe_is_fresh`
            # reads a missing field as 0, so a worker that had just been
            # probed would never be re-probed. An absent value means version
            # zero here exactly as it does everywhere else.
            condition["$expr"] = {"$eq": [
                {"$ifNull": ["$probe.config_version", 0]},
                int(expect_config_version),
            ]}
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].find_one_and_update(
                condition,
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
        now = utc_now()
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

    def touch_reservation(
        self, resource_id: Any, simulation_id: ObjectId, attempt_id: str
    ) -> bool:
        """Renew the lease on a reservation that is still being prepared.

        The stale-reservation sweep frees a container that was claimed and
        never started, and its TTL has to cover the slowest legitimate
        preparation: downloading the firmware from GridFS and sending it over
        a tunnel. Those are minutes, and a margin is not a guarantee — an
        upload slower than the TTL had its container freed underneath it while
        the thread went on writing to the directory.

        So the thread doing the work says so, and the sweep's
        ``reserved_before`` condition then refuses. This is a lease, with the
        properties a lease needs: only the holder can renew it (the attempt
        guard), renewal is refused once the reservation is gone, and a thread
        that dies stops renewing — so nothing is held for ever by a process
        that is not there.

        Refused once the reservation became ``busy``: from then on the run is
        governed by the simulation timeout, not by this.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = utc_now()
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                {
                    "_id": oid,
                    "status": EnumResourceStatus.RESERVED.value,
                    "current_simulation_id": simulation_id,
                    "attempt_id": attempt_id,
                },
                {"$set": {"reserved_at": now, "updated_time": now}},
            )
            return res.modified_count > 0

    def quarantine(
        self,
        resource_id: Any,
        reason: str,
        *,
        expect_attempt_id: Optional[str] = None,
        held_for: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Take a worker out of the pool **and** stop holding it, in one write.

        For the case where a run could not be confirmed stopped. Two writes
        were used before — ``set_enabled(False)`` and then the ordinary
        release — and the failure mode was the reason this method exists: when
        the first write failed, the second still ran, and a container that may
        still have had a JVM on it went back into the pool as ``available``.
        The protective mechanism failing was followed by the unprotective one
        succeeding.

        One document, one update: either the worker is disabled *and* released
        to ``disabled``, or nothing changed and it keeps holding the attempt —
        which is also out of the pool, because ``find_free_by_host`` requires
        ``available``. There is no interleaving that produces an eligible
        worker.

        ``remote_pid`` is deliberately kept. It is the only record of what may
        still be running in there, and an operator looking at a disabled
        worker needs it.

        ``held_for`` is the **persistent** half, and it is what makes the
        protection survive this process. Disabling a worker was enough to keep
        the *dispatcher* off it, but not the reconciliation: that pass found a
        simulation in a terminal state, concluded nothing was running and
        called ``force_release`` with no probe — so a quarantine whose write
        had failed, or one whose worker still had a JVM on it, was undone by
        the next health cycle. :meth:`force_release` and :meth:`release` now
        refuse while this field is set, and
        :meth:`clear_held_for` is the only way out.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = utc_now()
        condition: dict[str, Any] = {"_id": oid}
        if expect_attempt_id is not None:
            condition["attempt_id"] = expect_attempt_id
        fields: dict[str, Any] = {
            "enabled": False,
            "status": EnumResourceStatus.DISABLED.value,
            "current_simulation_id": None,
            "attempt_id": None,
            "reserved_at": None,
            "started_at": None,
            "quarantined_at": now,
            "last_check_message": reason,
            "updated_time": now,
        }
        if held_for is not None:
            fields["held_for"] = {**held_for, "at": now, "reason": reason}
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                condition, {"$set": fields})
            return res.modified_count > 0

    def hold(
        self, resource_id: Any, reason: str, held_for: dict[str, Any]
    ) -> bool:
        """Mark a worker as holding something unresolved, without releasing it.

        The difference from :meth:`quarantine` is that nothing about the
        reservation is cleared: the worker keeps whatever it was holding. Used
        when the attempt is over but the *container* still has something on it
        that nobody has accounted for — a run whose stop was never confirmed,
        or a result that could not be collected and could not be copied aside.

        Either way the worker is out of the pool and stays out until something
        looks. That is the point: the previous behaviour marked the simulation
        and let the ordinary release hand the worker on.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = utc_now()
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                {"_id": oid},
                {"$set": {
                    "enabled": False,
                    "held_for": {**held_for, "at": now, "reason": reason},
                    "last_check_message": reason,
                    "updated_time": now,
                }},
            )
            return res.modified_count > 0

    def find_held(self) -> list[CoojaResource]:
        """Workers held for something unresolved — the resolver's input."""
        with self.connection.connect() as db:
            return list(db[RESOURCES_COLLECTION].find(
                {"held_for": {"$ne": None}}))

    def clear_held_for(
        self, resource_id: Any, *, expect_attempt_id: Optional[str] = None
    ) -> bool:
        """Let a held worker back in, once something has established it is free.

        Deliberately narrow: it clears the hold and re-enables the worker, and
        the *status* is left to the next health cycle, which measures it. A
        caller that has not probed the container has no business calling this.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        condition: dict[str, Any] = {"_id": oid, "held_for": {"$ne": None}}
        if expect_attempt_id is not None:
            condition["held_for.attempt_id"] = expect_attempt_id
        now = utc_now()
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                condition,
                {"$unset": {"held_for": ""},
                 "$set": {
                     "enabled": True,
                     "status": EnumResourceStatus.OFFLINE.value,
                     "current_simulation_id": None,
                     "attempt_id": None,
                     "remote_pid": None,
                     "reserved_at": None,
                     "started_at": None,
                     "last_check_message": (
                         "released after the unresolved run was confirmed gone"),
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
        config_version: Optional[int] = None,
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
        now = utc_now()
        pipeline = [{"$set": {
            "status": released_status_expression(
                container_state_counts=container_state_counts,
                config_version=config_version),
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
                    # Nothing unresolved on the container. See `hold`.
                    "held_for": None,
                },
                pipeline,
                return_document=ReturnDocument.AFTER,
            )

    def is_held_by(
        self, resource_id: Any, simulation_id: ObjectId, attempt_id: str
    ) -> bool:
        """Whether this attempt still holds this container.

        A reservation is granted atomically, but everything that happens after
        it — assembling inputs, uploading them, launching — takes seconds to
        minutes, and a recovery pass or the stale-reservation sweep can hand
        the container on in between. Asking before touching the container is
        what keeps a lost attempt from writing over the inputs of the run that
        took over.

        ``reserved`` or ``busy`` both count: the question is who holds it, not
        how far along they are.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            return db[RESOURCES_COLLECTION].count_documents({
                "_id": oid,
                "current_simulation_id": simulation_id,
                "attempt_id": attempt_id,
                "status": {"$in": _OCCUPIED},
            }, limit=1) > 0

    def force_release(
        self,
        resource_id: Any,
        message: str = "",
        *,
        container_state_counts: bool,
        config_version: Optional[int] = None,
        expect_attempt_id: Optional[str] = None,
        reserved_before: Optional[datetime] = None,
        expect_reserved: bool = False,
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
        now = utc_now()
        pipeline = [{"$set": {
            "status": released_status_expression(
                container_state_counts=container_state_counts,
                config_version=config_version),
            "current_simulation_id": None,
            "attempt_id": None,
            "remote_pid": None,
            "reserved_at": None,
            "started_at": None,
            "last_check_message": message,
            "updated_time": now,
        }}]
        # A worker held for something unresolved is **not** released by this,
        # whatever the simulation says. That composition was the gap: the
        # runner correctly kept the worker when its quarantine write failed,
        # and the next reconciliation pass found a terminal simulation, assumed
        # nothing was running and freed it — with no probe and no guard. A
        # terminal status is not evidence that a remote process ended.
        condition: dict[str, Any] = {"_id": oid, "held_for": None}
        if expect_attempt_id is not None:
            condition["attempt_id"] = expect_attempt_id
        if expect_reserved:
            # Still waiting to start. A reservation that got as far as `busy`
            # is governed by the simulation timeout, not by this. Separate
            # from the attempt guard because the reconciliation needs the
            # guard over a container that *is* busy: it releases only the
            # attempt it read, and insisting on `reserved` there would make
            # the guard silently never match.
            condition["status"] = EnumResourceStatus.RESERVED.value
        if reserved_before is not None:
            condition["reserved_at"] = {"$lt": reserved_before}
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(condition, pipeline)
            return res.modified_count > 0

    # ── health and discovery (never touching an occupied resource) ─────────

    def mark_available(
        self,
        resource_id: Any,
        message: str = "",
        *,
        config_version: Optional[int] = None,
    ) -> bool:
        """Put a container into the pool after both check levels passed.

        ``config_version`` records which configuration the measurement behind
        this belongs to, in the same write. A reservation requires the two to
        match, so a probe that published after an access edit cannot leave a
        worker schedulable on evidence about the previous path.
        """
        extra = (
            {"probe.config_version": int(config_version)}
            if config_version is not None else None)
        return self._set_status(
            resource_id, EnumResourceStatus.AVAILABLE, message, extra=extra)

    def mark_offline(self, resource_id: Any, message: str = "") -> bool:
        return self._set_status(resource_id, EnumResourceStatus.OFFLINE, message)

    def mark_error(self, resource_id: Any, message: str = "") -> bool:
        return self._set_status(resource_id, EnumResourceStatus.ERROR, message)

    def _set_status(
        self,
        resource_id: Any,
        status: EnumResourceStatus,
        message: str,
        *,
        extra: Optional[dict[str, Any]] = None,
    ) -> bool:
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = utc_now()
        fields: dict[str, Any] = {
            "status": status.value,
            "last_check": now,
            "last_check_message": message,
            "updated_time": now,
        }
        if extra:
            fields.update(extra)
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                # Reserved and busy are absent from _REASSIGNABLE: a failing
                # check must never evict a simulation that may still be running.
                # `held_for` likewise: a worker holding something unresolved
                # must not be promoted into the pool by a measurement.
                {"_id": oid, "status": {"$in": _REASSIGNABLE}, "held_for": None},
                {"$set": fields},
            )
            return res.modified_count > 0

    def clear_probes_of_host(self, host_id: Any) -> int:
        """Forget what was measured about a host's workers.

        Called when the host's own access settings change. A worker inherits
        whatever it does not override — the address of hop 1, the container
        username, the port, the credential — so a stored probe describes a
        path that may no longer exist, exactly as it does when the worker's
        own settings change.

        Editing the *worker* already cleared its probe; editing the **host**
        did not, so the next placement could use a configuration nobody had
        measured while the page still showed the old result as evidence that
        it worked.

        A worker **holding a simulation is untouched**: the run in progress
        reached its container by a path that worked, and interrupting it over
        an edit would throw away work for nothing.

        Everything else leaves the pool. Forgetting the measurement while
        leaving the status at ``available`` was not enough — the scheduler
        decides on the status, so a placement could still land on a
        configuration nobody had measured in the window before the next cycle
        (R22). ``offline`` with a message saying so is the honest state: the
        next cycle measures it and puts it back, usually within seconds.

        Returns how many workers were affected.
        """
        oid = _oid(host_id)
        if oid is None:
            return 0
        now = utc_now()
        with self.connection.connect() as db:
            resources = db[RESOURCES_COLLECTION]
            affected = resources.update_many(
                {"host_id": oid},
                {"$set": {"probe": {}, "updated_time": now}},
            ).modified_count
            resources.update_many(
                {"host_id": oid, "status": {"$nin": _OCCUPIED}},
                {"$set": {
                    "status": EnumResourceStatus.OFFLINE.value,
                    "last_check_message": (
                        "waiting to be measured again: the host's access "
                        "settings changed"),
                    "updated_time": now,
                }},
            )
            return affected

    def set_enabled(self, resource_id: Any, enabled: bool) -> bool:
        """Administrative override. Survives every discovery pass.

        Disabling an occupied container lets it finish and leaves the pool when
        released, which is why the status is not forced here.

        Re-enabling moves the status **off** ``disabled``, to ``offline`` with
        a message. It used to leave it, and the status is not a second copy of
        the flag: a worker switched back on sat at ``disabled`` — with a stale
        "ready" from its last check — until the next health cycle, so the
        interface showed a toggle that was on next to a bar segment that said
        otherwise. ``offline`` is the honest reading of *enabled, and nothing
        has measured it since*; the next cycle promotes it, usually in
        seconds.

        Only from ``disabled``: a container that was offline or in error stays
        where it is, because re-enabling says nothing about what was measured.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = utc_now()
        updates: dict[str, Any] = {"enabled": enabled, "updated_time": now}
        with self.connection.connect() as db:
            if not enabled:
                db[RESOURCES_COLLECTION].update_one(
                    {"_id": oid, "status": {"$in": _REASSIGNABLE}},
                    {"$set": {"status": EnumResourceStatus.DISABLED.value}},
                )
            else:
                db[RESOURCES_COLLECTION].update_one(
                    {"_id": oid, "status": EnumResourceStatus.DISABLED.value},
                    {"$set": {
                        "status": EnumResourceStatus.OFFLINE.value,
                        "last_check_message":
                            "back in the pool; waiting to be measured",
                    }},
                )
            res = db[RESOURCES_COLLECTION].update_one({"_id": oid}, {"$set": updates})
            return res.matched_count > 0

    def update_probe(
        self,
        resource_id: Any,
        probe: ResourceProbe,
        *,
        config_version: Optional[int] = None,
    ) -> bool:
        """Record functional readiness without deciding the status.

        ``config_version`` is the host's at the moment the measurement was
        taken. Stored with it so a reader can tell a probe that describes the
        current configuration from one that describes the previous — a probe
        started before an access edit finishes after it, and otherwise writes
        a positive result for a path that no longer exists (R22).
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        now = utc_now()
        record: dict[str, Any] = {**probe, "checked_at": now}
        if config_version is not None:
            record["config_version"] = int(config_version)
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                {"_id": oid},
                {"$set": {"probe": record, "updated_time": now}},
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
        now = utc_now()
        endpoints = endpoints or {}
        seen: list[str] = []
        created: list[str] = []
        updated: list[str] = []

        # One transaction, because this both creates and removes workers and a
        # removal of the host may be running at the same time. A discovery
        # that started before the withdrawal would otherwise re-create the
        # very workers the removal is deleting: the pass may have read the
        # host minutes ago, so the check has to be where the writes are, and
        # it has to *write* the host to conflict with the other side.
        try:
            return self._sync_discovered(
                host_id, containers, remote_dir=remote_dir,
                endpoints=endpoints, now=now, seen=seen,
                created=created, updated=updated)
        except OperationFailure as error:
            # A write conflict with a concurrent removal or registration. The
            # pass does nothing this cycle and runs again in thirty seconds;
            # reporting nothing missing is what keeps it from acting on a
            # snapshot it could not complete.
            log.warning(
                "Discovery on host %s conflicted with a concurrent change: "
                "%s", host_id, error,
            )
            return {"created": [], "updated": [], "missing": [],
                    "missing_occupied": []}

    def _sync_discovered(
        self,
        host_id: ObjectId,
        containers: Iterable[ContainerInfo],
        *,
        remote_dir: str,
        endpoints: dict[str, dict[str, Any]],
        now: datetime,
        seen: list[str],
        created: list[str],
        updated: list[str],
    ) -> dict[str, list[str]]:
        with self.connection.transaction() as (db, session):
            self._claim_host_for_creation(db, host_id, session)
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

                # The container's id is the identity when there is one; the
                # name is only how a record is first recognised. Matching on
                # the name alone meant a record whose name had diverged — a
                # rename here, or one in Docker — was not recognised, and the
                # container was inserted a second time: two registrations and
                # two reservations for one machine, with nothing in the
                # database able to say they were the same.
                container_id = container.get("id")
                known = None
                if container_id:
                    known = coll.find_one(
                        {"host_id": host_id, "container.id": container_id},
                        {"_id": 1, "name": 1},
                        session=session,
                    )
                if known is not None:
                    if known.get("name") != name:
                        log.info(
                            "Container %s is registered as %r; following the "
                            "rename", container_id, known.get("name"),
                        )
                        set_fields["name"] = name
                    result = coll.update_one({"_id": known["_id"]}, {"$set": set_fields}, session=session)
                    identified_by = {"_id": known["_id"]}
                else:
                    result = coll.update_one(
                        {"host_id": host_id, "name": name},
                        {"$set": set_fields, "$setOnInsert": set_on_insert},
                        upsert=True,
                        session=session,
                    )
                    identified_by = {"host_id": host_id, "name": name}

                if endpoint and result.upserted_id is None:
                    # Refreshing the address of a worker that already exists
                    # is only discovery's business where discovery owns it. A
                    # worker whose address a person typed keeps it: the
                    # operator is the authority on how it is reached, and
                    # replacing that on the next inspection would silently
                    # undo a deliberate setting — the same division of
                    # authority the health cycle and the release apply.
                    coll.update_one(
                        {**identified_by, "origin": {"$ne": "manual"}},
                        {"$set": {"endpoint": endpoint}},
                        session=session,
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
            }, session=session))
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
                    session=session,
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
                    session=session,
                )

        return {
            "created": created,
            "updated": updated,
            "missing": missing,
            "missing_occupied": missing_occupied,
        }

    @staticmethod
    def _claim_host_for_creation(db: Any, host_id: Any, session: Any) -> None:
        """Refuse a withdrawn host, and make this creation conflict with one.

        Two things, and the second is the one that was missing. Reading the
        host refuses a withdrawal that is *already* visible; it does nothing
        about a removal that starts a moment later, because a transaction's
        snapshot does not see a document someone else inserts — so the
        removal's own counting could not see this worker either.

        Writing ``resource_epoch`` fixes that by making both sides write the
        same document. MongoDB then detects the write conflict and one of the
        two aborts, whichever order they arrive in.
        """
        if host_id is None:
            return
        host = db[HOSTS_COLLECTION].find_one(
            {"_id": host_id}, {"withdrawn_at": 1}, session=session)
        if host is None:
            raise HostWithdrawing(
                f"host {host_id} no longer exists; no worker may be added to it")
        if host.get("withdrawn_at") is not None:
            raise HostWithdrawing(
                f"host {host_id} is being removed; no worker may be added to it"
            )
        db[HOSTS_COLLECTION].update_one(
            {"_id": host_id}, {"$inc": {"resource_epoch": 1}}, session=session)

    def conflicting_target(
        self, host: Host, resource: CoojaResource, exclude_id: Any = None
    ) -> Optional[CoojaResource]:
        """Another registration that resolves to the same physical worker.

        Two names for one container mean two reservations for one run
        directory, and then each run's defensive cleanup deletes the other's
        inputs — while the lock that exists to prevent it is held on the wrong
        document (R08). Discovery cannot produce this, because it keys on the
        container id; a hand-registered worker can.

        Compared across *all* hosts, since the fingerprint includes the host's
        own address: the same container reached through two different host
        registrations is still one container.
        """
        fingerprint = target_fingerprint(host or {}, resource)
        if fingerprint is None:
            return None
        exclude = _oid(exclude_id) if exclude_id is not None else None
        with self.connection.connect() as db:
            hosts = {h["_id"]: h for h in db[HOSTS_COLLECTION].find()}
            for other in db[RESOURCES_COLLECTION].find():
                if exclude is not None and other["_id"] == exclude:
                    continue
                if target_fingerprint(
                    hosts.get(other.get("host_id")) or {}, other
                ) == fingerprint:
                    return other
        return None

    def insert_manual(self, resource: CoojaResource) -> ObjectId:
        """Register one worker with its own connection parameters.

        The counterpart to discovery, for a host whose SSH user cannot reach
        its Docker daemon — a shared machine — or wherever stating the
        addresses is simply clearer than deriving them.

        Starts ``offline`` like a discovered one: being registered is not
        evidence that it works, and the SSH probe is what puts it in the pool.
        """
        now = utc_now()
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
        with self.connection.transaction() as (db, session):
            self._claim_host_for_creation(db, doc.get("host_id"), session)
            return db[RESOURCES_COLLECTION].insert_one(
                doc, session=session).inserted_id

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
        payload["updated_time"] = utc_now()
        if "connection" in payload or "endpoint" in payload:
            # Configuring how a worker is reached takes ownership of it, even
            # if it first appeared through discovery. `origin` then means "who
            # decides this worker's connection", which is the distinction the
            # health cycle and the discovery sync both need — without it, the
            # next discovery pass marks a hand-configured worker as missing
            # and the probe never runs.
            #
            # And it has to be reversible, or "remove the override and inherit
            # again" would not actually inherit: the worker would stay flagged
            # as operator-configured for ever and Docker's view of it would go
            # on being ignored. Ownership follows the overrides, in both
            # directions — except for a worker that was registered by hand in
            # the first place, which Docker never found and cannot describe.
            payload["origin"] = self._origin_after(oid, payload)
            # The stored probe measured the *old* address, so it no longer
            # says anything about this worker: keeping it would show a result
            # for a path that no longer exists, and let the health cycle skip
            # the one container that most needs measuring. Cleared rather than
            # falsified — the next cycle fills it in, within one interval.
            payload["probe"] = {}
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one({"_id": oid}, {"$set": payload})
            return res.matched_count > 0

    def _origin_after(self, oid: ObjectId, payload: dict) -> str:
        """Who decides this worker's connection, after this edit.

        ``manual`` while any override stands. Back to ``discovered`` once the
        last one is gone — but only for a worker Docker actually found, which
        is the one case where there is something to inherit back to.
        """
        with self.connection.connect() as db:
            current = db[RESOURCES_COLLECTION].find_one(
                {"_id": oid}, {"container.id": 1, "connection": 1, "endpoint": 1})
        current = current or {}
        if not (current.get("container") or {}).get("id"):
            # Registered by hand: Docker never described it, so there is no
            # inherited configuration to fall back to.
            return "manual"

        # Only *this* edit's endpoint counts as a statement of intent. A
        # discovered worker always has one — discovery writes it — so reading
        # the stored endpoint as ownership kept every edited worker manual for
        # ever, which is the one-way door this is meant to remove.
        if "endpoint" in payload:
            return "manual"

        connection = payload.get("connection", current.get("connection") or {})
        if any(key != "transport" for key in connection):
            return "manual"
        return "discovered"

    def clear_connection_port(self, resource_id: Any) -> bool:
        """Remove a worker's port override, leaving its endpoint in charge.

        Separate from :meth:`update_manual` because an override is removed by
        the *absence* of the key, and ``$set`` cannot express absence. A
        stored ``null`` is not the same thing: it reads as the operator having
        configured something, which is one of the signals that decides whether
        Docker's view of this worker applies at all.

        Exists for the documents an older version of the form created, which
        wrote the port in both places. The override wins in the connection
        plan, so editing such a worker changed the endpoint, reported the new
        port and kept connecting to the old one (R17).
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].update_one(
                {"_id": oid},
                {"$unset": {"connection.port": ""},
                 "$set": {"updated_time": utc_now()}},
            )
            return res.modified_count > 0

    def delete_if_free(self, resource_id: Any) -> bool:
        """Remove one worker, and **only** if nothing is holding it.

        The condition is in the write, not in a check before it. Asking first
        and deleting after leaves a window the scheduler can reserve in: the
        API saw a free container, the dispatcher took it, and the delete then
        removed the record of a live run — with no way to find it again,
        because the reservation went with it.

        False means it was occupied, already gone, or never existed. The
        caller cannot tell those apart from here and does not need to: in
        every case there is nothing it may delete.
        """
        oid = _oid(resource_id)
        if oid is None:
            return False
        with self.connection.connect() as db:
            res = db[RESOURCES_COLLECTION].delete_one({
                "_id": oid,
                "status": {"$nin": _OCCUPIED},
                "current_simulation_id": None,
                "held_for": None,
            })
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
        try:
            with self.connection.transaction() as (db, session):
                self._claim_host_for_creation(
                    db, resource.get("host_id"), session)
                result = db[RESOURCES_COLLECTION].update_one(
                    {"host_id": resource["host_id"], "name": resource["name"]},
                    {"$setOnInsert": document},
                    upsert=True,
                    session=session,
                )
                return result.upserted_id is not None
        except (HostWithdrawing, OperationFailure) as error:
            # The one creation path that must not raise. It runs at every
            # start-up, before anything else: a withdrawal left set by an
            # interrupted removal — or a conflict with one in progress —
            # would stop the master-node from starting at all, which is a
            # worse failure than the orphan this guards against.
            log.warning(
                "Not seeding %r on host %s: %s",
                resource.get("name"), resource.get("host_id"), error,
            )
            return False

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
        now = utc_now()
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


class MasterStatusRepository:
    """What the master-node says about itself, in one document.

    Its liveness used to be inferred from the hosts' ``last_check``, which
    stops moving when every host is disabled or backing off — a healthy
    master then looked dead. And a resolver that stopped answering belongs to
    no host at all. Both are written here, once per health cycle.
    """

    DOC_ID = "master-node"
    #: Restarts kept for display; the loop guard only needs the last one.
    RESTARTS_KEPT = 10

    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def get(self) -> Optional[dict[str, Any]]:
        with self.connection.connect() as db:
            return db[MASTER_STATUS_COLLECTION].find_one({"_id": self.DOC_ID})

    def heartbeat(
        self, *, resolver: Optional[dict[str, Any]] = None,
        cycle_sec: Optional[float] = None,
    ) -> None:
        now = utc_now()
        fields: dict[str, Any] = {"heartbeat_at": now}
        if resolver is not None:
            fields["resolver"] = resolver
        if cycle_sec is not None:
            fields["cycle_sec"] = round(float(cycle_sec), 3)
        with self.connection.connect() as db:
            db[MASTER_STATUS_COLLECTION].update_one(
                {"_id": self.DOC_ID},
                {"$set": fields, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )

    def record_restart(self, reason: str) -> datetime:
        """Write down a restart the master-node is about to do to itself.

        Written *before* exiting: it is what keeps the next process from
        restarting again for the same reason straight away.
        """
        now = utc_now()
        entry = {"at": now, "reason": reason}
        with self.connection.connect() as db:
            db[MASTER_STATUS_COLLECTION].update_one(
                {"_id": self.DOC_ID},
                {
                    "$set": {"last_restart": entry},
                    "$push": {"restarts": {
                        "$each": [entry], "$slice": -self.RESTARTS_KEPT}},
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        return now


def host_access_secret(host: Host) -> Optional[SecretEnvelope]:
    """The hop-1 credential envelope, or ``None`` for a local host."""
    access = host.get("access") or {}
    return access.get("secret")


def container_access_secret(host: Host) -> Optional[SecretEnvelope]:
    """The hop-2 credential envelope."""
    return (host.get("container_access") or {}).get("secret")
