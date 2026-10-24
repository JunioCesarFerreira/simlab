"""Executing one placed simulation, and giving the container back.

Extracted from the worker loop so the dispatcher can hand a job to a thread
without that thread also deciding *where* the job goes. The split matters: the
decision has to be serialized to balance correctly, the execution must not be.

The release in ``finally`` is the contract with the dispatcher: a container is
returned exactly once, guarded by the attempt that holds it, and capacity is
signalled afterwards so a waiting decision wakes up.
"""
import logging
import shlex
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from bson import ObjectId

from lib.detached import RUN_ARTIFACTS, LiveRunPresent, assert_no_live_run
from lib.firmware_cache import FirmwareStateError
from lib.transport import TransportError, build_transport
from pylib.db.models.infra import operator_configured
from pylib.timeutil import utc_now

log = logging.getLogger(__name__)

# How many times a *failed* publish is retried before the result is handed to
# the collection-retry path. Bounded, and the backoff is linear: the write is
# one document and the failure it covers is a momentary one.
PUBLISH_ATTEMPTS = 3
PUBLISH_BACKOFF_SEC = 1.0

# What an unfetched result is renamed to, so the next run on this worker does
# not delete it. ``build_cleanup_command`` removes a fixed set of names plus
# the inputs it was given, and this is in neither.
PRESERVED_PREFIX = "simlab-unfetched"

# How often a reservation being prepared says "still working". Well under the
# reservation TTL (600s by default), and one small update per minute per
# preparation in flight.
#
# Renewing only *before* and *after* the upload was not enough, and the
# distinction matters: those two calls prove the reservation was ours at two
# instants, while the sweep acts on what is true in between. A long upload sat
# in that gap with no heartbeat at all.
#
# Derived from the reservation TTL by the entrypoint; this default matches a
# TTL of 600s at a quarter of it.
LEASE_HEARTBEAT_SEC = 60.0

# How long a preparation may go on being renewed. A heartbeat that renews for
# ever turns "the worker is not lost to a slow upload" into "the worker is
# never lost", and those are different claims: a transfer that stops making
# progress would hold its container until the simulation timeout, with the
# reservation looking healthy the whole time.
#
# Generously above anything real — the slowest preparation measured in a
# pilot was 22s — and well below SIM_TIMEOUT_SEC, so an abandoned preparation
# resolves in minutes rather than hours.
PREPARE_BUDGET_SEC = 1800.0


@dataclass
class RunReport:
    """What one execution produced, for the runner to publish.

    The executor used to write the simulation's state itself, and so did the
    runner — two layers translating the same outcome into database writes.
    With both doing it the inner one won, which is how the quarantine and the
    re-queue came to be unreachable: the executor's own ``except Exception``
    swallowed the exceptions that trigger them, marked the simulation in error
    and returned normally.

    So the executor now reports and the runner decides. Everything about who
    owns the simulation, what its status becomes and whether the container may
    be cleaned lives in one place.
    """
    log_id: Any = None
    csv_id: Any = None
    metrics: dict = field(default_factory=dict)
    dodag: Optional[dict] = None
    #: Whether the run's own completion was confirmed — an exit code was
    #: recorded and the CSV carried metrics. ``False`` means the artifacts
    #: exist and were kept, but nothing may read them as a result: a process
    #: reaped without recording its status, or a testlog that converted to an
    #: empty CSV. Published as a *failure with artifacts* rather than as a
    #: result, because the alternative is feeding the optimiser a mean over a
    #: different number of replicas without saying so.
    confirmed: bool = True
    #: Why it is unconfirmed, for the simulation's ``system_message``.
    reason: str = ""


class _PrepareFailed(Exception):
    """Internal: the inputs could not be assembled, so nothing was started."""


class RunFailed(Exception):
    """The run did not produce a usable result, and nothing is left running.

    An ordinary failure: the simulation is marked in error and the container
    goes back to the pool. Distinct from ``TerminationUnconfirmed``, where
    something may still be executing, and from ``LostOwnership``, where the
    work was fine but is no longer ours.
    """


class CollectionFailed(RunFailed):
    """The run finished, but its result could not be brought back.

    Kept apart from an ordinary failure because the work is not lost: the
    testlog is sitting on the container and the exit code was recorded. What
    failed is the fetching or the persisting, and that is worth trying again
    over a **new** session — the SCP retries inside the transport reopen the
    channel, not the connection, so a dropped session defeats them all three
    times.

    Carries the pid so the collection can be resumed exactly like a run
    adopted after a restart: attach, poll once, collect.

    ``stage`` names what failed, because the stages are not the same problem
    and were previously indistinguishable: ``download`` (nothing was stored
    yet, so the whole collection may be repeated), ``upload`` (GridFS refused),
    ``convert`` (the testlog is already stored — ``log_id`` says which blob, so
    a repeat removes it instead of leaving a second copy), ``evaluate`` (both
    blobs are stored and the metrics could not be computed) and ``publish``
    (everything is stored and only the state write failed).
    """

    def __init__(
        self,
        message: str,
        *,
        pid: Optional[int] = None,
        stage: str = "",
        log_id: Any = None,
        csv_id: Any = None,
        report: Any = None,
    ):
        super().__init__(message)
        self.pid = pid
        self.stage = stage
        self.log_id = log_id
        self.csv_id = csv_id
        #: The finished report, when the only thing that failed was recording
        #: it. Repeating *that* needs no container and no second upload.
        self.report = report


class ResultNotRecorded(Exception):
    """Artifacts were stored and the simulation does not reference them.

    The unconfirmed-result path stores the testlog in GridFS and then writes
    it onto the simulation. When that write failed the exception was caught,
    the method returned normally, and the caller went on to clear the remote
    artifacts and release the worker — so the blobs were orphaned *and* the
    evidence on the container was deleted, for a run whose result had been
    produced. No crash required: it was a path handled as a successful return.

    Raised instead, so the worker is held and nothing remote is cleaned.
    """

    def __init__(self, message: str, *, log_id: Any = None, csv_id: Any = None):
        super().__init__(message)
        self.log_id = log_id
        self.csv_id = csv_id


class TerminationUnconfirmed(Exception):
    """A run had to be stopped and the stop could not be confirmed.

    Raised instead of a plain failure because the two call for different
    things. A simulation that failed leaves a working container; a run that
    may still be alive leaves a container nobody should be given. The worker
    is taken out of the pool administratively — the sticky, operator-visible
    mechanism — rather than released, because a health probe cannot tell a
    leftover JVM from an idle container and would hand it straight back.

    ``started`` says whether *this* simulation ever ran. ``False`` is the case
    where something was found alive in the run directory **before** anything
    was sent: the worker is still unusable, but this simulation has no result
    to lose and no reason to be failed — it goes back to the queue and is
    placed somewhere else.
    """

    def __init__(self, message: str, *, started: bool = True):
        super().__init__(message)
        self.started = started


class LostOwnership(Exception):
    """This attempt no longer owns the container it is running on.

    Raised when a write guarded by ``attempt_id`` matches nothing: a recovery
    pass or the stale-reservation sweep handed the container to someone else
    while this thread was working. It is deliberately *not* a failure of the
    simulation — the work was sound — so the simulation goes back to the queue
    instead of being marked in error.

    The remote run is stopped on the way out. The pool already believes that
    container is free, so leaving a JVM on it would invite a second simulation
    onto the same worker, which is the one thing the reservation exists to
    prevent.

    ``stopped`` records whether that stop was *confirmed*. ``False`` is the
    dangerous combination and it used to be invisible: the executor called
    ``terminate()``, ignored its answer and raised this, and the runner
    re-queued the simulation without taking the worker out of the pool — so a
    container with a JVM still on it was handed to the next placement.
    """

    def __init__(self, message: str, *, stopped: Optional[bool] = None):
        super().__init__(message)
        self.stopped = stopped


class SimulationRunner:
    """Runs placed simulations. One instance, shared by every runner thread.

    ``execute`` is the function that actually drives Cooja — injected rather
    than imported so the orchestration here can be tested without a network,
    and so ``master-node.py`` keeps ownership of the execution sequence.
    """

    def __init__(
        self,
        mongo: Any,
        execute: Callable[..., RunReport],
        prepare: Callable[..., tuple[bool, list[str], list[str]]],
        *,
        opener: Any = None,
        on_release: Optional[Callable[[], None]] = None,
        cleanup_local: Optional[Callable[[list[str]], None]] = None,
        on_learn_host_key: Optional[Callable[[Any, str], None]] = None,
        lease_heartbeat_sec: float = LEASE_HEARTBEAT_SEC,
        prepare_budget_sec: float = PREPARE_BUDGET_SEC,
        firmware: Any = None,
    ):
        self._mongo = mongo
        self._execute = execute
        self._prepare = prepare
        self._opener = opener
        self._on_release = on_release
        self._cleanup_local = cleanup_local
        self._on_learn_host_key = on_learn_host_key
        # A ``FirmwareCache``, or ``None`` for the previous behaviour: every
        # run clears the build and sends everything.
        self._firmware = firmware
        # Only guarded against zero or negative, which would spin. The
        # operational floor belongs where the value is derived from the TTL,
        # not here — a test that wants millisecond beats is not a
        # misconfiguration.
        self._lease_heartbeat = max(0.001, lease_heartbeat_sec)
        self._prepare_budget = prepare_budget_sec
        # The attempts this process is following. Process-local on purpose:
        # it is the only honest answer to "is anyone watching this run?", and
        # nothing in the database can know it. The periodic reconciliation
        # asks before touching an occupied container, so a run that *is*
        # being followed is never reconciled underneath its follower.
        self._following: set[str] = set()
        self._following_lock = threading.Lock()

    def is_following(self, attempt_id: Optional[str]) -> bool:
        """Whether a thread of this process is watching that attempt."""
        if not attempt_id:
            return False
        with self._following_lock:
            return attempt_id in self._following

    def register_attempt(self, attempt_id: str) -> None:
        """Claim an attempt as this process's, before its thread exists.

        Called by the dispatcher the moment a reservation is granted, on the
        dispatcher's own thread. The registration used to happen inside
        ``_on_container``, which runs on the *runner* thread — so between the
        reservation being written and the thread reaching that line, the
        periodic reconciliation could read a container that was held, see that
        nobody was following it, and conclude it was the leftover of a dead
        process. A reservation seconds old was freed and its simulation failed
        while the thread that owned it was still starting up (R04).

        Registering here closes that window by construction: the attempt is
        known before the reservation can be observed by anything else.
        ``_on_container`` still registers, for the resume path, where there is
        no dispatcher.
        """
        if not attempt_id:
            return
        with self._following_lock:
            self._following.add(attempt_id)

    def _record_duration(self, resource: dict, seconds: float) -> None:
        """Feed one completed run into the worker's occupancy estimate.

        Advisory, and isolated for that reason: this number only chooses
        between equally loaded hosts, so losing it costs a slightly worse
        placement and nothing else. Letting it raise here would turn a failed
        statistics write into a failed simulation, which is not a trade worth
        making.
        """
        try:
            self._mongo.resource_repo.record_duration(resource["_id"], seconds)
        except Exception:
            log.warning(
                "Could not record the duration of %s (%.1fs); the estimate "
                "is stale, the result is not affected",
                resource.get("name"), seconds, exc_info=True,
            )

    def forget_attempt(self, attempt_id: str) -> None:
        """Drop a registration whose thread will never run.

        The counterpart of :meth:`register_attempt` for the dispatcher's own
        failure paths: a placement that registers and then cannot start its
        thread would otherwise leave an attempt permanently excluded from
        reconciliation — exactly the leak the registration exists to prevent.
        """
        if not attempt_id:
            return
        with self._following_lock:
            self._following.discard(attempt_id)

    def run(
        self,
        sim: dict,
        host: dict,
        resource: dict,
        attempt_id: str,
    ) -> None:
        """Execute one simulation on one reserved container.

        Every exit path releases the container. A simulation that fails is
        marked and its generation re-evaluated, exactly as before; what is new
        is that the container goes back to the pool either way.
        """
        sim_oid = _oid(sim["_id"])
        local_files: list[str] = []

        def body(transport) -> None:
            nonlocal local_files
            # Held for the whole of the slow part — the firmware out of
            # GridFS, then up a tunnel — rather than renewed at its two ends.
            # While this beats, the sweep cannot expire the reservation; a
            # refusal sets `lost` and nothing further is sent.
            with self._lease(
                    resource, sim_oid, attempt_id, abort=transport.close) as lost:
                success, local_files, remote_files = self._prepare(
                    sim, self._mongo)
                if not success:
                    raise _PrepareFailed()

                # Checked before anything is sent. The reservation was granted
                # a moment ago, but a recovery pass or the stale-reservation
                # sweep may have handed the container on since — and uploading
                # to a container that is someone else's now would overwrite the
                # inputs of a run in progress.
                self._require_ownership(resource, sim_oid, attempt_id)
                # A guarded write, before anything is sent: a refusal here is
                # definite, where the heartbeat only answers at its interval.
                self._renew_lease(resource, sim_oid, attempt_id)
                self._refuse_if_lost(lost, attempt_id)

                # Two attempts must never share this directory, and the next
                # line is a recursive delete. The reservation is not proof that
                # nothing is running in there, so the container is asked.
                try:
                    assert_no_live_run(transport)
                except LiveRunPresent as error:
                    # Nothing of ours has been sent, so this simulation has
                    # nothing to lose: the worker is unusable, the work is not.
                    raise TerminationUnconfirmed(
                        str(error), started=False) from error

                placement = self._place_firmware(
                    transport, local_files, remote_files)
                if placement is None:
                    # Defensive: clear artifacts left behind if a previous run
                    # crashed before its own cleanup could execute.
                    transport.cleanup()
                    transport.put_files(local_files, remote_files)
                else:
                    transport.put_files(
                        placement.local_files, placement.remote_files)
                self._refuse_if_lost(lost, attempt_id)
                # And a direct renewal, so the reservation is demonstrably
                # ours at the instant the run starts rather than one heartbeat
                # ago.
                self._renew_lease(resource, sim_oid, attempt_id)

            if not self._mongo.simulation_repo.mark_running(
                    sim_oid, attempt_id=attempt_id):
                raise LostOwnership(
                    "the simulation was taken over before this run started")

            report = self._execute(
                sim, transport, self._mongo, remote_files,
                on_started=self._started(resource["_id"], sim_oid, attempt_id),
                attempt_id=attempt_id,
            )
            self._publish(sim, sim_oid, attempt_id, report)
            # Kept only behind a confirmed result that was published: that is
            # the evidence the build in place is sound. Anything less clears
            # it as before, and the manifest left `pending` makes the next run
            # rebuild.
            if (placement is not None and getattr(report, "confirmed", True)
                    and self._firmware.keep(transport, placement)):
                return
            self._clear_remote(transport, remote_files)

        self._on_container(sim, host, resource, attempt_id, body,
                           cleanup=lambda: local_files, measure=True)

    def _place_firmware(
        self, transport: Any, local_files: list[str], remote_files: list[str]
    ) -> Any:
        """Reuse the firmware in place, or clear it completely.

        ``None`` without a ``FirmwareCache``. A state the cache could not
        reach is a failed run: nothing has been sent and nothing is running,
        and building on top of what the previous firmware left is exactly
        what must not happen.
        """
        if self._firmware is None:
            return None
        try:
            return self._firmware.place(transport, local_files, remote_files)
        except FirmwareStateError as error:
            raise RunFailed(str(error)) from error

    def resume(
        self,
        sim: dict,
        host: dict,
        resource: dict,
        attempt_id: str,
        pid: int,
    ) -> None:
        """Pick up a simulation that outlived a master-node restart.

        Nothing is prepared and nothing is uploaded: the inputs are already in
        the container and the JVM is already running. Only the following
        resumes, so the result is collected when it finishes instead of the
        run being abandoned to its timeout.
        """
        sim_oid = _oid(sim["_id"])

        def body(transport) -> None:
            report = self._execute(
                sim, transport, self._mongo, None,
                on_started=None, adopt_pid=int(pid), attempt_id=attempt_id,
            )
            self._publish(sim, sim_oid, attempt_id, report)
            self._clear_remote(transport, None)

        self._on_container(sim, host, resource, attempt_id, body)

    def _on_container(
        self,
        sim: dict,
        host: dict,
        resource: dict,
        attempt_id: str,
        body: Callable[[Any], None],
        cleanup: Optional[Callable[[], list[str]]] = None,
        measure: bool = False,
    ) -> None:
        """Open the container, run ``body``, and always give it back.

        Shared by starting and resuming so the release — the contract with the
        dispatcher — exists once and cannot drift between the two paths.

        ``measure`` times how long the container was held and feeds it to the
        worker's occupancy estimate, which a speed-aware scheduling policy
        reads. Off for ``resume``: adopting a JVM that was already running
        measures the remainder of someone else's run, and recording that as
        this worker's cost would make a container look fast for finishing
        work it did not start.
        """
        sim_oid = _oid(sim["_id"])
        resource_id = resource["_id"]
        where = f"{host.get('name')}/{resource.get('name')}"
        # Whether the container may go back to the pool on the way out. Set
        # *before* the quarantine is attempted, not from its result: the
        # release that followed a failed quarantine is what put a worker with
        # a possibly-live JVM back into the pool (R05.1). Once the stop is
        # unconfirmed, the ordinary release is forbidden regardless of whether
        # the protective write succeeded.
        skip_release = False

        self.register_attempt(attempt_id)
        try:
            transport = build_transport(
                host, resource, self._opener,
                on_learn_host_key=self._host_key_sink(host),
            )
            started_at = time.monotonic()
            with transport:
                body(transport)
            if measure:
                # Only here: a run that reached this line produced a result
                # and published it. Timing a failure that aborted in five
                # seconds would make the worst worker look like the best.
                self._record_duration(resource, time.monotonic() - started_at)
        except _PrepareFailed:
            log.warning("[%s] Skipping simulation %s (prepare failed)", where, sim_oid)
            self._fail(sim, sim_oid, "Failed to prepare simulation files", attempt_id)
        except ResultNotRecorded as e:
            # The artifacts exist and the simulation does not reference them.
            # Nothing remote may be cleaned and the worker may not be handed
            # on: both would destroy the only evidence of a run that produced
            # a result.
            log.error(
                "[%s] %s produced artifacts that could not be recorded: %s. "
                "Holding %s", where, sim_oid, e, resource.get("name"),
            )
            skip_release = True
            self._quarantine(
                resource, attempt_id, str(e),
                held_for=self._unresolved(
                    "collection_pending", sim_oid, attempt_id, resource))
        except CollectionFailed as e:
            # The run finished; bringing its result back did not. Tried again
            # over a new session, because the retries inside the transport
            # reopen the channel and not the connection.
            log.warning("[%s] %s: %s; retrying over a new session", where, sim_oid, e)
            if not self._retry_collection(sim, host, resource, attempt_id, e):
                # Exhausted *and* nothing was copied aside. The original
                # testlog is still under the name the next run deletes, so
                # the worker is held rather than released.
                skip_release = True
                self._quarantine(
                    resource, attempt_id,
                    f"result not collected and not preserved: {e}",
                    held_for=self._unresolved(
                        "collection_pending", sim_oid, attempt_id, resource,
                        remote_pid=e.pid))
        except RunFailed as e:
            # The run produced nothing usable and nothing is left executing.
            log.warning("[%s] %s did not complete: %s", where, sim_oid, e)
            self._fail(sim, sim_oid, str(e), attempt_id)
            self._clear_remote_after_failure(resource, attempt_id)
        except TerminationUnconfirmed as e:
            log.error(
                "[%s] Could not confirm that the run was stopped: %s. "
                "Disabling %s so nothing else is placed on it until someone "
                "has looked", where, e, resource.get("name"),
            )
            skip_release = True
            if e.started:
                self._fail(sim, sim_oid, str(e), attempt_id)
            else:
                # Found something alive before anything was sent. The worker
                # is out of the pool either way; the simulation is placed
                # again rather than written off.
                self._requeue(sim_oid, attempt_id, str(e))
            self._quarantine(
                resource, attempt_id, str(e),
                held_for=self._unresolved(
                    "termination_unknown", sim_oid, attempt_id, resource))
        except LostOwnership as e:
            # Not the simulation's fault, so it is not marked in error: it
            # goes back to the queue and is placed again. The container is
            # already someone else's, which is why the release below finds
            # nothing of ours to give back.
            log.warning(
                "[%s] Attempt %s lost %s: %s; returning it to the queue",
                where, attempt_id, sim_oid, e,
            )
            self._requeue(sim_oid, attempt_id, str(e))
            if e.stopped is False:
                # Ownership was lost *and* the run could not be stopped. The
                # pool already believes this container is free, so the only
                # thing that keeps a second simulation off a worker with a
                # live JVM is taking it out administratively.
                log.error(
                    "[%s] The run could not be stopped after losing the "
                    "container; disabling %s", where, resource.get("name"),
                )
                skip_release = True
                self._quarantine(
                    resource, None, f"ownership lost and the run survived: {e}",
                    held_for=self._unresolved(
                        "termination_unknown", sim_oid, attempt_id, resource))
        except TransportError as e:
            # Reaching the container failed. The simulation never started, so
            # it is a plain failure — distinct from losing contact with a run
            # already in progress, which the follower handles by reconnecting.
            log.warning("[%s] Could not reach the container for %s: %s", where, sim_oid, e)
            self._fail(sim, sim_oid, f"Could not reach {where}: {e}", attempt_id)
        except Exception as e:
            log.exception("[%s] General ERROR on %s: %s", where, sim_oid, e)
            self._fail(sim, sim_oid, str(e), attempt_id)
        finally:
            if self._cleanup_local is not None and cleanup is not None:
                self._cleanup_local(cleanup())
            try:
                if skip_release:
                    log.info(
                        "[%s] Not releasing: the stop was not confirmed, so "
                        "this worker stays out of the pool", where)
                else:
                    self._release(
                        resource_id, sim_oid, attempt_id, where,
                        # Where Docker's view of this worker does not apply, a
                        # stale `missing` must not take it out of the pool on
                        # the way back.
                        container_state_counts=not operator_configured(
                            host, resource),
                        config_version=int(host.get("config_version") or 0),
                    )
            finally:
                with self._following_lock:
                    self._following.discard(attempt_id)
                if self._on_release is not None:
                    self._on_release()

    def _started(
        self, resource_id: Any, sim_oid: ObjectId, attempt_id: str
    ) -> Callable[[int], None]:
        """Record the moment the simulation actually began running.

        ``reserved`` becomes ``busy`` here rather than before the execution,
        so the state means what it says — a container is busy once something
        is running on it — and so the transition carries the remote PID in a
        single atomic write.

        The PID is what lets a later liveness probe ask whether the run is
        still going instead of assuming a dropped connection killed it.
        """

        def on_started(pid: int) -> None:
            try:
                promoted = self._mongo.resource_repo.mark_busy(
                    resource_id, sim_oid, attempt_id, pid)
            except Exception:
                # The write itself failed — Mongo blinked. Bookkeeping only:
                # the simulation is already running, and failing it over a
                # lost write would waste the whole run.
                log.exception(
                    "Failed to record pid %s for simulation %s", pid, sim_oid)
                return

            if not promoted:
                # Not a lost write: a *refused* one. The reservation this
                # attempt held is gone, so the pool already considers the
                # container free and the run has to stop.
                raise LostOwnership(
                    f"the reservation for attempt {attempt_id} is gone")

            try:
                self._mongo.simulation_repo.set_remote_pid(sim_oid, attempt_id, pid)
            except Exception:
                log.exception(
                    "Failed to record pid %s for simulation %s", pid, sim_oid)

        return on_started

    # ── helpers ────────────────────────────────────────────────────────────

    def _require_ownership(
        self, resource: dict, sim_oid: ObjectId, attempt_id: str
    ) -> None:
        """Refuse to touch the container unless this attempt still holds it."""
        try:
            held = self._mongo.resource_repo.is_held_by(
                resource["_id"], sim_oid, attempt_id)
        except Exception as error:
            # Unlike a failed PID write after launch, this is permission for
            # destructive cleanup and upload. Unknown ownership cannot grant it.
            raise LostOwnership(
                f"could not confirm the reservation for attempt {attempt_id}") from error
        if not held:
            raise LostOwnership(
                f"the reservation for attempt {attempt_id} is gone")

    @staticmethod
    def _refuse_if_lost(lost: threading.Event, attempt_id: str) -> None:
        """Stop before the next remote effect if the lease ended.

        The heartbeat also closes the transport to interrupt an in-flight SCP
        transfer. This check covers the intervals between remote operations.
        """
        if lost.is_set():
            raise LostOwnership(
                f"the reservation for attempt {attempt_id} ended while the "
                "simulation was being prepared")

    @contextmanager
    def _lease(self, resource: dict, sim_oid: ObjectId, attempt_id: str,
               *, abort: Callable[[], None]):
        """Keep the reservation alive for as long as this block runs.

        Yields an event that is set the moment a renewal is **refused** —
        which is a definite statement that the reservation is gone, because
        the write is guarded by the attempt.

        A thread rather than two calls around the slow part. Renewing before
        and after an upload proves the reservation was ours at two instants;
        the sweep acts on what is true in between, and a transfer longer than
        the TTL sat in that gap. While this thread lives the sweep's
        ``reserved_before`` condition cannot match, so the expiry does not
        happen rather than being noticed late.
        """
        stop = threading.Event()
        lost = threading.Event()
        abort_lock = threading.Lock()
        deadline = time.monotonic() + self._prepare_budget

        def interrupt() -> None:
            # No delayed heartbeat may close the connection after this lease
            # context has ended and the executor has begun using it.
            with abort_lock:
                if stop.is_set() or lost.is_set():
                    return
                lost.set()
                try:
                    abort()
                except Exception:
                    log.exception("Could not interrupt preparation of %s", sim_oid)

        def beat() -> None:
            while not stop.wait(self._lease_heartbeat):
                if (not lost.is_set() and self._prepare_budget > 0
                        and time.monotonic() >= deadline):
                    log.error(
                        "Preparing %s on %s exceeded %.0fs; interrupting upload",
                        sim_oid, resource.get("name"), self._prepare_budget,
                    )
                    interrupt()
                    # Continue renewing until the body actually unwinds. A
                    # timer expiring is not proof that a writer has stopped.
                try:
                    renewed = self._mongo.resource_repo.touch_reservation(
                        resource["_id"], sim_oid, attempt_id)
                except Exception:
                    log.exception(
                        "Could not renew the lease on %s", resource.get("name"))
                    interrupt()
                    continue
                if not renewed:
                    log.warning(
                        "The reservation for attempt %s is gone; stopping the "
                        "lease", attempt_id)
                    interrupt()
                    return

        thread = threading.Thread(
            target=beat, daemon=True, name=f"lease-{attempt_id[:8]}")
        thread.start()
        try:
            yield lost
            self._refuse_if_lost(lost, attempt_id)
        except Exception:
            self._refuse_if_lost(lost, attempt_id)
            raise
        finally:
            with abort_lock:
                stop.set()

    def _renew_lease(
        self, resource: dict, sim_oid: ObjectId, attempt_id: str
    ) -> None:
        """Say "still preparing", and find out whether we may still say it.

        A guarded write, so it answers two questions at once: it pushes the
        reservation's TTL out past a slow upload, and a *refusal* is a
        definite statement that the reservation is gone. A failed write also
        interrupts preparation: without renewal the sweep may hand the worker
        to another attempt while an upload is still writing its inputs.
        """
        try:
            renewed = self._mongo.resource_repo.touch_reservation(
                resource["_id"], sim_oid, attempt_id)
        except Exception as error:
            raise LostOwnership(
                f"could not renew the reservation for attempt {attempt_id}") from error
        if not renewed:
            raise LostOwnership(
                f"the reservation for attempt {attempt_id} is gone")

    def _publish(
        self, sim: dict, sim_oid: ObjectId, attempt_id: str, report: Any
    ) -> None:
        """Record the result, if this attempt is still the one that may.

        The executor produced it; deciding what it means belongs here, with
        every other decision about the simulation's state.
        """
        if not getattr(report, "confirmed", True):
            self._publish_unconfirmed(sim, sim_oid, attempt_id, report)
            return

        published = self._mark_done_with_retry(sim_oid, attempt_id, report)
        if published is None:
            # The write kept failing. The blobs are in GridFS and the result
            # is sound, so this is a *publish* failure and is retried as one —
            # with the same ids, so repeating it cannot duplicate a blob.
            raise CollectionFailed(
                "the result could not be recorded in MongoDB",
                stage="publish", report=report,
                log_id=getattr(report, "log_id", None),
                csv_id=getattr(report, "csv_id", None),
            )
        if not published:
            # The result is sound but it is not ours to publish any more. The
            # blobs are already in GridFS, so they are removed rather than
            # left with nothing referencing them.
            self._discard_artifacts(report)
            raise LostOwnership(
                "the simulation was taken over while it was running")
        # From here the simulation **is** Done. Closing its generation is
        # separate, administrative bookkeeping over a different document, and
        # a failure of it used to propagate out of this method into the
        # generic handler — which marked the finished simulation in error
        # (R24). It is isolated, and the health cycle re-closes what is left.
        self._close_generation_safely(sim.get("generation_id"), sim_oid)

    def _mark_done_with_retry(
        self, sim_oid: ObjectId, attempt_id: str, report: Any
    ) -> Optional[bool]:
        """Publish, retrying a *failed* write but not a refused one.

        ``True`` published, ``False`` refused (someone else owns it, and no
        number of retries changes that), ``None`` the write itself never
        succeeded.
        """
        for attempt in range(1, PUBLISH_ATTEMPTS + 1):
            try:
                return bool(self._mongo.simulation_repo.mark_done(
                    sim_oid,
                    getattr(report, "log_id", None),
                    getattr(report, "csv_id", None),
                    getattr(report, "metrics", None) or {},
                    getattr(report, "dodag", None),
                    attempt_id=attempt_id,
                ))
            except Exception:
                log.exception(
                    "Recording the result of %s failed (attempt %d of %d)",
                    sim_oid, attempt, PUBLISH_ATTEMPTS,
                )
                if attempt < PUBLISH_ATTEMPTS:
                    time.sleep(PUBLISH_BACKOFF_SEC * attempt)
        return None

    def _publish_unconfirmed(
        self, sim: dict, sim_oid: ObjectId, attempt_id: str, report: Any
    ) -> None:
        """Keep the artifacts of a run whose completion was never confirmed.

        The policy, stated once: bytes are preserved, a result is not
        declared. A process reaped without recording an exit code, or a
        testlog that converted to an empty CSV, leaves real evidence — and
        publishing it as ``Done`` would put its metrics into the mean that
        becomes an objective, with nothing recording that this replica is
        different from the others. So the simulation is marked in error *with
        its artifacts attached*, and its metrics are stored under
        ``partial_metrics``, which the objective aggregation does not read.

        The replica then counts as missing, which is what makes the
        completeness of the evaluation above it true.
        """
        reason = getattr(report, "reason", "") or "the run was not confirmed complete"
        log.warning(
            "Simulation %s produced artifacts without a confirmed completion: "
            "%s. They are kept; the result is not published.", sim_oid, reason,
        )
        recorded: Optional[bool] = None
        for attempt in range(1, PUBLISH_ATTEMPTS + 1):
            try:
                recorded = self._mongo.simulation_repo.mark_error(
                    sim_oid, f"Unconfirmed result: {reason}",
                    attempt_id=attempt_id,
                    log_id=getattr(report, "log_id", None),
                    csv_id=getattr(report, "csv_id", None),
                    partial_metrics=getattr(report, "metrics", None) or {},
                )
                break
            except Exception:
                log.exception(
                    "Recording the unconfirmed result of %s failed "
                    "(attempt %d of %d)", sim_oid, attempt, PUBLISH_ATTEMPTS)
                if attempt < PUBLISH_ATTEMPTS:
                    time.sleep(PUBLISH_BACKOFF_SEC * attempt)
        if recorded is None:
            # The blobs are in GridFS and nothing points at them. Raised so
            # the caller holds the worker and leaves the container alone —
            # returning here let the remote evidence be cleared as well.
            raise ResultNotRecorded(
                f"the unconfirmed result of {sim_oid} could not be recorded",
                log_id=getattr(report, "log_id", None),
                csv_id=getattr(report, "csv_id", None),
            )
        if not recorded:
            self._discard_artifacts(report)
            raise LostOwnership(
                "the simulation was taken over while it was running")
        self._close_generation_safely(sim.get("generation_id"), sim_oid)

    def _close_generation_safely(
        self, generation_id: Any, sim_oid: ObjectId
    ) -> None:
        """Re-evaluate a generation, never at the cost of the simulation.

        Idempotent and recoverable by construction: ``close_if_finished``
        re-reads the generation's simulations every time, so repeating it
        decides the same thing, and the health cycle repeats it for any
        generation left open with nothing active.
        """
        if not generation_id:
            return
        try:
            self._close_generation(generation_id)
        except Exception:
            log.exception(
                "Failed to close generation %s after %s; it is left for the "
                "periodic pass to close", generation_id, sim_oid,
            )

    def _discard_artifacts(self, report: Any) -> None:
        for orphan in (getattr(report, "log_id", None),
                       getattr(report, "csv_id", None)):
            if orphan is None:
                continue
            try:
                self._mongo.fs_handler.delete_file(str(orphan))
            except Exception:
                log.exception("Could not remove orphaned file %s", orphan)

    def _retry_collection(
        self,
        sim: dict,
        host: dict,
        resource: dict,
        attempt_id: str,
        failure: "CollectionFailed",
    ) -> bool:
        """Try again to turn a finished run into a stored result.

        Returns whether the worker may go back to the pool. ``False`` means
        the result is neither stored nor copied aside, so the only evidence
        left is the original ``COOJA.testlog`` — under exactly the name the
        next run on this worker deletes.

        Which work to repeat depends on *where* it failed, and conflating the
        stages is how a Mongo failure after the upload came to re-download and
        re-upload the same bytes:

        ``publish``
            Nothing remote is needed at all. The blobs exist and the report is
            in hand; only the one document write is repeated, so a repeat
            cannot produce a second copy of anything.
        ``convert``
            The testlog is already in GridFS. That blob is removed first, and
            then the whole collection is repeated — otherwise the retry leaves
            two testlogs for one simulation.
        ``download``, ``upload``
            Nothing was stored. The collection is repeated over a **new**
            session, which is the point: the SCP retries inside the transport
            reopen the channel, not the connection, so a dropped session
            defeats all three of them.

        Before any of that, the artifacts are copied aside under a name the
        next run's cleanup does not know. That is what makes a second failure
        survivable instead of final.
        """
        sim_oid = _oid(sim["_id"])

        if failure.stage == "publish" and failure.report is not None:
            log.info(
                "Only the record of %s failed; repeating the write without "
                "fetching anything again", sim_oid,
            )
            try:
                self._publish(sim, sim_oid, attempt_id, failure.report)
            except (LostOwnership, ResultNotRecorded, TerminationUnconfirmed):
                # Control outcomes, not failures of this retry. Swallowing
                # them here is what kept the quarantine of the main flow from
                # applying to a second attempt.
                raise
            except CollectionFailed as again:
                log.error(
                    "The result of %s is in GridFS but could not be recorded: "
                    "%s", sim_oid, again,
                )
                self._record_unfetched(sim_oid, attempt_id, host, resource, failure)
                self._fail(
                    sim, sim_oid, f"{failure}", attempt_id,
                    log_id=failure.log_id, csv_id=failure.csv_id,
                )
                # The blobs exist and are referenced; nothing is on the
                # container that the next run would destroy.
                return True
            return True

        if not failure.pid:
            log.warning(
                "No pid recorded for %s; the result cannot be fetched again",
                sim_oid,
            )
            self._record_unfetched(sim_oid, attempt_id, host, resource, failure)
            self._fail(sim, sim_oid, str(failure), attempt_id)
            # Nothing to attach to, so nothing was copied aside either.
            return False

        if failure.log_id is not None or failure.csv_id is not None:
            # Stored by the attempt that is about to store them again. Without
            # this a repeat leaves two copies of the same testlog for one
            # simulation.
            self._discard_artifacts(failure)

        preserved: Optional[str] = None
        try:
            transport = build_transport(
                host, resource, self._opener,
                on_learn_host_key=self._host_key_sink(host),
            )
            with transport:
                preserved = self._preserve_remote(transport, attempt_id)
                report = self._execute(
                    sim, transport, self._mongo, None,
                    on_started=None, adopt_pid=int(failure.pid),
                    attempt_id=attempt_id,
                )
                self._publish(sim, sim_oid, attempt_id, report)
                self._clear_remote(transport, [preserved] if preserved else None)
        except (LostOwnership, ResultNotRecorded, TerminationUnconfirmed):
            raise
        except Exception as error:
            log.warning(
                "[%s/%s] The result of %s could not be fetched again: %s",
                host.get("name"), resource.get("name"), sim_oid, error,
            )
            self._record_unfetched(
                sim_oid, attempt_id, host, resource, failure, preserved)
            self._fail(sim, sim_oid, f"{failure} (retry: {error})", attempt_id)
            # Only safe to release if the bytes were copied somewhere the next
            # run will not delete. When the second session never opened, they
            # were not: `preserved` is None and the worker is held instead.
            return preserved is not None
        log.info("The result of %s was fetched on the second attempt", sim_oid)
        return True

    def _preserve_remote(self, transport: Any, attempt_id: str) -> Optional[str]:
        """Copy an unfetched testlog to a name the next run will not delete.

        The exhaustion policy used to be "leave the files alone", which is not
        preservation: the artifacts stay under ``COOJA.testlog``, and the
        defensive cleanup at the start of the *next* run on that worker
        removes exactly that name. The window was one placement wide.

        Copying is cheap, happens on the container, and the new name is
        outside the cleanup's fixed list — so the bytes survive until somebody
        decides what to do with them.
        """
        name = f"{PRESERVED_PREFIX}-{attempt_id}.testlog"
        command = (
            f"cd {shlex.quote(_remote_dir_of(transport))} 2>/dev/null; "
            f"[ -s COOJA.testlog ] && cp -f COOJA.testlog {shlex.quote(name)} "
            f"&& echo kept"
        )
        try:
            result = transport.exec_command(command, timeout=120)
        except Exception as error:
            log.warning("Could not preserve the result remotely: %s", error)
            return None
        if "kept" not in (result.stdout or ""):
            return None
        log.info("Kept the unfetched testlog as %s", name)
        return name

    def _record_unfetched(
        self,
        sim_oid: ObjectId,
        attempt_id: str,
        host: dict,
        resource: dict,
        failure: "CollectionFailed",
        preserved: Optional[str] = None,
    ) -> None:
        """Write down where an uncollected result is, and what it was.

        Without this the exhaustion is unrecoverable in practice even when the
        bytes survive: nothing records which worker, which directory or which
        process they belonged to, so there is nothing for a person or a later
        pass to act on.
        """
        try:
            self._mongo.simulation_repo.update(sim_oid, {"unfetched_result": {
                "host_id": host.get("_id"),
                "host_name": host.get("name"),
                "resource_id": resource.get("_id"),
                "resource_name": resource.get("name"),
                "container_id": (resource.get("container") or {}).get("id"),
                "attempt_id": attempt_id,
                "remote_pid": failure.pid,
                # The directory, not just the file name: the resource
                # registration is mutable, so a location that has to be
                # re-derived from it later is not a location.
                "remote_dir": resource.get("remote_dir"),
                "remote_file": preserved,
                "stage": failure.stage,
                "reason": str(failure),
                "recorded_at": utc_now(),
            }})
        except Exception:
            log.exception(
                "Could not record where the unfetched result of %s is", sim_oid)

    def _clear_remote_after_failure(self, resource: dict, attempt_id: str) -> None:
        """A failed run leaves artifacts; they are the next run's to clear.

        Deliberately a no-op. A run that failed on its own terms produced
        nothing worth keeping, and the defensive cleanup at the start of the
        next run on that worker removes what is there.

        A run that *succeeded* and could not be collected is a different case
        and is not handled here: :meth:`_preserve_remote` copies its testlog
        to a name the cleanup does not know, and
        :meth:`_record_unfetched` writes down where it is.
        """

    def _clear_remote(self, transport: Any, remote_files: Optional[list]) -> None:
        """Clear the run's remote artifacts, now that nothing needs them.

        Reached only after the result is published. The old code did this in a
        ``finally``, which meant it also ran when the run could not be stopped
        — deleting ``run.pgid`` and ``run.pid``, the only handles on a process
        that might still have been executing.
        """
        try:
            transport.cleanup(list(remote_files or []) + list(RUN_ARTIFACTS))
        except Exception:
            log.exception("Remote cleanup failed")

    def _unresolved(
        self,
        kind: str,
        sim_oid: ObjectId,
        attempt_id: Optional[str],
        resource: dict,
        *,
        remote_pid: Any = None,
        remote_file: Optional[str] = None,
    ) -> dict:
        """What is left unaccounted for on a worker, in one record.

        Enough for a later pass to *ask the container* rather than infer:
        which process, which directory, which attempt. The directory is
        included because the resource registration is mutable — an operator
        can change `remote_dir` — and a location that has to be re-derived
        from a document that may have changed is not a location.
        """
        return {
            "kind": kind,
            "simulation_id": sim_oid,
            "attempt_id": attempt_id,
            "remote_pid": remote_pid if remote_pid is not None
            else resource.get("remote_pid"),
            "remote_dir": resource.get("remote_dir"),
            "remote_file": remote_file,
        }

    def _quarantine(
        self,
        resource: dict,
        attempt_id: Optional[str],
        reason: str,
        *,
        held_for: Optional[dict] = None,
    ) -> None:
        """Take a worker out of the pool until a person says otherwise.

        ``enabled = False`` is the right lever: it is administrative, so a
        successful probe cannot undo it, and the page already explains a
        disabled worker as draining. The alternative — releasing it — gives
        the next simulation a container that may still be running the last
        one.

        One write, not two. It used to be ``set_enabled(False)`` followed by
        the ordinary release in ``finally``, and when the first failed the
        second still ran: the protective step failing was followed by the
        unprotective one succeeding, and the worker came back as
        ``available``. Now a single update disables it *and* stops it holding
        the attempt, and the caller skips the release either way — so a
        failure here leaves the worker holding its attempt, which is also
        ineligible.

        ``held_for`` is what makes the protection outlive this process. Being
        disabled kept the dispatcher off the worker but not the
        reconciliation, which found a terminal simulation and freed the worker
        with no probe — so a quarantine whose write had failed was undone by
        the next health cycle, exactly when the database came back. With the
        field set, ``release`` and ``force_release`` refuse, and only a pass
        that has *asked the container* clears it.
        """
        try:
            if self._mongo.resource_repo.quarantine(
                resource["_id"], reason, expect_attempt_id=attempt_id,
                held_for=held_for,
            ):
                return
            log.error(
                "Quarantine of %s matched nothing — it is held by another "
                "attempt now; leaving it to that attempt", resource.get("name"),
            )
        except Exception:
            log.exception(
                "Could not disable %s after an unconfirmed stop; trying to "
                "record the hold on its own", resource.get("name"),
            )
        # Either the guarded write was refused or it raised. Both leave the
        # worker without the one field that stops a later release, so the hold
        # is attempted unconditionally. If this fails too the worker keeps the
        # attempt — ineligible, but only until something reconciles it, which
        # is the residual stated in the plan.
        if held_for is not None:
            try:
                self._mongo.resource_repo.hold(
                    resource["_id"], reason, held_for)
            except Exception:
                log.exception(
                    "Could not record the unresolved hold on %s",
                    resource.get("name"),
                )

    def _requeue(self, sim_oid: ObjectId, attempt_id: str, reason: str) -> None:
        """Hand the simulation back to the queue, keeping no result.

        If the guard refuses this too, a newer attempt already owns the
        simulation and is doing the work — which is the outcome we wanted
        anyway, so it is logged and left alone.
        """
        try:
            requeued = self._mongo.simulation_repo.requeue(
                sim_oid, attempt_id, f"Re-queued: {reason}")
        except Exception:
            log.exception("Failed to re-queue simulation %s", sim_oid)
            return
        if not requeued:
            log.info(
                "Simulation %s is already owned by a newer attempt; "
                "leaving it alone", sim_oid,
            )

    def _release(
        self,
        resource_id: Any,
        sim_oid: ObjectId,
        attempt_id: str,
        where: str,
        *,
        container_state_counts: bool,
        config_version: int = 0,
    ) -> None:
        try:
            released = self._mongo.resource_repo.release(
                resource_id, sim_oid, attempt_id,
                container_state_counts=container_state_counts,
                config_version=config_version)
            if released is None:
                # Someone else already took it over — a recovery path decided
                # this attempt was gone. Releasing again would free a container
                # that is now running something else.
                log.info(
                    "[%s] Resource no longer held by attempt %s; nothing released",
                    where, attempt_id,
                )
            else:
                log.info("[%s] Released, now %s", where, released.get("status"))
        except Exception:
            log.exception("[%s] Failed to release resource %s", where, resource_id)

    def _fail(
        self,
        sim: dict,
        sim_oid: ObjectId,
        message: str,
        attempt_id: Optional[str],
        *,
        log_id: Any = None,
        csv_id: Any = None,
    ) -> None:
        """Record this attempt's failure, if it is still the owner.

        Guarded like every other write about the simulation: a thread whose
        container was handed on must not mark a failure against the placement
        that took over, and the repository additionally refuses to write a
        failure over a result that was already published.

        ``log_id``/``csv_id`` attach artifacts that exist despite the failure,
        so a simulation that could not be collected still points at what was
        stored.
        """
        try:
            if not self._mongo.simulation_repo.mark_error(
                    sim_oid, message, attempt_id=attempt_id,
                    log_id=log_id, csv_id=csv_id):
                log.info(
                    "Simulation %s is owned by a newer attempt, or already "
                    "carries a published result; not marking this failure "
                    "against it", sim_oid,
                )
                return
        except Exception:
            log.exception("Failed to mark error on simulation %s", sim_oid)
        self._close_generation_safely(sim.get("generation_id"), sim_oid)

    def _close_generation(self, generation_id: ObjectId) -> None:
        self._mongo.generation_repo.close_if_finished(generation_id)

    def _host_key_sink(self, host: dict) -> Optional[Callable[[str], None]]:
        if self._on_learn_host_key is None:
            return None
        host_id = host.get("_id")
        return lambda fingerprint: self._on_learn_host_key(host_id, fingerprint)


def _oid(value: Any) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)


def _remote_dir_of(transport: Any) -> str:
    return getattr(getattr(transport, "plan", None), "remote_dir", ".")
