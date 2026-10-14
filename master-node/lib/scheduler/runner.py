"""Executing one placed simulation, and giving the container back.

Extracted from the worker loop so the dispatcher can hand a job to a thread
without that thread also deciding *where* the job goes. The split matters: the
decision has to be serialized to balance correctly, the execution must not be.

The release in ``finally`` is the contract with the dispatcher: a container is
returned exactly once, guarded by the attempt that holds it, and capacity is
signalled afterwards so a waiting decision wakes up.
"""
import logging
from typing import Any, Callable, Optional

from bson import ObjectId

from lib.transport import TransportError, build_transport
from pylib.db.models.infra import operator_configured

log = logging.getLogger(__name__)


class _PrepareFailed(Exception):
    """Internal: the inputs could not be assembled, so nothing was started."""


class TerminationUnconfirmed(Exception):
    """A run had to be stopped and the stop could not be confirmed.

    Raised instead of a plain failure because the two call for different
    things. A simulation that failed leaves a working container; a run that
    may still be alive leaves a container nobody should be given. The worker
    is taken out of the pool administratively — the sticky, operator-visible
    mechanism — rather than released, because a health probe cannot tell a
    leftover JVM from an idle container and would hand it straight back.
    """


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
    """


class SimulationRunner:
    """Runs placed simulations. One instance, shared by every runner thread.

    ``execute`` is the function that actually drives Cooja — injected rather
    than imported so the orchestration here can be tested without a network,
    and so ``master-node.py`` keeps ownership of the execution sequence.
    """

    def __init__(
        self,
        mongo: Any,
        execute: Callable[..., None],
        prepare: Callable[..., tuple[bool, list[str], list[str]]],
        *,
        opener: Any = None,
        on_release: Optional[Callable[[], None]] = None,
        cleanup_local: Optional[Callable[[list[str]], None]] = None,
        on_learn_host_key: Optional[Callable[[Any, str], None]] = None,
    ):
        self._mongo = mongo
        self._execute = execute
        self._prepare = prepare
        self._opener = opener
        self._on_release = on_release
        self._cleanup_local = cleanup_local
        self._on_learn_host_key = on_learn_host_key

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
            success, local_files, remote_files = self._prepare(sim, self._mongo)
            if not success:
                raise _PrepareFailed()
            # Defensive: clear artifacts left behind if a previous run crashed
            # before its own cleanup could execute.
            transport.cleanup()
            transport.put_files(local_files, remote_files)
            self._execute(
                sim, transport, self._mongo, remote_files,
                on_started=self._started(resource["_id"], sim_oid, attempt_id),
                attempt_id=attempt_id,
            )

        self._on_container(sim, host, resource, attempt_id, body,
                           cleanup=lambda: local_files)

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
            self._execute(
                sim, transport, self._mongo, None,
                on_started=None, adopt_pid=int(pid), attempt_id=attempt_id,
            )

        self._on_container(sim, host, resource, attempt_id, body)

    def _on_container(
        self,
        sim: dict,
        host: dict,
        resource: dict,
        attempt_id: str,
        body: Callable[[Any], None],
        cleanup: Optional[Callable[[], list[str]]] = None,
    ) -> None:
        """Open the container, run ``body``, and always give it back.

        Shared by starting and resuming so the release — the contract with the
        dispatcher — exists once and cannot drift between the two paths.
        """
        sim_oid = _oid(sim["_id"])
        resource_id = resource["_id"]
        where = f"{host.get('name')}/{resource.get('name')}"

        try:
            transport = build_transport(
                host, resource, self._opener,
                on_learn_host_key=self._host_key_sink(host),
            )
            with transport:
                body(transport)
        except _PrepareFailed:
            log.warning("[%s] Skipping simulation %s (prepare failed)", where, sim_oid)
            self._fail(sim, sim_oid, "Failed to prepare simulation files", attempt_id)
        except TerminationUnconfirmed as e:
            log.error(
                "[%s] Could not confirm that the run was stopped: %s. "
                "Disabling %s so nothing else is placed on it until someone "
                "has looked", where, e, resource.get("name"),
            )
            self._fail(sim, sim_oid, str(e), attempt_id)
            self._quarantine(resource, str(e))
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
            self._release(
                resource_id, sim_oid, attempt_id, where,
                # Where Docker's view of this worker does not apply, a stale
                # `missing` must not take it out of the pool on the way back.
                container_state_counts=not operator_configured(host, resource),
            )

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

    def _quarantine(self, resource: dict, reason: str) -> None:
        """Take a worker out of the pool until a person says otherwise.

        ``enabled = False`` is the right lever: it is administrative, so a
        successful probe cannot undo it, and the page already explains a
        disabled worker as draining. The alternative — releasing it — gives
        the next simulation a container that may still be running the last
        one.
        """
        try:
            self._mongo.resource_repo.set_enabled(resource["_id"], False)
        except Exception:
            log.exception(
                "Could not disable %s after an unconfirmed stop",
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
    ) -> None:
        try:
            released = self._mongo.resource_repo.release(
                resource_id, sim_oid, attempt_id,
                container_state_counts=container_state_counts)
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
        finally:
            if self._on_release is not None:
                self._on_release()

    def _fail(
        self, sim: dict, sim_oid: ObjectId, message: str, attempt_id: str
    ) -> None:
        """Record this attempt's failure, if it is still the owner.

        Guarded like every other write about the simulation: a thread whose
        container was handed on must not mark a failure against the placement
        that took over.
        """
        try:
            if not self._mongo.simulation_repo.mark_error(
                    sim_oid, message, attempt_id=attempt_id):
                log.info(
                    "Simulation %s is owned by a newer attempt; not marking "
                    "this failure against it", sim_oid,
                )
                return
        except Exception:
            log.exception("Failed to mark error on simulation %s", sim_oid)
        generation_id = sim.get("generation_id")
        if generation_id:
            try:
                self._close_generation(generation_id)
            except Exception:
                log.exception("Failed to close generation %s", generation_id)

    def _close_generation(self, generation_id: ObjectId) -> None:
        self._mongo.generation_repo.close_if_finished(generation_id)

    def _host_key_sink(self, host: dict) -> Optional[Callable[[str], None]]:
        if self._on_learn_host_key is None:
            return None
        host_id = host.get("_id")
        return lambda fingerprint: self._on_learn_host_key(host_id, fingerprint)


def _oid(value: Any) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)
