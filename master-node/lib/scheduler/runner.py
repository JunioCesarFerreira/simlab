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

log = logging.getLogger(__name__)


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
        resource_id = resource["_id"]
        where = f"{host.get('name')}/{resource.get('name')}"
        local_files: list[str] = []

        try:
            success, local_files, remote_files = self._prepare(sim, self._mongo)
            if not success:
                log.warning("[%s] Skipping simulation %s (prepare failed)", where, sim_oid)
                self._fail(sim, sim_oid, "Failed to prepare simulation files")
                return

            transport = build_transport(
                host, resource, self._opener,
                on_learn_host_key=self._host_key_sink(host),
            )
            with transport:
                # Defensive: clear artifacts left behind if a previous run
                # crashed before its own cleanup could execute.
                transport.cleanup()
                transport.put_files(local_files, remote_files)
                self._execute(
                    sim, transport, self._mongo, remote_files,
                    on_started=self._started(resource_id, sim_oid, attempt_id),
                )

        except TransportError as e:
            # Reaching the container failed. The simulation never started, so
            # it is a plain failure — distinct from losing contact with a run
            # already in progress, which the reconciliation layer handles.
            log.warning("[%s] Could not reach the container for %s: %s", where, sim_oid, e)
            self._fail(sim, sim_oid, f"Could not reach {where}: {e}")
        except Exception as e:
            log.exception("[%s] General ERROR on %s: %s", where, sim_oid, e)
            self._fail(sim, sim_oid, str(e))
        finally:
            if self._cleanup_local is not None:
                self._cleanup_local(local_files)
            self._release(resource_id, sim_oid, attempt_id, where)

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
                self._mongo.resource_repo.mark_busy(
                    resource_id, sim_oid, attempt_id, pid)
                self._mongo.simulation_repo.set_remote_pid(sim_oid, attempt_id, pid)
            except Exception:
                # Bookkeeping only: the simulation is already running, and
                # failing it over a lost write would waste the whole run.
                log.exception(
                    "Failed to record pid %s for simulation %s", pid, sim_oid)

        return on_started

    # ── helpers ────────────────────────────────────────────────────────────

    def _release(self, resource_id: Any, sim_oid: ObjectId, attempt_id: str, where: str) -> None:
        try:
            released = self._mongo.resource_repo.release(resource_id, sim_oid, attempt_id)
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

    def _fail(self, sim: dict, sim_oid: ObjectId, message: str) -> None:
        try:
            self._mongo.simulation_repo.mark_error(sim_oid, message)
        except Exception:
            log.exception("Failed to mark error on simulation %s", sim_oid)
        generation_id = sim.get("generation_id")
        if generation_id:
            try:
                self._close_generation(generation_id)
            except Exception:
                log.exception("Failed to close generation %s", generation_id)

    def _close_generation(self, generation_id: ObjectId) -> None:
        gr = self._mongo.generation_repo
        if gr.any_simulation_active(generation_id):
            return
        if gr.all_simulations_done(generation_id):
            gr.mark_done(generation_id)
        else:
            gr.mark_error(generation_id)

    def _host_key_sink(self, host: dict) -> Optional[Callable[[str], None]]:
        if self._on_learn_host_key is None:
            return None
        host_id = host.get("_id")
        return lambda fingerprint: self._on_learn_host_key(host_id, fingerprint)


def _oid(value: Any) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)
