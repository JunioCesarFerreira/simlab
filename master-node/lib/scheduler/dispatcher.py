"""The placement loop: one thread decides, many threads execute.

Why a single decision thread, when the previous design had N workers racing for
a shared queue: balancing by host requires knowing how many placements each
host already has, and a race cannot know that. With eight local containers and
two remote ones, eight local workers drained the queue before any remote host
saw work — not a policy that could be fixed by choosing differently, but a
consequence of letting the competition decide.

So the decision is serialized and the execution is not. Each decision reads the
inventory, picks a host by policy, claims a container atomically and hands the
job to its own thread. Since reservations are visible to the next snapshot, the
counts the policy sees are always current.

With no capacity, the job is **held** rather than failed or dropped, and the
loop waits to be woken: by a release, by a change to the inventory, or by a
periodic tick that makes a lost signal cost a few seconds instead of a stall.
"""
import logging
import queue
import threading
import uuid
from typing import Any, Callable, Optional

from bson import ObjectId

from lib.scheduler.inventory import describe_loads, snapshot_loads
from lib.scheduler.policy import Policy, build_policy
from pylib.db.models.enums import EnumStatus

log = logging.getLogger(__name__)

# How long a decision waits before looking again on its own. Not the primary
# wake-up — releases and inventory changes are — but the backstop that keeps a
# missed signal from stalling the queue indefinitely.
IDLE_TICK_SEC = 5.0


class Dispatcher:
    """Serializes placement; parallelizes execution."""

    def __init__(
        self,
        mongo: Any,
        runner: Any,
        *,
        policy: Optional[Policy] = None,
        intercept: Optional[Callable[[dict], bool]] = None,
        idle_tick_sec: float = IDLE_TICK_SEC,
        thread_factory: Optional[Callable[..., Any]] = None,
    ):
        self._mongo = mongo
        self._runner = runner
        self._policy = policy or build_policy()
        self._intercept = intercept
        self._idle_tick = idle_tick_sec
        self._thread_factory = thread_factory or _daemon_thread

        self._queue: queue.Queue = queue.Queue()
        self._capacity = threading.Condition()
        self._held: Optional[str] = None
        self._rr_cursor = 0
        self._stop = threading.Event()

    # ── inbound ────────────────────────────────────────────────────────────

    def submit(self, sim_id: str) -> None:
        """Offer a simulation for placement."""
        self._queue.put(str(sim_id))

    def notify_capacity(self) -> None:
        """Wake a waiting decision: a container was freed, or the pool changed."""
        with self._capacity:
            self._capacity.notify_all()

    def stop(self) -> None:
        self._stop.set()
        self.notify_capacity()

    @property
    def pending(self) -> int:
        return self._queue.qsize() + (1 if self._held is not None else 0)

    # ── the loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info(
            "Dispatcher started (policy=%s, idle tick=%ss)",
            getattr(self._policy, "name", type(self._policy).__name__), self._idle_tick,
        )
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:
                # The decision loop is the one thread that must not die: if it
                # does, nothing is ever placed again.
                log.exception("Dispatcher step failed; continuing")
                self._wait_for_capacity()

    def step(self) -> bool:
        """One iteration. Returns True when a simulation was placed.

        Separated from ``run`` so the loop's behaviour can be tested one
        decision at a time, without threads or timing.
        """
        sim_id = self._next_job()
        if sim_id is None:
            return False

        outcome = self._place(sim_id)
        if outcome is _NO_CAPACITY:
            # Held, not dropped: the next wake-up tries the same job again.
            self._held = sim_id
            self._wait_for_capacity()
            return False

        self._held = None
        return bool(outcome)

    def _next_job(self) -> Optional[str]:
        if self._held is not None:
            return self._held
        try:
            sim_id = self._queue.get(timeout=self._idle_tick)
        except queue.Empty:
            return None
        self._queue.task_done()
        return sim_id

    def _wait_for_capacity(self) -> None:
        with self._capacity:
            self._capacity.wait(timeout=self._idle_tick)

    # ── one placement ──────────────────────────────────────────────────────

    def _place(self, sim_id: str) -> Any:
        """Claim a container for ``sim_id`` and start it.

        Returns True when started, False when the simulation needs no placement
        (gone, already running, already placed), and ``_NO_CAPACITY`` when it
        should be held for later.
        """
        sim = self._mongo.simulation_repo.get(sim_id)
        if sim is None:
            log.warning("Simulation %s not found; dropping.", sim_id)
            return False

        sim_oid = _oid(sim["_id"])
        if not self._still_wants_placement(sim, sim_oid, sim_id):
            return False

        # Some simulations need no container at all — a synthetic run is
        # evaluated analytically. Completing them here keeps them from holding
        # capacity that work needing Cooja is waiting for.
        if self._intercept is not None and self._intercept(sim):
            log.info("Simulation %s handled without a container", sim_id)
            return False

        loads = snapshot_loads(self._mongo.host_repo)
        chosen = self._policy.select(loads, self._rr_cursor)
        if chosen is None:
            log.debug("No host can take %s right now: %s", sim_id, describe_loads(loads))
            return _NO_CAPACITY

        placement = self._claim(chosen, sim_oid)
        if placement is None:
            # The host looked free a moment ago and no longer is. Re-deciding
            # beats waiting: another host may still have room.
            log.info(
                "Host %s had no free container after all; re-deciding for %s",
                chosen.name, sim_id,
            )
            return _NO_CAPACITY

        host, resource, attempt_id = placement
        self._rr_cursor += 1
        log.info(
            "Placed %s on %s/%s (attempt %s) — %s",
            sim_id, host.get("name"), resource.get("name"), attempt_id,
            describe_loads(loads),
        )
        self._thread_factory(
            target=self._runner.run,
            args=(sim, host, resource, attempt_id),
            name=f"sim-{sim_id[:8]}",
        )
        return True

    def _still_wants_placement(self, sim: dict, sim_oid: ObjectId, sim_id: str) -> bool:
        if sim.get("status") != EnumStatus.WAITING:
            log.info(
                "Simulation %s is already in status %r; skipping.",
                sim_id, sim.get("status"),
            )
            return False

        # A change stream can redeliver the same document. Without this check a
        # redelivery would place a second attempt for a simulation that is
        # already holding a container.
        existing = self._mongo.resource_repo.find_by_simulation(sim_oid)
        if existing is not None:
            log.info(
                "Simulation %s already holds %s; not placing it again.",
                sim_id, existing.get("name"),
            )
            return False
        return True

    def _claim(self, chosen: Any, sim_oid: ObjectId) -> Optional[tuple[dict, dict, str]]:
        """Reserve a free container on the chosen host, atomically.

        Walks that host's free containers because another dispatcher — or the
        health cycle — may have taken one between the snapshot and now. Only a
        reservation that the database granted counts.
        """
        host = self._mongo.host_repo.get(chosen.host_id)
        if host is None:
            log.warning("Host %s vanished between snapshot and claim", chosen.host_id)
            return None

        for candidate in self._mongo.resource_repo.find_free_by_host(chosen.host_id):
            attempt_id = uuid.uuid4().hex
            reserved = self._mongo.resource_repo.reserve(
                candidate["_id"], sim_oid, attempt_id)
            if reserved is None:
                continue  # lost the race for this one; try the next

            if self._record_assignment(host, reserved, sim_oid, attempt_id):
                return host, reserved, attempt_id

            # The reservation stands but the simulation does not know about it,
            # which would strand the container. Give it back and stop.
            self._mongo.resource_repo.release(reserved["_id"], sim_oid, attempt_id)
            return None
        return None

    def _record_assignment(
        self, host: dict, resource: dict, sim_oid: ObjectId, attempt_id: str
    ) -> bool:
        """Persist where this simulation went, before anything runs.

        Order matters: written now, a master-node that restarts mid-simulation
        can find its way back to the container instead of guessing.
        """
        try:
            self._mongo.simulation_repo.set_assignment(
                sim_oid,
                host_id=host["_id"],
                host_name=str(host.get("name") or "?"),
                resource_id=resource["_id"],
                resource_name=str(resource.get("name") or "?"),
                attempt_id=attempt_id,
                container_id=(resource.get("container") or {}).get("id"),
            )
            return True
        except Exception:
            log.exception(
                "Failed to record the assignment of %s on %s", sim_oid, resource.get("name"))
            return False


class _NoCapacity:
    """Sentinel: hold the job and wait, rather than fail or drop it."""

    def __bool__(self) -> bool:  # pragma: no cover - never used as a truth value
        return False

    def __repr__(self) -> str:
        return "<no capacity>"


_NO_CAPACITY = _NoCapacity()


def _daemon_thread(target: Callable, args: tuple, name: str) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, name=name, daemon=True)
    thread.start()
    return thread


def _oid(value: Any) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)
