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
import time
import uuid
from typing import Any, Callable, Optional

from bson import ObjectId

from lib.scheduler.inventory import describe_loads, snapshot_loads
from lib.scheduler.policy import Policy, build_policy
from pylib.db.models.enums import EnumStatus
from pylib.db.models.infra import operator_configured

log = logging.getLogger(__name__)

# How long a decision waits before looking again on its own. Not the primary
# wake-up — releases and inventory changes are — but the backstop that keeps a
# missed signal from stalling the queue indefinitely.
IDLE_TICK_SEC = 5.0

# How often to repeat the "nothing can take this" warning. The loop retries
# every few seconds; saying so every time would bury everything else.
STARVATION_LOG_INTERVAL_SEC = 60.0

# How many times a placement may fail in a row before the job goes to the back
# of the queue. Keeping it forever would let one unplaceable simulation block
# everything behind it; dropping it is what this exists to prevent.
PLACEMENT_ATTEMPTS = 3

# How often to re-read the simulations still marked Waiting. The queue is fed
# by change streams, so a missed event leaves a simulation that nothing will
# ever look at again.
RESCAN_INTERVAL_SEC = 60.0


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
        rescan_interval_sec: float = RESCAN_INTERVAL_SEC,
        placement_attempts: int = PLACEMENT_ATTEMPTS,
        thread_factory: Optional[Callable[..., Any]] = None,
    ):
        self._mongo = mongo
        self._runner = runner
        self._policy = policy or build_policy()
        self._intercept = intercept
        self._idle_tick = idle_tick_sec
        self._rescan_interval = rescan_interval_sec
        self._placement_attempts = max(1, placement_attempts)
        self._thread_factory = thread_factory or _daemon_thread

        self._queue: queue.Queue = queue.Queue()
        self._capacity = threading.Condition()
        self._held: Optional[str] = None
        self._held_failures = 0
        self._rr_cursor = 0
        self._stop = threading.Event()
        self._last_starved_log = 0.0
        self._last_rescan = 0.0

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

    @property
    def runner(self):
        """The runner this dispatcher hands work to.

        Exposed so an adopted run can be resumed through the same object, and
        therefore released through the same contract.
        """
        return self._runner

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
            self._rescan_pending()
            return False

        # Owned from the moment it leaves the queue. ``_next_job`` has already
        # called ``task_done``, so until this is set nothing holds the id: a
        # momentary MongoDB failure below used to leave the simulation Waiting
        # for ever, with the queue empty and nothing to bring it back. The
        # capacity tick only retries work it already knows about.
        self._held = sim_id
        try:
            outcome = self._place(sim_id)
        except Exception:
            self._hold_after_failure(sim_id)
            self._wait_for_capacity()
            return False

        self._held_failures = 0
        if outcome is _NO_CAPACITY:
            # Held, not dropped: the next wake-up tries the same job again.
            self._wait_for_capacity()
            return False

        self._held = None
        return bool(outcome)

    def _hold_after_failure(self, sim_id: str) -> None:
        """Keep a job whose placement raised, without blocking the rest.

        Retried in place a few times, because the usual cause is momentary —
        the database, or a host that answered a second ago. After that it goes
        to the back of the queue: still ours, but no longer ahead of work that
        might succeed.
        """
        self._held_failures += 1
        log.exception(
            "Placing %s failed (attempt %d of %d); it is still queued",
            sim_id, self._held_failures, self._placement_attempts,
        )
        if self._held_failures < self._placement_attempts:
            return
        log.error(
            "Placing %s has failed %d times; moving it to the back of the "
            "queue so it stops holding up the others",
            sim_id, self._held_failures,
        )
        self._held = None
        self._held_failures = 0
        self._queue.put(sim_id)

    def _rescan_pending(self) -> None:
        """Re-offer simulations the queue never heard about.

        Only runs when there is nothing queued and nothing held, so it costs
        one read per interval while idle. It exists because the queue is fed
        by change streams: a missed event, or a job lost some other way,
        leaves a simulation Waiting with nothing ever looking at it again, and
        the experiment only moves on when someone restarts the master-node.
        The database is the truth; the stream is an optimisation.
        """
        now = time.monotonic()
        if now - self._last_rescan < self._rescan_interval:
            return
        self._last_rescan = now
        try:
            pending = self._mongo.simulation_repo.find_pending()
        except Exception:
            log.exception("Could not re-read the simulations still waiting")
            return
        if not pending:
            return
        log.info(
            "Re-offering %d simulation(s) still waiting that the queue had "
            "lost track of", len(pending),
        )
        for sim in pending:
            self._queue.put(str(sim["_id"]))

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
            self._log_starvation(sim_id, loads)
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
        try:
            self._thread_factory(
                target=self._runner.run,
                args=(sim, host, resource, attempt_id),
                name=f"sim-{sim_id[:8]}",
            )
        except Exception:
            # The thread that would have owned this attempt does not exist, so
            # the registration taken in `_claim` has to go: an attempt nobody
            # follows must be visible to the reconciliation, which is the
            # whole reason the registration is conditional on a thread.
            self._runner.forget_attempt(attempt_id)
            raise
        return True

    def _log_starvation(self, sim_id: str, loads: list) -> None:
        """Say why nothing is being placed, visibly but without flooding.

        A queue that fills while nothing runs is the symptom an operator sees;
        this is the line that explains it. At DEBUG it was invisible, and the
        only diagnosis left was reading the database by hand — so it is a
        warning, rate-limited because the loop retries every few seconds.
        """
        now = time.monotonic()
        if now - self._last_starved_log < STARVATION_LOG_INTERVAL_SEC:
            return
        self._last_starved_log = now
        if not loads:
            log.warning(
                "Holding %s: no hosts are registered. The local host is created "
                "at startup — check for an earlier error.", sim_id,
            )
            return
        log.warning(
            "Holding %s: no host can take it. %s "
            "(a host needs enabled=true, a schedulable status, and a free container)",
            sim_id, describe_loads(loads),
        )

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
        if host.get("withdrawn_at"):
            # Being removed. The snapshot the policy chose from may predate
            # the withdrawal, and placing work on a host whose workers are
            # being deleted strands the run.
            log.info(
                "Host %s is being removed; not placing work on it",
                host.get("name"))
            return None

        # The configuration the measurement has to belong to. A probe that
        # started before an access edit and published after it can leave a
        # worker `available` describing a path that no longer exists, and the
        # status filter alone cannot see that.
        config_version = int(host.get("config_version") or 0)

        # The two halves of one decision: a policy that picked the host
        # finishing soonest must not then be handed its slowest container.
        prefer_fastest = bool(getattr(self._policy, "prefers_fastest", False))
        for candidate in self._mongo.resource_repo.find_free_by_host(
                chosen.host_id, prefer_fastest=prefer_fastest):
            attempt_id = uuid.uuid4().hex
            reserved = self._mongo.resource_repo.reserve(
                candidate["_id"], sim_oid, attempt_id,
                expect_config_version=config_version)
            if reserved is None:
                continue  # lost the race for this one; try the next

            # Claimed as this process's *before* anything else can observe
            # the reservation. The registration used to be taken by the runner
            # thread, which starts later — and in that gap the periodic
            # reconciliation saw a held container that nobody was following,
            # read it as the leftover of a dead process, freed it and failed
            # its simulation (R04). Registering on this thread, between the
            # atomic reservation and everything that follows, removes the gap
            # rather than making it smaller.
            self._runner.register_attempt(attempt_id)

            if self._record_assignment(host, reserved, sim_oid, attempt_id):
                return host, reserved, attempt_id

            # The reservation stands but the simulation does not know about it,
            # which would strand the container. Give it back and stop.
            self._runner.forget_attempt(attempt_id)
            self._mongo.resource_repo.release(
                reserved["_id"], sim_oid, attempt_id,
                container_state_counts=not operator_configured(host, reserved),
                config_version=config_version)
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
            # The return value matters: it is false when the write matched no
            # document, which means the simulation is gone. Treating that as
            # recorded would start a run whose result has nowhere to land.
            recorded = self._mongo.simulation_repo.set_assignment(
                sim_oid,
                host_id=host["_id"],
                host_name=str(host.get("name") or "?"),
                resource_id=resource["_id"],
                resource_name=str(resource.get("name") or "?"),
                attempt_id=attempt_id,
                container_id=(resource.get("container") or {}).get("id"),
            )
            if not recorded:
                log.warning(
                    "Assignment of %s on %s matched no simulation; not starting it",
                    sim_oid, resource.get("name"),
                )
            return bool(recorded)
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
