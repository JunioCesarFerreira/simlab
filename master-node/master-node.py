import os
import sys
import time
import logging
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from typing import Callable
from bson import ObjectId

# SysPath for common modules
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# lib master-node
from lib.bootstrap import ensure_local_host, publish_public_key
from lib.credentials import SecretOpener
from lib.detached import (
    RUN_ARTIFACTS,
    DetachedRun,
    LaunchUnconfirmed,
    RunOutcome,
    inspect_run,
)
from lib.health import HealthMonitor
from lib.recovery import reconcile_occupied_resources
from lib.scheduler import (
    CollectionFailed,
    Dispatcher,
    LostOwnership,
    RunFailed,
    RunReport,
    SimulationRunner,
    TerminationUnconfirmed,
    build_policy,
)
from lib.synthetic_data import run_synthetic_simulation, resolve_synthetic_settings
from lib.transport import (
    HostKeyMismatch,
    SSHTransport,
    TransportError,
    build_transport,
)

# pylib crypto — master-node only; the API deliberately cannot open a secret.
from pylib.crypto.sealed import DEFAULT_KEY_PATH, Keystore
from pylib.timeutil import elapsed_since, utc_now

# pylib
from pylib.db import create_mongo_repository_factory, MongoRepository, EnumStatus
from pylib.db.models import Simulation, SourceRepository
from pylib.db.repositories.simulation import SimulationRepository
from pylib import cooja_files
from pylib import statistics
from pylib import rpl_dodag

# --------------------------- Logging --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [master-node] %(message)s",
)
log = logging.getLogger("master-node")
# --------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    is_docker: bool
    mongo_uri: str
    db_name: str
    local_dir: str
    remote_dir: str
    local_log_dir: str
    hostnames: list[str]
    ports: list[int]
    sim_timeout_sec: int
    jvm_xms: str
    jvm_xmx: str
    sched_policy: str
    infra_key_path: str
    run_poll_interval_sec: float
    infra_check_interval_sec: float
    infra_failure_threshold: int
    reservation_ttl_sec: float

    @staticmethod
    def from_env() -> "Settings":
        def to_bool(s: str, default: bool = False) -> bool:
            if s is None:
                return default
            return s.strip().lower() in {"1", "true", "yes", "y", "on"}

        is_docker = to_bool(os.getenv("IS_DOCKER", "false"))
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
        db_name = os.getenv("DB_NAME", "simlab")
        local_dir = "."
        remote_dir = "/opt/contiki-ng/tools/cooja"
        local_log_dir = "logs"
        Path(local_log_dir).mkdir(exist_ok=True)

        default_n = int(os.getenv("NUMBER_OF_CONTAINERS", "8"))
        if is_docker:
            hostnames = [f"cooja{i+1}" for i in range(default_n)]
            ports = [22 for _ in range(default_n)]
        else:
            hostnames = ["localhost" for _ in range(default_n)]
            ports = [2231 + i for i in range(default_n)]

        sim_timeout_sec = int(os.getenv("SIM_TIMEOUT_SEC", "3600"))
        jvm_xms = os.getenv("COOJA_JVM_XMS", "1g")
        jvm_xmx = os.getenv("COOJA_JVM_XMX", "2g")

        sched_policy = os.getenv("SCHED_POLICY", "least_loaded_host")
        infra_key_path = os.getenv("INFRA_KEY_PATH", DEFAULT_KEY_PATH)
        # How often a running simulation is checked on. One cheap round trip
        # each time (~40ms measured), so this is about log latency rather than
        # load; it also bounds how long a finished run holds its container.
        run_poll_interval_sec = float(os.getenv("RUN_POLL_INTERVAL_SEC", "5"))
        infra_check_interval_sec = float(os.getenv("INFRA_CHECK_INTERVAL_SEC", "30"))
        infra_failure_threshold = int(os.getenv("INFRA_FAILURE_THRESHOLD", "3"))
        reservation_ttl_sec = float(os.getenv("RESERVATION_TTL_SEC", "600"))

        return Settings(
            is_docker=is_docker,
            mongo_uri=mongo_uri,
            db_name=db_name,
            local_dir=local_dir,
            remote_dir=remote_dir,
            local_log_dir=local_log_dir,
            hostnames=hostnames,
            ports=ports,
            sim_timeout_sec=sim_timeout_sec,
            jvm_xms=jvm_xms,
            jvm_xmx=jvm_xmx,
            sched_policy=sched_policy,
            infra_key_path=infra_key_path,
            run_poll_interval_sec=run_poll_interval_sec,
            infra_check_interval_sec=infra_check_interval_sec,
            infra_failure_threshold=infra_failure_threshold,
            reservation_ttl_sec=reservation_ttl_sec,
        )


SET = Settings.from_env()


def _check_and_close_generation(generation_id: ObjectId, mongo: MongoRepository) -> None:
    """Close the generation if this was the last simulation still running.

    The decision itself lives in ``GenerationRepository.close_if_finished``,
    because four places used to carry their own copy of it and a fifth — the
    stale-reservation sweep — had none at all.
    """
    try:
        mongo.generation_repo.close_if_finished(generation_id)
    except Exception:
        log.exception("Failed to close generation %s", generation_id)


def recover_abandoned_simulations(mongo: MongoRepository) -> None:
    """Fail the RUNNING simulations that **no container holds**.

    Runs *after* ``reconcile_occupied_resources``, and that order is the whole
    point. The reconciliation asks each occupied container whether the run is
    still going, adopts it if so, and collects or fails it if not. Only what
    is left over reaches here: a simulation marked RUNNING with no reservation
    behind it, so there is nothing to ask and the clock is all the evidence
    there is.

    It used to run *first*, and purely by time. The reconciliation then found
    a simulation already in a terminal state and freed its container without
    probing — so a JVM that had survived the master-node, which is exactly
    what detached execution is for, could keep running on a container handed
    to the next simulation. Narrowing this to the unheld ones is what makes
    "marked by the clock" honest.

    Skipped when SIM_TIMEOUT_SEC == 0.
    """
    if SET.sim_timeout_sec <= 0:
        return
    # UTC: `datetime.now()` is naive *local* time, and pymongo stores and
    # compares it as though it were UTC. On a host that is not on UTC the
    # cutoff was wrong by the offset — in one direction it recovers nothing,
    # in the other it recovers live simulations.
    cutoff = utc_now() - timedelta(seconds=SET.sim_timeout_sec)
    stuck = mongo.simulation_repo.find_running_before(cutoff)
    if not stuck:
        return

    abandoned = []
    for sim in stuck:
        sim_id = ObjectId(sim["_id"])
        try:
            holder = mongo.resource_repo.find_by_simulation(sim_id)
        except Exception:
            log.exception(
                "Could not check whether a container holds %s; leaving it", sim_id)
            continue
        if holder is not None:
            # The reconciliation owns this one: it has a container to ask.
            log.info(
                "Simulation %s is still held by %s; leaving it to the "
                "reconciliation", sim_id, holder.get("name"),
            )
            continue
        abandoned.append(sim)

    if not abandoned:
        return
    log.warning(
        "Found %d RUNNING simulation(s) no container holds; failing them",
        len(abandoned),
    )
    for sim in abandoned:
        sim_id = ObjectId(sim["_id"])
        log.warning(
            "Recovering abandoned simulation %s (start_time=%s)",
            sim_id, sim.get("start_time"),
        )
        try:
            mongo.simulation_repo.mark_error(
                sim_id,
                "Recovered: left RUNNING with no container holding it",
                attempt_id=None)
        except Exception:
            log.exception("Failed to recover simulation %s", sim_id)
        generation_id = sim.get("generation_id")
        if generation_id:
            _check_and_close_generation(generation_id, mongo)


def prepare_simulation_files(
    sim: Simulation,
    mongo: MongoRepository,
) -> tuple[bool, list[str], list[str]]:
    """
    Downloads CSC/positions from the simulation and files from the SourceRepository.
    Returns (success, local_files, remote_files).
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    # Keyed by simulation, not by worker: a thread is no longer bound to one
    # container, so two runs could otherwise share a directory and overwrite
    # each other's firmware before it is sent.
    tmp_dir = Path(f"tmp/sim_{sim_oid}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    local_xml = tmp_dir / f"simulation_{sim_oid}.xml"
    local_dat = tmp_dir / f"positions_{sim_oid}.dat"

    local_files: list[str] = []
    remote_files: list[str] = []

    # CSC/XML
    try:
        mongo.fs_handler.download_file(sim["csc_file_id"], str(local_xml))
    except Exception as e:
        # Never ship a partial input to Cooja: the run would look successful
        # and its metrics would be attributed to a valid individual.
        log.error("Failed to download simulation.csc %s for simulation %s: %s",
                  sim.get("csc_file_id"), sim.get("_id"), e)
        return False, local_files, remote_files
    local_files.append(str(local_xml))
    remote_files.append("simulation.csc")

    # Positions (optional)
    if sim.get("pos_file_id"):
        try:
            mongo.fs_handler.download_file(sim["pos_file_id"], str(local_dat))
        except Exception as e:
            log.error("Failed to download positions.dat %s for simulation %s: %s",
                      sim.get("pos_file_id"), sim.get("_id"), e)
            return False, local_files, remote_files
        local_files.append(str(local_dat))
        remote_files.append("positions.dat")

    # Source repository
    src_repo_id = sim.get("source_repository_id")
    src: SourceRepository = mongo.source_repo.get_by_id(src_repo_id)
    if not src or "source_files" not in src:
        log.warning("Source repository %s not found for experiment %s", src_repo_id, sim.get("experiment_id"))
        return False, local_files, remote_files

    for sf in src["source_files"]:
        file_path = str(tmp_dir / sf["file_name"])
        try:
            mongo.fs_handler.download_file(sf["id"], file_path)
        except Exception as e:
            log.error("Failed to download firmware file %s (%s) for simulation %s: %s",
                      sf.get("file_name"), sf.get("id"), sim.get("_id"), e)
            return False, local_files, remote_files
        local_files.append(file_path)
        remote_files.append(sf["file_name"])

    return True, local_files, remote_files


def _detached_pid(runner: DetachedRun) -> int | None:
    """The pid of the run being followed, if one was recorded.

    One accessor, because ``handle`` is a *property*: the expression
    ``runner.handle()`` raised ``TypeError`` before the exception that needed
    the pid could even be constructed, which made the retry over a fresh
    session unreachable (R23).
    """
    handle = runner.handle
    return handle.pid if handle else None


def _resume_outcome(
    sim: Simulation,
    transport: SSHTransport,
    runner: DetachedRun,
    adopt_pid: int,
    where: str,
    sim_oid: ObjectId,
) -> RunOutcome:
    """What became of a run we attached to, asked **before** any deadline.

    The order is the whole fix. ``_remaining_timeout`` used to be applied
    first, so a master-node that came back after a long stop failed a
    simulation that had in fact finished minutes in: the time the master-node
    spent down was counted as time the simulation spent executing, and the
    result sitting on the container was never looked for.

    Three answers, kept apart:

    * still alive → follow it, with whatever is left of its timeout;
    * ended with an exit code → nothing to follow, go straight to collection;
    * ended without one → the same, and the caller's policy for an
      unconfirmed result decides what the bytes are worth.
    """
    try:
        state = inspect_run(
            transport, adopt_pid, remote_dir=transport.plan.remote_dir)
    except Exception as error:
        # We hold a container running something we cannot ask about. A plain
        # failure here would release it, so this is the unconfirmed case: the
        # worker is taken out of the pool rather than handed to the next job.
        raise TerminationUnconfirmed(
            f"could not ask {where} what became of the resumed run: {error}"
        ) from error

    if not state.alive:
        log.info(
            "[%s] The resumed run for %s is no longer executing "
            "(exit %s, %d bytes of testlog)",
            where, sim_oid, state.exit_code, state.testlog_bytes,
        )
        return RunOutcome(
            completed=state.finished,
            exit_code=state.exit_code,
            message="" if state.finished
            else "the run ended without recording an exit code",
        )

    remaining = _remaining_timeout(sim)
    if remaining is not None and remaining <= 0:
        message = (
            "the simulation was still running and had already exceeded "
            "SIM_TIMEOUT_SEC when it was resumed"
        )
        if not runner.terminate():
            raise TerminationUnconfirmed(
                f"{message}, and it could not be stopped")
        raise RunFailed(message)
    return runner.wait(
        timeout_sec=remaining,
        on_output=lambda text: log.info("[cooja][%s] %s", where, text.rstrip()),
    )


def _leftover_result(
    transport: SSHTransport, runner: DetachedRun, where: str
) -> int | None:
    """Bytes of testlog left behind by a run that is no longer executing.

    ``None`` when there is nothing to collect: the process is still alive, the
    container could not be asked, or it left nothing. Conservative on purpose
    — a live process means the artifacts are still being written, so they are
    not a result.
    """
    try:
        state = inspect_run(
            transport, _detached_pid(runner),
            remote_dir=transport.plan.remote_dir)
    except Exception as error:
        log.warning("[%s] Could not look for a leftover result: %s", where, error)
        return None
    if not state.collectable:
        return None
    return state.testlog_bytes


def run_cooja_simulation(
    sim: Simulation,
    transport: SSHTransport,
    mongo: MongoRepository,
    remote_files: list[str] | None = None,
    on_started: Callable[[int], None] | None = None,
    adopt_pid: int | None = None,
    attempt_id: str | None = None,
) -> RunReport:
    """
    Executes the simulation in the container, follows it, and stores its
    artifacts in GridFS. The local temporary files are always removed; the
    remote ones are the caller's to clear, because only the caller knows
    whether anything may still be using them.

    The connection arrives already open: the caller owns it, because it also
    sent the input files over it and because closing it is what releases the
    container.

    With ``adopt_pid`` the run is *already going* — a simulation that outlived
    a master-node restart — so it is followed from where it is instead of
    being started. Collecting the result is identical either way, which is why
    this is one function with one branch rather than two that drift apart.

    ``attempt_id`` identifies the placement; it is passed on to the callback
    that promotes the container, which is where a takeover is detected.

    **This function does not write the simulation's state.** It returns a
    ``RunReport`` with what it produced, or raises: ``RunFailed`` for a run
    that produced nothing usable, ``TerminationUnconfirmed`` when something
    may still be executing, ``LostOwnership`` when the container was handed
    on. The caller decides what each means, because it used to decide *as
    well* — and with both writing, the quarantine and the re-queue in the
    runner were never reached.
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    where = transport.plan.describe()
    log_path = f"{SET.local_log_dir}/sim_{sim_oid}.log"
    csv_path = f"{SET.local_log_dir}/sim_{sim_oid}.csv"
    try:
        runner = DetachedRun(transport, poll_interval_sec=SET.run_poll_interval_sec)
        # Nothing below catches anything but its own business: a blanket
        # `except Exception` here is what hid the control exceptions from the
        # caller and made its quarantine and re-queue unreachable.

        if adopt_pid:
            log.info("[%s] Resuming simulation %s (pid %s)", where, sim_oid, adopt_pid)
            runner.attach(adopt_pid)
            outcome = _resume_outcome(sim, transport, runner, adopt_pid, where, sim_oid)
        else:
            log.info("[%s] Starting simulation %s", where, sim_oid)
            # No `cd` and no channel: the run is launched in its own session
            # with its output going to a file, so the JVM outlives a dropped
            # connection instead of being killed by SIGHUP.
            command = (
                f"/opt/java/openjdk/bin/java --enable-preview "
                f"-Xms{SET.jvm_xms} -Xmx{SET.jvm_xmx} "
                f"-jar build/libs/cooja.jar --no-gui simulation.csc"
            )
            try:
                handle = runner.launch(command)
            except LaunchUnconfirmed as error:
                # Something may be executing with nobody holding its pid.
                # Reported as an ordinary failure, the container went back to
                # the pool with a simulation on it (R05.4).
                raise TerminationUnconfirmed(str(error)) from error
            if on_started is not None:
                try:
                    on_started(handle.pid)
                except LostOwnership as error:
                    # The container was handed to someone else while this run
                    # was starting. The pool already believes it is free, so
                    # the JVM has to go before a second simulation lands on
                    # the same worker.
                    log.warning(
                        "[%s] Lost the container for %s during launch; "
                        "stopping the run", where, sim_oid,
                    )
                    # The answer is the point. It was discarded, so a run that
                    # refused to stop on a container the pool already believed
                    # free was re-queued with nothing taking that worker out
                    # of the pool (R05.2).
                    raise LostOwnership(
                        str(error), stopped=runner.terminate()) from error
            outcome = runner.wait(
                timeout_sec=SET.sim_timeout_sec or None,
                on_output=lambda text: log.info(
                    "[cooja][%s] %s", where, text.rstrip()),
            )

        confirmed = True
        if not outcome.completed:
            # Timed out, killed from outside, or contact never came back. The
            # distinction is already in the outcome; what matters here is that
            # no result is fabricated from an incomplete run.
            if outcome.stopped is False:
                # Something may still be running in there. Distinguished from
                # a plain failure because the container must not go back to
                # the pool as though it were idle.
                raise TerminationUnconfirmed(
                    outcome.message or "the run could not be stopped")
            # Before writing the run off: are there bytes on the container?
            # A process reaped without recording an exit code leaves a
            # complete-looking testlog behind, and deleting it was the only
            # thing the previous code could do with it. They are collected and
            # kept, but **not** published as a result — see `confirmed` below.
            leftover = _leftover_result(transport, runner, where)
            if leftover is None:
                raise RunFailed(
                    outcome.message or "the simulation did not complete"
                )
            log.warning(
                "[%s] %s left %d bytes of testlog with no exit code; keeping "
                "them as an unconfirmed result", where, sim_oid, leftover,
            )
            confirmed = False
        elif outcome.exit_code not in (0, None):
            # Recorded rather than fatal: Cooja has exited non-zero while
            # still producing a usable testlog, and the previous
            # implementation ignored the status entirely. The CSV check below
            # remains the arbiter of success.
            log.warning(
                "[%s] Cooja exited with status %s after %.0fs",
                where, outcome.exit_code, outcome.elapsed_sec,
            )

        log.info("[%s] Copying log to %s", where, log_path)
        # From here on the run is over and its result exists on the container.
        # A failure is therefore a *collection* failure, which the caller can
        # retry over a fresh session instead of writing the run off.
        try:
            transport.get_file("COOJA.testlog", log_path)
        except Exception as error:
            raise CollectionFailed(
                f"could not bring back the result: {error}",
                # `handle` is a property. Calling it raised TypeError before
                # this line could build the exception, so the retry over a
                # fresh session was unreachable and a dropped download cost a
                # finished simulation. The test that covered it drove a
                # MagicMock, on which `handle()` is valid.
                pid=_detached_pid(runner),
                stage="download",
            ) from error

        # Each stage of the collection fails differently, and conflating them
        # is what made "the download dropped" indistinguishable from "GridFS
        # refused the upload". The stage travels with the exception so the
        # caller knows what may be retried and which blobs already exist.
        try:
            log_id = mongo.fs_handler.upload_file(log_path, "sim_result.log")
        except Exception as error:
            raise CollectionFailed(
                f"the testlog could not be stored: {error}",
                pid=_detached_pid(runner), stage="upload",
            ) from error
        log.info("[%s] Log saved with ID: %s", where, log_id)

        try:
            cooja_files.convert_cooja_log_to_csv(log_path, csv_path)
            csv_id = mongo.fs_handler.upload_file(csv_path, "sim_result.csv")
        except Exception as error:
            raise CollectionFailed(
                f"the result could not be converted or stored: {error}",
                pid=_detached_pid(runner), stage="convert",
                # Already in GridFS. Named so the caller can remove it rather
                # than leave a blob nothing references.
                log_id=log_id,
            ) from error
        log.info("[%s] Log converted CSV and saved with ID: %s", where, csv_id)

        csv_file = Path(csv_path)
        if csv_file.exists() and csv_file.stat().st_size != 0:
            # Still collection, not execution. Reading the CSV back or
            # evaluating the metric configuration can fail for reasons that
            # have nothing to do with the run — and falling through to the
            # generic handler wrote the simulation off while its testlog sat
            # in GridFS, referenced by nothing.
            try:
                df = pd.read_csv(csv_path)
                exp_id = sim["experiment_id"]
                cfg = mongo.experiment_repo.get_metrics_data_conversion(
                    str(exp_id))
                net_meas = statistics.evaluate_config(df, cfg, log)
            except Exception as error:
                raise CollectionFailed(
                    f"the result was stored but could not be evaluated: {error}",
                    pid=_detached_pid(runner), stage="evaluate",
                    log_id=log_id, csv_id=csv_id,
                ) from error
            log.info("[%s] Metrics calculated: %s", where, net_meas)

            # RPL DODAG analysis (formation time + tree) from the raw testlog.
            # Best-effort: never let it fail the simulation result.
            dodag = None
            try:
                dodag = rpl_dodag.analyze_dodag_full(log_path)
                log.info(
                    "[%s] DODAG: %d/%d joined, convergence=%s ms",
                    where,
                    dodag["convergence"]["n_nodes"],
                    dodag["deployed_nodes"],
                    dodag["convergence"]["convergence_time_ms"],
                )
            except Exception:
                log.exception("[%s] DODAG analysis failed", where)

            return RunReport(
                log_id=log_id, csv_id=csv_id, metrics=net_meas, dodag=dodag,
                confirmed=confirmed,
                reason="" if confirmed else (
                    "the run process ended without recording an exit code, so "
                    "this result is not confirmed"
                ),
            )

        log.warning(
            "[%s] CSV file is missing or empty for simulation %s", where, sim_oid)
        # Not a plain failure any more: the testlog is in GridFS and it is the
        # only evidence of what happened. Previously it was uploaded and then
        # orphaned, with the simulation marked in error and nothing to read.
        return RunReport(
            log_id=log_id, csv_id=csv_id, metrics={}, dodag=None,
            confirmed=False,
            reason="the converted CSV was empty, so no metrics could be computed",
        )
    finally:
        # Only what this function created locally. The remote artifacts are
        # the caller's to clear, because clearing them is safe exactly when
        # the caller knows nothing is still using the container — and that is
        # the judgement the caller makes.
        for path in (log_path, csv_path):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as ex:
                log.warning("Failed to remove temp file %s: %s", path, ex)


def load_initial_waiting_jobs(
    mongo: MongoRepository, submit: Callable[[str], None]
) -> None:
    """Offer every WAITING simulation found at startup."""
    log.info("Searching for pending simulations...")
    pending = mongo.simulation_repo.find_pending()
    for sim in pending:
        sim_id = str(sim["_id"])
        log.info("Pending simulation: %s", sim_id)
        submit(sim_id)


def watch_status_waiting_enqueue(
    simRepo: SimulationRepository,
    submit: Callable[[str], None],
) -> None:
    """
    Subscribes to the simulations Change Stream and offers simulation IDs
    as they enter WAITING status.

    ``submit`` is a callable rather than a queue so the producer does not have
    to know how the dispatcher takes work.
    """
    log.info("[mongowatch] Watching for WAITING simulations...")

    def on_simulation_event(change: dict):
        doc = change.get("fullDocument")
        if not doc:
            log.warning("[mongowatch] Event without fullDocument; skipping.")
            return
        sim_id = str(doc["_id"])
        log.info("[mongowatch] Enqueuing simulation %s", sim_id)
        submit(sim_id)

    simRepo.watch_status_waiting(on_simulation_event)


def _remaining_timeout(sim: Simulation) -> float | None:
    """How much of SIM_TIMEOUT_SEC is left for a simulation already running.

    Without this a restart would reset the clock, so a stuck simulation could
    hold a container indefinitely by being adopted again and again.
    """
    if not SET.sim_timeout_sec:
        return None
    # Naive or aware: a resume is exactly the path where a TypeError would
    # be worst, because it only runs after something already went wrong.
    spent = elapsed_since(sim.get("start_time"))
    if spent is None:
        return float(SET.sim_timeout_sec)
    return float(SET.sim_timeout_sec) - spent


def _cleanup_local_files(paths: list[str]) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception as ex:
            log.warning("Failed to remove temp file %s: %s", path, ex)


def _run_synthetic_if_requested(sim: Simulation, mongo: MongoRepository) -> bool:
    """Handle a synthetic simulation without placing it on a container.

    Synthetic runs never touch Cooja, so holding a container for one would
    take capacity away from work that needs it.
    """
    exp_id = sim.get("experiment_id")
    exp_doc = mongo.experiment_repo.get(exp_id)
    if exp_doc is None:
        log.warning(
            "Experiment %s not found for simulation %s; "
            "falling back to the env-var synthetic setting.", exp_id, sim.get("_id"),
        )
    mode, bench, noise_std = resolve_synthetic_settings(exp_doc)
    if not mode:
        return False
    log.info("sim %s → Synthetic Data (exp=%s)", sim.get("_id"), exp_id)
    run_synthetic_simulation(sim, mongo, bench=bench, noise_std=noise_std)
    return True


def _resume_in_background(runner: SimulationRunner) -> Callable:
    """Follow an adopted run on its own thread.

    Reconciliation must not block on a simulation that may have hours left to
    go, and each adopted run needs its own follower for the same reason a
    freshly placed one does.
    """

    def on_adopt(sim, host, resource, attempt_id: str, pid: int) -> None:
        Thread(
            target=runner.resume,
            args=(sim, host, resource, attempt_id, pid),
            daemon=True,
            name=f"resume-{str(sim.get('_id'))[:8]}",
        ).start()

    return on_adopt


def build_liveness_probe(opener: SecretOpener) -> Callable:
    """A probe that asks a container what became of a recorded run.

    Returns a ``RunState`` — alive, finished with an exit code, or gone, plus
    whether a testlog is still sitting there — or ``None`` when the container
    could not be reached. That last distinction is load-bearing: conflating
    "unreachable" with "not running" is what would let a network blip destroy
    a simulation.

    It used to answer a bare boolean, and "not alive" was taken for "failed".
    A run that finished while the master-node was down therefore had its
    result discarded, and the next job cleared the artifacts that proved it.
    """

    def probe(host: dict, resource: dict, pid: int | None):
        transport = build_transport(host, resource, opener)
        try:
            with transport:
                return inspect_run(
                    transport, pid, remote_dir=transport.plan.remote_dir)
        except TransportError:
            return None

    return probe


def build_dispatcher(
    mongo: MongoRepository, keystore: Keystore, opener: SecretOpener
) -> Dispatcher:
    """Assemble the managed scheduling path."""
    def learn_host_key(host_id, fingerprint: str) -> None:
        """Pin a host key seen on first contact — without overwriting one.

        R11. This used to be an unconditional ``update``, so the *second*
        path to reach a host could replace the pin the first one established.
        Trust on first use only means something if the first use is the one
        that sticks, so the write is conditional on the field being unset, and
        a refusal is checked against what is stored rather than ignored: the
        two cases it can mean — someone else pinned the same key, or the host
        is answering with a different one — are not the same event.

        A conflict **raises**. Logging it and returning left the session that
        had just accepted the other key open and in use: the policy only
        calls this when it found no pin, so by the time the conflict is
        visible the connection is already established. ``HostKeyMismatch``
        propagates out of the policy and out of ``connect``, which is what
        refusing actually means.
        """
        try:
            if mongo.host_repo.pin_host_key(host_id, fingerprint):
                log.info("Pinned the host key of %s: %s", host_id, fingerprint)
                return
            host = mongo.host_repo.get(host_id) or {}
            stored = (host.get("access") or {}).get("host_key_fingerprint")
        except Exception:
            # Cannot tell. The connection stands, as it did before: refusing
            # a session over a database failure would take the pool out for a
            # reason that has nothing to do with the host.
            log.exception("Could not pin the host key of %s", host_id)
            return
        if stored and stored != fingerprint:
            log.error(
                "Host %s is pinned to %s but presented %s; refusing this "
                "session", host.get("name", host_id), stored, fingerprint,
            )
            raise HostKeyMismatch(
                str(host.get("name") or host_id), stored, fingerprint)

    runner = SimulationRunner(
        mongo,
        execute=run_cooja_simulation,
        prepare=prepare_simulation_files,
        opener=opener,
        cleanup_local=_cleanup_local_files,
        on_learn_host_key=learn_host_key,
    )
    dispatcher = Dispatcher(
        mongo,
        runner,
        policy=build_policy(SET.sched_policy),
        # Synthetic runs never touch Cooja, so they are completed here instead
        # of being placed: reserving a container for one would take capacity
        # away from work that actually needs it.
        intercept=lambda sim: _run_synthetic_if_requested(sim, mongo),
    )

    # A release frees a container, and a change to the pool may add one. Both
    # wake a decision that is holding a job for want of capacity.
    Thread(
        target=mongo.resource_repo.watch,
        args=(lambda _change: dispatcher.notify_capacity(),),
        daemon=True, name="watch-resources",
    ).start()
    Thread(
        target=mongo.host_repo.watch,
        args=(lambda _change: dispatcher.notify_capacity(),),
        daemon=True, name="watch-hosts",
    ).start()
    return dispatcher


def main() -> None:
    log.info("start")
    log.info("local containers seeded from the environment: %s", len(SET.hostnames))
    log.info(
        "env: MONGO_URI=%s | DB_NAME=%s | IS_DOCKER=%s",
        SET.mongo_uri, SET.db_name, SET.is_docker,
    )

    mongo = create_mongo_repository_factory(SET.mongo_uri, SET.db_name)

    keystore = Keystore.load_or_create(SET.infra_key_path)
    opener = SecretOpener(keystore)
    publish_public_key(mongo, keystore)
    ensure_local_host(
        mongo,
        is_docker=SET.is_docker,
        hostnames=SET.hostnames,
        ports=SET.ports,
        remote_dir=SET.remote_dir,
    )
    dispatcher = build_dispatcher(mongo, keystore, opener)
    def reconcile(*, quiet: bool = False):
        """One reconciliation pass, at startup or from the health cycle.

        Containers claimed by a process that is gone would stay claimed, and
        the pool would shrink by that many on every restart. The probe keeps
        this from also destroying runs that outlived the master-node, and the
        handler picks up following them so their results are collected
        instead of being abandoned to the timeout.

        The decision is the same either way. The periodic pass additionally
        skips the attempts this process is already following — their follower
        will collect and release — and stays quiet when there is nothing to
        report.
        """
        return reconcile_occupied_resources(
            mongo,
            probe=build_liveness_probe(opener),
            on_adopt=_resume_in_background(dispatcher.runner),
            is_followed=dispatcher.runner.is_following if quiet else None,
            # A reservation younger than its own TTL is a placement in
            # progress, and the sweep is what owns that timeout. Zero at
            # startup: nothing there can belong to this process.
            reserved_grace_sec=SET.reservation_ttl_sec if quiet else 0.0,
            quiet=quiet,
        )

    monitor = HealthMonitor(
        mongo,
        opener=opener,
        interval_sec=SET.infra_check_interval_sec,
        failure_threshold=SET.infra_failure_threshold,
        reservation_ttl_sec=SET.reservation_ttl_sec,
        on_capacity=dispatcher.notify_capacity,
        # Every cycle, not just at startup: a container whose host was
        # unreachable at boot kept its reservation and nothing ever came back
        # to it.
        reconcile=lambda: reconcile(quiet=True),
        # The counterpart of the quarantine: a worker held for something
        # unresolved comes back only when the container says it has nothing
        # alive and nothing left behind.
        inspect_run=build_liveness_probe(opener),
    )

    reconcile()

    # Only after the reconciliation, and only for what it did not cover: a
    # simulation marked RUNNING that no container holds. Before, this ran
    # first and by the clock alone, which let the reconciliation free a
    # container whose JVM was still running.
    recover_abandoned_simulations(mongo)

    # Synchronously, before anything is placed: the first placement
    # decision then reads measured statuses instead of assumptions. This is
    # what replaced seeding hosts and containers as already usable.
    # Forced, so it does not inherit the probes of the process that just
    # ended — later cycles may trust a recent measurement, but the pool
    # the dispatcher starts from is measured by this process.
    log.info("Running the first health cycle before placing any work...")
    monitor.cycle(force=True)

    Thread(target=monitor.run, daemon=True, name="health").start()
    Thread(target=dispatcher.run, daemon=True, name="dispatcher").start()
    submit = dispatcher.submit

    load_initial_waiting_jobs(mongo, submit)

    Thread(
        target=watch_status_waiting_enqueue,
        args=(mongo.simulation_repo, submit),
        daemon=True, name="watch-simulations",
    ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("closing...")


if __name__ == "__main__":
    main()
