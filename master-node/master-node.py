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
from lib.detached import RUN_ARTIFACTS, DetachedRun, is_running
from lib.health import HealthMonitor
from lib.recovery import reconcile_occupied_resources
from lib.scheduler import (
    Dispatcher,
    LostOwnership,
    SimulationRunner,
    TerminationUnconfirmed,
    build_policy,
)
from lib.synthetic_data import run_synthetic_simulation, resolve_synthetic_settings
from lib.transport import SSHTransport, TransportError, build_transport

# pylib crypto — master-node only; the API deliberately cannot open a secret.
from pylib.crypto.sealed import DEFAULT_KEY_PATH, Keystore

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


def recover_stuck_simulations(mongo: MongoRepository) -> None:
    """
    On startup, marks as ERROR any simulation that was left in RUNNING state
    (e.g. from a previous crash), then attempts to close affected generations.
    Skipped when SIM_TIMEOUT_SEC == 0.
    """
    if SET.sim_timeout_sec <= 0:
        return
    cutoff = datetime.now() - timedelta(seconds=SET.sim_timeout_sec)
    stuck = mongo.simulation_repo.find_running_before(cutoff)
    if not stuck:
        return
    log.warning("Found %d stuck simulation(s) in RUNNING; recovering...", len(stuck))
    for sim in stuck:
        sim_id = ObjectId(sim["_id"])
        log.warning("Recovering stuck simulation %s (start_time=%s)", sim_id, sim.get("start_time"))
        try:
            mongo.simulation_repo.mark_error(
                sim_id, "Recovered: stuck in RUNNING on master-node restart",
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


def run_cooja_simulation(
    sim: Simulation,
    transport: SSHTransport,
    mongo: MongoRepository,
    remote_files: list[str] | None = None,
    on_started: Callable[[int], None] | None = None,
    adopt_pid: int | None = None,
    attempt_id: str | None = None,
) -> None:
    """
    Executes the simulation in the container, monitors logs, and stores results in GridFS.
    Per-run files (remote artifacts and local log/csv) are always removed before returning.

    The connection arrives already open: the caller owns it, because it also
    sent the input files over it and because closing it is what releases the
    container.

    With ``adopt_pid`` the run is *already going* — a simulation that outlived
    a master-node restart — so it is followed from where it is instead of
    being started. Collecting the result is identical either way, which is why
    this is one function with one branch rather than two that drift apart.

    ``attempt_id`` is the placement this run belongs to, and every write about
    the simulation carries it. A run can lose its container without noticing —
    a recovery pass or the stale-reservation sweep may have handed it on — and
    the guard is what stops this thread from publishing over the attempt that
    took over.
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    where = transport.plan.describe()
    log_path = f"{SET.local_log_dir}/sim_{sim_oid}.log"
    csv_path = f"{SET.local_log_dir}/sim_{sim_oid}.csv"
    try:
        runner = DetachedRun(transport, poll_interval_sec=SET.run_poll_interval_sec)

        if adopt_pid:
            log.info("[%s] Resuming simulation %s (pid %s)", where, sim_oid, adopt_pid)
            runner.attach(adopt_pid)
            # The deadline counts from when the run actually started, not from
            # now: a restart must not hand a simulation a fresh timeout.
            timeout_sec = _remaining_timeout(sim)
            if timeout_sec is not None and timeout_sec <= 0:
                message = (
                    "the simulation had already exceeded SIM_TIMEOUT_SEC when "
                    "it was resumed"
                )
                if not runner.terminate():
                    raise TerminationUnconfirmed(
                        f"{message}, and it could not be stopped")
                raise RuntimeError(message)
        else:
            log.info("[%s] Starting simulation %s", where, sim_oid)
            if not mongo.simulation_repo.mark_running(
                    sim_oid, attempt_id=attempt_id):
                raise LostOwnership(
                    "the simulation was taken over before this run started")

            # No `cd` and no channel: the run is launched in its own session
            # with its output going to a file, so the JVM outlives a dropped
            # connection instead of being killed by SIGHUP.
            command = (
                f"/opt/java/openjdk/bin/java --enable-preview "
                f"-Xms{SET.jvm_xms} -Xmx{SET.jvm_xmx} "
                f"-jar build/libs/cooja.jar --no-gui simulation.csc"
            )
            handle = runner.launch(command)
            if on_started is not None:
                try:
                    on_started(handle.pid)
                except LostOwnership:
                    # The container was handed to someone else while this run
                    # was starting. The pool already believes it is free, so
                    # the JVM has to go before a second simulation lands on
                    # the same worker.
                    log.warning(
                        "[%s] Lost the container for %s during launch; "
                        "stopping the run", where, sim_oid,
                    )
                    runner.terminate()
                    raise
            timeout_sec = SET.sim_timeout_sec or None

        outcome = runner.wait(
            timeout_sec=timeout_sec,
            on_output=lambda text: log.info("[cooja][%s] %s", where, text.rstrip()),
        )
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
            raise RuntimeError(
                outcome.message or "the simulation did not complete"
            )
        if outcome.exit_code not in (0, None):
            # Recorded rather than fatal: Cooja has exited non-zero while
            # still producing a usable testlog, and the previous
            # implementation ignored the status entirely. The CSV check below
            # remains the arbiter of success.
            log.warning(
                "[%s] Cooja exited with status %s after %.0fs",
                where, outcome.exit_code, outcome.elapsed_sec,
            )

        log.info("[%s] Copying log to %s", where, log_path)
        transport.get_file("COOJA.testlog", log_path)

        log_id = mongo.fs_handler.upload_file(log_path, "sim_result.log")
        log.info("[%s] Log saved with ID: %s", where, log_id)

        cooja_files.convert_cooja_log_to_csv(log_path, csv_path)
        csv_id = mongo.fs_handler.upload_file(csv_path, "sim_result.csv")
        log.info("[%s] Log converted CSV and saved with ID: %s", where, csv_id)

        csv_file = Path(csv_path)
        if csv_file.exists() and csv_file.stat().st_size != 0:
            df = pd.read_csv(csv_path)
            exp_id = sim["experiment_id"]
            cfg = mongo.experiment_repo.get_metrics_data_conversion(str(exp_id))
            net_meas = statistics.evaluate_config(df, cfg, log)
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

            if not mongo.simulation_repo.mark_done(
                    sim_oid, log_id, csv_id, net_meas, dodag,
                    attempt_id=attempt_id):
                # The result is sound but it is not ours to publish any more:
                # something else owns this simulation now. The blobs were
                # already uploaded, so they are removed rather than left
                # behind with nothing referencing them.
                for orphan in (log_id, csv_id):
                    try:
                        mongo.fs_handler.delete_file(str(orphan))
                    except Exception:
                        log.exception("Could not remove orphaned file %s", orphan)
                raise LostOwnership(
                    "the simulation was taken over while it was running")
        else:
            log.warning("[%s] CSV file is missing or empty for simulation %s", where, sim_oid)
            mongo.simulation_repo.mark_error(
                sim_oid, "CSV file is missing or empty after conversion",
                attempt_id=attempt_id)

        if sim.get("generation_id"):
            _check_and_close_generation(sim["generation_id"], mongo)

    except Exception as e:
        log.exception("[%s] Simulation ERROR %s: %s", where, sim_oid, e)
        try:
            mongo.simulation_repo.mark_error(
                sim_oid, str(e), attempt_id=attempt_id)
        except Exception:
            log.exception("Failed to mark error on simulation %s", sim_oid)
        if sim.get("generation_id"):
            _check_and_close_generation(sim["generation_id"], mongo)
    finally:
        # The detached run's bookkeeping files are the caller's to declare:
        # the transport's cleanup stays generic and knows nothing about them.
        transport.cleanup(list(remote_files or []) + list(RUN_ARTIFACTS))
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
    started = sim.get("start_time")
    if not isinstance(started, datetime):
        return float(SET.sim_timeout_sec)
    spent = (datetime.now() - started).total_seconds()
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
    """A probe that answers whether a recorded run is still going.

    Returns None when the container cannot be reached, which the caller must
    keep distinct from "not running": conflating them is what would let a
    network blip destroy a simulation.
    """

    def probe(host: dict, resource: dict, pid: int | None) -> bool | None:
        transport = build_transport(host, resource, opener)
        try:
            with transport:
                return is_running(
                    transport, pid, remote_dir=transport.plan.remote_dir)
        except TransportError:
            return None

    return probe


def build_dispatcher(
    mongo: MongoRepository, keystore: Keystore, opener: SecretOpener
) -> Dispatcher:
    """Assemble the managed scheduling path."""
    def learn_host_key(host_id, fingerprint: str) -> None:
        mongo.host_repo.update(host_id, {"access.host_key_fingerprint": fingerprint})

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

    recover_stuck_simulations(mongo)

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
    monitor = HealthMonitor(
        mongo,
        opener=opener,
        interval_sec=SET.infra_check_interval_sec,
        failure_threshold=SET.infra_failure_threshold,
        reservation_ttl_sec=SET.reservation_ttl_sec,
        on_capacity=dispatcher.notify_capacity,
    )

    # Containers claimed by a process that is gone would stay claimed, and
    # the pool would shrink by that many on every restart. The probe keeps
    # this from also destroying runs that outlived the master-node, and the
    # handler picks up following them so their results are collected
    # instead of being abandoned to the timeout.
    reconcile_occupied_resources(
        mongo,
        probe=build_liveness_probe(opener),
        on_adopt=_resume_in_background(dispatcher.runner),
    )

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
