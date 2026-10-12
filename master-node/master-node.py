import os
import sys
import time
import queue
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
from lib.bootstrap import (
    ensure_local_host,
    local_host_document,
    local_resource_documents,
    publish_public_key,
)
from lib.credentials import SecretOpener
from lib.detached import RUN_ARTIFACTS, DetachedRun, is_running
from lib.recovery import reconcile_occupied_resources
from lib.scheduler import Dispatcher, SimulationRunner, build_policy
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
    infra_mode: str
    sched_policy: str
    infra_key_path: str
    run_poll_interval_sec: float

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

        # "managed" places work by policy across the registered hosts.
        # "legacy" keeps the pre-feature pool: one worker per container from
        # NUMBER_OF_CONTAINERS, all racing for the queue. Kept as an escape
        # hatch for the transition and meant to be removed once the managed
        # path has been exercised in deployment.
        infra_mode = os.getenv("INFRA_MODE", "managed").strip().lower()
        sched_policy = os.getenv("SCHED_POLICY", "least_loaded_host")
        infra_key_path = os.getenv("INFRA_KEY_PATH", DEFAULT_KEY_PATH)
        # How often a running simulation is checked on. One cheap round trip
        # each time (~40ms measured), so this is about log latency rather than
        # load; it also bounds how long a finished run holds its container.
        run_poll_interval_sec = float(os.getenv("RUN_POLL_INTERVAL_SEC", "5"))

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
            infra_mode=infra_mode,
            sched_policy=sched_policy,
            infra_key_path=infra_key_path,
            run_poll_interval_sec=run_poll_interval_sec,
        )


SET = Settings.from_env()


def _assert_capacity(num_workers: int) -> None:
    if num_workers > len(SET.hostnames) or num_workers > len(SET.ports):
        raise ValueError(
            f"Workers({num_workers}) > hostnames({len(SET.hostnames)})/ports({len(SET.ports)})"
        )


def _check_and_close_generation(generation_id: ObjectId, mongo: MongoRepository) -> None:
    """
    Called after every simulation status change.
    If no simulation in this generation is still WAITING or RUNNING:
      - DONE  : every simulation finished with DONE.
      - ERROR : at least one simulation finished with ERROR.
    """
    gr = mongo.generation_repo
    try:
        if gr.any_simulation_active(generation_id):
            return  # still work in progress — do nothing
        if gr.all_simulations_done(generation_id):
            gr.mark_done(generation_id)
        else:
            gr.mark_error(generation_id)
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
            mongo.simulation_repo.mark_error(sim_id, "Recovered: stuck in RUNNING on master-node restart")
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
) -> None:
    """
    Executes the simulation in the container, monitors logs, and stores results in GridFS.
    Per-run files (remote artifacts and local log/csv) are always removed before returning.

    The connection arrives already open: the caller owns it, because it also
    sent the input files over it and because closing it is what releases the
    container. Everything below is the same sequence as before — only the
    paramiko calls moved behind the transport.
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    where = transport.plan.describe()
    log_path = f"{SET.local_log_dir}/sim_{sim_oid}.log"
    csv_path = f"{SET.local_log_dir}/sim_{sim_oid}.csv"
    try:
        log.info("[%s] Starting simulation %s", where, sim_oid)
        mongo.simulation_repo.mark_running(sim_oid)

        # No `cd` and no channel: the run is launched in its own session with
        # its output going to a file, so the JVM outlives a dropped
        # connection instead of being killed by SIGHUP.
        command = (
            f"/opt/java/openjdk/bin/java --enable-preview "
            f"-Xms{SET.jvm_xms} -Xmx{SET.jvm_xmx} "
            f"-jar build/libs/cooja.jar --no-gui simulation.csc"
        )

        runner = DetachedRun(transport, poll_interval_sec=SET.run_poll_interval_sec)
        handle = runner.launch(command)
        if on_started is not None:
            on_started(handle.pid)

        outcome = runner.wait(
            timeout_sec=SET.sim_timeout_sec or None,
            on_output=lambda text: log.info("[cooja][%s] %s", where, text.rstrip()),
        )
        if not outcome.completed:
            # Timed out, killed from outside, or contact never came back. The
            # distinction is already in the outcome; what matters here is that
            # no result is fabricated from an incomplete run.
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

            mongo.simulation_repo.mark_done(sim_oid, log_id, csv_id, net_meas, dodag)
        else:
            log.warning("[%s] CSV file is missing or empty for simulation %s", where, sim_oid)
            mongo.simulation_repo.mark_error(sim_oid, "CSV file is missing or empty after conversion")

        if sim.get("generation_id"):
            _check_and_close_generation(sim["generation_id"], mongo)

    except Exception as e:
        log.exception("[%s] Simulation ERROR %s: %s", where, sim_oid, e)
        try:
            mongo.simulation_repo.mark_error(sim_oid, str(e))
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


def simulation_worker(
    worker_id: int,
    sim_queue: queue.Queue,
    host: dict,
    resource: dict,
) -> None:
    """
    Worker that consumes simulation IDs from the queue and executes each one.
    The full simulation document is fetched from MongoDB at execution time.
    Simulations not in WAITING status are skipped (already claimed by another worker).

    The worker is still bound to one container, as before. What changed is how
    it reaches it: a transport built from the host and resource records, so the
    same code serves a local container and one on a remote host. The placement
    decision itself still belongs to whoever fills the queue.
    """
    mongo = create_mongo_repository_factory(SET.mongo_uri, SET.db_name)
    where = f"{host.get('name')}/{resource.get('name')}"
    while True:
        sim_id = sim_queue.get()
        sim = None
        local_files: list[str] = []
        try:
            if sim_id is None:
                return

            sim = mongo.simulation_repo.get(sim_id)
            if sim is None:
                log.warning("[%s] Simulation %s not found; skipping.", where, sim_id)
                continue

            if sim.get("status") != EnumStatus.WAITING:
                log.info("[%s] Simulation %s already in status '%s'; skipping.",
                         where, sim_id, sim.get("status"))
                continue

            # Resolve synthetic mode: per-experiment config takes priority over env var.
            exp_id = sim.get("experiment_id")
            exp_doc = mongo.experiment_repo.get(exp_id)
            if exp_doc is None:
                log.warning(
                    "[%s] Experiment %s not found for simulation %s; "
                    "falling back to the env-var synthetic setting.",
                    where, exp_id, sim_id,
                )
            mode, bench, noise_std = resolve_synthetic_settings(exp_doc)
            log.info(
                "[%s] sim %s → mode=%s (exp=%s)",
                where, sim_id, "Synthetic Data" if mode else "Simulation", exp_id,
            )
            if mode:
                run_synthetic_simulation(sim, mongo, bench=bench, noise_std=noise_std)
                continue

            log.info("[%s] Preparing simulation %s", where, sim_id)

            success, local_files, remote_files = prepare_simulation_files(sim, mongo)
            if not success:
                log.warning("[%s] Skipping simulation %s (prepare failed)", where, sim_id)
                try:
                    mongo.simulation_repo.mark_error(ObjectId(sim["_id"]), "Failed to prepare simulation files")
                except Exception:
                    log.exception("Failed to mark error on simulation %s", sim_id)
                if sim.get("generation_id"):
                    _check_and_close_generation(sim["generation_id"], mongo)
                continue

            # One connection for the whole run, where there used to be two: the
            # upload and the execution share it, and closing it is what ends
            # the attempt. Over a tunnel a second handshake is not free.
            transport = build_transport(host, resource)
            with transport:
                # Defensive: clear artifacts left behind if a previous run
                # crashed before its own cleanup could execute.
                transport.cleanup()
                transport.put_files(local_files, remote_files)
                run_cooja_simulation(sim, transport, mongo, remote_files)

        except Exception as e:
            log.exception("[%s] General ERROR: %s", where, e)
            try:
                if sim and "_id" in sim:
                    mongo.simulation_repo.mark_error(ObjectId(sim["_id"]), str(e))
            except Exception:
                log.exception("Failed to mark error on simulation after general exception")
            if sim and sim.get("generation_id"):
                _check_and_close_generation(sim["generation_id"], mongo)
        finally:
            for f in local_files:
                try:
                    Path(f).unlink(missing_ok=True)
                except Exception as ex:
                    log.warning("Failed to remove temp file %s: %s", f, ex)
            sim_queue.task_done()


def start_workers(num_workers: int) -> queue.Queue:
    _assert_capacity(num_workers)
    host = local_host_document(is_docker=SET.is_docker)
    resources = local_resource_documents(
        hostnames=SET.hostnames,
        ports=SET.ports,
        remote_dir=SET.remote_dir,
    )
    q: queue.Queue = queue.Queue()
    for i in range(num_workers):
        t = Thread(
            target=simulation_worker,
            args=(i, q, host, resources[i]),
            daemon=True,
        )
        t.start()
    log.info("Workers started: %s", num_workers)
    return q


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

    ``submit`` is a callable rather than a queue so the same producer feeds
    the managed dispatcher and the legacy worker pool.
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
    number_of_containers = len(SET.hostnames)
    log.info("start")
    log.info("number of containers: %s", number_of_containers)
    log.info(
        "env: MONGO_URI=%s | DB_NAME=%s | IS_DOCKER=%s | INFRA_MODE=%s",
        SET.mongo_uri, SET.db_name, SET.is_docker, SET.infra_mode,
    )

    mongo = create_mongo_repository_factory(SET.mongo_uri, SET.db_name)

    recover_stuck_simulations(mongo)

    if SET.infra_mode == "legacy":
        log.warning(
            "INFRA_MODE=legacy: one worker per container from "
            "NUMBER_OF_CONTAINERS, no host balancing. This path exists for "
            "the transition and will be removed."
        )
        sim_queue = start_workers(number_of_containers)
        submit = sim_queue.put
    else:
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
        # Containers claimed by a process that is gone would stay claimed,
        # and the pool would shrink by that many on every restart. The probe
        # is what keeps this from also destroying runs that outlived the
        # master-node, which detached execution makes possible.
        reconcile_occupied_resources(mongo, build_liveness_probe(opener))
        dispatcher = build_dispatcher(mongo, keystore, opener)
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
