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
from bson import ObjectId
from paramiko import SSHClient
from scp import SCPClient

# SysPath for common modules
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# lib master-node
from lib.sshscp import create_ssh_client, send_files_scp
from lib.synthetic_data import run_synthetic_simulation

# pylib
from pylib.db import create_mongo_repository_factory, MongoRepository, EnumStatus
from pylib.db.models import Simulation, SourceRepository
from pylib.db.repositories.simulation import SimulationRepository
from pylib import cooja_files
from pylib import statistics

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

        default_n = int(os.getenv("NUMBER_OF_CONTAINERS", "3"))
        if is_docker:
            hostnames = [f"cooja{i+1}" for i in range(default_n)]
            ports = [22 for _ in range(default_n)]
        else:
            hostnames = ["localhost" for _ in range(default_n)]
            ports = [2231 + i for i in range(default_n)]

        sim_timeout_sec = int(os.getenv("SIM_TIMEOUT_SEC", "3600"))

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
    worker_id: int,
    mongo: MongoRepository,
) -> tuple[bool, list[str], list[str]]:
    """
    Downloads CSC/positions from the simulation and files from the SourceRepository.
    Returns (success, local_files, remote_files).
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    tmp_dir = Path(f"tmp/worker_{worker_id}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    local_xml = tmp_dir / f"simulation_{sim_oid}.xml"
    local_dat = tmp_dir / f"positions_{sim_oid}.dat"

    local_files: list[str] = []
    remote_files: list[str] = []

    # CSC/XML
    mongo.fs_handler.download_file(sim["csc_file_id"], str(local_xml))
    local_files.append(str(local_xml))
    remote_files.append("simulation.csc")

    # Positions (optional)
    if sim.get("pos_file_id"):
        mongo.fs_handler.download_file(sim["pos_file_id"], str(local_dat))
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
        mongo.fs_handler.download_file(sf["id"], file_path)
        local_files.append(file_path)
        remote_files.append(sf["file_name"])

    return True, local_files, remote_files


def run_cooja_simulation(
    sim: Simulation,
    port: int,
    hostname: str,
    mongo: MongoRepository,
) -> None:
    """
    Executes the simulation in the container via SSH, monitors logs, and stores results in GridFS.
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    ssh: SSHClient = create_ssh_client(hostname, port, "root", "root")
    try:
        log.info("[port=%s host=%s] Starting simulation %s", port, hostname, sim_oid)
        mongo.simulation_repo.mark_running(sim_oid)

        command = (
            f"cd {SET.remote_dir} && "
            f"/opt/java/openjdk/bin/java --enable-preview -Xms2g -Xmx4g "
            f"-jar build/libs/cooja.jar --no-gui simulation.csc"
        )

        stdin, stdout, stderr = ssh.exec_command(command, get_pty=True, timeout=SET.sim_timeout_sec or None)

        chan = stdout.channel
        chan.settimeout(5.0)
        while not chan.exit_status_ready():
            if chan.recv_ready():
                out = chan.recv(4096).decode("utf-8", errors="ignore")
                if out:
                    log.info("[ssh][%s][stdout] %s", hostname if SET.is_docker else port, out)
            if chan.recv_stderr_ready():
                err = chan.recv_stderr(4096).decode("utf-8", errors="ignore")
                if err:
                    log.error("[ssh][%s][stderr] %s", hostname if SET.is_docker else port, err)
            time.sleep(0.1)

        log_path = f"{SET.local_log_dir}/sim_{sim_oid}.log"
        with SCPClient(ssh.get_transport()) as scp:
            remote_log = f"{SET.remote_dir}/COOJA.testlog"
            log.info("[port=%s] Copying log to %s", port, log_path)
            scp.get(remote_log, log_path)

        log_id = mongo.fs_handler.upload_file(log_path, "sim_result.log")
        log.info("[port=%s] Log saved with ID: %s", port, log_id)

        csv_path = f"{SET.local_log_dir}/sim_{sim_oid}.csv"
        cooja_files.convert_cooja_log_to_csv(log_path, csv_path)
        csv_id = mongo.fs_handler.upload_file(csv_path, "sim_result.csv")
        log.info("[port=%s] Log converted CSV and saved with ID: %s", port, csv_id)

        csv_file = Path(csv_path)
        if csv_file.exists() and csv_file.stat().st_size != 0:
            df = pd.read_csv(csv_path)
            exp_id = sim["experiment_id"]
            cfg = mongo.experiment_repo.get_metrics_data_conversion(str(exp_id))
            net_meas = statistics.evaluate_config(df, cfg, log)
            log.info("[port=%s] Metrics calculated: %s", port, net_meas)
            mongo.simulation_repo.mark_done(sim_oid, log_id, csv_id, net_meas)
        else:
            log.warning("[port=%s] CSV file is missing or empty for simulation %s", port, sim_oid)
            mongo.simulation_repo.mark_error(sim_oid, "CSV file is missing or empty after conversion")

        for path in (log_path, csv_path):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as ex:
                log.warning("Failed to remove temp file %s: %s", path, ex)

        if sim.get("generation_id"):
            _check_and_close_generation(sim["generation_id"], mongo)

    except Exception as e:
        log.exception("[port=%s host=%s] Simulation ERROR %s: %s", port, hostname, sim_oid, e)
        try:
            mongo.simulation_repo.mark_error(sim_oid, str(e))
        except Exception:
            log.exception("Failed to mark error on simulation %s", sim_oid)
        if sim.get("generation_id"):
            _check_and_close_generation(sim["generation_id"], mongo)
    finally:
        ssh.close()


def simulation_worker(worker_id: int, sim_queue: queue.Queue, port: int, hostname: str) -> None:
    """
    Worker that consumes simulation IDs from the queue and executes each one.
    The full simulation document is fetched from MongoDB at execution time.
    Simulations not in WAITING status are skipped (already claimed by another worker).
    """
    mongo = create_mongo_repository_factory(SET.mongo_uri, SET.db_name)
    while True:
        sim_id = sim_queue.get()
        sim = None
        try:
            if sim_id is None:
                return

            sim = mongo.simulation_repo.get(sim_id)
            if sim is None:
                log.warning("[port=%s] Simulation %s not found; skipping.", port, sim_id)
                continue

            if sim.get("status") != EnumStatus.WAITING:
                log.info("[port=%s] Simulation %s already in status '%s'; skipping.",
                         port, sim_id, sim.get("status"))
                continue

            mode = os.getenv("ENABLE_DATA_SYNTHETIC", "False").lower() == "true"
            log.info("mode: %s", "Synthetic Data" if mode else "Simulation")
            if mode:
                run_synthetic_simulation(sim, mongo)
                continue

            log.info("[port=%s host=%s] Preparing simulation %s", port, hostname, sim_id)

            success, local_files, remote_files = prepare_simulation_files(sim, worker_id, mongo)
            if not success:
                log.warning("[port=%s] Skipping simulation %s (prepare failed)", port, sim_id)
                try:
                    mongo.simulation_repo.mark_error(ObjectId(sim["_id"]), "Failed to prepare simulation files")
                except Exception:
                    log.exception("Failed to mark error on simulation %s", sim_id)
                if sim.get("generation_id"):
                    _check_and_close_generation(sim["generation_id"], mongo)
                continue

            ssh = create_ssh_client(hostname, port, "root", "root")
            try:
                send_files_scp(ssh, SET.local_dir, SET.remote_dir, local_files, remote_files)
            finally:
                ssh.close()

            run_cooja_simulation(sim, port, hostname, mongo)

            for f in local_files:
                try:
                    Path(f).unlink(missing_ok=True)
                except Exception as ex:
                    log.warning("Failed to remove temp file %s: %s", f, ex)

        except Exception as e:
            log.exception("[port=%s host=%s] General ERROR: %s", port, hostname, e)
            try:
                if sim and "_id" in sim:
                    mongo.simulation_repo.mark_error(ObjectId(sim["_id"]), str(e))
            except Exception:
                log.exception("Failed to mark error on simulation after general exception")
            if sim and sim.get("generation_id"):
                _check_and_close_generation(sim["generation_id"], mongo)
        finally:
            sim_queue.task_done()


def start_workers(num_workers: int) -> queue.Queue:
    _assert_capacity(num_workers)
    q: queue.Queue = queue.Queue()
    for i in range(num_workers):
        t = Thread(
            target=simulation_worker,
            args=(i, q, SET.ports[i], SET.hostnames[i]),
            daemon=True,
        )
        t.start()
    log.info("Workers started: %s", num_workers)
    return q


def load_initial_waiting_jobs(mongo: MongoRepository, sim_queue: queue.Queue) -> None:
    """Enqueue IDs of all WAITING simulations found at startup."""
    log.info("Searching for pending simulations...")
    pending = mongo.simulation_repo.find_pending()
    for sim in pending:
        sim_id = str(sim["_id"])
        log.info("Pending simulation: %s", sim_id)
        sim_queue.put(sim_id)


def watch_status_waiting_enqueue(
    simRepo: SimulationRepository,
    sim_queue: queue.Queue
) -> None:
    """
    Subscribes to the simulations Change Stream and enqueues simulation IDs
    as they enter WAITING status.
    """
    log.info("[mongowatch] Watching for WAITING simulations...")

    def on_simulation_event(change: dict):
        doc = change.get("fullDocument")
        if not doc:
            log.warning("[mongowatch] Event without fullDocument; skipping.")
            return
        sim_id = str(doc["_id"])
        log.info("[mongowatch] Enqueuing simulation %s", sim_id)
        sim_queue.put(sim_id)

    simRepo.watch_status_waiting(on_simulation_event)


def main() -> None:
    number_of_containers = len(SET.hostnames)
    log.info("start")
    log.info("number of containers: %s", number_of_containers)
    log.info("env: MONGO_URI=%s | DB_NAME=%s | IS_DOCKER=%s", SET.mongo_uri, SET.db_name, SET.is_docker)

    mongo = create_mongo_repository_factory(SET.mongo_uri, SET.db_name)

    recover_stuck_simulations(mongo)
    sim_queue = start_workers(number_of_containers)
    load_initial_waiting_jobs(mongo, sim_queue)

    Thread(
        target=watch_status_waiting_enqueue,
        args=(mongo.simulation_repo, sim_queue,),
        daemon=True,
    ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("closing...")


if __name__ == "__main__":
    main()
