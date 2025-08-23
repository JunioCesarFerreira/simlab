import os
import sys
import time
import queue
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from bson import ObjectId
from paramiko import SSHClient
from scp import SCPClient

# lib master-node
from lib import sshscp

# --------------------------------------------------------------------
# SysPath for common modules
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db  
from dto import Simulation, Experiment, SourceRepository
# --------------------------------------------------------------------


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

        # Defaults
        default_n = int(os.getenv("NUMBER_OF_CONTAINERS", "10"))
        if is_docker:
            hostnames = [f"cooja{i+1}" for i in range(default_n)]
            ports = [22 for _ in range(default_n)]
        else:
            hostnames = ["localhost" for _ in range(default_n)]
            ports = [2231 + i for i in range(default_n)]

        sim_timeout_sec = int(os.getenv("SIM_TIMEOUT_SEC", "0"))  # 0 = sem timeout

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


def prepare_simulation_files(
    sim: Simulation,
    mongo: mongo_db.MongoRepository,
) -> tuple[bool, list[str], list[str]]:
    """
    Baixa XML/positions da simulação e os arquivos do SourceRepository.
    Retorna (success, local_files, remote_files).
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    local_xml = tmp_dir / f"simulation_{sim_oid}.xml"
    local_dat = tmp_dir / f"positions_{sim_oid}.dat"

    local_files: list[str] = []
    remote_files: list[str] = []

    # CSC/XML
    mongo.fs_handler.download_file(sim["csc_file_id"], str(local_xml))
    local_files.append(str(local_xml))
    remote_files.append("simulation.csc")

    # Positions (opcional)
    if sim.get("pos_file_id"):
        mongo.fs_handler.download_file(sim["pos_file_id"], str(local_dat))
        local_files.append(str(local_dat))
        remote_files.append("positions.dat")

    # Source repository
    exp: Experiment = mongo.experiment_repo.get_by_id(sim["experiment_id"])
    if not exp:
        log.warning("Experiment not found for sim %s", sim_oid)
        return False, local_files, remote_files

    src_repo_id = exp.get("source_repository_id")
    src: SourceRepository = mongo.source_repo.get_by_id(src_repo_id)
    if not src or "source_files" not in src:
        log.warning("Source repository %s not found for experiment %s", src_repo_id, exp.get("_id"))
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
    mongo: mongo_db.MongoRepository,
) -> None:
    """
    Executa a simulação no container via SSH, acompanha logs e envia resultado ao GridFS.
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    ssh: SSHClient = sshscp.create_ssh_client(hostname, port, "root", "root")
    try:
        log.info("[port=%s host=%s] Starting simulation %s", port, hostname, sim_oid)
        mongo.simulation_repo.mark_running(sim_oid)

        # Comando Cooja (sem '../' porque remote_dir é absoluto)
        command = (
            f"cd {SET.remote_dir} && "
            f"/opt/java/openjdk/bin/java --enable-preview -Xms4g -Xmx4g "
            f"-jar build/libs/cooja.jar --no-gui simulation.csc"
        )

        # Execução remota
        stdin, stdout, stderr = ssh.exec_command(command, get_pty=True, timeout=SET.sim_timeout_sec or None)

        # Leitura não-bloqueante simples (polling)
        chan = stdout.channel
        chan.settimeout(5.0)
        while not chan.exit_status_ready():
            if chan.recv_ready():
                out = chan.recv(4096).decode("utf-8", errors="ignore")
                if out:
                    print(f"[ssh][{hostname if SET.is_docker else port}][stdout] {out}", end="")
            if chan.recv_stderr_ready():
                err = chan.recv_stderr(4096).decode("utf-8", errors="ignore")
                if err:
                    print(f"[ssh][{hostname if SET.is_docker else port}][stderr] {err}", end="")
            time.sleep(0.1)

        # Coleta de log principal
        log_path = f"{SET.local_log_dir}/sim_{sim_oid}.log"
        with SCPClient(ssh.get_transport()) as scp:
            # ajuste o nome do arquivo se necessário
            remote_log = f"{SET.remote_dir}/COOJA.testlog"
            log.info("[port=%s] Copying log to %s", port, log_path)
            scp.get(remote_log, log_path)

        log_id = mongo.fs_handler.upload_file(log_path, "sim_result.log")
        log.info("[port=%s] Log saved with ID: %s", port, log_id)

        mongo.simulation_repo.mark_done(sim_oid, log_id)

        if SET.is_docker:
            try:
                Path(log_path).unlink(missing_ok=True)
            except Exception as ex:
                log.warning("Failed to remove temp log %s: %s", log_path, ex)

        # Finalização de geração
        gen_id = sim["generation_id"]
        if mongo.generation_repo.all_simulations_done(gen_id):
            mongo.generation_repo.mark_done(gen_id)

    except Exception as e:
        log.exception("[port=%s host=%s] Simulation ERROR %s: %s", port, hostname, sim_oid, e)
        try:
            mongo.simulation_repo.mark_error(sim_oid)
        except Exception:
            log.exception("Failed to mark error on simulation %s", sim_oid)
    finally:
        ssh.close()


def simulation_worker(sim_queue: queue.Queue, port: int, hostname: str) -> None:
    """
    Worker que consome a fila e executa simulações em um host/porta.
    """
    mongo = mongo_db.create_mongo_repository_factory(SET.mongo_uri, SET.db_name)
    while True:
        sim = sim_queue.get()
        try:
            if sim is None:
                return

            sim_id_str = str(sim.get("_id"))
            log.info("[port=%s host=%s] Preparing simulation %s", port, hostname, sim_id_str)

            success, local_files, remote_files = prepare_simulation_files(sim, mongo)
            if not success:
                log.warning("[port=%s] Skipping simulation %s (prepare failed)", port, sim_id_str)
                mongo.simulation_repo.mark_error(ObjectId(sim["_id"]))
                continue

            # Envio de arquivos
            log.debug("[port=%s] Creating SSH client %s@%s:%s", port, "root", hostname, port)
            ssh = sshscp.create_ssh_client(hostname, port, "root", "root")
            try:
                sshscp.send_files_scp(ssh, SET.local_dir, SET.remote_dir, local_files, remote_files)
            finally:
                ssh.close()

            # Execução
            run_cooja_simulation(sim, port, hostname, mongo)

            # Limpeza local
            if SET.is_docker:
                for f in local_files:
                    try:
                        Path(f).unlink(missing_ok=True)
                    except Exception as ex:
                        log.warning("Failed to remove temp file %s: %s", f, ex)

        except Exception as e:
            log.exception("[port=%s host=%s] General ERROR: %s", port, hostname, e)
            try:
                if sim and "_id" in sim:
                    mongo.simulation_repo.mark_error(ObjectId(sim["_id"]))
            except Exception:
                log.exception("Failed to mark error on simulation after general exception")
        finally:
            # Sempre dar baixa na tarefa
            sim_queue.task_done()


def start_workers(num_workers: int) -> queue.Queue:
    """
    Inicializa threads worker de acordo com a capacidade.
    """
    _assert_capacity(num_workers)

    q: queue.Queue = queue.Queue()
    for i in range(num_workers):
        t = Thread(
            target=simulation_worker,
            args=(q, SET.ports[i], SET.hostnames[i]),
            daemon=True,
        )
        t.start()
    log.info("Workers started: %s", num_workers)
    return q


def load_initial_waiting_jobs(mongo: mongo_db.MongoRepository, sim_queue: queue.Queue) -> None:
    """
    Enfileira simulações pendentes no arranque.
    """
    log.info("Searching for pending simulations...")
    pending = mongo.simulation_repo.find_pending()
    for sim in pending:
        log.info("Pending simulation: %s", sim.get("_id"))
        sim_queue.put(sim)


def main() -> None:
    number_of_containers = len(SET.hostnames)
    log.info("start")
    log.info("number of containers: %s", number_of_containers)
    log.info("env: MONGO_URI=%s | DB_NAME=%s | IS_DOCKER=%s", SET.mongo_uri, SET.db_name, SET.is_docker)

    mongo = mongo_db.create_mongo_repository_factory(SET.mongo_uri, SET.db_name)

    sim_queue = start_workers(number_of_containers)
    load_initial_waiting_jobs(mongo, sim_queue)

    # watch em thread separada
    Thread(
        target=mongo.generation_repo.watch_generations,
        args=(sim_queue,),
        daemon=True,
    ).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Encerrando...")


if __name__ == "__main__":
    main()
