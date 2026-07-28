"""
milp-engine service entry point.

Watches the milp_sweeps collection (change streams, same pattern as the
mo-engine) and executes each Waiting sweep: MILP parameter sweep -> genotype
deduplication -> handoff as a batch experiment for the existing pipeline.

On startup it validates the solver backends (Gurobi license / HiGHS) and
publishes the result to the milp_engine_status collection, exposed by the
REST API at GET /milp/status.
"""
import logging
import os
import sys
import time
from threading import Thread

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib.db import create_mongo_repository_factory, EnumStatus

from lib.runner import SweepRunner
from lib.solver import BACKEND_REGISTRY
from lib.solver.gurobi_backend import check_license

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [milp-engine] %(message)s",
)
log = logging.getLogger("milp-engine")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
DEFAULT_SOLVER = os.getenv("MILP_SOLVER", "gurobi")

mongo = create_mongo_repository_factory(MONGO_URI, DB_NAME)


def publish_engine_status() -> None:
    available = [name for name, cls in BACKEND_REGISTRY.items() if cls.is_available()]
    _, license_msg = check_license()
    status = {
        "status": "online",
        "solver": DEFAULT_SOLVER if DEFAULT_SOLVER in available else (available[0] if available else None),
        "gurobi_license": license_msg,
        "available_backends": available,
    }
    log.info("Solver status: %s", status)
    if not available:
        log.error("No solver backend available — sweeps will fail until one is installed/licensed.")
    mongo.milp_sweep_repo.publish_engine_status(status)


def process_sweep(sweep_doc: dict) -> bool:
    sweep_id = str(sweep_doc["_id"])
    log.info("Processing sweep id: %s", sweep_id)
    try:
        SweepRunner(sweep_doc, mongo).start()
        return True
    except Exception:
        log.exception("Failed to start sweep %s", sweep_id)
        return False


def on_sweep_event(change: dict) -> None:
    sweep_doc = change.get("fullDocument")
    if not sweep_doc:
        log.warning("Document missing from the change event.")
        return
    sweep_id = str(sweep_doc["_id"])
    if mongo.milp_sweep_repo.update_starting(sweep_id):
        if not process_sweep(sweep_doc):
            mongo.milp_sweep_repo.update_status(sweep_id, EnumStatus.ERROR)


def main() -> None:
    log.info("service started.")
    log.info("env:\n\tMONGO_URI: %s\n\tDB_NAME: %s\n\tMILP_SOLVER: %s", MONGO_URI, DB_NAME, DEFAULT_SOLVER)

    repo = mongo.milp_sweep_repo
    repo.connection.waiting_ping()
    publish_engine_status()

    # Startup recovery: resume sweeps orphaned in Running by a previous crash
    # (the per-combination checkpoint makes this safe), then claim Waiting ones.
    for sweep_doc in repo.find_running():
        if not process_sweep(sweep_doc):
            repo.update_status(str(sweep_doc["_id"]), EnumStatus.ERROR)

    for sweep_doc in repo.find_pending():
        sweep_id = str(sweep_doc["_id"])
        if repo.update_starting(sweep_id):
            if not process_sweep(sweep_doc):
                repo.update_status(sweep_id, EnumStatus.ERROR)

    Thread(
        target=repo.watch_status_waiting,
        args=(on_sweep_event,),
        daemon=True,
    ).start()

    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
