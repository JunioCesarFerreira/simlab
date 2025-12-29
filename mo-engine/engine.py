import os, sys, time, logging
from threading import Thread

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db
from pylib.mongo_db import EnumStatus
from lib.strategy.base import EngineStrategy 
from lib.strategy.generator_random import GeneratorRandomStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy  

# --------------------------- Logging --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [mo-engine] %(message)s",
)
log = logging.getLogger("mo-engine")
# --------------------------------------------------------------------

SimStatus = mongo_db.EnumStatus
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
IS_DOCKER = os.getenv("IS_DOCKER", False)
DB_NAME = os.getenv("DB_NAME", "simlab")

mongo = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

def select_strategy(exp_doc: dict) -> EngineStrategy:
    log.info("select strategy")
    exp_type = exp_doc.get("parameters", {}).get("strategy", "simple")
    log.info(f"selected: {exp_type}")
    if exp_type == "simple":
        return GeneratorRandomStrategy(exp_doc, mongo)
    elif exp_type == "nsga3":
        return NSGA3LoopStrategy(exp_doc, mongo)
    else:
        raise ValueError(f"[mo-engine] Experiment type unknown: {exp_type}")


def process_experiment(exp_doc: dict) -> bool:
    exp_id = str(exp_doc["_id"])
    log.info(f"Processing experiment id: {exp_id}")
    try:
        strategy = select_strategy(exp_doc)
        strategy.start()
        return True
    except Exception:
        log.exception("Failed to start strategy for experiment %s", exp_id)
        return False


def on_experiment_event(change: dict):
    log.info("on experiment event...")
    log.info(f"change: {change}")

    exp_doc = change.get("fullDocument")
    if not exp_doc:
        log.warning("Document missing from the event.")
        return
    exp_id = str(exp_doc["_id"])
        
    if mongo.experiment_repo.update_starting(exp_id):
        if process_experiment(exp_doc) == False:
            mongo.experiment_repo.update_status(exp_id, EnumStatus.ERROR)


def run_pending_experiment(change: dict):
    log.info("run pending experiment...")
    log.info(f"change: {change}")

    exp_id = str(change["_id"])
    
    if mongo.experiment_repo.update_starting(exp_id):
        if process_experiment(change) == False:
            mongo.experiment_repo.update_status(exp_id, EnumStatus.ERROR)
       
def main() -> None:
    log.info("service started.")
    log.info(f"env:\n\tMONGO_URI: {MONGO_URI}\n\tDB_NAME: {DB_NAME}")
    exp_repo = mongo.experiment_repo
    exp_repo.connection.waiting_ping()

    pending = exp_repo.find_by_status(EnumStatus.WAITING)

    while (len(pending) > 0):
        run_pending_experiment(pending.pop())

    Thread(
        target=exp_repo.watch_status_waiting, 
        args=(on_experiment_event,),
        daemon=True).start()

    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()
