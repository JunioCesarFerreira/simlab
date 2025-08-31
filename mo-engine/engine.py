import os, sys, time
from threading import Thread

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db
from pylib.mongo_db import EnumStatus
from strategy.generator_random import GeneratorRandomStrategy
from strategy.nsga3 import NSGA3LoopStrategy  # futuro

SimStatus = mongo_db.EnumStatus
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")

IS_DOCKER = os.getenv("IS_DOCKER", False)

DB_NAME = os.getenv("DB_NAME", "simlab")

mongo = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

def select_strategy(exp_doc: dict):
    print("[mo-engine] select strategy")
    exp_type = exp_doc.get("parameters", {}).get("strategy", "simple")
    print(f"[mo-engine] selected: {exp_type}")
    if exp_type == "simple":
        return GeneratorRandomStrategy(exp_doc, mongo)
    elif exp_type == "nsga3":
        return NSGA3LoopStrategy(exp_doc, mongo)
    else:
        raise ValueError(f"[mo-engine] Experiment type unknown: {exp_type}")


def process_experiment(exp_doc: dict):
    exp_id = str(exp_doc["_id"])
    print(f"[mo-engine] Processing experiment id: {exp_id}")
    try:
        strategy = select_strategy(exp_doc)
        strategy.start()
    except Exception as e:
        print(f"[Erro] Failed to start strategy for experiment {exp_id}: {e}")


def on_experiment_event(change: dict):
    print("[mo-engine] on experiment event...")
    print(f"[mo-engine] change: {change}")

    exp_doc = change.get("fullDocument")
    if not exp_doc:
        print("[mo-engine] Document missing from the event.")
        return
    exp_id = str(exp_doc["_id"])
        
    if mongo.experiment_repo.update_starting(exp_id):
        process_experiment(exp_doc)


def run_pending_experiment(change: dict):
    print("[mo-engine] run pending experiment...")
    print(f"[mo-engine] change: {change}")

    exp_id = str(change["_id"])
    
    if mongo.experiment_repo.update_starting(exp_id):
        process_experiment(change)
       

if __name__ == "__main__":
    print("[mo-engine] Service started.", flush=True)
    print(f"[mo-engine] env:\n\tMONGO_URI: {MONGO_URI}\n\tDB_NAME: {DB_NAME}")
    exp_repo = mongo.experiment_repo
    exp_repo.connection.waiting_ping()

    pending = exp_repo.find_by_status(EnumStatus.WAITING)

    while (len(pending) > 0):
        run_pending_experiment(pending.pop())

    Thread(
        target=exp_repo.watch_experiments, 
        args=(on_experiment_event,),
        daemon=True).start()

    while True:
        time.sleep(10)
