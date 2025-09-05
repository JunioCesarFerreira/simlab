import random
import logging
from bson import ObjectId
from pylib import mongo_db
from dto import Simulation

log = logging.getLogger(__name__)
def run_fake_simulation(sim: Simulation, mongo: mongo_db.MongoRepository) -> None:
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]

    log.info("Starting fake simulation %s", sim_oid)
    mongo.simulation_repo.mark_running(sim_oid)

    exp_id = sim["experiment_id"]
    cfg = mongo.experiment_repo.get_objectives_and_metrics(str(exp_id))

    objectives, metrics = {}, {}
    for item in cfg.get("objectives", []):
        name = item["name"]
        objectives[name] = round(random.uniform(0, 100), 2)

    for item in cfg.get("metrics", []):
        name = item["name"]
        metrics[name] = round(random.uniform(0, 100), 2)
            
    mongo.simulation_repo.mark_done(sim_oid, sim_oid, sim_oid, objectives, metrics)

    gen_id = sim["generation_id"]
    if mongo.generation_repo.all_simulations_done(gen_id):
        mongo.generation_repo.mark_done(gen_id)