import logging
from datetime import datetime
from typing import Optional, Callable
from bson import ObjectId, errors

from mongo.connection import MongoDBConnection, EnumStatus
from dto import Experiment, TransformConfig

log = logging.getLogger(__name__)

class ExperimentRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def insert(self, experiment: Experiment) -> ObjectId:
        with self.connection.connect() as db:
            return db["experiments"].insert_one(experiment).inserted_id

    def find_by_status(self, status: EnumStatus) -> list[Experiment]:
        with self.connection.connect() as db:
            return list(db["experiments"].find({"status": status}))

    def find_first_by_status(self, status: str) -> Optional[Experiment]:
        with self.connection.connect() as db:
            return db["experiments"].find_one({"status": status})

    def update(self, experiment_id: str, updates: dict) -> bool:
        updates["id"] = experiment_id
        with self.connection.connect() as db:
            result = db["experiments"].update_one({"_id": ObjectId(experiment_id)}, {"$set": updates})
            return result.modified_count > 0
        
    def update_status(self, sim_id: str, status: str):
        with self.connection.connect() as db:
            db["experiments"].update_one(
                {"_id": ObjectId(sim_id)},
                {"$set": {"status": status}}
            )    
            
    def update_starting(self, exp_id: str)->bool:
        success = self.update(exp_id, {
        "status": EnumStatus.RUNNING,
        "start_time": datetime.now()
        })  
        return success
            
    def get_by_id(self, experiment_id: str)->Experiment:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
        with self.connection.connect() as db:
            result = db["experiments"].find_one({"_id": oid})
            return result
        
    def get_objectives_and_metrics(self, experiment_id: str) -> TransformConfig:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            raise ValueError(f"Invalid experiment_id: {experiment_id!r}")

        with self.connection.connect() as db:
            doc = db["experiments"].find_one(
                {"_id": oid},
                {
                    "_id": 0,
                    "transform_config.objectives": 1,
                    "transform_config.metrics": 1,
                }
            )

        if not doc:
            return {}

        return doc.get("transform_config") or {}
    
    def add_generation(self, exp_id: ObjectId, gen_id: ObjectId) -> bool:
        with self.connection.connect() as db:
            result = db["experiments"].update_one(
                {"_id": exp_id},
                {"$push": {"generations": gen_id}}
            )
            return result.modified_count > 0
        
    def delete_by_id(self, experiment_id: str) -> dict[str, int]:
        """
        Delete an experiment by _id and cascade-delete all its generations and simulations.
        Returns counters: {"deleted_experiments": 0|1, "deleted_generations": N, "deleted_simulations": M}.
        """
        try:
            exp_oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return {"deleted_experiments": 0, "deleted_generations": 0, "deleted_simulations": 0}

        def _coerce_oid(x):
            if isinstance(x, ObjectId):
                return x
            try:
                return ObjectId(str(x))
            except Exception:
                return None

        with self.connection.connect() as db:
            # 1) Collect generations linked to the experiment
            gen_docs = list(db["generations"].find({"experiment_id": exp_oid}, {"_id": 1}))
            gen_ids = [doc["_id"] for doc in gen_docs]

            exp_doc = db["experiments"].find_one({"_id": exp_oid}, {"generations": 1})
            if exp_doc and isinstance(exp_doc.get("generations"), list):
                for g in exp_doc["generations"]:
                    go = _coerce_oid(g)
                    if go and go not in gen_ids:
                        gen_ids.append(go)

            # 2) delete simulations
            sim_filter = {"experiment_id": exp_oid}
            if gen_ids:
                sim_filter = {"$or": [{"generation_id": {"$in": gen_ids}}, {"experiment_id": exp_oid}]}

            sim_del_res = db["simulations"].delete_many(sim_filter)
            sims_deleted = int(sim_del_res.deleted_count)

            # 3) delete generations
            gens_deleted = 0
            if gen_ids:
                gen_del_res = db["generations"].delete_many({"_id": {"$in": gen_ids}})
                gens_deleted = int(gen_del_res.deleted_count)

            # 4) delete experiment
            exp_del_res = db["experiments"].delete_one({"_id": exp_oid})
            exps_deleted = int(exp_del_res.deleted_count)

            return {
                "deleted_experiments": exps_deleted,
                "deleted_generations": gens_deleted,
                "deleted_simulations": sims_deleted,
            }
    
    def watch_status_waiting(self, on_change: Callable[[dict], None]):
        log.info("[ExperimentRepository] Waiting new experiments...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": EnumStatus.WAITING
                }
            }
        ]
        self.connection.watch_collection(
            "experiments", 
            pipeline, 
            on_change, 
            full_document="updateLookup"
            )