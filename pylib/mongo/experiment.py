from datetime import datetime
from typing import Optional, Callable
from bson import ObjectId, errors

from mongo.connection import MongoDBConnection, EnumStatus
from dto import Experiment, TransformConfig

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
            print("ID inválido")
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
    
    
    def watch_experiments(self, on_change: Callable[[dict], None]):
        print("[ExperimentRepository] Waiting new experiments...")
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