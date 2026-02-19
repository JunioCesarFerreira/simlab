import logging
from datetime import datetime
from typing import Optional, Callable, Any
from bson import ObjectId, errors

from pylib.dto.database import Experiment, DataConversionConfig, Generation
from mongo.connection import MongoDBConnection, EnumStatus
from mongo.gridfs_handler import MongoGridFSHandler

log = logging.getLogger(__name__)

class ExperimentRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection


    def insert(self, experiment: Experiment) -> ObjectId:
        with self.connection.connect() as db:
            return db["experiments"].insert_one(experiment).inserted_id


    def find_by_status(self, status: EnumStatus) -> list[dict[str,Any]]:
        with self.connection.connect() as db:
            return list(db["experiments"].find(
                {"status": status},
                {"_id": 1, "name": 1, "system_message": 1, "start_time": 1, "end_time": 1}
                ))


    def find_analysis_files(self, experiment_id: str) -> dict[str,Any]:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
        with self.connection.connect() as db:
            result = db["experiments"].find_one(
                {"_id": oid},
                {"analysis_files": 1})
            return result["analysis_files"]
                        

    def update(self, experiment_id: str, updates: dict) -> bool:
        updates["id"] = experiment_id
        with self.connection.connect() as db:
            result = db["experiments"].update_one({"_id": ObjectId(experiment_id)}, {"$set": updates})
            return result.modified_count > 0
        
        
    def update_status(self, experiment_id: str, status: str)->bool:
        return self.update(experiment_id, {"status": status})
            
            
    def update_starting(self, experiment_id: str)->bool:
        success = self.update(experiment_id, {
        "status": EnumStatus.RUNNING,
        "start_time": datetime.now()
        })  
        return success
            
            
    def get(self, experiment_id: str)->Experiment:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
        with self.connection.connect() as db:
            result = db["experiments"].find_one({"_id": oid})
            return result
        
        
    def get_metrics_data_conversion(self, experiment_id: str) -> DataConversionConfig:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            raise ValueError(f"Invalid experiment_id: {experiment_id!r}")

        with self.connection.connect() as db:
            doc = db["experiments"].find_one(
                {"_id": oid},
                {
                    "_id": 0,
                    "data_conversion_config.node_col": 1,
                    "data_conversion_config.time_col": 1,
                    "data_conversion_config.metrics": 1,
                }
            )

        if not doc:
            return {}

        return doc.get("data_conversion_config") or {}
    
    
    def add_generation(self, experiment_id: ObjectId, generation: Generation) -> bool:
        with self.connection.connect() as db:
            result = db["experiments"].update_one(
                {"_id": experiment_id},
                {"$push": {"generations": generation}}
            )
            return result.modified_count > 0
        
        
    def delete(self, experiment_id: str) -> dict[str, int]:
        """
        Delete an experiment by _id and cascade-delete all its simulations.
        Returns counters: {"deleted_experiments": 0|1, "deleted_generations": N, "deleted_simulations": M}.
        """            
        try:
            exp_oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return {"deleted_experiments": 0, "deleted_simulations": 0}

        with self.connection.connect() as db:

            # delete simulations
            sim_del_res = db["simulations"].delete_many(
                {"experiment_id": exp_oid}
            )

            sims_deleted = int(sim_del_res.deleted_count)

            # delete experiment
            exp_del_res = db["experiments"].delete_one(
                {"_id": exp_oid}
            )

            return {
                "deleted_experiments": int(exp_del_res.deleted_count),
                "deleted_simulations": sims_deleted,
            }
            
            
    def add_analysis_file_to_experiment(self, 
            experiment_id: str, 
            description: str,
            path: str, 
            name: str
    ) -> ObjectId | None:
        with self.connection.connect() as db:
            grid = MongoGridFSHandler(self.connection)
            file_id = grid.upload_file(path, name)
            
            result = db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {
                    "$set": {
                        f"analysis_files.{description}": file_id
                    }
                }
            )
            if result.matched_count == 0:
                raise ValueError("Experiment not found")
            
            return file_id
    
    
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