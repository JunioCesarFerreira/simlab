import logging
from datetime import datetime
from bson import ObjectId, errors
from typing import Callable, Optional

from pylib.dto.database import Simulation
from mongo.connection import MongoDBConnection, EnumStatus

log = logging.getLogger(__name__)

class SimulationRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["simulations"].create_index([("status", 1)], name="idx_simulations_status")
            db["simulations"].create_index([("experiment_id", 1)], name="idx_simulations_experiment_id")


    def insert(self, simulation: Simulation) -> ObjectId:
        with self.connection.connect() as db:
            return db["simulations"].insert_one(simulation).inserted_id


    def get(self, simulation_id: str)->Simulation:
        try:
            oid = ObjectId(simulation_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", simulation_id)
            return None
        with self.connection.connect() as db:
            result = db["simulations"].find_one({"_id": oid})
            return result


    def get_topology_pic_file_id(self, simulation_id: str) -> ObjectId | None:
        try:
            oid = ObjectId(simulation_id)
        except errors.InvalidId:
            log.error("[get_topology_pic_file_id] Invalid simulation_id")
            return None

        with self.connection.connect() as db:
            result = db["simulations"].find_one(
                {"_id": oid},
                {"_id": 0, "topology_picture_id": 1}
            )

            if not result:
                log.warning(f"[get_topology_pic_file_id] Simulation not found: {simulation_id}")
                return None

            return result.get("topology_picture_id")
        
        
    def find_pending(self) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find({"status": EnumStatus.WAITING}))


    def find_running_before(self, cutoff: datetime) -> list[Simulation]:
        """Returns simulations stuck in RUNNING with start_time older than cutoff."""
        with self.connection.connect() as db:
            return list(db["simulations"].find({
                "status": EnumStatus.RUNNING,
                "start_time": {"$lt": cutoff}
            }))
        
        
    def find_pending_by(self, parent: str, object_id: ObjectId) -> list[Simulation]:
        if parent not in ["experiment_id", "batch_id"]:
            log.error(f"Invalid parent field: {parent}")
            return []
        with self.connection.connect() as db:
            return list(db["simulations"].find(
                {
                    "status": EnumStatus.WAITING,
                    parent: object_id
                }))
            
            
    def find_by_status(self, status: str) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find({"status": status}))
    
    
    def update_status(self, sim_id: str, status: str):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": ObjectId(sim_id)},
                {"$set": {"status": status}}
            )
    
    
    def mark_running(self, sim_id: ObjectId):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "status": EnumStatus.RUNNING, 
                    "start_time": datetime.now(),
                    }
                 }
            )
    
    
    def mark_done(self, 
                  sim_id: ObjectId, 
                  log_id: ObjectId, 
                  csv_id: ObjectId, 
                  network_metrics: dict[str,float]
                  ):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "status": EnumStatus.DONE, 
                    "end_time": datetime.now(),
                    "log_cooja_id": log_id,
                    "csv_log_id": csv_id,
                    "network_metrics": network_metrics,
                    }
                 }
            )
            
    def mark_error(self, sim_id: ObjectId, system_message: str):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "status": EnumStatus.ERROR, 
                    "end_time": datetime.now(),
                    "system_message": system_message
                    }
                 }
            )
    
    
    def delete_by_id(self, simulation_id: str) -> bool:
        try:
            oid = ObjectId(simulation_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return False

        with self.connection.connect() as db:
            result = db["simulations"].delete_one({"_id": oid})
            return result.deleted_count > 0