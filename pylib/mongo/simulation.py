from datetime import datetime
from bson import ObjectId, errors
from typing import Callable

from dto import Simulation
from mongo.connection import MongoDBConnection, EnumStatus

class SimulationRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def insert(self, simulation: Simulation) -> ObjectId:
        with self.connection.connect() as db:
            return db["simulations"].insert_one(simulation).inserted_id

    def get_by_id(self, simulation_id: str)->Simulation:
        try:
            oid = ObjectId(simulation_id)
        except errors.InvalidId:
            print("ID inválido")
        with self.connection.connect() as db:
            result = db["simulations"].find_one({"_id": oid})
            return result

    def find_pending(self) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find({"status": EnumStatus.WAITING}))
        
    def find_pending_by_generation(self, gen_id: ObjectId) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find(
                {
                    "status": EnumStatus.WAITING,
                    "generation_id": gen_id
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
                  objectives: dict[str,float],
                  metrics: dict[str, float]
                  ):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "status": EnumStatus.DONE, 
                    "end_time": datetime.now(),
                    "log_cooja_id": log_id,
                    "csv_log_id": csv_id,
                    "objectives": objectives,
                    "metrics": metrics
                    }
                 }
            )
            
    def mark_error(self, sim_id: ObjectId):
        with self.connection.connect() as db:
            db["simulations"].update_one(
                {"_id": sim_id},
                {"$set": {
                    "status": EnumStatus.ERROR, 
                    "end_time": datetime.now(),
                    }
                 }
            )
        
    def watch_simulations(self, on_change: Callable[[dict], None]):
        print("[SimulationRepository] Waiting changes...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": EnumStatus.DONE
                }
            }
        ]
        self.connection.watch_collection(
            "simulations", 
            pipeline, 
            on_change, 
            full_document="updateLookup"
            )