import queue
from datetime import datetime
from bson import ObjectId, errors

from dto import Simulation, Generation
from mongo.connection import MongoDBConnection, EnumStatus

class GenerationRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def insert(self, gen: Generation) -> ObjectId:
        with self.connection.connect() as db:
            return db["generations"].insert_one(gen).inserted_id

    def update(self, generation_id: ObjectId, updates: dict) -> bool:
        updates["id"] = generation_id
        with self.connection.connect() as db:
            result = db["generations"].update_one({"_id": generation_id}, {"$set": updates})
            return result.modified_count > 0
            
    def get_by_id(self, generation_id: str)->Generation:
        try:
            oid = ObjectId(generation_id)
        except errors.InvalidId:
            print("ID inválido")
        with self.connection.connect() as db:
            result = db["generations"].find_one({"_id": oid})
            return result
        
    def find_pending(self) -> list[Generation]:
        with self.connection.connect() as db:
            return list(db["generations"].find({"status": EnumStatus.WAITING}))
        
    def find_by_status(self, status: str) -> list[Generation]:
        with self.connection.connect() as db:
            return list(db["generations"].find({"status": status}))
        
    def mark_waiting(self, generation_id: ObjectId):
            with self.connection.connect() as db:
                db["generations"].update_one(
                    {"_id": generation_id},
                    {"$set": {"status": EnumStatus.WAITING}}
                )
                
    def mark_done(self, generation_id: ObjectId):
        with self.connection.connect() as db:
            db["generations"].update_one(
                {"_id": generation_id},
                {"$set": {"status": EnumStatus.DONE, "end_time": datetime.now()}}
            )
    
    def _find_pending_by_generation(self, gen_id: ObjectId) -> list[Simulation]:
        with self.connection.connect() as db:
            return list(db["simulations"].find(
                {
                    "status": EnumStatus.WAITING,
                    "generation_id": gen_id
                }))
    
    def update_status(self, sim_id: str, status: str):
        with self.connection.connect() as db:
            db["generations"].update_one(
                {"_id": ObjectId(sim_id)},
                {"$set": {"status": status}}
            )        
            
    def _make_generation_event_handler(self, sim_queue: queue.Queue) -> callable:
        def on_generation_event(change: dict):
            print("[GenerationRepository] on generation event...")
            print(f"[GenerationRepository] change: {change}")

            gen_doc = change.get("fullDocument")
            if not gen_doc:
                print("[GenerationRepository] Document missing from the event.")
                return
            gen_id = ObjectId(gen_doc["_id"])
            
            list_sim = self._find_pending_by_generation(gen_id)
            print(f"[GenerationRepository] len(list_sim)={len(list_sim)}")
            for sim in list_sim:
                print(f"[GenerationRepository] enqueue sim_id: {sim["_id"]}")
                sim_queue.put(sim)
            
            if len(list_sim) > 0:
                self.update(gen_id, {
                    "status": EnumStatus.RUNNING,
                    "start_time": datetime.now()
                    })    
        return on_generation_event

    def watch_generations(self, sim_queue: queue.Queue) -> None:
        print("[GenerationRepository] Waiting new generations...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": "Waiting"
                }
            }
        ]
        event_handler = self._make_generation_event_handler(sim_queue)
        self.connection.watch_collection(
            "generations", 
            pipeline, 
            event_handler, 
            full_document="updateLookup"
        )

    def all_simulations_done(self, generation_id: ObjectId) -> bool:
        with self.connection.connect() as db:
            generation = db["generations"].find_one({ "_id": generation_id })
            if not generation:
                print(f"[GenerationRepository] Generation {generation_id} not found.")
                return False

            sim_ids = generation.get("simulations_ids", [])
            if not sim_ids:
                print(f"[GenerationRepository] Generation {generation_id} does not have simulations.")
                return True 
            
            # Conta quantas simulações ainda NÃO estão com status Done
            count_not_done = db["simulations"].count_documents({
                "_id": { "$in": sim_ids },
                "status": { "$ne": "Done" }
            })
            return count_not_done == 0
