import logging
from datetime import datetime
from bson import ObjectId, errors
from pyparsing import Callable

from pylib.dto.database import Batch
from mongo.connection import MongoDBConnection, EnumStatus

log = logging.getLogger(__name__)

class BatchRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["batches"].create_index([("status", 1)], name="idx_batches_status")


    def insert(self, gen: Batch) -> ObjectId:
        with self.connection.connect() as db:
            batch_id = db["batches"].insert_one(gen).inserted_id
            for sim_id in gen["simulations_ids"]:
                db["simulations"].update_one(
                    {"_id": ObjectId(sim_id)},
                    {"$set": {"batch_id": batch_id}}
                )
            return batch_id


    def update(self, batch_id: ObjectId, updates: dict) -> bool:
        updates["id"] = batch_id
        with self.connection.connect() as db:
            result = db["batches"].update_one({"_id": batch_id}, {"$set": updates})
            return result.modified_count > 0
            
            
    def get(self, batch_id: str)->Batch:
        try:
            oid = ObjectId(batch_id)
        except errors.InvalidId:
            log.error("Invalid ID")
        with self.connection.connect() as db:
            result = db["batches"].find_one({"_id": oid})
            return result
        
        
    def find_by_status(self, status: EnumStatus) -> list[Batch]:
        with self.connection.connect() as db:
            return list(db["batches"].find({"status": status}))
        
        
    def find_pending(self) -> list[Batch]:
        return self.find_by_status(EnumStatus.WAITING)
        
        
    def mark_waiting(self, batch_id: ObjectId):
        self.update(batch_id, {"status": EnumStatus.WAITING})
                
                
    def mark_done(self, batch_id: ObjectId):
        self.update(batch_id, {
            "status": EnumStatus.DONE,
            "end_time": datetime.now()
        })


    def mark_error(self, batch_id: ObjectId):
        self.update(batch_id, {
            "status": EnumStatus.ERROR,
            "end_time": datetime.now()
        })


    def watch_status(self, status: EnumStatus, on_change: Callable[[dict], None]):
        log.info("[BatchRepository] Waiting changes...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": status
                }
            }
        ]
        self.connection.watch_collection(
            "batches", 
            pipeline, 
            on_change, 
            full_document="updateLookup"
            )


    def watch_status_done(self, on_change: Callable[[dict], None]):
        self.watch_status(EnumStatus.DONE, on_change)


    def all_simulations_done(self, batch_id: ObjectId) -> bool:
        """
        Checks if all simulations associated with the batch have been completed.

        Criterion:
            count(simulations where status != DONE) == 0
        """

        if not isinstance(batch_id, ObjectId):
            try:
                batch_id = ObjectId(batch_id)
            except errors.InvalidId:
                log.error(f"[BatchRepository] Invalid batch_id: {batch_id}")
                return False

        with self.connection.connect() as db:

            batch = db["batches"].find_one({"_id": batch_id})

            if not batch:
                log.warning(f"[BatchRepository] Batch {batch_id} not found.")
                return False

            sim_ids: list[ObjectId] = batch.get("simulations_ids", [])

            # Batch sem simulações → considerado concluído
            if not sim_ids:
                log.warning(
                    f"[BatchRepository] Batch {batch_id} has no simulations. "
                    "Marking as done by definition."
                )
                return True

            count_not_done = db["simulations"].count_documents({
                "_id": {"$in": sim_ids},
                "status": {"$ne": EnumStatus.DONE}
            })

            return count_not_done == 0


    def any_simulation_error(self, batch_id: ObjectId) -> bool:
        """
        Checks whether any simulation associated with the batch ended with an error.

        Criterion:
            exists(simulation where status == ERROR)
        """

        if not isinstance(batch_id, ObjectId):
            try:
                batch_id = ObjectId(batch_id)
            except errors.InvalidId:
                log.error(f"[BatchRepository] Invalid batch_id: {batch_id}")
                return False

        with self.connection.connect() as db:

            batch = db["batches"].find_one({"_id": batch_id})

            if not batch:
                log.warning(f"[BatchRepository] Batch {batch_id} not found.")
                return False

            sim_ids: list[ObjectId] = batch.get("simulations_ids", [])

            if not sim_ids:
                log.warning(
                    f"[BatchRepository] Batch {batch_id} has no simulations. "
                    "No error possible."
                )
                return False

            count_error = db["simulations"].count_documents({
                "_id": {"$in": sim_ids},
                "status": EnumStatus.ERROR
            })

            return count_error > 0


    def get_simulations_metrics_map(
        self,
        batch_id: ObjectId | str,
        metrics: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Given a batch_id and a list of metric names, returns a map: 
        simulation_id -> { metric_name -> value } 
        Only the requested metrics are returned. 
        Missing metrics are not included in the submap.
        """

        # --- coercion ----------------------------------------------------
        if isinstance(batch_id, str):
            try:
                batch_oid = ObjectId(batch_id)
            except Exception:
                raise ValueError(f"Invalid batch_id: {batch_id}")
        else:
            batch_oid = batch_id

        # --- fetch batch -------------------------------------------------
        with self.connection.connect() as db:

            batch_doc = db["batches"].find_one(
                {"_id": batch_oid},
                {"simulations_ids": 1}
            )

            if not batch_doc:
                raise ValueError(f"Batch {batch_oid} not found")

            sim_ids: list[ObjectId] = batch_doc.get("simulations_ids", [])

            if not sim_ids:
                return {}

            # --- fetch simulations ---------------------------------------
            sims_cursor = db["simulations"].find(
                {"_id": {"$in": sim_ids}},
                {
                    "_id": 1,
                    "network_metrics": 1
                }
            )

            result: dict[str, dict[str, float]] = {}

            for sim in sims_cursor:

                sim_id_str = str(sim["_id"])
                metrics_map = sim.get("network_metrics", {}) or {}

                filtered = {
                    m: metrics_map[m]
                    for m in metrics
                    if m in metrics_map
                }

                result[sim_id_str] = filtered

            return result
