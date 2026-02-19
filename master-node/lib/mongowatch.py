import logging
import queue
from datetime import datetime
from bson import ObjectId

from mongo.connection import MongoDBConnection, EnumStatus
from mongo.simulation import SimulationRepository

log = logging.getLogger(__name__)

#------------------------------------------------------------------------------------------------------------------------
# This is an extension of the BatchRepository. 
# However, since it is intrinsically context-specific, it is implemented here rather than within the repository itself. 
# It is responsible for monitoring the batches collection and enqueueing simulations when a batch is marked as waiting. 
# It also updates the batch status to running once simulation processing begins.
#------------------------------------------------------------------------------------------------------------------------

def _make_generation_event_handler(simRepo: SimulationRepository, sim_queue: queue.Queue) -> callable:
    def on_generation_event(change: dict):
        log.info("[BatchRepository] on batch event...")
        log.info(f"[BatchRepository] change: {change}")

        batch_doc = change.get("fullDocument")
        if not batch_doc:
            log.warning("[BatchRepository] Document missing from the event.")
            return
        sims = list(batch_doc["simulations_ids"])
        
        list_sim = [ObjectId(sim_id) for sim_id in sims]
        log.info(f"[BatchRepository] len(list_sim)={len(list_sim)}")
        for sim_id in list_sim:
            log.info(f"[BatchRepository] enqueue sim_id: {sim_id}")
            sim_queue.put(simRepo.get(str(sim_id)))
        
        if len(list_sim) > 0:
            updates: dict = {
                "status": EnumStatus.RUNNING,
                "start_time": datetime.now()
                }
            _id = ObjectId(batch_doc.get("_id"))
            with simRepo.connection.connect() as db:
                result = db["batches"].update_one({"_id": _id}, {"$set": updates})
                return result.modified_count > 0
            
    return on_generation_event


def watch_status_waiting_enqueue(simRepo: SimulationRepository, sim_queue: queue.Queue) -> None:
    log.info("[BatchRepository] Waiting new batches...")
    connection: MongoDBConnection = simRepo.connection
    pipeline = [
        {
            "$match": {
                "operationType": {"$in": ["insert", "update", "replace"]},
                "fullDocument.status": "Waiting"
            }
        }
    ]
    event_handler = _make_generation_event_handler(simRepo, sim_queue)
    connection.watch_collection(
        "batches", 
        pipeline, 
        event_handler, 
        full_document="updateLookup"
    )