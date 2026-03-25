import logging
import queue

from pylib.db.repositories.simulation import SimulationRepository

log = logging.getLogger(__name__)

# Watches the simulations collection for WAITING status.
# Each new WAITING simulation has its ID enqueued for a worker to process.


def watch_status_waiting_enqueue(
    simRepo: SimulationRepository,
    sim_queue: queue.Queue
) -> None:
    """
    Subscribes to the simulations Change Stream and enqueues simulation IDs
    as they enter WAITING status.
    """
    log.info("[mongowatch] Watching for WAITING simulations...")

    def on_simulation_event(change: dict):
        doc = change.get("fullDocument")
        if not doc:
            log.warning("[mongowatch] Event without fullDocument; skipping.")
            return
        sim_id = str(doc["_id"])
        log.info("[mongowatch] Enqueuing simulation %s", sim_id)
        sim_queue.put(sim_id)

    simRepo.watch_status_waiting(on_simulation_event)
