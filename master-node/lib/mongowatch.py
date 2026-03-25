import logging
import queue
from datetime import datetime
from bson import ObjectId

from pylib.db.connection import MongoDBConnection
from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.simulation import SimulationRepository
from pylib.db.repositories.generation import GenerationRepository

log = logging.getLogger(__name__)

# Watches the generations collection for WAITING status.
# When a new WAITING generation is detected, fetches its simulations and enqueues them.
# Marks the generation as RUNNING once the simulations are enqueued.


def _make_generation_event_handler(
    simRepo: SimulationRepository,
    genRepo: GenerationRepository,
    sim_queue: queue.Queue
) -> callable:
    def on_generation_event(change: dict):
        log.info("[mongowatch] on generation event...")

        gen_doc = change.get("fullDocument")
        if not gen_doc:
            log.warning("[mongowatch] Document missing from the event.")
            return

        gen_id = ObjectId(gen_doc.get("_id"))

        sims = simRepo.find_pending_by("generation_id", gen_id)
        if not sims:
            log.warning("[mongowatch] Generation %s has no pending simulations.", gen_id)
            return

        log.info("[mongowatch] Enqueuing %d simulation(s) for generation %s", len(sims), gen_id)
        for sim in sims:
            sim_queue.put(sim)

        genRepo.mark_running(gen_id)

    return on_generation_event


def watch_status_waiting_enqueue(
    simRepo: SimulationRepository,
    genRepo: GenerationRepository,
    sim_queue: queue.Queue
) -> None:
    log.info("[mongowatch] Watching for WAITING generations...")
    event_handler = _make_generation_event_handler(simRepo, genRepo, sim_queue)
    genRepo.watch_status_waiting(event_handler)
