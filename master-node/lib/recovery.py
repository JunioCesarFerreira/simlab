"""Recovering the pool after the master-node restarts.

Before this feature a restart cost nothing: workers held no state, so the
containers were usable again the moment the process came back. Reservations
changed that — a container claimed by a process that is gone stays claimed —
so the equivalent has to be done explicitly, or the pool shrinks by one
container per restart until nothing is left.

Scope and assumption
--------------------
This releases *every* occupied container at startup, on the assumption that
one master-node places work in this deployment, so anything still holding a
container belongs to a previous life of this process.

It is also safe for the simulations themselves, because Cooja is currently
launched attached to its SSH channel: when the master-node died, the channel
closed, and the JVM received SIGHUP. Those runs are genuinely over, so marking
them failed is accurate rather than pessimistic.

Both of those stop being true later. Detached execution (which keeps a run
alive across a restart) and the liveness probe that goes with it will replace
this with reconciliation proper: probe the container, **adopt** a run that is
still going, and release only what is really gone.
"""
import logging
from typing import Any

from bson import ObjectId

from pylib.db.models.enums import EnumStatus

log = logging.getLogger(__name__)

RECOVERY_MESSAGE = (
    "Recovered: the master-node restarted while this simulation was running"
)


def release_orphaned_reservations(mongo: Any) -> int:
    """Free containers held by a previous run, and fail what they were running.

    Returns how many containers were recovered.
    """
    occupied = mongo.resource_repo.find_occupied()
    if not occupied:
        return 0

    log.warning(
        "Found %d container(s) still held from a previous run; recovering...",
        len(occupied),
    )
    recovered = 0
    for resource in occupied:
        name = resource.get("name", "?")
        simulation_id = resource.get("current_simulation_id")
        try:
            _fail_simulation(mongo, simulation_id, name)
            if mongo.resource_repo.force_release(
                resource["_id"], "recovered on master-node restart"
            ):
                recovered += 1
                log.info("Recovered container %s", name)
        except Exception:
            log.exception("Failed to recover container %s", name)
    return recovered


def _fail_simulation(mongo: Any, simulation_id: Any, resource_name: str) -> None:
    """Mark the interrupted simulation failed, and close its generation.

    Skipped when the simulation already reached a terminal state: the run
    finished and only the release was lost, so there is nothing to correct.
    """
    if not simulation_id:
        return
    simulation = mongo.simulation_repo.get(str(simulation_id))
    if simulation is None:
        return
    if simulation.get("status") in (EnumStatus.DONE, EnumStatus.ERROR,
                                    EnumStatus.CANCELLED):
        return

    sim_oid = simulation["_id"]
    if not isinstance(sim_oid, ObjectId):
        sim_oid = ObjectId(sim_oid)
    log.warning(
        "Simulation %s was running on %s when the master-node stopped; "
        "marking it failed", sim_oid, resource_name,
    )
    mongo.simulation_repo.mark_error(sim_oid, RECOVERY_MESSAGE)

    generation_id = simulation.get("generation_id")
    if generation_id:
        _close_generation(mongo, generation_id)


def _close_generation(mongo: Any, generation_id: Any) -> None:
    gr = mongo.generation_repo
    try:
        if gr.any_simulation_active(generation_id):
            return
        if gr.all_simulations_done(generation_id):
            gr.mark_done(generation_id)
        else:
            gr.mark_error(generation_id)
    except Exception:
        log.exception("Failed to close generation %s", generation_id)
