from typing import TypedDict, Optional, NotRequired
from datetime import datetime

from pylib.config.simulator import SimulationConfig


class PlacementDto(TypedDict, total=False):
    """Where one simulation ran, as the master-node recorded it.

    Written at reservation time, before anything starts, so a finished
    simulation can still say where it executed after the host registration is
    edited or removed — which is why the names are stored alongside the ids
    rather than looked up later.

    It was persisted and then never published: the chain experiment →
    generation → individual → simulation stopped short of the machine, so an
    operator comparing two results had no way to tell whether they ran on the
    same worker, and the distribution of work across hosts could not be
    audited at all.
    """
    host_id: Optional[str]
    host_name: str
    resource_id: Optional[str]
    resource_name: str
    #: Docker's id for the container, when it was discovered rather than
    #: configured by hand. The worker's name can be reused; this cannot.
    container_id: Optional[str]
    #: The placement attempt. Two attempts of the same simulation ran in
    #: different conditions, and this is what tells them apart.
    attempt_id: Optional[str]
    assigned_at: Optional[datetime]
    remote_pid: Optional[int]


class SimulationDto(TypedDict):
    id: str
    experiment_id: str
    generation_id: str
    individual_id: str
    status: str
    system_message: str
    random_seed: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    parameters: SimulationConfig
    pos_file_id: str
    csc_file_id: str
    source_repository_id: str
    log_cooja_id: str
    runtime_log_id: str
    csv_log_id: str
    network_metrics: dict[str, float]
    # Populated by the master-node after a run (RPL DODAG formation + tree);
    # optional on create so existing simulation payloads remain valid.
    dodag: NotRequired[Optional[dict]]
    #: Where it ran. Absent on a simulation that was never placed, and on
    #: those recorded before the master-node started writing it down.
    placement: NotRequired[Optional[PlacementDto]]
