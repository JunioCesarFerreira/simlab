"""Infrastructure: hosts, Cooja containers, and the pool's occupancy.

What this API deliberately cannot do
------------------------------------
It cannot reach the containers — it is not on their network — and it cannot
open a stored credential, because the private key lives with the master-node
and this process has no crypto library at all. So anything that needs either
is *requested* rather than performed: a check or a discovery is recorded and
answered 202, and the master-node, which watches these documents, carries it
out and writes back what it observed.

That is not a limitation worked around; it is what keeps the API outside the
trust boundary. A credential travels sealed from the browser, is stored
sealed, and is opened exactly once, by the one process that runs simulations.
"""
import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Response, status
from bson import errors as bson_errors

from pylib.crypto import MalformedEnvelope
from pylib.db import MongoRepository
from pylib.db.models.infra import OCCUPIED_STATUSES, EnumAuthMethod
from pylib.db.repositories.infra import HostWithdrawing, RemovalConflicted

from api.dependencies import get_factory
from api.mappers.topology import topology_from_documents
from pylib.db.models.enums import EnumStatus
from pylib.timeutil import utc_now

# Where Cooja lives inside the stock image; a worker may override it.
DEFAULT_REMOTE_DIR = "/opt/contiki-ng/tools/cooja"
# The write DTOs (HostCreateDto, HostUpdateDto, ResourceEnabledDto) stay in
# api.domain.infra as the documented shape of each body, and are referenced by
# name in the Body() descriptions. They are deliberately not used as
# parameter types — see create_host for why.
from api.domain.infra import (
    AcceptedDto,
    HostDetailDto,
    HostDto,
    MachineDto,
    OverviewDto,
    PublicKeyDto,
    ResourceDto,
)
from api.schemas.infra import (
    HOST_CREATE,
    HOST_UPDATE,
    RESOURCE_CREATE,
    RESOURCE_ENABLED,
    RESOURCE_UPDATE,
    request_body,
)
from api.mappers.infra import (
    PlaintextRefused,
    as_bool,
    as_max_concurrent,
    dotted_update,
    host_from_mongo,
    host_to_mongo,
    machines_from_snapshot,
    occupancy_from_snapshot,
    overview_from_snapshot,
    public_key_from_mongo,
    resource_from_mongo,
    resource_to_mongo,
)

log = logging.getLogger(__name__)

router = APIRouter()

_OCCUPIED = [s.value for s in OCCUPIED_STATUSES]

#: Host fields a worker inherits. Changing one of them invalidates what was
#: measured about every worker on that host.
_ACCESS_BLOCKS = {"access", "container_access"}


# ── What the browser seals against ───────────────────────────────────────────

@router.get("/crypto/public-key", response_model=PublicKeyDto)
def get_public_key(
    factory: MongoRepository = Depends(get_factory),
) -> PublicKeyDto:
    """The master-node's public key, for sealing a credential in the browser.

    Published by the master-node at startup. Its absence means the master-node
    has not run yet in managed mode, and is reported as such rather than as a
    missing document — the fix is to start it, not to retry.
    """
    doc = factory.infra_crypto_repo.get_active()
    if not doc or not doc.get("public_key"):
        raise HTTPException(
            status_code=503,
            detail=(
                "No infrastructure public key has been published yet. The "
                "master-node publishes it on startup; "
                "start it once and retry."
            ),
        )
    return public_key_from_mongo(doc)


# ── Overview ─────────────────────────────────────────────────────────────────

@router.get("/overview", response_model=OverviewDto)
def get_overview(
    factory: MongoRepository = Depends(get_factory),
) -> OverviewDto:
    """Pool occupancy, from the same snapshot the scheduler reads.

    Sharing the query is the point: what an operator sees is what the
    placement decision used, rather than a second count that can disagree.
    """
    try:
        snapshot = factory.host_repo.inventory_snapshot()
        builds = [
            (r.get("probe") or {}).get("cooja_jar_sha256")
            for r in factory.resource_repo.find_all()
        ]
        return overview_from_snapshot(
            snapshot, builds, host_documents=factory.host_repo.find_all())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Physical machines ────────────────────────────────────────────────────────

def _machine_limits(factory: MongoRepository) -> dict[str, int]:
    repo = getattr(factory, "machine_repo", None)
    limits = repo.limits() if repo is not None else {}
    return limits if isinstance(limits, dict) else {}


@router.get("/machines", response_model=list[MachineDto])
def list_machines(factory: MongoRepository = Depends(get_factory)) -> list[MachineDto]:
    """The physical machines behind the hosts, with their limits.

    Several hosts can be one machine — VMs on one hypervisor — and then they
    compete for its CPUs. A limit set here caps the simulations the machine
    runs at once, across all its hosts, whatever the placement policy.
    """
    return machines_from_snapshot(
        factory.host_repo.inventory_snapshot(), _machine_limits(factory))


@router.put("/machines/{machine:path}", response_model=MachineDto)
def set_machine_limit(
    machine: str,
    body: dict = Body(..., description=(
        '`{"max_concurrent": <whole number from 1, or null>}`. Null removes '
        "the limit. Applies from the next placement.")),
    factory: MongoRepository = Depends(get_factory),
) -> MachineDto:
    """Set or clear how many simulations one machine may run at once."""
    if not isinstance(body, dict) or set(body) != {"max_concurrent"}:
        raise HTTPException(
            status_code=422, detail='send exactly {"max_concurrent": <int or null>}')
    try:
        limit = as_max_concurrent(body["max_concurrent"])
    except PlaintextRefused as e:
        raise HTTPException(status_code=422, detail=str(e))

    machines = {m["machine"]: m for m in machines_from_snapshot(
        factory.host_repo.inventory_snapshot(), _machine_limits(factory))}
    if machine not in machines:
        # A typo would otherwise create a limit that nothing ever matches.
        raise HTTPException(
            status_code=404,
            detail=f"no host runs on a machine called {machine!r}; known: "
                   + ", ".join(sorted(machines)))
    factory.machine_repo.set_limit(machine, limit)
    return {**machines[machine], "max_concurrent": limit}


# ── Hosts ────────────────────────────────────────────────────────────────────

@router.get("/topology")
def get_topology(factory: MongoRepository = Depends(get_factory)) -> dict:
    """The pool as a graph — master-node, hosts, workers — and what runs where.

    Built for a page that polls while experiments run, so it is a fixed
    number of queries however large the pool is: every host, every worker,
    then the simulations occupying workers and their experiments and
    generations in one ``$in`` each. Asking each worker separately was one
    request per container per refresh.
    """
    hosts = factory.host_repo.find_all()
    resources = factory.resource_repo.find_all()
    schedulable = {
        str(row["_id"]): bool(row.get("schedulable"))
        for row in factory.host_repo.inventory_snapshot()
    }
    occupied = {s.value for s in OCCUPIED_STATUSES}
    sim_ids = [r["current_simulation_id"] for r in resources
               if r.get("status") in occupied and r.get("current_simulation_id")]

    with factory.simulation_repo.connection.connect() as db:
        simulations = list(db["simulations"].find(
            {"_id": {"$in": sim_ids}},
            {"experiment_id": 1, "generation_id": 1, "status": 1,
             "random_seed": 1, "start_time": 1, "assigned_at": 1},
        )) if sim_ids else []
        running = list(db["experiments"].find(
            {"status": EnumStatus.RUNNING},
            {"name": 1, "status": 1, "parameters.strategy": 1}))
        exp_ids = {s["experiment_id"] for s in simulations if s.get("experiment_id")}
        exp_ids -= {e["_id"] for e in running}
        experiments = running + (list(db["experiments"].find(
            {"_id": {"$in": list(exp_ids)}}, {"name": 1})) if exp_ids else [])
        gen_ids = [s["generation_id"] for s in simulations if s.get("generation_id")]
        generations = list(db["generations"].find(
            {"_id": {"$in": gen_ids}}, {"index": 1})) if gen_ids else []

    master_repo = getattr(factory, "master_repo", None)
    master = master_repo.get() if master_repo is not None else None

    return topology_from_documents(
        hosts=hosts, schedulable=schedulable, resources=resources,
        simulations=simulations, experiments=experiments,
        generations=generations, running_experiments=running, now=utc_now(),
        master=master if isinstance(master, dict) else None)


@router.get("/hosts", response_model=list[HostDto])
def list_hosts(
    factory: MongoRepository = Depends(get_factory),
) -> list[HostDto]:
    try:
        return [host_from_mongo(d) for d in factory.host_repo.find_all()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/hosts", response_model=str, status_code=status.HTTP_201_CREATED,
    openapi_extra=request_body(HOST_CREATE, "Register a remote host and how the master-node reaches it."),
)
def create_host(
    host: dict = Body(...),
    factory: MongoRepository = Depends(get_factory),
) -> str:
    """Register a remote host. Returns its id.

    The body is taken as a raw mapping, not as a typed model, on purpose.
    FastAPI validates a ``TypedDict`` by *dropping* fields it does not
    declare — so a client sending ``password`` would have received 201 and
    believed its credential was stored while nothing was written, and a
    client sending a different ``alg`` would have had it silently rewritten.
    Silently discarding a credential is worse than refusing it, so every
    field reaches :func:`host_to_mongo`, which accepts what it recognizes and
    rejects the rest with 422.
    """
    try:
        doc = host_to_mongo(dict(host))
    except (PlaintextRefused, MalformedEnvelope) as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not doc.get("name"):
        raise HTTPException(status_code=422, detail="name is required")
    # Only remote hosts are registered through the API, and a remote host is
    # reached over SSH: without an address and a user there is nothing to
    # connect to, and the failure would surface much later, in a log.
    _require_reachable(doc.get("access") or {})
    _require_credential(doc.get("access") or {})
    if factory.host_repo.get_by_name(doc["name"]) is not None:
        raise HTTPException(
            status_code=409, detail=f"a host named {doc['name']!r} already exists")
    try:
        return str(factory.host_repo.insert(doc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hosts/{host_id}", response_model=HostDetailDto)
def get_host(
    host_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> HostDetailDto:
    """One host with its containers and its occupancy."""
    host = _require_host(factory, host_id)
    resources = factory.resource_repo.find_by_host(host["_id"])
    occupancy = next(
        (occupancy_from_snapshot(dict(row))
         for row in factory.host_repo.inventory_snapshot()
         if row["_id"] == host["_id"]),
        None,
    )
    return {
        "host": host_from_mongo(host),
        "resources": [
            resource_from_mongo(r, host_name=host.get("name", ""), host=host)
            for r in resources
        ],
        "occupancy": occupancy,
    }


@router.put(
    "/hosts/{host_id}", response_model=HostDto,
    openapi_extra=request_body(HOST_UPDATE, "Change a host. Partial by field: see each field's description."),
)
def update_host(
    host_id: str,
    updates: dict = Body(...),
    factory: MongoRepository = Depends(get_factory),
) -> HostDto:
    """Change a host's configuration.

    Only the fields sent are touched, so a credential is left alone unless a
    new sealed one is supplied. Taken as a raw mapping for the same reason as
    the create: a dropped field would be a silent failure.
    """
    host = _require_host(factory, host_id)
    try:
        payload = host_to_mongo(dict(updates), for_update=True)
    except (PlaintextRefused, MalformedEnvelope) as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not payload:
        raise HTTPException(status_code=422, detail="nothing to update")

    new_name = payload.get("name")
    if new_name and new_name != host.get("name"):
        clash = factory.host_repo.get_by_name(new_name)
        if clash is not None:
            raise HTTPException(
                status_code=409, detail=f"a host named {new_name!r} already exists")

    if "access" in payload:
        # Validate the merged result, not the patch: an update that only
        # changes the port must not be judged as if it had dropped the
        # credential.
        merged = {**(host.get("access") or {}), **payload["access"]}
        _require_credential(merged)

    # Written by path, so the fields the client did not send survive. Writing
    # `payload` directly replaced each nested block with just those fields,
    # which is how editing a port removed the credential and the pinned key.
    factory.host_repo.update(host["_id"], dotted_update(payload))

    if _ACCESS_BLOCKS & set(payload):
        # The workers inherit whatever they do not override, so what was
        # measured about them no longer describes how they are reached.
        # Forgetting it makes the next health cycle re-measure; leaving it
        # would let a placement use a configuration nobody has tried while
        # the page still showed the old result as evidence that it worked.
        # Bumped first: the version is what lets a probe that is *already
        # running* be recognised as describing the previous configuration
        # when it finishes (R22). Clearing alone could not see that one.
        version = factory.host_repo.bump_config_version(host["_id"])
        affected = factory.resource_repo.clear_probes_of_host(host["_id"])
        log.info("Host %s access changed; configuration is now v%s",
                 host_id, version)
        if affected:
            log.info(
                "Host %s access changed; cleared the probe of %d worker(s)",
                host_id, affected,
            )

    return host_from_mongo(_require_host(factory, host_id))


@router.post("/hosts/{host_id}/enable", response_model=HostDto)
def enable_host(
    host_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> HostDto:
    host = _require_host(factory, host_id)
    factory.host_repo.set_enabled(host["_id"], True)
    return host_from_mongo(_require_host(factory, host_id))


@router.post("/hosts/{host_id}/disable", response_model=HostDto)
def disable_host(
    host_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> HostDto:
    """Take a host out of the pool.

    Running simulations are left to finish — disabling drains a host rather
    than interrupting it — so this succeeds even while it is busy.
    """
    host = _require_host(factory, host_id)
    factory.host_repo.set_enabled(host["_id"], False)
    return host_from_mongo(_require_host(factory, host_id))


@router.delete("/hosts/{host_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_host(
    host_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> Response:
    """Remove a host and its containers.

    Refused with 409 while anything is running on it: deleting the records
    would orphan a live simulation and lose the trail of where it ran. Disable
    it first and let it drain.
    """
    host = _require_host(factory, host_id)
    # The condition travels with the write. Counting the busy ones here and
    # deleting afterwards left a window the scheduler could reserve in, and
    # then removed the host along with the record of a live run.
    try:
        deleted, blocking = factory.host_repo.delete_if_drained(host["_id"])
    except RemovalConflicted as e:
        # A registration or a discovery committed against this host while the
        # removal was running. Nothing was removed and nothing is wrong, so
        # this is neither a 404 nor a drain problem: it is "try again".
        raise HTTPException(status_code=409, detail=f"{e}. Try again.")
    if blocking:
        # Nothing was removed. The deletions run in one transaction, so a
        # worker that cannot go rolls the whole thing back — the refusal
        # costs nothing, where it used to leave the free workers deleted.
        raise HTTPException(
            status_code=409,
            detail=(
                f"host {host.get('name')!r} still has worker(s) holding an "
                f"active simulation or unresolved run/result: {', '.join(blocking)}. "
                "Nothing was removed and the host is still in service. Disable "
                "it and let active simulations drain; recover retained results "
                "or confirm termination of retained runs before removing it."
            ),
        )
    if not deleted:
        raise HTTPException(status_code=404, detail="host not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── Requests the master-node carries out ─────────────────────────────────────

@router.post(
    "/hosts/{host_id}/check",
    response_model=AcceptedDto,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_check(
    host_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> AcceptedDto:
    """Ask the master-node to probe this host."""
    host = _require_host(factory, host_id)
    factory.host_repo.request_check(host["_id"])
    return {
        "accepted": True,
        "requested_at": (factory.host_repo.get(host["_id"]) or {}).get(
            "check_requested_at"),
        "detail": (
            "The master-node performs the check: it holds the private key and "
            "the route to the containers, which this API does not. Poll the "
            "host's last_check to see the result."
        ),
    }


@router.post(
    "/hosts/{host_id}/discover",
    response_model=AcceptedDto,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_discovery(
    host_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> AcceptedDto:
    """Ask the master-node to enumerate this host's Cooja containers."""
    host = _require_host(factory, host_id)
    factory.host_repo.request_discovery(host["_id"])
    return {
        "accepted": True,
        "requested_at": (factory.host_repo.get(host["_id"]) or {}).get(
            "discovery_requested_at"),
        "detail": (
            "The master-node queries Docker on that host. Poll the host's "
            "discovery block, or its resources, to see what was found."
        ),
    }


# ── Resources ────────────────────────────────────────────────────────────────

@router.get("/resources", response_model=list[ResourceDto])
def list_resources(
    factory: MongoRepository = Depends(get_factory),
) -> list[ResourceDto]:
    """Every container in the pool, with the host it belongs to."""
    try:
        # Keyed by the whole record, not just the name: `docker_applies`
        # depends on the host's discovery and address modes too.
        hosts = {h["_id"]: h for h in factory.host_repo.find_all()}
        return [
            resource_from_mongo(
                r,
                host_name=hosts.get(r.get("host_id"), {}).get("name", ""),
                host=hosts.get(r.get("host_id")),
            )
            for r in factory.resource_repo.find_all()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resources/{resource_id}", response_model=ResourceDto)
def get_resource(
    resource_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> ResourceDto:
    resource = _require_resource(factory, resource_id)
    host = factory.host_repo.get(resource.get("host_id")) or {}
    return resource_from_mongo(resource, host_name=host.get("name", ""), host=host)


@router.put(
    "/resources/{resource_id}", response_model=ResourceDto,
    openapi_extra=request_body(RESOURCE_ENABLED, "Take one worker in or out of the pool."),
)
def set_resource_enabled(
    resource_id: str,
    body: dict = Body(...),
    factory: MongoRepository = Depends(get_factory),
) -> ResourceDto:
    """Take one container in or out of the pool.

    ``enabled`` is the only field a client may write: everything else about a
    resource is discovered or measured. A busy container is allowed to be
    disabled — it finishes first and leaves the pool when released.
    """
    resource = _require_resource(factory, resource_id)
    if "enabled" not in body:
        raise HTTPException(status_code=422, detail="enabled is required")
    try:
        # `bool("false")` is True, so a client sending the string put a worker
        # *into* the pool while asking for the opposite. Every other route
        # went through this check; this one still called `bool()`.
        enabled = as_bool(body["enabled"], field="enabled")
    except PlaintextRefused as e:
        raise HTTPException(status_code=422, detail=str(e))
    factory.resource_repo.set_enabled(resource["_id"], enabled)
    host = factory.host_repo.get(resource.get("host_id")) or {}
    return resource_from_mongo(
        _require_resource(factory, resource_id),
        host_name=host.get("name", ""), host=host)


@router.get("/resources/{resource_id}/simulation")
def get_resource_simulation(
    resource_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """What is running on this container, if anything."""
    resource = _require_resource(factory, resource_id)
    simulation_id = resource.get("current_simulation_id")
    if not simulation_id:
        return {"simulation_id": None, "status": None, "experiment_id": None}
    simulation = factory.simulation_repo.get(str(simulation_id)) or {}
    return {
        "simulation_id": str(simulation_id),
        "status": simulation.get("status"),
        "experiment_id": (
            str(simulation["experiment_id"])
            if simulation.get("experiment_id") else None
        ),
        "generation_id": (
            str(simulation["generation_id"])
            if simulation.get("generation_id") else None
        ),
        "started_at": resource.get("started_at"),
        "attempt_id": resource.get("attempt_id"),
    }



def _refuse_duplicate_target(
    factory: MongoRepository, host: dict, resource: dict, exclude_id=None
) -> None:
    """Refuse a second name for the same physical worker.

    Discovery cannot create one — it keys on the container id — but a
    hand-registered worker can, and then two reservations exist for one run
    directory: each run's defensive cleanup deletes the other's inputs, and
    the exclusivity the reservation provides is held on the wrong document
    (R08).
    """
    clash = factory.resource_repo.conflicting_target(
        host, resource, exclude_id=exclude_id)
    if clash is None:
        return
    raise HTTPException(
        status_code=409,
        detail=(
            f"worker {clash.get('name')!r} already points at the same "
            "address, port and directory. Two registrations for one container "
            "would let two simulations share it."
        ),
    )

# ── Workers configured one by one ────────────────────────────────────────────

@router.post(
    "/hosts/{host_id}/resources",
    response_model=ResourceDto,
    status_code=status.HTTP_201_CREATED,
    openapi_extra=request_body(
        RESOURCE_CREATE,
        "Register one Cooja worker on this host, with its own SSH parameters."),
)
def create_resource(
    host_id: str,
    body: dict = Body(...),
    factory: MongoRepository = Depends(get_factory),
) -> ResourceDto:
    """Register one Cooja worker on this host, with its own SSH parameters.

    The counterpart to Docker discovery, and the only way that works on a host
    whose SSH user cannot reach its Docker daemon. ``endpoint.address`` says
    where this worker's sshd answers; anything left out of ``connection``
    falls back to the host's ``container_access``.
    """
    host = _require_host(factory, host_id)
    try:
        doc = resource_to_mongo(dict(body))
    except (PlaintextRefused, MalformedEnvelope) as e:
        raise HTTPException(status_code=422, detail=str(e))

    if factory.resource_repo.find_by_host_and_name(host["_id"], doc["name"]):
        raise HTTPException(
            status_code=409,
            detail=f"host {host.get('name')!r} already has a worker named "
                   f"{doc['name']!r}",
        )
    doc["host_id"] = host["_id"]
    doc.setdefault("remote_dir", DEFAULT_REMOTE_DIR)
    _refuse_duplicate_target(factory, host, doc)
    try:
        resource_id = factory.resource_repo.insert_manual(doc)
    except HostWithdrawing as e:
        raise HTTPException(status_code=409, detail=str(e))
    created = factory.resource_repo.get(resource_id)
    return resource_from_mongo(created, host_name=host.get("name", ""), host=host)


@router.put(
    "/resources/{resource_id}/connection", response_model=ResourceDto,
    openapi_extra=request_body(RESOURCE_UPDATE, "Change a worker's address or SSH parameters."),
)
def update_resource_connection(
    resource_id: str,
    body: dict = Body(...),
    factory: MongoRepository = Depends(get_factory),
) -> ResourceDto:
    """Change a worker's address or SSH parameters.

    Only what is sent is touched, so a stored credential survives an edit that
    changes the port.
    """
    resource = _require_resource(factory, resource_id)
    try:
        payload = resource_to_mongo(dict(body), for_update=True)
    except (PlaintextRefused, MalformedEnvelope) as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not payload:
        raise HTTPException(status_code=422, detail="nothing to update")

    # What the **client** sent, read before anything is merged. This is the
    # whole of R17: the decision below asks whether a port override was
    # stated, and asking it after the merge asks the wrong question — the
    # merge has just reintroduced the stored one. The form sends `connection`
    # with `username` and `auth_method` (nulls included), so the merged block
    # always had a port and the override was never removed. Endpoint saved as
    # 2223, effective port still 2222.
    sent_connection = dict(payload.get("connection") or {})
    states_effective_port = "endpoint" in payload and "port" not in sent_connection

    if "connection" in payload:
        # Merge, then drop what was explicitly removed. A null means "inherit
        # this again", and inheriting is the *absence* of the key: a stored
        # null would still read as the operator having configured something,
        # which is what decides whether Docker's view of the worker is used.
        merged = {**(resource.get("connection") or {}), **payload["connection"]}
        payload["connection"] = {
            k: v for k, v in merged.items() if v is not None}

    if states_effective_port:
        # The port the form sends is the port it shows. An older version of
        # the form also wrote `connection.port`, and that override *wins* in
        # the connection plan — so editing such a worker changed the endpoint,
        # reported the new port, and went on connecting to the old one.
        #
        # Sending an endpoint without a connection override is therefore taken
        # as stating the effective port, and a conflicting override is
        # removed. A client that means to keep one sends it in the same body.
        stored = (resource.get("connection") or {}).get("port")
        new_port = (payload.get("endpoint") or {}).get("port")
        if stored is not None and stored != new_port:
            payload["connection"] = {
                k: v for k, v in (payload.get("connection") or {}).items()
                if k != "port"}
            payload["_unset_connection_port"] = True

    merged_for_target = {
        **resource,
        **{k: v for k, v in payload.items()
           if k in ("name", "remote_dir", "endpoint", "connection")},
    }
    _refuse_duplicate_target(
        factory, factory.host_repo.get(resource.get("host_id")) or {},
        merged_for_target, exclude_id=resource["_id"])

    new_name = payload.get("name")
    if new_name and new_name != resource.get("name"):
        # A discovered worker's name *is* its identity: it is the container's
        # name, and the next discovery pass matches on it. Renaming one left
        # the original name with no record, so discovery inserted it again —
        # two registrations, two reservations, one container, and the database
        # cannot see that they are the same machine.
        container_id = (resource.get("container") or {}).get("id")
        if resource.get("origin") != "manual" or container_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"worker {resource.get('name')!r} was found by Docker, so "
                    "its name identifies the container and cannot be changed. "
                    "Change its connection settings instead, or remove it and "
                    "register the worker by hand under the name you want."
                ),
            )
        clash = factory.resource_repo.find_by_host_and_name(
            resource["host_id"], new_name)
        if clash is not None:
            raise HTTPException(
                status_code=409,
                detail=f"this host already has a worker named {new_name!r}")

    drop_connection_port = payload.pop("_unset_connection_port", False)
    factory.resource_repo.update_manual(resource["_id"], payload)
    if drop_connection_port:
        factory.resource_repo.clear_connection_port(resource["_id"])
    updated = _require_resource(factory, resource_id)
    host = factory.host_repo.get(updated.get("host_id")) or {}
    return resource_from_mongo(updated, host_name=host.get("name", ""), host=host)


@router.delete("/resources/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_resource(
    resource_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> Response:
    """Remove one worker.

    Refused with 409 while a simulation is on it: removing the record would
    orphan a live run and lose the trail of where it executed.

    A discovered worker can be removed too, but discovery will bring it back
    on the next pass — disabling it is what keeps it out of the pool.
    """
    resource = _require_resource(factory, resource_id)
    # Conditional in the write: the read above is only for the message. Between
    # that read and this call the scheduler can reserve the container, and
    # deleting it then would orphan a live run with no way to find it again.
    if not factory.resource_repo.delete_if_free(resource["_id"]):
        raise HTTPException(
            status_code=409,
            detail=(
                f"worker {resource.get('name')!r} is running a simulation or "
                "retaining an unresolved run/result. Disable it and let active "
                "simulations finish; recover retained results or confirm "
                "termination of retained runs before removing it."
            ),
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── helpers ──────────────────────────────────────────────────────────────────

def _require_reachable(access: dict) -> None:
    """Refuse a remote host nobody could connect to.

    An empty ``access`` used to pass: a host could be registered with nothing
    but a name, and every later operation failed with "has no SSH address
    configured" — far from its cause, and only visible in the master-node's
    log. A registration says how to reach the host or it is not a
    registration.
    """
    missing = [field for field in ("address", "username") if not access.get(field)]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"a remote host needs {' and '.join('access.' + m for m in missing)}: "
                "the master-node reaches its containers through the host, so "
                "there is nothing to connect to without them."
            ),
        )


def _require_credential(access: dict) -> None:
    """Refuse a host whose authentication needs a credential it does not have.

    Without this the registration succeeds and every later operation fails
    with "no credential is stored" — which is what happened in practice, and
    the error arrives far from its cause.
    """
    if not access:
        return
    if access.get("secret"):
        return
    method = access.get("auth_method")
    if method in (EnumAuthMethod.KEY.value, EnumAuthMethod.PASSWORD.value):
        raise HTTPException(
            status_code=422,
            detail=(
                f"auth_method is {method!r} but no credential was sent. Seal "
                "one against the master-node's public key "
                "(GET /infra/crypto/public-key) and send it as "
                "`access.secret: {key_id, ciphertext}`."
            ),
        )

def _require_host(factory: MongoRepository, host_id: str) -> dict:
    try:
        host = factory.host_repo.get(host_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=422, detail="invalid host id")
    if host is None:
        raise HTTPException(status_code=404, detail="Host not found")
    return host


def _require_resource(factory: MongoRepository, resource_id: str) -> dict:
    try:
        resource = factory.resource_repo.get(resource_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=422, detail="invalid resource id")
    if resource is None:
        raise HTTPException(status_code=404, detail="Resource not found")
    return resource
