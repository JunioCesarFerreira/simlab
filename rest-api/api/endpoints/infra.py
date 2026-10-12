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
from fastapi import APIRouter, Body, Depends, HTTPException, Response, status
from bson import errors as bson_errors

from pylib.crypto import MalformedEnvelope
from pylib.db import MongoRepository
from pylib.db.models.infra import OCCUPIED_STATUSES

from api.dependencies import get_factory
# The write DTOs (HostCreateDto, HostUpdateDto, ResourceEnabledDto) stay in
# api.domain.infra as the documented shape of each body, and are referenced by
# name in the Body() descriptions. They are deliberately not used as
# parameter types — see create_host for why.
from api.domain.infra import (
    AcceptedDto,
    HostDetailDto,
    HostDto,
    OverviewDto,
    PublicKeyDto,
    ResourceDto,
)
from api.mappers.infra import (
    PlaintextRefused,
    host_from_mongo,
    host_to_mongo,
    occupancy_from_snapshot,
    overview_from_snapshot,
    public_key_from_mongo,
    resource_from_mongo,
)

router = APIRouter()

_OCCUPIED = [s.value for s in OCCUPIED_STATUSES]


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
                "master-node publishes it on startup (INFRA_MODE=managed); "
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
        return overview_from_snapshot(snapshot, builds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Hosts ────────────────────────────────────────────────────────────────────

@router.get("/hosts", response_model=list[HostDto])
def list_hosts(
    factory: MongoRepository = Depends(get_factory),
) -> list[HostDto]:
    try:
        return [host_from_mongo(d) for d in factory.host_repo.find_all()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hosts", response_model=str, status_code=status.HTTP_201_CREATED)
def create_host(
    host: dict = Body(..., description="See HostCreateDto for the accepted shape"),
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
            resource_from_mongo(r, host_name=host.get("name", "")) for r in resources
        ],
        "occupancy": occupancy,
    }


@router.put("/hosts/{host_id}", response_model=HostDto)
def update_host(
    host_id: str,
    updates: dict = Body(..., description="See HostUpdateDto for the accepted shape"),
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

    factory.host_repo.update(host["_id"], payload)
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
    busy = [
        r.get("name", "?")
        for r in factory.resource_repo.find_by_host(host["_id"])
        if r.get("status") in _OCCUPIED
    ]
    if busy:
        raise HTTPException(
            status_code=409,
            detail=(
                f"host {host.get('name')!r} still has simulations running on "
                f"{', '.join(busy)}. Disable the host and let it drain first."
            ),
        )
    factory.host_repo.delete(host["_id"])
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
        names = {h["_id"]: h.get("name", "") for h in factory.host_repo.find_all()}
        return [
            resource_from_mongo(r, host_name=names.get(r.get("host_id"), ""))
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
    return resource_from_mongo(resource, host_name=host.get("name", ""))


@router.put("/resources/{resource_id}", response_model=ResourceDto)
def set_resource_enabled(
    resource_id: str,
    body: dict = Body(..., description="See ResourceEnabledDto"),
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
    factory.resource_repo.set_enabled(resource["_id"], bool(body["enabled"]))
    return resource_from_mongo(_require_resource(factory, resource_id))


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


# ── helpers ──────────────────────────────────────────────────────────────────

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
