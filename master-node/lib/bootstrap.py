"""The local host, as host and resource documents.

The master-node has always derived its container list from ``IS_DOCKER`` and
``NUMBER_OF_CONTAINERS``. This expresses that same derivation in the shape the
infrastructure collections use, so one definition serves both the execution
path (which builds a transport from these documents) and the startup upsert
that persists them.

Keeping it in one place is what makes ``docker compose up`` continue to work
with no registration step: the documents the scheduler reads for the local host
are the documents the environment already described.
"""
import logging
from datetime import datetime
from typing import Any, Optional

from bson import ObjectId

from pylib.db.models.infra import (
    EnumAddressMode,
    EnumAuthMethod,
    EnumContainerState,
    EnumHostStatus,
    EnumHostType,
    EnumResourceStatus,
)

log = logging.getLogger(__name__)

LOCAL_HOST_NAME = "local"
# The label every Cooja service in docker-compose.yaml already carries.
DEFAULT_COOJA_LABEL = "simlab.type=cooja"


def local_host_document(
    *,
    is_docker: bool,
    container_user: str = "root",
    container_port: int = 22,
    label_filter: str = DEFAULT_COOJA_LABEL,
) -> dict[str, Any]:
    """The local host: no first hop, and discovery over the mounted socket.

    ``access`` is absent rather than empty — a local host has no host to SSH
    into, and the transport decides on the jump from the type alone.

    The address mode follows the deployment: inside Docker the master-node
    shares ``cooja-net`` and Docker's DNS answers for container names, which is
    exactly how it reaches ``cooja1..N`` today. Run from the host instead, it
    has no such DNS, so it goes to the ports published on loopback — again what
    it already did.
    """
    return {
        "name": LOCAL_HOST_NAME,
        "type": EnumHostType.LOCAL.value,
        "enabled": True,
        # ONLINE, not OFFLINE. The bootstrap is an assertion by the
        # environment — the same one that makes the seeded resources
        # `available` — and the pre-feature master-node trusted it, finding
        # out otherwise by failing a run. Starting the host OFFLINE while its
        # resources are AVAILABLE deadlocks the pool: nothing is schedulable,
        # and the health cycle that would promote it does not exist yet.
        #
        # A *registered* remote host is different and does start OFFLINE:
        # nothing asserts that it works until a probe says so.
        "status": EnumHostStatus.ONLINE.value,
        "access": None,
        "container_access": {
            "transport": "ssh",
            "username": container_user,
            "port": container_port,
            "address_mode": (
                EnumAddressMode.CONTAINER_NAME.value if is_docker
                else EnumAddressMode.PUBLISHED_PORT.value
            ),
            # No envelope: the credential falls back to the image default (see
            # lib/credentials.py), which is how the local containers have
            # always been reached and keeps a password out of the database.
            "auth_method": EnumAuthMethod.PASSWORD.value,
            "secret": None,
        },
        "discovery": {
            "mode": "docker",
            "label_filter": label_filter,
            "name_prefix": None,
            "endpoint": "socket",
            "last_discovery": None,
        },
        "origin": "bootstrap",
        "consecutive_failures": 0,
        "last_check": None,
        "last_check_message": "",
        "check_requested_at": None,
        "discovery_requested_at": None,
    }


def local_resource_documents(
    *,
    hostnames: list[str],
    ports: list[int],
    remote_dir: str,
    host_id: Optional[ObjectId] = None,
) -> list[dict[str, Any]]:
    """Resources for the containers the environment describes.

    A seed, not an inventory: a real discovery pass replaces the container
    facts and the endpoint with what Docker reports. It exists so the first
    start — before any discovery has run, or with no Docker socket mounted —
    still has somewhere to send work, exactly as the master-node did before.

    Each resource starts ``available``, which is the pre-feature behaviour:
    the environment asserted these containers exist and the worker found out
    otherwise by failing. Discovery and the probe take over that judgement as
    soon as they run.
    """
    now = datetime.now()
    documents: list[dict[str, Any]] = []
    for index, (hostname, port) in enumerate(zip(hostnames, ports), start=1):
        name = hostname if hostname not in ("localhost", "127.0.0.1") else f"cooja{index}"
        document: dict[str, Any] = {
            "name": name,
            "container": {
                "id": "",
                "name": name,
                "image": "",
                # Not yet observed. Discovery sets the real value; until then
                # nothing claims to know whether the container is running.
                "state": EnumContainerState.MISSING.value,
                "labels": {},
                "checked_at": None,
            },
            "endpoint": {"address": hostname, "port": port},
            "remote_dir": remote_dir,
            "enabled": True,
            "status": EnumResourceStatus.AVAILABLE.value,
            "current_simulation_id": None,
            "attempt_id": None,
            "remote_pid": None,
            "probe": {"ssh_ok": False, "checked_at": None},
            "origin": "bootstrap",
            "last_check": None,
            "last_check_message": "seeded from NUMBER_OF_CONTAINERS",
            "created_time": now,
            "updated_time": now,
        }
        if host_id is not None:
            document["host_id"] = host_id
        documents.append(document)
    return documents


def ensure_local_host(
    mongo: Any,
    *,
    is_docker: bool,
    hostnames: list[str],
    ports: list[int],
    remote_dir: str,
) -> ObjectId:
    """Persist the local host and its seeded resources, idempotently.

    Runs on every start, which is what makes ``docker compose up`` work with no
    registration step. It must therefore add what is missing without undoing
    anything an operator changed, so both writes are insert-only:
    ``upsert_bootstrap`` for the host and ``insert_seeded`` for each resource.
    A resource that discovery has since described, that was disabled, or that
    is holding a simulation is left exactly as it is.

    Returns the host's id.
    """
    document = local_host_document(is_docker=is_docker)
    host_id = mongo.host_repo.upsert_bootstrap(document)
    # upsert_bootstrap is insert-only, so an existing host keeps whatever
    # status it was created with. This corrects one written by an earlier
    # version as OFFLINE — and does nothing once a real check has run.
    mongo.host_repo.set_status_if_unchecked(host_id, EnumHostStatus.ONLINE)

    seeded = local_resource_documents(
        hostnames=hostnames, ports=ports, remote_dir=remote_dir, host_id=host_id)
    created = sum(
        1 for document in seeded if mongo.resource_repo.insert_seeded(document))

    # Until the health cycle exists, `offline` is a one-way door: nothing
    # promotes a container back. Any of this host's containers sitting there
    # with no observation behind it never actually failed, so it is returned
    # to the pool.
    restored = mongo.resource_repo.restore_unobserved(host_id)

    log.info(
        "Local host %s ready: %d of %d resource(s) seeded, %d restored to the pool",
        host_id, created, len(seeded), restored,
    )
    return host_id


def publish_public_key(mongo: Any, keystore: Any) -> None:
    """Publish the keystore's public half, so the browser can seal to it.

    The API serves this; it holds no private key and could not produce it. Only
    public material is written — the private key never leaves its mounted file.
    """
    mongo.infra_crypto_repo.publish(
        alg="crypto_box_seal",
        key_id=keystore.key_id,
        public_key=keystore.public_key_b64,
        fingerprint=keystore.fingerprint,
        retired_key_ids=keystore.retired_key_ids,
    )
    log.info(
        "Published infrastructure public key %s (%s)",
        keystore.key_id, keystore.fingerprint,
    )
