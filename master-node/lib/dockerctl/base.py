"""The Docker inspection contract.

Docker is the authority on which Cooja containers exist: resources are
discovered, not registered. That answers two questions at once — which
containers a host has, and *where* each one is reachable for the second SSH
hop — which is why the endpoint is resolved from inspection rather than typed
in by an operator.

Named ``dockerctl`` rather than ``docker`` so a reader never has to wonder
whether an import refers to this package or to the Docker SDK on PyPI.
"""
from typing import Optional, Protocol, runtime_checkable


class DockerError(RuntimeError):
    """Docker could not be queried, or answered something unusable."""


class DockerUnavailable(DockerError):
    """The daemon could not be reached, or the user may not talk to it.

    Kept distinct so a host whose SSH works but whose Docker socket is not
    accessible to the configured user is reported as misconfigured rather than
    as simply down — the fix is different.
    """


@runtime_checkable
class DockerInspector(Protocol):
    """Lists containers as raw ``docker inspect`` documents.

    Both implementations return the *inspect* shape rather than the shorter
    listing shape, so a single normalizer serves them. The local one pays a few
    extra calls over a unix socket for that; the uniformity is worth more than
    those microseconds, because the alternative is two parsers that drift.
    """

    def list_containers(
        self,
        label_filter: Optional[str] = None,
        name_prefix: Optional[str] = None,
    ) -> list[dict]: ...

    def close(self) -> None: ...
