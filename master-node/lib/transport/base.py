"""The transport contract: what the execution path needs from a connection.

Narrow on purpose. Everything above it — preparing files, running Cooja,
collecting the log — works against this interface, so adding a transport later
(or testing the execution path without a network) does not touch that code.

SSH is the first and only implementation, which is exactly the separation the
design asks for: SSH is how a resource is reached, not what it is.
"""
from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecResult:
    exit_status: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_status == 0


@dataclass(frozen=True)
class ProbeResult:
    """Functional readiness of a Cooja container.

    The second of the two availability levels: Docker says the container
    exists and runs, this says it can actually run a simulation.
    """
    reachable: bool
    cooja_present: bool = False
    #: The first line of `java -version`, when the runtime answered at all.
    #: Collected since the beginning and, until now, not part of the verdict.
    java_version: Optional[str] = None
    #: Whether the JVM that launches Cooja actually ran. Separate from having
    #: a version string, which is only what it printed.
    java_ok: bool = False
    cooja_jar_sha256: Optional[str] = None
    message: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.reachable and self.cooja_present and self.java_ok


class TransportError(RuntimeError):
    """A transport-level failure: connection, tunnel, or file movement."""


class TransportUnreachable(TransportError):
    """The target could not be reached.

    Distinct from every other failure on purpose: unreachable says nothing
    about whether a simulation already running there is still alive, so the
    caller must not treat it as a failed run.
    """


@runtime_checkable
class Transport(Protocol):
    """One connection to one Cooja container, for the length of one attempt."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def put_files(self, local_paths: list[str], remote_names: list[str]) -> None: ...

    def get_file(self, remote_name: str, local_path: str) -> None: ...

    def exec_command(self, command: str, timeout: Optional[float] = None) -> ExecResult: ...

    def stream_command(
        self, command: str, timeout: Optional[float] = None
    ) -> Iterator[tuple[str, str]]:
        """Yield ``(channel, text)`` as output arrives, then return."""
        ...

    def probe(self) -> ProbeResult: ...

    def cleanup(self, remote_names: Optional[list[str]] = None) -> bool: ...
