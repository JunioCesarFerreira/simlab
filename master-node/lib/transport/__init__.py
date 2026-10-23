"""Transport layer: how a Cooja container is reached.

``base`` holds the contract, ``plan`` resolves where to go, ``resolve`` turns
a host's name into an address (and remembers the last one that worked),
``hostkey`` the trust policy, ``ssh`` the only implementation, ``factory`` composes one from
the database records, and ``pool`` keeps hop 1 from being opened once per
container.
"""
from lib.transport.base import (
    ExecResult,
    ProbeResult,
    Transport,
    TransportError,
    TransportUnreachable,
)
from lib.transport.plan import ConnectionPlan, PlanError, SSHEndpoint, build_plan
from lib.transport.resolve import (
    DEFAULT_RESOLVER,
    HostResolver,
    NameResolutionFailed,
    Resolution,
)
from lib.transport.hostkey import HostKeyMismatch, fingerprint
from lib.transport.ssh import GENERATED_ARTIFACTS, SSHTransport, build_cleanup_command
from lib.transport.factory import build_transport
from lib.transport.pool import JumpPool, needs_jump
from lib.transport.session import HostSession, connect_host

__all__ = [
    "ExecResult",
    "ProbeResult",
    "Transport",
    "TransportError",
    "TransportUnreachable",
    "ConnectionPlan",
    "PlanError",
    "SSHEndpoint",
    "build_plan",
    "DEFAULT_RESOLVER",
    "HostResolver",
    "NameResolutionFailed",
    "Resolution",
    "HostKeyMismatch",
    "fingerprint",
    "GENERATED_ARTIFACTS",
    "SSHTransport",
    "build_cleanup_command",
    "build_transport",
    "JumpPool",
    "needs_jump",
    "HostSession",
    "connect_host",
]
