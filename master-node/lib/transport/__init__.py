"""Transport layer: how a Cooja container is reached.

``base`` holds the contract, ``plan`` resolves where to go, ``hostkey`` the
trust policy, ``ssh`` the only implementation, and ``factory`` composes one
from the database records.
"""
from lib.transport.base import (
    ExecResult,
    ProbeResult,
    Transport,
    TransportError,
    TransportUnreachable,
)
from lib.transport.plan import ConnectionPlan, PlanError, SSHEndpoint, build_plan
from lib.transport.hostkey import HostKeyMismatch, fingerprint
from lib.transport.ssh import GENERATED_ARTIFACTS, SSHTransport, build_cleanup_command
from lib.transport.factory import build_transport

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
    "HostKeyMismatch",
    "fingerprint",
    "GENERATED_ARTIFACTS",
    "SSHTransport",
    "build_cleanup_command",
    "build_transport",
]
