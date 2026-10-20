"""Remote Docker, through the host's own CLI over SSH.

Runs ``docker ps`` and ``docker inspect`` on the host, inside the session that
hop 1 already authenticated. No extra port, no daemon exposed over TCP, no
second credential: the one thing the host publishes stays its sshd.

The configured SSH user must be able to talk to the host's Docker socket —
usually membership of the ``docker`` group. That is a prerequisite worth
stating plainly, because it grants that user effective root on the host.
"""
import json
import logging
import shlex
from typing import Optional

from lib.dockerctl.base import DockerError, DockerUnavailable
from lib.dockerctl.normalize import matches_label, matches_name_prefix
from lib.transport.base import TransportError, TransportUnreachable
from lib.transport.session import HostSession

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0

# Two commands in one round trip. POSIX sh only — no `xargs -r`, which busybox
# does not reliably provide and the stand-in remote host in debug/ runs Alpine.
# The guard matters: `docker inspect` with no arguments exits non-zero, which
# would make "this host has no Cooja containers yet" look like a failure.
_LIST_SCRIPT = """\
ids=$(docker ps --all --quiet {filter_args} 2>/dev/null) || exit 90
if [ -z "$ids" ]; then exit 0; fi
docker inspect $ids --format '{{{{json .}}}}'
"""


class SSHDockerInspector:
    """Asks a remote host what Cooja containers it has."""

    def __init__(self, session: HostSession, timeout: float = DEFAULT_TIMEOUT):
        self._session = session
        self._timeout = timeout

    def list_containers(
        self,
        label_filter: Optional[str] = None,
        name_prefix: Optional[str] = None,
    ) -> list[dict]:
        filter_args = ""
        if label_filter:
            filter_args = f"--filter label={shlex.quote(label_filter)}"
        script = _LIST_SCRIPT.format(filter_args=filter_args)

        try:
            result = self._session.exec_command(script, timeout=self._timeout)
        except TransportUnreachable as e:
            raise DockerUnavailable(str(e)) from None
        except TransportError as e:
            raise DockerUnavailable(
                f"could not run docker on host {self._session.host_name}: {e}"
            ) from None

        if result.exit_status == 90:
            # `docker ps` itself failed: not installed, or the SSH user cannot
            # reach the socket. Either way the host is misconfigured, not down.
            raise DockerUnavailable(
                f"host {self._session.host_name} did not answer `docker ps`; "
                "check that Docker is installed and that the SSH user may use "
                f"its socket. {result.stderr.strip() or '(no stderr)'}"
            )
        if result.exit_status == 127:
            raise DockerUnavailable(
                f"host {self._session.host_name} has no `docker` command on PATH"
            )
        if not result.ok:
            raise DockerError(
                f"docker inspect on host {self._session.host_name} exited "
                f"{result.exit_status}: {result.stderr.strip()}"
            )

        containers: list[dict] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                log.warning(
                    "Host %s returned an unparsable docker inspect line",
                    self._session.host_name,
                )
                continue
            # The daemon already applied the label filter, but it travelled
            # through a shell to get there; re-checking locally costs nothing
            # and means a quoting mistake cannot widen the inventory.
            if matches_label(raw, label_filter) and matches_name_prefix(raw, name_prefix):
                containers.append(raw)
        return containers

    def ping(self) -> str:
        """Report the host's Docker version, for a health message."""
        try:
            result = self._session.exec_command(
                "docker version --format '{{.Server.Version}}'", timeout=self._timeout)
        except TransportError as e:
            raise DockerUnavailable(str(e)) from None
        if not result.ok:
            raise DockerUnavailable(
                f"host {self._session.host_name} did not answer `docker version`: "
                f"{result.stderr.strip()}"
            )
        return result.stdout.strip()

    def close(self) -> None:
        """The session's owner closes it: this inspector only borrows it."""
