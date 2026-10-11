"""Local Docker, over the mounted unix socket.

Plain HTTP on a unix socket rather than the Docker SDK: it is some thirty lines
of standard library against an API this code uses two endpoints of, and it
keeps a dependency — plus its transitive tree — out of an image whose job is to
run simulations. The SDK's value is in the parts not needed here.

The socket is mounted read-only in the compose file, which stops the *file*
from being replaced but does not restrict the API; the daemon exposes
everything to anyone who can talk to it. That is the same access cAdvisor
already has in this stack, and the documented hardening is a read-only socket
proxy in front of it.
"""
import http.client
import json
import logging
import socket
import urllib.parse
from typing import Any, Optional

from lib.dockerctl.base import DockerError, DockerUnavailable
from lib.dockerctl.normalize import matches_name_prefix

log = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = "/var/run/docker.sock"
DEFAULT_TIMEOUT = 10.0


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that dials a unix socket instead of a TCP port."""

    def __init__(self, socket_path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


class SocketDockerInspector:
    """Queries the daemon the master-node shares a kernel with."""

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET_PATH,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.socket_path = socket_path
        self.timeout = timeout

    # ── HTTP plumbing ──────────────────────────────────────────────────────

    def _get(self, path: str) -> Any:
        conn = _UnixHTTPConnection(self.socket_path, self.timeout)
        try:
            conn.request("GET", path, headers={"Host": "docker", "Accept": "application/json"})
            response = conn.getresponse()
            body = response.read()
        except FileNotFoundError:
            raise DockerUnavailable(
                f"Docker socket {self.socket_path} is not present; mount it to "
                "let the master-node discover local containers"
            ) from None
        except PermissionError:
            raise DockerUnavailable(
                f"no permission to use the Docker socket {self.socket_path}"
            ) from None
        except (OSError, http.client.HTTPException) as e:
            raise DockerUnavailable(
                f"Docker socket {self.socket_path} could not be queried: {e}"
            ) from None
        finally:
            conn.close()

        if response.status == 404:
            raise DockerError(f"Docker API path not found: {path}")
        if response.status >= 400:
            detail = body.decode("utf-8", errors="ignore").strip()
            raise DockerError(f"Docker API {path} returned {response.status}: {detail}")
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise DockerError(f"Docker API {path} returned invalid JSON: {e}") from None

    # ── inspection ─────────────────────────────────────────────────────────

    def list_containers(
        self,
        label_filter: Optional[str] = None,
        name_prefix: Optional[str] = None,
    ) -> list[dict]:
        """List matching containers as full inspect documents.

        ``all=1`` on purpose: a stopped container must still be seen, so it can
        be marked offline rather than silently vanishing from the inventory
        along with the record of what ran on it.
        """
        query: dict[str, str] = {"all": "1"}
        if label_filter:
            query["filters"] = json.dumps({"label": [label_filter]})
        # No API version in the path: the daemon then serves its own current
        # version, instead of this code pinning one the host may not have.
        path = "/containers/json?" + urllib.parse.urlencode(query)

        summaries = self._get(path)
        if not isinstance(summaries, list):
            raise DockerError("Docker returned an unexpected container listing")

        containers: list[dict] = []
        for summary in summaries:
            container_id = summary.get("Id")
            if not container_id:
                continue
            try:
                raw = self._get(f"/containers/{container_id}/json")
            except DockerError as e:
                # A container removed between listing and inspection is normal,
                # and not a reason to fail the whole discovery pass.
                log.debug("Skipping container %s: %s", container_id[:12], e)
                continue
            if matches_name_prefix(raw, name_prefix):
                containers.append(raw)
        return containers

    def ping(self) -> dict:
        """Confirm the daemon answers, and report what it is.

        ``/version`` rather than ``/_ping``: the latter replies with plain text
        that the JSON reader here would choke on, and the version is worth
        having in a health message anyway.
        """
        return self._get("/version")

    def close(self) -> None:
        """Nothing to close: each request uses its own short-lived connection."""
