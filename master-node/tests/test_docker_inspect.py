"""Tests for Docker inspection and endpoint resolution.

Discovery is what makes resources appear without anyone registering them, and
what resolves the second SSH hop's address. Both halves are locked in here:

  * a stopped or vanished container is *seen*, so it can leave the pool
    without taking its history with it;
  * the hop-2 address is derived per address mode, and is ``None`` rather than
    a guess when it cannot be known;
  * a host where Docker is not usable is reported as misconfigured, not down —
    the two have different fixes.
"""
import json
from unittest.mock import MagicMock

import pytest

from lib.dockerctl.base import DockerError, DockerUnavailable
from lib.dockerctl.normalize import (
    container_name,
    is_running,
    matches_label,
    matches_name_prefix,
    normalize_container,
    resolve_endpoint,
)
from lib.dockerctl.socket_api import SocketDockerInspector
from lib.dockerctl.ssh_cli import SSHDockerInspector
from lib.transport.base import ExecResult, TransportUnreachable


def inspect_doc(
    name="cooja1",
    state="running",
    ip="172.18.0.5",
    published=("127.0.0.1", "2231"),
    labels=None,
):
    """A ``docker inspect`` document, trimmed to the fields that are read."""
    ports = {}
    if published:
        ports["22/tcp"] = [{"HostIp": published[0], "HostPort": published[1]}]
    return {
        "Id": "a1b2c3d4e5f6",
        "Name": f"/{name}",
        "State": {"Status": state, "Running": state == "running"},
        "Config": {
            "Image": "juniocesarferreira/simulation-cooja:v1.1",
            "Labels": labels if labels is not None else {"simlab.type": "cooja"},
        },
        "NetworkSettings": {
            "Networks": {"cooja_net": {"IPAddress": ip}},
            "Ports": ports,
        },
    }


# ── Normalization ────────────────────────────────────────────────────────────

class TestNormalize:
    def test_name_loses_dockers_leading_slash(self):
        assert container_name(inspect_doc("cooja3")) == "cooja3"

    def test_fields_the_resource_record_needs(self):
        info = normalize_container(inspect_doc())
        assert info["name"] == "cooja1"
        assert info["state"] == "running"
        assert info["image"].endswith(":v1.1")
        assert info["labels"] == {"simlab.type": "cooja"}
        assert info["id"] == "a1b2c3d4e5f6"

    def test_docker_vocabulary_is_stored_verbatim(self):
        # Not mapped onto an enum: Docker has added states before, and a stale
        # mapping table would report a container as something it is not.
        assert normalize_container(inspect_doc(state="removing"))["state"] == "removing"

    def test_only_running_counts_as_usable(self):
        assert is_running(normalize_container(inspect_doc(state="running"))) is True
        for state in ("exited", "paused", "created", "dead", "restarting"):
            assert is_running(normalize_container(inspect_doc(state=state))) is False

    def test_boolean_only_payload_is_tolerated(self):
        raw = inspect_doc()
        raw["State"] = {"Running": True}
        assert normalize_container(raw)["state"] == "running"


# ── Endpoint resolution: where hop 2 connects ────────────────────────────────

class TestResolveEndpoint:
    def test_container_ip_is_the_remote_default(self):
        # The bridge address, which the host itself can route to — so no
        # container port has to be published anywhere.
        assert resolve_endpoint(
            inspect_doc(), address_mode="container_ip", container_port=22
        ) == {"address": "172.18.0.5", "port": 22}

    def test_container_ip_is_deterministic_across_networks(self):
        raw = inspect_doc()
        raw["NetworkSettings"]["Networks"] = {
            "zeta_net": {"IPAddress": "10.0.9.9"},
            "alpha_net": {"IPAddress": "10.0.1.1"},
        }
        # Sorted by network name so repeated passes agree with each other.
        assert resolve_endpoint(
            raw, address_mode="container_ip", container_port=22
        )["address"] == "10.0.1.1"

    def test_container_ip_falls_back_to_the_legacy_field(self):
        raw = inspect_doc()
        raw["NetworkSettings"]["Networks"] = {}
        raw["NetworkSettings"]["IPAddress"] = "172.17.0.9"
        assert resolve_endpoint(
            raw, address_mode="container_ip", container_port=22
        )["address"] == "172.17.0.9"

    def test_published_port_uses_the_binding(self):
        assert resolve_endpoint(
            inspect_doc(published=("127.0.0.1", "2235")),
            address_mode="published_port", container_port=22,
        ) == {"address": "127.0.0.1", "port": 2235}

    def test_wildcard_binding_becomes_loopback(self):
        # Reachable from the host over loopback, and it does not depend on the
        # host knowing its own external address.
        assert resolve_endpoint(
            inspect_doc(published=("0.0.0.0", "2232")),
            address_mode="published_port", container_port=22,
        ) == {"address": "127.0.0.1", "port": 2232}

    def test_container_name_is_the_local_mode(self):
        assert resolve_endpoint(
            inspect_doc("cooja7"), address_mode="container_name", container_port=22
        ) == {"address": "cooja7", "port": 22}

    def test_unresolvable_returns_none_rather_than_a_guess(self):
        raw = inspect_doc(ip="", published=None)
        assert resolve_endpoint(
            raw, address_mode="container_ip", container_port=22) is None
        assert resolve_endpoint(
            raw, address_mode="published_port", container_port=22) is None

    def test_published_port_without_that_port_is_unresolvable(self):
        raw = inspect_doc(published=("127.0.0.1", "2231"))
        assert resolve_endpoint(
            raw, address_mode="published_port", container_port=2222) is None


# ── Client-side filters ──────────────────────────────────────────────────────

class TestFilters:
    def test_key_value_label_matches(self):
        assert matches_label(inspect_doc(), "simlab.type=cooja") is True
        assert matches_label(inspect_doc(), "simlab.type=master") is False

    def test_bare_key_matches_on_presence(self):
        assert matches_label(inspect_doc(), "simlab.type") is True
        assert matches_label(inspect_doc(), "simlab.absent") is False

    def test_no_filter_matches_everything(self):
        assert matches_label(inspect_doc(labels={}), None) is True

    def test_name_prefix(self):
        assert matches_name_prefix(inspect_doc("cooja1"), "cooja") is True
        assert matches_name_prefix(inspect_doc("other1"), "cooja") is False
        assert matches_name_prefix(inspect_doc("other1"), None) is True


# ── Remote inspection over SSH ───────────────────────────────────────────────

def session_returning(stdout="", exit_status=0, stderr=""):
    session = MagicMock()
    session.host_name = "lab-01"
    session.exec_command.return_value = ExecResult(exit_status, stdout, stderr)
    return session


class TestSSHDockerInspector:
    def test_lists_containers_from_inspect_output(self):
        docs = [inspect_doc("cooja1"), inspect_doc("cooja2", state="exited")]
        stdout = "\n".join(json.dumps(d) for d in docs)
        session = session_returning(stdout)

        found = SSHDockerInspector(session).list_containers("simlab.type=cooja")

        assert [container_name(d) for d in found] == ["cooja1", "cooja2"]

    def test_stopped_containers_are_included(self):
        # --all, deliberately: a stopped container must be seen so it can be
        # marked offline instead of vanishing with its history.
        session = session_returning(json.dumps(inspect_doc(state="exited")))
        SSHDockerInspector(session).list_containers("simlab.type=cooja")
        assert "--all" in session.exec_command.call_args.args[0]

    def test_benign_label_filter_is_passed_through(self):
        session = session_returning("")
        SSHDockerInspector(session).list_containers("simlab.type=cooja")
        assert "--filter label=simlab.type=cooja" in session.exec_command.call_args.args[0]

    def test_dangerous_label_filter_cannot_break_out_of_the_command(self):
        # The filter comes from a host registration and travels through a
        # shell on the remote host, so it has to be quoted. shlex.quote only
        # adds quotes when they are needed, which is why the benign case above
        # looks bare and this one does not.
        session = session_returning("")
        SSHDockerInspector(session).list_containers("x=1; rm -rf /tmp/pwned")
        script = session.exec_command.call_args.args[0]
        assert "'x=1; rm -rf /tmp/pwned'" in script
        assert "; rm -rf /tmp/pwned 2>/dev/null" not in script

    def test_label_is_rechecked_locally(self):
        # The filter travels through a shell to reach the daemon; re-checking
        # means a quoting mistake cannot widen the inventory.
        session = session_returning(json.dumps(inspect_doc(labels={"other": "x"})))
        found = SSHDockerInspector(session).list_containers("simlab.type=cooja")
        assert found == []

    def test_no_containers_is_not_a_failure(self):
        # `docker inspect` with no arguments exits non-zero, so the script
        # guards it — otherwise an empty host would look broken.
        session = session_returning("", exit_status=0)
        assert SSHDockerInspector(session).list_containers() == []

    def test_docker_not_usable_is_misconfigured_not_down(self):
        session = session_returning(
            "", exit_status=90, stderr="permission denied while trying to connect")
        with pytest.raises(DockerUnavailable) as exc:
            SSHDockerInspector(session).list_containers()
        message = str(exc.value)
        assert "SSH user may use" in message and "permission denied" in message

    def test_missing_docker_binary_is_named(self):
        session = session_returning("", exit_status=127)
        with pytest.raises(DockerUnavailable, match="no `docker` command"):
            SSHDockerInspector(session).list_containers()

    def test_unreachable_host_surfaces_as_unavailable(self):
        session = MagicMock()
        session.host_name = "lab-01"
        session.exec_command.side_effect = TransportUnreachable("no route")
        with pytest.raises(DockerUnavailable, match="no route"):
            SSHDockerInspector(session).list_containers()

    def test_other_failures_are_plain_docker_errors(self):
        session = session_returning("", exit_status=1, stderr="boom")
        with pytest.raises(DockerError, match="exited 1"):
            SSHDockerInspector(session).list_containers()

    def test_unparsable_line_is_skipped_not_fatal(self):
        stdout = "not json\n" + json.dumps(inspect_doc("cooja1"))
        found = SSHDockerInspector(session_returning(stdout)).list_containers()
        assert [container_name(d) for d in found] == ["cooja1"]


# ── Local inspection over the unix socket ────────────────────────────────────

class TestSocketDockerInspector:
    def test_lists_then_inspects_each_container(self, monkeypatch):
        inspector = SocketDockerInspector("/tmp/does-not-matter.sock")
        calls: list[str] = []

        def fake_get(path):
            calls.append(path)
            if path.startswith("/containers/json"):
                return [{"Id": "a1b2c3d4e5f6"}, {"Id": "ffeeddccbbaa"}]
            return inspect_doc("cooja1")

        monkeypatch.setattr(inspector, "_get", fake_get)
        found = inspector.list_containers("simlab.type=cooja")

        assert len(found) == 2
        # Full inspect documents, so one normalizer serves both transports.
        assert calls[0].startswith("/containers/json?")
        assert calls[1] == "/containers/a1b2c3d4e5f6/json"

    def test_listing_requests_stopped_containers_and_the_label_filter(self, monkeypatch):
        inspector = SocketDockerInspector()
        captured: list[str] = []
        monkeypatch.setattr(inspector, "_get", lambda p: captured.append(p) or [])
        inspector.list_containers("simlab.type=cooja")
        path = captured[0]
        assert "all=1" in path
        assert "simlab.type%3Dcooja" in path or "simlab.type=cooja" in path

    def test_no_api_version_is_pinned(self, monkeypatch):
        # The daemon serves its own current version; pinning one here would
        # break on a host whose Docker is older or newer than expected.
        inspector = SocketDockerInspector()
        captured: list[str] = []
        monkeypatch.setattr(inspector, "_get", lambda p: captured.append(p) or [])
        inspector.list_containers()
        assert not captured[0].startswith("/v1.")

    def test_container_removed_between_calls_is_skipped(self, monkeypatch):
        inspector = SocketDockerInspector()

        def fake_get(path):
            if path.startswith("/containers/json"):
                return [{"Id": "gone"}, {"Id": "stays"}]
            if path == "/containers/gone/json":
                raise DockerError("404")
            return inspect_doc("cooja2")

        monkeypatch.setattr(inspector, "_get", fake_get)
        # Normal churn during a discovery pass, not a reason to fail it.
        assert len(inspector.list_containers()) == 1

    def test_absent_socket_says_what_to_mount(self):
        inspector = SocketDockerInspector("/nonexistent/docker.sock", timeout=1.0)
        with pytest.raises(DockerUnavailable, match="mount it"):
            inspector.list_containers()

    def test_name_prefix_filters_locally(self, monkeypatch):
        inspector = SocketDockerInspector()

        def fake_get(path):
            if path.startswith("/containers/json"):
                return [{"Id": "one"}]
            return inspect_doc("other1")

        monkeypatch.setattr(inspector, "_get", fake_get)
        assert inspector.list_containers(name_prefix="cooja") == []
