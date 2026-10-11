"""Tests for the two-hop transport, its credentials and its host-key policy.

Paramiko is mocked, in the style of ``test_cleanup.py``: what matters here is
*which* connections are opened, in what order, with which credential, and that
a tunnel is used exactly when the host is remote.

Properties locked in:
  * the local host still takes the single-hop path, unchanged;
  * a remote host is reached by connecting to it and tunnelling to the
    container, so no container port need be published;
  * hop 1 is pinned on first use and refuses a changed key; hop 2 is not;
  * a secret never appears in a repr, a log or an exception;
  * a failed open leaves no half-connected pair behind.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest

from lib.credentials import (
    CredentialUnavailable,
    SSHCredential,
    SecretOpener,
    resolve_container_credential,
    resolve_host_credential,
)
from lib.transport.base import TransportError, TransportUnreachable
from lib.transport.hostkey import (
    HostKeyMismatch,
    PinnedHostKeyPolicy,
    TunnelledHostKeyPolicy,
    fingerprint,
)
from lib.transport.plan import (
    DEFAULT_REMOTE_DIR,
    ConnectionPlan,
    PlanError,
    SSHEndpoint,
    build_plan,
)
from lib.transport.ssh import GENERATED_ARTIFACTS, SSHTransport, build_cleanup_command

SECRET_PEM = "-----BEGIN OPENSSH PRIVATE KEY-----\nnotarealkey\n-----END OPENSSH PRIVATE KEY-----"

LOCAL_HOST = {
    "name": "local",
    "type": "local",
    "container_access": {"username": "root", "port": 22},
}
REMOTE_HOST = {
    "name": "lab-01",
    "type": "remote",
    "access": {
        "address": "192.168.0.42", "port": 2222, "username": "simlab",
        "auth_method": "key", "secret": {"alg": "crypto_box_seal",
                                         "key_id": "0" * 16, "ciphertext": "x"},
    },
    "container_access": {"username": "root", "port": 22},
}
RESOURCE = {
    "name": "cooja1",
    "endpoint": {"address": "172.18.0.5", "port": 22},
    "remote_dir": "/opt/contiki-ng/tools/cooja",
}


# ── Connection plan ──────────────────────────────────────────────────────────

class TestBuildPlan:
    def test_local_host_has_no_jump(self):
        plan = build_plan(LOCAL_HOST, RESOURCE)
        # The master-node sits on cooja-net and resolves container names
        # directly, so the local path is what it always was.
        assert plan.jump is None and plan.is_tunnelled is False
        assert plan.target == SSHEndpoint("172.18.0.5", 22)

    def test_remote_host_jumps_through_its_sshd(self):
        plan = build_plan(REMOTE_HOST, RESOURCE)
        assert plan.jump == SSHEndpoint("192.168.0.42", 2222)
        assert plan.target == SSHEndpoint("172.18.0.5", 22)

    def test_endpoint_port_overrides_the_host_default(self):
        host = {**REMOTE_HOST, "container_access": {"username": "root", "port": 22}}
        resource = {**RESOURCE, "endpoint": {"address": "127.0.0.1", "port": 2231}}
        assert build_plan(host, resource).target.port == 2231

    def test_host_container_port_is_the_fallback(self):
        resource = {**RESOURCE, "endpoint": {"address": "127.0.0.1"}}
        host = {**REMOTE_HOST, "container_access": {"username": "root", "port": 2022}}
        assert build_plan(host, resource).target.port == 2022

    def test_resource_without_endpoint_is_refused(self):
        # Never guess an address: discovery resolves it, and a simulation must
        # not be sent somewhere nobody verified.
        with pytest.raises(PlanError, match="discovery pass"):
            build_plan(LOCAL_HOST, {"name": "cooja1"})

    def test_remote_host_without_address_is_refused(self):
        host = {"name": "lab-01", "type": "remote", "access": {}}
        with pytest.raises(PlanError, match="no SSH address"):
            build_plan(host, RESOURCE)

    def test_remote_dir_falls_back_to_the_image_default(self):
        plan = build_plan(LOCAL_HOST, {**RESOURCE, "remote_dir": None})
        assert plan.remote_dir == DEFAULT_REMOTE_DIR

    def test_describe_shows_the_path_and_no_credential(self):
        described = build_plan(REMOTE_HOST, RESOURCE).describe()
        assert "192.168.0.42:2222" in described and "172.18.0.5:22" in described
        assert "simlab" not in described


# ── Credentials ──────────────────────────────────────────────────────────────

class FakeKeystore:
    def __init__(self, plaintext=SECRET_PEM, error=None):
        self._plaintext = plaintext
        self._error = error

    def open(self, envelope):
        if self._error:
            raise self._error
        return self._plaintext


class TestCredentials:
    def test_host_key_credential_is_parsed(self):
        opener = SecretOpener(FakeKeystore())
        with patch("lib.credentials.load_private_key", return_value="PKEY") as load:
            cred = resolve_host_credential(
                REMOTE_HOST["access"], opener, host_name="lab-01")
        load.assert_called_once_with(SECRET_PEM)
        assert cred.pkey == "PKEY" and cred.password is None

    def test_host_password_credential_is_used_verbatim(self):
        access = {**REMOTE_HOST["access"], "auth_method": "password"}
        cred = resolve_host_credential(
            access, SecretOpener(FakeKeystore("hunter2")), host_name="lab-01")
        assert cred.password == "hunter2" and cred.pkey is None

    def test_missing_host_credential_names_the_host_only(self):
        access = {**REMOTE_HOST["access"], "secret": None}
        with pytest.raises(CredentialUnavailable) as exc:
            resolve_host_credential(access, SecretOpener(FakeKeystore()), host_name="lab-01")
        assert "lab-01" in str(exc.value)
        assert SECRET_PEM not in str(exc.value)

    def test_container_credential_falls_back_to_the_image_default(self, monkeypatch):
        # What keeps the bootstrapped local host working with no registration
        # step, and without a password in the database.
        monkeypatch.delenv("COOJA_SSH_PASSWORD", raising=False)
        cred = resolve_container_credential({"username": "root"}, None, host_name="local")
        assert cred == SSHCredential(username="root", password="root")

    def test_container_fallback_is_configurable(self, monkeypatch):
        monkeypatch.setenv("COOJA_SSH_PASSWORD", "changed")
        cred = resolve_container_credential({}, None, host_name="local")
        assert cred.password == "changed" and cred.username == "root"

    def test_stored_container_credential_wins_over_the_fallback(self, monkeypatch):
        monkeypatch.setenv("COOJA_SSH_PASSWORD", "image-default")
        access = {"username": "root", "auth_method": "password",
                  "secret": {"alg": "crypto_box_seal", "key_id": "0" * 16, "ciphertext": "x"}}
        cred = resolve_container_credential(
            access, SecretOpener(FakeKeystore("from-db")), host_name="lab-01")
        assert cred.password == "from-db"

    def test_repr_never_shows_the_secret(self):
        shown = repr(SSHCredential(username="root", password="hunter2"))
        assert "hunter2" not in shown and "auth=password" in shown

    def test_connect_kwargs_disable_implicit_credentials(self):
        kwargs = SSHCredential(username="root", password="x").connect_kwargs()
        # A silent fallback to an agent or ~/.ssh would make a misconfigured
        # host work on a developer machine and fail in deployment.
        assert kwargs["look_for_keys"] is False and kwargs["allow_agent"] is False

    def test_secret_is_not_logged_on_failure(self, caplog):
        opener = SecretOpener(FakeKeystore(SECRET_PEM))
        with caplog.at_level(logging.DEBUG, logger="lib.credentials"):
            resolve_container_credential({}, opener, host_name="local")
        assert SECRET_PEM not in caplog.text


# ── Host key policy ──────────────────────────────────────────────────────────

def fake_key(material: bytes) -> MagicMock:
    key = MagicMock()
    key.asbytes.return_value = material
    return key


class TestHostKeyPolicy:
    def test_first_contact_pins_and_reports(self):
        learned = []
        policy = PinnedHostKeyPolicy(None, learned.append)
        key = fake_key(b"host-key-1")

        policy.missing_host_key(MagicMock(), "192.168.0.42", key)

        assert learned == [fingerprint(key)]
        assert policy.expected == fingerprint(key)

    def test_matching_key_is_accepted_without_a_write(self):
        key = fake_key(b"host-key-1")
        learned = []
        policy = PinnedHostKeyPolicy(fingerprint(key), learned.append)

        policy.missing_host_key(MagicMock(), "192.168.0.42", key)

        # A steady state must not generate a database write per connection.
        assert learned == []

    def test_changed_key_refuses_the_connection(self):
        policy = PinnedHostKeyPolicy(fingerprint(fake_key(b"old")))
        with pytest.raises(HostKeyMismatch) as exc:
            policy.missing_host_key(MagicMock(), "192.168.0.42", fake_key(b"new"))
        assert "192.168.0.42" in str(exc.value)
        assert "clear the pinned fingerprint" in str(exc.value).lower()

    def test_container_key_is_accepted_without_recording(self):
        # Containers are recreated routinely with a new key, and are reached
        # only through a tunnel hop 1 already authenticated.
        client = MagicMock()
        TunnelledHostKeyPolicy().missing_host_key(client, "172.18.0.5", fake_key(b"k"))
        client.get_host_keys.assert_not_called()

    def test_fingerprint_is_openssh_shaped(self):
        fp = fingerprint(fake_key(b"whatever"))
        assert fp.startswith("SHA256:") and not fp.endswith("=")


# ── Opening the transport ────────────────────────────────────────────────────

@pytest.fixture
def ssh_clients():
    """Patch paramiko.SSHClient, handing out one mock per construction."""
    created: list[MagicMock] = []

    def factory():
        client = MagicMock()
        created.append(client)
        return client

    with patch("lib.transport.ssh.paramiko.SSHClient", side_effect=factory), \
         patch("lib.transport.session.paramiko.SSHClient", side_effect=factory):
        yield created


def local_transport() -> SSHTransport:
    return SSHTransport(
        build_plan(LOCAL_HOST, RESOURCE),
        target_credential=SSHCredential(username="root", password="root"),
    )


def remote_transport() -> SSHTransport:
    return SSHTransport(
        build_plan(REMOTE_HOST, RESOURCE),
        target_credential=SSHCredential(username="root", password="root"),
        jump_credential=SSHCredential(username="simlab", password="k"),
    )


class TestOpenLocal:
    def test_single_connection_and_no_tunnel(self, ssh_clients):
        transport = local_transport()
        transport.open()

        assert len(ssh_clients) == 1
        client = ssh_clients[0]
        client.connect.assert_called_once()
        assert client.connect.call_args.args[0] == "172.18.0.5"
        assert client.connect.call_args.kwargs["port"] == 22
        # No jump means no channel: byte-for-byte the previous behaviour.
        assert "sock" not in client.connect.call_args.kwargs or \
               client.connect.call_args.kwargs["sock"] is None

    def test_open_is_idempotent(self, ssh_clients):
        transport = local_transport()
        transport.open()
        transport.open()
        assert len(ssh_clients) == 1


class TestOpenRemote:
    def test_connects_the_host_then_tunnels_to_the_container(self, ssh_clients):
        transport = remote_transport()
        transport.open()

        assert len(ssh_clients) == 2
        jump, target = ssh_clients
        assert jump.connect.call_args.args[0] == "192.168.0.42"
        assert jump.connect.call_args.kwargs["port"] == 2222
        assert jump.connect.call_args.kwargs["username"] == "simlab"

        channel = jump.get_transport.return_value.open_channel
        channel.assert_called_once()
        assert channel.call_args.kwargs["dest_addr"] == ("172.18.0.5", 22)

        # The container is reached over that channel, not over the network.
        assert target.connect.call_args.kwargs["sock"] is channel.return_value
        assert target.connect.call_args.kwargs["username"] == "root"

    def test_keepalive_is_set_on_the_tunnel(self, ssh_clients):
        # A simulation runs for hours; without this an idle NAT or hop drops
        # the channel and the run is lost.
        remote_transport().open()
        jump = ssh_clients[0]
        jump.get_transport.return_value.set_keepalive.assert_called_with(30)

    def test_remote_without_jump_credential_is_refused(self):
        with pytest.raises(TransportError, match="host credential"):
            SSHTransport(
                build_plan(REMOTE_HOST, RESOURCE),
                target_credential=SSHCredential(username="root", password="x"),
            )

    def test_unreachable_host_never_attempts_the_tunnel(self, ssh_clients):
        transport = remote_transport()
        # Patched where ssh.py bound the name, not where it is defined.
        with patch("lib.transport.ssh.connect_host",
                   side_effect=TransportUnreachable("no route to host")):
            with pytest.raises(TransportUnreachable):
                transport.open()
        # Unreachable is kept its own failure kind because it says nothing
        # about work already running there — and no second client is built.
        assert ssh_clients == []

    def test_failed_open_leaves_nothing_connected(self, ssh_clients):
        transport = remote_transport()
        with patch.object(SSHTransport, "_connect_target",
                          side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                transport.open()

        # The jump must have been closed: a half-open pair leaks a descriptor
        # and makes a reservation look live while nothing is connected.
        assert ssh_clients[0].close.called
        with pytest.raises(TransportError, match="not open"):
            transport.exec_command("true")


# ── Cleanup: unchanged guarantees, now on the transport ──────────────────────

class TestCleanupCommand:
    def test_generated_artifacts_always_included(self):
        cmd = build_cleanup_command("/opt/cooja")
        assert cmd.startswith("cd /opt/cooja && rm -rf -- ")
        for artifact in GENERATED_ARTIFACTS:
            assert artifact in cmd

    @pytest.mark.parametrize("unsafe", ["/etc/passwd", "../../etc/passwd", ".."])
    def test_names_that_escape_are_refused(self, unsafe):
        # Names come from GridFS documents, so this is the boundary that keeps
        # a crafted file name from deleting what the image shipped.
        assert unsafe not in build_cleanup_command("/opt/cooja", [unsafe])

    def test_uploaded_files_are_removed_too(self):
        cmd = build_cleanup_command("/opt/cooja", ["simulation.csc", "node.c"])
        assert "simulation.csc" in cmd and "node.c" in cmd

    def test_shell_metacharacters_are_quoted(self):
        cmd = build_cleanup_command("/opt/cooja", ["a b;reboot.c"])
        assert "'a b;reboot.c'" in cmd


class TestCleanupIsBestEffort:
    """Cleanup must never abort the simulation flow.

    Ported from the single-hop implementation these behaviours were written
    for: the command composition moved to the transport, but a failure to tidy
    up is still reported through the return value rather than raised.
    """

    def _transport_with(self, result=None, error=None) -> SSHTransport:
        transport = local_transport()
        if error is not None:
            patcher = patch.object(SSHTransport, "exec_command", side_effect=error)
        else:
            patcher = patch.object(SSHTransport, "exec_command", return_value=result)
        self._patcher = patcher
        patcher.start()
        return transport

    def teardown_method(self):
        patcher = getattr(self, "_patcher", None)
        if patcher is not None:
            patcher.stop()
            self._patcher = None

    def test_success_runs_the_composed_command(self):
        from lib.transport.base import ExecResult
        transport = self._transport_with(result=ExecResult(0, "", ""))

        assert transport.cleanup(["node.c"]) is True

        sent = SSHTransport.exec_command.call_args.args[0]
        assert sent == build_cleanup_command(transport.plan.remote_dir, ["node.c"])

    def test_nonzero_exit_returns_false(self):
        from lib.transport.base import ExecResult
        transport = self._transport_with(
            result=ExecResult(1, "", "rm: permission denied"))
        assert transport.cleanup() is False

    def test_transport_failure_is_swallowed(self):
        transport = self._transport_with(error=TransportError("connection lost"))
        assert transport.cleanup() is False
