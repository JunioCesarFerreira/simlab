"""Unit tests for the remote Cooja cleanup (lib/sshscp.py).

These lock in the container-hygiene behavior:
  * generated artifacts (build/cooja, COOJA logs) are always removed;
  * uploaded per-run files are removed alongside them;
  * names that could escape remote_dir are refused;
  * cleanup is best-effort and never raises into the simulation flow.
"""
from unittest.mock import MagicMock

from lib.sshscp import (
    GENERATED_ARTIFACTS,
    build_cleanup_command,
    cleanup_remote_files,
)

REMOTE_DIR = "/opt/contiki-ng/tools/cooja"


# ── Command construction ─────────────────────────────────────────────────────

class TestBuildCleanupCommand:
    def test_generated_artifacts_always_included(self):
        cmd = build_cleanup_command(REMOTE_DIR)
        assert cmd.startswith(f"cd {REMOTE_DIR} && rm -rf -- ")
        for artifact in GENERATED_ARTIFACTS:
            assert artifact in cmd

    def test_uploaded_files_appended(self):
        cmd = build_cleanup_command(REMOTE_DIR, ["simulation.csc", "positions.dat", "node.c"])
        for name in ("simulation.csc", "positions.dat", "node.c"):
            assert name in cmd

    def test_absolute_paths_are_refused(self):
        cmd = build_cleanup_command(REMOTE_DIR, ["/etc/passwd", "ok.c"])
        assert "/etc/passwd" not in cmd
        assert "ok.c" in cmd

    def test_parent_traversal_is_refused(self):
        cmd = build_cleanup_command(REMOTE_DIR, ["../../opt/java", "ok.c"])
        assert "opt/java" not in cmd
        assert "ok.c" in cmd

    def test_shell_metacharacters_are_quoted(self):
        cmd = build_cleanup_command(REMOTE_DIR, ["a b;reboot.c"])
        assert "'a b;reboot.c'" in cmd


# ── Remote execution wrapper ─────────────────────────────────────────────────

def _client_with_exit(status: int, stderr_text: str = "") -> MagicMock:
    client = MagicMock()
    stdout = MagicMock()
    stdout.channel.recv_exit_status.return_value = status
    stderr = MagicMock()
    stderr.read.return_value = stderr_text.encode()
    client.exec_command.return_value = (MagicMock(), stdout, stderr)
    return client


class TestCleanupRemoteFiles:
    def test_success_runs_expected_command(self):
        client = _client_with_exit(0)
        assert cleanup_remote_files(client, REMOTE_DIR, ["node.c"]) is True
        (cmd,), _ = client.exec_command.call_args
        assert cmd == build_cleanup_command(REMOTE_DIR, ["node.c"])

    def test_nonzero_exit_returns_false(self):
        client = _client_with_exit(1, "rm: permission denied")
        assert cleanup_remote_files(client, REMOTE_DIR) is False

    def test_ssh_failure_is_swallowed(self):
        client = MagicMock()
        client.exec_command.side_effect = OSError("connection lost")
        assert cleanup_remote_files(client, REMOTE_DIR) is False
