"""Tests for reusing a container's firmware, and clearing it when it changes.

The commands are *executed* in ``/bin/sh`` against a directory laid out like
the Cooja image, in the style of ``TestProbeCommandAgainstARealShell``: what
matters is what the shell deletes and keeps, not the text of the command.

Reusing the wrong firmware does not fail a run — it produces a result that
belongs to a different program. So most of what is here is the refusals.
"""
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from lib.firmware_cache import (
    MANIFEST_NAME,
    FirmwareCache,
    describe_firmware,
    parse_manifest,
    safe_name,
)
from lib.scheduler.runner import RunReport, SimulationRunner
from lib.transport.base import ExecResult

CSC = """<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <simulation>
    <title>t</title>
    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <description>server</description>
      <source>/opt/contiki-ng/tools/cooja/root.c</source>
      <commands>$(MAKE) -j$(CPUS) root.cooja TARGET=cooja</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      {motes}
    </motetype>
  </simulation>
</simconf>
"""

MOTE = ("<mote><interface_config>org.contikios.cooja.interfaces.Position"
        "<pos x=\"{x}\" y=\"0\" /></interface_config></mote>")

FIRMWARE = {
    "Makefile": "CONTIKI_PROJECT = root node\n",
    "project-conf.h": "#define X 1\n",
    "root.c": "int main(void) { return 0; }\n",
    "node.c": "int main(void) { return 1; }\n",
}


class LocalContainer:
    """A transport whose container is a directory and whose shell is real."""

    def __init__(self, remote_dir: Path):
        self.dir = remote_dir
        self.plan = SimpleNamespace(
            remote_dir=str(remote_dir), describe=lambda: "test/cooja1")
        self.sent: list[str] = []

    def exec_command(self, command, timeout=None) -> ExecResult:
        done = subprocess.run(["/bin/sh", "-c", command], capture_output=True,
                              text=True, timeout=30)
        return ExecResult(done.returncode, done.stdout, done.stderr)

    def put_files(self, local, remote) -> None:
        for source, name in zip(local, remote):
            shutil.copyfile(source, self.dir / name)
            self.sent.append(name)

    def build(self) -> None:
        """What Cooja's ``make`` leaves: objects, and one library per JVM."""
        obj = self.dir / "build/cooja/obj"
        obj.mkdir(parents=True, exist_ok=True)
        (obj / "root.o").write_text("object")
        (self.dir / "build/cooja/mtype123.cooja").write_text("library")
        (self.dir / "COOJA.testlog").write_text("log")

    def exists(self, name: str) -> bool:
        return (self.dir / name).exists()


@pytest.fixture
def container(tmp_path):
    remote = tmp_path / "cooja"
    # What the image ships, and what nothing here may ever touch.
    (remote / "build/libs").mkdir(parents=True)
    (remote / "build/libs/cooja.jar").write_text("jar")
    (remote / "README.md").write_text("cooja readme")
    (remote / "sim-config.csc").write_text("<simconf/>")
    return LocalContainer(remote)


def write_inputs(where: Path, firmware=None, csc=None, motes=1):
    """A simulation's local inputs, as ``prepare_simulation_files`` lays them out."""
    where.mkdir(parents=True, exist_ok=True)
    local, remote = [], []
    csc_text = csc or CSC.format(
        motes="".join(MOTE.format(x=i) for i in range(motes)))
    (where / "sim.xml").write_text(csc_text)
    local.append(str(where / "sim.xml"))
    remote.append("simulation.csc")
    (where / "pos.dat").write_text("0 0\n")
    local.append(str(where / "pos.dat"))
    remote.append("positions.dat")
    for name, text in (firmware or FIRMWARE).items():
        (where / name).write_text(text)
        local.append(str(where / name))
        remote.append(name)
    return local, remote


def run_once(cache, container, local, remote, *, confirmed=True):
    """Place, upload, build and — if confirmed — keep, as the runner does."""
    placement = cache.place(container, local, remote)
    container.put_files(placement.local_files, placement.remote_files)
    container.build()
    if confirmed:
        assert cache.keep(container, placement)
    return placement


def manifest_of(container):
    return parse_manifest((container.dir / MANIFEST_NAME).read_text())


# ── The fingerprint ──────────────────────────────────────────────────────────

class TestFingerprint:
    def test_positions_do_not_change_it(self, tmp_path):
        # The mo-engine rewrites the motes of every individual. If that
        # changed the fingerprint, nothing would ever be reused.
        a = describe_firmware(*write_inputs(tmp_path / "a", motes=2))
        b = describe_firmware(*write_inputs(tmp_path / "b", motes=5))
        assert a.fingerprint is not None
        assert a.fingerprint == b.fingerprint

    def test_a_changed_source_changes_it(self, tmp_path):
        a = describe_firmware(*write_inputs(tmp_path / "a"))
        b = describe_firmware(*write_inputs(
            tmp_path / "b", firmware={**FIRMWARE, "project-conf.h": "#define X 2\n"}))
        assert a.fingerprint != b.fingerprint

    def test_changed_compile_commands_change_it(self, tmp_path):
        # make does not track flags: same sources, different command, and the
        # objects in place are stale without make noticing.
        a = describe_firmware(*write_inputs(tmp_path / "a"))
        csc = CSC.replace("TARGET=cooja", "TARGET=cooja DEFINES=FOO=1").format(motes="")
        b = describe_firmware(*write_inputs(tmp_path / "b", csc=csc))
        assert a.fingerprint != b.fingerprint

    def test_changed_mote_interfaces_change_it(self, tmp_path):
        # Cooja turns the interfaces a mote type leaves out into make flags.
        a = describe_firmware(*write_inputs(tmp_path / "a"))
        csc = CSC.replace(
            "<moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>",
            "").format(motes="")
        b = describe_firmware(*write_inputs(tmp_path / "b", csc=csc))
        assert a.fingerprint != b.fingerprint

    def test_an_unreadable_csc_disables_reuse(self, tmp_path):
        firmware = describe_firmware(*write_inputs(tmp_path, csc="not xml"))
        assert firmware.fingerprint is None
        # Still tracked, so it can be cleared.
        assert set(firmware.names) == set(FIRMWARE)

    def test_an_unsafe_name_disables_reuse_and_is_not_tracked(self, tmp_path):
        firmware = describe_firmware(*write_inputs(
            tmp_path, firmware={**FIRMWARE, "-rf": "x"}))
        assert firmware.fingerprint is None
        assert "-rf" not in firmware.names

    @pytest.mark.parametrize("name", [
        "", ".", "..", "../x.c", "a/b.c", "/etc/passwd", ".hidden", "a b.c",
        "a;rm.c", "x\n.c", "-rf",
    ])
    def test_unsafe_names(self, name):
        assert not safe_name(name)


class TestManifestParsing:
    def test_unknown_header_reads_as_no_manifest(self):
        assert parse_manifest("something else\nstate valid\n") is None

    def test_unknown_state_reads_as_no_manifest(self):
        assert parse_manifest("simlab-firmware 1\nstate maybe\n") is None

    def test_unsafe_names_are_dropped_and_the_rest_kept(self):
        manifest = parse_manifest(
            "simlab-firmware 1\nstate pending\nfingerprint -\n"
            "file ../../etc/passwd\nfile node.c\n")
        assert manifest.names == ("node.c",)
        assert manifest.fingerprint is None


# ── Against a real shell ─────────────────────────────────────────────────────

class TestFirstRun:
    def test_nothing_in_place_sends_everything(self, container, tmp_path):
        local, remote = write_inputs(tmp_path / "in")
        placement = FirmwareCache().place(container, local, remote)
        assert placement.reused is False
        assert placement.remote_files == remote

    def test_the_manifest_is_pending_before_anything_is_sent(self, container, tmp_path):
        # So a run that dies during the upload still leaves the names behind.
        local, remote = write_inputs(tmp_path / "in")
        FirmwareCache().place(container, local, remote)
        manifest = manifest_of(container)
        assert manifest.state == "pending"
        assert set(manifest.names) == set(FIRMWARE)

    def test_what_the_image_ships_survives(self, container, tmp_path):
        local, remote = write_inputs(tmp_path / "in")
        FirmwareCache().place(container, local, remote)
        for kept in ("build/libs/cooja.jar", "README.md", "sim-config.csc"):
            assert container.exists(kept), kept

    def test_leftovers_with_no_manifest_are_removed(self, container, tmp_path):
        # A firmware left by a crash before the manifest existed.
        (container.dir / "old-app.c").write_text("stale")
        (container.dir / "old-conf.h").write_text("stale")
        (container.dir / "Makefile.old").write_text("stale")
        container.build()
        local, remote = write_inputs(tmp_path / "in")
        FirmwareCache().place(container, local, remote)
        for gone in ("old-app.c", "old-conf.h", "Makefile.old", "build/cooja",
                     "COOJA.testlog"):
            assert not container.exists(gone), gone


class TestReuse:
    def test_a_confirmed_run_keeps_the_firmware_and_the_objects(self, container, tmp_path):
        local, remote = write_inputs(tmp_path / "in")
        run_once(FirmwareCache(), container, local, remote)

        assert manifest_of(container).state == "valid"
        for name in FIRMWARE:
            assert container.exists(name), name
        assert container.exists("build/cooja/obj/root.o")
        # Per run, and gone: the inputs, the result and the JVM's library.
        for gone in ("simulation.csc", "positions.dat", "COOJA.testlog",
                     "build/cooja/mtype123.cooja"):
            assert not container.exists(gone), gone

    def test_the_same_firmware_sends_only_the_simulation_inputs(self, container, tmp_path):
        cache = FirmwareCache()
        run_once(cache, container, *write_inputs(tmp_path / "a", motes=2))
        container.sent.clear()

        placement = cache.place(container, *write_inputs(tmp_path / "b", motes=7))
        assert placement.reused is True
        assert placement.remote_files == ["simulation.csc", "positions.dat"]
        assert container.exists("build/cooja/obj/root.o")
        assert not container.exists("build/cooja/mtype123.cooja")

    def test_reuse_consumes_the_validation(self, container, tmp_path):
        # From the moment a run starts, the build is unproven again.
        cache = FirmwareCache()
        local, remote = write_inputs(tmp_path / "in")
        run_once(cache, container, local, remote)
        cache.place(container, local, remote)
        assert manifest_of(container).state == "pending"

    def test_an_unconfirmed_run_is_never_reused(self, container, tmp_path):
        cache = FirmwareCache()
        local, remote = write_inputs(tmp_path / "in")
        run_once(cache, container, local, remote, confirmed=False)

        placement = cache.place(container, local, remote)
        assert placement.reused is False
        assert not container.exists("build/cooja")

    def test_a_failure_after_reuse_forces_a_rebuild(self, container, tmp_path):
        cache = FirmwareCache()
        local, remote = write_inputs(tmp_path / "in")
        run_once(cache, container, local, remote)
        run_once(cache, container, local, remote, confirmed=False)
        assert cache.place(container, local, remote).reused is False

    def test_a_file_changed_on_the_container_is_not_reused(self, container, tmp_path):
        cache = FirmwareCache()
        local, remote = write_inputs(tmp_path / "in")
        run_once(cache, container, local, remote)
        (container.dir / "node.c").write_text("tampered")
        assert cache.place(container, local, remote).reused is False

    def test_a_file_missing_on_the_container_is_not_reused(self, container, tmp_path):
        cache = FirmwareCache()
        local, remote = write_inputs(tmp_path / "in")
        run_once(cache, container, local, remote)
        (container.dir / "root.c").unlink()
        assert cache.place(container, local, remote).reused is False

    def test_a_different_build_command_is_not_reused(self, container, tmp_path):
        cache = FirmwareCache()
        run_once(cache, container, *write_inputs(tmp_path / "a"))
        csc = CSC.replace("TARGET=cooja", "TARGET=cooja DEFINES=FOO=1").format(motes="")
        placement = cache.place(container, *write_inputs(tmp_path / "b", csc=csc))
        assert placement.reused is False
        assert not container.exists("build/cooja")

    def test_disabled_reuse_still_clears_and_never_reuses(self, container, tmp_path):
        cache = FirmwareCache(reuse=False)
        local, remote = write_inputs(tmp_path / "in")
        run_once(cache, container, local, remote)
        placement = cache.place(container, local, remote)
        assert placement.reused is False
        assert placement.remote_files == remote


class TestChangedFirmware:
    def test_the_previous_firmware_is_removed_completely(self, container, tmp_path):
        cache = FirmwareCache()
        previous = {**FIRMWARE, "extra-helper.inc": "only in the old firmware"}
        run_once(cache, container, *write_inputs(tmp_path / "a", firmware=previous))

        new = {"Makefile": "x\n", "project-conf.h": "#define Y\n", "app.c": "int y;\n"}
        placement = cache.place(container, *write_inputs(tmp_path / "b", firmware=new))

        assert placement.reused is False
        # Not a C source or a header, so only the manifest could know of it.
        assert not container.exists("extra-helper.inc")
        for name in ("root.c", "node.c", "build/cooja"):
            assert not container.exists(name), name
        assert set(manifest_of(container).names) == set(new)

    def test_a_pending_manifest_still_names_what_to_clear(self, container, tmp_path):
        # A run that failed after its upload: the names are all that is left.
        cache = FirmwareCache()
        previous = {**FIRMWARE, "notes.txt": "from the failed run"}
        run_once(cache, container, *write_inputs(tmp_path / "a", firmware=previous),
                 confirmed=False)
        cache.place(container, *write_inputs(tmp_path / "b"))
        assert not container.exists("notes.txt")

    def test_a_crafted_manifest_cannot_reach_outside(self, container, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("must survive")
        (container.dir / MANIFEST_NAME).write_text(
            "simlab-firmware 1\nstate pending\nfingerprint -\n"
            "file ../outside.txt\nfile old.c\n")
        (container.dir / "old.c").write_text("stale")
        FirmwareCache().place(container, *write_inputs(tmp_path / "in"))
        assert outside.exists()
        assert not container.exists("old.c")


class TestFailures:
    def test_a_reset_that_fails_raises(self, tmp_path):
        from lib.firmware_cache import FirmwareStateError
        missing = LocalContainer(tmp_path / "does-not-exist")
        with pytest.raises(FirmwareStateError):
            FirmwareCache().place(missing, *write_inputs(tmp_path / "in"))

    def test_an_unanswered_inspection_does_not_reuse(self, container, tmp_path):
        cache = FirmwareCache()
        local, remote = write_inputs(tmp_path / "in")
        run_once(cache, container, local, remote)
        real = container.exec_command
        calls = []

        def flaky(command, timeout=None):
            calls.append(command)
            if len(calls) == 1:
                raise TimeoutError("inspection timed out")
            return real(command, timeout)

        container.exec_command = flaky
        assert cache.place(container, local, remote).reused is False

    def test_keep_that_fails_reports_it(self, container, tmp_path):
        cache = FirmwareCache()
        local, remote = write_inputs(tmp_path / "in")
        placement = cache.place(container, local, remote)
        container.exec_command = MagicMock(side_effect=TimeoutError())
        assert cache.keep(container, placement) is False


# ── The runner ───────────────────────────────────────────────────────────────

SIM = {"_id": ObjectId("507f1f77bcf86cd799430003"), "status": "Waiting",
       "generation_id": ObjectId("507f1f77bcf86cd799430004")}
HOST = {"_id": ObjectId("507f1f77bcf86cd799430011"), "name": "host-a", "type": "local"}
RESOURCE = {"_id": ObjectId("507f1f77bcf86cd799430021"), "name": "cooja1"}
ATTEMPT = "8f14e45fceea167a5a36dedd4bea2543"


def fake_mongo():
    mongo = MagicMock()
    mongo.resource_repo.release.return_value = {"status": "available"}
    mongo.simulation_repo.mark_done.return_value = True
    mongo.simulation_repo.mark_running.return_value = True
    return mongo


class TestRunner:
    @pytest.fixture
    def transport(self, container):
        with patch("lib.scheduler.runner.build_transport") as factory, \
                patch("lib.scheduler.runner.assert_no_live_run"):
            container.close = lambda: None
            container.cleanup = MagicMock(return_value=True)
            factory.return_value = _ContextManager(container)
            yield container

    def _runner(self, mongo, inputs, report):
        def execute(sim, transport, mongo_, remote_files, **kwargs):
            transport.build()
            return report

        return SimulationRunner(
            mongo, execute=execute,
            prepare=lambda sim, repo: (True, *inputs),
            firmware=FirmwareCache(),
        )

    def test_a_confirmed_run_keeps_the_firmware(self, transport, tmp_path):
        mongo = fake_mongo()
        inputs = write_inputs(tmp_path / "in")
        self._runner(mongo, inputs, RunReport(confirmed=True)).run(
            SIM, HOST, RESOURCE, ATTEMPT)
        assert manifest_of(transport).state == "valid"
        # The old full cleanup would have taken build/cooja with it.
        transport.cleanup.assert_not_called()

        transport.sent.clear()
        self._runner(mongo, write_inputs(tmp_path / "again"),
                     RunReport(confirmed=True)).run(SIM, HOST, RESOURCE, ATTEMPT)
        assert transport.sent == ["simulation.csc", "positions.dat"]

    def test_an_unconfirmed_run_clears_as_before(self, transport, tmp_path):
        mongo = fake_mongo()
        inputs = write_inputs(tmp_path / "in")
        self._runner(mongo, inputs, RunReport(confirmed=False, reason="empty")).run(
            SIM, HOST, RESOURCE, ATTEMPT)
        assert transport.cleanup.called
        assert manifest_of(transport).state == "pending"

    def test_a_firmware_state_that_cannot_be_reached_fails_the_run(self, tmp_path):
        mongo = fake_mongo()
        broken = LocalContainer(tmp_path / "does-not-exist")
        broken.close = lambda: None
        with patch("lib.scheduler.runner.build_transport") as factory, \
                patch("lib.scheduler.runner.assert_no_live_run"):
            factory.return_value = _ContextManager(broken)
            execute = MagicMock()
            SimulationRunner(
                mongo, execute=execute,
                prepare=lambda sim, repo: (True, *write_inputs(tmp_path / "in")),
                firmware=FirmwareCache(),
            ).run(SIM, HOST, RESOURCE, ATTEMPT)
        execute.assert_not_called()
        assert broken.sent == []
        assert "previous firmware" in mongo.simulation_repo.mark_error.call_args.args[1]
        mongo.resource_repo.release.assert_called_once()


class _ContextManager:
    """Hand out the container itself, the way ``with transport:`` does."""

    def __init__(self, container):
        self._container = container

    def __enter__(self):
        return self._container

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        return getattr(self._container, name)
