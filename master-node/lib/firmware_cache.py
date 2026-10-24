"""Reusing a container's compiled firmware, and clearing it when it changes.

Cooja compiles every ``ContikiMoteType`` when it loads ``simulation.csc``: it
always runs ``make``, and the library it links is named after an identifier
drawn at random per JVM (``build/cooja/mtype<random>.cooja``). What survives
between runs is ``build/cooja/obj``, and ``make`` reuses those objects only if
the sources are older than them. Sending the firmware again — SCP stamps the
upload time — makes everything that includes ``project-conf.h`` stale, which
is the whole of Contiki-NG. So reuse here means *not sending the firmware*,
and letting ``make`` relink from the objects already there.

Measured on the Cooja image, rpl-udp-csma: a full build is 0.3s at -j16 and
3s at -j1 when warm, 8.9s cold; a relink from ``obj`` is 0.12s, and the
library it produces is byte-identical to the one a clean build produces.

What makes it safe is the manifest the container keeps next to the firmware.
It names the files of the firmware in place, their fingerprint, and a state:

``pending``
    Written *before* anything is sent and again before each run starts. The
    firmware in place has not yet produced a confirmed result, so it is never
    reused. A run that fails, times out or is cut short leaves this behind,
    and the next run rebuilds from nothing.
``valid``
    Written only after a run built from these files was confirmed and
    published. The only state in which the next run may skip the upload — and
    even then the bytes on the container are checked against the ones about
    to be sent.

The fingerprint covers the content of every firmware file and every
``<motetype>`` of the ``.csc`` except its ``<mote>`` children: the source,
the compile commands and the mote interfaces, which Cooja turns into extra
``make`` flags. ``make`` does not track flags, so a changed command with the
same sources would otherwise link stale objects.

When the firmware is not reused, the previous one is removed *completely*
before the new one is sent: the files the manifest names, ``build/cooja``,
and any top-level C source, header or Makefile — the image ships none, so a
firmware left behind by a crash before the manifest existed is caught too.
"""
import hashlib
import logging
import re
import shlex
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional

from lib.detached import RUN_ARTIFACTS
from lib.transport import GENERATED_ARTIFACTS

log = logging.getLogger(__name__)

#: The per-run inputs. Everything else a simulation sends is firmware.
CSC_NAME = "simulation.csc"
POSITIONS_NAME = "positions.dat"
SIMULATION_INPUTS = (CSC_NAME, POSITIONS_NAME)

MANIFEST_NAME = "simlab-firmware.manifest"
MANIFEST_HEADER = "simlab-firmware 1"
STATE_PENDING = "pending"
STATE_VALID = "valid"

#: Where Cooja builds. ``obj`` is what reuse keeps; the libraries next to it
#: are one per JVM and are removed after every run.
BUILD_DIR = "build/cooja"
#: The run's outputs apart from the build, from the transport's own list so
#: the two cannot drift.
RUN_OUTPUTS = tuple(a for a in GENERATED_ARTIFACTS if a != BUILD_DIR)

#: Top-level files that can only have come from a firmware. Checked against
#: the image: the Cooja directory ships no C source, header or Makefile at its
#: top level, so removing these cannot touch anything Cooja needs.
LEFTOVER_PATTERNS = ("*.c", "*.h", "Makefile", "Makefile.*", "*.mk")

_MARK = "SIMLABFW"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_+][A-Za-z0-9._+-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FirmwareStateError(RuntimeError):
    """The container's firmware could not be put in a known state.

    Raised instead of carrying on, because carrying on is exactly what this
    module exists to prevent: a run built on top of what a previous firmware
    left behind.
    """


def safe_name(name: str) -> bool:
    """Whether a name may be used in a remote ``rm`` and written to a manifest.

    Stricter than ``build_cleanup_command``: no path at all, no whitespace,
    no leading dot or dash. Names come from GridFS documents and from a file
    on the container, so neither is trusted.
    """
    return bool(_SAFE_NAME.match(name or "")) and ".." not in name


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config_digest(csc_path: str) -> Optional[str]:
    """Hash what of the ``.csc`` decides how the firmware is built.

    Every ``<motetype>`` with its ``<mote>`` children removed: those carry the
    positions and ids of one individual and are the only part the mo-engine
    changes per simulation. ``None`` when the file cannot be read, which turns
    reuse off for the run rather than guessing.
    """
    try:
        root = ET.parse(csc_path).getroot()
    except Exception as error:
        log.warning("Could not read %s for the firmware fingerprint: %s",
                    csc_path, error)
        return None
    mote_types = root.findall("./simulation/motetype")
    if not mote_types:
        return None
    digest = hashlib.sha256()
    for mote_type in mote_types:
        stripped = ET.Element(mote_type.tag, mote_type.attrib)
        stripped.text = mote_type.text
        for child in mote_type:
            if child.tag != "mote":
                stripped.append(child)
        digest.update(ET.tostring(stripped, encoding="utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class FirmwareSet:
    """The firmware one simulation sends, as the container will hold it."""
    #: ``(remote name, sha256)``, sorted by name.
    files: tuple[tuple[str, str], ...]
    #: ``None`` when it could not be established; the firmware is then
    #: tracked for cleanup but never reused.
    fingerprint: Optional[str]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.files)


def describe_firmware(
    local_files: list[str], remote_files: list[str]
) -> FirmwareSet:
    """Split a simulation's inputs into firmware and fingerprint it.

    A name that is not :func:`safe_name` disables reuse for the run and is
    left out of the manifest: it is not something a remote ``rm`` may be
    given.
    """
    csc_local: Optional[str] = None
    files: list[tuple[str, str]] = []
    reusable = True
    for local, remote in zip(local_files, remote_files):
        if remote == CSC_NAME:
            csc_local = local
            continue
        if remote in SIMULATION_INPUTS:
            continue
        if not safe_name(remote):
            log.warning("Firmware file %r has an unsafe name; not reusing "
                        "this firmware", remote)
            reusable = False
            continue
        files.append((remote, _sha256_file(local)))
    files.sort()

    build = build_config_digest(csc_local) if csc_local else None
    if not files or build is None or not reusable:
        return FirmwareSet(files=tuple(files), fingerprint=None)
    digest = hashlib.sha256()
    digest.update(f"{MANIFEST_HEADER}\nbuild {build}\n".encode())
    for name, sha in files:
        digest.update(f"file {sha} {name}\n".encode())
    return FirmwareSet(files=tuple(files), fingerprint=digest.hexdigest())


@dataclass(frozen=True)
class Manifest:
    state: str
    fingerprint: Optional[str]
    names: tuple[str, ...]


def render_manifest(state: str, firmware: FirmwareSet) -> list[str]:
    lines = [MANIFEST_HEADER, f"state {state}",
             f"fingerprint {firmware.fingerprint or '-'}"]
    lines.extend(f"file {name}" for name in firmware.names)
    return lines


def parse_manifest(text: str) -> Optional[Manifest]:
    """Read a manifest found on the container, trusting nothing in it.

    Anything malformed reads as no manifest at all, which costs a rebuild and
    nothing else. Unsafe names are dropped rather than failing the whole
    file, so the safe ones can still be cleared.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines or lines[0] != MANIFEST_HEADER:
        return None
    state: Optional[str] = None
    fingerprint: Optional[str] = None
    names: list[str] = []
    for line in lines[1:]:
        key, _, value = line.partition(" ")
        if key == "state":
            state = value
        elif key == "fingerprint":
            fingerprint = value if _SHA256.match(value) else None
        elif key == "file":
            if safe_name(value):
                names.append(value)
            else:
                log.warning("Ignoring unsafe name %r in the firmware manifest", value)
    if state not in (STATE_PENDING, STATE_VALID):
        return None
    return Manifest(state=state, fingerprint=fingerprint, names=tuple(names))


def _quoted(names) -> str:
    return " ".join(shlex.quote(name) for name in names)


def _write_manifest(state: str, firmware: FirmwareSet) -> str:
    """Replace the manifest atomically: a reader sees the old or the new one."""
    tmp = f"{MANIFEST_NAME}.tmp"
    lines = _quoted(render_manifest(state, firmware))
    return (f"printf '%s\\n' {lines} > {shlex.quote(tmp)} && "
            f"mv -f -- {shlex.quote(tmp)} {shlex.quote(MANIFEST_NAME)}")


def _remove_libraries() -> str:
    # Unquoted on purpose: the glob is the point, and it is a fixed string.
    return f"rm -f -- {BUILD_DIR}/*.cooja"


def build_inspect_command(remote_dir: str, names: tuple[str, ...]) -> str:
    """One round trip: the manifest, and the checksums of the files in place."""
    sums = (f"sha256sum -- {_quoted(names)} 2>/dev/null; " if names else "")
    return (
        f"cd {shlex.quote(remote_dir)} || exit 3; "
        f"echo {_MARK}-manifest; "
        f"cat {shlex.quote(MANIFEST_NAME)} 2>/dev/null; "
        f"echo {_MARK}-sums; {sums}"
        f"echo {_MARK}-end"
    )


def parse_inspect_output(stdout: str) -> tuple[Optional[Manifest], dict[str, str], bool]:
    """``(manifest, {name: sha256}, complete)``.

    ``complete`` is false when the end marker never arrived; nothing in an
    answer cut short may be relied on.
    """
    section = None
    manifest_lines: list[str] = []
    sums: dict[str, str] = {}
    complete = False
    for line in (stdout or "").splitlines():
        if line == f"{_MARK}-manifest":
            section = "manifest"
        elif line == f"{_MARK}-sums":
            section = "sums"
        elif line == f"{_MARK}-end":
            complete = True
            break
        elif section == "manifest":
            manifest_lines.append(line)
        elif section == "sums":
            sha, _, name = line.partition("  ")
            if _SHA256.match(sha) and name:
                sums[name] = sha
    return parse_manifest("\n".join(manifest_lines)), sums, complete


def build_reset_command(
    remote_dir: str, previous: tuple[str, ...], firmware: FirmwareSet
) -> str:
    """Remove every trace of the previous firmware, then record the new one.

    The new manifest is written last, as ``pending``, and before anything is
    sent: if the upload dies halfway, the next run still knows which names to
    clear. The previous manifest is never removed on its own before the files
    it names — overwriting it is the final step.
    """
    names = sorted(set(previous) | set(firmware.names)
                   | set(SIMULATION_INPUTS) | set(RUN_OUTPUTS))
    leftovers = " -o ".join(f"-name {shlex.quote(p)}" for p in LEFTOVER_PATTERNS)
    return (
        f"cd {shlex.quote(remote_dir)} && "
        f"rm -rf -- {BUILD_DIR} {_quoted(names)} && "
        f"find . -maxdepth 1 -type f \\( {leftovers} \\) -exec rm -f -- {{}} + && "
        f"{_write_manifest(STATE_PENDING, firmware)}"
    )


def build_reuse_command(remote_dir: str, firmware: FirmwareSet) -> str:
    """Consume the validation, and clear what the previous run left.

    ``pending`` first: from this instant the firmware is unproven again, so a
    run that fails leaves it to be rebuilt rather than reused.
    """
    names = sorted(set(SIMULATION_INPUTS) | set(RUN_OUTPUTS))
    return (
        f"cd {shlex.quote(remote_dir)} && "
        f"{_write_manifest(STATE_PENDING, firmware)} && "
        f"rm -rf -- {_quoted(names)} && {_remove_libraries()}"
    )


def build_validate_command(
    remote_dir: str, firmware: FirmwareSet, inputs: list[str]
) -> str:
    """After a confirmed run: clear the run, keep the firmware, mark it valid."""
    names = sorted(set(inputs) | set(RUN_OUTPUTS) | set(RUN_ARTIFACTS))
    names = [name for name in names if safe_name(name)]
    return (
        f"cd {shlex.quote(remote_dir)} && "
        f"rm -rf -- {_quoted(names)} && {_remove_libraries()} && "
        f"{_write_manifest(STATE_VALID, firmware)}"
    )


@dataclass(frozen=True)
class FirmwarePlacement:
    """What :meth:`FirmwareCache.place` decided for one run."""
    firmware: FirmwareSet
    reused: bool
    #: What still has to be sent.
    local_files: list[str]
    remote_files: list[str]
    #: The per-run inputs, for the cleanup after the run.
    inputs: list[str]


class FirmwareCache:
    """Decide, per run, whether the firmware on a container may be reused.

    ``reuse=False`` keeps the complete cleanup and the manifest and never
    skips an upload — the switch that turns the optimisation off without
    turning off the guarantee.
    """

    def __init__(self, *, reuse: bool = True, timeout_sec: float = 60.0):
        self._reuse = reuse
        self._timeout = timeout_sec

    def place(
        self, transport: Any, local_files: list[str], remote_files: list[str]
    ) -> FirmwarePlacement:
        """Put the container in a known state for this run.

        Raises :class:`FirmwareStateError` when that state could not be
        reached; nothing has been sent at that point.
        """
        remote_dir = transport.plan.remote_dir
        where = transport.plan.describe()
        firmware = describe_firmware(local_files, remote_files)
        inputs = [r for r in remote_files if r in SIMULATION_INPUTS]

        manifest, sums, complete = self._inspect(transport, remote_dir, firmware)
        reason = self._refusal(firmware, manifest, sums, complete)
        if reason is None:
            self._run(transport, build_reuse_command(remote_dir, firmware),
                      "prepare the firmware in place for reuse")
            log.info(
                "[%s] Reusing the firmware in place (%d files, fingerprint "
                "%s…); sending only the simulation inputs",
                where, len(firmware.files), firmware.fingerprint[:12],
            )
            pairs = [(l, r) for l, r in zip(local_files, remote_files)
                     if r in SIMULATION_INPUTS]
            return FirmwarePlacement(
                firmware=firmware, reused=True,
                local_files=[l for l, _ in pairs],
                remote_files=[r for _, r in pairs],
                inputs=inputs,
            )

        previous = manifest.names if manifest else ()
        self._run(transport, build_reset_command(remote_dir, previous, firmware),
                  "clear the previous firmware")
        log.info(
            "[%s] Not reusing the firmware (%s); cleared %d previous file(s) "
            "and %s, sending %d", where, reason, len(previous), BUILD_DIR,
            len(remote_files),
        )
        return FirmwarePlacement(
            firmware=firmware, reused=False,
            local_files=list(local_files), remote_files=list(remote_files),
            inputs=inputs,
        )

    def keep(self, transport: Any, placement: FirmwarePlacement) -> bool:
        """After a confirmed, published run: keep the firmware for the next one.

        Best-effort, like every cleanup after a result: ``False`` leaves the
        manifest ``pending``, which costs the next run a rebuild and nothing
        else.
        """
        command = build_validate_command(
            transport.plan.remote_dir, placement.firmware, placement.inputs)
        try:
            result = transport.exec_command(command, timeout=self._timeout)
        except Exception as error:
            log.warning("Could not keep the firmware on %s: %s",
                        transport.plan.describe(), error)
            return False
        if not result.ok:
            log.warning("Keeping the firmware exited with status %s on %s: %s",
                        result.exit_status, transport.plan.describe(),
                        (result.stderr or "").strip())
            return False
        return True

    def _inspect(self, transport: Any, remote_dir: str, firmware: FirmwareSet):
        try:
            result = transport.exec_command(
                build_inspect_command(remote_dir, firmware.names),
                timeout=self._timeout)
        except Exception as error:
            # Not fatal: an unanswered question is a rebuild. If the container
            # is really unreachable, the reset that follows says so.
            log.warning("Could not inspect the firmware on %s: %s",
                        transport.plan.describe(), error)
            return None, {}, False
        return parse_inspect_output(result.stdout)

    def _refusal(
        self,
        firmware: FirmwareSet,
        manifest: Optional[Manifest],
        sums: dict[str, str],
        complete: bool,
    ) -> Optional[str]:
        """Why the firmware in place may not be reused; ``None`` if it may."""
        if not self._reuse:
            return "reuse is disabled"
        if firmware.fingerprint is None:
            return "this firmware has no fingerprint"
        if not complete:
            return "the container did not answer completely"
        if manifest is None:
            return "no manifest on the container"
        if manifest.state != STATE_VALID:
            return "the firmware in place never produced a confirmed result"
        if manifest.fingerprint != firmware.fingerprint:
            return "a different firmware is in place"
        if tuple(sorted(manifest.names)) != firmware.names:
            return "the manifest names other files"
        for name, sha in firmware.files:
            if sums.get(name) != sha:
                return f"{name} on the container does not match"
        return None

    def _run(self, transport: Any, command: str, what: str) -> None:
        try:
            result = transport.exec_command(command, timeout=self._timeout)
        except Exception as error:
            raise FirmwareStateError(
                f"could not {what} on {transport.plan.describe()}: {error}"
            ) from error
        if not result.ok:
            raise FirmwareStateError(
                f"could not {what} on {transport.plan.describe()} (status "
                f"{result.exit_status}): {(result.stderr or '').strip()}"
            )
