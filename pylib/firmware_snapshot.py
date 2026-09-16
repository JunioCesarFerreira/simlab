"""Firmware traceability: freeze the firmware an experiment was executed with.

Source repositories (collection ``sources``) are shared between experiments and
remain editable after a run: files can be replaced, removed, or the whole
repository deleted. A finished experiment would then reference firmware that is
no longer the one it ran. To keep results auditable, every referenced file is
copied into GridFS when the experiment starts, under the sole ownership of that
experiment, and recorded in the ``firmware_snapshot`` block of its document.

The capture is deliberately **best-effort**: a failure is persisted as
``failed`` on the experiment (and surfaced in the GUI) but never prevents the
run from starting — traceability must not become a new way for experiments to
die. The copies are removed by ``ExperimentRepository.delete`` together with
the experiment; the shared originals are never touched.
"""
import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

from bson import ObjectId, errors as bson_errors

from pylib.db import MongoRepository
from pylib.db.models import FirmwareFile, FirmwareRepositorySnapshot, FirmwareSnapshot

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

STATUS_CAPTURED = "captured"
STATUS_PARTIAL = "partial"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"

# GridFS marker on every copy, so the artifacts can be audited (or swept) by
# provenance rather than by guessing from the file name.
ARTIFACT_KIND = "firmware_snapshot"

_NO_SOURCES_REASON = "Experiment references no source repository (e.g. synthetic run)."


def capture_firmware_snapshot(
    mongo: MongoRepository, experiment: dict[str, Any]
) -> Optional[FirmwareSnapshot]:
    """Copy the experiment's firmware into GridFS and record the snapshot.

    Returns the persisted block, or ``None`` when the capture was not performed
    (experiment without a usable id, or a snapshot already claimed by an
    earlier start — the claim makes this idempotent across engine restarts).
    Never raises.
    """
    experiment_id = _experiment_id(experiment)
    if not experiment_id:
        log.warning("[firmware] Experiment without a usable id; capture skipped.")
        return None

    try:
        if not mongo.experiment_repo.claim_firmware_snapshot(experiment_id, datetime.now()):
            log.info("[firmware] Snapshot already present for experiment %s.", experiment_id)
            return None
    except Exception:
        log.exception("[firmware] Failed to claim snapshot for experiment %s", experiment_id)
        return None

    try:
        block = _build_snapshot(mongo, experiment_id, experiment)
    except Exception as e:  # pragma: no cover - defensive, see module docstring
        log.exception("[firmware] Capture failed for experiment %s", experiment_id)
        block = _block(STATUS_FAILED, repositories=[], error=str(e))

    try:
        mongo.experiment_repo.set_firmware_snapshot(experiment_id, dict(block))
    except Exception:
        log.exception("[firmware] Failed to persist snapshot for experiment %s", experiment_id)
        return None

    log.info(
        "[firmware] Snapshot %s for experiment %s (%d repositories, %d files).",
        block.get("status"), experiment_id,
        len(block.get("repositories") or []), _file_count(block),
    )
    return block


def iter_snapshot_files(snapshot: Optional[dict[str, Any]]):
    """Yield ``(repository_name, file_entry)`` for every captured copy.

    Shared by the API endpoints that serve a single file or the whole ZIP, so
    both agree on what belongs to a snapshot. ``missing_files`` are excluded:
    they have no copy to serve.
    """
    for index, repo in enumerate((snapshot or {}).get("repositories") or []):
        repo = repo or {}
        name = str(repo.get("name") or "") or f"repository_{index}"
        for entry in repo.get("files") or []:
            if entry:
                yield name, entry


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _experiment_id(experiment: dict[str, Any]) -> str:
    raw = experiment.get("_id") or experiment.get("id") or ""
    return str(raw)


def _build_snapshot(
    mongo: MongoRepository, experiment_id: str, experiment: dict[str, Any]
) -> FirmwareSnapshot:
    options: dict[str, Any] = experiment.get("source_repository_options") or {}
    if not options:
        return _block(STATUS_SKIPPED, repositories=[], reason=_NO_SOURCES_REASON)

    repositories: list[FirmwareRepositorySnapshot] = []
    degraded = False
    # One entry per distinct repository: several options (one per MAC protocol)
    # routinely point at the same firmware and must not be copied twice.
    for repo_id, option_keys in _distinct_repositories(options).items():
        entry, entry_degraded = _capture_repository(mongo, experiment_id, repo_id, option_keys)
        repositories.append(entry)
        degraded = degraded or entry_degraded

    status = STATUS_PARTIAL if degraded else STATUS_CAPTURED
    return _block(status, repositories=repositories)


def _distinct_repositories(options: dict[str, Any]) -> dict[str, list[str]]:
    """Map repository id -> the option keys resolving to it, order preserved."""
    grouped: dict[str, list[str]] = {}
    for key in sorted(options):
        value = options[key]
        if value in (None, ""):
            continue
        grouped.setdefault(str(value), []).append(str(key))
    return grouped


def _capture_repository(
    mongo: MongoRepository, experiment_id: str, repo_id: str, option_keys: list[str]
) -> tuple[FirmwareRepositorySnapshot, bool]:
    entry: FirmwareRepositorySnapshot = {
        "option_keys": option_keys,
        "source_repository_id": _as_object_id(repo_id) or repo_id,
        "name": "",
        "description": "",
        "files": [],
        "missing_files": [],
    }

    try:
        doc = mongo.source_repo.get_by_id(repo_id)
    except Exception as e:
        log.warning("[firmware] Could not read source repository %s: %s", repo_id, e)
        doc = None

    if not doc:
        # The repository was deleted between experiment creation and start.
        log.warning("[firmware] Source repository %s not found for experiment %s",
                    repo_id, experiment_id)
        entry["name"] = f"<missing repository {repo_id}>"
        return entry, True

    entry["name"] = str(doc.get("name") or "")
    entry["description"] = str(doc.get("description") or "")

    degraded = False
    for source_file in doc.get("source_files") or []:
        file_name = str(source_file.get("file_name") or "")
        origin_id = source_file.get("id")
        captured = _copy_file(mongo, experiment_id, repo_id, file_name, origin_id)
        if captured is None:
            entry["missing_files"].append({
                "file_name": file_name,
                "origin_file_id": _as_object_id(origin_id) or origin_id,
            })
            degraded = True
        else:
            entry["files"].append(captured)

    return entry, degraded


def _copy_file(
    mongo: MongoRepository, experiment_id: str, repo_id: str,
    file_name: str, origin_id: Any,
) -> Optional[FirmwareFile]:
    """Duplicate one source file into the experiment's own GridFS space."""
    if origin_id in (None, ""):
        return None
    try:
        content: bytes = mongo.fs_handler.read_file_content(str(origin_id))
    except Exception as e:
        log.warning("[firmware] Source file %s (%s) unreadable: %s", origin_id, file_name, e)
        return None

    copy_id = mongo.fs_handler.upload_bytes(
        content, file_name,
        metadata={
            "kind": ARTIFACT_KIND,
            "experiment_id": _as_object_id(experiment_id) or experiment_id,
            "source_repository_id": _as_object_id(repo_id) or repo_id,
            "origin_file_id": _as_object_id(origin_id) or origin_id,
        },
    )
    return {
        "file_name": file_name,
        "file_id": copy_id,
        "origin_file_id": _as_object_id(origin_id) or origin_id,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _block(status: str, *, repositories: list[FirmwareRepositorySnapshot],
           **extra: Any) -> FirmwareSnapshot:
    block: FirmwareSnapshot = {
        "status": status,
        "captured_at": datetime.now(),
        "schema_version": SCHEMA_VERSION,
        "repositories": repositories,
    }
    block.update(extra)  # type: ignore[typeddict-item]
    return block


def _file_count(block: FirmwareSnapshot) -> int:
    return sum(len(r.get("files") or []) for r in block.get("repositories") or [])


def _as_object_id(value: Any) -> Optional[ObjectId]:
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(value)
    except (bson_errors.InvalidId, TypeError):
        return None
