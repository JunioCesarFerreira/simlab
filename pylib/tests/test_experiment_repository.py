"""Tests for ExperimentRepository artifact cleanup on delete.

The delete cascade must remove every GridFS artifact owned exclusively by an
experiment (topology pictures, simulation position/config/log/csv files and
analysis files) while leaving shared artifacts — source-code repositories —
untouched.  A live MongoDB / GridFS is out of scope here, so we exercise the
pure helpers and invariants that guarantee that contract.
"""
from bson import ObjectId

from pylib.db.repositories.experiment import ExperimentRepository


class _FakeGridFS:
    """Minimal GridFS stand-in recording deletions and simulating failures."""

    def __init__(self, missing: set[ObjectId] | None = None):
        self.deleted: list[ObjectId] = []
        self._missing = missing or set()

    def delete(self, file_id: ObjectId) -> None:
        if file_id in self._missing:
            raise RuntimeError("no such file")
        self.deleted.append(file_id)


class TestDeleteFiles:
    def test_deletes_each_valid_id_once(self):
        a, b = ObjectId(), ObjectId()
        fs = _FakeGridFS()
        # b is duplicated and must only be deleted once
        deleted = ExperimentRepository._delete_files(fs, [a, b, b])
        assert deleted == 2
        assert sorted(fs.deleted, key=str) == sorted([a, b], key=str)

    def test_skips_none_and_non_objectid(self):
        valid = ObjectId()
        fs = _FakeGridFS()
        deleted = ExperimentRepository._delete_files(fs, [None, "", valid, 123])
        assert deleted == 1
        assert fs.deleted == [valid]

    def test_tolerates_missing_files(self):
        present, missing = ObjectId(), ObjectId()
        fs = _FakeGridFS(missing={missing})
        # missing raises inside fs.delete but must not abort the loop
        deleted = ExperimentRepository._delete_files(fs, [missing, present])
        assert deleted == 1
        assert fs.deleted == [present]


class TestSharedArtifactExclusion:
    def test_source_repository_not_collected(self):
        # Shared source code must never be part of the per-experiment purge.
        assert "source_repository_id" not in ExperimentRepository._SIM_FILE_FIELDS

    def test_owned_simulation_files_collected(self):
        for field in ("pos_file_id", "csc_file_id",
                      "log_cooja_id", "runtime_log_id", "csv_log_id"):
            assert field in ExperimentRepository._SIM_FILE_FIELDS


class TestFirmwareSnapshotCleanup:
    """The firmware copies belong to one experiment and die with it."""

    def _snapshot(self, copy_a, copy_b, origin):
        return {"repositories": [
            {"name": "csma-fw", "files": [
                {"file_name": "node.c", "file_id": copy_a, "origin_file_id": origin},
            ]},
            {"name": "tsch-fw", "files": [
                {"file_name": "node.c", "file_id": copy_b, "origin_file_id": origin},
            ]},
        ]}

    def test_collects_every_copy(self):
        copy_a, copy_b, origin = ObjectId(), ObjectId(), ObjectId()
        ids = ExperimentRepository._collect_firmware_file_ids(
            self._snapshot(copy_a, copy_b, origin)
        )
        assert ids == [copy_a, copy_b]

    def test_never_collects_the_shared_original(self):
        copy_a, copy_b, origin = ObjectId(), ObjectId(), ObjectId()
        ids = ExperimentRepository._collect_firmware_file_ids(
            self._snapshot(copy_a, copy_b, origin)
        )
        # Deleting origin_file_id would corrupt the shared source repository.
        assert origin not in ids

    def test_ignores_missing_files_without_a_copy(self):
        copy_a = ObjectId()
        snapshot = {"repositories": [{
            "name": "fw",
            "files": [{"file_name": "node.c", "file_id": copy_a}],
            "missing_files": [{"file_name": "root.c", "origin_file_id": ObjectId()}],
        }]}
        assert ExperimentRepository._collect_firmware_file_ids(snapshot) == [copy_a]

    def test_tolerates_absent_or_empty_snapshots(self):
        assert ExperimentRepository._collect_firmware_file_ids(None) == []
        assert ExperimentRepository._collect_firmware_file_ids({}) == []
        assert ExperimentRepository._collect_firmware_file_ids(
            {"status": "skipped", "repositories": []}
        ) == []
