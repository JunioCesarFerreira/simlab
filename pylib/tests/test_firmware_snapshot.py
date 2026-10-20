"""Tests for the per-experiment firmware capture.

The capture exists so a finished experiment keeps the exact firmware it ran
with, even after the shared source repository is edited or deleted. These
tests pin the contract that matters for traceability: every referenced file is
copied, degradation is reported instead of hidden, the capture never raises,
and it never runs twice for the same experiment.
"""
import hashlib
from types import SimpleNamespace

from bson import ObjectId

from pylib.firmware_snapshot import (
    STATUS_CAPTURED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_SKIPPED,
    ARTIFACT_KIND,
    capture_firmware_snapshot,
    iter_snapshot_files,
)

EXP_ID = "507f1f77bcf86cd799439011"
REPO_A = "507f1f77bcf86cd799439021"
REPO_B = "507f1f77bcf86cd799439022"


class _FakeExperimentRepo:
    """Records the persisted block and simulates the atomic claim."""

    def __init__(self, already_claimed: bool = False, claim_raises: bool = False):
        self.claimed = already_claimed
        self.claim_raises = claim_raises
        self.saved: dict | None = None
        self.claims: list[str] = []

    def claim_firmware_snapshot(self, experiment_id: str, claimed_at) -> bool:
        if self.claim_raises:
            raise RuntimeError("mongo down")
        self.claims.append(experiment_id)
        if self.claimed:
            return False
        self.claimed = True
        return True

    def set_firmware_snapshot(self, experiment_id: str, block: dict) -> bool:
        self.saved = block
        return True


class _FakeSourceRepo:
    def __init__(self, docs: dict[str, dict] | None = None, raises: bool = False):
        self.docs = docs or {}
        self.raises = raises

    def get_by_id(self, repository_id: str):
        if self.raises:
            raise RuntimeError("mongo down")
        return self.docs.get(str(repository_id))


class _FakeGridFS:
    """In-memory GridFS: unknown ids raise, uploads get a fresh ObjectId."""

    def __init__(self, contents: dict[str, bytes] | None = None):
        self.contents = contents or {}
        self.uploaded: list[tuple[bytes, str, dict | None]] = []

    def read_file_content(self, file_id: str) -> bytes:
        if str(file_id) not in self.contents:
            raise RuntimeError(f"no such file {file_id}")
        return self.contents[str(file_id)]

    def upload_bytes(self, data: bytes, name: str, metadata=None) -> ObjectId:
        self.uploaded.append((data, name, metadata))
        return ObjectId()


def _mongo(experiment_repo=None, source_repo=None, fs_handler=None):
    return SimpleNamespace(
        experiment_repo=experiment_repo or _FakeExperimentRepo(),
        source_repo=source_repo or _FakeSourceRepo(),
        fs_handler=fs_handler or _FakeGridFS(),
    )


def _source_doc(name: str, files: list[tuple[str, str]]) -> dict:
    return {
        "name": name,
        "description": f"{name} description",
        "source_files": [{"id": fid, "file_name": fname} for fid, fname in files],
    }


def _experiment(options: dict) -> dict:
    return {"_id": ObjectId(EXP_ID), "source_repository_options": options}


class TestSuccessfulCapture:
    def _run(self):
        contents = {"f1": b"int main(void) {}", "f2": b"CONTIKI_PROJECT = node\n"}
        source_repo = _FakeSourceRepo({REPO_A: _source_doc(
            "rpl-udp-csma", [("f1", "node.c"), ("f2", "Makefile")]
        )})
        fs = _FakeGridFS(contents)
        exp_repo = _FakeExperimentRepo()
        mongo = _mongo(exp_repo, source_repo, fs)
        block = capture_firmware_snapshot(mongo, _experiment({"csma": REPO_A}))
        return block, exp_repo, fs, contents

    def test_status_and_persistence(self):
        block, exp_repo, _, _ = self._run()
        assert block["status"] == STATUS_CAPTURED
        assert exp_repo.saved == block
        assert block["schema_version"] == 1
        assert block["captured_at"] is not None

    def test_every_file_is_copied(self):
        block, _, fs, contents = self._run()
        files = block["repositories"][0]["files"]
        assert [f["file_name"] for f in files] == ["node.c", "Makefile"]
        # Copies are new GridFS ids, distinct from the shared originals.
        assert all(isinstance(f["file_id"], ObjectId) for f in files)
        assert [str(f["origin_file_id"]) for f in files] != [str(f["file_id"]) for f in files]
        assert [data for data, _, _ in fs.uploaded] == [contents["f1"], contents["f2"]]

    def test_records_size_and_digest(self):
        block, _, _, contents = self._run()
        node_c = block["repositories"][0]["files"][0]
        assert node_c["size_bytes"] == len(contents["f1"])
        assert node_c["sha256"] == hashlib.sha256(contents["f1"]).hexdigest()

    def test_copies_carry_provenance_metadata(self):
        _, _, fs, _ = self._run()
        _, _, metadata = fs.uploaded[0]
        assert metadata["kind"] == ARTIFACT_KIND
        assert metadata["experiment_id"] == ObjectId(EXP_ID)
        assert metadata["source_repository_id"] == ObjectId(REPO_A)

    def test_repository_metadata_is_frozen(self):
        block, _, _, _ = self._run()
        repo = block["repositories"][0]
        assert repo["name"] == "rpl-udp-csma"
        assert repo["description"] == "rpl-udp-csma description"
        assert repo["option_keys"] == ["csma"]
        assert repo["source_repository_id"] == ObjectId(REPO_A)


class TestRepositoryDeduplication:
    def test_shared_repository_is_copied_once(self):
        fs = _FakeGridFS({"f1": b"code"})
        source_repo = _FakeSourceRepo({REPO_A: _source_doc("shared", [("f1", "node.c")])})
        mongo = _mongo(_FakeExperimentRepo(), source_repo, fs)

        block = capture_firmware_snapshot(
            mongo, _experiment({"csma": REPO_A, "tsch": REPO_A})
        )

        assert len(block["repositories"]) == 1
        assert len(fs.uploaded) == 1
        # Both mappings are still recorded, so the resolution stays auditable.
        assert block["repositories"][0]["option_keys"] == ["csma", "tsch"]

    def test_distinct_repositories_are_both_captured(self):
        fs = _FakeGridFS({"f1": b"a", "f2": b"b"})
        source_repo = _FakeSourceRepo({
            REPO_A: _source_doc("csma-fw", [("f1", "node.c")]),
            REPO_B: _source_doc("tsch-fw", [("f2", "node.c")]),
        })
        mongo = _mongo(_FakeExperimentRepo(), source_repo, fs)

        block = capture_firmware_snapshot(
            mongo, _experiment({"csma": REPO_A, "tsch": REPO_B})
        )

        assert block["status"] == STATUS_CAPTURED
        assert sorted(r["name"] for r in block["repositories"]) == ["csma-fw", "tsch-fw"]
        assert len(fs.uploaded) == 2


class TestDegradedCapture:
    def test_unreadable_file_is_reported_as_partial(self):
        fs = _FakeGridFS({"f1": b"code"})  # f2 is absent from GridFS
        source_repo = _FakeSourceRepo({REPO_A: _source_doc(
            "fw", [("f1", "node.c"), ("f2", "root.c")]
        )})
        mongo = _mongo(_FakeExperimentRepo(), source_repo, fs)

        block = capture_firmware_snapshot(mongo, _experiment({"csma": REPO_A}))

        assert block["status"] == STATUS_PARTIAL
        repo = block["repositories"][0]
        assert [f["file_name"] for f in repo["files"]] == ["node.c"]
        assert [f["file_name"] for f in repo["missing_files"]] == ["root.c"]

    def test_deleted_repository_is_reported_as_partial(self):
        mongo = _mongo(_FakeExperimentRepo(), _FakeSourceRepo({}), _FakeGridFS())

        block = capture_firmware_snapshot(mongo, _experiment({"csma": REPO_A}))

        assert block["status"] == STATUS_PARTIAL
        repo = block["repositories"][0]
        assert repo["files"] == []
        assert REPO_A in repo["name"]
        assert repo["source_repository_id"] == ObjectId(REPO_A)

    def test_unreachable_source_collection_is_reported_as_partial(self):
        mongo = _mongo(_FakeExperimentRepo(), _FakeSourceRepo(raises=True), _FakeGridFS())

        block = capture_firmware_snapshot(mongo, _experiment({"csma": REPO_A}))

        assert block["status"] == STATUS_PARTIAL


class TestSkippedCapture:
    def test_experiment_without_sources_is_skipped(self):
        exp_repo = _FakeExperimentRepo()
        mongo = _mongo(exp_repo)

        block = capture_firmware_snapshot(mongo, _experiment({}))

        assert block["status"] == STATUS_SKIPPED
        assert block["repositories"] == []
        assert block["reason"]
        assert exp_repo.saved == block

    def test_blank_option_values_are_ignored(self):
        fs = _FakeGridFS()
        mongo = _mongo(_FakeExperimentRepo(), _FakeSourceRepo({}), fs)

        block = capture_firmware_snapshot(mongo, _experiment({"csma": None, "tsch": ""}))

        assert block["status"] == STATUS_CAPTURED
        assert block["repositories"] == []
        assert fs.uploaded == []


class TestFailureIsolation:
    def test_gridfs_write_error_is_recorded_not_raised(self):
        class _ReadOnlyGridFS(_FakeGridFS):
            def upload_bytes(self, data, name, metadata=None):
                raise RuntimeError("boom")

        exp_repo = _FakeExperimentRepo()
        source_repo = _FakeSourceRepo({REPO_A: _source_doc("fw", [("f1", "node.c")])})
        mongo = _mongo(exp_repo, source_repo, _ReadOnlyGridFS({"f1": b"code"}))

        block = capture_firmware_snapshot(mongo, _experiment({"csma": REPO_A}))

        assert block["status"] == STATUS_FAILED
        assert "boom" in block["error"]
        assert exp_repo.saved["status"] == STATUS_FAILED

    def test_malformed_experiment_document_is_recorded_not_raised(self):
        exp_repo = _FakeExperimentRepo()
        mongo = _mongo(exp_repo)

        # A corrupted options field must not take the experiment down with it.
        block = capture_firmware_snapshot(
            mongo, {"_id": ObjectId(EXP_ID), "source_repository_options": 42}
        )

        assert block["status"] == STATUS_FAILED
        assert exp_repo.saved["status"] == STATUS_FAILED

    def test_claim_failure_does_not_raise(self):
        mongo = _mongo(_FakeExperimentRepo(claim_raises=True))
        assert capture_firmware_snapshot(mongo, _experiment({"csma": REPO_A})) is None

    def test_experiment_without_id_is_skipped(self):
        exp_repo = _FakeExperimentRepo()
        mongo = _mongo(exp_repo)
        assert capture_firmware_snapshot(mongo, {"source_repository_options": {}}) is None
        assert exp_repo.saved is None


class TestIdempotency:
    def test_second_capture_is_a_no_op(self):
        fs = _FakeGridFS({"f1": b"code"})
        source_repo = _FakeSourceRepo({REPO_A: _source_doc("fw", [("f1", "node.c")])})
        exp_repo = _FakeExperimentRepo()
        mongo = _mongo(exp_repo, source_repo, fs)
        experiment = _experiment({"csma": REPO_A})

        first = capture_firmware_snapshot(mongo, experiment)
        second = capture_firmware_snapshot(mongo, experiment)

        assert first is not None
        # A restarted engine must not duplicate the copies in GridFS.
        assert second is None
        assert len(fs.uploaded) == 1

    def test_accepts_the_string_id_variant(self):
        exp_repo = _FakeExperimentRepo()
        mongo = _mongo(exp_repo)
        capture_firmware_snapshot(mongo, {"id": EXP_ID, "source_repository_options": {}})
        assert exp_repo.claims == [EXP_ID]


class TestIterSnapshotFiles:
    def test_yields_repository_name_with_each_file(self):
        snapshot = {"repositories": [
            {"name": "csma-fw", "files": [{"file_name": "node.c"}, {"file_name": "root.c"}]},
            {"name": "tsch-fw", "files": [{"file_name": "node.c"}]},
        ]}
        assert [(name, f["file_name"]) for name, f in iter_snapshot_files(snapshot)] == [
            ("csma-fw", "node.c"), ("csma-fw", "root.c"), ("tsch-fw", "node.c"),
        ]

    def test_skips_missing_files_and_tolerates_empty_input(self):
        snapshot = {"repositories": [
            {"name": "fw", "files": [], "missing_files": [{"file_name": "gone.c"}]},
        ]}
        assert list(iter_snapshot_files(snapshot)) == []
        assert list(iter_snapshot_files(None)) == []
        assert list(iter_snapshot_files({})) == []

    def test_unnamed_repository_gets_a_positional_folder(self):
        snapshot = {"repositories": [{"name": "", "files": [{"file_name": "node.c"}]}]}
        assert [name for name, _ in iter_snapshot_files(snapshot)] == ["repository_0"]
