import io
from pathlib import Path
from bson import ObjectId

from tests.conftest import SRC_ID, FILE_ID, sample_source

BASE = "/api/v1/sources"


# ── GET / ─────────────────────────────────────────────────────────────────────
class TestListSources:
    def test_returns_list(self, client, mock_factory):
        mock_factory.source_repo.get_all.return_value = [sample_source()]
        resp = client.get(f"{BASE}/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test Source"
        assert data[0]["id"] == SRC_ID

    def test_empty_list(self, client, mock_factory):
        mock_factory.source_repo.get_all.return_value = []
        resp = client.get(f"{BASE}/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.source_repo.get_all.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/")
        assert resp.status_code == 500


# ── GET /{repository_id} ──────────────────────────────────────────────────────
class TestGetSourceRepository:
    def test_found(self, client, mock_factory):
        mock_factory.source_repo.get_by_id.return_value = sample_source()
        resp = client.get(f"{BASE}/{SRC_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == SRC_ID
        assert resp.json()["description"] == "A test source repository"

    def test_not_found_returns_404(self, client, mock_factory):
        mock_factory.source_repo.get_by_id.return_value = None
        resp = client.get(f"{BASE}/{SRC_ID}")
        assert resp.status_code == 404

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.source_repo.get_by_id.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/{SRC_ID}")
        assert resp.status_code == 500


# ── POST / ────────────────────────────────────────────────────────────────────
class TestCreateSourceRepository:
    def test_success(self, client, mock_factory):
        mock_factory.fs_handler.upload_file.return_value = ObjectId(FILE_ID)
        mock_factory.source_repo.insert.return_value = ObjectId(SRC_ID)

        resp = client.post(
            f"{BASE}/",
            data={"name": "My Source", "description": "desc"},
            files={"files": ("main.c", io.BytesIO(b"int main(){}"), "text/plain")},
        )

        assert resp.status_code == 200
        assert resp.json() == SRC_ID
        mock_factory.fs_handler.upload_file.assert_called_once()
        mock_factory.source_repo.insert.assert_called_once()

    def test_multiple_files(self, client, mock_factory):
        mock_factory.fs_handler.upload_file.side_effect = [ObjectId(FILE_ID), ObjectId(FILE_ID)]
        mock_factory.source_repo.insert.return_value = ObjectId(SRC_ID)

        resp = client.post(
            f"{BASE}/",
            data={"name": "Multi Source"},
            files=[
                ("files", ("a.c", io.BytesIO(b"void a(){}"), "text/plain")),
                ("files", ("b.c", io.BytesIO(b"void b(){}"), "text/plain")),
            ],
        )

        assert resp.status_code == 200
        assert mock_factory.fs_handler.upload_file.call_count == 2

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.fs_handler.upload_file.side_effect = RuntimeError("gridfs error")
        resp = client.post(
            f"{BASE}/",
            data={"name": "Source"},
            files={"files": ("main.c", io.BytesIO(b"code"), "text/plain")},
        )
        assert resp.status_code == 500


# ── GET /{repository_id}/download ─────────────────────────────────────────────
class TestDownloadSourceRepository:
    def test_success(self, client, mock_factory):
        mock_factory.source_repo.get_by_id.return_value = sample_source()

        def fake_download(oid, path):
            Path(path).write_bytes(b"C SOURCE CODE")

        mock_factory.fs_handler.download_file.side_effect = fake_download

        resp = client.get(f"{BASE}/{SRC_ID}/download")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_not_found_returns_404(self, client, mock_factory):
        mock_factory.source_repo.get_by_id.return_value = None
        resp = client.get(f"{BASE}/{SRC_ID}/download")
        assert resp.status_code == 404

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.source_repo.get_by_id.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/{SRC_ID}/download")
        assert resp.status_code == 500
