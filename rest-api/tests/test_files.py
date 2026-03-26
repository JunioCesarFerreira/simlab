from pathlib import Path
from bson import ObjectId
from gridfs.errors import NoFile

from tests.conftest import EXP_ID, GEN_ID, SIM_ID, FILE_ID, sample_experiment, sample_generation, sample_individual

BASE = "/api/v1/files"


def _fake_download(content: bytes):
    """Returns a side_effect function that writes `content` to the given path."""
    def _write(oid, path):
        Path(path).write_bytes(content)
    return _write


# ── GET /{file_id}/as/{extension} ─────────────────────────────────────────────
class TestDownloadFile:
    def test_success_txt(self, client, mock_factory):
        mock_factory.fs_handler.download_file.side_effect = _fake_download(b"LOG DATA")
        resp = client.get(f"{BASE}/{FILE_ID}/as/txt")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_success_csv(self, client, mock_factory):
        mock_factory.fs_handler.download_file.side_effect = _fake_download(b"a,b,c\n1,2,3")
        resp = client.get(f"{BASE}/{FILE_ID}/as/csv")
        assert resp.status_code == 200

    def test_invalid_id_returns_400(self, client, mock_factory):
        resp = client.get(f"{BASE}/not-an-objectid/as/txt")
        assert resp.status_code == 400

    def test_file_not_found_returns_404(self, client, mock_factory):
        mock_factory.fs_handler.download_file.side_effect = NoFile
        resp = client.get(f"{BASE}/{FILE_ID}/as/txt")
        assert resp.status_code == 404

    def test_gridfs_error_returns_500(self, client, mock_factory):
        mock_factory.fs_handler.download_file.side_effect = RuntimeError("gridfs error")
        resp = client.get(f"{BASE}/{FILE_ID}/as/txt")
        assert resp.status_code == 500


# ── GET /simulations/{simulation_id}/topology ─────────────────────────────────
class TestDownloadTopologyBySimulation:
    def test_success(self, client, mock_factory):
        mock_factory.simulation_repo.get_topology_pic_file_id.return_value = ObjectId(FILE_ID)
        mock_factory.fs_handler.download_file.side_effect = _fake_download(b"\x89PNG\r\n")
        resp = client.get(f"{BASE}/simulations/{SIM_ID}/topology")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_invalid_sim_id_returns_400(self, client, mock_factory):
        resp = client.get(f"{BASE}/simulations/not-valid/topology")
        assert resp.status_code == 400

    def test_topology_not_found_returns_404(self, client, mock_factory):
        mock_factory.simulation_repo.get_topology_pic_file_id.return_value = None
        resp = client.get(f"{BASE}/simulations/{SIM_ID}/topology")
        assert resp.status_code == 404

    def test_gridfs_file_missing_returns_404(self, client, mock_factory):
        mock_factory.simulation_repo.get_topology_pic_file_id.return_value = ObjectId(FILE_ID)
        mock_factory.fs_handler.download_file.side_effect = NoFile
        resp = client.get(f"{BASE}/simulations/{SIM_ID}/topology")
        assert resp.status_code == 404

    def test_gridfs_error_returns_500(self, client, mock_factory):
        mock_factory.simulation_repo.get_topology_pic_file_id.return_value = ObjectId(FILE_ID)
        mock_factory.fs_handler.download_file.side_effect = RuntimeError("error")
        resp = client.get(f"{BASE}/simulations/{SIM_ID}/topology")
        assert resp.status_code == 500


# ── GET /experiments/{experiment_id}/analysis/zip ─────────────────────────────
class TestDownloadAnalysisZip:
    def test_invalid_id_returns_400(self, client, mock_factory):
        resp = client.get(f"{BASE}/experiments/bad-id/analysis/zip")
        assert resp.status_code == 400

    def test_no_analysis_files_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.find_analysis_files.return_value = {}
        resp = client.get(f"{BASE}/experiments/{EXP_ID}/analysis/zip")
        assert resp.status_code == 404

    def test_none_analysis_files_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.find_analysis_files.return_value = None
        resp = client.get(f"{BASE}/experiments/{EXP_ID}/analysis/zip")
        assert resp.status_code == 404

    def test_success(self, client, mock_factory):
        mock_factory.experiment_repo.find_analysis_files.return_value = {
            "pareto_front": FILE_ID
        }
        mock_factory.fs_handler.download_file.side_effect = _fake_download(b"\x89PNG\r\n")

        resp = client.get(f"{BASE}/experiments/{EXP_ID}/analysis/zip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    def test_gridfs_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.find_analysis_files.return_value = {"file": FILE_ID}
        mock_factory.fs_handler.download_file.side_effect = RuntimeError("error")
        resp = client.get(f"{BASE}/experiments/{EXP_ID}/analysis/zip")
        assert resp.status_code == 500


# ── GET /experiments/{experiment_id}/topologies/zip ───────────────────────────
class TestDownloadTopologiesZip:
    def test_invalid_id_returns_400(self, client, mock_factory):
        resp = client.get(f"{BASE}/experiments/bad-id/topologies/zip")
        assert resp.status_code == 400

    def test_experiment_not_found_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = None
        resp = client.get(f"{BASE}/experiments/{EXP_ID}/topologies/zip")
        assert resp.status_code == 404

    def test_no_generations_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = sample_experiment()
        mock_factory.generation_repo.find_by_experiment.return_value = []
        resp = client.get(f"{BASE}/experiments/{EXP_ID}/topologies/zip")
        assert resp.status_code == 404

    def test_no_topologies_downloaded_returns_404(self, client, mock_factory):
        gen = sample_generation()
        ind = sample_individual()
        ind["topology_picture_id"] = None  # no topology stored

        mock_factory.experiment_repo.get.return_value = sample_experiment()
        mock_factory.generation_repo.find_by_experiment.return_value = [gen]
        mock_factory.individual_repo.find_by_generation.return_value = [ind]

        resp = client.get(f"{BASE}/experiments/{EXP_ID}/topologies/zip")
        assert resp.status_code == 404

    def test_success(self, client, mock_factory):
        gen = sample_generation()
        ind = sample_individual()
        ind["topology_picture_id"] = FILE_ID

        mock_factory.experiment_repo.get.return_value = sample_experiment()
        mock_factory.generation_repo.find_by_experiment.return_value = [gen]
        mock_factory.individual_repo.find_by_generation.return_value = [ind]
        mock_factory.fs_handler.download_file.side_effect = _fake_download(b"\x89PNG\r\n")

        resp = client.get(f"{BASE}/experiments/{EXP_ID}/topologies/zip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
