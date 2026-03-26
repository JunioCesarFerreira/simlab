from bson import ObjectId

from tests.conftest import EXP_ID, GEN_ID, SIM_ID, FILE_ID, sample_simulation

BASE = "/api/v1/simulations"

_VALID_PAYLOAD = {
    "id": "",
    "experiment_id": EXP_ID,
    "generation_id": GEN_ID,
    "individual_id": "abc123hash",
    "status": "Waiting",
    "system_message": "",
    "random_seed": 42,
    "start_time": None,
    "end_time": None,
    "parameters": {
        "name": "sim-0",
        "duration": 120,
        "randomSeed": 42,
        "radiusOfReach": 100.0,
        "radiusOfInter": 50.0,
        "region": [0.0, 0.0, 100.0, 100.0],
        "simulationElements": {"fixedMotes": [], "mobileMotes": []},
    },
    "pos_file_id": "",
    "csc_file_id": "",
    "source_repository_id": "",
    "log_cooja_id": "",
    "runtime_log_id": "",
    "csv_log_id": "",
    "network_metrics": {},
}


# ── POST / ─────────────────────────────────────────────────────────────────────
class TestCreateSimulation:
    def test_returns_simulation_id(self, client, mock_factory):
        mock_factory.simulation_repo.insert.return_value = ObjectId(SIM_ID)
        resp = client.post(f"{BASE}/", json=_VALID_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json() == SIM_ID

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.simulation_repo.insert.side_effect = RuntimeError("db error")
        resp = client.post(f"{BASE}/", json=_VALID_PAYLOAD)
        assert resp.status_code == 500


# ── GET /by-status/{status} ───────────────────────────────────────────────────
class TestGetSimulationsByStatus:
    def test_returns_list(self, client, mock_factory):
        mock_factory.simulation_repo.find_by_status.return_value = [sample_simulation()]
        resp = client.get(f"{BASE}/by-status/Waiting")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == SIM_ID
        assert data[0]["status"] == "Waiting"
        assert data[0]["individual_id"] == "abc123hash"

    def test_empty_list(self, client, mock_factory):
        mock_factory.simulation_repo.find_by_status.return_value = []
        resp = client.get(f"{BASE}/by-status/Done")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.simulation_repo.find_by_status.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/by-status/Waiting")
        assert resp.status_code == 500


# ── GET /{sim_id} ─────────────────────────────────────────────────────────────
class TestGetSimulation:
    def test_found(self, client, mock_factory):
        mock_factory.simulation_repo.get.return_value = sample_simulation()
        resp = client.get(f"{BASE}/{SIM_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == SIM_ID

    def test_not_found_returns_404(self, client, mock_factory):
        mock_factory.simulation_repo.get.return_value = None
        resp = client.get(f"{BASE}/{SIM_ID}")
        assert resp.status_code == 404

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.simulation_repo.get.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/{SIM_ID}")
        assert resp.status_code == 500


# ── PUT /{sim_id} ─────────────────────────────────────────────────────────────
class TestUpdateSimulation:
    def test_success(self, client, mock_factory):
        mock_factory.simulation_repo.update.return_value = True
        resp = client.put(f"{BASE}/{SIM_ID}", json={"status": "Running"})
        assert resp.status_code == 200
        assert resp.json() is True


# ── DELETE /{sim_id} ──────────────────────────────────────────────────────────
class TestDeleteSimulation:
    def test_success(self, client, mock_factory):
        mock_factory.simulation_repo.delete_by_id.return_value = True
        resp = client.delete(f"{BASE}/{SIM_ID}")
        assert resp.status_code == 200
        assert resp.json() is True

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.simulation_repo.delete_by_id.side_effect = RuntimeError("db error")
        resp = client.delete(f"{BASE}/{SIM_ID}")
        assert resp.status_code == 500


# ── GET /{sim_id}/file/{field_name} ───────────────────────────────────────────
class TestDownloadSimulationFile:
    def test_simulation_not_found_returns_404(self, client, mock_factory):
        mock_factory.simulation_repo.get.return_value = None
        resp = client.get(f"{BASE}/{SIM_ID}/file/log_cooja_id")
        assert resp.status_code == 404

    def test_field_empty_returns_404(self, client, mock_factory):
        sim = sample_simulation()
        sim["log_cooja_id"] = ""
        mock_factory.simulation_repo.get.return_value = sim
        resp = client.get(f"{BASE}/{SIM_ID}/file/log_cooja_id")
        assert resp.status_code == 404

    def test_field_none_returns_404(self, client, mock_factory):
        sim = sample_simulation()
        sim["missing_field"] = None
        mock_factory.simulation_repo.get.return_value = sim
        resp = client.get(f"{BASE}/{SIM_ID}/file/missing_field")
        assert resp.status_code == 404

    def test_success(self, client, mock_factory):
        from pathlib import Path

        sim = sample_simulation()
        sim["log_cooja_id"] = FILE_ID
        mock_factory.simulation_repo.get.return_value = sim

        def fake_download(oid, path):
            Path(path).write_bytes(b"LOG_CONTENT")

        mock_factory.fs_handler.download_file.side_effect = fake_download

        resp = client.get(f"{BASE}/{SIM_ID}/file/log_cooja_id")
        assert resp.status_code == 200


# ── PATCH /{sim_id}/status ────────────────────────────────────────────────────
class TestUpdateSimulationStatus:
    def test_success(self, client, mock_factory):
        mock_factory.simulation_repo.update_status.return_value = None
        resp = client.patch(f"{BASE}/{SIM_ID}/status", params={"new_status": "Running"})
        assert resp.status_code == 200
        assert resp.json() is True

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.simulation_repo.update_status.side_effect = RuntimeError("db error")
        resp = client.patch(f"{BASE}/{SIM_ID}/status", params={"new_status": "Running"})
        assert resp.status_code == 500
