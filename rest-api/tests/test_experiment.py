import io
from bson import ObjectId, errors as bson_errors

from tests.conftest import (
    EXP_ID, GEN_ID, IND_ID,
    sample_experiment, sample_generation, sample_individual,
)

BASE = "/api/v1/experiments"

_VALID_PAYLOAD = {
    "id": None,
    "name": "Test Experiment",
    "status": "Waiting",
    "system_message": "",
    "created_time": None,
    "start_time": None,
    "end_time": None,
    "parameters": {
        "strategy": "nsga3",
        "algorithm": {},
        "simulation": {},
        "problem": {},
        "objectives": [],
    },
    "source_repository_options": {},
    "data_conversion_config": {"node_col": "node", "time_col": "time", "metrics": []},
    "pareto_front": None,
}


# ── GET /{experiment_id}/full ─────────────────────────────────────────────────
class TestGetExperimentFull:
    def test_returns_experiment_with_generations_and_individuals(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = sample_experiment()
        mock_factory.generation_repo.find_by_experiment.return_value = [sample_generation()]
        mock_factory.individual_repo.find_by_generation.return_value = [sample_individual()]

        resp = client.get(f"{BASE}/{EXP_ID}/full")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == EXP_ID
        assert data["name"] == "Test Experiment"
        assert len(data["generations"]) == 1
        assert data["generations"][0]["id"] == GEN_ID
        assert len(data["generations"][0]["population"]) == 1
        assert data["generations"][0]["population"][0]["individual_id"] == "abc123hash"

    def test_generations_empty_when_none_exist(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = sample_experiment()
        mock_factory.generation_repo.find_by_experiment.return_value = []

        resp = client.get(f"{BASE}/{EXP_ID}/full")

        assert resp.status_code == 200
        assert resp.json()["generations"] == []

    def test_not_found_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = None
        resp = client.get(f"{BASE}/{EXP_ID}/full")
        assert resp.status_code == 404

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.get.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/{EXP_ID}/full")
        assert resp.status_code == 500


# ── POST / ─────────────────────────────────────────────────────────────────────
class TestCreateExperiment:
    def test_returns_experiment_id(self, client, mock_factory):
        mock_factory.experiment_repo.insert.return_value = ObjectId(EXP_ID)
        resp = client.post(f"{BASE}/", json=_VALID_PAYLOAD)
        assert resp.status_code == 200
        assert resp.json() == EXP_ID

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.insert.side_effect = RuntimeError("db error")
        resp = client.post(f"{BASE}/", json=_VALID_PAYLOAD)
        assert resp.status_code == 500


# ── GET /by-status/{status} ────────────────────────────────────────────────────
class TestGetExperimentsByStatus:
    def test_returns_list(self, client, mock_factory):
        mock_factory.experiment_repo.find_by_status.return_value = [sample_experiment()]
        resp = client.get(f"{BASE}/by-status/Waiting")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "Test Experiment"

    def test_empty_list(self, client, mock_factory):
        mock_factory.experiment_repo.find_by_status.return_value = []
        resp = client.get(f"{BASE}/by-status/Done")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.find_by_status.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/by-status/Waiting")
        assert resp.status_code == 500


# ── GET /{experiment_id} ───────────────────────────────────────────────────────
class TestGetExperiment:
    def test_found(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = sample_experiment()
        resp = client.get(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == EXP_ID
        assert resp.json()["name"] == "Test Experiment"

    def test_not_found_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = None
        resp = client.get(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 404

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.get.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 500


# ── PUT /{experiment_id} ───────────────────────────────────────────────────────
class TestUpdateExperiment:
    def test_success(self, client, mock_factory):
        mock_factory.experiment_repo.update.return_value = True
        resp = client.put(f"{BASE}/{EXP_ID}", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json() is True

    def test_not_modified(self, client, mock_factory):
        mock_factory.experiment_repo.update.return_value = False
        resp = client.put(f"{BASE}/{EXP_ID}", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json() is False


# ── DELETE /{experiment_id} ────────────────────────────────────────────────────
class TestDeleteExperiment:
    def test_success_dict_result(self, client, mock_factory):
        mock_factory.experiment_repo.delete.return_value = {"deleted_experiments": 1}
        resp = client.delete(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 200
        assert resp.json() is True

    def test_success_bool_result(self, client, mock_factory):
        mock_factory.experiment_repo.delete.return_value = True
        resp = client.delete(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 200
        assert resp.json() is True

    def test_invalid_id_returns_400(self, client, mock_factory):
        mock_factory.experiment_repo.delete.side_effect = bson_errors.InvalidId
        resp = client.delete(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 400

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.delete.side_effect = RuntimeError("db error")
        resp = client.delete(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 500


# ── PATCH /{experiment_id}/status ─────────────────────────────────────────────
class TestUpdateExperimentStatus:
    def test_success(self, client, mock_factory):
        mock_factory.experiment_repo.update_status.return_value = None
        resp = client.patch(f"{BASE}/{EXP_ID}/status", params={"new_status": "Running"})
        assert resp.status_code == 200
        assert resp.json() is True

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.update_status.side_effect = RuntimeError("db error")
        resp = client.patch(f"{BASE}/{EXP_ID}/status", params={"new_status": "Running"})
        assert resp.status_code == 500


# ── PATCH /{experiment_id}/analysis-file ──────────────────────────────────────
class TestAttachAnalysisFile:
    def test_success(self, client, mock_factory):
        mock_factory.experiment_repo.add_analysis_file_to_experiment.return_value = ObjectId(EXP_ID)
        resp = client.patch(
            f"{BASE}/{EXP_ID}/analysis-file",
            data={"name": "pareto", "description": "Pareto front"},
            files={"file": ("pareto.png", io.BytesIO(b"PNG"), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json() == EXP_ID

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.add_analysis_file_to_experiment.side_effect = RuntimeError
        resp = client.patch(
            f"{BASE}/{EXP_ID}/analysis-file",
            data={"name": "pareto"},
            files={"file": ("pareto.png", io.BytesIO(b"PNG"), "image/png")},
        )
        assert resp.status_code == 500
