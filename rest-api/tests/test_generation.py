from bson import ObjectId

from tests.conftest import EXP_ID, GEN_ID, sample_generation, sample_individual

BASE = "/api/v1/generations"


# ── GET /by-experiment/{experiment_id} ────────────────────────────────────────
class TestGetGenerationsByExperiment:
    def test_returns_list_with_population(self, client, mock_factory):
        gen = sample_generation()
        ind = sample_individual()
        mock_factory.generation_repo.find_by_experiment.return_value = [gen]
        mock_factory.individual_repo.find_by_generation.return_value = [ind]

        resp = client.get(f"{BASE}/by-experiment/{EXP_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == GEN_ID
        assert data[0]["index"] == 0
        assert data[0]["status"] == "Done"
        assert len(data[0]["population"]) == 1
        assert data[0]["population"][0]["individual_id"] == "abc123hash"

    def test_individual_id_as_int_is_coerced_to_str(self, client, mock_factory):
        """individual_id stored as int (python hash) must be serialised as str."""
        gen = sample_generation()
        ind = sample_individual()
        ind["individual_id"] = 3099568720531170764   # int hash
        mock_factory.generation_repo.find_by_experiment.return_value = [gen]
        mock_factory.individual_repo.find_by_generation.return_value = [ind]

        resp = client.get(f"{BASE}/by-experiment/{EXP_ID}")

        assert resp.status_code == 200
        assert resp.json()[0]["population"][0]["individual_id"] == "3099568720531170764"

    def test_empty_list(self, client, mock_factory):
        mock_factory.generation_repo.find_by_experiment.return_value = []
        resp = client.get(f"{BASE}/by-experiment/{EXP_ID}")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.generation_repo.find_by_experiment.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/by-experiment/{EXP_ID}")
        assert resp.status_code == 500


# ── GET /by-status/{status} ───────────────────────────────────────────────────
class TestGetGenerationsByStatus:
    def test_returns_list(self, client, mock_factory):
        mock_factory.generation_repo.find_by_status.return_value = [sample_generation()]
        mock_factory.individual_repo.find_by_generation.return_value = []
        resp = client.get(f"{BASE}/by-status/Done")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.generation_repo.find_by_status.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/by-status/Done")
        assert resp.status_code == 500


# ── GET /{generation_id} ──────────────────────────────────────────────────────
class TestGetGeneration:
    def test_found(self, client, mock_factory):
        mock_factory.generation_repo.get.return_value = sample_generation()
        mock_factory.individual_repo.find_by_generation.return_value = [sample_individual()]
        resp = client.get(f"{BASE}/{GEN_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == GEN_ID

    def test_not_found_returns_404(self, client, mock_factory):
        mock_factory.generation_repo.get.return_value = None
        resp = client.get(f"{BASE}/{GEN_ID}")
        assert resp.status_code == 404

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.generation_repo.get.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/{GEN_ID}")
        assert resp.status_code == 500


# ── PATCH /{generation_id}/status ─────────────────────────────────────────────
class TestUpdateGenerationStatus:
    def test_success(self, client, mock_factory):
        mock_factory.generation_repo.update.return_value = True
        resp = client.patch(f"{BASE}/{GEN_ID}/status", params={"new_status": "Done"})
        assert resp.status_code == 200
        assert resp.json() is True

    def test_not_modified(self, client, mock_factory):
        mock_factory.generation_repo.update.return_value = False
        resp = client.patch(f"{BASE}/{GEN_ID}/status", params={"new_status": "Done"})
        assert resp.status_code == 200
        assert resp.json() is False

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.generation_repo.update.side_effect = RuntimeError("db error")
        resp = client.patch(f"{BASE}/{GEN_ID}/status", params={"new_status": "Done"})
        assert resp.status_code == 500
