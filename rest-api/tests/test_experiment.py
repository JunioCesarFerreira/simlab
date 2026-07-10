import io
import pytest
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


# ── GET /{experiment_id}/hv-gd ────────────────────────────────────────────────
class TestGetHvGd:
    def _call(self, client, objectives=("f1", "f2")):
        q = "&".join([f"objectives={o}" for o in objectives] + ["minimize=true"] * len(objectives))
        return client.get(f"{BASE}/{EXP_ID}/hv-gd?{q}")

    def _setup(self, mock_factory, doc, ind_objs):
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.generation_repo.find_by_experiment.return_value = [{"_id": ObjectId(GEN_ID), "index": 0}]
        mock_factory.individual_repo.find_by_generation.return_value = [{"objectives": o} for o in ind_objs]

    def test_synthetic_uses_true_front_and_returns_igd(self, client, mock_factory):
        doc = sample_experiment()
        doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": "ZDT1"}}
        # Population sits AT the stored final front but OFF the true ZDT1 front.
        doc["pareto_front"] = [{"objectives": {"f1": 0.5, "f2": 0.5}}]
        self._setup(mock_factory, doc, [[0.5, 0.5]])

        data = self._call(client).json()
        assert data["reference"] == "true_front"
        assert data["igd"][0] is not None
        # GD is measured against the TRUE front, so an off-front point has GD > 0
        # (against the run's own final front it would trivially be 0).
        assert data["gd"][0] > 0.1

    def test_non_synthetic_uses_final_front_and_adds_igd(self, client, mock_factory):
        doc = sample_experiment()  # no synthetic block → WSN behaviour preserved
        doc["pareto_front"] = [{"objectives": {"f1": 0.5, "f2": 0.3}}]
        self._setup(mock_factory, doc, [[0.5, 0.3]])

        data = self._call(client).json()
        assert data["reference"] == "final_front"
        assert data["gd"][0] == pytest.approx(0.0, abs=1e-9)   # equals its own front
        assert data["igd"][0] == pytest.approx(0.0, abs=1e-9)

    def test_unknown_bench_falls_back_to_final_front(self, client, mock_factory):
        doc = sample_experiment()
        doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": "NOPE"}}
        doc["pareto_front"] = [{"objectives": {"f1": 0.5, "f2": 0.3}}]
        self._setup(mock_factory, doc, [[0.5, 0.3]])
        assert self._call(client).json()["reference"] == "final_front"

    def test_empty_pareto_front_returns_empty_shape(self, client, mock_factory):
        doc = sample_experiment()
        doc["pareto_front"] = None
        mock_factory.experiment_repo.get.return_value = doc
        data = self._call(client).json()
        assert data["generations"] == [] and data["igd"] == []


# ── POST /{experiment_id}/plot-pareto ─────────────────────────────────────────
class TestPlotPareto:
    _BODY = {"objectives": ["f1", "f2", "f3"], "minimize": [True, True, True]}

    @staticmethod
    def _patch_subprocess(monkeypatch):
        """Capture the subprocess command instead of running the analysis script."""
        from types import SimpleNamespace
        import api.endpoints.experiment as mod
        calls: dict = {}

        def fake_run(cmd, **kwargs):
            calls["cmd"] = cmd
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        return calls

    def test_synthetic_passes_true_front_flags(self, client, mock_factory, monkeypatch):
        doc = sample_experiment()
        doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": "DTLZ2"}}
        mock_factory.experiment_repo.get.return_value = doc
        calls = self._patch_subprocess(monkeypatch)

        resp = client.post(f"{BASE}/{EXP_ID}/plot-pareto", json=self._BODY)
        assert resp.status_code == 200
        cmd = calls["cmd"]
        assert cmd[cmd.index("--true-front-bench") + 1] == "DTLZ2"
        assert cmd[cmd.index("--true-front-m") + 1] == "3"

    def test_non_synthetic_omits_true_front_flags(self, client, mock_factory, monkeypatch):
        doc = sample_experiment()  # no synthetic block → empirical references
        mock_factory.experiment_repo.get.return_value = doc
        calls = self._patch_subprocess(monkeypatch)

        resp = client.post(f"{BASE}/{EXP_ID}/plot-pareto", json=self._BODY)
        assert resp.status_code == 200
        assert "--true-front-bench" not in calls["cmd"]

    def test_synthetic_with_maximize_objective_omits_flags(self, client, mock_factory, monkeypatch):
        # A maximization objective has no closed-form analytical front here.
        doc = sample_experiment()
        doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": "DTLZ2"}}
        mock_factory.experiment_repo.get.return_value = doc
        calls = self._patch_subprocess(monkeypatch)

        body = {"objectives": ["f1", "f2", "f3"], "minimize": [True, True, False]}
        resp = client.post(f"{BASE}/{EXP_ID}/plot-pareto", json=body)
        assert resp.status_code == 200
        assert "--true-front-bench" not in calls["cmd"]
