from datetime import datetime

from bson import ObjectId

BASE = "/api/v1/milp"

SWEEP_ID = "60f1b2a3c4d5e6f7a8b9c0d1"
EXP_ID_M = "60f1b2a3c4d5e6f7a8b9c0d2"


def sample_problem2():
    return {
        "name": "problem2",
        "radius_of_reach": 20.0,
        "radius_of_inter": 40.0,
        "region": [-50.0, -50.0, 50.0, 50.0],
        "sink": [0.0, 0.0],
        "candidates": [[0.0, 40.0], [15.0, 0.0]],
        "mobile_nodes": [],
    }


def sample_sweep_create():
    return {
        "name": "Test Sweep",
        "model_key": "milp_p2_mobile",
        "problem": sample_problem2(),
        "parameter_grid": {"C0": [10.0, 20.0], "B": [1.0]},
        "fixed_parameters": {"kdecay": 0.01},
    }


def sample_sweep_doc():
    return {
        "_id": ObjectId(SWEEP_ID),
        "name": "Test Sweep",
        "model_key": "milp_p2_mobile",
        "problem": sample_problem2(),
        "problem_id": None,
        "parameter_grid": {"C0": [10.0, 20.0], "B": [1.0]},
        "fixed_parameters": {"kdecay": 0.01},
        "solver": {"backend": "gurobi", "time_limit_s": 300.0, "mip_gap": 0.01, "allow_fallback": True},
        "batch_options": {},
        "status": "Waiting",
        "system_message": None,
        "cancel_requested": False,
        "created_time": datetime(2026, 7, 12),
        "start_time": None,
        "end_time": None,
        "progress": {"total_combos": 2, "done": 0, "solved": 0, "infeasible": 0, "unique_genotypes": 0},
        "checkpoint": {"next_index": 0, "genotypes": []},
        "solutions": [],
        "experiment_id": None,
        "campaign_id": None,
    }


# ── GET /models ───────────────────────────────────────────────────────────────
class TestModelCatalog:
    def test_list_models(self, client):
        resp = client.get(f"{BASE}/models")
        assert resp.status_code == 200
        keys = [m["key"] for m in resp.json()]
        assert "milp_p2_mobile" in keys

    def test_get_model_detail(self, client):
        resp = client.get(f"{BASE}/models/milp_p2_mobile")
        assert resp.status_code == 200
        data = resp.json()
        assert data["problem_key"] == "problem2"
        assert data["formulation"]
        param_names = [p["name"] for p in data["parameters"]]
        assert param_names == ["C0", "kdecay", "B", "w_install", "duration"]
        duration = next(p for p in data["parameters"] if p["name"] == "duration")
        assert duration["sweepable"] is False

    def test_get_unknown_model_404(self, client):
        resp = client.get(f"{BASE}/models/nope")
        assert resp.status_code == 404


# ── GET /status ───────────────────────────────────────────────────────────────
class TestEngineStatus:
    def test_status_unknown_when_never_published(self, client, mock_factory):
        mock_factory.milp_sweep_repo.get_engine_status.return_value = None
        resp = client.get(f"{BASE}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown"

    def test_status_published(self, client, mock_factory):
        mock_factory.milp_sweep_repo.get_engine_status.return_value = {
            "status": "online",
            "solver": "highs",
            "gurobi_license": "gurobi license unavailable: no license",
            "available_backends": ["highs"],
            "updated_time": datetime(2026, 7, 12),
        }
        resp = client.get(f"{BASE}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online"
        assert data["available_backends"] == ["highs"]


# ── POST /sweeps ──────────────────────────────────────────────────────────────
class TestCreateSweep:
    def test_create_happy_path(self, client, mock_factory):
        mock_factory.milp_sweep_repo.insert.return_value = ObjectId(SWEEP_ID)

        resp = client.post(f"{BASE}/sweeps", json=sample_sweep_create())

        assert resp.status_code == 200
        assert resp.json() == SWEEP_ID
        doc = mock_factory.milp_sweep_repo.insert.call_args[0][0]
        assert doc["status"] == "Waiting"
        assert doc["progress"]["total_combos"] == 2
        assert doc["checkpoint"] == {"next_index": 0, "genotypes": []}
        assert doc["solver"]["backend"] == "gurobi"

    def test_unknown_model_key(self, client):
        payload = sample_sweep_create()
        payload["model_key"] = "milp_unknown"
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400

    def test_unknown_parameter(self, client):
        payload = sample_sweep_create()
        payload["parameter_grid"] = {"not_a_param": [1.0]}
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400
        assert "not_a_param" in resp.json()["detail"]

    def test_swept_and_fixed_overlap(self, client):
        payload = sample_sweep_create()
        payload["fixed_parameters"] = {"C0": 5.0}
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400
        assert "both swept and fixed" in resp.json()["detail"]

    def test_empty_grid_values(self, client):
        payload = sample_sweep_create()
        payload["parameter_grid"] = {"C0": []}
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400

    def test_non_sweepable_parameter_with_multiple_values(self, client):
        payload = sample_sweep_create()
        payload["parameter_grid"] = {"duration": [30.0, 60.0]}
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400
        assert "not sweepable" in resp.json()["detail"]

    def test_combination_limit(self, client):
        payload = sample_sweep_create()
        payload["parameter_grid"] = {
            "C0": list(range(150)),
            "B": list(range(100)),
        }
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400
        assert "maximum" in resp.json()["detail"]

    def test_problem_kind_mismatch(self, client):
        payload = sample_sweep_create()
        payload["problem"]["name"] = "problem3"
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400
        assert "expects a 'problem2'" in resp.json()["detail"]

    def test_unknown_backend(self, client):
        payload = sample_sweep_create()
        payload["solver"] = {"backend": "cplex"}
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400

    def test_invalid_problem_id(self, client):
        payload = sample_sweep_create()
        payload["problem_id"] = "not-an-oid"
        resp = client.post(f"{BASE}/sweeps", json=payload)
        assert resp.status_code == 400


# ── GET /sweeps ───────────────────────────────────────────────────────────────
class TestListSweeps:
    def test_list(self, client, mock_factory):
        mock_factory.milp_sweep_repo.find_all.return_value = [sample_sweep_doc()]
        resp = client.get(f"{BASE}/sweeps")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == SWEEP_ID
        assert data[0]["progress"]["total_combos"] == 2
        # info view must not embed solutions
        assert "solutions" not in data[0]

    def test_empty(self, client, mock_factory):
        mock_factory.milp_sweep_repo.find_all.return_value = []
        resp = client.get(f"{BASE}/sweeps")
        assert resp.status_code == 200
        assert resp.json() == []


# ── GET /sweeps/{id} ──────────────────────────────────────────────────────────
class TestGetSweep:
    def test_found(self, client, mock_factory):
        doc = sample_sweep_doc()
        doc["solutions"] = [
            {"index": 0, "params": {"C0": 10.0}, "status": "OPTIMAL", "genotype": "01"}
        ]
        doc["experiment_id"] = ObjectId(EXP_ID_M)
        mock_factory.milp_sweep_repo.get.return_value = doc

        resp = client.get(f"{BASE}/sweeps/{SWEEP_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["solutions"][0]["genotype"] == "01"
        assert data["experiment_id"] == EXP_ID_M
        assert data["parameter_grid"]["C0"] == [10.0, 20.0]

    def test_not_found(self, client, mock_factory):
        mock_factory.milp_sweep_repo.get.return_value = None
        resp = client.get(f"{BASE}/sweeps/{SWEEP_ID}")
        assert resp.status_code == 404


# ── PATCH /sweeps/{id}/cancel ─────────────────────────────────────────────────
class TestCancelSweep:
    def test_cancel_running(self, client, mock_factory):
        mock_factory.milp_sweep_repo.get.return_value = sample_sweep_doc()
        mock_factory.milp_sweep_repo.request_cancel.return_value = True
        resp = client.patch(f"{BASE}/sweeps/{SWEEP_ID}/cancel")
        assert resp.status_code == 200
        assert resp.json() is True

    def test_cancel_terminal_state_409(self, client, mock_factory):
        doc = sample_sweep_doc()
        doc["status"] = "Done"
        mock_factory.milp_sweep_repo.get.return_value = doc
        mock_factory.milp_sweep_repo.request_cancel.return_value = False
        resp = client.patch(f"{BASE}/sweeps/{SWEEP_ID}/cancel")
        assert resp.status_code == 409

    def test_cancel_not_found(self, client, mock_factory):
        mock_factory.milp_sweep_repo.get.return_value = None
        resp = client.patch(f"{BASE}/sweeps/{SWEEP_ID}/cancel")
        assert resp.status_code == 404


# ── DELETE /sweeps/{id} ───────────────────────────────────────────────────────
class TestDeleteSweep:
    def test_delete_done(self, client, mock_factory):
        doc = sample_sweep_doc()
        doc["status"] = "Done"
        mock_factory.milp_sweep_repo.get.return_value = doc
        mock_factory.milp_sweep_repo.delete.return_value = True
        resp = client.delete(f"{BASE}/sweeps/{SWEEP_ID}")
        assert resp.status_code == 200
        assert resp.json() is True

    def test_delete_running_409(self, client, mock_factory):
        doc = sample_sweep_doc()
        doc["status"] = "Running"
        mock_factory.milp_sweep_repo.get.return_value = doc
        resp = client.delete(f"{BASE}/sweeps/{SWEEP_ID}")
        assert resp.status_code == 409

    def test_delete_not_found(self, client, mock_factory):
        mock_factory.milp_sweep_repo.get.return_value = None
        resp = client.delete(f"{BASE}/sweeps/{SWEEP_ID}")
        assert resp.status_code == 404
