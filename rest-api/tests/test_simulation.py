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


# ── Where it ran ─────────────────────────────────────────────────────────────

class TestPlacementIsPublished:
    """The chain has to reach the machine.

    `assigned_host_*`, `assigned_resource_*`, the container id, the attempt
    and the pid were all persisted at reservation time and then never
    published. An operator comparing two results could not tell whether they
    had run on the same worker, and the distribution of work across hosts
    could not be audited at all.
    """

    def _placed(self, **extra):
        return {**sample_simulation(),
                "assigned_host_id": ObjectId(EXP_ID),
                "assigned_host_name": "vm8",
                "assigned_resource_id": ObjectId(GEN_ID),
                "assigned_resource_name": "vm8-cooja2",
                "assigned_container_id": "abc123",
                "attempt_id": "attempt-A",
                "remote_pid": 4242,
                **extra}

    def _fetch(self, client, mock_factory, doc):
        mock_factory.simulation_repo.find_by_status.return_value = [doc]
        return client.get(f"{BASE}/by-status/Done").json()[0]

    def test_it_says_which_host_and_which_worker(self, client, mock_factory):
        body = self._fetch(client, mock_factory, self._placed())

        placement = body["placement"]
        assert placement["host_name"] == "vm8"
        assert placement["resource_name"] == "vm8-cooja2"

    def test_it_carries_the_container_and_the_attempt(self, client, mock_factory):
        # The worker's name can be reused; the container id cannot. And two
        # attempts of one simulation ran in different conditions.
        body = self._fetch(client, mock_factory, self._placed())

        placement = body["placement"]
        assert placement["container_id"] == "abc123"
        assert placement["attempt_id"] == "attempt-A"
        assert placement["remote_pid"] == 4242

    def test_a_simulation_that_never_ran_has_no_placement(self, client, mock_factory):
        """Distinguishable from one that ran somewhere unknown."""
        body = self._fetch(client, mock_factory, sample_simulation())

        assert body["placement"] is None

    def test_a_record_from_before_this_still_reads(self, client, mock_factory):
        # Only the names were ever denormalised, so an older document may
        # carry those and nothing else.
        doc = {**sample_simulation(), "assigned_host_name": "local",
               "assigned_resource_name": "cooja3"}

        placement = self._fetch(client, mock_factory, doc)["placement"]

        assert placement["host_name"] == "local"
        assert placement["container_id"] is None

    def test_it_is_published_by_individual_too(self, client, mock_factory):
        """Which is where the GUI reads it from."""
        mock_factory.simulation_repo.find_by_individual.return_value = [
            self._placed()]

        body = client.get(
            f"{BASE}/by-individual/abc123hash",
            params={"experiment_id": EXP_ID},
        ).json()[0]

        assert body["placement"]["resource_name"] == "vm8-cooja2"


# ── GET /by-individual/{individual_id} ───────────────────────────────────────
class TestGetSimulationsByIndividual:
    """The individual id is a chromosome hash, so the experiment scopes it.

    The same chromosome recurs across experiments and all of its runs share
    that id; without the scope the panel lists other experiments' simulations.
    """

    def test_passes_experiment_as_objectid(self, client, mock_factory):
        mock_factory.simulation_repo.find_by_individual.return_value = []
        resp = client.get(
            f"{BASE}/by-individual/abc123hash", params={"experiment_id": EXP_ID}
        )
        assert resp.status_code == 200
        args = mock_factory.simulation_repo.find_by_individual.call_args.args
        assert args[0] == "abc123hash"
        assert args[1] == ObjectId(EXP_ID)

    def test_returns_list(self, client, mock_factory):
        mock_factory.simulation_repo.find_by_individual.return_value = [
            sample_simulation()
        ]
        resp = client.get(
            f"{BASE}/by-individual/abc123hash", params={"experiment_id": EXP_ID}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == SIM_ID
        assert data[0]["individual_id"] == "abc123hash"

    def test_without_experiment_queries_unscoped(self, client, mock_factory):
        """Still answered, so an older client keeps working."""
        mock_factory.simulation_repo.find_by_individual.return_value = []
        resp = client.get(f"{BASE}/by-individual/abc123hash")
        assert resp.status_code == 200
        assert mock_factory.simulation_repo.find_by_individual.call_args.args == (
            "abc123hash",
            None,
        )

    def test_malformed_experiment_is_rejected(self, client, mock_factory):
        """A bad id must not fall through to an unscoped query."""
        resp = client.get(
            f"{BASE}/by-individual/abc123hash", params={"experiment_id": "not-an-id"}
        )
        assert resp.status_code == 422
        mock_factory.simulation_repo.find_by_individual.assert_not_called()

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.simulation_repo.find_by_individual.side_effect = RuntimeError("db")
        resp = client.get(
            f"{BASE}/by-individual/abc123hash", params={"experiment_id": EXP_ID}
        )
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

    def test_dodag_is_exposed(self, client, mock_factory):
        doc = sample_simulation()
        doc["dodag"] = {
            "deployed_nodes": 2,
            "tree": {"root": "fd00::201", "edges": {"fd00::202": "fd00::201"},
                     "depth": {"fd00::202": 1}, "disconnected": []},
            "convergence": {"converged": True, "convergence_time_ms": 5000.0,
                            "n_nodes": 2, "expected_nodes": 2},
        }
        mock_factory.simulation_repo.get.return_value = doc
        resp = client.get(f"{BASE}/{SIM_ID}")
        assert resp.status_code == 200
        dodag = resp.json()["dodag"]
        assert dodag["convergence"]["convergence_time_ms"] == 5000.0
        assert dodag["tree"]["root"] == "fd00::201"

    def test_dodag_absent_maps_to_null(self, client, mock_factory):
        mock_factory.simulation_repo.get.return_value = sample_simulation()
        resp = client.get(f"{BASE}/{SIM_ID}")
        assert resp.status_code == 200
        assert resp.json()["dodag"] is None

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
