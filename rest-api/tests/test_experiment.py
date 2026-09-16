import io
import math
import pytest
from bson import ObjectId, errors as bson_errors

from tests.conftest import (
    EXP_ID, GEN_ID, IND_ID, SRC_ID, FW_FILE_ID,
    sample_experiment, sample_generation, sample_individual,
    sample_firmware_snapshot,
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
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [sample_individual()]}

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


# ── GET / (list all) ───────────────────────────────────────────────────────────
class TestGetAllExperiments:
    def test_returns_list_with_status(self, client, mock_factory):
        exp = sample_experiment()
        exp["status"] = "Running"
        mock_factory.experiment_repo.find_all_info.return_value = [exp]
        resp = client.get(f"{BASE}/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test Experiment"
        assert data[0]["status"] == "Running"

    def test_empty_list(self, client, mock_factory):
        mock_factory.experiment_repo.find_all_info.return_value = []
        resp = client.get(f"{BASE}/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.find_all_info.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/")
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

    def test_cancelled_is_accepted(self, client, mock_factory):
        mock_factory.experiment_repo.update_status.return_value = None
        resp = client.patch(f"{BASE}/{EXP_ID}/status", params={"new_status": "Cancelled"})
        assert resp.status_code == 200
        assert resp.json() is True

    def test_invalid_status_returns_422(self, client, mock_factory):
        resp = client.patch(f"{BASE}/{EXP_ID}/status", params={"new_status": "Bogus"})
        assert resp.status_code == 422
        mock_factory.experiment_repo.update_status.assert_not_called()

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
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [{"objectives": o} for o in ind_objs]}

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
        # Every series key must be present even when there is nothing to plot,
        # so the frontend never has to special-case a missing field.
        for key in ("hv", "hv_cumulative", "gd", "igd", "igd_plus"):
            assert data[key] == []
        assert data["reference"] is None and data["reference_size"] == 0

    def test_penalized_reference_row_does_not_corrupt_igd(self, client, mock_factory):
        # Regression: the stored front may carry penalized individuals (the
        # engine writes 1e9+ for infeasible solutions). IGD averages over the
        # REFERENCE, so one such row used to drag the mean to ~1e9. GD never
        # exposed this — it minimizes over the reference instead.
        doc = sample_experiment()
        doc["pareto_front"] = [
            {"objectives": {"f1": 0.0, "f2": 1.0}},
            {"objectives": {"f1": 1.0, "f2": 0.0}},
            {"objectives": {"f1": 1e9, "f2": 1e9}},     # penalized
        ]
        self._setup(mock_factory, doc, [[0.0, 1.0], [1.0, 0.0]])

        data = self._call(client).json()
        assert data["reference_size"] == 2                    # penalized row dropped
        assert data["igd"][0] == pytest.approx(0.0, abs=1e-9)

    def test_dominated_reference_row_is_filtered(self, client, mock_factory):
        # A reference front must be non-dominated: charging the population for
        # failing to cover a point that is not on the front biases IGD upward.
        doc = sample_experiment()
        doc["pareto_front"] = [
            {"objectives": {"f1": 0.0, "f2": 1.0}},
            {"objectives": {"f1": 1.0, "f2": 0.0}},
            {"objectives": {"f1": 5.0, "f2": 5.0}},     # dominated by both
        ]
        self._setup(mock_factory, doc, [[0.0, 1.0], [1.0, 0.0]])

        data = self._call(client).json()
        assert data["reference_size"] == 2
        assert data["igd"][0] == pytest.approx(0.0, abs=1e-9)

    def test_reference_row_missing_an_objective_is_dropped(self, client, mock_factory):
        # Defaulting a missing objective to 0.0 would read as optimal on a
        # minimized axis and pull the whole reference towards the origin.
        doc = sample_experiment()
        doc["pareto_front"] = [
            {"objectives": {"f1": 3.0, "f2": 3.0}},
            {"objectives": {"f1": 3.0}},                # no f2
        ]
        self._setup(mock_factory, doc, [[3.0, 3.0]])
        assert self._call(client).json()["reference_size"] == 1

    def test_normalization_is_on_by_default_and_rescales_axes(self, client, mock_factory):
        # f2 spans 1000x the range of f1: unnormalized, it alone decides the
        # distance. Normalizing by the reference front's ideal-nadir range puts
        # both axes on comparable footing, so the two numbers must differ.
        doc = sample_experiment()
        doc["pareto_front"] = [
            {"objectives": {"f1": 0.0, "f2": 1000.0}},
            {"objectives": {"f1": 1.0, "f2": 0.0}},
        ]
        self._setup(mock_factory, doc, [[0.5, 900.0]])

        default = self._call(client).json()
        assert default["normalized"] is True

        q = "&".join(["objectives=f1", "objectives=f2", "minimize=true", "minimize=true",
                      "normalize=false"])
        raw = client.get(f"{BASE}/{EXP_ID}/hv-gd?{q}").json()
        assert raw["normalized"] is False
        assert raw["gd"][0] > default["gd"][0]      # raw distance carries f2's magnitude
        # HV is measured in raw units either way and must not move.
        assert raw["hv"] == default["hv"]

    def test_gd_matches_the_documented_p1_mean(self, client, mock_factory):
        # Regression lock on the definition itself: GD is the arithmetic mean of
        # each front point's distance to its nearest reference point (p=1), not
        # the RMS variant. The three population points are mutually
        # non-dominated, so all three survive into the generation's front, and
        # their distances to the reference {(0,0)} are 1, 1 and √0.5 — unequal,
        # which is what makes mean and RMS distinguishable here.
        doc = sample_experiment()
        doc["pareto_front"] = [{"objectives": {"f1": 0.0, "f2": 0.0}}]
        self._setup(mock_factory, doc, [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])

        q = "&".join(["objectives=f1", "objectives=f2", "minimize=true", "minimize=true",
                      "normalize=false"])
        data = client.get(f"{BASE}/{EXP_ID}/hv-gd?{q}").json()
        p1_mean = (1.0 + 1.0 + math.sqrt(0.5)) / 3.0
        rms = math.sqrt((1.0 + 1.0 + 0.5) / 3.0)
        assert data["gd"][0] == pytest.approx(p1_mean)
        assert data["gd"][0] != pytest.approx(rms)

    def test_igd_plus_never_exceeds_igd(self, client, mock_factory):
        # d+ only counts the components where the solution is worse than the
        # reference point, so IGD+ is bounded above by IGD by construction.
        doc = sample_experiment()
        doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": "ZDT1"}}
        doc["pareto_front"] = [{"objectives": {"f1": 0.5, "f2": 0.5}}]
        self._setup(mock_factory, doc, [[0.2, 0.9], [0.8, 0.4]])

        data = self._call(client).json()
        assert data["igd_plus"][0] <= data["igd"][0] + 1e-12
        assert data["igd_plus"][0] > 0

    def test_max_objective_hv_reference_in_min_space(self, client, mock_factory):
        # Regression: with a MAXimized objective the HV reference must be built in
        # minimization space. Two generations, each a single individual; gen 1
        # improves only on the maximized f2 (5 → 8).
        doc = sample_experiment()
        doc["pareto_front"] = [{"objectives": {"f1": 5.0, "f2": 5.0}}]
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.generation_repo.find_by_experiment.return_value = [
            {"_id": ObjectId(GEN_ID), "index": 0},
            {"_id": ObjectId(IND_ID), "index": 1},
        ]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [{"objectives": [5.0, 5.0]}], ObjectId(IND_ID): [{"objectives": [5.0, 8.0]}]}

        q = "objectives=f1&objectives=f2&minimize=true&minimize=false"
        data = client.get(f"{BASE}/{EXP_ID}/hv-gd?{q}").json()

        # worst (min-space) = [max f1, max(-f2)] = [5, -5]
        # ref = [5 + 5*0.05 + 1, -5 + 5*0.05 + 1] = [6.25, -3.75]
        # HV gen0 = (6.25-5)*(-3.75-(-5)) = 1.25 * 1.25   = 1.5625
        # HV gen1 = (6.25-5)*(-3.75-(-8)) = 1.25 * 4.25   = 5.3125
        assert data["hv"][0] == pytest.approx(1.5625, rel=1e-6)
        assert data["hv"][1] == pytest.approx(5.3125, rel=1e-6)
        assert data["hv"][1] > data["hv"][0]  # HV grows as the front improves

    def test_cumulative_hv_is_best_so_far_and_monotonic(self, client, mock_factory):
        # Per-generation HV can regress when a generation's own front is worse
        # than an earlier one; the cumulative ("best-so-far") HV must not.
        doc = sample_experiment()  # non-synthetic → empirical HV reference
        doc["pareto_front"] = [{"objectives": {"f1": 1.0, "f2": 1.0}}]
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.generation_repo.find_by_experiment.return_value = [
            {"_id": ObjectId(GEN_ID), "index": 0},
            {"_id": ObjectId(IND_ID), "index": 1},
        ]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [{"objectives": [1.0, 1.0]}], ObjectId(IND_ID): [{"objectives": [2.0, 2.0]}]}

        data = self._call(client).json()   # objectives f1,f2 both minimized

        # worst (min-space) = [2, 2] → ref = [2 + 2*0.05 + 1] = [3.1, 3.1]
        # HV gen0 = (3.1-1)^2 = 4.41 ; HV gen1 = (3.1-2)^2 = 1.21
        assert data["hv"][0] == pytest.approx(4.41, rel=1e-6)
        assert data["hv"][1] == pytest.approx(1.21, rel=1e-6)
        assert data["hv"][1] < data["hv"][0]                    # own front regresses
        # Cumulative keeps the best-so-far front ([1,1]) → holds at 4.41
        assert data["hv_cumulative"][0] == pytest.approx(4.41, rel=1e-6)
        assert data["hv_cumulative"][1] == pytest.approx(4.41, rel=1e-6)
        assert data["hv_cumulative"][1] >= data["hv_cumulative"][0]
        assert data["hv_cumulative"][1] >= data["hv"][1]





# ── GET /{experiment_id}/hv-gd — objective resolution & degenerate inputs ─────
class TestHvGdObjectiveResolution:
    """Phase 5: finding 9 of the audit.

    Individuals store their objectives as a positional list in the order the
    experiment declared them, but the reference front was assembled by NAME.
    Reading the first n_obj entries therefore mismatched the moment a request
    reordered or subsetted the objectives.
    """

    def _setup(self, mock_factory, *, declared, individuals, stored_front=None,
               synthetic=None):
        doc = sample_experiment()
        doc["parameters"]["objectives"] = [
            {"metric_name": name, "goal": "min"} for name in declared
        ]
        if synthetic:
            doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": synthetic}}
        doc["pareto_front"] = stored_front
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.generation_repo.find_by_experiment.return_value = [
            {"_id": ObjectId(GEN_ID), "index": 0}
        ]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [
            {"objectives": o} for o in individuals
        ]}
        return doc

    def _call(self, client, objectives):
        q = "&".join(
            [f"objectives={o}" for o in objectives] + ["minimize=true"] * len(objectives)
        )
        return client.get(f"{BASE}/{EXP_ID}/hv-gd?{q}")

    def test_reordering_objectives_does_not_change_the_distance(self, mock_factory, client):
        """The audit's reproduction: the same solution used as its own reference
        scored GD 0 in declared order and 11.313708 reordered."""
        self._setup(
            mock_factory,
            declared=["f1", "f2"],
            individuals=[[1.0, 9.0]],
            stored_front=[{"objectives": {"f1": 1.0, "f2": 9.0}}],
        )
        forward = self._call(client, ["f1", "f2"]).json()

        self._setup(
            mock_factory,
            declared=["f1", "f2"],
            individuals=[[1.0, 9.0]],
            stored_front=[{"objectives": {"f1": 1.0, "f2": 9.0}}],
        )
        reversed_ = self._call(client, ["f2", "f1"]).json()

        assert forward["gd"][0] == pytest.approx(0.0, abs=1e-12)
        assert reversed_["gd"][0] == pytest.approx(0.0, abs=1e-12)

    def test_unknown_objective_is_rejected(self, mock_factory, client):
        self._setup(mock_factory, declared=["f1", "f2"], individuals=[[1.0, 9.0]],
                    stored_front=[{"objectives": {"f1": 1.0, "f2": 9.0}}])
        response = self._call(client, ["f1", "nope"])
        assert response.status_code == 422
        assert "nope" in response.json()["detail"]

    def test_a_subset_request_refuses_the_analytical_front(self, mock_factory, client):
        """DTLZ2 with M=3 read on two axes is not DTLZ2 with M=2."""
        self._setup(
            mock_factory,
            declared=["f1", "f2", "f3"],
            individuals=[[0.5, 0.5, 0.5]],
            stored_front=[{"objectives": {"f1": 0.5, "f2": 0.5, "f3": 0.5}}],
            synthetic="DTLZ2",
        )
        data = self._call(client, ["f1", "f2"]).json()
        assert data["reference"] == "final_front"
        assert data["gd_method"] == "reference_front"

    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize("point", [[0.25, 0.5], [0.25, 0.8]])
    def test_a_permuted_request_keeps_the_analytical_front(
        self, mock_factory, client, normalize, point,
    ):
        """An interior ZDT1 point exposes the asymmetry hidden by its endpoints.

        Both on-front and off-front distances must be invariant under an axis
        permutation, as must HV/IGD/IGD+. Test raw and normalized distances.
        """
        self._setup(mock_factory, declared=["f1", "f2"],
                    individuals=[point], synthetic="ZDT1")
        results = []
        for order in (["f1", "f2"], ["f2", "f1"]):
            params = [("objectives", o) for o in order]
            params += [("minimize", "true")] * 2 + [("normalize", str(normalize).lower())]
            response = client.get(f"{BASE}/{EXP_ID}/hv-gd", params=params)
            assert response.status_code == 200
            results.append(response.json())
        forward, permuted = results
        assert forward["reference"] == permuted["reference"] == "true_front"
        assert forward["gd_method"] == permuted["gd_method"] == "analytical"
        for metric in ("gd", "hv", "igd", "igd_plus"):
            assert permuted[metric] == pytest.approx(forward[metric], abs=1e-12)
        if point == [0.25, 0.5]:
            assert forward["gd"][0] == pytest.approx(0.0, abs=1e-12)
        else:
            assert forward["gd"][0] > 0.0

    def test_all_individuals_penalized_returns_the_empty_shape(self, mock_factory, client):
        """`max()` over the empty set used to surface as a 500."""
        self._setup(
            mock_factory,
            declared=["f1", "f2"],
            individuals=[[1e9, 1e9], [1e10, 1e10]],
            stored_front=[{"objectives": {"f1": 1.0, "f2": 9.0}}],
        )
        response = self._call(client, ["f1", "f2"])
        assert response.status_code == 200
        assert response.json()["generations"] == []

    def test_synthetic_run_without_a_stored_front_still_reports(self, mock_factory, client):
        """The analytical front needs no stored one; the early return denied it."""
        self._setup(
            mock_factory,
            declared=["f1", "f2"],
            individuals=[[0.5, 0.5]],
            stored_front=None,
            synthetic="ZDT1",
        )
        data = self._call(client, ["f1", "f2"]).json()
        assert data["reference"] == "true_front"
        assert data["generations"] == [0]
        assert data["gd"][0] is not None

    def test_non_synthetic_run_without_a_stored_front_is_still_empty(self, mock_factory, client):
        """There is no reference to measure against in that case."""
        self._setup(mock_factory, declared=["f1", "f2"], individuals=[[0.5, 0.5]],
                    stored_front=None)
        assert self._call(client, ["f1", "f2"]).json()["generations"] == []

    def test_experiments_without_declared_objectives_keep_positional_order(
        self, mock_factory, client
    ):
        """Documents written before objectives were declared have nothing to
        resolve against."""
        self._setup(mock_factory, declared=[], individuals=[[1.0, 9.0]],
                    stored_front=[{"objectives": {"f1": 1.0, "f2": 9.0}}])
        data = self._call(client, ["f1", "f2"]).json()
        assert data["gd"][0] == pytest.approx(0.0, abs=1e-12)

# ── GET /{experiment_id}/hv-gd — analytical GD ────────────────────────────────
class TestHvGdAnalyticalDistance:
    """Phase 4: for a known benchmark, GD is the exact distance to the true
    front instead of the mean nearest-neighbour distance to a sampled one.

    The audit measured points lying EXACTLY on the DTLZ2 front scoring GD 0.19
    at M=6 against the 500-point reference — pure discretisation, reported as
    lack of convergence.
    """

    def _setup(self, mock_factory, bench, objectives, individuals):
        doc = sample_experiment()
        doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": bench}}
        doc["pareto_front"] = [{"objectives": dict(zip(objectives, individuals[0]))}]
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.generation_repo.find_by_experiment.return_value = [
            {"_id": ObjectId(GEN_ID), "index": 0}
        ]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [
            {"objectives": o} for o in individuals
        ]}
        return doc

    def _call(self, client, objectives):
        q = "&".join(
            [f"objectives={o}" for o in objectives] + ["minimize=true"] * len(objectives)
        )
        return client.get(f"{BASE}/{EXP_ID}/hv-gd?{q}").json()

    def test_points_on_the_true_front_score_zero(self, client, mock_factory):
        """Three points exactly on the DTLZ2 unit sphere, M=3."""
        on_sphere = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3 ** -0.5, 3 ** -0.5, 3 ** -0.5],
        ]
        self._setup(mock_factory, "DTLZ2", ["f1", "f2", "f3"], on_sphere)
        data = self._call(client, ["f1", "f2", "f3"])
        assert data["gd_method"] == "analytical"
        assert data["gd"][0] == pytest.approx(0.0, abs=1e-12)

    def test_distance_is_the_radial_error_on_dtlz2(self, client, mock_factory):
        """A point at twice the radius sits exactly 1.0 from the front."""
        self._setup(mock_factory, "DTLZ2", ["f1", "f2", "f3"], [[2.0, 0.0, 0.0]])
        assert self._call(client, ["f1", "f2", "f3"])["gd"][0] == pytest.approx(1.0, abs=1e-12)

    def test_sch1_distance_is_divided_by_the_analytical_range(self, client, mock_factory):
        """SCH1's theoretical range is [0,4] on both axes, so raw / 4."""
        # (0, 4) is the front's endpoint at x=0; (0, 5) is 1.0 away from it.
        self._setup(mock_factory, "SCH1", ["f1", "f2"], [[0.0, 5.0]])
        data = self._call(client, ["f1", "f2"])
        assert data["gd_method"] == "analytical"
        assert data["gd"][0] == pytest.approx(0.25, abs=1e-9)

    def test_non_synthetic_keeps_the_reference_front(self, client, mock_factory):
        doc = sample_experiment()          # no synthetic block
        doc["pareto_front"] = [{"objectives": {"f1": 0.5, "f2": 0.3}}]
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.generation_repo.find_by_experiment.return_value = [
            {"_id": ObjectId(GEN_ID), "index": 0}
        ]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [
            {"objectives": [0.5, 0.3]}
        ]}
        data = self._call(client, ["f1", "f2"])
        assert data["gd_method"] == "reference_front"
        assert data["normalization"] == "reference front ideal-nadir range"

    def test_response_states_how_gd_was_measured(self, client, mock_factory):
        self._setup(mock_factory, "DTLZ2", ["f1", "f2", "f3"], [[1.0, 0.0, 0.0]])
        data = self._call(client, ["f1", "f2", "f3"])
        assert data["normalization"] == "analytical ideal-nadir range"
        assert "distance to the true front" in data["gd_formula"]

    def test_unknown_bench_falls_back_to_the_reference_front(self, client, mock_factory):
        self._setup(mock_factory, "NOPE", ["f1", "f2"], [[0.5, 0.3]])
        assert self._call(client, ["f1", "f2"])["gd_method"] == "reference_front"

# ── GET /{experiment_id}/hv-gd?population=… ───────────────────────────────────
class TestHvGdMeasuredPopulation:
    """Phase 1 of the NSGA metrics fix plan: which set each generation is
    measured on.

    The scenario is the one the audit reproduced. Generation 1's offspring are
    all worse than the parents environmental selection kept, so the offspring
    curve regresses while the search has not: measuring Q_t instead of P_t makes
    a healthy run look like it is losing ground.
    """

    GEN0 = ObjectId("507f1f77bcf86cd799439021")
    GEN1 = ObjectId("507f1f77bcf86cd799439022")

    # Gen 0 population; "c" is dominated by neither extreme but is mid-front.
    _GEN0 = [
        {"individual_id": "a", "objectives": [0.1, 0.9]},
        {"individual_id": "b", "objectives": [0.9, 0.1]},
        {"individual_id": "c", "objectives": [0.5, 0.5]},
    ]
    # Gen 1 offspring: a single bad child, dominated by "c".
    _GEN1 = [{"individual_id": "d", "objectives": [0.8, 0.8]}]

    def _setup(self, mock_factory, survivors=None):
        doc = sample_experiment()
        doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": "ZDT1"}}
        doc["pareto_front"] = [{"objectives": {"f1": 0.1, "f2": 0.9}}]
        mock_factory.experiment_repo.get.return_value = doc

        gen0 = {"_id": self.GEN0, "index": 0}
        gen1 = {"_id": self.GEN1, "index": 1}
        if survivors is not None:
            gen0["survivors"] = ["a", "b", "c"]
            # Every gen-1 offspring lost: P_1 is carried over from gen 0, whose
            # documents live in the PREVIOUS generation. Resolving these hashes
            # is only possible experiment-wide.
            gen1["survivors"] = survivors
        mock_factory.generation_repo.find_by_experiment.return_value = [gen0, gen1]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {self.GEN0: self._GEN0, self.GEN1: self._GEN1}
        return doc

    def _call(self, client, population=None):
        q = "objectives=f1&objectives=f2&minimize=true&minimize=true"
        if population is not None:
            q += f"&population={population}"
        return client.get(f"{BASE}/{EXP_ID}/hv-gd?{q}")

    def test_survivors_is_the_default(self, client, mock_factory):
        self._setup(mock_factory, survivors=["a", "b"])
        data = self._call(client).json()
        assert data["population"] == "survivors"
        assert data["population_source"] == "survivors"

    def test_survivor_curve_holds_where_the_offspring_curve_regresses(self, client, mock_factory):
        self._setup(mock_factory, survivors=["a", "b"])
        survivors = self._call(client, "survivors").json()
        self._setup(mock_factory, survivors=["a", "b"])
        offspring = self._call(client, "offspring").json()

        # Same run, same evaluations, same reference point — only the measured
        # set differs. This is finding 1 in one assertion.
        assert offspring["hv"][1] < offspring["hv"][0]
        assert survivors["hv"][1] > offspring["hv"][1]
        assert survivors["gd"][1] < offspring["gd"][1]

    def test_survivors_resolve_across_generations(self, client, mock_factory):
        """A survivor kept from an older generation has no document of its own
        in the generation that kept it."""
        self._setup(mock_factory, survivors=["a", "b"])
        data = self._call(client, "survivors").json()
        # ND{(0.1,0.9),(0.9,0.1)} against ref 1.1·nadir = [1.1, 1.1]:
        # 1.0·0.2 + 0.2·1.0 − overlap 0.2·0.2 = 0.36
        assert data["hv"][1] == pytest.approx(0.36, rel=1e-9)

    def test_missing_survivor_sets_fall_back_to_offspring(self, client, mock_factory):
        """Runs recorded before survivors were persisted must still plot."""
        self._setup(mock_factory, survivors=None)
        data = self._call(client, "survivors").json()
        assert data["population"] == "survivors"
        assert data["population_source"] == "offspring"
        assert data["hv"][1] > 0.0

    @pytest.mark.parametrize("missing_index", [0, 1])
    @pytest.mark.parametrize("include_cumulative", [True, False])
    def test_partial_survivor_history_reports_each_measured_population(
        self, client, mock_factory, missing_index, include_cumulative,
    ):
        """Legacy generations and a currently evaluating generation lack P_t.

        Only those generations fall back to Q_t; the other generations must
        still measure survivors, and the archive remains independent of this.
        """
        self._setup(mock_factory, survivors=["a", "b"])
        expected_survivors = self._call(client, "survivors").json()
        expected_offspring = self._call(client, "offspring").json()
        gens = mock_factory.generation_repo.find_by_experiment.return_value
        del gens[missing_index]["survivors"]
        response = client.get(f"{BASE}/{EXP_ID}/hv-gd", params={
            "objectives": ["f1", "f2"], "minimize": ["true", "true"],
            "population": "survivors", "include_cumulative": include_cumulative,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["population_source"] == "mixed"
        assert data["population_sources"] == [
            "offspring" if i == missing_index else "survivors" for i in range(2)
        ]
        for metric in ("hv", "gd", "igd", "igd_plus"):
            expected = [
                (expected_offspring if i == missing_index else expected_survivors)[metric][i]
                for i in range(2)
            ]
            assert data[metric] == pytest.approx(expected)
        assert data["hv_cumulative"] == (
            expected_survivors["hv_cumulative"] if include_cumulative else []
        )

    def test_explicit_empty_survivors_do_not_fall_back(self, client, mock_factory):
        self._setup(mock_factory, survivors=[])
        data = self._call(client, "survivors").json()
        assert data["population_source"] == "survivors"
        assert data["population_sources"] == ["survivors", "survivors"]
        assert data["hv"][1] == 0.0
        assert data["gd"][1] is None

    def test_archive_makes_every_metric_cumulative(self, client, mock_factory):
        """Not only HV: the 'Cumulative' view used to leave GD/IGD on Q_t."""
        self._setup(mock_factory, survivors=["a", "b"])
        data = self._call(client, "archive").json()
        assert data["hv"] == pytest.approx(data["hv_cumulative"])
        assert data["gd"][1] <= data["gd"][0]

    def test_cumulative_hv_ignores_the_measured_population(self, client, mock_factory):
        """The archive folds in the offspring whichever set is reported —
        survivors are a subset of earlier offspring, so restricting the fold to
        them would silently shrink the best-so-far front."""
        self._setup(mock_factory, survivors=["a", "b"])
        survivors = self._call(client, "survivors").json()
        self._setup(mock_factory, survivors=["a", "b"])
        offspring = self._call(client, "offspring").json()
        assert survivors["hv_cumulative"] == pytest.approx(offspring["hv_cumulative"])

    def test_unknown_population_is_rejected(self, client, mock_factory):
        self._setup(mock_factory, survivors=["a", "b"])
        assert self._call(client, "elite").status_code == 422

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


# ── GET /{experiment_id}/firmware ─────────────────────────────────────────────
class TestGetExperimentFirmware:
    def test_returns_snapshot_with_string_ids(self, client, mock_factory):
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = \
            sample_firmware_snapshot()

        resp = client.get(f"{BASE}/{EXP_ID}/firmware")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "captured"
        assert body["schema_version"] == 1

        repo = body["repositories"][0]
        assert repo["name"] == "rpl-udp-csma"
        assert repo["option_keys"] == ["csma"]
        assert repo["source_repository_id"] == SRC_ID
        assert [f["file_name"] for f in repo["files"]] == ["main.c", "Makefile"]
        assert repo["files"][0]["file_id"] == FW_FILE_ID
        assert repo["files"][0]["sha256"] == "a" * 64

    def test_internal_claim_marker_is_not_exposed(self, client, mock_factory):
        snapshot = sample_firmware_snapshot()
        snapshot["claimed_at"] = "2024-01-01T00:00:00"
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = snapshot

        resp = client.get(f"{BASE}/{EXP_ID}/firmware")
        assert resp.status_code == 200
        assert "claimed_at" not in resp.json()

    def test_skipped_snapshot_keeps_its_reason(self, client, mock_factory):
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = {
            "status": "skipped", "repositories": [], "reason": "no source repository",
        }
        resp = client.get(f"{BASE}/{EXP_ID}/firmware")
        assert resp.status_code == 200
        assert resp.json()["reason"] == "no source repository"

    def test_absent_snapshot_returns_404(self, client, mock_factory):
        # Runs that predate firmware tracking are never backfilled.
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = {}
        resp = client.get(f"{BASE}/{EXP_ID}/firmware")
        assert resp.status_code == 404

    def test_invalid_id_returns_400(self, client, mock_factory):
        mock_factory.experiment_repo.get_firmware_snapshot.side_effect = \
            bson_errors.InvalidId()
        resp = client.get(f"{BASE}/bad-id/firmware")
        assert resp.status_code == 400

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.get_firmware_snapshot.side_effect = RuntimeError("db")
        resp = client.get(f"{BASE}/{EXP_ID}/firmware")
        assert resp.status_code == 500


# ── GET /{experiment_id}/firmware/files/{file_id}/content ─────────────────────
class TestGetExperimentFirmwareFileContent:
    def test_returns_raw_text(self, client, mock_factory):
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = \
            sample_firmware_snapshot()
        mock_factory.fs_handler.read_file_content.return_value = b"int main(void) {}"

        resp = client.get(f"{BASE}/{EXP_ID}/firmware/files/{FW_FILE_ID}/content")
        assert resp.status_code == 200
        assert resp.text == "int main(void) {}"

    def test_decodes_invalid_utf8_without_failing(self, client, mock_factory):
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = \
            sample_firmware_snapshot()
        mock_factory.fs_handler.read_file_content.return_value = b"\xff\xfe"

        resp = client.get(f"{BASE}/{EXP_ID}/firmware/files/{FW_FILE_ID}/content")
        assert resp.status_code == 200

    def test_file_outside_the_snapshot_returns_404(self, client, mock_factory):
        # The endpoint is scoped to the experiment: an arbitrary GridFS id must
        # not be readable through it.
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = \
            sample_firmware_snapshot()

        resp = client.get(f"{BASE}/{EXP_ID}/firmware/files/{IND_ID}/content")
        assert resp.status_code == 404
        mock_factory.fs_handler.read_file_content.assert_not_called()

    def test_absent_snapshot_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = None
        resp = client.get(f"{BASE}/{EXP_ID}/firmware/files/{FW_FILE_ID}/content")
        assert resp.status_code == 404

    def test_gridfs_error_returns_500(self, client, mock_factory):
        mock_factory.experiment_repo.get_firmware_snapshot.return_value = \
            sample_firmware_snapshot()
        mock_factory.fs_handler.read_file_content.side_effect = RuntimeError("gridfs")

        resp = client.get(f"{BASE}/{EXP_ID}/firmware/files/{FW_FILE_ID}/content")
        assert resp.status_code == 500


# ── firmware_snapshot in GET /{experiment_id} ─────────────────────────────────
class TestExperimentCarriesFirmwareSnapshot:
    def test_exposed_on_the_experiment_document(self, client, mock_factory):
        doc = sample_experiment()
        doc["firmware_snapshot"] = sample_firmware_snapshot()
        mock_factory.experiment_repo.get.return_value = doc

        resp = client.get(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 200
        snapshot = resp.json()["firmware_snapshot"]
        assert snapshot["status"] == "captured"
        assert snapshot["repositories"][0]["files"][0]["file_id"] == FW_FILE_ID

    def test_null_when_never_captured(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = sample_experiment()
        resp = client.get(f"{BASE}/{EXP_ID}")
        assert resp.status_code == 200
        assert resp.json()["firmware_snapshot"] is None
