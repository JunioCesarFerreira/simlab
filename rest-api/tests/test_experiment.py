import io
import math
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
        mock_factory.individual_repo.find_by_generation.side_effect = [
            [{"objectives": [5.0, 5.0]}],   # gen 0
            [{"objectives": [5.0, 8.0]}],   # gen 1 (better f2)
        ]

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
        mock_factory.individual_repo.find_by_generation.side_effect = [
            [{"objectives": [1.0, 1.0]}],   # gen 0 (good)
            [{"objectives": [2.0, 2.0]}],   # gen 1 (worse — regression)
        ]

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
        mock_factory.individual_repo.find_by_generation.side_effect = (
            lambda gid: self._GEN0 if gid == self.GEN0 else self._GEN1
        )
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
