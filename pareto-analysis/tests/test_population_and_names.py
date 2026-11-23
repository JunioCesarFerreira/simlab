"""The offline tools measure the set /hv-gd measures, and read objectives by name.

They measured only each generation's offspring while the API and the GUI
default to survivors, and paired the stored vectors with --objectives by
position: another order, or a subset, gave wrong numbers without an error.
"""
import math

import numpy as np
import pytest

from lib.api import canonical_order, load_generations, objective_columns, synthetic_bench
from lib.stats import summary, t975
from plot_pareto_results import compute_convergence_metrics, population_sources

OBJ = ["f1", "f2"]
MIN = [True, True]
REF = np.array([[0.0, 0.0]])
HV_REF = [10.0, 10.0]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, generations):
        self.generations = generations

    def get(self, url, timeout=None):
        return FakeResponse(self.generations)


EXPERIMENT = {"parameters": {"objectives": [
    {"metric_name": "f1"}, {"metric_name": "f2"}, {"metric_name": "f3"}]}}


class TestObjectivesByName:
    def test_a_reordered_subset_finds_its_columns(self):
        assert objective_columns(["f3", "f1"], EXPERIMENT) == [2, 0]

    def test_an_unknown_name_stops_the_tool(self):
        with pytest.raises(SystemExit, match="f9"):
            objective_columns(["f1", "f9"], EXPERIMENT)

    def test_without_declared_objectives_the_order_is_positional(self):
        assert objective_columns(["a", "b"], {"parameters": {}}) == [0, 1]

    def test_the_values_go_under_the_right_names(self):
        session = FakeSession([{"index": 0, "population": [
            {"id": "i", "individual_id": "h", "objectives": [1.0, 2.0, 3.0]}]}])
        per_gen, _ = load_generations(session, "api", "e", ["f3", "f1"], [2, 0])
        assert per_gen[0][0]["objectives"] == {"f3": 3.0, "f1": 1.0}


def gen(*points, ids=None):
    ids = ids or [f"g{i}" for i in range(len(points))]
    return [{"id": h, "individual_id": h, "objectives": {"f1": x, "f2": y}}
            for h, (x, y) in zip(ids, points)]


class TestPopulation:
    PER_GEN = {
        0: gen((1.0, 1.0), (5.0, 5.0), ids=["a", "b"]),
        # Generation 1 evaluated worse offspring; selection kept "a" from 0.
        1: gen((6.0, 6.0), ids=["c"]),
    }

    def run(self, population, survivors):
        return compute_convergence_metrics(
            self.PER_GEN, OBJ, MIN, HV_REF, REF,
            population=population, survivors_per_gen=survivors)

    def test_survivors_are_measured_and_resolved_across_generations(self):
        survivors = {0: ["a", "b"], 1: ["a", "c"]}
        on_survivors = self.run("survivors", survivors)
        on_offspring = self.run("offspring", survivors)
        # P_1 still holds (1, 1); generation 1's offspring alone do not.
        assert on_survivors.hv[1] == pytest.approx((10 - 1) * (10 - 1))
        assert on_offspring.hv[1] == pytest.approx((10 - 6) * (10 - 6))
        assert on_survivors.gd[1] < on_offspring.gd[1]

    def test_a_generation_without_survivors_falls_back_to_its_offspring(self):
        sources = population_sources([0, 1], "survivors", {0: ["a", "b"]})
        assert sources == ["survivors", "offspring"]
        conv = self.run("survivors", {0: ["a", "b"]})
        assert conv.hv[1] == pytest.approx(16.0)

    def test_the_archive_is_the_best_so_far(self):
        conv = self.run("archive", None)
        assert conv.hv == conv.hv_cumulative

    def test_an_unknown_population_is_refused(self):
        with pytest.raises(ValueError):
            self.run("elite", None)


class TestStudentInterval:
    def test_few_runs_use_t_not_the_normal(self):
        assert t975(4) == 2.776
        s = summary([1.0, 2.0, 3.0, 4.0, 5.0])
        assert s["ci95"] == pytest.approx(2.776 * s["std"] / math.sqrt(5))

    def test_the_expansion_joins_the_table(self):
        assert t975(31) == pytest.approx(2.0395, abs=1e-3)
        assert t975(10_000) == pytest.approx(1.9602, abs=1e-3)


class TestTheSameReferenceAsTheApi:
    SYNTHETIC = {"parameters": {
        "objectives": [{"metric_name": "f1"}, {"metric_name": "f2"}, {"metric_name": "f3"}],
        "simulation": {"synthetic": {"enabled": True, "bench": "dtlz2"}}}}

    def test_a_synthetic_full_request_uses_the_analytical_front(self):
        assert synthetic_bench(self.SYNTHETIC, ["f3", "f1", "f2"], [True] * 3) == "DTLZ2"

    @pytest.mark.parametrize("objectives, minimize", [
        (["f1", "f2"], [True, True]),            # a projection is not the smaller benchmark
        (["f1", "f2", "f3"], [True, False, True]),
    ])
    def test_otherwise_the_empirical_reference(self, objectives, minimize):
        assert synthetic_bench(self.SYNTHETIC, objectives, minimize) is None
        assert synthetic_bench({"parameters": {}}, objectives, minimize) is None

    def test_a_permutation_is_measured_in_the_declared_order(self):
        assert canonical_order(["f3", "f1", "f2"], [True, False, True], self.SYNTHETIC) == (
            ["f1", "f2", "f3"], [False, True, True])
        assert canonical_order(["f3", "f1"], [True, True], self.SYNTHETIC) == (["f3", "f1"], [True, True])
