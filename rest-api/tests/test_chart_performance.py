from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

from bson import ObjectId
import numpy as np
import pytest

from api.endpoints import experiment
from api.metrics_cache import MetricsCache
from tests.conftest import EXP_ID, GEN_ID, sample_experiment, sample_generation, sample_individual


@pytest.mark.parametrize("dimensions", [2, 3, 6])
def test_native_front_matches_pairwise_definition(dimensions):
    rng = np.random.default_rng(42)
    points = rng.integers(0, 10, size=(120, dimensions)).tolist()
    points += points[:10]  # duplicates must retain their original indices
    minimize = [i % 2 == 0 for i in range(dimensions)]
    def dominates(a, b):
        return (all(x <= y if m else x >= y for x, y, m in zip(a, b, minimize))
                and any(x != y for x, y in zip(a, b)))

    expected = {tuple(v if m else -v for v, m in zip(p, minimize)) for p in points
                if not any(dominates(q, p) for q in points)}
    actual = experiment._front_rows(points, minimize)
    assert len(actual) == len(expected)
    assert {tuple(p) for p in actual} == expected


def test_full_uses_one_individual_query_and_keeps_generation_membership(client, mock_factory):
    gen0, gen1 = sample_generation(), sample_generation()
    gen1.update(_id=ObjectId(), index=1)
    a, b = sample_individual(), sample_individual()
    b.update(individual_id="second", generation_id=gen1["_id"])
    mock_factory.experiment_repo.get.return_value = sample_experiment()
    mock_factory.generation_repo.find_by_experiment.return_value = [gen0, gen1]
    mock_factory.individual_repo.find_grouped_by_experiment.return_value = {
        gen0["_id"]: [a], gen1["_id"]: [b],
    }
    response = client.get(f"/api/v1/experiments/{EXP_ID}/full")
    assert response.status_code == 200
    assert [[i["individual_id"] for i in g["population"]]
            for g in response.json()["generations"]] == [["abc123hash"], ["second"]]
    mock_factory.individual_repo.find_grouped_by_experiment.assert_called_once_with(ObjectId(EXP_ID))
    mock_factory.individual_repo.find_by_generation.assert_not_called()


def test_cache_reuses_only_identical_metric_inputs(mock_factory, monkeypatch):
    monkeypatch.setattr(experiment, "hv_gd_cache", MetricsCache())
    original = experiment._compute_hv_gd
    compute = MagicMock(wraps=original)
    monkeypatch.setattr(experiment, "_compute_hv_gd", compute)
    doc = sample_experiment()
    doc["pareto_front"] = [{"objectives": {"f1": 1, "f2": 1}}]
    gen = {"_id": ObjectId(GEN_ID), "index": 0, "survivors": ["a"]}
    ind = {"individual_id": "a", "objectives": [1, 1]}
    mock_factory.experiment_repo.get.return_value = doc
    mock_factory.generation_repo.find_by_experiment.return_value = [gen]
    mock_factory.individual_repo.find_grouped_by_experiment.return_value = {gen["_id"]: [ind]}

    def call(population="survivors", normalize=True):
        return experiment.get_hv_gd(EXP_ID, ["f1", "f2"], ["true", "true"],
                                    population, normalize, mock_factory)

    before = call()
    assert call() == before
    assert compute.call_count == 1
    # A rename does not change the mathematics.
    doc["name"] = "renamed"
    call()
    assert compute.call_count == 1
    ind["objectives"] = [2, 2]
    assert call()["gd"] != before["gd"]
    gen["survivors"] = ["missing"]
    assert call()["gd"] == [None]
    doc["pareto_front"] = [{"objectives": {"f1": 2, "f2": 2}}]
    call()
    call(population="offspring")
    call(population="offspring", normalize=False)
    assert compute.call_count == 6
    mock_factory.individual_repo.find_by_generation.assert_not_called()
    assert all(c.kwargs == {"objectives_only": True}
               for c in mock_factory.individual_repo.find_grouped_by_experiment.call_args_list)


def test_cache_coalesces_concurrent_identical_computations():
    cache = MetricsCache()
    started, release, requested = Event(), Event(), Event()
    calls = []

    def compute():
        calls.append(True)
        started.set()
        assert release.wait(2)
        return {"hv": [1]}

    def second_request():
        requested.set()
        return cache.get_or_compute(["same"], compute)

    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(cache.get_or_compute, ["same"], compute)
        assert started.wait(2)
        second = executor.submit(second_request)
        assert requested.wait(2)
        release.set()
        assert first.result() == second.result() == {"hv": [1]}
    assert len(calls) == 1


def test_cache_is_bounded_and_does_not_cache_errors():
    cache = MetricsCache(max_entries=1)
    compute = MagicMock(return_value={"hv": [1]})
    cache.get_or_compute("a", compute)
    cache.get_or_compute("b", compute)
    cache.get_or_compute("a", compute)
    assert compute.call_count == 3
    fail = MagicMock(side_effect=[RuntimeError("failed"), {"hv": []}])
    with pytest.raises(RuntimeError):
        cache.get_or_compute("c", fail)
    assert cache.get_or_compute("c", fail) == {"hv": []}
    tiny = MetricsCache(max_bytes=1)
    tiny.get_or_compute("a", compute)
    tiny.get_or_compute("a", compute)
    assert compute.call_count == 5


def test_slow_archive_does_not_block_a_different_metrics_request():
    cache = MetricsCache()
    started, release = Event(), Event()

    def slow():
        started.set()
        assert release.wait(2)
        return {"hv": [1]}

    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(cache.get_or_compute, "archive", slow)
        assert started.wait(2)
        try:
            other = executor.submit(cache.get_or_compute, "survivors", lambda: {"hv": [2]})
            assert other.result(timeout=1) == {"hv": [2]}
        finally:
            release.set()
        assert first.result() == {"hv": [1]}


def test_unchanged_archive_does_not_recompute_hypervolume(monkeypatch):
    doc = {"parameters": {}, "pareto_front": [{"objectives": {"x": 1, "y": 1}}]}
    gens = [{"_id": i, "index": i} for i in range(3)]
    individuals = {i: [{"objectives": [i + 1, i + 1]}] for i in range(3)}
    hv = MagicMock(wraps=experiment.moocore.hypervolume)
    monkeypatch.setattr(experiment.moocore, "hypervolume", hv)
    result = experiment._compute_hv_gd(doc, gens, individuals, ["x", "y"],
                                       ["true", "true"], "archive", True)
    assert hv.call_count == 1
    assert result["hv"] == result["hv_cumulative"]
    assert len(set(result["hv"])) == 1


@pytest.mark.parametrize("population", ["offspring", "survivors", "archive"])
@pytest.mark.parametrize("normalize", [False, True])
def test_omitting_extra_cumulative_series_preserves_every_selected_indicator(population, normalize):
    doc = {"parameters": {}, "pareto_front": [{"objectives": {"x": 1, "y": 3}},
                                               {"objectives": {"x": 3, "y": 1}}]}
    gens = [{"_id": 0, "index": 0, "survivors": ["a"]},
            {"_id": 1, "index": 1, "survivors": ["a", "b"]}]
    individuals = {0: [{"individual_id": "a", "objectives": [1, 3]}],
                   1: [{"individual_id": "b", "objectives": [3, 1]}]}
    args = (doc, gens, individuals, ["x", "y"], ["true", "true"], population, normalize)
    full = experiment._compute_hv_gd(*args)
    selected = experiment._compute_hv_gd(*args, include_cumulative=False)
    assert full["hv_cumulative"]
    assert selected == {**full, "hv_cumulative": []}


def test_population_request_skips_unused_archive_hypervolume(client, mock_factory, monkeypatch):
    monkeypatch.setattr(experiment, "hv_gd_cache", MetricsCache())
    doc = sample_experiment()
    doc["pareto_front"] = [{"objectives": {"f1": 1, "f2": 1}}]
    mock_factory.experiment_repo.get.return_value = doc
    mock_factory.generation_repo.find_by_experiment.return_value = [sample_generation()]
    mock_factory.individual_repo.find_grouped_by_experiment.return_value = {
        ObjectId(GEN_ID): [{"objectives": [1, 1]}],
    }
    hv = MagicMock(wraps=experiment.moocore.hypervolume)
    monkeypatch.setattr(experiment.moocore, "hypervolume", hv)
    url = f"/api/v1/experiments/{EXP_ID}/hv-gd?objectives=f1&objectives=f2&minimize=true&minimize=true"
    selected = client.get(url + "&include_cumulative=false")
    assert selected.status_code == 200
    assert hv.call_count == 1
    full = client.get(url).json()
    assert hv.call_count == 3
    assert selected.json() == {**full, "hv_cumulative": []}
