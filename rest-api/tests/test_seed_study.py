"""The seed study routes: creating one, listing them, reading the analysis.

The analysis itself is tested in ``pylib/tests/test_seed_study.py``; here the
contract around it, what a body may contain, what is written, and how the
documents of a run become a study.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest
from bson import ObjectId

from api.endpoints import seed_study as module
from api.metrics_cache import MetricsCache
from pylib.seed_study import SEED_LIMIT, generate_seeds
from tests.conftest import EXP_ID, GEN_ID, SRC_ID

BASE = "/api/v1/seed-studies"
CONTROL_GEN = ObjectId("507f1f77bcf86cd7994390aa")

PROBLEM = {"name": "problem1", "region": [-100, -100, 100, 100]}
OBJECTIVES = [{"metric_name": "latency", "goal": "min", "aggregator": "mean"},
              {"metric_name": "throughput", "goal": "max", "aggregator": "mean"}]
SEEDS = generate_seeds(12, 5)


def body(**over) -> dict:
    base = {
        "name": "p1 · 30 seeds",
        "problem": PROBLEM,
        "source_repository_options": {"csma": SRC_ID},
        "duration": 180,
        "seeds": {"mode": "generated", "count": 30, "master_seed": 20261030},
        "panel": {"source": "generated", "size": 20, "random_seed": 7741},
        "objectives": OBJECTIVES,
    }
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def empty_cache(monkeypatch):
    """Every test uses one id and one clock, so a shared cache would answer for them."""
    monkeypatch.setattr(module, "_cache", MetricsCache())


@pytest.fixture
def inserted(mock_factory):
    mock_factory.experiment_repo.insert.return_value = ObjectId(EXP_ID)
    return lambda: mock_factory.experiment_repo.insert.call_args.args[0]


class TestCreating:
    def test_a_study_is_an_experiment_the_engine_will_start(self, client, inserted):
        resp = client.post(f"{BASE}/", json=body())

        assert resp.status_code == 201
        doc = inserted()
        assert doc["status"] == "Waiting", "the engine's trigger"
        assert doc["parameters"]["strategy"] == "seed_study"
        assert doc["parameters"]["objectives"] == OBJECTIVES
        assert doc["source_repository_options"] == {"csma": ObjectId(SRC_ID)}

    def test_the_answer_states_the_cost_up_front(self, client, inserted):
        resp = client.post(f"{BASE}/", json=body())

        assert resp.json() == {
            "id": EXP_ID, "seeds": 30, "chromosomes": 20, "control": 40,
            "simulations": 640,
            "seed_diagnostics": {"low16_collisions": 0, "consecutive_pairs": 0, "guard": 64},
        }

    def test_the_generated_list_is_stored_with_where_it_came_from(self, client, inserted):
        client.post(f"{BASE}/", json=body())

        doc = inserted()
        assert doc["parameters"]["simulation"]["random_seeds"] == generate_seeds(30, 20261030)
        assert doc["parameters"]["seed_study"]["seeds"]["master_seed"] == 20261030

    def test_a_master_seed_is_drawn_when_not_given(self, client, inserted):
        client.post(f"{BASE}/", json=body(seeds={"mode": "generated", "count": 5}))

        origin = inserted()["parameters"]["seed_study"]["seeds"]
        assert isinstance(origin["master_seed"], int)

    def test_an_imported_list_keeps_its_order(self, client, inserted):
        seeds = {"mode": "imported", "values": [9, 3, 7, 5], "file_name": "seeds.json"}

        client.post(f"{BASE}/", json=body(seeds=seeds))

        doc = inserted()
        assert doc["parameters"]["simulation"]["random_seeds"] == [9, 3, 7, 5]
        assert doc["parameters"]["seed_study"]["seeds"]["file_name"] == "seeds.json"

    def test_an_imported_list_with_structure_is_accepted_with_a_diagnosis(self, client, inserted):
        """It may be an earlier experiment that someone wants to reproduce."""
        resp = client.post(f"{BASE}/", json=body(
            seeds={"mode": "imported", "values": [1, 2, 3]}))

        assert resp.status_code == 201
        assert resp.json()["seed_diagnostics"]["consecutive_pairs"] == 3

    def test_two_studies_do_not_repeat_a_panel_by_default(self, client, inserted):
        client.post(f"{BASE}/", json=body(panel={"source": "generated", "size": 3}))
        first = inserted()["parameters"]["random_seed"]

        client.post(f"{BASE}/", json=body(panel={"source": "generated", "size": 3}))

        assert inserted()["parameters"]["random_seed"] != first

    def test_a_failed_seed_is_repeated_then_reported(self, client, inserted):
        client.post(f"{BASE}/", json=body())

        simulation = inserted()["parameters"]["simulation"]
        assert simulation["replica_policy"] == "repeat"
        assert simulation["incomplete_objective"] == "accept"

    def test_a_provided_panel_is_carried_to_the_engine(self, client, inserted):
        chromosomes = [{"mac_protocol": 0, "relays": [{"x": 1.0, "y": 2.0}]}]

        resp = client.post(f"{BASE}/", json=body(
            panel={"source": "provided"}, chromosomes=chromosomes))

        assert resp.json()["chromosomes"] == 1
        assert inserted()["parameters"]["seed_study"]["chromosomes"] == chromosomes

    def test_a_synthetic_study_needs_no_firmware(self, client, inserted):
        """The way the whole path is tested without Cooja."""
        resp = client.post(f"{BASE}/", json=body(
            source_repository_options={},
            synthetic={"enabled": True, "bench": "DTLZ2", "noise_std": 0.1}))

        assert resp.status_code == 201
        assert inserted()["parameters"]["simulation"]["synthetic"]["noise_std"] == 0.1


class TestRefusing:
    """Every refusal is a 422 that names the field, and nothing is written."""

    @pytest.mark.parametrize("over, mentions", [
        ({"name": "  "}, "name"),
        ({"problem": {}}, "problem"),
        ({"seedz": {}}, "unknown field"),
        ({"duration": 2}, "duration"),
        ({"source_repository_options": {}}, "source_repository_options"),
        ({"source_repository_options": {"csma": "nope"}}, "not a valid id"),
        ({"source_repository_options": {"csma": SRC_ID, "tsch": SRC_ID}}, "exactly one firmware"),
        ({"seeds": {"mode": "magic"}}, "seeds.mode"),
        ({"seeds": {"mode": "generated", "count": 2}}, "seeds.count"),
        ({"seeds": {"mode": "generated", "count": 501}}, "seeds.count"),
        ({"seeds": {"mode": "imported", "values": [1, 2, 2]}}, "duplicate"),
        ({"seeds": {"mode": "imported", "values": [1, 2, SEED_LIMIT]}}, "must be in"),
        ({"panel": {"source": "generated", "size": 0}}, "panel.size"),
        ({"panel": {"source": "generated", "size": 101}}, "panel.size"),
        ({"panel": {"source": "front"}}, "panel.source"),
        ({"panel": {"source": "provided"}}, "chromosomes"),
        ({"panel": {"source": "generated", "size": 2},
          "chromosomes": [{"mac_protocol": 0}]}, "chromosomes"),
        ({"panel": {"source": "generated", "size": 100},
          "seeds": {"mode": "generated", "count": 60}}, "over the limit"),
        ({"objectives": []}, "objectives"),
        ({"objectives": [{"metric_name": "latency", "goal": "best"}]}, "goal"),
        ({"objectives": [{"metric_name": "ghost", "goal": "min"}]}, "ghost"),
        ({"objectives": [{"metric_name": "latency", "goal": "min", "aggregator": "mode"}]},
         "Unknown aggregator"),
        ({"targets": {"confidence": 0.5}}, "targets.confidence"),
        ({"targets": {"relative_error": 0}}, "targets.relative_error"),
        ({"targets": {"reliability": 1}}, "targets.reliability"),
        ({"targets": {"precision": 1}}, "unknown target"),
        ({"control_replicates": 9}, "control_replicates"),
        ({"synthetic": {"enabled": False}}, "synthetic"),
    ], ids=lambda value: value if isinstance(value, str) else None)
    def test_what_is_refused(self, client, mock_factory, over, mentions):
        resp = client.post(f"{BASE}/", json=body(**over))

        assert resp.status_code == 422, resp.text
        assert mentions in resp.json()["detail"]
        mock_factory.experiment_repo.insert.assert_not_called()


def connection_returning(simulations, aggregated=()):
    collection = MagicMock()
    collection.find.return_value = simulations
    collection.aggregate.return_value = list(aggregated)
    db = MagicMock()
    db.__getitem__.return_value = collection

    @contextmanager
    def connect():
        yield db

    conn = MagicMock()
    conn.connect = connect
    return conn


T0 = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def run(genome, seed, *, gen=GEN_ID, status="Done", latency=200.0, throughput=3000.0,
        minutes=0, message="", resource="cooja1"):
    return {
        "_id": ObjectId(), "generation_id": ObjectId(gen) if isinstance(gen, str) else gen,
        "individual_id": genome, "random_seed": seed, "status": status,
        "network_metrics": ({"latency": latency, "rtt_latency": latency,
                             "throughput": throughput, "bytes_tx": 0.0}
                            if status == "Done" else {}),
        "dodag": {"convergence": {"total_parent_switches": 40 + seed % 7}},
        "system_message": message, "end_time": T0 + timedelta(minutes=minutes),
        "assigned_resource_name": resource,
    }


def setup_study(mock_factory, runs, *, genomes=("a", "b"), seeds=None, status="Done",
                control_generation=True):
    seeds = SEEDS if seeds is None else seeds
    mock_factory.experiment_repo.get.return_value = {
        "_id": ObjectId(EXP_ID), "name": "study", "status": status,
        "parameters": {
            "strategy": "seed_study",
            "problem": {"name": "problem1"},
            "objectives": OBJECTIVES,
            "simulation": {"duration": 180, "random_seeds": seeds},
            "seed_study": {"panel": {"source": "generated", "size": len(genomes)},
                           "control_replicates": 2,
                           "seeds": {"mode": "generated", "master_seed": 5},
                           "targets": {"confidence": 0.95, "relative_error": 0.05,
                                       "reliability": 0.9}},
        },
        "seed_study_panel": {"source": "generated", "size": len(genomes),
                             "genome_hashes": list(genomes), "chromosomes": []},
    }
    generations = [{"_id": ObjectId(GEN_ID), "index": 0, "status": "Done"}]
    if control_generation:
        generations.append({"_id": CONTROL_GEN, "index": 1, "status": "Done",
                            "seed_study": {"role": "control"}})
    mock_factory.generation_repo.find_by_experiment.return_value = generations
    mock_factory.simulation_repo.connection = connection_returning(runs)
    mock_factory.resource_repo.find_all.return_value = [
        {"name": "cooja1", "container": {"image": "simulation-cooja:v1.1"}},
        {"name": "cooja2", "container": {"image": "simulation-cooja:v2"}},
    ]


def full_study(*, genomes=("a", "b"), noise=30.0):
    rng = np.random.default_rng(8)
    runs = []
    for g, genome in enumerate(genomes):
        for seed in SEEDS:
            runs.append(run(genome, seed, latency=200 + 40 * g + rng.normal(0, noise),
                            throughput=3000 + 100 * g + rng.normal(0, 50)))
    return runs


def control_for(runs, seed=SEEDS[0], replicates=2):
    out = []
    for source in [r for r in runs if r["random_seed"] == seed]:
        for _ in range(replicates):
            out.append(run(source["individual_id"], seed, gen=CONTROL_GEN,
                           latency=source["network_metrics"]["latency"],
                           throughput=source["network_metrics"]["throughput"]))
    return out


class TestReading:
    def test_it_serves_the_study_and_its_analysis(self, client, mock_factory):
        runs = full_study()
        setup_study(mock_factory, runs + control_for(runs))

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert data["seed_list"] == SEEDS and data["panel"] == ["a", "b"]
        assert data["simulations"] == 2 * 12 + 4
        assert data["finished"] == data["simulations"]
        names = [m["name"] for m in data["analysis"]["metrics"]]
        assert names[:2] == ["latency", "throughput"]
        assert data["analysis"]["recommendation"]["optimise"]["by_objective"]["latency"]

    def test_the_control_stays_out_of_the_distribution(self, client, mock_factory):
        """Repeating a seed adds no information, and it must not count twice."""
        runs = full_study()
        setup_study(mock_factory, runs + control_for(runs))

        data = client.get(f"{BASE}/{EXP_ID}").json()

        latency = data["analysis"]["metrics"][0]
        assert latency["observed"] == 24
        assert {row["n"] for row in latency["by_chromosome"]} == {12}

    def test_identical_repeats_are_reported(self, client, mock_factory):
        runs = full_study()
        setup_study(mock_factory, runs + control_for(runs))

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert data["determinism"]["checked"] and data["determinism"]["identical"]
        assert data["determinism"]["replicates"] == 2
        assert "non_deterministic" not in {w["code"] for w in data["top_warnings"]}

    def test_a_repeat_that_differs_goes_above_everything(self, client, mock_factory):
        runs = full_study()
        repeats = control_for(runs)
        repeats[0]["network_metrics"]["latency"] += 1
        setup_study(mock_factory, runs + repeats)

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert data["determinism"]["identical"] is False
        assert data["top_warnings"][0]["code"] == "non_deterministic"

    def test_a_finished_study_with_no_control_says_so(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert "determinism_unchecked" in {w["code"] for w in data["top_warnings"]}

    def test_a_repeated_replica_counts_once_and_the_later_one_wins(self, client, mock_factory):
        runs = full_study(genomes=("a",))
        seed = SEEDS[3]
        runs.append(run("a", seed, latency=999.0, minutes=30))
        setup_study(mock_factory, runs, genomes=("a",))

        data = client.get(f"{BASE}/{EXP_ID}/metrics/latency").json()

        assert len(data["values"]) == 12
        assert next(v for v in data["values"] if v["seed"] == seed)["value"] == 999

    def test_a_seed_that_never_finished_is_named(self, client, mock_factory):
        runs = [r for r in full_study() if not (r["individual_id"] == "a"
                                                and r["random_seed"] == SEEDS[2])]
        runs.append(run("a", SEEDS[2], status="Error", message="Could not reach vm8"))
        setup_study(mock_factory, runs)

        data = client.get(f"{BASE}/{EXP_ID}").json()

        completeness = data["completeness"]
        assert completeness["observed"] == 23 and completeness["failed_total"] == 1
        assert completeness["failed_seeds"][0] == {
            "genome_hash": "a", "seed": SEEDS[2], "message": "Could not reach vm8",
            "kind": "infrastructure"}
        assert completeness["failures"]["failed"] == 1
        assert "incomplete" in {w["code"] for w in data["top_warnings"]}

    def test_a_failure_that_was_repaired_is_not_a_hole(self, client, mock_factory):
        runs = full_study()
        runs.append(run("a", SEEDS[1], status="Error", message="boom"))
        setup_study(mock_factory, runs)

        completeness = client.get(f"{BASE}/{EXP_ID}").json()["completeness"]

        assert completeness["failed_total"] == 0
        assert completeness["failures"]["failed"] == 1, "the attempt still counts"

    def test_a_running_study_can_be_read(self, client, mock_factory):
        setup_study(mock_factory, full_study()[:15], status="Running",
                    control_generation=False)

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert data["analysis"]["metrics"][0]["observed"] == 15
        assert "incomplete" not in {w["code"] for w in data["top_warnings"]}

    def test_a_study_with_no_result_yet_has_no_analysis(self, client, mock_factory):
        setup_study(mock_factory, [], status="Waiting", control_generation=False)

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert data["analysis"] is None and data["finished"] == 0

    def test_parent_switches_are_read_from_the_tree(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert "parent_switches" in {m["name"] for m in data["analysis"]["metrics"]}

    def test_a_value_json_cannot_carry_was_not_measured(self, client, mock_factory):
        runs = full_study()
        runs[0]["network_metrics"]["latency"] = float("nan")
        setup_study(mock_factory, runs)

        resp = client.get(f"{BASE}/{EXP_ID}")

        assert resp.status_code == 200
        assert resp.json()["analysis"]["metrics"][0]["observed"] == 23

    def test_changing_a_target_recomputes_without_simulating(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        tight = client.get(f"{BASE}/{EXP_ID}", params={"relative_error": 0.01}).json()
        loose = client.get(f"{BASE}/{EXP_ID}", params={"relative_error": 0.2}).json()

        def needed(data):
            return data["analysis"]["recommendation"]["report"]["by_objective"]["latency"]["value"]

        assert needed(tight) > needed(loose)

    def test_the_same_data_gives_the_same_numbers(self, client, mock_factory, monkeypatch):
        runs = full_study()
        setup_study(mock_factory, runs)
        first = client.get(f"{BASE}/{EXP_ID}").json()["analysis"]

        monkeypatch.setattr(module, "_cache", MetricsCache())
        setup_study(mock_factory, runs)

        assert client.get(f"{BASE}/{EXP_ID}").json()["analysis"] == first

    def test_an_analysis_is_not_computed_twice(self, client, mock_factory, monkeypatch):
        calls = []
        real = module.seed_study.analyse_study
        monkeypatch.setattr(module.seed_study, "analyse_study",
                            lambda *a, **k: calls.append(1) or real(*a, **k))
        setup_study(mock_factory, full_study())

        client.get(f"{BASE}/{EXP_ID}")
        client.get(f"{BASE}/{EXP_ID}")

        assert len(calls) == 1

    def test_a_new_result_is_not_answered_from_the_cache(self, client, mock_factory):
        runs = full_study()
        setup_study(mock_factory, runs[:20], status="Running", control_generation=False)
        client.get(f"{BASE}/{EXP_ID}")

        setup_study(mock_factory, runs[:21], status="Running", control_generation=False)

        assert client.get(f"{BASE}/{EXP_ID}").json()["analysis"]["metrics"][0]["observed"] == 21

    def test_a_target_out_of_range_is_refused(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        resp = client.get(f"{BASE}/{EXP_ID}", params={"confidence": 2})

        assert resp.status_code == 422

    def test_a_malformed_absolute_target_is_refused(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        resp = client.get(f"{BASE}/{EXP_ID}", params={"absolute": "latency"})

        assert resp.status_code == 422

    def test_the_image_of_the_workers_is_reported(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert data["provenance"]["images"] == ["simulation-cooja:v1.1"]

    def test_two_images_in_one_study_are_warned_about(self, client, mock_factory):
        runs = full_study()
        runs[0]["assigned_resource_name"] = "cooja2"
        setup_study(mock_factory, runs)

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert "mixed_images" in {w["code"] for w in data["top_warnings"]}

    def test_a_list_with_structure_is_warned_about(self, client, mock_factory):
        seeds = [1, 2, 3, 4]
        runs = [run("a", seed, latency=200 + seed) for seed in seeds]
        setup_study(mock_factory, runs, genomes=("a",), seeds=seeds)

        data = client.get(f"{BASE}/{EXP_ID}").json()

        assert "seed_structure" in {w["code"] for w in data["top_warnings"]}

    def test_an_experiment_that_is_not_a_study_is_a_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = {
            "_id": ObjectId(EXP_ID), "parameters": {"strategy": "nsga3"}}

        assert client.get(f"{BASE}/{EXP_ID}").status_code == 404

    def test_an_unknown_id_is_a_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = None

        assert client.get(f"{BASE}/{EXP_ID}").status_code == 404

    def test_a_malformed_id_is_a_400(self, client):
        assert client.get(f"{BASE}/not-an-id").status_code == 400


class TestMetricDetail:
    def test_it_serves_one_chromosome(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        data = client.get(f"{BASE}/{EXP_ID}/metrics/latency",
                          params={"chromosome": "b"}).json()

        assert data["genome_hash"] == "b"
        assert len(data["values"]) == 12
        assert data["values"][0]["simulation_id"]
        assert data["histogram"] and data["qq"] and data["running"]

    def test_the_first_chromosome_is_the_default(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        assert client.get(f"{BASE}/{EXP_ID}/metrics/latency").json()["genome_hash"] == "a"

    def test_it_uses_the_aggregator_of_the_objective(self, client, mock_factory):
        setup_study(mock_factory, full_study())
        doc = mock_factory.experiment_repo.get.return_value
        doc["parameters"]["objectives"][0] = {"metric_name": "latency", "goal": "min",
                                              "aggregator": "max"}

        try:
            data = client.get(f"{BASE}/{EXP_ID}/metrics/latency").json()
        finally:
            doc["parameters"]["objectives"][0] = OBJECTIVES[0]

        assert data["aggregator"] == "max"

    def test_an_unknown_metric_is_a_404(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        assert client.get(f"{BASE}/{EXP_ID}/metrics/ghost").status_code == 404

    def test_an_unknown_chromosome_is_a_404(self, client, mock_factory):
        setup_study(mock_factory, full_study())

        resp = client.get(f"{BASE}/{EXP_ID}/metrics/latency", params={"chromosome": "z"})

        assert resp.status_code == 404


class TestExport:
    def test_values_are_exported_in_long_format(self, client, mock_factory):
        runs = full_study(genomes=("a",))
        setup_study(mock_factory, runs, genomes=("a",))

        resp = client.get(f"{BASE}/{EXP_ID}/values.csv")

        lines = resp.text.strip().split("\n")
        assert resp.headers["content-type"].startswith("text/csv")
        assert lines[0] == "chromosome,seed,order,metric,value,status,simulation_id"
        # 12 seeds × (latency, rtt_latency, throughput, bytes_tx, parent_switches)
        assert len(lines) == 1 + 12 * 5
        assert lines[1].startswith(f"a,{SEEDS[0]},0,")


class TestSolutions:
    def test_the_panel_is_served_with_its_instance(self, client, mock_factory):
        setup_study(mock_factory, [])
        doc = mock_factory.experiment_repo.get.return_value
        doc["seed_study_panel"]["chromosomes"] = [
            {"mac_protocol": 0, "relays": [{"x": 1.0, "y": 2.0}]},
            {"mac_protocol": 0, "relays": [{"x": 3.0, "y": 4.0}]},
        ]

        data = client.get(f"{BASE}/{EXP_ID}/solutions").json()

        assert data["problem"] == {"name": "problem1"}
        assert [s["genome_hash"] for s in data["solutions"]] == ["a", "b"]
        assert data["solutions"][1]["chromosome"]["relays"] == [{"x": 3.0, "y": 4.0}]

    def test_before_the_panel_is_chosen_there_is_nothing_to_draw(self, client, mock_factory):
        setup_study(mock_factory, [])
        mock_factory.experiment_repo.get.return_value.pop("seed_study_panel")

        data = client.get(f"{BASE}/{EXP_ID}/solutions").json()

        assert data["solutions"] == []

    def test_an_experiment_that_is_not_a_study_is_a_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = {
            "_id": ObjectId(EXP_ID), "parameters": {"strategy": "nsga2"}}

        assert client.get(f"{BASE}/{EXP_ID}/solutions").status_code == 404


class TestListing:
    def test_progress_is_reported_per_study(self, client, mock_factory):
        oid = ObjectId(EXP_ID)
        mock_factory.experiment_repo.find_by_strategy.return_value = [{
            "_id": oid, "name": "s", "status": "Running",
            "parameters": {"simulation": {"duration": 180, "random_seeds": [1, 2, 3]},
                           "objectives": OBJECTIVES,
                           "problem": {"name": "problem1"},
                           "seed_study": {"panel": {"source": "generated", "size": 20},
                                          "control_replicates": 2}},
        }]
        mock_factory.simulation_repo.connection = connection_returning(
            [], aggregated=[{"_id": oid, "finished": 11}])

        rows = client.get(f"{BASE}/").json()

        assert rows[0]["simulations"] == 100 and rows[0]["finished"] == 11
        assert rows[0]["panel_source"] == "generated"
        assert mock_factory.experiment_repo.find_by_strategy.call_args.args[0] == "seed_study"


class TestEstimate:
    def test_it_averages_recent_runs_of_that_duration(self, client, mock_factory):
        mock_factory.simulation_repo.connection = connection_returning([
            {"start_time": T0, "end_time": T0 + timedelta(seconds=100)},
            {"start_time": T0, "end_time": T0 + timedelta(seconds=300)},
            {"start_time": None, "end_time": T0},
        ])

        data = client.get(f"{BASE}/estimate", params={"duration": 180}).json()

        assert data == {"samples": 2, "mean_seconds": 200}

    def test_no_history_means_no_estimate(self, client, mock_factory):
        mock_factory.simulation_repo.connection = connection_returning([])

        data = client.get(f"{BASE}/estimate", params={"duration": 180}).json()

        assert data == {"samples": 0, "mean_seconds": None}

    def test_a_duration_out_of_range_is_refused(self, client):
        assert client.get(f"{BASE}/estimate", params={"duration": 2}).status_code == 422
