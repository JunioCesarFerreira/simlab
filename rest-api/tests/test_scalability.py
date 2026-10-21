"""The scalability routes: creating a benchmark, listing them, analysing one.

The analysis itself is tested in ``pylib/tests/test_scalability.py``; here
the contract around it — what a body may contain, what is written, and which
experiments these routes agree to describe.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from bson import ObjectId

from tests.conftest import EXP_ID, SRC_ID

BASE = "/api/v1/scalability"

PROBLEM = {"name": "problem1", "region": [-100, -100, 100, 100]}


def body(**over) -> dict:
    base = {
        "name": "pool scalability",
        "problem": PROBLEM,
        "source_repository_options": {"csma": SRC_ID},
        "levels": [1, 2, 4, 8],
        "repetitions": 3,
        "duration": 60,
    }
    base.update(over)
    return base


class TestCreating:
    def test_a_benchmark_is_an_experiment_the_engine_will_start(
            self, client, mock_factory):
        mock_factory.experiment_repo.insert.return_value = ObjectId(EXP_ID)

        resp = client.post(f"{BASE}/", json=body())

        assert resp.status_code == 201
        doc = mock_factory.experiment_repo.insert.call_args.args[0]
        assert doc["status"] == "Waiting", "the engine's trigger"
        assert doc["parameters"]["strategy"] == "usl_benchmark"
        assert doc["parameters"]["benchmark"] == {
            "levels": [1, 2, 4, 8], "repetitions": 3}
        assert doc["parameters"]["simulation"] == {
            "duration": 60, "random_seeds": [123456]}
        assert doc["source_repository_options"] == {"csma": ObjectId(SRC_ID)}

    def test_the_answer_states_the_cost_up_front(self, client, mock_factory):
        """A benchmark spends simulations; the operator should see how many
        before the first one runs."""
        mock_factory.experiment_repo.insert.return_value = ObjectId(EXP_ID)

        resp = client.post(f"{BASE}/", json=body(levels=[1, 2, 4], repetitions=2))

        assert resp.json() == {"id": EXP_ID, "levels": [1, 2, 4],
                               "repetitions": 2, "waves": 6, "simulations": 14}

    def test_levels_are_sorted_and_deduplicated(self, client, mock_factory):
        mock_factory.experiment_repo.insert.return_value = ObjectId(EXP_ID)

        client.post(f"{BASE}/", json=body(levels=[8, 1, 4, 4]))

        doc = mock_factory.experiment_repo.insert.call_args.args[0]
        assert doc["parameters"]["benchmark"]["levels"] == [1, 4, 8]

    def test_a_missing_conversion_falls_back_to_one_that_works(
            self, client, mock_factory):
        """A run whose metrics cannot be evaluated ends in error, and a
        benchmark of failures measures nothing."""
        mock_factory.experiment_repo.insert.return_value = ObjectId(EXP_ID)

        client.post(f"{BASE}/", json=body())

        doc = mock_factory.experiment_repo.insert.call_args.args[0]
        assert doc["data_conversion_config"]["metrics"]

    def test_a_given_chromosome_is_carried_to_the_engine(
            self, client, mock_factory):
        mock_factory.experiment_repo.insert.return_value = ObjectId(EXP_ID)
        chromosome = {"mac_protocol": 0, "relays": [{"x": 1.0, "y": 2.0}]}

        client.post(f"{BASE}/", json=body(chromosome=chromosome))

        doc = mock_factory.experiment_repo.insert.call_args.args[0]
        assert doc["parameters"]["benchmark"]["chromosome"] == chromosome


class TestRefusing:
    """Every refusal is a 422 that names the field — and nothing is written."""

    def _refused(self, client, mock_factory, payload, *, mentions):
        resp = client.post(f"{BASE}/", json=payload)

        assert resp.status_code == 422, resp.text
        assert mentions in resp.json()["detail"]
        mock_factory.experiment_repo.insert.assert_not_called()

    def test_no_levels(self, client, mock_factory):
        self._refused(client, mock_factory, body(levels=[]), mentions="levels")

    def test_a_level_below_one(self, client, mock_factory):
        self._refused(client, mock_factory, body(levels=[0, 2]), mentions="levels")

    def test_a_fractional_level(self, client, mock_factory):
        self._refused(client, mock_factory, body(levels=[1, 2.5]),
                      mentions="whole number")

    def test_a_boolean_level(self, client, mock_factory):
        """``True`` is an int in Python and would be read as one container."""
        self._refused(client, mock_factory, body(levels=[True, 2]),
                      mentions="whole number")

    def test_too_many_repetitions(self, client, mock_factory):
        self._refused(client, mock_factory, body(repetitions=50),
                      mentions="repetitions")

    def test_an_absurd_duration(self, client, mock_factory):
        self._refused(client, mock_factory, body(duration=2),
                      mentions="duration")

    def test_no_firmware(self, client, mock_factory):
        self._refused(client, mock_factory, body(source_repository_options={}),
                      mentions="source_repository_options")

    def test_a_bad_firmware_id(self, client, mock_factory):
        self._refused(client, mock_factory,
                      body(source_repository_options={"csma": "nope"}),
                      mentions="not a valid id")

    def test_a_problem_without_a_name(self, client, mock_factory):
        self._refused(client, mock_factory, body(problem={}), mentions="problem")

    def test_an_unknown_field(self, client, mock_factory):
        """Silently ignored, a misspelt ``repetitons`` would run the default."""
        self._refused(client, mock_factory, body(repetitons=5),
                      mentions="unknown field")

    def test_no_name(self, client, mock_factory):
        self._refused(client, mock_factory, body(name="  "), mentions="name")


def sim(gen_id, offset, held, status="Done", host="local"):
    t0 = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    assigned = t0 + timedelta(seconds=offset)
    return {"generation_id": gen_id, "status": status, "assigned_at": assigned,
            "start_time": assigned, "end_time": assigned + timedelta(seconds=held),
            "assigned_host_name": host}


def connection_returning(simulations):
    collection = MagicMock()
    collection.find.return_value = simulations
    db = MagicMock()
    db.__getitem__.return_value = collection

    @contextmanager
    def connect():
        yield db

    conn = MagicMock()
    conn.connect = connect
    return conn


class TestAnalysing:
    def _benchmark(self, mock_factory, waves):
        """``waves`` is a list of (level, held seconds per simulation)."""
        generations, simulations = [], []
        for index, (level, held) in enumerate(waves):
            gid = ObjectId()
            generations.append({"_id": gid, "index": index, "status": "Done",
                                "benchmark": {"level": level, "repetition": 0,
                                              "wave": index}})
            simulations += [sim(gid, 0, held) for _ in range(level)]
        mock_factory.experiment_repo.get.return_value = {
            "_id": ObjectId(EXP_ID), "name": "b", "status": "Done",
            "parameters": {"strategy": "usl_benchmark",
                           "benchmark": {"levels": [w[0] for w in waves],
                                         "repetitions": 1},
                           "simulation": {"duration": 60},
                           "problem": {"name": "problem1"}},
        }
        mock_factory.generation_repo.find_by_experiment.return_value = generations
        mock_factory.simulation_repo.connection = connection_returning(simulations)

    def test_it_serves_the_waves_and_both_fits(self, client, mock_factory):
        self._benchmark(mock_factory, [(1, 100), (2, 105), (4, 120), (8, 170)])

        resp = client.get(f"{BASE}/{EXP_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["waves"] == 4 and data["waves_finished"] == 4
        analysis = data["analysis"]
        assert analysis["valid_waves"] == 4
        assert set(analysis["fits"]) == {"batch", "per_simulation"}
        assert analysis["fits"]["batch"]["status"] == "ok"
        assert analysis["fits"]["batch"]["curve"]

    def test_a_slower_wave_means_contention(self, client, mock_factory):
        """Each simulation takes longer as concurrency grows: σ or κ > 0."""
        self._benchmark(mock_factory, [(1, 100), (2, 110), (4, 135), (8, 200)])

        fitted = client.get(f"{BASE}/{EXP_ID}").json()["analysis"]["fits"]["batch"]

        assert fitted["sigma"] + fitted["kappa"] > 0

    def test_an_experiment_that_is_not_a_benchmark_is_a_404(
            self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = {
            "_id": ObjectId(EXP_ID), "parameters": {"strategy": "nsga3"}}

        resp = client.get(f"{BASE}/{EXP_ID}")

        assert resp.status_code == 404

    def test_an_unknown_id_is_a_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = None

        assert client.get(f"{BASE}/{EXP_ID}").status_code == 404

    def test_a_malformed_id_is_a_400(self, client, mock_factory):
        assert client.get(f"{BASE}/not-an-id").status_code == 400


class TestListing:
    def test_progress_is_reported_per_benchmark(self, client, mock_factory):
        mock_factory.experiment_repo.find_by_strategy.return_value = [{
            "_id": ObjectId(EXP_ID), "name": "b", "status": "Running",
            "parameters": {"benchmark": {"levels": [1, 2, 4], "repetitions": 2},
                           "simulation": {"duration": 60},
                           "problem": {"name": "problem1"}},
        }]
        mock_factory.generation_repo.find_by_experiment.return_value = [
            {"benchmark": {"wave": 0}, "status": "Done"},
            {"benchmark": {"wave": 1}, "status": "Running"},
        ]

        rows = client.get(f"{BASE}/").json()

        assert rows[0]["waves"] == 6 and rows[0]["waves_finished"] == 1
        assert rows[0]["simulations"] == 14
        assert mock_factory.experiment_repo.find_by_strategy.call_args.args[0] == (
            "usl_benchmark")
