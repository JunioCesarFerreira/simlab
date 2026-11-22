"""A batch can be created and read back through the API.

The model declared neither ``chromosomes`` nor ``random_seed``, so both were
dropped on the way in and on the way out, and it required an ``algorithm``
that batch does not read.
"""
import copy
import json
import math
from pathlib import Path

import pytest
from bson import ObjectId

from tests.conftest import EXP_ID, sample_experiment

BASE = "/api/v1/experiments"
EXAMPLE = Path(__file__).resolve().parents[2] / "debug/requests/post-batch-experiment-p2.json"


@pytest.fixture
def batch():
    return json.loads(EXAMPLE.read_text())


@pytest.fixture
def created(client, mock_factory):
    """POST a payload; return the response and the document that was inserted."""
    mock_factory.experiment_repo.insert.return_value = ObjectId(EXP_ID)

    def post(payload):
        response = client.post(f"{BASE}/", json=payload)
        call = mock_factory.experiment_repo.insert.call_args
        return response, call.args[0] if call else None

    return post


def other_strategy(strategy="nsga3", **parameters):
    doc = sample_experiment()
    return {
        "id": None, "name": "x", "status": "Waiting", "system_message": None,
        "created_time": None, "start_time": None, "end_time": None,
        "parameters": {**doc["parameters"], "strategy": strategy, **parameters},
        "source_repository_options": {},
        "data_conversion_config": doc["data_conversion_config"],
        "pareto_front": None,
    }


class TestCreateBatch:
    def test_the_example_is_accepted_as_it_is(self, created, batch):
        response, stored = created(batch)

        assert response.status_code == 200, response.text
        assert stored["parameters"]["chromosomes"] == batch["parameters"]["chromosomes"]
        assert stored["parameters"]["random_seed"] == 42
        assert stored["parameters"]["algorithm"] == {}

    def test_an_algorithm_block_is_stored_as_sent(self, created, batch):
        batch["parameters"]["algorithm"] = {"note": 1}

        response, stored = created(batch)

        assert response.status_code == 200, response.text
        assert stored["parameters"]["algorithm"] == {"note": 1}

    def test_the_seed_is_optional_and_not_invented(self, created, batch):
        del batch["parameters"]["random_seed"]

        response, stored = created(batch)

        assert response.status_code == 200, response.text
        assert "random_seed" not in stored["parameters"]

    def test_genes_are_stored_to_the_last_bit(self, created, batch):
        # A front read from the API and sent back must keep its hashes.
        relays = [{"x": 0.1 + 0.2, "y": -1 / 3}, {"x": 5e-324, "y": 123.45678901234568}]
        batch["parameters"]["problem"].update(name="problem1", number_of_relays=2)
        batch["parameters"]["chromosomes"] = [{"mac_protocol": 1, "relays": relays}]

        response, stored = created(batch)

        assert response.status_code == 200, response.text
        assert stored["parameters"]["chromosomes"][0]["relays"] == relays

    @pytest.mark.parametrize("field, change", [
        ("parameters.chromosomes", lambda p: p.pop("chromosomes")),
        ("parameters.chromosomes", lambda p: p.update(chromosomes=[])),
        ("parameters.chromosomes[1].mask", lambda p: p["chromosomes"][1]["mask"].pop()),
        ("parameters.chromosomes[2].mac_protocol", lambda p: p["chromosomes"][2].update(mac_protocol=7)),
        ("parameters.chromosomes[0].mask", lambda p: p["chromosomes"][0].update(mask=[math.nan] * 30)),
        ("parameters.random_seed", lambda p: p.update(random_seed="42")),
        ("parameters.problem.name", lambda p: p["problem"].update(name="problem9")),
    ])
    def test_what_would_not_run_is_refused_naming_the_field(
            self, client, mock_factory, batch, field, change):
        change(batch["parameters"])

        # json.dumps, because httpx refuses to send the NaN that FastAPI accepts.
        response = client.post(f"{BASE}/", content=json.dumps(batch),
                               headers={"content-type": "application/json"})

        assert response.status_code == 422, response.text
        assert response.json()["detail"].startswith(field + " ")
        mock_factory.experiment_repo.insert.assert_not_called()

    def test_a_source_left_empty_does_not_count(self, created, batch):
        # The mapper does not store it, so the engine would never see it.
        batch["source_repository_options"] = {"csma": ""}

        response, stored = created(batch)

        assert response.status_code == 422, response.text
        assert response.json()["detail"].startswith("source_repository_options ")
        assert stored is None


class TestOtherStrategiesKeepTheirContract:
    @pytest.mark.parametrize("strategy", ["nsga3", "nsga2", "random_search"])
    def test_algorithm_is_still_required(self, client, mock_factory, strategy):
        payload = other_strategy(strategy)
        del payload["parameters"]["algorithm"]

        response = client.post(f"{BASE}/", json=payload)

        assert response.status_code == 422
        [error] = response.json()["detail"]
        assert error["type"] == "missing"
        assert error["loc"] == ["body", "parameters", "algorithm"]
        assert error["msg"] == "Field required"
        assert error["input"] == payload["parameters"]
        mock_factory.experiment_repo.insert.assert_not_called()

    @pytest.mark.parametrize("value", [42, None, "not a number", [1]])
    def test_the_keys_of_batch_are_not_stored_for_them(self, created, value):
        response, stored = created(other_strategy(random_seed=value, chromosomes=value))

        assert response.status_code == 200, response.text
        assert stored["parameters"] == sample_experiment()["parameters"]


class TestReadBatch:
    @staticmethod
    def stored(batch, **changes):
        doc = sample_experiment()
        doc["parameters"] = {**copy.deepcopy(batch["parameters"]), **changes}
        return doc

    @pytest.mark.parametrize("suffix", ["", "/full"])
    def test_what_was_stored_is_returned(self, client, mock_factory, batch, suffix):
        doc = self.stored(batch, algorithm={})
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.generation_repo.find_by_experiment.return_value = []

        response = client.get(f"{BASE}/{EXP_ID}{suffix}")

        assert response.status_code == 200, response.text
        assert response.json()["parameters"]["chromosomes"] == batch["parameters"]["chromosomes"]
        assert response.json()["parameters"]["random_seed"] == 42

    @pytest.mark.parametrize("suffix", ["", "/full"])
    @pytest.mark.parametrize("algorithm", ["absent", None])
    def test_one_inserted_without_algorithm_can_be_read(
            self, client, mock_factory, batch, suffix, algorithm):
        doc = self.stored(batch, **({} if algorithm == "absent" else {"algorithm": algorithm}))
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.generation_repo.find_by_experiment.return_value = []

        response = client.get(f"{BASE}/{EXP_ID}{suffix}")

        assert response.status_code == 200, response.text
        assert response.json()["parameters"]["algorithm"] == {}

    def test_it_does_not_break_the_campaign_that_holds_it(self, client, mock_factory, batch):
        mock_factory.campaign_repo.get.return_value = {
            "_id": ObjectId(EXP_ID), "name": "c", "description": "",
            "created_time": None, "experiment_ids": [ObjectId(EXP_ID)],
        }
        mock_factory.experiment_repo.get.return_value = self.stored(batch)

        response = client.get(f"/api/v1/campaigns/{EXP_ID}/full")

        assert response.status_code == 200, response.text
        assert response.json()["experiments"][0]["parameters"]["algorithm"] == {}

    def test_an_experiment_without_them_gets_no_new_keys(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = sample_experiment()

        response = client.get(f"{BASE}/{EXP_ID}")

        assert response.json()["parameters"] == sample_experiment()["parameters"]
