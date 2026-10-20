from bson import ObjectId

from tests.conftest import EXP_ID, GEN_ID, sample_generation, sample_individual

BASE = "/api/v1/generations"


# ── GET /by-experiment/{experiment_id} ────────────────────────────────────────
class TestGetGenerationsByExperiment:
    def test_returns_list_with_population(self, client, mock_factory):
        gen = sample_generation()
        ind = sample_individual()
        mock_factory.generation_repo.find_by_experiment.return_value = [gen]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [ind]}

        resp = client.get(f"{BASE}/by-experiment/{EXP_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == GEN_ID
        assert data[0]["index"] == 0
        assert data[0]["status"] == "Done"
        assert len(data[0]["population"]) == 1
        assert data[0]["population"][0]["individual_id"] == "abc123hash"

    def test_individual_id_as_int_is_coerced_to_str(self, client, mock_factory):
        """individual_id stored as int (python hash) must be serialised as str."""
        gen = sample_generation()
        ind = sample_individual()
        ind["individual_id"] = 3099568720531170764   # int hash
        mock_factory.generation_repo.find_by_experiment.return_value = [gen]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {ObjectId(GEN_ID): [ind]}

        resp = client.get(f"{BASE}/by-experiment/{EXP_ID}")

        assert resp.status_code == 200
        assert resp.json()[0]["population"][0]["individual_id"] == "3099568720531170764"

    def test_empty_list(self, client, mock_factory):
        mock_factory.generation_repo.find_by_experiment.return_value = []
        resp = client.get(f"{BASE}/by-experiment/{EXP_ID}")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.generation_repo.find_by_experiment.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/by-experiment/{EXP_ID}")
        assert resp.status_code == 500


# ── GET /by-status/{status} ───────────────────────────────────────────────────
class TestGetGenerationsByStatus:
    def test_returns_list(self, client, mock_factory):
        mock_factory.generation_repo.find_by_status.return_value = [sample_generation()]
        mock_factory.individual_repo.find_by_generation.return_value = []
        resp = client.get(f"{BASE}/by-status/Done")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.generation_repo.find_by_status.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/by-status/Done")
        assert resp.status_code == 500


# ── GET /{generation_id} ──────────────────────────────────────────────────────
class TestGetGeneration:
    def test_found(self, client, mock_factory):
        mock_factory.generation_repo.get.return_value = sample_generation()
        mock_factory.individual_repo.find_by_generation.return_value = [sample_individual()]
        resp = client.get(f"{BASE}/{GEN_ID}")
        assert resp.status_code == 200
        assert resp.json()["id"] == GEN_ID

    def test_not_found_returns_404(self, client, mock_factory):
        mock_factory.generation_repo.get.return_value = None
        resp = client.get(f"{BASE}/{GEN_ID}")
        assert resp.status_code == 404

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.generation_repo.get.side_effect = RuntimeError("db error")
        resp = client.get(f"{BASE}/{GEN_ID}")
        assert resp.status_code == 500


# ── PATCH /{generation_id}/status ─────────────────────────────────────────────
class TestUpdateGenerationStatus:
    def test_success(self, client, mock_factory):
        mock_factory.generation_repo.update.return_value = True
        resp = client.patch(f"{BASE}/{GEN_ID}/status", params={"new_status": "Done"})
        assert resp.status_code == 200
        assert resp.json() is True

    def test_not_modified(self, client, mock_factory):
        mock_factory.generation_repo.update.return_value = False
        resp = client.patch(f"{BASE}/{GEN_ID}/status", params={"new_status": "Done"})
        assert resp.status_code == 200
        assert resp.json() is False

    def test_repo_error_returns_500(self, client, mock_factory):
        mock_factory.generation_repo.update.side_effect = RuntimeError("db error")
        resp = client.patch(f"{BASE}/{GEN_ID}/status", params={"new_status": "Done"})
        assert resp.status_code == 500


# ── R19: the evidence travels with the objectives ──────────────────────────

class TestCompletenessIsPublished:
    """An objective vector alone cannot be read.

    A mean over one replica and a mean over five look identical in the
    response, and the difference is the difference between a measurement and
    an anecdote. ``replica_completeness`` existed in the persistence library
    and nothing published it.
    """

    COMPLETENESS = {
        "state": "partial", "expected": 2, "observed": 1, "failed": 1,
        "pending": 0, "absent": 0, "expected_seeds": [1, 2],
        "observed_seeds": [1], "missing_seeds": [2], "failed_seeds": [2],
        "per_metric": {"latency": 1}, "attempts": 2,
        "objective_used": False, "policy": "repeat missing replicas once",
    }

    def _population(self, mock_factory, individual):
        mock_factory.generation_repo.find_by_experiment.return_value = [
            sample_generation()]
        mock_factory.individual_repo.find_grouped_by_experiment.return_value = {
            ObjectId(GEN_ID): [individual]}

    def test_the_record_reaches_the_response(self, client, mock_factory):
        ind = sample_individual()
        ind["completeness"] = dict(self.COMPLETENESS)
        self._population(mock_factory, ind)

        body = client.get(f"{BASE}/by-experiment/{EXP_ID}").json()

        published = body[0]["population"][0]["completeness"]
        assert published["state"] == "partial"
        assert published["missing_seeds"] == [2]
        assert published["per_metric"] == {"latency": 1}
        assert published["objective_used"] is False

    def test_an_individual_without_the_record_reports_null(
        self, client, mock_factory
    ):
        """Not recorded and complete are different states.

        Every individual evaluated before this existed is in the first, and
        an empty record would read as the second — "nothing expected, nothing
        observed" — which is worse than saying nothing.
        """
        self._population(mock_factory, sample_individual())

        body = client.get(f"{BASE}/by-experiment/{EXP_ID}").json()

        assert body[0]["population"][0]["completeness"] is None

    def test_unknown_fields_in_the_stored_record_are_not_published(
        self, client, mock_factory
    ):
        """The contract is a fixed list, so a field the engine adds to its own
        record cannot leak into the API unnoticed."""
        ind = sample_individual()
        ind["completeness"] = {**self.COMPLETENESS, "internal_note": "x"}
        self._population(mock_factory, ind)

        body = client.get(f"{BASE}/by-experiment/{EXP_ID}").json()

        assert "internal_note" not in body[0]["population"][0]["completeness"]

    def test_a_reused_evaluation_is_published_as_such(self, client, mock_factory):
        """A reused individual has no simulations in its own generation. The
        record describes the evaluation its numbers came from, and the
        response has to say that rather than imply a fresh measurement."""
        ind = sample_individual()
        ind["completeness"] = {
            **self.COMPLETENESS, "state": "complete", "observed": 2,
            "reused": True, "source": "genome_cache",
        }
        self._population(mock_factory, ind)

        published = client.get(
            f"{BASE}/by-experiment/{EXP_ID}").json()[0]["population"][0][
                "completeness"]

        assert published["reused"] is True
        assert published["source"] == "genome_cache"
        assert published["state"] == "complete"

    def test_an_evaluation_without_replicas_says_why(self, client, mock_factory):
        ind = sample_individual()
        ind["completeness"] = {
            "state": "infeasible", "expected": 0, "observed": 0,
            "reason": "the individual violates a constraint",
        }
        self._population(mock_factory, ind)

        published = client.get(
            f"{BASE}/by-experiment/{EXP_ID}").json()[0]["population"][0][
                "completeness"]

        assert published["state"] == "infeasible"
        assert "constraint" in published["reason"]
