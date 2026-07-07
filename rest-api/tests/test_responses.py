"""Tests for non-finite float tolerance in API responses.

Regression: an experiment whose stored objectives contain inf/nan (e.g. an
individual that never produced metrics) must not make the API return HTTP 500
during JSON serialization.
"""
import math

from bson import ObjectId

from api.responses import sanitize_non_finite, SafeJSONResponse
from tests.conftest import sample_experiment, EXP_ID


class TestSanitizeNonFinite:
    def test_replaces_inf_and_nan(self):
        out = sanitize_non_finite(
            {"a": float("inf"), "b": float("-inf"), "c": float("nan"), "d": 1.5}
        )
        assert out == {"a": 1e12, "b": -1e12, "c": None, "d": 1.5}

    def test_recurses_into_lists_and_dicts(self):
        out = sanitize_non_finite({"objectives": [float("inf"), 2.0, {"x": float("nan")}]})
        assert out == {"objectives": [1e12, 2.0, {"x": None}]}

    def test_leaves_finite_values_untouched(self):
        data = {"n": 3, "s": "ok", "f": 0.25, "b": True, "none": None}
        assert sanitize_non_finite(data) == data

    def test_render_produces_valid_json(self):
        body = SafeJSONResponse({"v": [float("inf"), float("nan")]}).body
        assert b"Infinity" not in body and b"NaN" not in body
        assert b"1000000000000" in body and b"null" in body


class TestExperimentEndpointWithNonFinite:
    def test_get_experiment_with_inf_objectives_returns_200(self, client, mock_factory):
        doc = sample_experiment()
        # An individual/pareto point that ended up with worst (non-finite) objectives
        doc["pareto_front"] = [
            {"chromosome": {"relays": []}, "objectives": {"f1": float("inf"), "f2": float("nan")}}
        ]
        mock_factory.experiment_repo.get.return_value = doc

        resp = client.get(f"/api/v1/experiments/{EXP_ID}")

        assert resp.status_code == 200  # would be 500 without SafeJSONResponse
        pf = resp.json()["pareto_front"][0]["objectives"]
        assert pf["f1"] == 1e12
        assert pf["f2"] is None
