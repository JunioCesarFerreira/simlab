"""Penalties are dropped by the evidence stored with the vector, not by size.

The quality-indicator endpoint and the experiment mapper used to treat any
objective ≥ 1e8 as a penalty. A long run's energy already reaches 1.8e7, so a
legitimate front could lose points in silence; and the mapper dropped the
`evaluation` mark the engine now writes on each Pareto item.
"""
from api.endpoints import experiment
from api.mappers.experiment import experiment_from_mongo
from api.mappers.generation import _completeness_from_mongo

VALID = {"state": "complete", "objective_used": True, "valid": True}
PENALTY = {"state": "infeasible", "objective_used": True, "valid": False}


def _hv_inputs(individuals, front):
    doc = {"parameters": {}, "pareto_front": front}
    gens = [{"_id": 0, "index": 0}]
    return doc, gens, {0: individuals}


def test_a_large_measurement_is_kept_and_a_small_penalty_is_dropped():
    individuals = [
        {"individual_id": "big", "objectives": [2e8, 1.0],
         "completeness": {"state": "complete", "objective_used": True}},
        {"individual_id": "pen", "objectives": [0.5, 0.5],
         "completeness": {"state": "infeasible"}},
        {"individual_id": "mid", "objectives": [1.0, 3.0],
         "completeness": {"state": "complete", "objective_used": True}},
    ]
    front = [
        {"objectives": {"x": 2e8, "y": 1.0}, "evaluation": VALID},
        {"objectives": {"x": 0.5, "y": 0.5}, "evaluation": PENALTY},
        {"objectives": {"x": 1.0, "y": 3.0}, "evaluation": VALID},
    ]
    doc, gens, grouped = _hv_inputs(individuals, front)

    result = experiment._compute_hv_gd(
        doc, gens, grouped, ["x", "y"], ["true", "true"], "offspring", False)

    # Both measurements are on the reference front and in the population, so
    # GD is exactly zero; had the penalty at (0.5, 0.5) survived, it would
    # dominate both and GD would be positive.
    assert result["gd"] == [0.0]


def test_legacy_individuals_keep_the_magnitude_rule():
    assert experiment._is_excluded([1e9, 1.0]) is True
    assert experiment._is_excluded([1.0, 1.0]) is False
    assert experiment._is_excluded([float("nan"), 1.0]) is True


def test_the_mapper_carries_the_evaluation_and_the_validity():
    doc = {
        "_id": "507f1f77bcf86cd799450301", "name": "e",
        "pareto_front": [{"chromosome": {}, "objectives": {"x": 1.0}, "evaluation": PENALTY}],
        "pareto_front_validity": {"items": 1, "valid": 0, "infeasible": 1, "refused": 0},
    }
    dto = experiment_from_mongo(doc)
    assert dto["pareto_front"][0]["evaluation"] == PENALTY
    assert dto["pareto_front_validity"]["valid"] == 0


def test_legacy_pareto_items_have_no_evaluation_key():
    dto = experiment_from_mongo({"_id": "507f1f77bcf86cd799450302", "name": "e",
                                 "pareto_front": [{"chromosome": {}, "objectives": {}}]})
    assert "evaluation" not in dto["pareto_front"][0]


def test_the_refusal_is_published_with_the_completeness():
    out = _completeness_from_mongo({"state": "complete", "refused": "metric 'x' is nan"})
    assert out["refused"] == "metric 'x' is nan"
