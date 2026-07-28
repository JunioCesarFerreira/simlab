"""
End-to-end SweepRunner tests: real model + real solver, mocked MongoDB.
"""
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.runner import SweepRunner
from lib.solver import GurobiBackend, HighsBackend

if not any(cls.is_available() for cls in (GurobiBackend, HighsBackend)):
    pytest.skip("No MILP solver backend available", allow_module_level=True)

EXP_OID = ObjectId()


def sample_sweep_doc(**overrides):
    doc = {
        "_id": ObjectId(),
        "name": "Runner sweep",
        "model_key": "milp_p2_mobile",
        "problem": {
            "name": "problem2",
            "radius_of_reach": 20.0,
            "radius_of_inter": 40.0,
            "region": [-50.0, -50.0, 50.0, 50.0],
            "sink": [0.0, 0.0],
            "candidates": [[0.0, 40.0], [15.0, 0.0]],
            "mobile_nodes": [
                {
                    "path_segments": [("30", "0")],
                    "is_closed": False,
                    "is_round_trip": False,
                    "speed": 1.0,
                    "time_step": 1.0,
                }
            ],
        },
        "parameter_grid": {"C0": [10.0, 20.0]},
        "fixed_parameters": {"kdecay": 0.01, "B": 1.0, "w_install": 1000.0, "duration": 1.0},
        "solver": {"backend": "highs" if HighsBackend.is_available() else "gurobi",
                   "time_limit_s": 30, "mip_gap": 0.01, "allow_fallback": True},
        "batch_options": {
            "objectives": [{"metric_name": "rtt_latency", "goal": "min"}],
            "simulation": {"duration": 60},
        },
        "status": "Running",
        "progress": {"total_combos": 2, "done": 0, "solved": 0, "infeasible": 0, "unique_genotypes": 0},
        "checkpoint": {"next_index": 0, "genotypes": []},
        "campaign_id": None,
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def mongo():
    m = MagicMock()
    m.milp_sweep_repo.is_cancel_requested.return_value = False
    m.experiment_repo.insert.return_value = EXP_OID
    return m


def test_runner_solves_and_hands_off(mongo):
    doc = sample_sweep_doc()
    runner = SweepRunner(doc, mongo)
    runner._run_safe()

    # Two combos persisted incrementally
    assert mongo.milp_sweep_repo.append_solution.call_count == 2
    _, record, progress, checkpoint = mongo.milp_sweep_repo.append_solution.call_args[0]
    assert progress == {
        "total_combos": 2, "done": 2, "solved": 2, "infeasible": 0, "unique_genotypes": 1,
    }
    assert checkpoint == {"next_index": 2, "genotypes": ["01"]}

    # Batch experiment created with the deduplicated topology
    exp = mongo.experiment_repo.insert.call_args[0][0]
    assert exp["parameters"]["strategy"] == "batch"
    assert exp["parameters"]["chromosomes"] == [{"mac_protocol": 0, "mask": [0, 1]}]
    assert exp["milp_sweep_id"] == doc["_id"]

    mongo.milp_sweep_repo.finish.assert_called_once_with(
        str(doc["_id"]), "Done", experiment_id=EXP_OID
    )
    mongo.campaign_repo.add_experiment.assert_not_called()


def test_runner_links_campaign(mongo):
    campaign_oid = ObjectId()
    doc = sample_sweep_doc(campaign_id=campaign_oid)
    SweepRunner(doc, mongo)._run_safe()

    mongo.campaign_repo.add_experiment.assert_called_once_with(
        str(campaign_oid), str(EXP_OID)
    )


def test_runner_resumes_from_checkpoint(mongo):
    doc = sample_sweep_doc(
        checkpoint={"next_index": 1, "genotypes": ["01"]},
        progress={"total_combos": 2, "done": 1, "solved": 1, "infeasible": 0, "unique_genotypes": 1},
    )
    SweepRunner(doc, mongo)._run_safe()

    # Only the second combo runs, and it is a duplicate of the recovered genotype
    assert mongo.milp_sweep_repo.append_solution.call_count == 1
    _, record, progress, _ = mongo.milp_sweep_repo.append_solution.call_args[0]
    assert record["index"] == 1
    assert record["is_duplicate"] is True
    assert progress["done"] == 2
    assert progress["unique_genotypes"] == 1

    # Handoff still includes the mask recovered from the checkpoint genotype
    exp = mongo.experiment_repo.insert.call_args[0][0]
    assert exp["parameters"]["chromosomes"] == [{"mac_protocol": 0, "mask": [0, 1]}]


def test_runner_cancellation(mongo):
    mongo.milp_sweep_repo.is_cancel_requested.return_value = True
    doc = sample_sweep_doc()
    SweepRunner(doc, mongo)._run_safe()

    mongo.experiment_repo.insert.assert_not_called()
    args = mongo.milp_sweep_repo.finish.call_args
    assert args[0][1] == "Cancelled"


def test_runner_all_infeasible_finishes_done_without_experiment(mongo):
    doc = sample_sweep_doc()
    doc["problem"]["mobile_nodes"] = [
        {
            "path_segments": [("100", "0")],  # unreachable
            "is_closed": False,
            "is_round_trip": False,
            "speed": 1.0,
            "time_step": 1.0,
        }
    ]
    SweepRunner(doc, mongo)._run_safe()

    mongo.experiment_repo.insert.assert_not_called()
    args, kwargs = mongo.milp_sweep_repo.finish.call_args
    assert args[1] == "Done"
    assert "without feasible topologies" in kwargs["message"]


def test_runner_missing_objectives_finishes_error(mongo):
    doc = sample_sweep_doc()
    doc["batch_options"] = {}
    SweepRunner(doc, mongo)._run_safe()

    mongo.experiment_repo.insert.assert_not_called()
    args, kwargs = mongo.milp_sweep_repo.finish.call_args
    assert args[1] == "Error"
    assert "objectives" in kwargs["message"]


def test_runner_unknown_model_finishes_error(mongo):
    doc = sample_sweep_doc(model_key="milp_nope")
    SweepRunner(doc, mongo)._run_safe()

    args, kwargs = mongo.milp_sweep_repo.finish.call_args
    assert args[1] == "Error"
    assert "Unknown MILP model" in kwargs["message"]
