import pytest
from bson import ObjectId

from lib.handoff import build_batch_experiment

MASKS = {"01": [0, 1], "11": [1, 1]}


def sample_sweep(**overrides):
    sweep = {
        "_id": ObjectId(),
        "name": "P2 sweep",
        "model_key": "milp_p2_mobile",
        "problem": {"name": "problem2", "candidates": [[0, 40], [15, 0]]},
        "batch_options": {
            "objectives": [{"metric_name": "rtt_latency", "goal": "min"}],
            "simulation": {"duration": 300},
            "source_repository_options": {"csma": "0" * 24},
            "data_conversion_config": {"node_col": "n", "time_col": "t", "metrics": []},
        },
        "campaign_id": None,
    }
    sweep.update(overrides)
    return sweep


def test_handoff_builds_batch_experiment():
    sweep = sample_sweep()
    exp = build_batch_experiment(sweep, MASKS)

    assert exp["parameters"]["strategy"] == "batch"
    assert exp["parameters"]["problem"] is sweep["problem"]
    assert exp["parameters"]["simulation"]["duration"] == 300
    assert exp["parameters"]["objectives"][0]["metric_name"] == "rtt_latency"
    assert exp["source_repository_options"] == {"csma": "0" * 24}
    assert exp["status"] == "Waiting"
    assert exp["milp_sweep_id"] == sweep["_id"]
    # one chromosome per genotype (default MAC = csma), deterministic order
    assert exp["parameters"]["chromosomes"] == [
        {"mac_protocol": 0, "mask": [0, 1]},
        {"mac_protocol": 0, "mask": [1, 1]},
    ]


def test_handoff_expands_mac_protocols():
    sweep = sample_sweep()
    sweep["batch_options"]["mac_protocols"] = [0, 1]
    exp = build_batch_experiment(sweep, {"01": [0, 1]})
    assert exp["parameters"]["chromosomes"] == [
        {"mac_protocol": 0, "mask": [0, 1]},
        {"mac_protocol": 1, "mask": [0, 1]},
    ]


def test_handoff_experiment_name_override_and_default():
    sweep = sample_sweep()
    assert build_batch_experiment(sweep, MASKS)["name"] == "P2 sweep — batch"
    sweep["batch_options"]["experiment_name"] = "Custom name"
    assert build_batch_experiment(sweep, MASKS)["name"] == "Custom name"


def test_handoff_rejects_empty_masks():
    with pytest.raises(ValueError, match="no feasible topologies"):
        build_batch_experiment(sample_sweep(), {})


def test_handoff_requires_objectives():
    sweep = sample_sweep()
    sweep["batch_options"]["objectives"] = []
    with pytest.raises(ValueError, match="objectives"):
        build_batch_experiment(sweep, MASKS)


def test_handoff_rejects_invalid_mac():
    sweep = sample_sweep()
    sweep["batch_options"]["mac_protocols"] = [0, 7]
    with pytest.raises(ValueError, match="mac_protocols"):
        build_batch_experiment(sweep, MASKS)


def test_handoff_default_simulation_duration():
    sweep = sample_sweep()
    del sweep["batch_options"]["simulation"]
    exp = build_batch_experiment(sweep, MASKS)
    assert exp["parameters"]["simulation"]["duration"] == 120
