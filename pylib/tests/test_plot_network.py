"""Smoke tests for topology plotting (pylib/plot_network.py).

These lock in the P3 target rendering without disturbing P1/P2:
  * a config with a ``targets`` key renders and saves;
  * a config without the key (P1/P2 shape) still renders — the key is optional.
"""
import pytest

matplotlib = pytest.importorskip("matplotlib")

from pylib.plot_network import plot_network_save_from_sim


def _sim_model(with_targets: bool) -> dict:
    elements = {
        "fixedMotes": [
            {"name": "sink", "position": [0.0, 0.0]},
            {"name": "relay_0", "position": [20.0, 0.0]},
        ],
        "mobileMotes": [],
    }
    if with_targets:
        elements["targets"] = [[15.0, 10.0], [30.0, -5.0]]
    return {
        "name": "test",
        "duration": 60,
        "randomSeed": 42,
        "radiusOfReach": 50.0,
        "radiusOfInter": 60.0,
        "region": [-100.0, -100.0, 100.0, 100.0],
        "simulationElements": elements,
    }


def test_p3_config_with_targets_renders(tmp_path):
    out = tmp_path / "p3.png"
    plot_network_save_from_sim(str(out), _sim_model(with_targets=True))
    assert out.exists() and out.stat().st_size > 0


def test_config_without_targets_key_renders(tmp_path):
    """P1/P2 regression: SimulationElements has no 'targets' key."""
    out = tmp_path / "p1.png"
    plot_network_save_from_sim(str(out), _sim_model(with_targets=False))
    assert out.exists() and out.stat().st_size > 0
