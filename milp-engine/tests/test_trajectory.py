"""
Fidelity tests: lib/trajectory.py must reproduce the simulator's
positions.dat discretization (pylib/cooja_builder/parse_json_pos_dat.py)
exactly, otherwise MILP topologies and Cooja metrics diverge.
"""
import pytest

from lib.trajectory import position_at, sample_step_positions

pylib_dat = pytest.importorskip(
    "pylib.cooja_builder.parse_json_pos_dat",
    reason="pylib not importable (run tests from milp-engine/ with repo root on path)",
)

TWO_SEGMENT_NODE = {
    # (0,0) -> (10,0) then (10,0) -> (10,5): different arc lengths exercise
    # the proportional per-segment step split.
    "path_segments": [("10*t", "0"), ("10", "5*t")],
    "is_closed": False,
    "is_round_trip": True,
    "speed": 1.0,
    "time_step": 1.0,
}


def _positions_from_simulator(node: dict, tmp_path) -> list[tuple[float, float]]:
    sim_elements = {
        "fixedMotes": [],
        "mobileMotes": [
            {
                "functionPath": node["path_segments"],
                "speed": node["speed"],
                "timeStep": node["time_step"],
                "isRoundTrip": node["is_round_trip"],
            }
        ],
    }
    dat_file = tmp_path / "positions.dat"
    pylib_dat.generate_positions_from_json(sim_elements, str(dat_file))

    positions = []
    for line in dat_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        mote_id, _t, x, y = line.split()
        assert mote_id == "0"
        positions.append((float(x), float(y)))
    return positions


def test_step_positions_match_simulator_dat(tmp_path):
    ours = sample_step_positions(TWO_SEGMENT_NODE)
    simulator = _positions_from_simulator(TWO_SEGMENT_NODE, tmp_path)

    assert len(ours) == len(simulator)
    for (ox, oy), (sx, sy) in zip(ours, simulator):
        # the .dat file rounds to 2 decimals
        assert ox == pytest.approx(sx, abs=0.005)
        assert oy == pytest.approx(sy, abs=0.005)


def test_stationary_node_single_step():
    node = {
        "path_segments": [("30", "0")],
        "is_round_trip": False,
        "speed": 1.0,
        "time_step": 1.0,
    }
    positions = sample_step_positions(node)
    assert positions == [(30.0, 0.0)]


def test_position_at_holds_last_position():
    node = {
        "path_segments": [("10*t", "0")],
        "is_round_trip": False,
        "speed": 1.0,
        "time_step": 1.0,
    }
    positions = sample_step_positions(node)
    # Trajectory lasts 10 steps; far beyond the end the mote stays put.
    assert position_at(positions, 1.0, 1e6) == positions[-1]
    assert position_at(positions, 1.0, 0.0) == positions[0]


def test_round_trip_doubles_and_mirrors():
    node = {
        "path_segments": [("10*t", "0")],
        "is_round_trip": True,
        "speed": 1.0,
        "time_step": 1.0,
    }
    positions = sample_step_positions(node)
    assert len(positions) % 2 == 0
    half = len(positions) // 2
    assert positions[half:] == positions[:half][::-1]
