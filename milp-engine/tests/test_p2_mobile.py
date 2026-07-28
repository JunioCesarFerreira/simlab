import pytest

from lib.models import get_model
from lib.models.p2_mobile import P2MobileModel
from lib.solver import GurobiBackend, HighsBackend, STATUS_INFEASIBLE

AVAILABLE_BACKENDS = [
    cls() for cls in (GurobiBackend, HighsBackend) if cls.is_available()
]


def _backend_params():
    if not AVAILABLE_BACKENDS:
        pytest.skip("No MILP solver backend available (gurobipy/highspy).")
    return [pytest.param(b, id=b.name) for b in AVAILABLE_BACKENDS]


def _stationary_mobile(x: float, y: float) -> dict:
    return {
        "path_segments": [(str(x), str(y))],
        "is_closed": False,
        "is_round_trip": False,
        "speed": 1.0,
        "time_step": 1.0,
    }


# Sink at origin, stationary mobile at (30, 0), R_comm = 20:
# the mobile cannot reach the sink directly (d = 30); the candidate at
# (15, 0) bridges both hops (d = 15 each); the candidate at (0, 40) is
# isolated and must never be installed.
BASE_PROBLEM = {
    "name": "problem2",
    "radius_of_reach": 20.0,
    "radius_of_inter": 40.0,
    "region": [-50.0, -50.0, 50.0, 50.0],
    "sink": [0.0, 0.0],
    "candidates": [[0.0, 40.0], [15.0, 0.0]],
    "mobile_nodes": [_stationary_mobile(30.0, 0.0)],
}

PARAMS = {"C0": 10.0, "kdecay": 0.01, "B": 1.0, "w_install": 1000.0, "duration": 1.0}


@pytest.mark.parametrize("backend", _backend_params())
def test_p2_installs_only_the_relay(backend):
    model = get_model("milp_p2_mobile")
    built = model.build(BASE_PROBLEM, PARAMS)

    result = backend.solve(built.ir, time_limit_s=30)

    assert result.has_solution
    mask = built.decode_mask(result.values)
    assert mask == [0, 1]
    # obj = w_install + d²·flow on mobile->relay and relay->sink (15² each, B=1)
    assert result.obj_value == pytest.approx(1000.0 + 225.0 + 225.0)


@pytest.mark.parametrize("backend", _backend_params())
def test_p2_mask_follows_candidates_list_order(backend):
    """Genotype convention: mask[i] == candidates[i], regardless of position
    in the plane. Swapping the candidate list must swap the mask bits."""
    swapped = dict(BASE_PROBLEM)
    swapped["candidates"] = [[15.0, 0.0], [0.0, 40.0]]
    model = get_model("milp_p2_mobile")
    built = model.build(swapped, PARAMS)

    result = backend.solve(built.ir, time_limit_s=30)

    assert result.has_solution
    assert built.decode_mask(result.values) == [1, 0]


@pytest.mark.parametrize("backend", _backend_params())
def test_p2_unreachable_mobile_is_infeasible(backend):
    problem = dict(BASE_PROBLEM)
    problem["mobile_nodes"] = [_stationary_mobile(100.0, 0.0)]
    model = get_model("milp_p2_mobile")
    built = model.build(problem, PARAMS)

    result = backend.solve(built.ir, time_limit_s=30)

    assert result.status == STATUS_INFEASIBLE


@pytest.mark.parametrize("backend", _backend_params())
def test_p2_multi_period_moving_mote(backend):
    """A mote moving from (25,0) to (35,0) at speed 5 with time_step 1 yields
    two trajectory steps; duration 2 gives T=2 and the relay at (16,0) must
    stay installed to cover both instants."""
    problem = dict(BASE_PROBLEM)
    problem["candidates"] = [[16.0, 0.0]]
    problem["mobile_nodes"] = [
        {
            "path_segments": [("25 + 10*t", "0")],
            "is_closed": False,
            "is_round_trip": False,
            "speed": 5.0,
            "time_step": 1.0,
        }
    ]
    model = get_model("milp_p2_mobile")
    built = model.build(problem, {**PARAMS, "duration": 2.0})

    assert built.meta["T"] == 2

    result = backend.solve(built.ir, time_limit_s=30)

    assert result.has_solution
    assert built.decode_mask(result.values) == [1]


def test_p2_rejects_unknown_parameter():
    model = get_model("milp_p2_mobile")
    with pytest.raises(ValueError, match="Unknown parameters"):
        model.build(BASE_PROBLEM, {"C0": 10.0, "not_a_param": 1.0})


def test_p2_defaults_applied():
    model = P2MobileModel()
    resolved = model.resolve_params({"C0": 42.0})
    assert resolved["C0"] == 42.0
    assert resolved["kdecay"] == 0.25
    assert resolved["w_install"] == 1e6


def test_p2_requires_candidates():
    model = get_model("milp_p2_mobile")
    problem = dict(BASE_PROBLEM)
    problem["candidates"] = []
    with pytest.raises(ValueError, match="no candidates"):
        model.build(problem, PARAMS)
