import pytest

from lib.models import get_model
from lib.solver import GurobiBackend, HighsBackend
from lib.sweep import expand_grid, run_sweep

AVAILABLE_BACKENDS = [
    cls() for cls in (GurobiBackend, HighsBackend) if cls.is_available()
]

PROBLEM = {
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
}

GRID = {"C0": [10.0, 20.0], "B": [1.0, 2.0]}
FIXED = {"kdecay": 0.01, "w_install": 1000.0, "duration": 1.0}


def _first_backend():
    if not AVAILABLE_BACKENDS:
        pytest.skip("No MILP solver backend available (gurobipy/highspy).")
    return AVAILABLE_BACKENDS[0]


# ---------------------------------------------------------------- expand_grid

def test_expand_grid_is_deterministic_and_sorted_by_key():
    combos = expand_grid(GRID, FIXED)
    assert len(combos) == 4
    # keys sorted: B varies outermost... B < C0, so B is the slower axis
    assert [c["B"] for c in combos] == [1.0, 1.0, 2.0, 2.0]
    assert [c["C0"] for c in combos] == [10.0, 20.0, 10.0, 20.0]
    assert all(c["kdecay"] == 0.01 for c in combos)
    assert combos == expand_grid(GRID, FIXED)


def test_expand_grid_rejects_overlap_and_empty_lists():
    with pytest.raises(ValueError, match="both swept and fixed"):
        expand_grid({"C0": [1.0]}, {"C0": 2.0})
    with pytest.raises(ValueError, match="Empty value list"):
        expand_grid({"C0": []})


# ------------------------------------------------------------------ run_sweep

def test_sweep_dedups_by_genotype():
    backend = _first_backend()
    model = get_model("milp_p2_mobile")
    emitted = []

    result = run_sweep(
        model, PROBLEM, GRID, backend,
        fixed_params=FIXED, time_limit_s=30,
        on_record=emitted.append,
    )

    # All 4 combos are feasible and produce the same topology (only the relay)
    assert len(result.records) == 4
    assert len(emitted) == 4
    assert result.unique_masks == {"01": [0, 1]}
    duplicates = [r for r in result.records if r.is_duplicate]
    assert len(duplicates) == 3
    assert all(r.genotype == "01" for r in result.records)
    assert result.records[0].n_installed == 1


def test_sweep_resumes_from_checkpoint():
    backend = _first_backend()
    model = get_model("milp_p2_mobile")

    result = run_sweep(
        model, PROBLEM, GRID, backend,
        fixed_params=FIXED, time_limit_s=30,
        start_index=2, seen_genotypes={"01"},
    )

    # Only combos 2 and 3 run; the genotype is already known -> all duplicates
    assert len(result.records) == 2
    assert result.records[0].index == 2
    assert all(r.is_duplicate for r in result.records)
    assert result.unique_masks == {}


def test_sweep_cooperative_cancellation():
    backend = _first_backend()
    model = get_model("milp_p2_mobile")
    solved = []

    result = run_sweep(
        model, PROBLEM, GRID, backend,
        fixed_params=FIXED, time_limit_s=30,
        on_record=solved.append,
        should_stop=lambda: len(solved) >= 1,
    )

    assert result.cancelled
    assert len(result.records) == 1


def test_sweep_records_infeasible_combos():
    backend = _first_backend()
    model = get_model("milp_p2_mobile")
    problem = dict(PROBLEM)
    problem["mobile_nodes"] = [
        {
            "path_segments": [("100", "0")],  # unreachable mobile
            "is_closed": False,
            "is_round_trip": False,
            "speed": 1.0,
            "time_step": 1.0,
        }
    ]

    result = run_sweep(
        model, problem, {"C0": [10.0]}, backend,
        fixed_params=FIXED, time_limit_s=30,
    )

    assert len(result.records) == 1
    assert result.records[0].genotype is None
    assert result.n_failed == 1
    assert result.unique_masks == {}
