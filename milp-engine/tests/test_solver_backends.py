import pytest

from lib.solver import (
    GurobiBackend,
    HighsBackend,
    MilpBuilder,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
)

AVAILABLE_BACKENDS = [
    cls() for cls in (GurobiBackend, HighsBackend) if cls.is_available()
]


def _backend_params():
    if not AVAILABLE_BACKENDS:
        pytest.skip("No MILP solver backend available (gurobipy/highspy).")
    return [pytest.param(b, id=b.name) for b in AVAILABLE_BACKENDS]


@pytest.mark.parametrize("backend", _backend_params())
def test_simple_binary_min(backend):
    # min x + 2y  s.t. x + y >= 1, x,y binary  ->  x=1, y=0, obj=1
    b = MilpBuilder()
    x = b.add_var("x", vtype="B")
    y = b.add_var("y", vtype="B")
    b.add_objective_term(x, 1.0)
    b.add_objective_term(y, 2.0)
    b.add_constr([(x, 1.0), (y, 1.0)], ">=", 1.0)

    result = backend.solve(b.build(), time_limit_s=10)

    assert result.status == STATUS_OPTIMAL
    assert result.has_solution
    assert result.obj_value == pytest.approx(1.0)
    assert result.values[x] == pytest.approx(1.0)
    assert result.values[y] == pytest.approx(0.0)


@pytest.mark.parametrize("backend", _backend_params())
def test_mixed_continuous_and_capacity(backend):
    # min 3z + x  s.t. x >= 2, x <= 5z, z binary  ->  z=1, x=2, obj=5
    b = MilpBuilder()
    z = b.add_var("z", vtype="B")
    x = b.add_var("x", lb=0.0)
    b.add_objective_term(z, 3.0)
    b.add_objective_term(x, 1.0)
    b.add_constr([(x, 1.0)], ">=", 2.0)
    b.add_constr([(x, 1.0), (z, -5.0)], "<=", 0.0)

    result = backend.solve(b.build(), time_limit_s=10)

    assert result.status == STATUS_OPTIMAL
    assert result.obj_value == pytest.approx(5.0)
    assert result.values[z] == pytest.approx(1.0)
    assert result.values[x] == pytest.approx(2.0)


@pytest.mark.parametrize("backend", _backend_params())
def test_infeasible(backend):
    # x >= 2 and x <= 1 is infeasible
    b = MilpBuilder()
    x = b.add_var("x", lb=0.0)
    b.add_objective_term(x, 1.0)
    b.add_constr([(x, 1.0)], ">=", 2.0)
    b.add_constr([(x, 1.0)], "<=", 1.0)

    result = backend.solve(b.build(), time_limit_s=10)

    assert result.status == STATUS_INFEASIBLE
    assert not result.has_solution


@pytest.mark.parametrize("backend", _backend_params())
def test_empty_coefficient_constraint_infeasible(backend):
    # An empty flow-balance row with rhs == 1 (isolated mobile node case)
    # must be reported infeasible, not crash.
    b = MilpBuilder()
    x = b.add_var("x", lb=0.0)
    b.add_objective_term(x, 1.0)
    b.add_constr([], "==", 1.0)

    result = backend.solve(b.build(), time_limit_s=10)

    assert result.status == STATUS_INFEASIBLE
