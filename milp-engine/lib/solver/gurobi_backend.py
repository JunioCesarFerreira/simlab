"""
Gurobi backend.

License resolution follows gurobipy's standard chain, all container-friendly:
  1. GRB_LICENSE_FILE pointing to a mounted gurobi.lic (WLS credentials,
     COMPUTESERVER=... or TOKENSERVER=... entries all work through this file);
  2. GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID environment variables;
  3. the pip-installed size-limited trial (~2000 vars/constraints) as fallback.

check_license() reports which of these is active so the engine can publish
the licensing state without attempting a real solve.
"""
import logging
import math
from typing import Optional

from .base import (
    MilpIR,
    SolveResult,
    SolverBackend,
    STATUS_ERROR,
    STATUS_INFEASIBLE,
    STATUS_NO_SOLUTION,
    STATUS_OPTIMAL,
    STATUS_TIME_LIMIT,
)

log = logging.getLogger(__name__)

_license_cache: Optional[tuple[bool, str]] = None


def check_license() -> tuple[bool, str]:
    """Try to build a 1-variable model; returns (usable, description)."""
    global _license_cache
    if _license_cache is not None:
        return _license_cache
    try:
        import gurobipy as gp
    except ImportError:
        _license_cache = (False, "gurobipy is not installed")
        return _license_cache
    try:
        env = gp.Env(params={"OutputFlag": 0})
        model = gp.Model("license_probe", env=env)
        model.addVar()
        model.update()
        model.dispose()
        env.dispose()
        _license_cache = (True, "gurobi license OK")
    except gp.GurobiError as exc:
        _license_cache = (False, f"gurobi license unavailable: {exc}")
    return _license_cache


class GurobiBackend(SolverBackend):
    name = "gurobi"

    @classmethod
    def is_available(cls) -> bool:
        return check_license()[0]

    def solve(
        self,
        ir: MilpIR,
        time_limit_s: Optional[float] = None,
        mip_gap: Optional[float] = None,
    ) -> SolveResult:
        import gurobipy as gp
        from gurobipy import GRB

        try:
            env = gp.Env(params={"OutputFlag": 0})
            mdl = gp.Model("milp", env=env)
            if time_limit_s is not None:
                mdl.Params.TimeLimit = float(time_limit_s)
            if mip_gap is not None:
                mdl.Params.MIPGap = float(mip_gap)

            xs = []
            for i in range(ir.n_vars):
                ub = ir.var_ub[i]
                xs.append(
                    mdl.addVar(
                        lb=ir.var_lb[i],
                        ub=GRB.INFINITY if math.isinf(ub) else ub,
                        vtype=GRB.BINARY if ir.var_types[i] == "B" else GRB.CONTINUOUS,
                        name=ir.var_names[i],
                    )
                )
            mdl.update()

            for c in ir.constraints:
                expr = gp.quicksum(coef * xs[idx] for idx, coef in c.coeffs)
                if c.sense == "<=":
                    mdl.addConstr(expr <= c.rhs)
                elif c.sense == ">=":
                    mdl.addConstr(expr >= c.rhs)
                else:
                    mdl.addConstr(expr == c.rhs)

            mdl.setObjective(
                gp.quicksum(coef * xs[idx] for idx, coef in ir.objective),
                GRB.MINIMIZE,
            )
            mdl.optimize()

            status = mdl.Status
            runtime = float(mdl.Runtime)

            if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                result = SolveResult(status=STATUS_INFEASIBLE, runtime_s=runtime)
            elif mdl.SolCount == 0:
                result = SolveResult(
                    status=STATUS_NO_SOLUTION,
                    runtime_s=runtime,
                    message=f"gurobi status {status} with no incumbent",
                )
            else:
                unified = (
                    STATUS_TIME_LIMIT if status == GRB.TIME_LIMIT else STATUS_OPTIMAL
                )
                try:
                    gap = float(mdl.MIPGap)
                    if math.isinf(gap):
                        gap = None
                except (AttributeError, gp.GurobiError):
                    gap = None
                result = SolveResult(
                    status=unified,
                    values=[float(v.X) for v in xs],
                    obj_value=float(mdl.ObjVal),
                    mip_gap=gap,
                    runtime_s=runtime,
                )

            mdl.dispose()
            env.dispose()
            return result
        except gp.GurobiError as exc:
            log.error("Gurobi solve failed: %s", exc)
            return SolveResult(status=STATUS_ERROR, message=str(exc))
