"""
HiGHS backend (open source, no license required).

Serves as the first-class fallback when no Gurobi license is available:
P2/P3 are pure MILPs (binary + continuous linear), fully within HiGHS's scope.
"""
import logging
from typing import Optional

import numpy as np

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


class HighsBackend(SolverBackend):
    name = "highs"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import highspy  # noqa: F401
            return True
        except ImportError:
            return False

    def solve(
        self,
        ir: MilpIR,
        time_limit_s: Optional[float] = None,
        mip_gap: Optional[float] = None,
    ) -> SolveResult:
        import highspy

        try:
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            if time_limit_s is not None:
                h.setOptionValue("time_limit", float(time_limit_s))
            if mip_gap is not None:
                h.setOptionValue("mip_rel_gap", float(mip_gap))

            inf = highspy.kHighsInf
            n = ir.n_vars
            costs = np.zeros(n, dtype=np.double)
            for idx, coef in ir.objective:
                costs[idx] = coef
            lower = np.array(
                [-inf if lb == float("-inf") else lb for lb in ir.var_lb],
                dtype=np.double,
            )
            upper = np.array(
                [inf if ub == float("inf") else ub for ub in ir.var_ub],
                dtype=np.double,
            )
            h.addCols(
                n, costs, lower, upper,
                0, np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                np.array([], dtype=np.double),
            )
            for i, vtype in enumerate(ir.var_types):
                if vtype == "B":
                    h.changeColIntegrality(i, highspy.HighsVarType.kInteger)

            row_lower, row_upper = [], []
            starts, indices, values = [], [], []
            nnz = 0
            for c in ir.constraints:
                starts.append(nnz)
                for idx, coef in c.coeffs:
                    indices.append(idx)
                    values.append(coef)
                    nnz += 1
                if c.sense == "<=":
                    row_lower.append(-inf)
                    row_upper.append(c.rhs)
                elif c.sense == ">=":
                    row_lower.append(c.rhs)
                    row_upper.append(inf)
                else:
                    row_lower.append(c.rhs)
                    row_upper.append(c.rhs)
            h.addRows(
                len(ir.constraints),
                np.array(row_lower, dtype=np.double),
                np.array(row_upper, dtype=np.double),
                nnz,
                np.array(starts, dtype=np.int32),
                np.array(indices, dtype=np.int32),
                np.array(values, dtype=np.double),
            )

            h.run()

            status = h.getModelStatus()
            info = h.getInfo()
            runtime = float(h.getRunTime())
            has_incumbent = (
                info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
            )
            gap = float(info.mip_gap) if has_incumbent and info.mip_gap >= 0 else None

            if status == highspy.HighsModelStatus.kInfeasible:
                return SolveResult(status=STATUS_INFEASIBLE, runtime_s=runtime)
            if status == highspy.HighsModelStatus.kOptimal and has_incumbent:
                values_out = list(h.getSolution().col_value)
                return SolveResult(
                    status=STATUS_OPTIMAL,
                    values=values_out,
                    obj_value=float(info.objective_function_value),
                    mip_gap=gap,
                    runtime_s=runtime,
                )
            if has_incumbent:
                # Time/iteration limit with a feasible incumbent
                values_out = list(h.getSolution().col_value)
                return SolveResult(
                    status=STATUS_TIME_LIMIT,
                    values=values_out,
                    obj_value=float(info.objective_function_value),
                    mip_gap=gap,
                    runtime_s=runtime,
                    message=f"highs status {status}",
                )
            return SolveResult(
                status=STATUS_NO_SOLUTION,
                runtime_s=runtime,
                message=f"highs status {status} with no incumbent",
            )
        except Exception as exc:  # highspy raises plain Exceptions
            log.error("HiGHS solve failed: %s", exc)
            return SolveResult(status=STATUS_ERROR, message=str(exc))
