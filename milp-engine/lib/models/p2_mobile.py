"""
P2 — Mobile coverage MILP (multi-period flow with installation decisions).

Ported from wsn-milp/wsn-mobile/mobile.py and wsn-milp-nsga3-p2/milp/runner.py
(solve_p2), adapted to consume the SimLab ProblemP2 draft format directly:

  { "name": "problem2", "radius_of_reach": R, "radius_of_inter": R_int,
    "region": [...], "sink": [x, y], "candidates": [[x, y], ...],
    "mobile_nodes": [{path_segments, is_closed, is_round_trip, speed, time_step}] }

Formulation (see docs/markdown/MILP_INTEGRATION_PLAN.md and the wsn-milp README):

  min  w·Σ_j y_j + Σ_t Σ_e e_ij(t)·x_ij(t)
  s.t. z_ij(t) ≤ y_j                    (installation gating, candidate endpoints)
       0 ≤ x_ij(t) ≤ C_ij(t)·z_ij(t)    (link capacity when active)
       flow conservation: mobiles inject B, candidates relay, sink absorbs Σ B
       y_j, z_ij(t) ∈ {0,1},  x_ij(t) ≥ 0

  C_ij(t) = max{0, C0·(1 − k_decay·d_ij(t))²},  e_ij(t) = d_ij(t)²  for d ≤ R_com.

Time discretization and mobile positions mirror the simulator exactly
(lib/trajectory.py): dt = min mote time_step, T = duration // dt, positions
sampled at s = τ·dt for τ = 0..T−1.
"""
import math
from typing import Any, Mapping

from pylib.config.milp_models import MILP_MODEL_SPECS

from ..solver.base import MilpBuilder
from ..trajectory import Point2D, position_at, sample_step_positions
from .base import BuiltModel, MilpModel, ParamSpec


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# Metadata lives in pylib so the REST API serves the same catalog the engine
# solves with (see pylib/config/milp_models.py).
_SPEC = MILP_MODEL_SPECS["milp_p2_mobile"]


class P2MobileModel(MilpModel):
    key = _SPEC["key"]
    problem_key = _SPEC["problem_key"]
    title = _SPEC["title"]
    description = _SPEC["description"]
    parameters = tuple(
        ParamSpec(
            name=p["name"],
            description=p["description"],
            default=p["default"],
            sweepable=p["sweepable"],
        )
        for p in _SPEC["parameters"]
    )
    solver_defaults = dict(_SPEC["solver_defaults"])

    def build(self, problem: Mapping[str, Any], params: Mapping[str, float]) -> BuiltModel:
        p = self.resolve_params(params)
        C0 = p["C0"]
        kdecay = p["kdecay"]
        B = p["B"]
        w_install = p["w_install"]
        duration = p["duration"]

        R_comm = float(problem["radius_of_reach"])
        sink: Point2D = (float(problem["sink"][0]), float(problem["sink"][1]))
        candidates: list[Point2D] = [
            (float(c[0]), float(c[1])) for c in problem["candidates"]
        ]
        mobile_nodes = list(problem.get("mobile_nodes", []))
        if not candidates:
            raise ValueError("P2 problem has no candidates.")

        # --- time grid (must match the simulator's positions.dat timeline) ---
        time_steps = [float(m["time_step"]) for m in mobile_nodes]
        dt = min(time_steps) if time_steps else 1.0
        if dt <= 0:
            raise ValueError(f"Mobile time_step must be positive, got {dt}.")
        T = max(1, int(duration // dt))

        mobile_tracks = [
            (sample_step_positions(m), float(m["time_step"])) for m in mobile_nodes
        ]

        def capacity(d: float) -> float:
            return max(0.0, C0 * (1.0 - kdecay * d) ** 2)

        def energy_cost(d: float) -> float:
            # Only edges with 0 < d <= R_comm exist, so cost is always d².
            return d * d

        # --- node universe: sink, candidates (by index), mobiles (by index) ---
        SINK = ("s", 0)
        cand_nodes = [("j", j) for j in range(len(candidates))]
        mob_nodes = [("m", k) for k in range(len(mobile_nodes))]

        def node_pos(node: tuple[str, int], tau: int) -> Point2D:
            kind, idx = node
            if kind == "s":
                return sink
            if kind == "j":
                return candidates[idx]
            positions, step = mobile_tracks[idx]
            return position_at(positions, step, tau * dt)

        builder = MilpBuilder()

        # y_j in candidates-list order: y_indices[j] == j by construction.
        # This ordering IS the ChromosomeP2.mask convention — do not reorder.
        y_indices = [builder.add_var(f"y_{j}", vtype="B") for j in range(len(candidates))]
        for j in y_indices:
            builder.add_objective_term(j, w_install)

        n_mobiles = len(mobile_nodes)
        total_demand = B * n_mobiles

        for tau in range(T):
            nodes = [SINK] + cand_nodes + mob_nodes
            positions = {n: node_pos(n, tau) for n in nodes}

            # Directed edges with distance in (0, R_comm] and positive capacity
            edges: dict[tuple, tuple[int, int]] = {}  # edge -> (z_idx, x_idx)
            edge_data: dict[tuple, tuple[float, float]] = {}  # edge -> (cap, cost)
            for i in nodes:
                for j in nodes:
                    if i == j:
                        continue
                    d = _distance(positions[i], positions[j])
                    if 0.0 < d <= R_comm:
                        cap = capacity(d)
                        if cap > 0.0:
                            z = builder.add_var(f"z_{i}_{j}_t{tau}", vtype="B")
                            x = builder.add_var(f"x_{i}_{j}_t{tau}", lb=0.0)
                            edges[(i, j)] = (z, x)
                            edge_data[(i, j)] = (cap, energy_cost(d))

            for (i, j), (z, x) in edges.items():
                cap, cost = edge_data[(i, j)]
                # Capacity: x <= cap * z
                builder.add_constr([(x, 1.0), (z, -cap)], "<=", 0.0)
                # Installation gating on candidate endpoints: z <= y
                if i[0] == "j":
                    builder.add_constr([(z, 1.0), (y_indices[i[1]], -1.0)], "<=", 0.0)
                if j[0] == "j":
                    builder.add_constr([(z, 1.0), (y_indices[j[1]], -1.0)], "<=", 0.0)
                builder.add_objective_term(x, cost)

            def flow_balance(node: tuple[str, int]) -> list[tuple[int, float]]:
                coeffs: list[tuple[int, float]] = []
                for (i, j), (_, x) in edges.items():
                    if i == node:
                        coeffs.append((x, 1.0))   # outflow
                    elif j == node:
                        coeffs.append((x, -1.0))  # inflow
                return coeffs

            # Mobiles inject B each
            for m_node in mob_nodes:
                builder.add_constr(flow_balance(m_node), "==", B)
            # Candidates are pure relays
            for c_node in cand_nodes:
                builder.add_constr(flow_balance(c_node), "==", 0.0)
            # Sink absorbs the total demand (inflow - outflow == total)
            sink_coeffs = [(x, -c) for (x, c) in flow_balance(SINK)]
            builder.add_constr(sink_coeffs, "==", total_demand)

        return BuiltModel(
            ir=builder.build(),
            y_indices=y_indices,
            meta={"T": T, "dt": dt, "n_candidates": len(candidates), "n_mobiles": n_mobiles},
        )
