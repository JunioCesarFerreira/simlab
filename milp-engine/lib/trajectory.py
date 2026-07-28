"""
Time-discretized mobile-node positions for the MILP models.

CRITICAL: this module must produce the SAME trajectory the simulator executes,
otherwise MILP topologies and Cooja metrics disagree. It therefore mirrors the
discretization of pylib/cooja_builder/parse_json_pos_dat.generate_positions_from_json
step by step (100 samples per parametric segment, step count proportional to
arc length, truncating int conversions, round-trip duplication, and the mote
holding its final position once the path ends). ``is_closed`` is intentionally
ignored — the simulator's positions.dat generator ignores it too.

Problem-format mobile nodes are dicts with keys:
  path_segments: list[[x_expr, y_expr]]  (parametric in t ∈ [0, 1])
  speed: float
  time_step: float
  is_round_trip: bool
"""
from typing import Any, Mapping

import numpy as np

Point2D = tuple[float, float]

# Same sampling density used by the simulator's .dat generator.
_SEGMENT_SAMPLES = 100


def _evaluate(expression: str, t_values: np.ndarray) -> np.ndarray:
    return np.array([eval(expression, {"t": t, "np": np}) for t in t_values])


def sample_step_positions(mobile_node: Mapping[str, Any]) -> list[Point2D]:
    """
    Positions the mote occupies at each of its own time steps
    (index k <=> simulator time k * time_step), exactly as written to
    positions.dat by the simulator.
    """
    path_segments = mobile_node["path_segments"]
    speed = float(mobile_node["speed"])
    time_step = float(mobile_node["time_step"])
    is_round_trip = bool(mobile_node.get("is_round_trip", False))

    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    segment_distances: list[float] = []
    for x_expr, y_expr in path_segments:
        t_values = np.linspace(0, 1, num=_SEGMENT_SAMPLES)
        x_vals = _evaluate(x_expr, t_values)
        y_vals = _evaluate(y_expr, t_values)
        x_all.append(x_vals)
        y_all.append(y_vals)
        segment_distances.append(
            float(np.sum(np.sqrt(np.diff(x_vals) ** 2 + np.diff(y_vals) ** 2)))
        )

    total_distance = float(np.sum(segment_distances))
    total_duration = total_distance / speed
    total_steps = max(1, int(total_duration / time_step))

    x_full: list[float] = []
    y_full: list[float] = []
    for x_vals, y_vals, seg_dist in zip(x_all, y_all, segment_distances):
        proportion = seg_dist / total_distance if total_distance > 0 else 1
        seg_steps = max(1, int(proportion * total_steps))
        interp_t = np.linspace(0, 1, seg_steps)
        x_interp = np.interp(interp_t, np.linspace(0, 1, len(x_vals)), x_vals)
        y_interp = np.interp(interp_t, np.linspace(0, 1, len(y_vals)), y_vals)
        x_full.extend(float(v) for v in x_interp)
        y_full.extend(float(v) for v in y_interp)

    if is_round_trip:
        x_full = x_full + x_full[::-1]
        y_full = y_full + y_full[::-1]

    return list(zip(x_full, y_full))


def position_at(
    step_positions: list[Point2D], time_step: float, sim_time: float
) -> Point2D:
    """
    Position of the mote at absolute simulation time ``sim_time`` (seconds).

    The simulator moves the mote to step_positions[k] at time k * time_step and,
    once the trajectory is exhausted, the mote stays at its final position.
    """
    k = int(sim_time // time_step)
    if k < 0:
        k = 0
    if k >= len(step_positions):
        k = len(step_positions) - 1
    return step_positions[k]
