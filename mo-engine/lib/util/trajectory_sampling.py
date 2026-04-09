"""
Parametric trajectory sampling for Problems P1 and P2.

Each trajectory is described by one or more parametric segments (x(t), y(t))
with t in [0, 1]. The sampling function evaluates these expressions at
uniformly spaced t values, where the number of points is derived from the
arc length of the segment divided by a caller-supplied step parameter.

Arc length is estimated numerically using a fine reference grid of
_ARC_REF_SAMPLES points before the actual sampling is performed.
"""
import math
from pylib.config.problems import MobileNode

Point2D = tuple[float, float]

# Number of points used for the internal arc-length estimate.
# Higher values give a more accurate length but add negligible overhead.
_ARC_REF_SAMPLES = 500


def _arc_length(x_expr: str, y_expr: str) -> float:
    """
    Estimate the arc length of a parametric segment (x(t), y(t)), t in [0, 1],
    by summing chord lengths over _ARC_REF_SAMPLES uniformly spaced t values.
    """
    prev_x = float(eval(x_expr, {"t": 0.0}))
    prev_y = float(eval(y_expr, {"t": 0.0}))
    length = 0.0

    for k in range(1, _ARC_REF_SAMPLES):
        t = k / (_ARC_REF_SAMPLES - 1)
        cx = float(eval(x_expr, {"t": t}))
        cy = float(eval(y_expr, {"t": t}))
        length += math.hypot(cx - prev_x, cy - prev_y)
        prev_x, prev_y = cx, cy

    return length


def sample_trajectories(mobile_nodes: list[MobileNode], step: float) -> list[Point2D]:
    """
    Convert all P1-P2 mobile-node trajectories into a set of plane points.

    For each parametric segment (x(t), y(t)) of each mobile node, the
    arc length of the segment is estimated numerically and then divided by
    ``step`` to determine how many sample points to generate.  The actual
    points are placed at uniformly spaced t values in [0, 1], always
    including both endpoints (t=0 and t=1).

    Parameters
    ----------
    mobile_nodes : list[MobileNode]
        Mobile nodes of the P1-P2 problems.  Each node must expose a
        ``path_segments`` attribute — a list of (x_expr, y_expr) string
        tuples representing parametric equations in the variable ``t``.
    step : float
        Maximum arc-length distance between consecutive sampled points.
        Smaller values produce denser sampling.

    Returns
    -------
    list[Point2D]
        All sampled points concatenated in order
        (node 0 seg 0, node 0 seg 1, …, node 1 seg 0, …).

    Raises
    ------
    ValueError
        If ``step`` is not positive.
    """
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step!r}")

    points: list[Point2D] = []

    for mn in mobile_nodes:
        for x_expr, y_expr in mn.path_segments:
            length = _arc_length(x_expr, y_expr)

            # At least 2 points (the two endpoints); ceil ensures the step
            # constraint is satisfied even for very short segments.
            n_points = max(2, math.ceil(length / step) + 1)

            t_values = [i / (n_points - 1) for i in range(n_points)]

            for t in t_values:
                x = float(eval(x_expr, {"t": t}))  # noqa: S307
                y = float(eval(y_expr, {"t": t}))
                points.append((x, y))

    return points
