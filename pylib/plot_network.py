import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import numpy as np
from matplotlib.patches import Circle
from pylib.dto.simulator import SimulationConfig

log = logging.getLogger(__name__)

def _build_plot_network(
    points: list[tuple[float, float]],
    region: tuple[float, float, float, float],
    radius: float,
    interference_radius: float,
    paths: list[list[str]] = None
) -> Figure:
    """
    Plots a node network with two concentric disks for each node:
    - Communication disk (smaller, green)        - radius = ``radius``
    - Interference disk (larger, light gray)     - radius = ``interference_radius``

    Connections (edges) are drawn **only** when the distance
    between two nodes is less than or equal to ``radius``.

    Parameters
    ----------
    points : list[tuple[float, float]]
        List of node coordinates (x, y).
    region : tuple[float, float, float, float]
        Visualization rectangle in the form (x_min, y_min, x_max, y_max).
    radius : float
        Node communication radius (inner disk - green).
    interference_radius : float
        Node interference radius (outer disk - gray).
        Must be ≥ ``radius``.

    Additional parameters
    ---------------------
    paths : list of paths. Each path is a list of segments.
        Each segment is a sublist with 2 strings: expressions for x(t) and y(t),
        with t ∈ [0,1].

        Example of paths:
        [
            [ ["x_expr1", "y_expr1"] ],                         # Path with 1 segment
            [ ["x_expr1", "y_expr1"], ["x_expr2", "y_expr2"] ]  # Path with 2 segments
        ]
    """
    if interference_radius < radius:
        raise ValueError("interference_radius must be greater than or equal to radius")

    fig, ax = plt.subplots(figsize=(10, 8))

    # ~~~ Plot configuration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax.set_xlim(region[0], region[2])
    ax.set_ylim(region[1], region[3])
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_title(
        f"Network with {len(points)} nodes "
        f"(communication radius = {radius}, interference radius = {interference_radius})"
    )

    # rectangular region
    ax.add_patch(
        plt.Rectangle(
            (region[0], region[1]),
            region[2] - region[0],
            region[3] - region[1],
            fill=False,
            linestyle="--",
            edgecolor="red",
            linewidth=1,
        )
    )

    # ~~~ Communication edges ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.hypot(
                points[i][0] - points[j][0],
                points[i][1] - points[j][1],
            )
            if d <= radius:
                ax.plot(
                    [points[i][0], points[j][0]],
                    [points[i][1], points[j][1]],
                    color="#EE0A0A",
                    linewidth=1,
                    alpha=0.7,
                )

    # ~~~ Interference disks (larger – gray) + communication disks (smaller – green) ~
    for (x, y) in points:
        # interference disk (drawn first)
        ax.add_patch(
            Circle(
                (x, y),
                interference_radius,
                facecolor="lightgray",
                edgecolor="gray",
                alpha=0.25,
                linewidth=1.0,
            )
        )
        # communication disk (drawn on top)
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="#4ECDC4",   # aqua green
                edgecolor="green",
                alpha=0.35,
                linewidth=1.0,
            )
        )

    # ~~~ Nodes (markers) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i, (x, y) in enumerate(points):
        ax.plot(x, y, "o", markersize=8, color="gray")
        ax.text(
            x,
            y,
            str(i + 1),
            color="black",
            ha="center",
            va="center",
            fontsize=8,
        )

    # ~~~ Path plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if paths:
        for path_idx, path in enumerate(paths):
            if not isinstance(path, list) or any(len(part) != 2 for part in path):
                log.warning(f"Warning: Path ignored due to invalid structure: {path}")
                continue

            num_parts = len(path)
            ts = np.linspace(0, 1, 100 * num_parts)

            xs_total = []
            ys_total = []

            for part_idx, (x_expr, y_expr) in enumerate(path):
                t_start = part_idx / num_parts
                t_end = (part_idx + 1) / num_parts

                ts_segment = ts[(ts >= t_start) & (ts <= t_end)]
                t_local = (ts_segment - t_start) / (t_end - t_start)

                try:
                    x_eval = eval(x_expr, {"np": np, "t": t_local})
                    y_eval = eval(y_expr, {"np": np, "t": t_local})

                    # Adjust if the result is scalar
                    if np.isscalar(x_eval):
                        x_vals = np.full_like(t_local, x_eval, dtype=float)
                    else:
                        x_vals = np.array(x_eval, dtype=float)

                    if np.isscalar(y_eval):
                        y_vals = np.full_like(t_local, y_eval, dtype=float)
                    else:
                        y_vals = np.array(y_eval, dtype=float)

                except Exception:
                    log.exception(
                        f"Error while evaluating segment {part_idx} of path {path_idx}."
                    )
                    continue

                xs_total.extend(x_vals)
                ys_total.extend(y_vals)

            ax.plot(xs_total, ys_total, linestyle="--", color="blue", alpha=0.6)

    return fig


def plot_network_save_from_sim(
    file_path: str,
    sim_model: SimulationConfig
) -> None:

    fixed_motes = sim_model["simulationElements"]["fixedMotes"]
    mobile_motes = sim_model["simulationElements"]["mobileMotes"]

    points = [tuple(mote["position"]) for mote in fixed_motes]
    region = tuple(sim_model["region"])
    radius = sim_model["radiusOfReach"]
    interference_radius = sim_model["radiusOfInter"]
    paths = [list[str](mote["functionPath"]) for mote in mobile_motes]

    fig = _build_plot_network(points, region, radius, interference_radius, paths)
    plt.savefig(file_path)
    plt.close(fig)
