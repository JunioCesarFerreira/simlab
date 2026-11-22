import logging
import math
import numpy as np
from pylib.config.simulator import SimulationElements
from pylib.trajectory import path_segments as travelled_segments

def evaluate_function(expression: str, t_values: np.ndarray) -> np.ndarray:
    return np.array([eval(expression, {"t": t, "np": np}) for t in t_values])

ROOT_SOURCE = "sink.c"


def mote_order(simElements: SimulationElements) -> list[tuple[str, int]]:
    """The motes as Cooja numbers them, ``("fixed" | "mobile", index)``, the root first.

    Cooja's mote 1 runs the root firmware, and the log reader takes the root's
    lines as ``Mote:1``. The root is the mote whose source is ``sink.c``: P1–P3
    already list it first among the fixed motes, P4's sink is mobile. Without
    one, the first mote is the root, as before.
    """
    order = [("fixed", i) for i in range(len(simElements["fixedMotes"]))]
    order += [("mobile", j) for j in range(len(simElements["mobileMotes"]))]
    motes = {"fixed": simElements["fixedMotes"], "mobile": simElements["mobileMotes"]}
    root = next((k for k in order if motes[k[0]][k[1]].get("sourceCode") == ROOT_SOURCE), None)
    if root is not None and root != order[0]:
        order.remove(root)
        order.insert(0, root)
    return order


def generate_positions_from_json(
    simElements: SimulationElements, 
    output_filename: str = "positions.dat",
    debug = False,
    duration_sec: float | None = None,
    ) -> tuple[list[tuple[float, float]], int]:
    """Write the positions file and return every mote's start position in
    ``mote_order``, with the number of mobile motes.

    With ``duration_sec``, each mobile mote repeats its own path. Cooja's
    Mobility plugin replays the whole file with the period of its longest
    trajectory, so a mote with a shorter path stood at its end until the
    longest one finished. The file now spans a whole number of every mote's
    periods (their least common multiple, plus the first point again, so the
    replay joins) when that is shorter than the simulation, and the whole
    simulation otherwise.
    """

    order = mote_order(simElements)
    index_of = {key: i for i, key in enumerate(order)}
    fixed_positions = [(mote["position"][0], mote["position"][1]) for mote in simElements["fixedMotes"]]
    mobile_motes = simElements["mobileMotes"]
    start: dict[tuple[str, int], tuple[float, float]] = {
        ("fixed", i): p for i, p in enumerate(fixed_positions)}

    with open(output_filename, "w") as file:
        file.write("# Fixed positions\n")
        for i, (x, y) in enumerate(fixed_positions):
            file.write(f"{index_of[('fixed', i)]} 0.00000000 {x:.2f} {y:.2f}\n")
        file.write("\n")

        file.write("# Mobile nodes\n")
        max_steps = 0
        mobile_trajectories = []

        for j, mote in enumerate(mobile_motes):
            speed = mote["speed"]
            time_step = mote["timeStep"]
            is_round_trip = mote.get("isRoundTrip", False)
            path_segments = travelled_segments(
                mote["functionPath"], mote.get("isClosed", False), is_round_trip)
            
            if debug:
                log = logging.getLogger(__name__)
                log.debug("path_segments", path_segments)

            # Avaliação dos segmentos
            x_all, y_all, segment_distances = [], [], []
            for x_expr, y_expr in path_segments:
                t_values = np.linspace(0, 1, num=100)
                x_vals = evaluate_function(x_expr, t_values)
                y_vals = evaluate_function(y_expr, t_values)
                x_all.append(x_vals)
                y_all.append(y_vals)
                segment_distances.append(np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)))

            total_distance = np.sum(segment_distances)
            total_duration = total_distance / speed
            total_steps = max(1, int(total_duration / time_step))
            max_steps = max(max_steps, total_steps)

            # Interpolação proporcional por segmento
            x_full, y_full = [], []
            for x_vals, y_vals, seg_dist in zip(x_all, y_all, segment_distances):
                proportion = seg_dist / total_distance if total_distance > 0 else 1
                seg_steps = max(1, int(proportion * total_steps))
                interp_t = np.linspace(0, 1, seg_steps)
                x_interp = np.interp(interp_t, np.linspace(0, 1, len(x_vals)), x_vals)
                y_interp = np.interp(interp_t, np.linspace(0, 1, len(y_vals)), y_vals)
                x_full.extend(x_interp)
                y_full.extend(y_interp)

            x_full = np.array(x_full)
            y_full = np.array(y_full)

            if is_round_trip:
                x_full = np.concatenate((x_full, x_full[::-1]))
                y_full = np.concatenate((y_full, y_full[::-1]))

            start[("mobile", j)] = (x_full[0], y_full[0])
            mobile_trajectories.append((index_of[("mobile", j)], x_full, y_full, time_step))

        if duration_sec is not None and mobile_trajectories:
            steps_of = {ts for *_, ts in mobile_trajectories}
            horizon = max(math.ceil(duration_sec / ts) + 1 for ts in steps_of)
            if len(steps_of) == 1:
                cycle = math.lcm(*(len(xs) for _, xs, _, _ in mobile_trajectories)) + 1
                horizon = min(horizon, cycle)
            tiled = []
            for mote_id, xs, ys, ts in mobile_trajectories:
                repeat = np.arange(horizon) % len(xs)
                tiled.append((mote_id, xs[repeat], ys[repeat], ts))
            mobile_trajectories = tiled
            max_steps = horizon

        for step in range(max_steps if duration_sec is not None else 2 * max_steps):
            for mote_id, x_full, y_full, time_step in mobile_trajectories:
                if step < len(x_full):
                    file.write(f"{mote_id} {step * time_step:.8f} {x_full[step]:.2f} {y_full[step]:.2f}\n")
            file.write("\n")

    return [start[key] for key in order], len(mobile_motes)
