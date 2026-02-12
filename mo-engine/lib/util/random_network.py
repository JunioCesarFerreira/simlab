import random
import math
import numpy as np
from collections import deque
from typing import Optional

from lib.util.connectivity import is_connected

Point2D = tuple[float, float]
RectangleRegion = tuple[float, float, float, float]

#=====================================
# PRIVATE HELPERS
#=====================================

def _get_components(points: list[Point2D], radius: float) -> list[list[int]]:
    """
    Computes connected components of the geometric graph.
    """
    components: list[list[int]] = []
    visited: set[int] = set()

    for i in range(len(points)):
        if i in visited:
            continue

        component: list[int] = []
        queue: deque[int] = deque([i])

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                component.append(node)

                for j in range(len(points)):
                    if j != node and math.dist(points[node], points[j]) <= radius:
                        queue.append(j)

        components.append(component)

    return components


def _fallback_insert_point(
    points: list[Point2D],
    region: RectangleRegion,
    radius: float,
    rng: random.Random,
) -> Point2D:
    """
    Executes the fallback insertion strategy when no valid best_point
    is found during stochastic sampling.

    Strategy hierarchy:
    1) Try to connect disconnected components.
    2) Otherwise perform anchored random insertion.
    """
    x_min, y_min, x_max, y_max = region

    # -------------------------------------------------
    # Try to connect components
    # -------------------------------------------------
    if len(points) > 1:
        components = _get_components(points, radius)

        if len(components) > 1:
            comp1, comp2 = rng.sample(components, 2)
            p1 = points[rng.choice(comp1)]
            p2 = points[rng.choice(comp2)]

            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2

            direction = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

            new_x = mid_x + rng.uniform(-0.3 * radius, 0.3 * radius) * math.sin(direction)
            new_y = mid_y + rng.uniform(-0.3 * radius, 0.3 * radius) * math.cos(direction)

            new_x = float(np.clip(new_x, x_min, x_max))
            new_y = float(np.clip(new_y, y_min, y_max))

            return (new_x, new_y)

    # -------------------------------------------------
    # Anchored random insertion
    # -------------------------------------------------
    anchor = rng.choice(points)

    angle = rng.uniform(0, 2 * math.pi)
    distance = rng.uniform(0, radius)

    new_x = anchor[0] + distance * math.cos(angle)
    new_y = anchor[1] + distance * math.sin(angle)

    new_x = float(np.clip(new_x, x_min, x_max))
    new_y = float(np.clip(new_y, y_min, y_max))

    return (new_x, new_y)


def _repair_connectivity(
    points: list[Point2D],
    radius: float,
    rng: random.Random,
) -> list[Point2D]:
    """
    Ensures global connectivity by iteratively inserting midpoint relays
    between disconnected components.
    """
    components = _get_components(points, radius)

    while len(components) > 1:
        comp1, comp2 = rng.sample(components, 2)

        p1 = points[rng.choice(comp1)]
        p2 = points[rng.choice(comp2)]

        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2

        points.append((mid_x, mid_y))

        components = _get_components(points, radius)

    return points


#=====================================
# PUBLIC API
#=====================================

def continuous_network_gen(
    amount: int,
    region: RectangleRegion,
    radius: float,
    sink: Point2D,
    rng: random.Random,
    max_attempts: int = 100
) -> list[Point2D]:
    """
    Generates spatial points attempting to maximize coverage while
    preserving geometric connectivity.

    Growth model:
    - Seed at region center.
    - Iteratively sample anchored candidates.
    - Select the point maximizing minimum distance.
    - Apply fallback insertion when sampling fails.
    - Perform final connectivity repair if necessary.

    Parameters
    ----------
    amount : int
        Number of points to generate.
    region : (x_min, y_min, x_max, y_max)
        Rectangular spatial domain.
    radius : float
        Communication radius.
    sink: (float, float)
        Sink position (if None, uses region center).
    rng : random.Random
        RNG for reproducibility.
    max_attempts : int
        Candidate samples per iteration.

    Returns
    -------
    list[(float, float)]
        Generated point set.
    """
    if amount <= 0:
        return []

    x_min, y_min, x_max, y_max = region

    points: list[Point2D] = []

    # -------------------------------------------------
    # Sink position or seed at geometric center as sink
    # -------------------------------------------------
    first_x = sink[0] if sink else (x_min + x_max) / 2
    first_y = sink[1] if sink else (y_min + y_max) / 2
    points.append((first_x, first_y))

    # -------------------------------------------------
    # Growth process
    # -------------------------------------------------
    while len(points) < amount + 1: # +1 for sink

        best_point: Optional[Point2D] = None
        max_min_distance = 0.0

        for _ in range(max_attempts):

            anchor = rng.choice(points)

            angle = rng.uniform(0, 2 * math.pi)
            distance = rng.uniform(0.5 * radius, radius)

            new_x = anchor[0] + distance * math.cos(angle)
            new_y = anchor[1] + distance * math.sin(angle)

            if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                continue

            candidate = (new_x, new_y)
            temp_points = points + [candidate]

            if is_connected(temp_points, radius):

                min_dist = min(math.dist(candidate, p) for p in points)

                if min_dist > max_min_distance:
                    max_min_distance = min_dist
                    best_point = candidate

        # -------------------------------------------------
        # Insert best candidate or fallback
        # -------------------------------------------------
        if best_point is not None:
            points.append(best_point)
        else:
            new_point = _fallback_insert_point(points, region, radius, rng)
            points.append(new_point)

    # -------------------------------------------------
    # Final repair
    # -------------------------------------------------
    if not is_connected(points, radius):
        points = _repair_connectivity(points, radius, rng)

    return points[1:] # exclude sink


def stochastic_reachability_mask(
    candidates: list[tuple[float, float]],
    sink: tuple[float, float],
    radius: float,
    rng: random.Random
) -> list[int]:
    """
    Stochastic reachability-guided growth algorithm.

    Parameters
    ----------
    candidates : list[(float, float)]
        list of candidate positions Q.
    sink : (float, float)
        Root position (must be in candidates).
    radius : float
        Reachability radius.
    rng : random.Random
        RNG for reproducibility.

    Returns
    -------
    mask : list[int]
        Binary mask over candidates (1 = selected, 0 = not selected).
    """
    n = len(candidates)

    # ------------------------------------------------------------
    # State sets (indices)
    # ------------------------------------------------------------
    FREE = set(range(n))
    SELECTED = set()
    DISCARDED = set()

    # Frontier (active expansion nodes)
    frontier = set()

    # ------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------
    def dist_sink(i: int) -> float:
        x1, y1 = sink
        x2, y2 = candidates[i]
        return math.hypot(x1 - x2, y1 - y2)
    
    def dist(i: int, j: int) -> float:
        x1, y1 = candidates[i]
        x2, y2 = candidates[j]
        return math.hypot(x1 - x2, y1 - y2)

    # ------------------------------------------------------------
    # Initial expansion from sink (virtual root)
    # ------------------------------------------------------------
    reachable_from_sink = [
        i for i in FREE
        if dist_sink(i) <= radius
    ]

    if reachable_from_sink:
        c = rng.randint(1, len(reachable_from_sink))
        chosen = set(rng.sample(reachable_from_sink, c))
        rejected = set(reachable_from_sink) - chosen

        SELECTED.update(chosen)
        frontier.update(chosen)

        FREE.difference_update(chosen)
        FREE.difference_update(rejected)
        DISCARDED.update(rejected)

    # ------------------------------------------------------------
    # Main growth loop (from selected candidates)
    # ------------------------------------------------------------
    while frontier and FREE:
        u = frontier.pop()

        reachable = [
            v for v in FREE
            if dist(u, v) <= radius
        ]

        if not reachable:
            continue

        c = rng.randint(1, len(reachable))
        chosen = set(rng.sample(reachable, c))
        rejected = set(reachable) - chosen

        SELECTED.update(chosen)
        frontier.update(chosen)

        FREE.difference_update(chosen)
        FREE.difference_update(rejected)
        DISCARDED.update(rejected)

    # ------------------------------------------------------------
    # Build binary mask
    # ------------------------------------------------------------
    mask = [1 if i in SELECTED else 0 for i in range(n)]
    return mask