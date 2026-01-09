from typing import Iterable
from collections import defaultdict, deque
import math

Point2D = tuple[float, float]

#-------------------------------------------------
# Verification Methods
#-------------------------------------------------

def is_globally_connected(
    positions: Iterable[Point2D],
    radius: float
) -> bool:
    """
    Checks whether a set of 2D points forms a globally connected
    geometric graph under a fixed communication radius.

    Two points are adjacent if their Euclidean distance is <= radius.

    Spatial hashing (uniform grid) is used to reduce the number
    of distance checks.

    Parameters
    ----------
    positions : Iterable[(float, float)]
        Points in R^2.
    radius : float
        Communication radius.

    Returns
    -------
    bool
        True if the induced geometric graph is connected,
        False otherwise.
    """
    points: list[Point2D] = list(positions)
    n = len(points)

    if n == 0:
        return False
    if n == 1:
        return True

    radius_sq = radius * radius
    cell_size = radius

    # --------------------------------------------------
    # 1. Spatial hashing: build uniform grid
    # --------------------------------------------------
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    for idx, (x, y) in enumerate(points):
        cell = (
            int(math.floor(x / cell_size)),
            int(math.floor(y / cell_size))
        )
        grid[cell].append(idx)

    # --------------------------------------------------
    # 2. BFS over the implicit geometric graph
    # --------------------------------------------------
    visited = set()
    queue = deque([0])
    visited.add(0)

    while queue:
        i = queue.popleft()
        xi, yi = points[i]

        cx = int(math.floor(xi / cell_size))
        cy = int(math.floor(yi / cell_size))

        # Check current cell and the 8 neighbors
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_cell = (cx + dx, cy + dy)

                for j in grid.get(neighbor_cell, []):
                    if j in visited or j == i:
                        continue

                    xj, yj = points[j]
                    dx_ = xi - xj
                    dy_ = yi - yj

                    if dx_ * dx_ + dy_ * dy_ <= radius_sq:
                        visited.add(j)
                        queue.append(j)

    return len(visited) == n


def is_connected_and_k_covered(
    sensor_positions: Iterable[Point2D],
    target_positions: Iterable[Point2D],
    communication_radius: float,
    sensing_radius: float,
    k: int
) -> bool:
    """
    Checks whether a WSN configuration satisfies:
    (1) Global connectivity among sensors
    (2) k-coverage of all targets

    Parameters
    ----------
    sensor_positions : Iterable[(float, float)]
        Sensor node positions in R^2.
    target_positions : Iterable[(float, float)]
        Target positions in R^2.
    communication_radius : float
        Maximum communication distance between sensors.
    sensing_radius : float
        Sensing/coverage radius of sensors.
    k : int
        Required coverage degree for each target.

    Returns
    -------
    bool
        True if the network is globally connected AND all targets
        are k-covered; False otherwise.
    """
    sensors = list(sensor_positions)
    targets = list(target_positions)

    if k <= 0:
        raise ValueError("k must be a positive integer")

    if not sensors or not targets:
        return False

    # --------------------------------------------------
    # 1. Global connectivity check (reuse previous logic)
    # --------------------------------------------------
    if not is_globally_connected(sensors, communication_radius):
        return False

    # --------------------------------------------------
    # 2. k-coverage check (sensor -> target)
    # --------------------------------------------------
    sensing_radius_sq = sensing_radius * sensing_radius
    cell_size = sensing_radius

    # Spatial hashing for sensors
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)

    for idx, (x, y) in enumerate(sensors):
        cell = (
            int(math.floor(x / cell_size)),
            int(math.floor(y / cell_size))
        )
        grid[cell].append(idx)

    # For each target, count covering sensors
    for tx, ty in targets:
        cx = int(math.floor(tx / cell_size))
        cy = int(math.floor(ty / cell_size))

        cover_count = 0

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_cell = (cx + dx, cy + dy)

                for i in grid.get(neighbor_cell, []):
                    sx, sy = sensors[i]
                    dx_ = sx - tx
                    dy_ = sy - ty

                    if dx_ * dx_ + dy_ * dy_ <= sensing_radius_sq:
                        cover_count += 1
                        if cover_count >= k:
                            break

                if cover_count >= k:
                    break
            if cover_count >= k:
                break

        if cover_count < k:
            return False

    return True


#-------------------------------------------------
# Modification Methods
#-------------------------------------------------

def _dist(p: Point2D, q: Point2D) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _connected_components(points: list[Point2D], radius: float) -> list[list[int]]:
    """
    Retorna as componentes conexas como listas de índices.
    """
    n = len(points)
    visited = [False] * n
    adj = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _dist(points[i], points[j]) <= radius:
                adj[i].append(j)
                adj[j].append(i)

    components = []

    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            comp = []

            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)

            components.append(comp)

    return components


def _translate_component(
    points: list[Point2D],
    component: list[int],
    direction: tuple[float, float],
    delta: float
) -> None:
    """
    Move uma componente inteira ao longo de um vetor unitário 'direction'.
    """
    dx = direction[0] * delta
    dy = direction[1] * delta

    for i in component:
        x, y = points[i]
        points[i] = (x + dx, y + dy)


def make_graph_connected(
    points: list[Point2D],
    radius: float,
    step: float = 0.1
) -> list[Point2D]:
    """
    Modifica as posições dos pontos até que o grafo fique conexo,
    preservando a geometria interna das componentes.
    """
    points = points.copy()
    root = points[0]

    while True:
        components = _connected_components(points, radius)

        # Se já é conexo
        if len(components) == 1:
            break

        # Identifica componente da raiz
        root_comp = None
        for comp in components:
            if 0 in comp:
                root_comp = comp
                break

        assert root_comp is not None

        # Processa uma componente externa por vez
        for comp in components:
            if comp is root_comp:
                continue

            # Escolhe um ponto qualquer da componente
            idx = comp[0]
            px, py = points[idx]

            # Direção radial para a raiz
            vx = root[0] - px
            vy = root[1] - py
            norm = math.hypot(vx, vy)
            direction = (vx / norm, vy / norm)

            # Move incrementalmente até conectar
            while True:
                _translate_component(points, comp, direction, step)

                # Testa se houve conexão
                for i in comp:
                    for j in root_comp:
                        if _dist(points[i], points[j]) <= radius:
                            break
                    else:
                        continue
                    break
                else:
                    continue

                break

            break  # reavaliar componentes após cada conexão

    return points


def repair_connectivity_to_sink(
    candidates: list[tuple[float, float]],
    mask: list[int],
    sink: tuple[float, float],
    radius: float,
) -> tuple[bool, list[int]]:
    """
    Repairs a binary mask by connecting all components to the sink
    using minimal (heuristic) node activations.

    Parameters
    ----------
    candidates : list[(float, float)]
        Candidate positions Q.
    mask : list[int]
        Binary mask (may be disconnected).
    sink : (float, float)
        Sink position (must be active or will be activated).
    radius : float
        Reachability radius.

    Returns
    -------
    repaired_mask : tuple[bool, list[int]]
        Error flag and repaired binary mask (globally connected to sink).
    """

    n = len(candidates)
    mask = mask[:]  # copy

    # ------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------
    def dist(i: int, j: int) -> float:
        x1, y1 = candidates[i]
        x2, y2 = candidates[j]
        return math.hypot(x1 - x2, y1 - y2)

    def dist_sink(i: int) -> float:
        x1, y1 = sink
        x2, y2 = candidates[i]
        return math.hypot(x1 - x2, y1 - y2)

    # ------------------------------------------------------------
    # Build full reachability graph over candidates
    # ------------------------------------------------------------
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dist(i, j) <= radius:
                adj[i].append(j)
                adj[j].append(i)

    # ------------------------------------------------------------
    # Active connected components (ignoring sink)
    # ------------------------------------------------------------
    def active_components():
        visited = set()
        components = []

        for i in range(n):
            if mask[i] == 1 and i not in visited:
                comp = set()
                queue = deque([i])
                visited.add(i)

                while queue:
                    u = queue.popleft()
                    comp.add(u)
                    for v in adj[u]:
                        if mask[v] == 1 and v not in visited:
                            visited.add(v)
                            queue.append(v)

                components.append(comp)

        return components

    # ------------------------------------------------------------
    # Main repair loop
    # ------------------------------------------------------------
    while True:
        components = active_components()
        if not components:
            break

        # --------------------------------------------------------
        # Define sink component: active nodes reachable from sink
        # --------------------------------------------------------
        sink_comp = {
            i for i in range(n)
            if mask[i] == 1 and dist_sink(i) <= radius
        }

        # If no active node reaches sink, connect the closest one
        if not sink_comp:
            closest = min(
                (i for i in range(n) if mask[i] == 1),
                key=lambda i: dist_sink(i)
            )
            mask[closest] = 1
            sink_comp.add(closest)

        # If all components already intersect sink_comp, done
        if all(comp & sink_comp for comp in components):
            break

        # Pick one disconnected component
        other_comp = next(comp for comp in components if not (comp & sink_comp))

        # --------------------------------------------------------
        # BFS from sink_comp to other_comp on full graph
        # --------------------------------------------------------
        queue = deque()
        parent = {}

        # Virtual BFS start from all nodes in sink_comp
        for u in sink_comp:
            queue.append(u)
            parent[u] = None

        target = None

        while queue and target is None:
            u = queue.popleft()
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    if v in other_comp:
                        target = v
                        break
                    queue.append(v)

        if target is None:
            # Cannot repair connectivity: unreachable component"
            return True, []

        # --------------------------------------------------------
        # Activate path nodes
        # --------------------------------------------------------
        cur = target
        while cur is not None:
            mask[cur] = 1
            cur = parent[cur]

    return False, mask


def repair_k_coverage(
    candidates: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    mask: list[int],
    Rcov: float,
    k: int,
) -> list[int]:

    mask = mask[:]

    for tx, ty in targets:
        covering = [
            i for i, (x, y) in enumerate(candidates)
            if mask[i] == 1 and math.hypot(x - tx, y - ty) <= Rcov
        ]

        if len(covering) >= k:
            continue

        # Candidates that could cover this target
        possible = [
            i for i, (x, y) in enumerate(candidates)
            if math.hypot(x - tx, y - ty) <= Rcov
        ]

        possible.sort(
            key=lambda i: math.hypot(
                candidates[i][0] - tx,
                candidates[i][1] - ty
            )
        )

        for i in possible:
            if mask[i] == 0:
                mask[i] = 1
                covering.append(i)
                if len(covering) >= k:
                    break

    return mask
