import random
import math
import numpy as np
from collections import deque

def network_gen(
    amount: int,
    region: tuple[float, float, float, float],
    radius: float,
    max_attempts: int = 100
) -> list[tuple[float, float]]:
    """
    Gera pontos aleatórios maximizando a cobertura da região, garantindo que o grafo seja conexo.

    Parâmetros:
    - amount: número de pontos a serem gerados
    - region: (x_min, y_min, x_max, y_max) definindo a região retangular
    - radius: raio de conexão entre pontos
    - max_attempts: tentativas máximas para encontrar posição válida para cada ponto

    Retorna:
    - Lista de coordenadas dos pontos gerados
    """

    if amount <= 0:
        return []

    x_min, y_min, x_max, y_max = region
    points: list[tuple[float, float]] = []

    # Primeiro ponto no centro da região
    first_x = (x_min + x_max) / 2
    first_y = (y_min + y_max) / 2
    points.append((first_x, first_y))

    def is_connected(points: list[tuple[float, float]], radius: float) -> bool:
        if not points:
            return True
        visited: set[int] = set()
        queue: deque[int] = deque([0])
        adjacency: dict[int, list[int]] = {i: [] for i in range(len(points))}

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if math.dist(points[i], points[j]) <= radius:
                    adjacency[i].append(j)
                    adjacency[j].append(i)

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(adjacency[node])

        return len(visited) == len(points)

    while len(points) < amount:
        best_point: tuple[float, float] = None
        max_min_distance = 0.0

        for _ in range(max_attempts):
            anchor_idx = random.choice(range(len(points)))
            anchor = points[anchor_idx]

            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0.5 * radius, radius)

            new_x = anchor[0] + distance * math.cos(angle)
            new_y = anchor[1] + distance * math.sin(angle)

            if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                continue

            temp_points = points + [(new_x, new_y)]

            if is_connected(temp_points, radius):
                min_dist = min(math.dist((new_x, new_y), p) for p in points)
                if min_dist > max_min_distance:
                    max_min_distance = min_dist
                    best_point = (new_x, new_y)

        if best_point:
            points.append(best_point)
        else:
            if len(points) > 1:
                components: list[list[int]] = []
                visited: set[int] = set()
                for i in range(len(points)):
                    if i not in visited:
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

                if len(components) > 1:
                    comp1, comp2 = random.sample(components, 2)
                    point1 = points[random.choice(comp1)]
                    point2 = points[random.choice(comp2)]

                    mid_x = (point1[0] + point2[0]) / 2
                    mid_y = (point1[1] + point2[1]) / 2
                    direction = math.atan2(point2[1] - point1[1], point2[0] - point1[0])

                    new_x = mid_x + random.uniform(-0.3 * radius, 0.3 * radius) * math.sin(direction)
                    new_y = mid_y + random.uniform(-0.3 * radius, 0.3 * radius) * math.cos(direction)

                    new_x = float(np.clip(new_x, x_min, x_max))
                    new_y = float(np.clip(new_y, y_min, y_max))

                    points.append((new_x, new_y))
                    continue

            anchor = random.choice(points)
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius)
            new_x = anchor[0] + distance * math.cos(angle)
            new_y = anchor[1] + distance * math.sin(angle)

            new_x = float(np.clip(new_x, x_min, x_max))
            new_y = float(np.clip(new_y, y_min, y_max))
            points.append((new_x, new_y))

    # Verificação final de conectividade
    if not is_connected(points, radius):
        components: list[list[int]] = []
        visited: set[int] = set()
        for i in range(len(points)):
            if i not in visited:
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

        while len(components) > 1:
            comp1, comp2 = random.sample(components, 2)
            point1 = points[random.choice(comp1)]
            point2 = points[random.choice(comp2)]

            mid_x = (point1[0] + point2[0]) / 2
            mid_y = (point1[1] + point2[1]) / 2
            points.append((mid_x, mid_y))

            # Recalcular componentes
            components = []
            visited = set()
            for i in range(len(points)):
                if i not in visited:
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

    return points
