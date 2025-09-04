import logging
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import numpy as np
from matplotlib.patches import Circle
from dto import SimulationConfig

log = logging.getLogger(__name__)

def _build_plot_network(
    points: list[tuple[float, float]],
    region: tuple[float, float, float, float],
    radius: float,
    interference_radius: float,
    paths: list[list[str]] = None
    ) -> Figure:
    """
    Plota uma rede de nós com dois discos concêntricos para cada nó:
    - Disco de comunicação (menor, verde)           - raio = ``radius``
    - Disco de interferência (maior, cinza-claro)   - raio = ``interference_radius``

    Conexões (arestas) são traçadas **apenas** quando a distância
    entre dois nós é menor ou igual a ``radius``.

    Parâmetros
    ----------
    points : list[tuple[float, float]]
        Lista de coordenadas (x, y) dos nós.
    region : tuple[float, float, float, float]
        Retângulo de visualização na forma (x_min, y_min, x_max, y_max).
    radius : float
        Raio de comunicação do nó (disco interno - verde).
    interference_radius : float
        Raio de interferência do nó (disco externo - cinza).  
        Deve ser ≥ ``radius``.
    -----------
    Parâmetros adicionais:
    - paths: lista de caminhos. Cada caminho é uma lista de partes.
      Cada parte é uma sublista com 2 strings: expressão para x(t) e y(t), com t ∈ [0,1].

      Exemplo de paths:
      [
          [ ["x_expr1", "y_expr1"] ],                        # Caminho com 1 parte
          [ ["x_expr1", "y_expr1"], ["x_expr2", "y_expr2"] ] # Caminho com 2 partes
      ]
    """
    if interference_radius < radius:
        raise ValueError("interference_radius deve ser maior ou igual a radius")

    fig, ax = plt.subplots(figsize=(10, 8))

    # ~~~ Configuração do plot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax.set_xlim(region[0], region[2])
    ax.set_ylim(region[1], region[3])
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_title(
        f"Rede com {len(points)} nós "
        f"(raio comunicação = {radius}, raio interferência = {interference_radius})"
    )

    # região retangular
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

    # ~~~ Arestas de comunicação ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.hypot(points[i][0] - points[j][0],
                         points[i][1] - points[j][1])
            if d <= radius:
                ax.plot(
                    [points[i][0], points[j][0]],
                    [points[i][1], points[j][1]],
                    color="#EE0A0A",
                    linewidth=1,
                    alpha=0.7,
                )

    # ~~~ Discos de interferência (maiores – cinza) + comunicação (menores – verde) ~
    for (x, y) in points:
        # disco de interferência (desenhado primeiro)
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
        # disco de comunicação (desenhado por cima)
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="#4ECDC4",   # verde-água
                edgecolor="green",
                alpha=0.35,
                linewidth=1.0,
            )
        )

    # ~~~ Nós (marcadores) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i, (x, y) in enumerate(points):
        ax.plot(x, y, "o", markersize=8, color="gray")
        ax.text(x, y, str(i+1), color="black",
                ha="center", va="center", fontsize=8)

    # ~~~ Plotagem dos caminhos (paths) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                    
                    # Ajuste caso o resultado seja escalar
                    if np.isscalar(x_eval):
                        x_vals = np.full_like(t_local, x_eval, dtype=float)
                    else:
                        x_vals = np.array(x_eval, dtype=float)

                    if np.isscalar(y_eval):
                        y_vals = np.full_like(t_local, y_eval, dtype=float)
                    else:
                        y_vals = np.array(y_eval, dtype=float)
                        
                except Exception as e:
                    log.error(f"Erro ao avaliar parte {part_idx} do caminho {path_idx}: {e}")
                    continue

                xs_total.extend(x_vals)
                ys_total.extend(y_vals)

            ax.plot(xs_total, ys_total, linestyle='--', color='blue', alpha=0.6)
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