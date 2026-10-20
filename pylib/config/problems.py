from typing import Any

# ---------------------------------------------------------------------------------------------------------
# Problems Definitions -----------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

# Alias para coordenadas 2D (Ω ⊂ R²)
Position = tuple[float, float]

# Nível de cobertura padrão (α) quando o problema não o declara: 100% dos
# pontos amostrados das trajetórias devem permanecer cobertos.
DEFAULT_MIN_COVERAGE_PERCENTAGE = 100.0


def parse_min_coverage_percentage(map: dict[str, Any]) -> float:
    """
    Lê e valida α (`min_coverage_percentage`), o nível mínimo de cobertura de
    trajetória exigido pelos problemas P1 e P2, em porcentagem.

    O campo é opcional: ausente, assume 100% (cobertura total). Valores fora
    de [0, 100] são rejeitados aqui — na borda de desserialização — para que
    um experimento mal configurado falhe antes de qualquer simulação, e não
    silenciosamente penalize toda a população.
    """
    raw = map.get("min_coverage_percentage", DEFAULT_MIN_COVERAGE_PERCENTAGE)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"problem['min_coverage_percentage'] must be a number, got {raw!r}."
        )
    if not 0.0 <= value <= 100.0:
        raise ValueError(
            f"problem['min_coverage_percentage'] must be in [0, 100], got {value}."
        )
    return value


class MobileNode:
    # Trajetória parametrizada de forma simbólica
    path_segments: list[tuple[str, str]]

    is_closed: bool       # laço fechado (True) ou não
    is_round_trip: bool   # faz ida-e-volta entre extremos

    speed: float          # velocidade média ao longo da trajetória
    time_step: float      # Δt da discretização temporal


class SojournLocation:
    id: int
    position: Position

    adjacency: list[int]    # Adjacência no grafo (L, A): ids de outros sojourns alcançáveis diretamente
    visibleNodes: list[int]  # Nós visíveis a partir desta posição (|p_i - ℓ| ≤ R_com)


# -------------------------------------------------------------------
# Problemas Homogêneos
# -------------------------------------------------------------------
class HomogeneousProblem:
    name: str

    radius_of_reach: float  # R_com
    radius_of_inter: float  # R_inter
    region: list[float]     # Ω ⊂ R²


# --- P1: cobertura de comunicação com motes móveis -----
class ProblemP1(HomogeneousProblem):
    """
    P1: dado um sink fixo, motes móveis (trajetórias Γ)
    e n motes a serem posicionados de forma que
    haja caminho até o sink para todos os instantes t.
    """
    sink: Position                  # σ
    mobile_nodes: list[MobileNode]  # Γ
    number_of_relays: int           # n
    # α: nível mínimo de cobertura das trajetórias, em % (opcional, default 100%)
    min_coverage_percentage: float = DEFAULT_MIN_COVERAGE_PERCENTAGE

    def cast(map: dict[str, Any]) -> "ProblemP1":
        obj = ProblemP1()
        obj.name = map["name"]
        obj.radius_of_reach = map["radius_of_reach"]
        obj.radius_of_inter = map["radius_of_inter"]
        obj.region = map["region"]
        obj.sink = tuple(map["sink"])
        obj.number_of_relays = map["number_of_relays"]
        obj.min_coverage_percentage = parse_min_coverage_percentage(map)
        obj.mobile_nodes = []
        for mn in map["mobile_nodes"]:
            mobile_node = MobileNode()
            mobile_node.path_segments = [tuple(seg) for seg in mn["path_segments"]]
            mobile_node.is_closed = mn["is_closed"]
            mobile_node.is_round_trip = mn["is_round_trip"]
            mobile_node.speed = mn["speed"]
            mobile_node.time_step = mn["time_step"]
            obj.mobile_nodes.append(mobile_node)
        return obj


# --- P2: cobertura de comunicação com motes móveis -----
class ProblemP2(HomogeneousProblem):
    """
    P2: dado um sink fixo, motes móveis (trajetórias Γ)
    e posições candidatas Q, instalar motes fixos de forma que
    haja caminho até o sink para todos os instantes t.
    """
    sink: Position                  # σ
    mobile_nodes: list[MobileNode]  # Γ
    candidates: list[Position]      # Q
    # α: nível mínimo de cobertura das trajetórias, em % (opcional, default 100%)
    min_coverage_percentage: float = DEFAULT_MIN_COVERAGE_PERCENTAGE

    def cast(map: dict[str, Any]) -> "ProblemP2":
        obj = ProblemP2()
        obj.name = map["name"]
        obj.radius_of_reach = map["radius_of_reach"]
        obj.radius_of_inter = map["radius_of_inter"]
        obj.region = map["region"]
        obj.sink = Position(map["sink"])
        obj.candidates = [Position(cand) for cand in map["candidates"]]
        obj.min_coverage_percentage = parse_min_coverage_percentage(map)
        obj.mobile_nodes = []
        for mn in map["mobile_nodes"]:
            mobile_node = MobileNode()
            mobile_node.path_segments = [tuple(seg) for seg in mn["path_segments"]]
            mobile_node.is_closed = mn["is_closed"]
            mobile_node.is_round_trip = mn["is_round_trip"]
            mobile_node.speed = mn["speed"]
            mobile_node.time_step = mn["time_step"]
            obj.mobile_nodes.append(mobile_node)
        return obj


# --- P3: cobertura de sensoriamento estática -----------
class ProblemP3(HomogeneousProblem):
    """
    P3: k-cobertura + conectividade com motes fixos.
    """
    sink: Position                  # σ
    targets: list[Position]         # Ξ
    candidates: list[Position]      # Q
    radius_of_cover: float          # R_cov
    k_required: int                 # k (min coverage degree)
    g_required: int                 # g (min connectivity degree)

    def cast(map: dict[str, Any]) -> "ProblemP3":
        obj = ProblemP3()
        obj.name = map["name"]
        obj.radius_of_reach = map["radius_of_reach"]
        obj.radius_of_inter = map["radius_of_inter"]
        obj.region = map["region"]
        obj.sink = Position(map["sink"])
        obj.radius_of_cover = map["radius_of_cover"]
        obj.k_required = map["k_required"]
        obj.candidates = [Position(cand) for cand in map["candidates"]]
        obj.targets = [Position(tgt) for tgt in map["targets"]]
        return obj


# --- P4: mobilidade do sink para coleta ----------------
class ProblemP4(HomogeneousProblem):
    """
    P4: sink móvel, motes fixos com geração contínua, energia e buffer,
    sojourn locations e grafo de mobilidade.
    """
    nodes: list[Position]            # N
    sink_base: Position              # B
    initial_energy: float            # E_i^0
    buffer_capacity: float           # W_i
    data_rate: float                 # δ_i
    sojourns: list[SojournLocation]  # Posições possíveis de parada (L) e grafo (L, A) via adjacency

    speed: float      # velocidade média ao longo da trajetória
    time_step: float  # Δt da discretização temporal

    max_route_length: int             # comprimento máximo da rota do sink
    tau_bounds: tuple[float, float]   # (τ_min, τ_max) tempos de sojourn

    def cast(map: dict[str, Any]) -> "ProblemP4":
        obj = ProblemP4()
        obj.name = map["name"]
        obj.radius_of_reach = map["radius_of_reach"]
        obj.radius_of_inter = map["radius_of_inter"]
        obj.region = map["region"]
        obj.nodes = [Position(node) for node in map["nodes"]]
        obj.sink_base = Position(map["sink_base"])
        obj.initial_energy = map["initial_energy"]
        obj.buffer_capacity = map["buffer_capacity"]
        obj.data_rate = map["data_rate"]
        obj.speed = map["speed"]
        obj.time_step = map["time_step"]
        obj.max_route_length = map["max_route_length"]
        obj.tau_bounds = tuple(map["tau_bounds"])
        obj.sojourns = []
        for sl in map["sojourns"]:
            sojourn = SojournLocation()
            sojourn.id = sl["id"]
            sojourn.position = Position(sl["position"])
            sojourn.adjacency = sl["adjacency"]
            sojourn.visibleNodes = sl["visibleNodes"]
            obj.sojourns.append(sojourn)
        return obj
