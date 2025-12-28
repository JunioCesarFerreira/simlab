from typing import Any
#---------------------------------------------------------------------------------------------------------
# Problems Definitions -----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# Alias para coordenadas 2D (Ω ⊂ R²)
Position = tuple[float, float]

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
    
    adjacency: list[int] # Adjacência no grafo (L, A): ids de outros sojourns alcançáveis diretamente
    visibleNodes: list[int] # Nós visíveis a partir desta posição (|p_i - ℓ| ≤ R_com)

# -------------------------------------------------------------------
# Problemas Homogêneos
# -------------------------------------------------------------------
class HomogeneousProblem:
    name: str

    radius_of_reach: float            # R_com
    radius_of_inter: float            # R_inter
    region: list[float]               # Ω ⊂ R²

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
    
    def cast(map: dict[str, Any]) -> "ProblemP1":
        # Conversão de dicionário genérico para DTO fortemente tipado
        obj = ProblemP1()
        obj.name = map["name"]
        obj.radius_of_reach = map["radius_of_reach"]
        obj.radius_of_inter = map["radius_of_inter"]
        obj.region = map["region"]
        obj.sink = tuple(map["sink"])
        obj.number_of_relays = map["number_of_relays"]
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
    
    def cast(map: dict[str, Any]) -> "ProblemP2":
        # Conversão de dicionário genérico para DTO fortemente tipado
        obj = ProblemP2()
        obj.name = map["name"]
        obj.radius_of_reach = map["radius_of_reach"]
        obj.radius_of_inter = map["radius_of_inter"]
        obj.region = map["region"]
        obj.sink = tuple(map["sink"])
        obj.candidates = [tuple(cand["position"]) for cand in map["candidates"]]
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
    k_required: int                 # k
    
    def cast(map: dict[str, Any]) -> "ProblemP3":
        # Conversão de dicionário genérico para DTO fortemente tipado
        obj = ProblemP3()
        obj.name = map["name"]
        obj.radius_of_reach = map["radius_of_reach"]
        obj.radius_of_inter = map["radius_of_inter"]
        obj.region = map["region"]
        obj.sink = tuple(map["sink"])
        obj.radius_of_cover = map["radius_of_cover"]
        obj.k_required = map["k_required"]
        obj.candidates = [tuple(cand["position"]) for cand in map["candidates"]]
        obj.targets = [tuple(tgt["position"]) for tgt in map["targets"]]
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
    
    speed: float          # velocidade média ao longo da trajetória
    time_step: float      # Δt da discretização temporal
    
    def cast(map: dict[str, Any]) -> "ProblemP4":
        # Conversão de dicionário genérico para DTO fortemente tipado
        obj = ProblemP4()
        obj.name = map["name"]
        obj.radius_of_reach = map["radius_of_reach"]
        obj.radius_of_inter = map["radius_of_inter"]
        obj.region = map["region"]
        obj.nodes = [tuple(node) for node in map["nodes"]]
        obj.sink_base = tuple(map["sink_base"])
        obj.initial_energy = map["initial_energy"]
        obj.buffer_capacity = map["buffer_capacity"]
        obj.data_rate = map["data_rate"]
        obj.speed = map["speed"]
        obj.time_step = map["time_step"]
        obj.sojourns = []
        for sl in map["sojourns"]:
            sojourn = SojournLocation()
            sojourn.id = sl["id"]
            sojourn.position = tuple(sl["position"])
            sojourn.adjacency = sl["adjacency"]
            sojourn.visibleNodes = sl["visibleNodes"]
            obj.sojourns.append(sojourn)
        return obj