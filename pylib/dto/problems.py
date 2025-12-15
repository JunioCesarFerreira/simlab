from typing import TypedDict

#---------------------------------------------------------------------------------------------------------
# Problems Definitions -----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# Alias para coordenadas 2D (Ω ⊂ R²)
Position = tuple[float, float]

class MobileNode(TypedDict):
    # Trajetória parametrizada de forma simbólica
    path_segments: list[tuple[str, str]]

    is_closed: bool       # laço fechado (True) ou não
    is_round_trip: bool   # faz ida-e-volta entre extremos

    speed: float          # velocidade média ao longo da trajetória
    time_step: float      # Δt da discretização temporal


class SojournLocation(TypedDict):
    id: int
    position: Position
    
    adjacency: list[int] # Adjacência no grafo (L, A): ids de outros sojourns alcançáveis diretamente
    visibleNodes: list[int] # Nós visíveis a partir desta posição (|p_i - ℓ| ≤ R_com)

# -------------------------------------------------------------------
# Problemas Homogêneos
# -------------------------------------------------------------------
class HomogeneousProblem(TypedDict):
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