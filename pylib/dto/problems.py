from typing import TypedDict

#---------------------------------------------------------------------------------------------------------
# Problems Definitions -----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# Alias para coordenadas 2D (Ω ⊂ R²)
Position = tuple[float, float]

class MobileNode(TypedDict):
    # Trajetória parametrizada de forma simbólica
    pathSegments: list[tuple[str, str]]

    isClosed: bool       # laço fechado (True) ou não
    isRoundTrip: bool    # faz ida-e-volta entre extremos

    speed: float         # velocidade média ao longo da trajetória
    timeStep: float      # Δt da discretização temporal


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

    radiusOfReach: float            # R_com
    radiusOfInter: float            # R_inter

# --- P1: cobertura de comunicação com motes móveis -----
class ProblemP1(HomogeneousProblem):
    """
    P1: dado um sink fixo, motes móveis (trajetórias Γ)
    e n motes a serem posicionados de forma que
    haja caminho até o sink para todos os instantes t.
    """
    sink: Position                  # σ
    mobileNodes: list[MobileNode]   # Γ
    numberOfRelays: int             # n
    
# --- P2: cobertura de comunicação com motes móveis -----
class ProblemP2(HomogeneousProblem):
    """
    P2: dado um sink fixo, motes móveis (trajetórias Γ)
    e posições candidatas Q, instalar motes fixos de forma que
    haja caminho até o sink para todos os instantes t.
    """
    sink: Position                  # σ
    mobileNodes: list[MobileNode]   # Γ
    candidates: list[Position]      # Q


# --- P3: cobertura de sensoriamento estática -----------
class ProblemP3(HomogeneousProblem):
    """
    P3: k-cobertura + conectividade com motes fixos.
    """
    sink: Position                  # σ
    targets: list[Position]         # Ξ 
    candidates: list[Position]      # Q
    radiusOfCover: float            # R_cov
    k_required: int                 # k

# --- P4: mobilidade do sink para coleta ----------------
class ProblemP4(HomogeneousProblem):
    """
    P4: sink móvel, motes fixos com geração contínua, energia e buffer,
    sojourn locations e grafo de mobilidade.
    """
    nodes: list[Position]           # N
    sinkBase: Position              # B
    initialEnergy: float            # E_i^0
    bufferCapacity: float           # W_i
    dataRate: float                 # δ_i
    sojourns: list[SojournLocation] # Posições possíveis de parada (L) e grafo (L, A) via adjacency
    
    speed: float         # velocidade média ao longo da trajetória
    timeStep: float      # Δt da discretização temporal