from typing import TypedDict

#---------------------------------------------------------------------------------------------------------
# Simulation Structure -----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

class BaseMote(TypedDict):
    name: str
    sourceCode: str
    radiusOfReach: float # Atenção! no Cooja este atributo não é utilizado.
    radiusOfInter: float # Atenção! no Cooja este atributo não é utilizado.

class FixedMote(BaseMote, TypedDict):
    position: list[float] # A priori no plano

class MobileMote(BaseMote, TypedDict):
    functionPath: list[tuple[str, str]]  # Lista de tuplas cada par define uma parte da paremetrização
    isClosed: bool
    isRoundTrip: bool
    speed: float
    timeStep: float

class SimulationElements(TypedDict):
    fixedMotes: list[FixedMote]
    mobileMotes: list[MobileMote]
    
class SimulationConfig(TypedDict):
    name: str
    duration: float
    radiusOfReach: float # Cooja admite apenas redes homogeneas
    radiusOfInter: float # Cooja admite apenas redes homogeneas
    region: tuple[float, float, float, float]
    simulationElements: SimulationElements