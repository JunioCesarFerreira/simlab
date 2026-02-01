from typing import TypedDict

#---------------------------------------------------------------------------------------------------------
# Simulation Structure -----------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

class BaseMote(TypedDict):
    name: str
    sourceCode: str
    radiusOfReach: float # Attention! This attribute is not used in Cooja.
    radiusOfInter: float # Attention! This attribute is not used in Cooja.

class FixedMote(BaseMote, TypedDict):
    position: list[float]

class MobileMote(BaseMote, TypedDict):
    functionPath: list[tuple[str, str]]  # List of tuples; each pair defines a part of the parameterization.
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
    randomSeed: int
    radiusOfReach: float # Cooja only accepts homogeneous networks.
    radiusOfInter: float # Cooja only accepts homogeneous networks.
    region: tuple[float, float, float, float]
    simulationElements: SimulationElements