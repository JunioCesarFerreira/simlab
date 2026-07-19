from typing import NotRequired, TypedDict


class BaseMote(TypedDict):
    name: str
    sourceCode: str
    radiusOfReach: float  # Attention! This attribute is not used in Cooja.
    radiusOfInter: float  # Attention! This attribute is not used in Cooja.


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
    # Present only for analytical (synthetic) problems: the benchmark decision
    # vector x ∈ [0,1]^n, evaluated directly by the master-node with no motes.
    decisionVector: NotRequired[list[float]]
    # Present only for P3: sensing target positions [x, y]. Targets are not
    # motes — the Cooja builder ignores them; they exist so topology plots can
    # draw the sensing goals alongside the deployed network.
    targets: NotRequired[list[list[float]]]


class SimulationConfig(TypedDict):
    name: str
    duration: float
    randomSeed: int
    radiusOfReach: float  # Cooja only accepts homogeneous networks.
    radiusOfInter: float  # Cooja only accepts homogeneous networks.
    region: tuple[float, float, float, float]
    simulationElements: SimulationElements
