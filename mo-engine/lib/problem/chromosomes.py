from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

# Alias for 2D coordinates (Omega ⊂ R^2)
Position = tuple[float, float]

# Binary MAC selector
# 0 -> CSMA/CA
# 1 -> TDMA
MacGene = Literal[0, 1]

class Chromosome(ABC):
    """
    Abstract base class for chromosomes.
    """

    @abstractmethod
    def to_dict(self) -> dict:
        """
        Converts the chromosome to a dictionary representation.
        """
        pass


@dataclass(frozen=True, slots=True)
class ChromosomeP1(Chromosome):
    """
    Chromosome for Problem 1 (continuous relay placement + MAC selection).

    Genes:
      - mac_protocol: binary gene selecting the MAC protocol
      - relays: continuous positions for N fixed relays in Omega
    """
    mac_protocol: MacGene
    relays: list[Position]
    
    def to_dict(self) -> dict:
        return {
            "mac_protocol": self.mac_protocol,
            "relays": [ {"x": pos[0], "y": pos[1]} for pos in self.relays ]
        }


@dataclass(frozen=True, slots=True)
class ChromosomeP2(Chromosome):
    """
    Chromosome for Problem 2 (discrete candidate selection + MAC selection).

    Genes:
      - mac_protocol: binary gene selecting the MAC protocol
      - mask: binary selection mask over candidate set Q (length J)
    """
    mac_protocol: MacGene
    mask: list[int]  # each entry in {0,1}
    
    def to_dict(self) -> dict:
        return {
            "mac_protocol": self.mac_protocol,
            "mask": self.mask
        }


@dataclass(frozen=True, slots=True)
class ChromosomeP3(Chromosome):
    """
    Chromosome for Problem 3 (discrete candidate selection + MAC selection).

    Genes:
      - mac_protocol: binary gene selecting the MAC protocol
      - mask: binary selection mask over candidate set Q (length J)
    """
    mac_protocol: MacGene
    mask: list[int]  # each entry in {0,1}
    
    def to_dict(self) -> dict:
        return {
            "mac_protocol": self.mac_protocol,
            "mask": self.mask
        }


@dataclass(frozen=True, slots=True)
class ChromosomeP4(Chromosome):
    """
    Chromosome for Problem 4 (mobile sink route + sojourn times + MAC selection).

    Genes:
      - mac_protocol: binary gene selecting the MAC protocol
      - route: sequence of indices over L (discrete mobility graph)
      - sojourn_times: nonnegative real times aligned to route length
    """
    mac_protocol: MacGene
    route: list[int]
    sojourn_times: list[float]
    
    def to_dict(self) -> dict:
        return {
            "mac_protocol": self.mac_protocol,
            "route": self.route,
            "sojourn_times": self.sojourn_times
        }
