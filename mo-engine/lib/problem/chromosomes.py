from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Literal, Self
from bson import ObjectId

# Alias for 2D coordinates (Omega ⊂ R^2)
Position = tuple[float, float]

# Binary MAC selector
# 0 -> CSMA/CA
# 1 -> TSCH
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
        
    @abstractmethod
    def get_source_by_mac_protocol(
        self, options: dict[str, ObjectId]
    ) -> tuple["Chromosome", ObjectId]:
        """
        Returns to the selected source and adjusts the MacGene if there are no options.
        """
        pass


class ChromosomeBase:
    mac_protocol: MacGene

    def mac_protocol_str(self) -> str:
        return "tsch" if self.mac_protocol == 1 else "csma"

    def get_source_by_mac_protocol(
        self, options: dict[str, ObjectId]
    ) -> tuple[Self, ObjectId]:
        """
        Gets source ID and optionally returns a new chromosome if adjustment needed.
        
        Returns:
            Tuple of (possibly new chromosome, source_id)
        """
        protocol_str = self.mac_protocol_str()

        if len(options) == 1:
            only_key = next(iter(options))
            new_protocol = 0 if only_key == "csma" else 1

            if new_protocol != self.mac_protocol:
                return replace(self, mac_protocol=new_protocol), options[only_key]

        return self, options[protocol_str]


@dataclass(frozen=True, slots=True)
class ChromosomeP1(ChromosomeBase, Chromosome):
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
class ChromosomeP2(ChromosomeBase, Chromosome):
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
class ChromosomeP3(ChromosomeBase, Chromosome):
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
class ChromosomeP4(ChromosomeBase, Chromosome):
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