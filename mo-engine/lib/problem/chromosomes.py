from dataclasses import dataclass
from typing import Literal

# Alias for 2D coordinates (Omega ⊂ R^2)
Position = tuple[float, float]

# Binary MAC selector
# 0 -> CSMA/CA
# 1 -> TDMA
MacGene = Literal[0, 1]


@dataclass(frozen=True, slots=True)
class ChromosomeP1:
    """
    Chromosome for Problem 1 (continuous relay placement + MAC selection).

    Genes:
      - mac_protocol: binary gene selecting the MAC protocol
      - relays: continuous positions for N fixed relays in Omega
    """
    mac_protocol: MacGene
    relays: list[Position]


@dataclass(frozen=True, slots=True)
class ChromosomeP2:
    """
    Chromosome for Problem 2 (discrete candidate selection + MAC selection).

    Genes:
      - mac_protocol: binary gene selecting the MAC protocol
      - mask: binary selection mask over candidate set Q (length J)
    """
    mac_protocol: MacGene
    mask: list[int]  # each entry in {0,1}


@dataclass(frozen=True, slots=True)
class ChromosomeP3:
    """
    Chromosome for Problem 3 (discrete candidate selection + MAC selection).

    Genes:
      - mac_protocol: binary gene selecting the MAC protocol
      - mask: binary selection mask over candidate set Q (length J)
    """
    mac_protocol: MacGene
    mask: list[int]  # each entry in {0,1}


@dataclass(frozen=True, slots=True)
class ChromosomeP4:
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
