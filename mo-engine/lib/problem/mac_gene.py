"""The MAC gene, held to the firmware the experiment can actually run.

P1–P4 chromosomes carry a binary ``mac_protocol`` gene (0 = CSMA, 1 = TSCH).
When an experiment registers a single firmware source, that gene cannot
express anything: every individual runs the one firmware there is.
``ChromosomeBase.get_source_by_mac_protocol`` already knew this and returned
an adjusted chromosome, but the strategies kept only the source id and threw
the adjusted chromosome away. Three consequences, all measured on the paper-1
database:

* the stored gene was false — 152 of 210 individuals of a CSMA-only P2 run
  were recorded as "TSCH";
* two chromosomes differing only in that gene hash differently, so the same
  phenotype could be simulated twice and occupy two population slots;
* crossover and mutation spent part of their variation on a gene with no
  effect.

The fix is to pin the gene where chromosomes are *made* — the initial
population, crossover and mutation — so that what is hashed, simulated and
stored is the chromosome that ran. Chromosomes read back from the database
(a resume) are not touched: their hashes key the existing documents.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

#: Rounds of top-up when pinning collapses initial individuals into duplicates.
_MAX_TOP_UP_ROUNDS = 10


def pinned_mac_protocol(source_options: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The gene value forced by a single firmware source, else ``None``.

    Same rule as ``ChromosomeBase.get_source_by_mac_protocol``: the only key
    decides, ``"csma"`` is 0 and anything else is 1.
    """
    keys = list(source_options or {})
    if len(keys) != 1:
        return None
    return 0 if keys[0] == "csma" else 1


def pin_mac(chromosome: Any, protocol: Optional[int]) -> Any:
    """``chromosome`` with its MAC gene set to ``protocol`` (when it has one)."""
    if protocol is None or not hasattr(chromosome, "mac_protocol"):
        return chromosome
    if chromosome.mac_protocol == protocol:
        return chromosome
    return dataclasses.replace(chromosome, mac_protocol=protocol)


class MacPinnedAdapter:
    """A problem adapter whose new chromosomes carry the pinned MAC gene.

    Delegates everything else to the wrapped adapter. The random streams are
    untouched: the inner operators still draw for the MAC bit, so every other
    gene receives exactly the variation it would have received.
    """

    def __init__(self, inner: Any, protocol: int) -> None:
        self._inner = inner
        self._protocol = protocol

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def pinned_protocol(self) -> int:
        return self._protocol

    def _pin(self, chromosome: Any) -> Any:
        return pin_mac(chromosome, self._protocol)

    def random_individual_generator(self, size: int) -> list:
        population: list = []
        duplicates: list = []
        seen: set = set()
        for _ in range(_MAX_TOP_UP_ROUNDS + 1):
            missing = size - len(population)
            if missing <= 0:
                break
            for chromosome in self._inner.random_individual_generator(missing):
                pinned = self._pin(chromosome)
                if pinned in seen:
                    # Distinct only in the gene that was pinned away.
                    duplicates.append(pinned)
                    continue
                seen.add(pinned)
                population.append(pinned)
        # A decision space too small to fill the population with distinct
        # individuals keeps the size the strategy asked for, as the unwrapped
        # generator would.
        population.extend(duplicates[: max(0, size - len(population))])
        return population[:size]

    def crossover(self, parents: Sequence[Any]) -> Sequence[Any]:
        children = self._inner.crossover(parents)
        pinned = [self._pin(c) for c in children]
        return tuple(pinned) if isinstance(children, tuple) else pinned

    def mutate(self, chromosome: Any) -> Any:
        return self._pin(self._inner.mutate(chromosome))


def pin_mac_gene(adapter: Any, source_options: Optional[Mapping[str, Any]]) -> Any:
    """Wrap ``adapter`` when the sources force the MAC gene; else return it."""
    protocol = pinned_mac_protocol(source_options)
    if protocol is None:
        return adapter
    return MacPinnedAdapter(adapter, protocol)
