"""Observed link importance ``R = (r_ij)`` for the topology-aware P2 variant.

``RoutingKnowledge`` is deliberately decoupled from the Cooja log format: it
consumes :class:`RoutingObservation` values (link lists over *candidate
indices*), whatever produced them.  Two producers exist today:

* :meth:`RoutingKnowledge.observe_tree` — the structural sink-rooted tree of an
  individual that was really simulated.  Always available.
* :func:`observation_from_dodag` — the exact RPL parent map persisted on
  ``Simulation.dodag`` (see ``pylib/rpl_dodag.py``), usable only when the
  caller can map node addresses back to candidate indices.  Returns ``None``
  when that mapping is unavailable, so nothing here is mandatory.

The initial importance model is the frequency one from the design notes,

    r_ij = n_ij / N,

with ``n_ij`` the number of observations containing the link and ``N`` the
number of observations.  Richer weightings (traffic volume, link quality,
stability) can be folded in later through ``RoutingObservation.weights``
without changing any caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from .rooted_tree import SINK_NODE, ParentArrayTree

log = logging.getLogger(__name__)

Link = tuple[int, int]


@dataclass(frozen=True, slots=True)
class RoutingObservation:
    """One observed routing structure over candidate indices.

    ``links`` are ``(child, parent)`` pairs; ``SINK_NODE`` denotes the sink.
    ``weights`` optionally scales the contribution of individual links (traffic
    volume, forwarded bytes, ETX-derived quality, ...); missing entries count
    as ``1.0``.
    """

    links: tuple[Link, ...]
    weights: Mapping[Link, float] = field(default_factory=dict)
    source: str = "structural"


class RoutingKnowledge:
    """Accumulates ``r_ij`` over the observations of one scenario."""

    def __init__(self, scenario_fingerprint: str = "") -> None:
        self.scenario_fingerprint = scenario_fingerprint
        self._counts: dict[Link, float] = {}
        self._observations: int = 0
        self._seen: set[str] = set()
        # node_importance_map() is called once per repair pass, i.e. thousands
        # of times per generation, while R only changes when an individual is
        # really simulated. Cache it and invalidate on ingestion.
        self._importance_cache: Optional[dict[int, float]] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def observe(self, observation: RoutingObservation, key: str | None = None) -> bool:
        """Fold one observation into ``R``.

        ``key`` makes ingestion idempotent across restarts (pass the simulation
        id or the genome hash).  Returns ``False`` when the observation was
        already known.
        """
        if key is not None:
            if key in self._seen:
                return False
            self._seen.add(key)
        self._observations += 1
        self._importance_cache = None
        for link in observation.links:
            w = float(observation.weights.get(link, 1.0))
            self._counts[link] = self._counts.get(link, 0.0) + w
            # Importance is undirected for the repair heuristic: a candidate
            # observed forwarding towards the sink is equally valuable when the
            # bridge is rebuilt in the opposite orientation.
            rev = (link[1], link[0])
            self._counts[rev] = self._counts.get(rev, 0.0) + w
        return True

    def observe_tree(
        self,
        tree: ParentArrayTree,
        key: str | None = None,
        weight: float = 1.0,
    ) -> bool:
        """Record the structural sink-rooted tree of a simulated individual."""
        links: list[Link] = []
        weights: dict[Link, float] = {}
        for node in tree.nodes():
            parent = tree.parent(node)
            if parent is None:
                continue
            link = (node, parent)
            links.append(link)
            weights[link] = weight
        return self.observe(
            RoutingObservation(links=tuple(links), weights=weights, source="structural"),
            key=key,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def observation_count(self) -> int:
        return self._observations

    def importance(self, i: int, j: int) -> float:
        """``r_ij`` in ``[0, 1]``; ``0.0`` while no history exists."""
        if self._observations <= 0:
            return 0.0
        return min(1.0, self._counts.get((i, j), 0.0) / self._observations)

    def node_importance(self, node: int) -> float:
        """``I_v``: importance of the best link incident to ``node``."""
        if self._observations <= 0:
            return 0.0
        best = 0.0
        for (a, b), count in self._counts.items():
            if a == node or b == node:
                best = max(best, count)
        return min(1.0, best / self._observations)

    def node_importance_map(self) -> dict[int, float]:
        """``I_v`` for every node seen so far (cached batch form)."""
        if self._observations <= 0:
            return {}
        if self._importance_cache is None:
            best: dict[int, float] = {}
            for (a, b), count in self._counts.items():
                best[a] = max(best.get(a, 0.0), count)
                best[b] = max(best.get(b, 0.0), count)
            self._importance_cache = {
                n: min(1.0, c / self._observations) for n, c in best.items()
            }
        return self._importance_cache

    # ------------------------------------------------------------------
    # Persistence helpers (restart / resume)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_fingerprint": self.scenario_fingerprint,
            "observations": self._observations,
            "counts": [[a, b, w] for (a, b), w in sorted(self._counts.items())],
            "seen": sorted(self._seen),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoutingKnowledge":
        obj = cls(str(data.get("scenario_fingerprint", "")))
        obj._observations = int(data.get("observations", 0))
        obj._counts = {
            (int(a), int(b)): float(w) for a, b, w in data.get("counts", [])
        }
        obj._seen = {str(k) for k in data.get("seen", [])}
        obj._importance_cache = None
        return obj


# ----------------------------------------------------------------------
# Optional Cooja/RPL bridge
# ----------------------------------------------------------------------
def observation_from_dodag(
    dodag: Optional[Mapping[str, Any]],
    address_to_index: Optional[Mapping[str, int]],
) -> Optional[RoutingObservation]:
    """Convert a persisted ``Simulation.dodag`` into a routing observation.

    ``address_to_index`` maps the node addresses used in the log to candidate
    indices (``SINK_NODE`` for the root).  Returns ``None`` whenever the DODAG
    is absent or the mapping cannot resolve the addresses — RPL statistics are
    an optional enrichment, never a requirement.
    """
    if not dodag or not address_to_index:
        return None
    edges = ((dodag.get("tree") or {}).get("edges")) or {}
    if not edges:
        return None

    links: list[Link] = []
    for child, parent in edges.items():
        ci = address_to_index.get(str(child))
        pi = address_to_index.get(str(parent))
        if ci is None or pi is None:
            log.debug("[routing] Unmapped DODAG edge %s -> %s; skipped.", child, parent)
            continue
        links.append((ci, pi))
    if not links:
        return None
    return RoutingObservation(links=tuple(links), source="rpl_dodag")


def merge_observations(observations: Iterable[RoutingObservation]) -> RoutingObservation:
    """Flatten several observations into one (used for multi-seed runs)."""
    links: list[Link] = []
    weights: dict[Link, float] = {}
    source_parts: list[str] = []
    for obs in observations:
        source_parts.append(obs.source)
        for link in obs.links:
            links.append(link)
            weights[link] = weights.get(link, 0.0) + float(obs.weights.get(link, 1.0))
    unique: list[Link] = list(dict.fromkeys(links))
    return RoutingObservation(
        links=tuple(unique),
        weights=weights,
        source="+".join(sorted(set(source_parts))) or "merged",
    )


def sink_aware_links(parent_map: Mapping[int, int]) -> Sequence[Link]:
    """Normalise a ``child -> parent`` map into ``(child, parent)`` links."""
    return tuple((int(c), int(p) if p is not None else SINK_NODE) for c, p in parent_map.items())
