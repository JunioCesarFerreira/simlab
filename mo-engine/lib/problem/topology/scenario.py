"""Scenario-level topology caches for the topology-aware P2 variant.

Everything in this module depends on the *scenario* (candidate set, sink,
mobility, communication radius) and never on a chromosome, so it is computed
once per adapter instance and reused by every individual.

It also produces the **scenario fingerprint**: a stable digest that guards the
knowledge base against reusing observations across incompatible scenarios.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pylib.config.problems import MobileNode

from lib.util.trajectory_sampling import sample_trajectories

Point2D = tuple[float, float]

# Upper bound on the number of pseudo-time slices kept for the temporal
# connectivity descriptors.  Trajectories are sampled at R/2 resolution, so a
# couple hundred slices already resolve any gap a mobile node can produce.
MAX_TIME_SLICES: int = 200

# Quantisation applied to coordinates before fingerprinting, so that float
# round-trips through MongoDB/JSON cannot change a scenario's identity.
_FINGERPRINT_EPS = 1e-6


def _q(value: float) -> int:
    return int(round(float(value) / _FINGERPRINT_EPS))


@dataclass(frozen=True, slots=True)
class TimeSlice:
    """One pseudo-time sample of the mobile fleet.

    ``positions`` holds one point per mobile node.  ``cover_bits`` holds, for
    each mobile node, the bitmask of candidates within ``radius`` of it, and
    ``sink_covered`` flags nodes already inside the sink's own range.
    """

    index: int
    positions: tuple[Point2D, ...]
    cover_bits: tuple[int, ...]
    sink_covered: tuple[bool, ...]


class ScenarioTopology:
    """Pre-computed, chromosome-independent topology of a P2 scenario."""

    def __init__(
        self,
        candidates: Sequence[Point2D],
        sink: Point2D,
        radius: float,
        mobile_nodes: Sequence[MobileNode],
        *,
        max_time_slices: int = MAX_TIME_SLICES,
    ) -> None:
        self.candidates: list[Point2D] = [(float(x), float(y)) for x, y in candidates]
        self.sink: Point2D = (float(sink[0]), float(sink[1]))
        self.radius: float = float(radius)
        self.n_candidates: int = len(self.candidates)

        self._build_distance_cache()
        self._build_adjacency()
        self._build_time_slices(mobile_nodes, max_time_slices)
        self._build_structural_quality()

    # ------------------------------------------------------------------
    # Geometry caches
    # ------------------------------------------------------------------
    def _build_distance_cache(self) -> None:
        n = self.n_candidates
        self._dist: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = self.candidates[i]
            for j in range(i + 1, n):
                xj, yj = self.candidates[j]
                d = math.hypot(xi - xj, yi - yj)
                self._dist[i][j] = d
                self._dist[j][i] = d
        self._dist_sink: list[float] = [
            math.hypot(x - self.sink[0], y - self.sink[1]) for x, y in self.candidates
        ]
        finite = [d for row in self._dist for d in row] + self._dist_sink
        self.max_distance: float = max(finite) if finite else 1.0
        if self.max_distance <= 0.0:
            self.max_distance = 1.0

    def _build_adjacency(self) -> None:
        n = self.n_candidates
        self.adjacency: list[tuple[int, ...]] = []
        for i in range(n):
            self.adjacency.append(
                tuple(j for j in range(n) if j != i and self._dist[i][j] <= self.radius)
            )
        self.sink_neighbours: frozenset[int] = frozenset(
            i for i in range(n) if self._dist_sink[i] <= self.radius
        )
        self.degree: tuple[int, ...] = tuple(
            len(self.adjacency[i]) + (1 if i in self.sink_neighbours else 0)
            for i in range(n)
        )
        self.max_degree: int = max(self.degree) if self.degree else 1

    # ------------------------------------------------------------------
    # Temporal caches
    # ------------------------------------------------------------------
    def _build_time_slices(
        self, mobile_nodes: Sequence[MobileNode], max_time_slices: int
    ) -> None:
        """Sample every trajectory and align the samples on a pseudo-time axis.

        Each node is sampled independently at ``radius/2`` arc-length
        resolution (the same rule the base P2 coverage matrix uses).  Because
        nodes have different path lengths and speeds, slice ``t`` maps to the
        proportionally-indexed sample of every node — an alignment that is
        deterministic and monotone in time, which is all the descriptors need.
        """
        step = self.radius / 2.0 if self.radius > 0 else 1.0
        per_node: list[list[Point2D]] = []
        for node in mobile_nodes:
            samples = sample_trajectories([node], step=step)
            per_node.append(samples if samples else [])

        n_slices = min(max_time_slices, max((len(s) for s in per_node), default=0))
        self.time_slices: list[TimeSlice] = []
        if n_slices <= 0 or not per_node:
            self.n_mobile_nodes = len(per_node)
            return

        radius_sq = self.radius * self.radius
        for t in range(n_slices):
            positions: list[Point2D] = []
            cover_bits: list[int] = []
            sink_covered: list[bool] = []
            for samples in per_node:
                if not samples:
                    continue
                idx = min(len(samples) - 1, (t * len(samples)) // n_slices)
                px, py = samples[idx]
                positions.append((px, py))
                bits = 0
                for j, (cx, cy) in enumerate(self.candidates):
                    dx, dy = px - cx, py - cy
                    if dx * dx + dy * dy <= radius_sq:
                        bits |= 1 << j
                cover_bits.append(bits)
                sx, sy = self.sink
                sink_covered.append((px - sx) ** 2 + (py - sy) ** 2 <= radius_sq)
            self.time_slices.append(
                TimeSlice(
                    index=t,
                    positions=tuple(positions),
                    cover_bits=tuple(cover_bits),
                    sink_covered=tuple(sink_covered),
                )
            )
        self.n_mobile_nodes = len(per_node)

    def _build_structural_quality(self) -> None:
        """Per-candidate trajectory-coverage counts, used by the repair score."""
        counts = [0] * self.n_candidates
        for slice_ in self.time_slices:
            for bits in slice_.cover_bits:
                j = 0
                rest = bits
                while rest:
                    if rest & 1:
                        counts[j] += 1
                    rest >>= 1
                    j += 1
        self.coverage_count: tuple[int, ...] = tuple(counts)
        self.max_coverage_count: int = max(counts) if counts else 1

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def distance(self, i: int, j: int) -> float:
        """Euclidean distance between two candidates."""
        return self._dist[i][j]

    def distance_to_sink(self, i: int) -> float:
        """Euclidean distance between candidate ``i`` and the sink."""
        return self._dist_sink[i]

    def normalized_distance(self, i: int, j: int) -> float:
        """Distance in ``[0, 1]``; ``j`` may be the sink id (negative)."""
        d = self._dist_sink[i] if j < 0 else self._dist[i][j]
        return d / self.max_distance

    def neighbours(self, node: int) -> tuple[int, ...]:
        """Candidates within ``radius`` of ``node`` (``node < 0`` → the sink)."""
        if node < 0:
            return tuple(sorted(self.sink_neighbours))
        return self.adjacency[node]

    def structural_quality(self, i: int) -> float:
        """``Q_v`` in ``[0, 1]``: half connectivity degree, half trajectory reach."""
        deg = self.degree[i] / self.max_degree if self.max_degree else 0.0
        cov = (
            self.coverage_count[i] / self.max_coverage_count
            if self.max_coverage_count
            else 0.0
        )
        return 0.5 * deg + 0.5 * cov

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------
    def scenario_payload(self) -> dict[str, Any]:
        """Canonical, quantised description of the scenario geometry."""
        return {
            "candidates": [[_q(x), _q(y)] for x, y in self.candidates],
            "sink": [_q(self.sink[0]), _q(self.sink[1])],
            "radius_of_reach": _q(self.radius),
            "time_slices": [
                [[_q(x), _q(y)] for x, y in slice_.positions]
                for slice_ in self.time_slices
            ],
        }

    def fingerprint(self, extra: Mapping[str, Any] | None = None) -> str:
        """SHA-1 digest of the scenario plus any caller-supplied context.

        ``extra`` carries everything outside the geometry that changes the
        meaning of an objective vector — radio configuration, firmware/source
        repositories, the objective list and their senses, the metric
        transform configuration, aggregator, simulation duration, seeds.
        """
        payload: dict[str, Any] = {"scenario": self.scenario_payload()}
        if extra:
            payload["context"] = _canonicalize(extra)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()


def _canonicalize(value: Any) -> Any:
    """Recursively convert ``value`` into a JSON-stable structure."""
    if isinstance(value, Mapping):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, float):
        return _q(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
