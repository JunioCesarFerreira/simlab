"""Historical base of **really evaluated** individuals.

``EvaluationKnowledgeBase`` holds

    D = { (phi(x_i), x_i, f(x_i)) }_{i=1..N}

for one ``scenario_fingerprint``.  Only ground truth ever enters it — objectives
produced by a Cooja/synthetic simulation or replayed from the exact genome
cache.  Estimated objectives are *never* stored: that separation is what keeps
the estimator from training on its own output.

Nothing new is persisted for it: the base is rebuilt at start-up from the
``genome_cache`` collection the strategies already maintain (chromosome +
objectives in minimization space), which is also what makes restart/resume
work without a second source of truth.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np

from .dominance import non_dominated, objective_ranges

log = logging.getLogger(__name__)

# Objective magnitude above which a record is treated as a sentinel (an
# infeasibility penalty or the "no metrics" marker) rather than a measurement.
# Matches the frontend PENALTY_THRESHOLD, and keeps penalties out of the
# regression input where they would dwarf every real objective.
SENTINEL_THRESHOLD: float = 1e8

EvaluationType = str  # "simulated" | "cached" | "analytical"


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One really-evaluated individual."""

    scenario_fingerprint: str
    chromosome_hash: str
    chromosome: dict[str, Any]
    descriptors: dict[str, float]
    descriptor_vector: tuple[float, ...]
    objectives: tuple[float, ...]
    objectives_minimization: bool = True
    evaluation_type: EvaluationType = "simulated"
    seed_count: int = 0
    simulation_ids: tuple[str, ...] = ()
    routing_statistics: Optional[dict[str, Any]] = None
    generation: int = -1
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def mask(self) -> tuple[int, ...]:
        return tuple(int(b) for b in self.chromosome.get("mask", ()))

    @property
    def is_measurement(self) -> bool:
        """Whether the objectives are a real measurement, not a sentinel."""
        values = np.asarray(self.objectives, dtype=float)
        return bool(values.size and np.all(np.isfinite(values)) and np.all(np.abs(values) < SENTINEL_THRESHOLD))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_fingerprint": self.scenario_fingerprint,
            "chromosome_hash": self.chromosome_hash,
            "chromosome": self.chromosome,
            "descriptors": self.descriptors,
            "objectives": list(self.objectives),
            "objectives_minimization": self.objectives_minimization,
            "evaluation_type": self.evaluation_type,
            "seed_count": self.seed_count,
            "simulation_ids": list(self.simulation_ids),
            "routing_statistics": self.routing_statistics,
            "generation": self.generation,
            "created_at": self.created_at,
        }


class EvaluationKnowledgeBase:
    """In-memory index of the evaluated individuals of one scenario."""

    def __init__(self, scenario_fingerprint: str) -> None:
        self.scenario_fingerprint = scenario_fingerprint
        self._records: dict[str, EvaluationRecord] = {}
        self._front_cache: Optional[list[tuple[float, ...]]] = None
        self._range_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, record: EvaluationRecord) -> bool:
        """Insert ``record``; returns ``False`` when it is a duplicate.

        Records carrying a different ``scenario_fingerprint`` are rejected —
        knowledge is never reused across incompatible scenarios.
        """
        if record.scenario_fingerprint != self.scenario_fingerprint:
            log.debug(
                "[kb] Rejected record %s: fingerprint %s != %s.",
                record.chromosome_hash,
                record.scenario_fingerprint,
                self.scenario_fingerprint,
            )
            return False
        if record.chromosome_hash in self._records:
            return False
        self._records[record.chromosome_hash] = record
        self._front_cache = None
        self._range_cache = None
        return True

    def extend(self, records: Iterable[EvaluationRecord]) -> int:
        return sum(1 for r in records if self.add(r))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, chromosome_hash: str) -> bool:
        return chromosome_hash in self._records

    def __iter__(self) -> Iterator[EvaluationRecord]:
        return iter(self._records.values())

    def get(self, chromosome_hash: str) -> Optional[EvaluationRecord]:
        return self._records.get(chromosome_hash)

    def records(self) -> list[EvaluationRecord]:
        """Every stored record, in insertion order."""
        return list(self._records.values())

    def measurements(self) -> list[EvaluationRecord]:
        """Records usable as regression targets (penalties excluded)."""
        return [r for r in self._records.values() if r.is_measurement]

    @property
    def training_size(self) -> int:
        return len(self.measurements())

    def training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, Y)`` for the estimator; empty arrays when nothing is usable."""
        usable = self.measurements()
        if not usable:
            return np.zeros((0, 0)), np.zeros((0, 0))
        X = np.asarray([r.descriptor_vector for r in usable], dtype=float)
        Y = np.asarray([r.objectives for r in usable], dtype=float)
        return X, Y

    def known_front(self) -> list[tuple[float, ...]]:
        """Non-dominated objective vectors among the real measurements.

        Comparing an optimistic estimate against this set is equivalent to
        comparing it against the whole archive: clear dominance is transitive,
        so if any evaluated solution clearly dominates a point, a front member
        does too.
        """
        if self._front_cache is None:
            usable = self.measurements()
            points = [r.objectives for r in usable]
            self._front_cache = [points[i] for i in non_dominated(points)] if points else []
        return self._front_cache

    def objective_ranges(self) -> np.ndarray:
        """Per-objective spread of the measurements (for scaled thresholds)."""
        if self._range_cache is None:
            self._range_cache = objective_ranges([r.objectives for r in self.measurements()])
        return self._range_cache

    # ------------------------------------------------------------------
    # Novelty support
    # ------------------------------------------------------------------
    def nearest_hamming(self, mask: Sequence[int]) -> float:
        """Normalised Hamming distance to the closest stored chromosome.

        Returns ``1.0`` when the base is empty — an unknown region is maximally
        novel, which biases the very first decisions towards simulation.
        """
        bits = np.asarray(mask, dtype=int)
        if bits.size == 0:
            return 0.0
        best = 1.0
        found = False
        for record in self._records.values():
            other = np.asarray(record.mask, dtype=int)
            if other.size != bits.size:
                continue
            found = True
            best = min(best, float(np.count_nonzero(bits != other)) / bits.size)
            if best == 0.0:
                break
        return best if found else 1.0

    def nearest_descriptor_distance(self, vector: Sequence[float]) -> float:
        """Normalised descriptor distance to the closest stored individual.

        The distance is min-max normalised per feature with the base's own
        statistics and divided by ``sqrt(D)``, so it lands in ``[0, 1]`` for
        points inside the observed hull and stays comparable across scenarios.
        """
        usable = self.measurements()
        if not usable:
            return 1.0
        X = np.asarray([r.descriptor_vector for r in usable], dtype=float)
        query = np.asarray(vector, dtype=float)
        if query.shape[0] != X.shape[1]:
            return 1.0
        offset = X.min(axis=0)
        scale = X.max(axis=0) - offset
        scale[scale <= 0.0] = 1.0
        Xn = (X - offset) / scale
        qn = (query - offset) / scale
        distances = np.linalg.norm(Xn - qn, axis=1) / np.sqrt(X.shape[1])
        return float(np.min(distances))

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------
    @classmethod
    def from_genome_cache(
        cls,
        scenario_fingerprint: str,
        entries: Iterable[dict[str, Any]],
        descriptor_fn,
        evaluation_type: EvaluationType = "cached",
    ) -> "EvaluationKnowledgeBase":
        """Rebuild the base from ``genome_cache`` documents.

        ``descriptor_fn(chromosome_dict) -> TopologyDescriptors`` recomputes the
        descriptors deterministically, so nothing beyond the chromosome and its
        objectives needs to be persisted.  Entries without objectives (queued
        but never finished) are skipped.
        """
        kb = cls(scenario_fingerprint)
        skipped = 0
        for entry in entries:
            objectives = entry.get("objectives")
            chromosome = entry.get("chromosome")
            if not objectives or not chromosome:
                skipped += 1
                continue
            try:
                descriptors = descriptor_fn(chromosome)
            except Exception:  # pragma: no cover - defensive
                log.exception("[kb] Could not rebuild descriptors for %s.", entry.get("genome_hash"))
                skipped += 1
                continue
            kb.add(
                EvaluationRecord(
                    scenario_fingerprint=scenario_fingerprint,
                    chromosome_hash=str(entry.get("genome_hash", "")),
                    chromosome=dict(chromosome),
                    descriptors=descriptors.as_dict(),
                    descriptor_vector=tuple(float(v) for v in descriptors.vector()),
                    objectives=tuple(float(v) for v in objectives),
                    evaluation_type=evaluation_type,
                )
            )
        if skipped:
            log.info("[kb] Rebuilt from genome cache: %d records, %d skipped.", len(kb), skipped)
        return kb
