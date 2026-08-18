"""Bookkeeping of how much simulation cost the heuristic actually saved.

Every number here exists so the thesis can compare *NSGA-III traditional*
against *NSGA-III adaptive simulation* on cost (simulations, wall-clock) and on
quality (front recall/precision, HV/IGD computed downstream from the stored
Pareto fronts).

The baseline is defined operationally: ``baseline_simulations`` counts the
individuals a plain NSGA-III run would have simulated in the same generation —
i.e. every new genome that was neither an exact cache hit nor rejected by the
problem's hard-constraint penalty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(slots=True)
class GenerationAdaptiveMetrics:
    """Per-generation accounting of the adaptive evaluation."""

    generation: int
    generated_individuals: int = 0
    unique_individuals: int = 0
    exact_cache_hits: int = 0
    penalized_individuals: int = 0
    estimated_only: int = 0
    initial_simulations: int = 0
    promotions: int = 0
    audit_simulations: int = 0
    total_actual_simulations: int = 0
    baseline_simulations: int = 0
    avoided_simulations: int = 0
    simulation_reduction_ratio: float = 0.0
    prediction_mae: Optional[float] = None
    prediction_rmse: Optional[float] = None
    prediction_error_per_objective: Optional[list[float]] = None
    false_skip_rate: Optional[float] = None
    mean_uncertainty: Optional[float] = None
    mean_novelty: Optional[float] = None
    simulation_documents: int = 0

    def finalize(self) -> "GenerationAdaptiveMetrics":
        """Derive the aggregate ratios once the generation is complete."""
        self.total_actual_simulations = self.initial_simulations + self.promotions
        self.avoided_simulations = max(0, self.baseline_simulations - self.total_actual_simulations)
        self.simulation_reduction_ratio = (
            1.0 - (self.total_actual_simulations / self.baseline_simulations)
            if self.baseline_simulations > 0
            else 0.0
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PredictionSample:
    """One (prediction, ground truth) pair produced by a promotion or audit."""

    individual_id: str
    predicted: tuple[float, ...]
    actual: tuple[float, ...]
    was_skipped: bool = False        # the policy wanted to skip this individual
    became_relevant: bool = False    # ... but the real objectives proved it useful


class AdaptiveMetricsTracker:
    """Accumulates per-generation metrics and the experiment-level totals."""

    def __init__(self) -> None:
        self.per_generation: list[GenerationAdaptiveMetrics] = []
        self._samples: list[PredictionSample] = []

    # ------------------------------------------------------------------
    def start_generation(self, generation: int) -> GenerationAdaptiveMetrics:
        metrics = GenerationAdaptiveMetrics(generation=generation)
        self.per_generation.append(metrics)
        return metrics

    def current(self) -> Optional[GenerationAdaptiveMetrics]:
        return self.per_generation[-1] if self.per_generation else None

    def add_prediction_sample(self, sample: PredictionSample) -> None:
        self._samples.append(sample)

    # ------------------------------------------------------------------
    def score_generation(
        self,
        metrics: GenerationAdaptiveMetrics,
        samples: Sequence[PredictionSample],
        uncertainties: Sequence[float],
        novelties: Sequence[float],
    ) -> GenerationAdaptiveMetrics:
        """Attach estimator-quality statistics to a finished generation."""
        if samples:
            predicted = np.asarray([s.predicted for s in samples], dtype=float)
            actual = np.asarray([s.actual for s in samples], dtype=float)
            error = predicted - actual
            metrics.prediction_mae = float(np.mean(np.abs(error)))
            metrics.prediction_rmse = float(np.sqrt(np.mean(error**2)))
            metrics.prediction_error_per_objective = [
                float(v) for v in np.mean(np.abs(error), axis=0)
            ]
            skipped = [s for s in samples if s.was_skipped]
            if skipped:
                metrics.false_skip_rate = sum(
                    1 for s in skipped if s.became_relevant
                ) / len(skipped)
        if uncertainties:
            metrics.mean_uncertainty = float(np.mean(uncertainties))
        if novelties:
            metrics.mean_novelty = float(np.mean(novelties))
        return metrics.finalize()

    # ------------------------------------------------------------------
    def experiment_summary(self) -> dict[str, Any]:
        """Cumulative view written to the experiment document on completion."""
        totals = {
            "generations": len(self.per_generation),
            "generated_individuals": sum(m.generated_individuals for m in self.per_generation),
            "exact_cache_hits": sum(m.exact_cache_hits for m in self.per_generation),
            "penalized_individuals": sum(m.penalized_individuals for m in self.per_generation),
            "estimated_only": sum(m.estimated_only for m in self.per_generation),
            "initial_simulations": sum(m.initial_simulations for m in self.per_generation),
            "promotions": sum(m.promotions for m in self.per_generation),
            "audit_simulations": sum(m.audit_simulations for m in self.per_generation),
            "total_actual_simulations": sum(m.total_actual_simulations for m in self.per_generation),
            "baseline_simulations": sum(m.baseline_simulations for m in self.per_generation),
            "simulation_documents": sum(m.simulation_documents for m in self.per_generation),
        }
        totals["avoided_simulations"] = max(
            0, totals["baseline_simulations"] - totals["total_actual_simulations"]
        )
        totals["simulation_reduction_ratio"] = (
            1.0 - (totals["total_actual_simulations"] / totals["baseline_simulations"])
            if totals["baseline_simulations"] > 0
            else 0.0
        )
        if self._samples:
            predicted = np.asarray([s.predicted for s in self._samples], dtype=float)
            actual = np.asarray([s.actual for s in self._samples], dtype=float)
            error = predicted - actual
            totals["prediction_mae"] = float(np.mean(np.abs(error)))
            totals["prediction_rmse"] = float(np.sqrt(np.mean(error**2)))
            totals["prediction_error_per_objective"] = [
                float(v) for v in np.mean(np.abs(error), axis=0)
            ]
            skipped = [s for s in self._samples if s.was_skipped]
            if skipped:
                totals["false_skip_rate"] = sum(
                    1 for s in skipped if s.became_relevant
                ) / len(skipped)
        totals["per_generation"] = [m.to_dict() for m in self.per_generation]
        return totals
