"""Objective estimators for the adaptive-simulation strategy.

The estimator never *replaces* the simulator: it only produces a cheap guess
plus an uncertainty band, which the decision policy uses to decide where the
simulation budget is worth spending.

The first implementation is deliberately the simplest thing that can work and
be audited: distance-weighted k-nearest-neighbours over normalised structural
descriptors, with the weighted standard deviation of the neighbours as the
uncertainty.  No neural network, no Gaussian process, no external dependency.
``ObjectiveEstimator`` is the seam where a richer model can be dropped in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

# Two descriptor vectors closer than this are treated as identical, so an exact
# structural match returns the stored objectives instead of a weighted blend.
EXACT_MATCH_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ObjectivePrediction:
    """Estimated objective vector plus everything needed to audit it."""

    mean: np.ndarray
    uncertainty: np.ndarray
    neighbors: tuple[int, ...]
    neighbor_distances: tuple[float, ...]
    descriptor_distance: float
    confidence: float
    exact: bool = False

    def as_dict(self) -> dict:
        return {
            "mean": [float(v) for v in self.mean],
            "uncertainty": [float(v) for v in self.uncertainty],
            "neighbors": list(self.neighbors),
            "neighbor_distances": [float(d) for d in self.neighbor_distances],
            "descriptor_distance": float(self.descriptor_distance),
            "confidence": float(self.confidence),
            "exact": bool(self.exact),
        }


@runtime_checkable
class ObjectiveEstimator(Protocol):
    """Replaceable regression backend used by the adaptive policy."""

    @property
    def n_samples(self) -> int:
        """Number of training samples currently held."""
        ...

    def fit(self, descriptors: np.ndarray, objectives: np.ndarray) -> None:
        """(Re)train on the full historical sample set."""
        ...

    def predict(self, descriptors: Sequence[float]) -> Optional[ObjectivePrediction]:
        """Predict one individual; ``None`` when the model is untrained."""
        ...


class WeightedKNNEstimator:
    """Distance-weighted k-NN regression with a weighted-deviation uncertainty.

    For the ``k`` nearest training samples with distances ``d_i``:

        w_i    = 1 / (d_i + eps)
        f_j(x) = sum_i w_i f_j(x_i) / sum_i w_i
        s_j(x) = sqrt( sum_i w_i (f_j(x_i) - f_j(x))^2 / sum_i w_i )

    Descriptors are min-max normalised with the statistics of the training set,
    so features on different scales (a relay count and a coverage ratio)
    contribute comparably to the distance.
    """

    def __init__(self, k: int = 7, epsilon: float = 1e-9) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}.")
        self.k = int(k)
        self.epsilon = float(epsilon)
        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._offset: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return 0 if self._X is None else int(self._X.shape[0])

    @property
    def n_objectives(self) -> int:
        return 0 if self._Y is None else int(self._Y.shape[1])

    # ------------------------------------------------------------------
    def fit(self, descriptors: np.ndarray, objectives: np.ndarray) -> None:
        X = np.asarray(descriptors, dtype=float)
        Y = np.asarray(objectives, dtype=float)
        if X.size == 0 or Y.size == 0:
            self._X = self._Y = self._offset = self._scale = None
            return
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(
                f"Expected 2-D arrays, got descriptors{X.shape} objectives{Y.shape}."
            )
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Sample count mismatch: {X.shape[0]} descriptors vs {Y.shape[0]} objectives."
            )

        offset = X.min(axis=0)
        scale = X.max(axis=0) - offset
        scale[scale <= 0.0] = 1.0  # constant feature contributes nothing

        self._offset = offset
        self._scale = scale
        self._X = (X - offset) / scale
        self._Y = Y

    # ------------------------------------------------------------------
    def normalize(self, descriptors: Sequence[float]) -> np.ndarray:
        """Apply the fitted min-max normalisation to one descriptor vector."""
        vec = np.asarray(descriptors, dtype=float)
        if self._offset is None or self._scale is None:
            return vec
        return (vec - self._offset) / self._scale

    def predict(self, descriptors: Sequence[float]) -> Optional[ObjectivePrediction]:
        if self._X is None or self._Y is None:
            return None

        query = self.normalize(descriptors)
        if query.shape[0] != self._X.shape[1]:
            raise ValueError(
                f"Descriptor length mismatch: got {query.shape[0]}, "
                f"model trained on {self._X.shape[1]}."
            )

        distances = np.linalg.norm(self._X - query, axis=1)
        k = min(self.k, distances.shape[0])
        order = np.argsort(distances, kind="stable")[:k]
        d_sel = distances[order]
        y_sel = self._Y[order]

        exact_mask = d_sel <= EXACT_MATCH_TOLERANCE
        if bool(np.any(exact_mask)):
            # An identical structure is known: report it verbatim rather than
            # blending it with distant neighbours.
            y_exact = y_sel[exact_mask]
            mean = y_exact.mean(axis=0)
            sigma = y_exact.std(axis=0)
            return ObjectivePrediction(
                mean=mean,
                uncertainty=sigma,
                neighbors=tuple(int(i) for i in order[exact_mask]),
                neighbor_distances=tuple(float(d) for d in d_sel[exact_mask]),
                descriptor_distance=0.0,
                confidence=1.0,
                exact=True,
            )

        weights = 1.0 / (d_sel + self.epsilon)
        total = float(weights.sum())
        mean = (weights[:, None] * y_sel).sum(axis=0) / total
        variance = (weights[:, None] * (y_sel - mean) ** 2).sum(axis=0) / total
        sigma = np.sqrt(np.maximum(variance, 0.0))
        nearest = float(d_sel[0])

        return ObjectivePrediction(
            mean=mean,
            uncertainty=sigma,
            neighbors=tuple(int(i) for i in order),
            neighbor_distances=tuple(float(d) for d in d_sel),
            descriptor_distance=nearest,
            confidence=1.0 / (1.0 + nearest),
            exact=False,
        )
