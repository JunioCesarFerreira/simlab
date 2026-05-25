"""
DTLZ2 (Deb et al., 2002) — scalable multi-objective benchmark.

The same definition is used by ``master-node/lib/synthetic_data.py`` in the
SimLab dummy backend, kept here as a self-contained reference so the
external benchmark does not depend on the running platform.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def n_variables(M: int, k: int) -> int:
    """Number of decision variables for DTLZ2 with M objectives and parameter k."""
    return M - 1 + k


def evaluate(x: Sequence[float], M: int) -> list[float]:
    """
    DTLZ2 in minimisation form.
    Domain: x_i ∈ [0, 1].  Number of decision variables n = M - 1 + k, where
    k controls the size of the "tail" (Deb et al. recommend k ≈ 10 but we
    parameterise it explicitly).
    """
    n = len(x)
    k = max(1, n - (M - 1))
    tail = x[n - k:]
    g = sum((xi - 0.5) ** 2 for xi in tail)
    f: list[float] = []
    for m in range(M):
        val = 1.0 + g
        for i in range(0, M - 1 - m):
            val *= math.cos(0.5 * math.pi * x[i])
        if m > 0:
            val *= math.sin(0.5 * math.pi * x[M - 1 - m])
        f.append(float(val))
    return f


def evaluate_batch(X: np.ndarray, M: int) -> np.ndarray:
    """Vectorised evaluation for a (pop, n_var) matrix."""
    return np.asarray([evaluate(row.tolist(), M) for row in X], dtype=float)
