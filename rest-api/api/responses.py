"""JSON response that tolerates non-finite floats.

Objective values can legitimately be a large sentinel, but a stray ``inf`` or
``nan`` (e.g. from legacy documents written before the finite-sentinel fix, or
an individual that never produced metrics) makes Starlette's default encoder
(``allow_nan=False``) raise ``ValueError: Out of range float values are not JSON
compliant`` and return HTTP 500. This response sanitizes such values so the API
never crashes when reading an experiment:

  * ``+inf`` → ``1e12`` (kept above the frontend PENALTY_THRESHOLD of 1e8)
  * ``-inf`` → ``-1e12``
  * ``nan``  → ``None``
"""
import math
from typing import Any

from starlette.responses import JSONResponse

_POS_SENTINEL = 1e12
_NEG_SENTINEL = -1e12


def sanitize_non_finite(obj: Any) -> Any:
    """Recursively replace non-finite floats with JSON-safe values."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return _POS_SENTINEL if obj > 0 else _NEG_SENTINEL
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_non_finite(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSONResponse that sanitizes non-finite floats before encoding."""

    def render(self, content: Any) -> bytes:
        return super().render(sanitize_non_finite(content))
