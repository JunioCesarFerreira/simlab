"""What a mobile mote's path is, as both the coverage sampling and the Cooja
positions file must read it.

A path is a list of parametric segments ``(x(t), y(t))``, ``t`` in [0, 1].
``is_closed`` means the mote goes back from the end of the path to its start;
when the segments do not already end where they begin (an ellipse does, a
polyline does not), that return is a straight segment. A round trip goes back
along the path itself, so it needs none.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Segment = tuple[str, str]

#: Below this distance (in the problem's units) a path counts as already closed.
CLOSED_TOLERANCE = 1e-6


def point_at(segment: Sequence[str], t: float) -> tuple[float, float]:
    x_expr, y_expr = segment
    return (float(eval(x_expr, {"np": np, "t": t})),  # noqa: S307 - problem expressions
            float(eval(y_expr, {"np": np, "t": t})))


def _number(value: float) -> str:
    return f"{0.0 if abs(value) < 1e-12 else value:.10g}"


def closing_segment(segments: Sequence[Sequence[str]]) -> Segment | None:
    """The straight segment from the end of the path back to its start, or
    ``None`` when the path already ends where it starts."""
    if not segments:
        return None
    x0, y0 = point_at(segments[0], 0.0)
    x1, y1 = point_at(segments[-1], 1.0)
    if math.hypot(x1 - x0, y1 - y0) <= CLOSED_TOLERANCE:
        return None
    return (f"{_number(x1)} + {_number(x0 - x1)} * t", f"{_number(y1)} + {_number(y0 - y1)} * t")


def path_segments(segments: Sequence[Sequence[str]], is_closed: bool, is_round_trip: bool) -> list[Segment]:
    """The segments a mote travels in one pass of its path."""
    out = [tuple(s) for s in segments]
    if is_closed and not is_round_trip:
        closing = closing_segment(out)
        if closing is not None:
            out.append(closing)
    return out
