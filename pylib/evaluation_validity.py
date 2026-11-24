"""Whether a stored objective vector is a measurement.

Every reader of SimLab results used to answer this by magnitude: an objective
at or above ``1e8`` was taken for a penalty and dropped. That is a heuristic
with two failure modes. A legitimate value can cross it — energy summed over a
long run already reaches ``1.8e7`` — and be dropped in silence; and a
placeholder below it, or a NaN, passes.

The engine stores the evidence next to every vector (the ``completeness``
record on an individual, the ``evaluation`` summary on a Pareto item), so the
question can be answered from what happened instead of from what the number
looks like. The magnitude rule survives only for documents written before the
evidence existed.

No dependencies on purpose: the engine, the REST API and the analysis scripts
all import it. ``pareto-analysis`` keeps a standalone copy, held to this one
by a parity test.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

#: Rejected before simulating: the individual violates a constraint and its
#: vector is a penalty.
STATE_INFEASIBLE = "infeasible"

#: Magnitude marker for legacy documents with no evidence record.
LEGACY_PENALTY_THRESHOLD = 1.0e8


def is_valid_evaluation(record: Optional[Mapping[str, Any]]) -> bool:
    """Whether the evidence record says the vector is a measurement.

    Excluded: a constraint penalty (``state: infeasible``), a vector the
    strategy refused (``refused``: a metric missing or non-finite), and one
    the replica policy did not use (``objective_used: false``). A Pareto-item
    summary carries the verdict directly as ``valid``. A missing record says
    nothing against the vector.
    """
    if not record:
        return True
    if "valid" in record:
        return bool(record["valid"])
    if record.get("state") == STATE_INFEASIBLE:
        return False
    if record.get("refused"):
        return False
    return bool(record.get("objective_used", True))


def has_evidence(record: Any) -> bool:
    """Whether ``record`` is an evidence record a verdict can be read from."""
    return isinstance(record, Mapping) and bool(record)


def all_finite(objectives: Sequence[Any]) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in objectives)
    except (TypeError, ValueError):
        return False


def is_excluded(
    objectives: Sequence[Any],
    record: Optional[Mapping[str, Any]] = None,
    *,
    legacy_threshold: float = LEGACY_PENALTY_THRESHOLD,
) -> bool:
    """Whether a stored vector must stay out of fronts and indicators.

    Non-finite vectors are always excluded. With an evidence record the
    verdict is the record's; without one (documents written before the record
    existed) the legacy magnitude rule applies.
    """
    if not all_finite(objectives):
        return True
    if has_evidence(record):
        return not is_valid_evaluation(record)
    return any(abs(float(v)) >= legacy_threshold for v in objectives)
