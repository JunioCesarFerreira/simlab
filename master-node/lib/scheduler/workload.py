"""Which runs share one estimate of how long a worker holds a job.

Held time follows the work far more than the worker. Measured on this
deployment: 21 s for a CSMA run of 60 simulated minutes, 1437 s for a TSCH run
of 180, on workers of similar speed. One mean per worker compared the two,
and the workers that had last run long jobs looked up to 100 times slower
than the one that had just run a short one — so they were never placed on,
and never measured again.
"""
from __future__ import annotations

import re
from typing import Any, Mapping


def workload_of(sim: Mapping[str, Any]) -> str:
    """A key for the cost of ``sim``, safe to use as a MongoDB field name.

    What sets that cost, as far as a simulation document tells: the firmware,
    the simulated duration, and the size of the network to within a factor of
    two. Not finer on purpose — P2 varies its relays from one individual to
    the next, and an estimate per exact size would never gather samples.
    """
    parameters = sim.get("parameters") or {}
    elements = parameters.get("simulationElements") or {}
    nodes = sum(len(elements.get(kind) or ()) for kind in ("fixedMotes", "mobileMotes"))
    # The largest power of two not above the number of nodes.
    size = (1 << (nodes.bit_length() - 1)) if nodes else 0
    key = f"w{sim.get('source_repository_id')}_{parameters.get('duration')}m_n{size}"
    return re.sub(r"[^A-Za-z0-9_]", "-", key)
