"""From the simulations of a scalability benchmark to a USL curve.

A benchmark runs the **same** simulation — one chromosome, one seed — in
waves of ``N`` copies at a time, for several values of ``N``. Each wave is one
generation, tagged with ``benchmark: {level, repetition, wave}`` by the
engine's ``usl_benchmark`` strategy. This module turns those documents into
measurements and fits :mod:`pylib.usl` to them.

Two throughputs are measured per wave, because they answer two questions:

* **Batch** — ``X = N / makespan``, from the first placement of the wave to
  its last result. What a batch of ``N`` actually costs: it includes the
  serialised dispatch, the transfers, and the slowest container of the wave.
  This is the default, because the question a benchmark exists for is how
  long a batch takes.
* **Per simulation** — ``X = N / R̄``, Little's law for a closed system with
  ``N`` jobs and no think time, where ``R̄`` is the mean time a container was
  held. Insensitive to a single straggler, so it isolates how much *each*
  simulation slows down as concurrency grows — contention proper.

They agree on a uniform pool. On a pool whose hosts differ in speed they do
not, and the gap between them is itself a measurement: it is the cost of the
slowest machine in the wave.

Only a wave whose every simulation produced a result is measured. A wave with
a failure ran fewer than ``N`` simulations to completion, so ``N`` is not the
concurrency it measured.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Optional

from pylib import usl
from pylib.db.models.enums import EnumStatus
from pylib.timeutil import as_aware

#: The two throughput definitions, by the name the API and the GUI use.
METRIC_BATCH = "batch"
METRIC_PER_SIMULATION = "per_simulation"
METRICS = (METRIC_BATCH, METRIC_PER_SIMULATION)

#: Throughput unit. Simulations per minute reads naturally for runs that
#: take one to a few minutes each; per second would be all decimals.
SECONDS_PER_UNIT = 60.0


def _seconds(later: Optional[datetime], earlier: Optional[datetime]) -> Optional[float]:
    if later is None or earlier is None:
        return None
    try:
        return (as_aware(later) - as_aware(earlier)).total_seconds()
    except TypeError:
        return None


def measure_wave(generation: dict, simulations: Iterable[dict]) -> dict:
    """One wave's measurement, valid or not, with the reason when not.

    ``held`` runs from ``assigned_at`` — the reservation — to ``end_time``.
    The simulation's own ``start_time`` is written *after* the inputs are
    uploaded, so it leaves out exactly the part of the cost that differs most
    between a local container and a remote one.
    """
    tag = generation.get("benchmark") or {}
    level = int(tag.get("level") or 0)
    runs = list(simulations)
    done = [s for s in runs if s.get("status") == EnumStatus.DONE]
    out: dict[str, Any] = {
        "generation_id": str(generation.get("_id")),
        "wave": tag.get("wave"),
        "level": level,
        "repetition": tag.get("repetition"),
        "status": generation.get("status"),
        "simulations": len(runs),
        "done": len(done),
        "failed": sum(1 for s in runs if s.get("status") in (
            EnumStatus.ERROR, EnumStatus.CANCELLED)),
        "hosts": dict(Counter(
            str(s.get("assigned_host_name") or "?") for s in runs)),
        "makespan_sec": None,
        "mean_held_sec": None,
        "throughput": {metric: None for metric in METRICS},
        "valid": False,
        "reason": "",
    }

    if level < 1:
        out["reason"] = "not a benchmark wave"
        return out
    if len(runs) < level:
        out["reason"] = f"only {len(runs)} of {level} simulations exist yet"
        return out
    if any(s.get("status") in (EnumStatus.WAITING, EnumStatus.RUNNING) for s in runs):
        out["reason"] = "still running"
        return out
    if len(done) < level:
        out["reason"] = (
            f"{level - len(done)} of {level} simulations did not produce a "
            "result, so this wave did not run at concurrency "
            f"{level}")
        return out

    held = [_seconds(s.get("end_time"), s.get("assigned_at")) for s in done]
    if any(h is None or h <= 0 for h in held):
        out["reason"] = "a simulation is missing its placement or end time"
        return out
    first = min(as_aware(s["assigned_at"]) for s in done)
    last = max(as_aware(s["end_time"]) for s in done)
    makespan = (last - first).total_seconds()
    if makespan <= 0:
        out["reason"] = "the wave has no measurable duration"
        return out

    mean_held = sum(held) / len(held)
    out.update({
        "makespan_sec": makespan,
        "mean_held_sec": mean_held,
        "throughput": {
            METRIC_BATCH: SECONDS_PER_UNIT * level / makespan,
            METRIC_PER_SIMULATION: SECONDS_PER_UNIT * level / mean_held,
        },
        "valid": True,
    })
    return out


def summarize_levels(waves: list[dict], metric: str) -> list[dict]:
    """Mean and spread per concurrency level, over the valid repetitions."""
    by_level: dict[int, list[float]] = {}
    for wave in waves:
        value = wave["throughput"].get(metric)
        if wave["valid"] and value:
            by_level.setdefault(wave["level"], []).append(float(value))
    rows = []
    for level in sorted(by_level):
        values = by_level[level]
        mean = sum(values) / len(values)
        spread = (
            (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
            if len(values) > 1 else 0.0)
        rows.append({"level": level, "mean": mean, "std": spread,
                     "count": len(values), "min": min(values), "max": max(values)})
    return rows


def analyse(
    generations: list[dict],
    simulations_by_generation: dict[str, list[dict]],
    *,
    planned_levels: Optional[list[int]] = None,
) -> dict:
    """Measurements and a USL fit for each throughput definition."""
    waves = []
    for generation in sorted(
            generations,
            key=lambda g: (g.get("benchmark") or {}).get("wave", g.get("index", 0))):
        if not generation.get("benchmark"):
            continue
        waves.append(measure_wave(
            generation, simulations_by_generation.get(str(generation["_id"]), [])))

    fits: dict[str, Any] = {}
    for metric in METRICS:
        points = [(w["level"], w["throughput"][metric])
                  for w in waves if w["valid"] and w["throughput"][metric]]
        fitted = usl.fit([p[0] for p in points], [p[1] for p in points])
        tested = sorted({w["level"] for w in waves}) or (planned_levels or [1])
        fits[metric] = {
            **fitted.to_dict(),
            "levels": summarize_levels(waves, metric),
            "curve": usl.curve(fitted, usl.display_range(tested, fitted)),
            # Whether the peak sits inside what was measured. Outside it the
            # model is extrapolating, and the chart has to say so.
            "peak_within_tested": (
                fitted.n_peak is not None and tested
                and fitted.n_peak <= max(tested)),
        }

    return {
        "unit": "simulations per minute",
        "metrics": list(METRICS),
        "default_metric": METRIC_BATCH,
        "planned_levels": planned_levels or [],
        "waves": waves,
        "valid_waves": sum(1 for w in waves if w["valid"]),
        "fits": fits,
    }
