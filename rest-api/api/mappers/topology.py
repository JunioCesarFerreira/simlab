"""The infrastructure as a graph: master-node → hosts → workers, and what runs where.

One answer for the topology page, assembled from documents the caller has
already read, so it can be tested without a database. The page polls it
while experiments run, so it has to be one round of queries and not one per
worker — the per-worker ``/resources/{id}/simulation`` route is the N+1 this
replaces.

What this adds over the existing inventory routes is **what is running on
each container** — the simulation, its experiment and generation — and
**whether the master-node is alive**, which no document records directly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Optional

from pylib.db.models.infra import OCCUPIED_STATUSES
from pylib.timeutil import as_aware

#: After this long without any host being checked, the master-node is taken to
#: have stopped. Its health cycle writes ``last_check`` on every enabled host
#: every ``INFRA_CHECK_INTERVAL_SEC`` (30 s by default) — even when the host is
#: unreachable — so silence across *every* host means the cycle itself is not
#: running. Four intervals, so one slow cycle is not reported as a dead one.
MASTER_STALE_AFTER_SEC = 120

MASTER_ALIVE = "alive"
MASTER_STALE = "stale"
#: No enabled host to observe it by — a fresh install, or every host disabled.
MASTER_UNKNOWN = "unknown"

_OCCUPIED = {s.value for s in OCCUPIED_STATUSES}


def _oid(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def master_status(hosts: Iterable[dict], now: datetime) -> dict:
    """Whether the master-node's health cycle is running, judged by its writes.

    Only enabled, non-withdrawing hosts count: a disabled host is never
    checked, so its ``last_check`` is old by design and would make a healthy
    master look stopped.
    """
    observed = [
        as_aware(h["last_check"]) for h in hosts
        if h.get("enabled", True) and not h.get("withdrawn_at") and h.get("last_check")
    ]
    if not observed:
        return {"status": MASTER_UNKNOWN, "last_seen": None,
                "stale_after_sec": MASTER_STALE_AFTER_SEC}
    last_seen = max(observed)
    age = (as_aware(now) - last_seen).total_seconds()
    return {
        "status": MASTER_ALIVE if age <= MASTER_STALE_AFTER_SEC else MASTER_STALE,
        "last_seen": last_seen,
        "age_sec": max(0.0, age),
        "stale_after_sec": MASTER_STALE_AFTER_SEC,
    }


def _run(resource: dict, simulations: dict[str, dict], experiments: dict[str, dict],
         generations: dict[str, dict]) -> Optional[dict]:
    """What occupies a worker, or ``None``.

    Only an occupied worker has a run. A worker whose status is ``available``
    but still carries a stale ``current_simulation_id`` is not running it —
    the status is what the dispatcher acts on.
    """
    if resource.get("status") not in _OCCUPIED:
        return None
    sim_id = _oid(resource.get("current_simulation_id"))
    sim = simulations.get(sim_id or "", {})
    exp_id = _oid(sim.get("experiment_id"))
    gen_id = _oid(sim.get("generation_id"))
    experiment = experiments.get(exp_id or "", {})
    generation = generations.get(gen_id or "", {})
    return {
        "simulation_id": sim_id,
        "simulation_status": sim.get("status"),
        "experiment_id": exp_id,
        "experiment_name": experiment.get("name"),
        "generation_index": generation.get("index"),
        "random_seed": sim.get("random_seed"),
        "assigned_at": resource.get("reserved_at") or sim.get("assigned_at"),
        "started_at": resource.get("started_at") or sim.get("start_time"),
    }


def topology_from_documents(
    *,
    hosts: list[dict],
    schedulable: dict[str, bool],
    resources: list[dict],
    simulations: Iterable[dict],
    experiments: Iterable[dict],
    generations: Iterable[dict],
    running_experiments: Iterable[dict],
    now: datetime,
) -> dict:
    """The graph and the runs on it, from documents already read.

    ``schedulable`` comes from the inventory snapshot, the same aggregation the
    dispatcher decides on, rather than being re-derived here: a second rule
    for "can this host take work" would sooner or later disagree with the
    first.
    """
    sims = {str(s["_id"]): s for s in simulations}
    exps = {str(e["_id"]): e for e in experiments}
    gens = {str(g["_id"]): g for g in generations}

    by_host: dict[str, list[dict]] = {}
    for resource in resources:
        by_host.setdefault(_oid(resource.get("host_id")) or "", []).append(resource)

    out_hosts = []
    on_workers: dict[str, dict[str, int]] = {}
    for host in sorted(hosts, key=lambda h: (h.get("type") != "local", h.get("name", ""))):
        host_id = str(host["_id"])
        workers = []
        for resource in sorted(by_host.get(host_id, []), key=lambda r: r.get("name", "")):
            run = _run(resource, sims, exps, gens)
            if run and run["experiment_id"]:
                counts = on_workers.setdefault(run["experiment_id"], {})
                counts[host.get("name", "?")] = counts.get(host.get("name", "?"), 0) + 1
            held = resource.get("held_for") or None
            workers.append({
                "id": str(resource["_id"]),
                "name": resource.get("name", ""),
                "status": resource.get("status", ""),
                "enabled": bool(resource.get("enabled", True)),
                "held": held is not None,
                "held_kind": (held or {}).get("kind"),
                "message": resource.get("last_check_message", ""),
                "run": run,
            })
        out_hosts.append({
            "id": host_id,
            "name": host.get("name", ""),
            "type": host.get("type", "remote"),
            "status": host.get("status", ""),
            "enabled": bool(host.get("enabled", True)),
            "schedulable": bool(schedulable.get(host_id, False)),
            "withdrawing": bool(host.get("withdrawn_at")),
            "last_check": host.get("last_check"),
            "message": host.get("last_check_message", ""),
            "workers": workers,
        })

    running = []
    for experiment in running_experiments:
        exp_id = str(experiment["_id"])
        counts = on_workers.get(exp_id, {})
        running.append({
            "id": exp_id,
            "name": experiment.get("name", ""),
            "strategy": (experiment.get("parameters") or {}).get("strategy"),
            "on_workers": sum(counts.values()),
            "by_host": counts,
        })
    running.sort(key=lambda e: (-e["on_workers"], e["name"]))

    return {
        "generated_at": now,
        "master": master_status(hosts, now),
        "hosts": out_hosts,
        "running_experiments": running,
    }
