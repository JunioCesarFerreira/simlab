"""In-process fakes that drive a full SimLab strategy without MongoDB.

The real loop is event-driven (change streams + master-node workers), which is
untestable in CI.  These fakes reproduce the *contract* the strategies rely on
— generation documents created before simulations, simulations aggregated per
individual, a generation closing only when no simulation is active — while a
deterministic evaluator plays the role of the Cooja/synthetic worker.

That is enough to run NSGA-III (baseline) and NSGA-III adaptive simulation over
the same problem, with the same seeds, and compare what they actually cost.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional

from bson import ObjectId

from pylib.db import EnumStatus


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------
class FakeExperimentRepository:
    def __init__(self) -> None:
        self.documents: dict[str, dict] = {}
        self.cancelled: set[str] = set()

    def update(self, experiment_id: str, updates: dict) -> bool:
        self.documents.setdefault(str(experiment_id), {}).update(updates)
        return True

    def update_status(self, experiment_id: str, status: str) -> None:
        self.update(experiment_id, {"status": status})

    def is_cancelled(self, experiment_id: str) -> bool:
        return str(experiment_id) in self.cancelled


class FakeGenerationRepository:
    def __init__(self, simulations: "FakeSimulationRepository") -> None:
        self.documents: dict[ObjectId, dict] = {}
        self._simulations = simulations

    # --- writes ---------------------------------------------------------
    def insert(self, generation: dict) -> ObjectId:
        self.documents[generation["_id"]] = dict(generation)
        return generation["_id"]

    def update(self, generation_id: ObjectId, updates: dict) -> bool:
        doc = self.documents.get(generation_id)
        if doc is None:
            return False
        doc.update(updates)
        return True

    def mark_done(self, generation_id: ObjectId) -> None:
        self.update(generation_id, {"status": EnumStatus.DONE})

    def mark_error(self, generation_id: ObjectId) -> None:
        self.update(generation_id, {"status": EnumStatus.ERROR})

    def mark_running(self, generation_id: ObjectId) -> None:
        self.update(generation_id, {"status": EnumStatus.RUNNING})

    # --- reads ----------------------------------------------------------
    def get(self, generation_id) -> Optional[dict]:
        if isinstance(generation_id, str):
            for oid, doc in self.documents.items():
                if str(oid) == generation_id:
                    return doc
            return None
        return self.documents.get(generation_id)

    def find_by_experiment(self, experiment_id: ObjectId) -> list[dict]:
        return sorted(
            (d for d in self.documents.values() if d["experiment_id"] == experiment_id),
            key=lambda d: d["index"],
        )

    def any_simulation_active(self, generation_id: ObjectId) -> bool:
        return any(
            sim["generation_id"] == generation_id
            and sim["status"] in (EnumStatus.WAITING, EnumStatus.RUNNING)
            for sim in self._simulations.documents.values()
        )

    def all_simulations_done(self, generation_id: ObjectId) -> bool:
        sims = [s for s in self._simulations.documents.values() if s["generation_id"] == generation_id]
        return all(s["status"] == EnumStatus.DONE for s in sims)

    def get_simulations_metrics_by_individual(
        self,
        generation_id: ObjectId,
        metrics: list[str],
        aggregator: Any = "mean",
    ) -> dict[str, dict[str, float]]:
        accumulated: dict[str, dict[str, list[float]]] = {}
        for sim in self._simulations.documents.values():
            if sim["generation_id"] != generation_id or sim["status"] != EnumStatus.DONE:
                continue
            values = sim.get("network_metrics") or {}
            bucket = accumulated.setdefault(sim["individual_id"], {})
            for name in metrics:
                if name in values:
                    bucket.setdefault(name, []).append(float(values[name]))
        return {
            ind: {m: sum(v) / len(v) for m, v in per_metric.items() if v}
            for ind, per_metric in accumulated.items()
        }

    # --- watchers (no-ops: the driver replaces the change stream) --------
    def watch_status_terminal(self, on_change: Callable[[dict], None]) -> None:
        return None

    def watch_status_waiting(self, on_change: Callable[[dict], None]) -> None:
        return None


class FakeSimulationRepository:
    def __init__(self) -> None:
        self.documents: dict[ObjectId, dict] = {}

    def insert(self, simulation: dict) -> ObjectId:
        oid = ObjectId()
        doc = dict(simulation)
        doc["_id"] = oid
        self.documents[oid] = doc
        return oid

    def find_pending_by(self, parent: str, object_id: ObjectId) -> list[dict]:
        return [
            s for s in self.documents.values()
            if s.get(parent) == object_id and s["status"] == EnumStatus.WAITING
        ]

    def mark_done(self, sim_id: ObjectId, network_metrics: dict[str, float]) -> None:
        self.documents[sim_id]["status"] = EnumStatus.DONE
        self.documents[sim_id]["network_metrics"] = network_metrics

    def watch_status_terminal(self, on_change: Callable[[dict], None]) -> None:
        return None


class FakeIndividualRepository:
    def __init__(self) -> None:
        self.documents: dict[tuple[ObjectId, str], dict] = {}

    def insert(self, individual: dict) -> ObjectId:
        key = (individual["generation_id"], individual["individual_id"])
        self.documents[key] = dict(individual)
        return ObjectId()

    def find_by_generation(self, generation_id: ObjectId) -> list[dict]:
        return [d for (g, _), d in self.documents.items() if g == generation_id]

    def update_objectives(self, individual_id: str, generation_id: ObjectId, objectives: list[float]) -> bool:
        doc = self.documents.get((generation_id, individual_id))
        if doc is None:
            return False
        doc["objectives"] = objectives
        return True

    def update_fields(self, individual_id: str, generation_id: ObjectId, fields: dict) -> bool:
        doc = self.documents.get((generation_id, individual_id))
        if doc is None:
            return False
        doc.update(fields)
        return True

    def update_topology_picture(self, individual_id: str, generation_id: ObjectId, picture_id) -> bool:
        return True


class FakeGenomeCacheRepository:
    def __init__(self) -> None:
        self.documents: dict[tuple[ObjectId, str], dict] = {}

    def insert(self, experiment_id: ObjectId, genome_hash: str, chromosome: dict) -> ObjectId:
        self.documents.setdefault(
            (experiment_id, genome_hash),
            {"genome_hash": genome_hash, "chromosome": chromosome, "objectives": None},
        )
        return ObjectId()

    def set_objectives(
        self,
        experiment_id: ObjectId,
        genome_hash: str,
        objectives: list[float],
        evaluation_source: str | None = None,
    ) -> bool:
        doc = self.documents.setdefault(
            (experiment_id, genome_hash),
            {"genome_hash": genome_hash, "chromosome": {}, "objectives": None},
        )
        doc["objectives"] = list(objectives)
        if evaluation_source is not None:
            doc["evaluation_source"] = evaluation_source
        return True

    def get_all_by_experiment(self, experiment_id: ObjectId) -> list[dict]:
        return [
            {"genome_hash": d["genome_hash"], "objectives": d["objectives"]}
            for (exp, _), d in self.documents.items() if exp == experiment_id
        ]

    def get_all_full_by_experiment(self, experiment_id: ObjectId) -> list[dict]:
        return [dict(d) for (exp, _), d in self.documents.items() if exp == experiment_id]


class FakeAdaptiveRepository:
    def __init__(self) -> None:
        self.decisions: dict[tuple[ObjectId, int, str], dict] = {}
        self.metrics: dict[tuple[ObjectId, int], dict] = {}

    def upsert_decision(self, experiment_id, scenario_fingerprint, generation_index, decision) -> None:
        key = (experiment_id, int(generation_index), decision["individual_id"])
        doc = dict(decision)
        doc["scenario_fingerprint"] = scenario_fingerprint
        doc["generation_index"] = int(generation_index)
        self.decisions[key] = doc

    def update_actual_objectives(
        self, experiment_id, generation_index, individual_id, actual_objectives, evaluation_source="simulated"
    ) -> bool:
        key = (experiment_id, int(generation_index), individual_id)
        if key not in self.decisions:
            return False
        self.decisions[key]["actual_objectives"] = actual_objectives
        self.decisions[key]["evaluation_source"] = evaluation_source
        return True

    def find_by_experiment(self, experiment_id, generation_index=None) -> list[dict]:
        return [
            d for (exp, gen, _), d in self.decisions.items()
            if exp == experiment_id and (generation_index is None or gen == int(generation_index))
        ]

    def upsert_metrics(self, experiment_id, scenario_fingerprint, generation_index, metrics) -> None:
        self.metrics[(experiment_id, int(generation_index))] = dict(metrics)

    def find_metrics(self, experiment_id) -> list[dict]:
        return [m for (exp, _), m in self.metrics.items() if exp == experiment_id]


class FakeMongo:
    """Duck-typed stand-in for ``pylib.db.MongoRepository``."""

    def __init__(self) -> None:
        self.simulation_repo = FakeSimulationRepository()
        self.generation_repo = FakeGenerationRepository(self.simulation_repo)
        self.individual_repo = FakeIndividualRepository()
        self.genome_cache_repo = FakeGenomeCacheRepository()
        self.experiment_repo = FakeExperimentRepository()
        self.adaptive_repo = FakeAdaptiveRepository()
        self.fs_handler = None


# ---------------------------------------------------------------------------
# Deterministic evaluator (plays the master-node worker)
# ---------------------------------------------------------------------------
class TopologyEvaluator:
    """Closed-form, deterministic surrogate of a Cooja run for P2.

    The metrics are smooth functions of the deployed topology, which is exactly
    the regime the heuristic targets: structurally similar solutions really do
    behave similarly, so an estimate carries information — while every value
    still has to be "measured" through a Simulation document.
    """

    def __init__(self, sink: tuple[float, float], noise: float = 0.02) -> None:
        self.sink = sink
        self.noise = noise
        self.calls: int = 0

    def metrics(self, config: dict) -> dict[str, float]:
        """Metrics of one deployment, as a function of its topology.

        The three objectives are in genuine conflict, the way they are in a
        real WSN: more relays cost energy but shorten routes and raise
        capacity, while a wider network extent hurts both delay and goodput.
        Everything is a smooth function of quantities the descriptors also
        capture (relay count, distance to the sink), so a structural estimate
        carries real information — which is the regime the heuristic targets.
        """
        self.calls += 1
        elements = config.get("simulationElements") or {}
        relays = [m for m in elements.get("fixedMotes", []) if m.get("name") != "sink"]
        n = len(relays)
        seed = int(config.get("randomSeed", 0))

        distances = [
            math.hypot(m["position"][0] - self.sink[0], m["position"][1] - self.sink[1])
            for m in relays
        ]
        # Mean (not max) distance: the extent of a candidate grid is nearly
        # constant, so a max-based metric would make every deployment with a
        # different relay count trivially non-dominated.
        spread = sum(distances) / n if n else 0.0

        # Latency and goodput are *not* monotone in the relay count: extra
        # relays shorten routes up to a point, past which contention and
        # forwarding overhead dominate. That inflection is what creates a
        # bounded Pareto front with genuinely dominated regions, exactly like a
        # real WSN — and it is what makes an "is this worth simulating?"
        # question meaningful at all.
        jitter = self.noise * math.sin(seed * 0.37 + n * 0.11)
        energy = 5.0 + 2.0 * n + 0.01 * n * spread / 10.0 + jitter        # minimise
        latency = 8.0 + 120.0 / (1.0 + n) + 0.20 * n + 0.06 * spread + jitter   # minimise
        throughput = 100.0 * n / (n + 8.0) - 0.60 * n - 0.15 * spread - jitter  # maximise
        return {"latency": latency, "energy": energy, "throughput": throughput}

    def complete_generation(self, mongo: FakeMongo, generation_id: ObjectId) -> int:
        """Evaluate every WAITING simulation of ``generation_id``."""
        pending = mongo.simulation_repo.find_pending_by("generation_id", generation_id)
        for sim in pending:
            mongo.simulation_repo.mark_done(sim["_id"], self.metrics(sim["parameters"]))
        if pending and not mongo.generation_repo.any_simulation_active(generation_id):
            mongo.generation_repo.mark_done(generation_id)
        return len(pending)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def harness_strategy(cls, experiment: dict, mongo: FakeMongo):
    """Instantiate ``cls`` with threads, watchers and plotting disabled."""

    class _Harness(cls):  # type: ignore[misc, valid-type]
        def _start_watcher(self) -> None:
            return None

        def _start_generation_poll(self) -> None:
            return None

        def _fire_generation_done(self, gen_oid) -> None:
            return None  # the driver loop calls the handler explicitly

        def _upload_topology_async(self, *args, **kwargs) -> None:
            return None

    _Harness.__name__ = f"Harnessed{cls.__name__}"
    return _Harness(experiment, mongo)


def run_to_completion(strategy, mongo: FakeMongo, evaluator: TopologyEvaluator, max_steps: int = 400) -> int:
    """Drive a strategy from ``start()`` to experiment completion.

    Each step evaluates whatever the current generation queued and then fires
    the generation-done handler, which is precisely the sequence the real
    change stream produces.
    """
    strategy.start()
    steps = 0
    stalled = 0
    while steps < max_steps and not strategy._stop_flag:
        steps += 1
        gen_oid = strategy._generation_id
        if gen_oid is None:
            break
        before = (gen_oid, evaluator.calls)
        evaluator.complete_generation(mongo, gen_oid)
        with strategy._lock:
            strategy._handle_generation_done(gen_oid)
        after = (strategy._generation_id, evaluator.calls)
        stalled = stalled + 1 if before == after else 0
        if stalled >= 3:
            break
    return steps
