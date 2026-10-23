"""Scalability benchmarks of the simulation infrastructure.

A benchmark is an experiment run by the engine's ``usl_benchmark`` strategy:
the same simulation, in waves of ``N`` copies, for several ``N``. These routes
create one from a validated body, list them, and serve the analysis — the
measurements per wave and a Universal Scalability Law fit — computed on
demand from the simulations, so it always reflects what has run so far.

Created here rather than through ``POST /experiments`` because the benchmark
block is a contract of its own: the generic experiment body is typed, and a
key it does not declare could be dropped on the way in without an error.
"""
from typing import Any, Optional

from bson import ObjectId
from bson import errors as bson_errors
from fastapi import APIRouter, Body, Depends, HTTPException, status

from pylib import scalability
from pylib.db import MongoRepository
from pylib.db.models.enums import EnumStatus
from pylib.db.models.infra import machine_of
from pylib.timeutil import utc_now
from api.dependencies import get_factory

router = APIRouter()

STRATEGY = "usl_benchmark"

#: Bounds on what a benchmark may ask for. Generous on purpose — they are
#: there to refuse a typo, not to second-guess a large pool.
MAX_LEVEL = 256
MAX_LEVELS = 32
MAX_REPETITIONS = 10
DURATION_RANGE = (10, 3600)

#: What the master-node needs to turn a run's log into metrics. Not used by
#: the benchmark's own analysis, which only reads timestamps — but a run
#: whose metrics cannot be evaluated ends in error, and a benchmark of
#: failures measures nothing. Sent by the GUI; this is the fallback.
DEFAULT_DATA_CONVERSION = {
    "node_col": "node",
    "time_col": "root_time_now",
    "metrics": [
        {"name": "latency", "kind": "mean", "column": "rtt_latency"},
        {"name": "energy", "kind": "sum_all", "column": "total_energy_mj"},
        {"name": "throughput", "kind": "sum_last_minus_first",
         "column": "server_received"},
    ],
}


def _refuse(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


def _as_int(value: Any, field: str, low: int, high: int) -> int:
    """A whole number in range, or a 422 that says which field and why.

    ``bool`` is refused although it is an ``int`` in Python: ``True`` would be
    read as 1, which is never what a client sending it meant.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(f"{field} must be a whole number, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise _refuse(f"{field} must be a whole number, got {value!r}")
    number = int(value)
    if not low <= number <= high:
        raise _refuse(f"{field} must be between {low} and {high}, got {number}")
    return number


def _levels(value: Any) -> list[int]:
    if not isinstance(value, list) or not value:
        raise _refuse("levels must be a non-empty list of concurrency levels")
    levels = sorted({_as_int(v, "levels[]", 1, MAX_LEVEL) for v in value})
    if len(levels) > MAX_LEVELS:
        raise _refuse(f"at most {MAX_LEVELS} distinct levels, got {len(levels)}")
    return levels


def _source_options(value: Any) -> dict[str, ObjectId]:
    if not isinstance(value, dict) or not value:
        raise _refuse(
            "source_repository_options must map a MAC protocol to a source "
            "repository id")
    out: dict[str, ObjectId] = {}
    for protocol, repo_id in value.items():
        try:
            out[str(protocol)] = ObjectId(str(repo_id))
        except (bson_errors.InvalidId, TypeError):
            raise _refuse(f"source_repository_options.{protocol} is not a valid id")
    return out


def _plan_summary(levels: list[int], repetitions: int) -> dict:
    return {
        "levels": levels,
        "repetitions": repetitions,
        "waves": len(levels) * repetitions,
        "simulations": sum(levels) * repetitions,
    }


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_benchmark(
    body: dict = Body(...),
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """Queue a scalability benchmark. The engine starts it like any experiment."""
    if not isinstance(body, dict):
        raise _refuse("the body must be an object")
    known = {"name", "problem", "source_repository_options", "levels",
             "repetitions", "duration", "seed", "random_seed", "chromosome",
             "data_conversion_config"}
    unknown = sorted(set(body) - known)
    if unknown:
        raise _refuse(f"unknown field(s): {', '.join(unknown)}")

    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _refuse("name is required")
    problem = body.get("problem")
    if not isinstance(problem, dict) or not problem.get("name"):
        raise _refuse("problem must be the exported problem configuration")
    chromosome = body.get("chromosome")
    if chromosome is not None and not isinstance(chromosome, dict):
        raise _refuse("chromosome must be an object when given")
    conversion = body.get("data_conversion_config") or DEFAULT_DATA_CONVERSION
    if not isinstance(conversion, dict) or not conversion.get("metrics"):
        raise _refuse("data_conversion_config must define metrics")

    levels = _levels(body.get("levels"))
    repetitions = _as_int(body.get("repetitions", 3), "repetitions", 1, MAX_REPETITIONS)
    duration = _as_int(body.get("duration", 60), "duration", *DURATION_RANGE)
    seed = _as_int(body.get("seed", 123456), "seed", 0, 2**31 - 1)
    random_seed = _as_int(body.get("random_seed", 42), "random_seed", 0, 2**31 - 1)

    benchmark: dict[str, Any] = {"levels": levels, "repetitions": repetitions}
    if chromosome:
        benchmark["chromosome"] = chromosome

    document = {
        "name": name.strip(),
        "status": EnumStatus.WAITING,
        "system_message": "",
        "created_time": utc_now(),
        "start_time": None,
        "end_time": None,
        "parameters": {
            "strategy": STRATEGY,
            "random_seed": random_seed,
            "algorithm": {},
            "simulation": {"duration": duration, "random_seeds": [seed]},
            "problem": problem,
            "objectives": [],
            "benchmark": benchmark,
        },
        "source_repository_options": _source_options(
            body.get("source_repository_options")),
        "data_conversion_config": conversion,
        "pareto_front": None,
    }
    experiment_id = factory.experiment_repo.insert(document)
    return {"id": str(experiment_id), **_plan_summary(levels, repetitions)}


def _summary(doc: dict, generations: Optional[list[dict]] = None) -> dict:
    params = doc.get("parameters") or {}
    benchmark = params.get("benchmark") or {}
    levels = [int(n) for n in benchmark.get("levels") or []]
    repetitions = int(benchmark.get("repetitions") or 0)
    plan = _plan_summary(levels, repetitions)
    workload = doc.get("benchmark_workload") or {}
    out = {
        "id": str(doc["_id"]),
        "name": doc.get("name", ""),
        "status": doc.get("status"),
        "system_message": doc.get("system_message") or "",
        "created_time": doc.get("created_time"),
        "start_time": doc.get("start_time"),
        "end_time": doc.get("end_time"),
        "problem": (params.get("problem") or {}).get("name"),
        "duration": (params.get("simulation") or {}).get("duration"),
        "workload": {
            "genome_hash": workload.get("genome_hash"),
            "source": workload.get("source"),
            "seed": workload.get("seed"),
        } if workload else None,
        **plan,
    }
    if generations is not None:
        out["waves_finished"] = sum(
            1 for g in generations
            if g.get("benchmark") and g.get("status") in (EnumStatus.DONE, EnumStatus.ERROR))
    return out


@router.get("/")
def list_benchmarks(factory: MongoRepository = Depends(get_factory)) -> list[dict]:
    """Every benchmark, newest first, with how far it has got."""
    out = []
    for doc in factory.experiment_repo.find_by_strategy(STRATEGY):
        generations = factory.generation_repo.find_by_experiment(doc["_id"])
        out.append(_summary(doc, generations))
    return out


@router.get("/{experiment_id}")
def get_benchmark(
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """The benchmark, its waves and the USL fit, as of now.

    Served while the benchmark runs as well as after: the waves measured so
    far are fitted, and the ones still running are reported as such.
    """
    try:
        oid = ObjectId(experiment_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    doc = factory.experiment_repo.get(experiment_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    if (doc.get("parameters") or {}).get("strategy") != STRATEGY:
        raise HTTPException(status_code=404, detail="This experiment is not a benchmark")

    generations = factory.generation_repo.find_by_experiment(oid)
    with factory.simulation_repo.connection.connect() as db:
        simulations = list(db["simulations"].find(
            {"experiment_id": oid},
            {"generation_id": 1, "status": 1, "assigned_at": 1, "start_time": 1,
             "end_time": 1, "assigned_host_name": 1, "assigned_resource_name": 1}))
    by_generation: dict[str, list[dict]] = {}
    for sim in simulations:
        by_generation.setdefault(str(sim.get("generation_id")), []).append(sim)

    # Which machine each host is on, as the hosts are registered now: the
    # simulations record the host's name, and a host renamed or removed since
    # is reported as a machine of its own.
    machine_of_host = {
        str(host.get("name")): machine_of(host) for host in factory.host_repo.find_all()}
    machine_repo = getattr(factory, "machine_repo", None)
    limits = machine_repo.limits() if machine_repo is not None else {}

    summary = _summary(doc, generations)
    return {
        **summary,
        "analysis": scalability.analyse(
            generations, by_generation, planned_levels=summary["levels"],
            machine_of_host=machine_of_host,
            limits=limits if isinstance(limits, dict) else {}),
    }
