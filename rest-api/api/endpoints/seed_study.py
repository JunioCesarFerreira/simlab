"""Seed studies: one simulation per seed, and how many seeds an instance needs.

A study is an experiment run by the engine's ``seed_study`` strategy. These
routes create one from a validated body, list them, and serve the analysis of
:mod:`pylib.seed_study`, computed on demand from the simulations: a running
study can be read as it fills in, and changing a target needs no new run.

Created here rather than through ``POST /experiments`` for the reason written
in :mod:`api.endpoints.scalability`: the generic experiment body is typed, and a
key it does not declare could be dropped on the way in without an error.
"""
import csv
import io
import math
import secrets
from typing import Any, Optional

import numpy as np
from bson import ObjectId
from bson import errors as bson_errors
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, status

from pylib import seed_study
from pylib.db import MongoRepository
from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.generation import latest_per_seed
from pylib.seed_aggregation import objective_aggregators
from pylib.timeutil import utc_now
from api.dependencies import get_factory
from api.endpoints.scalability import (
    DEFAULT_DATA_CONVERSION,
    DURATION_RANGE,
    _as_int,
    _refuse,
    _source_options,
)
from api.metrics_cache import MetricsCache

router = APIRouter()

STRATEGY = "seed_study"
CONTROL_ROLE = "control"
MAX_CONTROL_REPLICATES = 5
#: Read from each run besides ``network_metrics``. ``convergence_time_ms`` is
#: left out: with mobile nodes it reports the last parent switch of the run.
PARENT_SWITCHES = "parent_switches"
MAX_LISTED_HOLES = 200

#: One entry per study, at most a few per set of targets while someone explores.
_cache = MetricsCache(max_entries=64, max_bytes=64 * 1024 * 1024)


def _as_rate(value: Any, field: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(f"{field} must be a number, got {value!r}")
    if not low <= float(value) <= high:
        raise _refuse(f"{field} must be between {low} and {high}, got {value}")
    return float(value)


def _targets(value: Any) -> dict[str, float]:
    if value is None:
        return dict(seed_study.DEFAULT_TARGETS)
    if not isinstance(value, dict):
        raise _refuse("targets must be an object")
    unknown = sorted(set(value) - set(seed_study.DEFAULT_TARGETS))
    if unknown:
        raise _refuse(f"unknown target(s): {', '.join(unknown)}")
    merged = {**seed_study.DEFAULT_TARGETS, **value}
    return {
        "confidence": _as_rate(merged["confidence"], "targets.confidence", 0.8, 0.999),
        "relative_error": _as_rate(merged["relative_error"], "targets.relative_error", 0.001, 1.0),
        "reliability": _as_rate(merged["reliability"], "targets.reliability", 0.5, 0.99),
    }


def _seeds(value: Any) -> tuple[list[int], dict[str, Any]]:
    """The list of the study, and how it came to be."""
    if not isinstance(value, dict):
        raise _refuse("seeds must be an object with a mode")
    mode = value.get("mode")
    if mode == "generated":
        count = _as_int(value.get("count"), "seeds.count",
                        seed_study.MIN_SEEDS, seed_study.MAX_SEEDS)
        master = value.get("master_seed")
        master_seed = (secrets.randbelow(2**31 - 1) if master is None
                       else _as_int(master, "seeds.master_seed", 0, 2**31 - 1))
        origin = {"mode": mode, "count": count, "master_seed": master_seed,
                  "guard": seed_study.DEFAULT_GUARD}
        return seed_study.generate_seeds(count, master_seed), origin
    if mode == "imported":
        try:
            seeds = seed_study.validate_seeds(value.get("values"))
        except ValueError as error:
            raise _refuse(f"seeds.values: {error}")
        return seeds, {"mode": mode, "count": len(seeds), "file_name": value.get("file_name")}
    raise _refuse("seeds.mode must be 'generated' or 'imported'")


def _panel(panel: Any, chromosomes: Any) -> tuple[dict[str, Any], Optional[list[dict]]]:
    if not isinstance(panel, dict):
        raise _refuse("panel must be an object with a source")
    source = panel.get("source")
    seed = panel.get("random_seed")
    # Drawn when not given: the same default would make two studies repeat
    # the same panel.
    random_seed = (secrets.randbelow(2**31 - 1) if seed is None
                   else _as_int(seed, "panel.random_seed", 0, 2**31 - 1))
    if source == "generated":
        if chromosomes:
            raise _refuse("chromosomes are only taken with panel.source 'provided'")
        size = _as_int(panel.get("size"), "panel.size", 1, seed_study.MAX_CHROMOSOMES)
        return {"source": source, "size": size, "random_seed": random_seed}, None
    if source == "provided":
        if not isinstance(chromosomes, list) or not chromosomes or not all(
                isinstance(c, dict) for c in chromosomes):
            raise _refuse("chromosomes must be a non-empty list of objects")
        if len(chromosomes) > seed_study.MAX_CHROMOSOMES:
            raise _refuse(f"at most {seed_study.MAX_CHROMOSOMES} chromosomes")
        return ({"source": source, "size": len(chromosomes), "random_seed": random_seed},
                chromosomes)
    raise _refuse("panel.source must be 'generated' or 'provided'")


def _objectives(value: Any, conversion: dict, synthetic: bool) -> list[dict]:
    if not isinstance(value, list) or not value or not all(isinstance(o, dict) for o in value):
        raise _refuse("objectives must be a non-empty list")
    produced = {m.get("name") for m in conversion.get("metrics") or []}
    objectives: list[dict] = []
    for index, objective in enumerate(value):
        name = objective.get("metric_name")
        if not isinstance(name, str) or not name:
            raise _refuse(f"objectives[{index}].metric_name is required")
        if objective.get("goal") not in ("min", "max"):
            raise _refuse(f"objectives[{index}].goal must be 'min' or 'max'")
        # A synthetic run writes the objectives by name and reads no log.
        if not synthetic and name not in produced:
            raise _refuse(f"objectives[{index}]: no metric '{name}' in data_conversion_config")
        objectives.append(dict(objective))
    try:
        objective_aggregators(objectives)
    except ValueError as error:
        raise _refuse(str(error))
    return objectives


def _plan(seeds: int, chromosomes: int, replicates: int) -> dict[str, int]:
    control = chromosomes * replicates
    return {"seeds": seeds, "chromosomes": chromosomes, "control": control,
            "simulations": seeds * chromosomes + control}


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_study(
    body: dict = Body(...),
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """Queue a seed study. The engine starts it like any experiment."""
    if not isinstance(body, dict):
        raise _refuse("the body must be an object")
    known = {"name", "problem", "source_repository_options", "duration", "seeds",
             "panel", "chromosomes", "objectives", "targets", "control_replicates",
             "data_conversion_config", "synthetic"}
    unknown = sorted(set(body) - known)
    if unknown:
        raise _refuse(f"unknown field(s): {', '.join(unknown)}")

    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _refuse("name is required")
    problem = body.get("problem")
    if not isinstance(problem, dict) or not problem.get("name"):
        raise _refuse("problem must be the exported problem configuration")
    synthetic = body.get("synthetic")
    if synthetic is not None and (not isinstance(synthetic, dict)
                                  or not synthetic.get("enabled")):
        raise _refuse("synthetic must be an object with enabled set")
    conversion = body.get("data_conversion_config") or DEFAULT_DATA_CONVERSION
    if not isinstance(conversion, dict) or not conversion.get("metrics"):
        raise _refuse("data_conversion_config must define metrics")

    # One protocol per study: on a mixed panel the difference between protocols
    # reads as a difference between solutions. Measured on a csma+tsch
    # experiment, it explains 82% to 95% of the variation of the objectives,
    # which would make m* far too small.
    firmware = body.get("source_repository_options") or {}
    if synthetic:
        options = _source_options(firmware) if firmware else {}
    else:
        options = _source_options(firmware)
        if len(options) != 1:
            raise _refuse(
                "source_repository_options must name exactly one firmware: a panel "
                "that mixes protocols would count the protocol as a difference "
                "between solutions. Run one study per protocol.")

    duration = _as_int(body.get("duration", 180), "duration", *DURATION_RANGE)
    seeds, origin = _seeds(body.get("seeds"))
    panel, chromosomes = _panel(body.get("panel"), body.get("chromosomes"))
    replicates = _as_int(body.get("control_replicates", 2), "control_replicates",
                         0, MAX_CONTROL_REPLICATES)
    if panel["size"] * len(seeds) > seed_study.MAX_SIMULATIONS:
        raise _refuse(
            f"{panel['size']} chromosomes × {len(seeds)} seeds is over the limit of "
            f"{seed_study.MAX_SIMULATIONS} simulations")
    objectives = _objectives(body.get("objectives"), conversion, bool(synthetic))
    targets = _targets(body.get("targets"))

    simulation: dict[str, Any] = {
        "duration": duration,
        "random_seeds": seeds,
        # A failed seed is repeated once. If it fails again, the study reports
        # what it has and what is missing instead of refusing the chromosome.
        "replica_policy": "repeat",
        "max_replica_retries": 1,
        "incomplete_objective": "accept",
    }
    if synthetic:
        simulation["synthetic"] = synthetic

    document = {
        "name": name.strip(),
        "status": EnumStatus.WAITING,
        "system_message": "",
        "created_time": utc_now(),
        "start_time": None,
        "end_time": None,
        "parameters": {
            "strategy": STRATEGY,
            "random_seed": panel["random_seed"],
            "algorithm": {},
            "simulation": simulation,
            "problem": problem,
            "objectives": objectives,
            "seed_study": {
                "seeds": origin,
                "panel": panel,
                "chromosomes": chromosomes,
                "control_replicates": replicates,
                "targets": targets,
            },
        },
        "source_repository_options": options,
        "data_conversion_config": conversion,
        "pareto_front": None,
    }
    experiment_id = factory.experiment_repo.insert(document)
    return {
        "id": str(experiment_id),
        **_plan(len(seeds), panel["size"], replicates),
        "seed_diagnostics": seed_study.diagnose_seeds(seeds),
    }


def _summary(doc: dict, finished: Optional[int] = None) -> dict:
    params = doc.get("parameters") or {}
    study = params.get("seed_study") or {}
    simulation = params.get("simulation") or {}
    panel = study.get("panel") or {}
    recorded = doc.get("seed_study_panel") or {}
    plan = _plan(len(simulation.get("random_seeds") or []),
                 int(recorded.get("size") or panel.get("size") or 0),
                 int(study.get("control_replicates") or 0))
    out = {
        "id": str(doc["_id"]),
        "name": doc.get("name", ""),
        "status": doc.get("status"),
        "system_message": doc.get("system_message") or "",
        "created_time": doc.get("created_time"),
        "start_time": doc.get("start_time"),
        "end_time": doc.get("end_time"),
        "problem": (params.get("problem") or {}).get("name"),
        "duration": simulation.get("duration"),
        "panel_source": recorded.get("source") or panel.get("source"),
        "objectives": params.get("objectives") or [],
        "targets": study.get("targets") or seed_study.DEFAULT_TARGETS,
        **plan,
    }
    if finished is not None:
        out["finished"] = min(finished, plan["simulations"])
    return out


@router.get("/")
def list_studies(factory: MongoRepository = Depends(get_factory)) -> list[dict]:
    """Every study, newest first, with how far it has got."""
    docs = factory.experiment_repo.find_by_strategy(STRATEGY)
    with factory.simulation_repo.connection.connect() as db:
        rows = db["simulations"].aggregate([
            {"$match": {"experiment_id": {"$in": [d["_id"] for d in docs]},
                        "status": EnumStatus.DONE}},
            {"$group": {"_id": "$experiment_id", "finished": {"$sum": 1}}},
        ])
        finished = {row["_id"]: row["finished"] for row in rows}
    return [_summary(doc, finished.get(doc["_id"], 0)) for doc in docs]


def _study_or_error(experiment_id: str, factory: MongoRepository) -> tuple[ObjectId, dict]:
    try:
        oid = ObjectId(experiment_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    doc = factory.experiment_repo.get(experiment_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Seed study not found")
    if (doc.get("parameters") or {}).get("strategy") != STRATEGY:
        raise HTTPException(status_code=404, detail="This experiment is not a seed study")
    return oid, doc


def _read_runs(factory: MongoRepository, oid: ObjectId) -> tuple[list[dict], list[dict]]:
    """Every attempt of the main generation, and of the control."""
    generations = factory.generation_repo.find_by_experiment(oid)
    control = {g["_id"] for g in generations
               if (g.get("seed_study") or {}).get("role") == CONTROL_ROLE}
    with factory.simulation_repo.connection.connect() as db:
        runs = list(db["simulations"].find(
            {"experiment_id": oid},
            {"generation_id": 1, "individual_id": 1, "random_seed": 1, "status": 1,
             "network_metrics": 1, "dodag.convergence.total_parent_switches": 1,
             "system_message": 1, "end_time": 1, "assigned_host_name": 1,
             "assigned_resource_name": 1}))
    return ([r for r in runs if r.get("generation_id") not in control],
            [r for r in runs if r.get("generation_id") in control])


def _measured(run: dict) -> dict[str, float]:
    """The numbers of one run. Anything JSON could not carry was not measured."""
    out: dict[str, float] = {}
    for name, value in (run.get("network_metrics") or {}).items():
        if (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value)):
            out[name] = float(value)
    switches = ((run.get("dodag") or {}).get("convergence") or {}).get("total_parent_switches")
    if switches is not None:
        out[PARENT_SWITCHES] = float(switches)
    return out


class _Results:
    """The runs of a study, arranged as chromosomes × seeds."""

    def __init__(self, doc: dict, main: list[dict], control: list[dict]):
        params = doc.get("parameters") or {}
        panel = doc.get("seed_study_panel") or {}
        self.seeds: list[int] = [int(s) for s in (params.get("simulation") or {}).get(
            "random_seeds") or []]
        self.chromosomes: list[str] = list(panel.get("genome_hashes") or [])
        self.objectives: list[dict] = params.get("objectives") or []
        self.attempts = main
        self.control_runs = control
        self.done = latest_per_seed(r for r in main if r.get("status") == EnumStatus.DONE)
        row_of = {genome_hash: i for i, genome_hash in enumerate(self.chromosomes)}
        column_of = {seed: i for i, seed in enumerate(self.seeds)}
        K, n = len(self.chromosomes), len(self.seeds)

        self.cells: dict[tuple[int, int], tuple[str, dict[str, float]]] = {}
        names: dict[str, None] = {}
        for run in self.done:
            row = row_of.get(run.get("individual_id"))
            column = column_of.get(run.get("random_seed"))
            if row is None or column is None:
                continue
            measured = _measured(run)
            self.cells[(row, column)] = (str(run["_id"]), measured)
            names.update(dict.fromkeys(measured))
        self.values = {name: np.full((K, n), np.nan) for name in names}
        for (row, column), (_, measured) in self.cells.items():
            for name, value in measured.items():
                self.values[name][row, column] = value

    def stamp(self) -> list[Any]:
        """What makes a cached analysis stale: a result that was not there."""
        latest = max((r["end_time"] for r in self.done if r.get("end_time")), default=None)
        return [len(self.done), len(self.control_runs), latest.isoformat() if latest else None]


def _determinism(results: _Results) -> dict[str, Any]:
    if not results.seeds:
        return seed_study.compare_control({}, {}, [])
    first = results.seeds[0]
    main: dict[str, dict[str, float]] = {}
    for run in results.done:
        if run.get("random_seed") == first:
            main[run.get("individual_id")] = _measured(run)
    control: dict[str, list[dict[str, float]]] = {}
    for run in results.control_runs:
        if run.get("status") == EnumStatus.DONE:
            control.setdefault(run.get("individual_id"), []).append(_measured(run))
    return seed_study.compare_control(main, control, sorted(results.values))


def _completeness(results: _Results, confidence: float) -> dict[str, Any]:
    planned = len(results.chromosomes) * len(results.seeds)
    message_of: dict[tuple, str] = {}
    active = 0
    for run in results.attempts:
        identity = (run.get("individual_id"), run.get("random_seed"))
        if run.get("status") == EnumStatus.ERROR:
            message_of[identity] = run.get("system_message") or ""
        elif run.get("status") in (EnumStatus.WAITING, EnumStatus.RUNNING):
            active += 1
    holes = []
    for (genome_hash, seed), message in message_of.items():
        if not any(r.get("individual_id") == genome_hash and r.get("random_seed") == seed
                   for r in results.done):
            holes.append({"genome_hash": genome_hash, "seed": seed, "message": message,
                          "kind": seed_study.failure_kind(message)})
    return {
        "planned": planned,
        "observed": len(results.cells),
        "pending": active,
        "failed_seeds": holes[:MAX_LISTED_HOLES],
        "failed_total": len(holes),
        "failures": seed_study.failure_summary(results.attempts, confidence),
    }


def _provenance(doc: dict, results: _Results, factory: MongoRepository) -> dict[str, Any]:
    """Where the numbers come from. A study only holds for the image it ran on.

    The image is read from the workers as they are registered now, since a
    simulation does not record it, and it is a tag rather than a digest.
    """
    used = {r.get("assigned_resource_name") for r in (*results.attempts, *results.control_runs)}
    images = sorted({
        (resource.get("container") or {}).get("image")
        for resource in factory.resource_repo.find_all()
        if resource.get("name") in used and (resource.get("container") or {}).get("image")
    })
    snapshot = doc.get("firmware_snapshot") or {}
    params = doc.get("parameters") or {}
    return {
        "images": images,
        "firmware": snapshot.get("status") or snapshot.get("state"),
        "seeds": (params.get("seed_study") or {}).get("seeds"),
        "panel": {k: v for k, v in (doc.get("seed_study_panel") or {}).items()
                  if k != "chromosomes"},
    }


def _top_warnings(determinism: dict, completeness: dict, diagnostics: dict,
                  provenance: dict, finished: bool) -> list[dict[str, Any]]:
    """What undermines the whole study, shown above any number."""
    warnings: list[dict[str, Any]] = []
    if determinism["checked"] and not determinism["identical"]:
        warnings.append({"code": "non_deterministic", "metric": None, "message": (
            f"Repeating a seed changed {determinism['divergent_total']} result(s). Something "
            "other than the seed adds noise, so the premise of this study does not hold.")})
    elif finished and not determinism["checked"]:
        warnings.append({"code": "determinism_unchecked", "metric": None, "message": (
            "The control did not run, so it is unverified that the seed is the only "
            "source of variation.")})
    missing = completeness["planned"] - completeness["observed"]
    if finished and missing > 0:
        warnings.append({"code": "incomplete", "metric": None, "message": (
            f"{missing} of {completeness['planned']} simulations have no result. The "
            "distributions are conditional on the seeds that finished.")})
    if diagnostics["low16_collisions"] or diagnostics["consecutive_pairs"]:
        warnings.append({"code": "seed_structure", "metric": None, "message": (
            f"The list has {diagnostics['low16_collisions']} pair(s) with the same low 16 "
            f"bits and {diagnostics['consecutive_pairs']} close pair(s). The firmware only "
            "sees those bits, so these seeds are not fully independent.")})
    if len(provenance["images"]) > 1:
        warnings.append({"code": "mixed_images", "metric": None, "message": (
            f"This study ran on {len(provenance['images'])} worker images. What a seed "
            "means depends on the image, so the runs may not be comparable.")})
    return warnings


def _parse_absolute(text: Optional[str]) -> dict[str, float]:
    """``latency:5,energy:100``, the absolute half-widths of a request."""
    out: dict[str, float] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        name, _, value = part.partition(":")
        try:
            number = float(value)
        except ValueError:
            raise _refuse(f"absolute: {part!r} is not metric:value")
        if not math.isfinite(number) or number <= 0:
            raise _refuse(f"absolute: {part!r} must be positive")
        out[name.strip()] = number
    return out


def _requested_targets(doc: dict, confidence: Optional[float], relative_error: Optional[float],
                       reliability: Optional[float]) -> dict[str, float]:
    stored = ((doc.get("parameters") or {}).get("seed_study") or {}).get("targets") or {}
    given = {"confidence": confidence, "relative_error": relative_error,
             "reliability": reliability}
    return _targets({**stored, **{k: v for k, v in given.items() if v is not None}})


@router.get("/{experiment_id}")
def get_study(
    experiment_id: str,
    confidence: Optional[float] = Query(None),
    relative_error: Optional[float] = Query(None),
    reliability: Optional[float] = Query(None),
    absolute: Optional[str] = Query(None),
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """The study and its analysis, as of now.

    The targets default to the ones stored with the study. Changing them
    recomputes the analysis without simulating anything.
    """
    oid, doc = _study_or_error(experiment_id, factory)
    targets = _requested_targets(doc, confidence, relative_error, reliability)
    absolutes = _parse_absolute(absolute)
    main, control = _read_runs(factory, oid)
    results = _Results(doc, main, control)

    summary = _summary(doc, len(results.done) + sum(
        1 for r in control if r.get("status") == EnumStatus.DONE))
    diagnostics = seed_study.diagnose_seeds(results.seeds) if results.seeds else {
        "low16_collisions": 0, "consecutive_pairs": 0, "guard": seed_study.DEFAULT_GUARD}
    determinism = _determinism(results)
    completeness = _completeness(results, targets["confidence"])
    provenance = _provenance(doc, results, factory)
    finished = doc.get("status") in (EnumStatus.DONE, EnumStatus.ERROR)

    analysis = None
    if results.values:
        analysis = _cache.get_or_compute(
            {"id": experiment_id, "version": seed_study.ANALYSIS_VERSION, "targets": targets,
             "absolute": absolutes, "stamp": results.stamp()},
            lambda: seed_study.analyse_study(
                results.values, seeds=results.seeds, chromosomes=results.chromosomes,
                objectives=results.objectives, targets=targets, absolute=absolutes))
    return {
        **summary,
        "seed_list": results.seeds,
        "panel": results.chromosomes,
        "seed_diagnostics": diagnostics,
        "determinism": determinism,
        "completeness": completeness,
        "provenance": provenance,
        "top_warnings": _top_warnings(determinism, completeness, diagnostics, provenance,
                                      finished),
        "analysis": analysis,
    }


@router.get("/{experiment_id}/metrics/{metric}")
def get_metric(
    experiment_id: str,
    metric: str,
    chromosome: Optional[str] = Query(None),
    confidence: Optional[float] = Query(None),
    relative_error: Optional[float] = Query(None),
    absolute: Optional[float] = Query(None, gt=0),
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """One metric on one chromosome, with what its charts need.

    Apart from the main read because its resampling is what costs, and a page
    opens a few of these, not one per metric and chromosome.
    """
    oid, doc = _study_or_error(experiment_id, factory)
    targets = _requested_targets(doc, confidence, relative_error, None)
    main, control = _read_runs(factory, oid)
    results = _Results(doc, main, control)
    if metric not in results.values:
        raise HTTPException(status_code=404, detail=f"No metric '{metric}' in this study")
    genome_hash = chromosome or (results.chromosomes[0] if results.chromosomes else None)
    if genome_hash not in results.chromosomes:
        raise HTTPException(status_code=404, detail="That chromosome is not in this study")
    row = results.chromosomes.index(genome_hash)

    spec = next((o.get("aggregator", "mean") for o in results.objectives
                 if o.get("metric_name") == metric), "mean")
    values = [results.cells.get((row, c), (None, {}))[1].get(metric)
              for c in range(len(results.seeds))]
    identifiers = [results.cells.get((row, c), (None, {}))[0]
                   for c in range(len(results.seeds))]
    return _cache.get_or_compute(
        {"id": experiment_id, "version": seed_study.ANALYSIS_VERSION, "metric": metric,
         "chromosome": genome_hash, "targets": targets, "absolute": absolute,
         "stamp": results.stamp()},
        lambda: seed_study.metric_detail(
            values, seeds=results.seeds, simulation_ids=identifiers,
            genome_hash=genome_hash, metric=metric, spec=spec, targets=targets,
            absolute=absolute))


@router.get("/{experiment_id}/values.csv")
def export_values(
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> Response:
    """Every value in long format, for analysis elsewhere."""
    oid, doc = _study_or_error(experiment_id, factory)
    main, control = _read_runs(factory, oid)
    results = _Results(doc, main, control)

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["chromosome", "seed", "order", "metric", "value", "status", "simulation_id"])
    for (row, column), (simulation_id, measured) in sorted(results.cells.items()):
        for metric, value in sorted(measured.items()):
            writer.writerow([results.chromosomes[row], results.seeds[column], column, metric,
                             repr(value), "Done", simulation_id])
    filename = f"seed-study-{experiment_id}.csv"
    return Response(content=out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})
