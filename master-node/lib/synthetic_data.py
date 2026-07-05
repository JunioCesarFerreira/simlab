import os
import math
import random
import logging
from bson import ObjectId
from pylib.db import create_mongo_repository_factory, EnumStatus, MongoRepository
from pylib.db.models import Simulation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Genome extraction
# ---------------------------------------------------------------------------

def _extract_genome_from_sim(sim: dict) -> list[float]:
    """Reconstructs the [x0,y0,x1,y1,...] vector from fixedMotes."""
    fixed = (((sim or {}).get("parameters") or {})
             .get("simulationElements") or {}).get("fixedMotes") or []
    genome: list[float] = []
    for mote in fixed:
        pos = mote.get("position") or [0.0, 0.0]
        genome.extend([float(pos[0]), float(pos[1])])
    return genome


def _scale_to_unit(genome: list[float], region: tuple[float, float, float, float]) -> list[float]:
    """Maps (x,y) coordinates from *region* bounding box to [0,1]. Clips outliers."""
    x1, y1, x2, y2 = region
    dx = max(1e-9, x2 - x1)
    dy = max(1e-9, y2 - y1)
    xs: list[float] = []
    for i in range(0, len(genome), 2):
        x = (genome[i]     - x1) / dx
        y = (genome[i + 1] - y1) / dy
        xs.extend([min(1.0, max(0.0, x)), min(1.0, max(0.0, y))])
    return xs

# ---------------------------------------------------------------------------
# Benchmark functions (minimisation)
# ---------------------------------------------------------------------------

def _dtlz2(x: list[float], M: int) -> list[float]:
    """DTLZ2 — Pareto front is the unit hypersphere segment in the first orthant.

    Reference: Deb, Thiele, Laumanns, Zitzler (2005).
    n = len(x); k = n - (M-1).
    f_m = (1+g) * prod_{i=0}^{M-2-m} cos(π/2·x_i) * (m>0 ? sin(π/2·x_{M-1-m}) : 1)
    """
    n = len(x)
    k = max(1, n - (M - 1))
    tail = x[n - k:] if k > 0 else []
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


def _zdt1(x: list[float]) -> list[float]:
    """ZDT1 — convex Pareto front, exactly 2 objectives.

    Reference: Zitzler, Deb, Thiele (2000).
    f1 = x[0]
    g  = 1 + 9·sum(x[1:])/(n-1)
    f2 = g·(1 - sqrt(f1/g))
    """
    if not x:
        return [1.0, 1.0]
    f1 = float(x[0])
    g = 1.0 if len(x) == 1 else 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - math.sqrt(max(0.0, f1 / g)))
    return [f1, float(f2)]


def _sch1(x01: list[float], region: tuple[float, float, float, float]) -> list[float]:
    """SCH1 (Schaffer) — 2 objectives, 1 effective decision variable.

    f1 = x², f2 = (x-2)²  where x is the first decision variable
    mapped back to the original region scale.

    Using the normalised x01[0] ∈ [0,1] mapped to the region's x-axis:
      raw_x = x_min + x01[0] * (x_max - x_min)
    This is consistent with DTLZ2/ZDT1 which also operate on normalised values.
    """
    x_min, _, x_max, _ = region
    raw_x = x_min + x01[0] * (x_max - x_min) if x01 else 0.0
    return [float(raw_x ** 2), float((raw_x - 2.0) ** 2)]


def _eval_benchmark(
    genome_xy: list[float],
    region: tuple[float, float, float, float],
    bench: str,
    M: int,
    noise_std: float,
) -> list[float]:
    x01 = _scale_to_unit(genome_xy, region)
    bench_upper = bench.upper()
    if bench_upper == "ZDT1":
        vals = _zdt1(x01)
    elif bench_upper == "SCH1":
        vals = _sch1(x01, region)
    else:
        M = max(2, int(M))
        vals = _dtlz2(x01, M)
    if noise_std > 0.0:
        vals = [v + random.gauss(0.0, noise_std) for v in vals]
    return vals

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_synthetic_simulation(
    sim: Simulation,
    mongo: MongoRepository,
    bench: str = "DTLZ2",
    noise_std: float = 0.0,
) -> None:
    """Emulates execution using a classical benchmark function (DTLZ2, ZDT1 or SCH1)
    and writes the objectives according to the experiment's data_conversion_config.

    Args:
        sim:       Simulation document from MongoDB.
        mongo:     Repository factory (used to read the experiment config and write results).
        bench:     Benchmark identifier — takes priority over the BENCH env var.
        noise_std: Gaussian noise std-dev applied to each objective value.
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    log.info("Starting benchmark simulation %s (bench=%s noise_std=%s)", sim_oid, bench, noise_std)
    mongo.simulation_repo.mark_running(sim_oid)

    exp_id = sim["experiment_id"]
    cfg = mongo.experiment_repo.get_metrics_data_conversion(str(exp_id))
    obj_items = cfg.get("objectives", []) or []
    objective_names = [it["name"] for it in obj_items if "name" in it]

    params = (mongo.experiment_repo.get(exp_id) or {}).get("parameters", {}) or {}
    region = tuple(params.get("region", (-100.0, -100.0, 100.0, 100.0)))
    M = len(cfg.get("objectives", []))

    genome_xy = _extract_genome_from_sim(sim)
    vals = _eval_benchmark(genome_xy, region, bench=bench, M=M, noise_std=noise_std)

    bench_upper = bench.upper()
    if bench_upper == "ZDT1":
        if len(objective_names) != 2:
            log.warning(
                "ZDT1 requires exactly 2 objectives; data_conversion_config has %d. "
                "Truncating/padding to match.",
                len(objective_names),
            )
        vals = vals[:2]
    else:
        if len(vals) < len(objective_names):
            vals = vals + [1e9] * (len(objective_names) - len(vals))
        vals = vals[:len(objective_names)]

    objectives = {name: float(vals[i]) for i, name in enumerate(objective_names)}

    try:
        mongo.simulation_repo.mark_done(sim_oid, None, None, objectives)
    except Exception as e:
        log.exception("Failed to mark simulation %s as done; marking as error", sim_oid)
        try:
            mongo.simulation_repo.mark_error(sim_oid, str(e))
        except Exception:
            log.exception("Failed to mark simulation %s as error", sim_oid)
        return

    generation_id = sim.get("generation_id")
    if not generation_id:
        log.warning("Simulation %s has no generation_id; skipping generation close check", sim_oid)
        return
    gr = mongo.generation_repo
    try:
        if gr.all_simulations_done(generation_id):
            gr.mark_done(generation_id)
        elif not gr.any_simulation_active(generation_id):
            gr.mark_error(generation_id)
    except Exception:
        log.exception("Failed to close generation %s", generation_id)
