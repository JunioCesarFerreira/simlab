import os
import random
import logging
from typing import Mapping, Optional
from bson import ObjectId
from pylib import benchmarks
from pylib.db import MongoRepository
from pylib.db.models import Simulation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode resolution (per-experiment config takes precedence over env vars)
# ---------------------------------------------------------------------------

def resolve_synthetic_settings(
    exp_doc: Optional[dict],
    env: Optional[Mapping[str, str]] = None,
) -> tuple[bool, str, float]:
    """Resolve ``(enabled, bench, noise_std)`` for a simulation's experiment.

    The per-experiment ``parameters.simulation.synthetic`` block takes priority
    over the ``ENABLE_DATA_SYNTHETIC`` / ``BENCH`` / ``NOISE_STD`` environment
    variables. The nested lookup tolerates missing or present-but-None levels,
    and ``enabled`` accepts both real booleans and string flags
    (``"true"``/``"false"``) so legacy/stringified configs still work.
    """
    env = os.environ if env is None else env
    syn_cfg = (
        (((exp_doc or {}).get("parameters") or {}).get("simulation") or {}).get("synthetic") or {}
    )
    env_enabled = str(env.get("ENABLE_DATA_SYNTHETIC", "False")).strip().lower() == "true"
    enabled_raw = syn_cfg.get("enabled", env_enabled)
    enabled = (
        enabled_raw.strip().lower() == "true"
        if isinstance(enabled_raw, str)
        else bool(enabled_raw)
    )
    bench = syn_cfg.get("bench") or env.get("BENCH", "DTLZ2")
    noise_std = float(syn_cfg.get("noise_std", env.get("NOISE_STD", "0.0")))
    return enabled, bench, noise_std

# ---------------------------------------------------------------------------
# Genome extraction
# ---------------------------------------------------------------------------

def _extract_genome_from_sim(sim: dict) -> list[float]:
    """Reconstructs the [x0,y0,x1,y1,...] decision-variable vector from fixedMotes.

    The sink is a fixed infrastructure node (not a decision variable), so it is
    excluded — only relay positions form the genome. This keeps the number of
    decision variables consistent with n_relays (genome length = 2·n_relays).
    """
    fixed = (((sim or {}).get("parameters") or {})
             .get("simulationElements") or {}).get("fixedMotes") or []
    genome: list[float] = []
    for mote in fixed:
        if mote.get("name") == "sink":
            continue
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

# The benchmark objective functions and true fronts live in the canonical
# pylib.benchmarks module (single source of truth, shared with rest-api and the
# in-process P0 adapter). These module-level aliases preserve the historical
# import surface used by the tests.
_dtlz2 = benchmarks.dtlz2
_zdt1 = benchmarks.zdt1
_sch1 = benchmarks.sch1


def _benchmark_values(x01: list[float], bench: str, M: int) -> list[float]:
    """Evaluate the benchmark on an already-normalised vector x01 ∈ [0,1]^n."""
    return benchmarks.evaluate(bench, x01, M)


def _apply_noise(vals: list[float], noise_std: float) -> list[float]:
    if noise_std > 0.0:
        return [v + random.gauss(0.0, noise_std) for v in vals]
    return vals


def _eval_benchmark(
    genome_xy: list[float],
    region: tuple[float, float, float, float],
    bench: str,
    M: int,
    noise_std: float,
) -> list[float]:
    """Physical-genome path (P1): scale relay coordinates from *region* to
    [0,1]^n, then evaluate. Kept for backward compatibility with P1-encoded
    synthetic experiments."""
    x01 = _scale_to_unit(genome_xy, region)
    return _apply_noise(_benchmark_values(x01, bench, M), noise_std)


def _decision_vector_from_sim(sim: dict) -> Optional[list[float]]:
    """Return the analytical decision vector x ∈ [0,1]^n for a P0 synthetic
    simulation, or ``None`` when the simulation encodes a physical (P1) genome.

    P0 exposes the benchmark decision variables verbatim under
    ``parameters.simulationElements.decisionVector`` — no motes, no scaling.
    """
    elements = (((sim or {}).get("parameters") or {}).get("simulationElements") or {})
    dv = elements.get("decisionVector")
    if dv is None:
        return None
    return [float(v) for v in dv]

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
    params = (mongo.experiment_repo.get(exp_id) or {}).get("parameters", {}) or {}

    # Objective names must match parameters.objectives[].metric_name, since the
    # mo-engine reads simulation results keyed by those exact names (in order).
    obj_items = params.get("objectives", []) or []
    objective_names = [
        it.get("metric_name") or it.get("name")
        for it in obj_items
        if it.get("metric_name") or it.get("name")
    ]
    # Legacy fallback: derive names from data_conversion_config.metrics.
    if not objective_names:
        cfg = mongo.experiment_repo.get_metrics_data_conversion(str(exp_id)) or {}
        objective_names = [m["name"] for m in (cfg.get("metrics") or []) if m.get("name")]

    # 'region' lives inside the problem sub-document (parameters.problem.region);
    # fall back to a legacy top-level 'region' and finally to a default Ω.
    problem_cfg = params.get("problem", {}) or {}
    region = tuple(
        problem_cfg.get("region")
        or params.get("region")
        or (-100.0, -100.0, 100.0, 100.0)
    )
    M = len(objective_names)

    # Optional SCH1 decision-domain override (default is the wide convergence
    # domain defined in pylib.benchmarks). Ignored by the other benchmarks.
    syn_cfg = ((params.get("simulation") or {}).get("synthetic") or {})
    raw_domain = syn_cfg.get("sch1_domain")
    sch1_domain = tuple(raw_domain) if raw_domain else benchmarks.SCH1_DEFAULT_DOMAIN

    # Reproducible per-(seed, genome) observation noise: the same genome under the
    # same simulation seed always yields the same objectives, so noisy synthetic
    # runs are reproducible and the objective cache stays consistent (a cache hit
    # returns exactly what a fresh evaluation would produce).
    noise_rng = random.Random(f"{sim.get('random_seed', 0)}:{sim.get('individual_id', '')}")

    decision_vector = _decision_vector_from_sim(sim)
    if decision_vector is not None:
        # P0 (pure synthetic): the decision vector is already in [0,1]^n; clip defensively.
        x01 = [min(1.0, max(0.0, v)) for v in decision_vector]
    else:
        # P1-encoded synthetic (legacy): reconstruct the genome from relay motes
        # and scale from the physical region to [0,1]^n.
        x01 = _scale_to_unit(_extract_genome_from_sim(sim), region)

    try:
        # evaluate_noisy validates dimensions (e.g. DTLZ2 needs n >= M-1, raising a
        # clear ValueError instead of a latent IndexError) and clamps noisy
        # non-negative objectives at 0 so HV/GD/IGD stay well-defined.
        vals = benchmarks.evaluate_noisy(bench, x01, M, noise_std, noise_rng, sch1_domain=sch1_domain)
    except ValueError as exc:
        log.error("Invalid synthetic configuration for simulation %s: %s", sim_oid, exc)
        mongo.simulation_repo.mark_error(sim_oid, str(exc))
        return

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
