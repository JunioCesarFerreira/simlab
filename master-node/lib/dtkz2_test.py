import os
import math
import random
import logging
from bson import ObjectId
from pylib import mongo_db
from dto import Simulation

log = logging.getLogger(__name__)

# ========= Helpers de benchmark =========

def _extract_genome_from_sim(sim: dict) -> list[float]:
    """Reconstrói o vetor [x0,y0,x1,y1,...] a partir de fixedMotes."""
    fixed = (((sim or {}).get("parameters") or {})
             .get("simulationElements") or {}).get("fixedMotes") or []
    genome: list[float] = []
    for mote in fixed:
        pos = mote.get("position") or [0.0, 0.0]
        genome.extend([float(pos[0]), float(pos[1])])
    return genome

def _scale_to_unit(genome: list[float], region: tuple[float,float,float,float]) -> list[float]:
    """Mapeia (x,y) do retângulo region -> [0,1]. Clipa valores."""
    x1, y1, x2, y2 = region
    dx = max(1e-9, (x2 - x1))
    dy = max(1e-9, (y2 - y1))
    xs: list[float] = []
    # pares (x,y) sequenciais
    for i in range(0, len(genome), 2):
        x = (genome[i]   - x1) / dx
        y = (genome[i+1] - y1) / dy
        xs.extend([min(1.0, max(0.0, x)), min(1.0, max(0.0, y))])
    return xs

def _dtlz2(x: list[float], M: int) -> list[float]:
    """
    DTLZ2 (minimização). Pareto-Front: hiperesfera no 1º ortante.
    n = len(x); k = n - (M-1).
    f_m = (1+g) * ∏_{i=1}^{M-1-m} cos(π x_i / 2) * (m>0 ? sin(π x_{M-m} / 2) : 1)
    """
    n = len(x)
    k = max(1, n - (M - 1))
    tail = x[n-k:] if k > 0 else []
    g = sum((xi - 0.5)**2 for xi in tail)
    f: list[float] = []
    for m in range(M):
        val = 1.0 + g
        # produto de cos
        for i in range(0, M-1-m):
            val *= math.cos(0.5 * math.pi * x[i])
        # um termo de sen a partir do “fim”
        if m > 0:
            val *= math.sin(0.5 * math.pi * x[M-1-m])
        f.append(float(val))
    return f

def _zdt1(x: list[float]) -> list[float]:
    """
    ZDT1 (M=2, minimização). Front contínuo, convexo.
    f1 = x1
    g  = 1 + 9 * sum(x2..xn)/(n-1)
    f2 = g * (1 - sqrt(f1/g))
    """
    if not x:
        return [1.0, 1.0]
    f1 = float(x[0])
    if len(x) == 1:
        g = 1.0
    else:
        g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - math.sqrt(f1 / g))
    return [f1, float(f2)]

def _eval_benchmark(genome_xy: list[float], region: tuple[float,float,float,float],
                    bench: str, M: int, noise_std: float) -> list[float]:
    x01 = _scale_to_unit(genome_xy, region)
    if bench.upper() == "ZDT1":
        vals = _zdt1(x01)
    else:  # DTLZ2 default
        M = max(2, int(M))
        vals = _dtlz2(x01, M)
    if noise_std > 0.0:
        vals = [v + random.gauss(0.0, noise_std) for v in vals]
    return vals

# ========= Substitui o run_fake_simulation =========

def run_benchmark_simulation(sim: Simulation, mongo: mongo_db.MongoRepository) -> None:
    """
    Emula execução usando um benchmark clássico (DTLZ2 ou ZDT1) e grava os
    objetivos conforme o transform_config do experimento.
    """
    sim_oid = ObjectId(sim["_id"]) if not isinstance(sim["_id"], ObjectId) else sim["_id"]
    log.info("Starting benchmark simulation %s", sim_oid)
    mongo.simulation_repo.mark_running(sim_oid)

    # config do experimento (objetivos e região)
    exp_id = sim["experiment_id"]
    cfg = mongo.experiment_repo.get_objectives_and_metrics(str(exp_id))
    obj_items = cfg.get("objectives", []) or []
    objective_names = [it["name"] for it in obj_items if "name" in it]

    # parâmetros necessários para reescalar
    params = (mongo.experiment_repo.get_by_id(exp_id) or {}).get("parameters", {}) or {}
    region = tuple(params.get("region", (-100.0, -100.0, 100.0, 100.0)))
    M = int(os.getenv("BENCH_M", str(max(2, len(objective_names) or 3))))
    bench = os.getenv("BENCH", "DTLZ2").upper()  # "DTLZ2" | "ZDT1"
    noise_std = float(os.getenv("NOISE_STD", "0.0"))

    # extrai genoma e avalia
    genome_xy = _extract_genome_from_sim(sim)
    vals = _eval_benchmark(genome_xy, region, bench=bench, M=M, noise_std=noise_std)

    # garante número correto de objetivos
    if bench == "ZDT1":
        if len(objective_names) != 2:
            log.warning("ZDT1 requer 2 objetivos; transform_config tem %d. Ajuste recomendado.", len(objective_names))
        vals = vals[:2]
    else:
        if len(vals) < len(objective_names):
            # completa com grandes valores (pior) para evitar shape mismatch
            vals = vals + [1e9] * (len(objective_names) - len(vals))
        vals = vals[:len(objective_names)]

    # monta dict na ordem correta
    objectives = {name: float(vals[i]) for i, name in enumerate(objective_names)}
    metrics = {}  # opcional: deixar vazio

    # marca como DONE (sem CSV/logs reais)
    mongo.simulation_repo.mark_done(sim_oid, sim_oid, sim_oid, objectives, metrics)

    # checa conclusão da geração
    gen_id = sim["generation_id"]
    if mongo.generation_repo.all_simulations_done(gen_id):
        mongo.generation_repo.mark_done(gen_id)
