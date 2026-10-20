"""Read-only numerical audit of SimLab's NSGA kernels and NSGA-Studies.

Run from the repository root with .venv/bin/python. No MongoDB is needed.
Only the requested output directory is written; application code is untouched.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.metadata
import itertools
import json
import logging
from pathlib import Path
import random
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import moocore
from deap import tools
from pymoo.indicators.gd import GD
from pymoo.problems import get_problem

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "mo-engine")]
sys.dont_write_bytecode = True

from pylib import benchmarks, moo_metrics
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.nsga3_deap import NSGA3DeapStrategy
from lib.strategy.nsga3_pymoo import NSGA3PymooStrategy
from lib.genetic_operators.crossover.simulated_binary_crossover import sbx


def make_strategy(cls, seed=42, bench="DTLZ2", n=10, m=3, pop=50,
                  generations=20, mutation=0.1, per_gene=0.05, divisions=10):
    return cls({"parameters": {
        "problem": {"name": "problem0", "n": n},
        "simulation": {"synthetic": {"enabled": True, "bench": bench},
                       "random_seeds": [42]},
        "algorithm": {"population_size": pop, "number_of_generations": generations,
                      "random_seed": seed, "prob_cx": 0.9, "prob_mt": mutation,
                      "per_gene_prob": per_gene, "eta_cx": 20, "eta_mt": 20,
                      "divisions": divisions},
        "objectives": [{"metric_name": f"f{i+1}", "goal": "min"} for i in range(m)],
    }}, None)


def front(F):
    return np.asarray(moocore.filter_dominated(np.unique(F, axis=0)))


def indicators(F, reference, hv_ref):
    F = front(F)
    return {"hv": float(moocore.hypervolume(F, ref=hv_ref)),
            "gd": moo_metrics.gd(F, reference),
            "igd": moo_metrics.igd(F, reference), "front_size": len(F)}


def primitives():
    rng = np.random.default_rng(37)
    errors = {"hv_inclusion_exclusion": 0.0, "gd_pymoo_raw": 0.0, "gd_direct_raw": 0.0,
              "dtlz2_pymoo": 0.0, "zdt1_pymoo": 0.0}
    for m in (2, 3, 5):
        for _ in range(10):
            F, R = rng.random((30, m)), rng.random((80, m))
            # Exact independent union of six boxes, by inclusion-exclusion.
            boxes = F[:6]
            exact_hv = sum((-1)**(k+1) * float(np.prod(1.1-np.max(subset, axis=0)))
                for k in range(1, 7) for subset in itertools.combinations(boxes, k))
            errors["hv_inclusion_exclusion"] = max(errors["hv_inclusion_exclusion"], abs(
                moocore.hypervolume(boxes, ref=np.ones(m)*1.1) - exact_hv))
            actual = moo_metrics.gd(F, R, normalized=False)
            errors["gd_pymoo_raw"] = max(errors["gd_pymoo_raw"], abs(actual-GD(R)(F)))
            direct = np.linalg.norm(F[:, None, :] - R[None, :, :], axis=2).min(axis=1).mean()
            errors["gd_direct_raw"] = max(errors["gd_direct_raw"], abs(actual-direct))
        X = rng.random((80, m+7))
        expected = get_problem("dtlz2", n_var=m+7, n_obj=m).evaluate(X)
        actual = np.array([benchmarks.dtlz2(x, m) for x in X])
        errors["dtlz2_pymoo"] = max(errors["dtlz2_pymoo"], float(np.max(abs(expected-actual))))
    X = rng.random((80, 10))
    errors["zdt1_pymoo"] = float(np.max(abs(get_problem("zdt1", n_var=10).evaluate(X)
        - np.array([benchmarks.zdt1(x.tolist()) for x in X]))))
    assert max(errors.values()) < 1e-12, errors

    class FixedRng:
        def __init__(self):
            self.values = iter([0.99, 0.75])
        def random(self):
            return next(self.values)

    local = sbx(0.8, 0.99, FixedRng(), 20, (0, 1))
    with patch("random.random", side_effect=[0.0, 0.99, 0.75]):
        reference_children = tools.cxSimulatedBinaryBounded([0.8], [0.99], 20, 0, 1)
    sbx_result = {"parents": [0.8, 0.99], "eta": 20, "draw": 0.99,
                  "simlab": list(local), "deap": [x[0] for x in reference_children]}
    # Independent front samples on the exact sphere expose discretization error.
    discretization = []
    for m in (2, 3, 6):
        exact = benchmarks.true_front("DTLZ2", m, 200, seed=123)
        R = benchmarks.true_front("DTLZ2", m)
        discretization.append({"m": m, "reference_size": len(R),
            "exact_radial_distance": float(np.mean(abs(np.linalg.norm(exact, axis=1)-1))),
            "gd_to_sample_normalized": moo_metrics.gd(exact, R)})
    return {"parity_max_absolute_errors": errors, "sbx": sbx_result,
            "reference_discretization": discretization}


def repeatability():
    # Same input, fresh strategy with the same algorithm seed on every call.
    F = benchmarks.true_front("DTLZ2", 3, 100, seed=9).tolist()
    result = {}
    np.random.seed(123)
    for cls in (NSGA3LoopStrategy, NSGA3DeapStrategy, NSGA3PymooStrategy):
        selections = []
        for _ in range(5):
            s = make_strategy(cls, pop=30, divisions=6)
            selections.append(sorted(s._select_next_parents(list(range(100)), F)))
        result[cls.__name__] = {"unique_selected_sets_in_5_calls":
                               len({tuple(x) for x in selections})}
    return result


def run_kernel(cls, seed, mutation, per_gene, generations=20):
    """Original mating + survival methods; record Q, selected P, and archive.

    This is a synchronous kernel harness, not a MongoDB lifecycle simulation.
    Unlike the production finalizer, it performs survival after the last Q too.
    """
    s = make_strategy(cls, seed=seed, mutation=mutation, per_gene=per_gene)
    s._parents = s._problem_adapter.random_individual_generator(s._pop_size)
    def evaluate(pop):
        for c in pop:
            s._map_genome_objectives[c] = benchmarks.dtlz2(c.x, 3)
        return np.array([s._map_genome_objectives[c] for c in pop])
    R = benchmarks.true_front("DTLZ2", 3)
    hv_ref = [1.1]*3
    F = evaluate(s._parents)
    archive = front(F)
    history = [{"generation": 0, "offspring": indicators(F, R, hv_ref),
                "survivors": indicators(F, R, hv_ref), "archive_hv":
                float(moocore.hypervolume(archive, ref=hv_ref))}]
    for g in range(1, generations+1):
        children = s._run_genetic_algorithm()
        Q = evaluate(children)
        union = s._parents + children
        union_F = np.array([s._map_genome_objectives[c] for c in union])
        s._parents = s._select_next_parents(union, union_F.tolist())
        F = np.array([s._map_genome_objectives[c] for c in s._parents])
        archive = front(np.vstack([archive, Q]))
        history.append({"generation": g, "offspring": indicators(Q, R, hv_ref),
                        "survivors": indicators(F, R, hv_ref),
                        "archive_hv": float(moocore.hypervolume(archive, ref=hv_ref))})
    return history


def notebook_comparison(studies):
    path = studies / "notebooks/evaluation/sch1-nsga-metrics.ipynb"
    if not path.exists():
        return {"unavailable": str(path)}
    sys.path.insert(0, str(studies))
    from modules.nsga2 import nsga2_func
    from modules.nsga3 import nsga3_func
    notebook = json.loads(path.read_text())
    namespace = {"np": np, "random": random, "copy": copy,
        "nsga2_func": nsga2_func, "nsga3_func": nsga3_func,
        "POP_SIZE": 50, "GENERATIONS": 20, "BOUNDS": [(-10., 10.)],
        "DIVISIONS": 49, "SEED": 42, "ETA_C": 20., "ETA_M": 20.,
        "P_CROSSOVER": 0.9, "P_MUTATION": 1., "REF_POINT": np.array([4.4, 4.4])}
    # Load function definitions only; do not execute plotting/output cells.
    for i in (3, 6, 8, 10, 14):
        tree = ast.parse("".join(notebook["cells"][i]["source"]))
        defs = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
        exec(compile(defs, str(path)+f":cell{i}", "exec"), namespace)
    namespace["OBJECTIVES"] = [namespace["f1"], namespace["f2"]]
    initial = namespace["initial_population"](50, [(-10., 10.)], 42)
    R1000 = namespace["true_pareto_front"](1000)
    R500 = benchmarks.true_front("SCH1", 2)
    result = {}
    for label, fn in (("NSGA-II", "run_nsga2"), ("NSGA-III", "run_nsga3")):
        F = namespace[fn](20, initial)
        distances = namespace["min_distances"](F, R1000)
        result[label] = {"front_size": len(F),
            "notebook_hv": namespace["hypervolume"](F),
            "simlab_hv_same_front": float(moocore.hypervolume(F, ref=[4.4, 4.4])),
            "notebook_gd_p2_raw_ref1000": namespace["generational_distance"](F, R1000),
            "gd_p1_raw_ref1000": float(distances.mean()),
            "gd_p1_normalized_ref1000": moo_metrics.gd(F, R1000),
            "simlab_gd_same_front_ref500": moo_metrics.gd(F, R500)}
        assert abs(result[label]["notebook_hv"]-result[label]["simlab_hv_same_front"]) < 1e-12
    return result


def lifecycle():
    results = {}
    for cls in (NSGA2LoopStrategy, NSGA3LoopStrategy):
        s = make_strategy(cls, pop=10, generations=2)
        recorded, selected = [], []
        original_selection = s._select_next_parents
        def select(pop, F):
            selected.append(s._gen_index-1)
            return original_selection(pop, F)
        s._select_next_parents = select
        def enqueue():
            recorded.append(s._gen_index)
            s._gen_index += 1
            for c in s._current_population:
                s._map_genome_objectives[c] = benchmarks.dtlz2(c.x, 3)
        done = []
        s._generation_enqueue = enqueue
        s._finalize_experiment = lambda **kw: done.append(kw)
        s._current_population = s._problem_adapter.random_individual_generator(10)
        enqueue()
        while not done:
            s._evolution()
            assert len(recorded) < 10
        results[cls.__name__] = {"configured_generations": 2,
            "recorded_generation_indices": recorded,
            "survival_applied_at_indices": selected,
            "final_front_size": len(done[0]["pareto_front"])}
    return results


def checkpoint_restore():
    from bson import ObjectId
    result = {}
    for cls in (NSGA2LoopStrategy, NSGA3LoopStrategy):
        s = make_strategy(cls, pop=10)
        initial = s._problem_adapter.random_individual_generator(10)
        s._parents = initial
        for c in initial:
            s._map_genome_objectives[c] = benchmarks.dtlz2(c.x, 3)
        q1 = s._run_genetic_algorithm()
        for c in q1:
            s._map_genome_objectives[c] = benchmarks.dtlz2(c.x, 3)
        union = initial + q1
        s._parents = s._select_next_parents(union, [s._map_genome_objectives[c] for c in union])
        expected = {c.get_hash() for c in s._parents}
        q2 = s._run_genetic_algorithm()
        for c in q2:
            s._map_genome_objectives[c] = benchmarks.dtlz2(c.x, 3)
        gens = [{"_id": ObjectId(), "index": i} for i in range(3)]
        stored = {g["_id"]: [{"individual_id": c.get_hash(), "chromosome": c.to_dict(),
                             "objectives": s._map_genome_objectives[c]} for c in pop]
                  for g, pop in zip(gens, (initial, q1, q2))}
        fresh = make_strategy(cls, pop=10)
        fresh.mongo = MagicMock()
        fresh.mongo.individual_repo.find_by_generation.side_effect = lambda gid: stored[gid]
        fresh._restore_population_state(gens, gens[-1])
        restored = {c.get_hash() for c in fresh._parents}
        result[cls.__name__] = {"expected_survivors": len(expected),
            "retained_parents_missing_after_restore": len(expected-restored),
            "rng_state_preserved": s._ga_rng.getstate() == fresh._ga_rng.getstate()}
    return result


def plot_history(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [r for r in result["kernel_runs"]
            if r["algorithm"] == "NSGA3LoopStrategy" and r["config"] == "gui_defaults"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, metric, label in zip(axes, ("hv", "gd", "igd"), ("HV ↑", "GD ↓", "IGD ↓")):
        for field, title, color in (("offspring", "Descendentes (curva atual)", "#dc702f"),
                                    ("survivors", "População após seleção", "#256ba6")):
            values = np.array([[g[field][metric] for g in r["history"]] for r in runs])
            x = np.arange(values.shape[1])
            ax.plot(x, values.mean(axis=0), label=title, color=color, linewidth=2)
        if metric == "hv":
            values = np.array([[g["archive_hv"] for g in r["history"]] for r in runs])
            ax.plot(x, values.mean(axis=0), label="Arquivo acumulado", color="#4c8c57", linestyle="--")
        ax.set(xlabel="Geração", ylabel=label)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Mesmo NSGA-III e mesmas avaliações; conjuntos medidos diferentes\n"
                 "DTLZ2: M=3, n=10, população 50 · média de 5 sementes · parâmetros padrão da interface",
                 fontsize=11)
    fig.savefig(output / "population-comparison.svg")
    fig.savefig(output / "population-comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, default=ROOT.parent / "NSGA-Studies")
    parser.add_argument("--output", type=Path, default=Path("/tmp/simlab-nsga-audit"))
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    result = {"versions": {p: importlib.metadata.version(p)
        for p in ("numpy", "pymoo", "deap", "moocore")},
        "primitives": primitives(), "repeatability": repeatability(),
        "notebook_same_front": notebook_comparison(args.studies),
        "lifecycle": lifecycle(), "checkpoint_restore": checkpoint_restore(), "kernel_runs": []}
    for cls in (NSGA2LoopStrategy, NSGA3LoopStrategy):
        for label, mutation, per_gene in (("gui_defaults", 0.1, 0.05),
                                          ("mutation_1_over_n", 1.0, 0.1)):
            for seed in (1, 2, 3, 5, 7):
                history = run_kernel(cls, seed, mutation, per_gene)
                result["kernel_runs"].append({"algorithm": cls.__name__, "config": label,
                    "seed": seed, "history": history})
            print(cls.__name__, label, "done", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    plot_history(result, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "kernel_runs"}, indent=2))
    for cls in (NSGA2LoopStrategy, NSGA3LoopStrategy):
        for config in ("gui_defaults", "mutation_1_over_n"):
            runs = [r for r in result["kernel_runs"] if r["algorithm"] == cls.__name__ and r["config"] == config]
            means = {metric: float(np.mean([r["history"][-1]["survivors"][metric] for r in runs]))
                     for metric in ("hv", "gd", "igd")}
            print(cls.__name__, config, means)
    print("Results:", args.output / "results.json")


if __name__ == "__main__":
    main()
