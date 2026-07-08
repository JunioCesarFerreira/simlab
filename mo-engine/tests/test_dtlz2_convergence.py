"""End-to-end convergence test: the NSGA optimizers must reach the DTLZ2 true
Pareto front when driving the P1 synthetic vehicle.

This is the core guarantee behind the synthetic-benchmark mode — that the
optimization algorithms actually work. It runs a compact NSGA-III / NSGA-II loop
(real SimLab selection operators + P1 crossover/mutation) evaluating the DTLZ2
benchmark, and asserts the non-dominated front converges to the unit hypersphere
(||f||_2 = 1). It also acts as a regression guard on the hyperparameters needed
for convergence (strong mutation) and on the objective/normalization pipeline.
"""
import math
import random

import numpy as np
import pytest

from lib.problem.resolve import build_adapter
from lib.nsga import (
    fast_nondominated_sort,
    niching_selection,
    generate_reference_points,
    select_next_population,
)
from lib.genetic_operators.selection import tournament_selection, compute_individual_ranks

REGION = (-100.0, -100.0, 100.0, 100.0)
M = 3
N_VARS = 6            # 3 relays × 2 coords
POP = 40
GENS = 60
DIVISIONS = 8


# ── DTLZ2 evaluation (mirrors master-node/lib/synthetic_data.py) ─────────────

def _scale_to_unit(genome):
    xs = []
    for i in range(0, len(genome), 2):
        xs.append(min(1.0, max(0.0, (genome[i] + 100.0) / 200.0)))
        xs.append(min(1.0, max(0.0, (genome[i + 1] + 100.0) / 200.0)))
    return xs


def _dtlz2(x, m):
    n = len(x)
    k = max(1, n - (m - 1))
    g = sum((xi - 0.5) ** 2 for xi in x[n - k:])
    f = []
    for j in range(m):
        val = 1.0 + g
        for i in range(0, m - 1 - j):
            val *= math.cos(0.5 * math.pi * x[i])
        if j > 0:
            val *= math.sin(0.5 * math.pi * x[m - 1 - j])
        f.append(float(val))
    return f


def _evaluate(chrom):
    genome = [v for xy in chrom.relays for v in xy]
    return _dtlz2(_scale_to_unit(genome), M)


def _make_adapter(rng):
    problem = {
        "name": "problem1", "radius_of_reach": 200.0, "radius_of_inter": 200.0,
        "region": list(REGION), "sink": [0.0, 0.0],
        "mobile_nodes": [], "min_coverage_percentage": 0.0,
        "number_of_relays": (N_VARS + 1) // 2,
    }
    algo = {
        "population_size": POP, "prob_cx": 0.9, "prob_mt": 1.0,
        "per_gene_prob": 1.0 / N_VARS,          # ≈ standard 1/n mutation
        "crossover_method": "sbx_with_radial_translate",
        "mutation_method": "polynomial", "eta_cx": 20.0, "eta_mt": 20.0,
    }
    return build_adapter(problem, algo, rng)


def _make_offspring(adapter, parents, parents_obj, rng):
    ranks = compute_individual_ranks(fast_nondominated_sort(parents_obj))
    children, seen, attempts = [], set(), 0
    while len(children) < POP and attempts < POP * 10:
        attempts += 1
        p1 = tournament_selection(parents, ranks, rng)
        p2 = tournament_selection(parents, ranks, rng)
        c1, c2 = adapter.crossover([p1, p2]) if rng.random() < 0.9 else (p1, p2)
        c1 = adapter.mutate(c1)
        if c1 not in seen:
            children.append(c1); seen.add(c1)
        if len(children) >= POP:
            break
        c2 = adapter.mutate(c2)
        if c2 not in seen:
            children.append(c2); seen.add(c2)
    return children[:POP]


def _front_norms(objectives):
    front = [objectives[i] for i in fast_nondominated_sort(objectives)[0]]
    return [math.sqrt(sum(v * v for v in f)) for f in front]


def _run(select_fn):
    rng = random.Random(0)
    adapter = _make_adapter(rng)
    pop = adapter.random_individual_generator(POP)
    pop_obj = [_evaluate(c) for c in pop]
    initial_mean = float(np.mean(_front_norms(pop_obj)))

    for _ in range(GENS):
        offspring = _make_offspring(adapter, pop, pop_obj, rng)
        off_obj = [_evaluate(c) for c in offspring]
        pop, pop_obj = select_fn(pop + offspring, pop_obj + off_obj, rng)

    norms = _front_norms(pop_obj)
    return initial_mean, float(np.mean(norms)), float(np.min(norms))


def _nsga3_select(R, R_obj, rng):
    ref_points = generate_reference_points(M, DIVISIONS)
    fronts = fast_nondominated_sort(R_obj)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= POP:
            selected.extend(front)
        else:
            remaining = POP - len(selected)
            if remaining > 0:
                selected.extend(niching_selection(front, R_obj, ref_points, remaining, rng))
            break
    return [R[i] for i in selected], [R_obj[i] for i in selected]


def _nsga2_select(R, R_obj, rng):
    fronts = fast_nondominated_sort(R_obj)
    idx = select_next_population(fronts, R_obj, list(range(len(R))), POP)
    return [R[i] for i in idx], [R_obj[i] for i in idx]


class TestDtlz2Convergence:
    def test_nsga3_converges_to_unit_sphere(self):
        initial_mean, final_mean, final_min = _run(_nsga3_select)
        # DTLZ2 front: ||f||_2 == 1. Converged run gets the mean well below the
        # ~1.3–1.5 a starved (weak-mutation) search stalls at.
        assert final_mean < initial_mean - 0.3, (initial_mean, final_mean)
        assert final_mean < 1.15, f"NSGA-III did not converge: mean||f||={final_mean:.3f}"
        assert final_min < 1.05, f"no point near the front: min||f||={final_min:.3f}"

    def test_nsga2_converges_to_unit_sphere(self):
        initial_mean, final_mean, final_min = _run(_nsga2_select)
        assert final_mean < initial_mean - 0.3, (initial_mean, final_mean)
        assert final_mean < 1.20, f"NSGA-II did not converge: mean||f||={final_mean:.3f}"
        assert final_min < 1.05, f"no point near the front: min||f||={final_min:.3f}"
