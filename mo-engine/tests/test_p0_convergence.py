"""Convergence / spread guard for the P0 synthetic adapter.

Validates the algorithmic fix that motivated P0. With the WSN P1 base, SCH1 runs
collapsed to a partial front (≈(0,4)→(1,0)) with no spread, because the radial
init + connectivity/coverage repair biased the decoded variable toward the region
centre. With P0 (uniform init, textbook SBX + polynomial mutation, evaluation
directly on x) an NSGA-II loop must span the full Pareto set (0,4)→(4,0) for SCH1
and converge to the unit hypersphere for DTLZ2.
"""
import math
import random

from lib.problem.resolve import build_adapter
from lib.nsga import fast_nondominated_sort, select_next_population
from lib.genetic_operators.selection import tournament_selection, compute_individual_ranks

POP = 40


def _sch1(x):  # mirrors master-node/lib/synthetic_data.py::_sch1
    x0 = x[0] * 2.0
    return [x0 * x0, (x0 - 2.0) ** 2]


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
        f.append(val)
    return f


def _adapter(n, seed):
    return build_adapter(
        {"name": "problem0", "n": n},
        {"eta_cx": 20.0, "eta_mt": 20.0, "per_gene_prob": 1.0 / n},
        random.Random(seed),
    )


def _offspring(adapter, parents, parents_obj, rng):
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


def _nsga2_front(evaluate, n, gens, seed=0):
    rng = random.Random(seed)
    adapter = _adapter(n, seed)
    pop = adapter.random_individual_generator(POP)
    pop_obj = [evaluate(c) for c in pop]
    for _ in range(gens):
        off = _offspring(adapter, pop, pop_obj, rng)
        off_obj = [evaluate(c) for c in off]
        R, R_obj = pop + off, pop_obj + off_obj
        fronts = fast_nondominated_sort(R_obj)
        idx = select_next_population(fronts, R_obj, list(range(len(R))), POP)
        pop, pop_obj = [R[i] for i in idx], [R_obj[i] for i in idx]
    return [pop_obj[i] for i in fast_nondominated_sort(pop_obj)[0]]


class TestP0Sch1Spread:
    def test_front_spans_full_pareto_range(self):
        front = _nsga2_front(lambda c: _sch1(c.x), n=1, gens=40, seed=0)
        f1 = [p[0] for p in front]
        f2 = [p[1] for p in front]
        # Both extremes reached: some solution near (0,4) and near (4,0).
        assert min(f1) < 0.2, f"f1 min too high (x≈0 not found): {min(f1):.3f}"
        assert min(f2) < 0.2, f"f2 min too high (x≈2 not found): {min(f2):.3f}"
        # And the front actually spreads (not collapsed to a single arc).
        assert max(f1) > 3.0, f"f1 max too low (no spread): {max(f1):.3f}"


class TestP0Dtlz2Convergence:
    def test_front_converges_to_unit_sphere(self):
        front = _nsga2_front(lambda c: _dtlz2(c.x, 3), n=6, gens=60, seed=0)
        norms = [math.sqrt(sum(v * v for v in p)) for p in front]
        mean = sum(norms) / len(norms)
        assert mean < 1.20, f"NSGA-II did not converge: mean||f||={mean:.3f}"
        assert min(norms) < 1.05, f"no point near the front: min||f||={min(norms):.3f}"
