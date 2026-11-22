"""The three runners vary alike, return the same set, and the table runs.

    cd experiments/external-nsga3-benchmark && python -m pytest -q test_benchmark.py
"""
from __future__ import annotations

import contextlib
import io
import random

import numpy as np
import pandas as pd
import pytest
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.problem import Problem

import regenerate_table3
from metrics import hypervolume, non_dominated
from run_deap import _tournament, run_deap
from run_native import run_native
from run_pymoo import build_algorithm, run_pymoo

RUNNERS = {"native": run_native, "deap": run_deap, "pymoo": run_pymoo}


def quiet(call, *args, **kwargs):
    # pymoo prints a warning when there are more directions than individuals.
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args, **kwargs)


def test_hypervolume_is_exact_and_ignores_what_does_not_dominate_the_reference():
    assert hypervolume(np.array([[0.5, 0.5]]), np.array([1.0, 1.0])) == 0.25
    assert hypervolume(np.array([[0.5, 0.5], [1.2, 0.1]]), np.array([1.0, 1.0])) == 0.25


def test_non_dominated_keeps_one_copy_of_each_point():
    front = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert non_dominated(front).tolist() == [[0.0, 1.0], [1.0, 0.0]]


@pytest.mark.parametrize("name", list(RUNNERS))
@pytest.mark.parametrize("M, k", [(3, 2), (5, 4)])
def test_every_runner_returns_its_final_population(name, M, k):
    F = quiet(RUNNERS[name], M=M, k=k, pop_size=20, n_generations=15, seed=3)
    assert F.shape == (20, M)


@pytest.mark.parametrize("name", list(RUNNERS))
def test_a_runner_repeats_itself_with_the_same_seed(name):
    a = quiet(RUNNERS[name], M=3, k=2, pop_size=12, n_generations=5, seed=9)
    b = quiet(RUNNERS[name], M=3, k=2, pop_size=12, n_generations=5, seed=9)
    assert np.array_equal(a, b)


def test_pymoo_mutates_one_variable_in_n():
    # PM's prob is per individual; passing 1/n there mutated 1/n² of them.
    n = 12

    class Box(Problem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=2, xl=0.0, xu=1.0)

    mutation = build_algorithm(3, 12, 4, 20.0, 25.0, 0.9, 1.0 / n).mating.mutation
    X = np.random.default_rng(0).random((20000, n))
    np.random.seed(1)
    Y = mutation(Box(), Population.new(X=X.copy())).get("X")
    assert np.mean(Y != X) == pytest.approx(1.0 / n, rel=0.05)


def test_pymoo_mates_by_rank_and_keeps_duplicates():
    algorithm = build_algorithm(3, 12, 4, 20.0, 25.0, 0.9, 0.1)
    pop = Population.new(rank=np.array([0, 1, 1, 0]))
    P = np.array([[0, 1], [1, 0], [1, 2], [2, 3]])
    assert algorithm.mating.selection.func_comp(pop, P).tolist() == [0, 0, 1, 3]
    assert isinstance(algorithm.eliminate_duplicates, NoDuplicateElimination)


def test_deap_mates_by_rank():
    random.seed(0)
    pop, ranks = ["first", "second"], [1, 0]
    assert {_tournament(pop, ranks) for _ in range(50)} == {"second"}


def test_the_table_is_written(tmp_path, monkeypatch):
    out = tmp_path / "table3.csv"
    monkeypatch.setattr("sys.argv", [
        "regenerate_table3.py", "--seeds", "2", "--pop-size", "12",
        "--generations", "3", "--divisions", "4", "--out", str(out)])
    quiet(regenerate_table3.main)

    table = pd.read_csv(out)
    assert len(table) == 3 * len(regenerate_table3.TABLE3_GRID)
    assert set(table["method"]) == {"nsga3_func", "nsga3_deap_func", "nsga3_pymoo_func"}
    # The same reference point everywhere: no HV above the box it bounds.
    assert (table["mean_hv"] <= regenerate_table3.HV_REFERENCE ** table["M"] + 1e-9).all()
    # Three generations are too few for (6, 14) to reach the box: 0 there.
    assert (table[table["M"] <= 5]["mean_hv"] > 0).all()
