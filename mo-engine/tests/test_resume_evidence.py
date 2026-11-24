"""A resume keeps the evidence behind every vector (NSGA-II/III and batch).

The vectors decided at enqueue — infeasible, analytical, reused — have no
simulations to count. After a restart their evidence was looked up in a map
that died with the process: NSGA rewrote them as ``unknown`` and used, so a
penalty read as a measurement, and batch did not find them at all and replaced
the penalty by the 1e12 placeholder.
"""
from __future__ import annotations

import math
import random

import pytest

from lib.problem.resolve import build_adapter
from lib.strategy.batch import BatchStrategy
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga2_deap import NSGA2DeapStrategy
from lib.strategy.nsga2_pymoo import NSGA2PymooStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.nsga3_deap import NSGA3DeapStrategy
from lib.strategy.nsga3_pymoo import NSGA3PymooStrategy
from pylib.db.models.enums import EnumStatus
from pylib.evaluation_validity import is_valid_evaluation
from tests.resume_harness import GENERATIONS, KEYS, PROBLEMS, Killed, Run

NATIVE = [NSGA2LoopStrategy, NSGA3LoopStrategy]
VARIANTS = [NSGA2DeapStrategy, NSGA2PymooStrategy, NSGA3DeapStrategy, NSGA3PymooStrategy]
POINTS = ["waiting", "closed", "evaluating", "evaluated"]


def name(value):
    return getattr(value, "__name__", str(value))


def assert_resume_is_identical(problem, strategy, checkpoint, point, **options):
    straight = Run(problem, strategy=strategy, **options).finish()
    run = Run(problem, strategy=strategy, **options).until(checkpoint, point)

    run.restart()

    assert run.finish().result() == straight.result()


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("checkpoint", [0, 2, GENERATIONS])
@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("strategy", NATIVE, ids=name)
def test_nsga_resume_is_the_uninterrupted_run(strategy, problem, checkpoint, point):
    assert_resume_is_identical(problem, strategy, checkpoint, point)


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("checkpoint", [0, 2])
@pytest.mark.parametrize("strategy", VARIANTS, ids=name)
def test_library_variants_resume_the_same_way(strategy, checkpoint, point):
    assert_resume_is_identical("infeasible", strategy, checkpoint, point)


@pytest.mark.parametrize("strategy", NATIVE, ids=name)
def test_a_penalty_is_not_a_measurement_after_a_restart(strategy):
    run = Run("infeasible", strategy=strategy).until(0, "waiting")
    penalised = {
        i["individual_id"] for i in run.db.individuals
        if (i.get("completeness") or {}).get("state") == "infeasible"
    }
    assert penalised, "the scenario must hold an infeasible individual"

    run.restart()
    run.finish()

    first = run.db.generations[0]["_id"]
    records = [
        i["completeness"] for i in run.db.individuals
        if i["generation_id"] == first and i["individual_id"] in penalised
    ]
    assert not any(is_valid_evaluation(record) for record in records)


def some_refused(digest, metrics):
    if digest[2] < 90:
        return EnumStatus.DONE, {**metrics, KEYS[0]: math.nan}
    return EnumStatus.DONE, metrics


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("policy", ["worst", "accept"])
@pytest.mark.parametrize("strategy", NATIVE, ids=name)
def test_a_refusal_survives_a_restart(strategy, policy, point):
    assert_resume_is_identical(
        "repeated", strategy, 0, point,
        simulation={"incomplete_objective": policy}, outcome=some_refused)


@pytest.mark.parametrize("strategy", NATIVE, ids=name)
def test_a_survivor_the_cache_lost_keeps_its_evidence(strategy):
    """Killed between the individual's write and the cache's: the individual knows."""
    straight = Run("infeasible", strategy=strategy).finish()
    run = Run("infeasible", strategy=strategy).until(2, "waiting")
    lost = {
        i["individual_id"] for i in run.db.individuals
        if (i.get("completeness") or {}).get("state") == "infeasible"
    }
    for genome_hash in lost:
        run.db.cache[genome_hash].update(objectives=None)
        run.db.cache[genome_hash].pop("completeness", None)

    run.restart()
    run.finish()

    assert run.result()["individuals"] == straight.result()["individuals"]
    assert run.result()["front"] == straight.result()["front"]


# ── Batch ────────────────────────────────────────────────────────────────────

def a_batch(penalised_only=False):
    adapter = build_adapter(
        PROBLEMS["infeasible"], {"apply_coverage_repair": False}, random.Random(42))
    drawn = adapter.random_individual_generator(30)
    bad = [c for c in drawn if adapter.penalty_objectives(c, 2) is not None]
    good = [c for c in drawn if adapter.penalty_objectives(c, 2) is None]
    assert bad and good
    chosen = bad if penalised_only else [*good[:3], *bad, good[0]]     # with a repeat
    return {"chromosomes": [c.to_dict() for c in chosen]}


def batch_run(outcome=None, **batch):
    return Run("infeasible", strategy=BatchStrategy, outcome=outcome,
               parameters=a_batch(**batch))


def killed_batch(point, outcome=None, **batch):
    run = batch_run(outcome, **batch)
    if point in ("closed", "finishing"):
        run.work()
    if point == "finishing":
        run.db.die_before_finish = True
        with pytest.raises(Killed):
            run.handle()
    run.restart()
    return run


@pytest.mark.parametrize("outcome", [None, some_refused], ids=["all_done", "some_refused"])
@pytest.mark.parametrize("point", ["waiting", "closed", "finishing"])
def test_batch_resume_is_the_uninterrupted_run(point, outcome):
    straight = batch_run(outcome).finish()

    assert killed_batch(point, outcome).finish().result() == straight.result()


@pytest.mark.parametrize("point", ["closed", "finishing"])
def test_a_batch_of_penalties_stays_one_after_a_restart(point):
    straight = batch_run(penalised_only=True).finish()

    assert killed_batch(point, penalised_only=True).finish().result() == straight.result()
