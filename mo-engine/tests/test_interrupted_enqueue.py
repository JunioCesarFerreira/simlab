"""An engine killed at any write resumes into the run nobody killed.

The case that was left is the enqueue: a generation is several documents, so a
death in the middle left a fragment. NSGA and batch evaluated it as if it were
whole or waited for ever, random search refused it. A resume now finishes
writing it: the previous generation's handler is deterministic, so it draws
the same population, and what is already written is not written again.

Every test below counts the writes of an uninterrupted run and kills at them,
one run per write, so no point between two states is assumed to be safe.
"""
from __future__ import annotations

import random

import pytest

from lib.problem.resolve import build_adapter
from lib.strategy.batch import BatchStrategy
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga2_deap import NSGA2DeapStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.nsga3_pymoo import NSGA3PymooStrategy
from lib.strategy.random_search import RandomSearchStrategy
from tests.resume_harness import PROBLEMS, Killed, Run

LOOPS = [NSGA2LoopStrategy, NSGA3LoopStrategy, RandomSearchStrategy]
#: One write in this many, starting from each offset that the case asks for.
STRIDE = 7


def batch_parameters():
    adapter = build_adapter(
        PROBLEMS["infeasible"], {"apply_coverage_repair": False}, random.Random(42))
    drawn = adapter.random_individual_generator(30)
    bad = [c for c in drawn if adapter.penalty_objectives(c, 2) is not None]
    good = [c for c in drawn if adapter.penalty_objectives(c, 2) is None]
    return {"chromosomes": [c.to_dict() for c in [*good[:3], *bad, good[0]]]}


def killed_at(write, problem, **options):
    return Run(problem, die_at_write=write, **options).finish_unless_killed()


def assert_every_death_resumes(problem, *, stride=1, offset=0, finished_meanwhile=False,
                               **options):
    straight = Run(problem, **options).finish()
    expected, total = straight.result(), straight.db.writes
    for write in range(1 + offset, total + 1, stride):
        run = killed_at(write, problem, **options)
        assert run.killed, f"write {write} of {total} was not reached"
        if finished_meanwhile and run.db.generations:
            # The workers go on and the master closes what there is of it.
            run.work()
        run.restart()
        assert run.finish().result() == expected, f"killed at write {write} of {total}"


@pytest.mark.parametrize("finished_meanwhile", [False, True], ids=["open", "closed_by_master"])
@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("strategy", LOOPS, ids=lambda cls: cls.__name__)
def test_killed_at_any_write(strategy, problem, finished_meanwhile):
    assert_every_death_resumes(
        problem, strategy=strategy, stride=STRIDE,
        offset=3 if finished_meanwhile else 0, finished_meanwhile=finished_meanwhile)


@pytest.mark.parametrize("strategy", [NSGA2DeapStrategy, NSGA3PymooStrategy],
                         ids=lambda cls: cls.__name__)
def test_library_variants_killed_at_any_write(strategy):
    assert_every_death_resumes("infeasible", strategy=strategy, stride=STRIDE)


@pytest.mark.parametrize("finished_meanwhile", [False, True], ids=["open", "closed_by_master"])
def test_batch_killed_at_every_write(finished_meanwhile):
    assert_every_death_resumes(
        "infeasible", strategy=BatchStrategy, parameters=batch_parameters(),
        finished_meanwhile=finished_meanwhile)


def cut_mid_enqueue(strategy, **options):
    run = Run("continuous", strategy=strategy, **options)
    run.db.die_while_enqueueing = (2, "individual", 3)
    with pytest.raises(Killed):
        run.finish()
    return run


@pytest.mark.parametrize("strategy", LOOPS, ids=lambda cls: cls.__name__)
def test_killed_again_while_finishing_the_cut(strategy):
    expected = Run("continuous", strategy=strategy).finish().result()
    run = cut_mid_enqueue(strategy)
    for _ in range(3):
        # A few writes further each time, then dead again.
        run.db.die_at_write = run.db.writes + 9
        try:
            run.boot()
        except Killed:
            pass

    run.restart()

    assert run.finish().result() == expected


@pytest.mark.parametrize("strategy", LOOPS, ids=lambda cls: cls.__name__)
def test_a_cut_that_cannot_be_reproduced_is_refused(strategy):
    """The generator changed since: finishing it would mix two searches."""
    run = cut_mid_enqueue(strategy)
    run.db.generations[-1]["rng_state"]["internal"][0] ^= 1
    written = len(run.db.generations), len(run.db.individuals), len(run.db.simulations)

    with pytest.raises(RuntimeError, match="cannot be finished"):
        run.restart()

    assert (len(run.db.generations), len(run.db.individuals), len(run.db.simulations)) == written
    assert "cannot be finished" in run.db.experiment["system_message"]
    assert run.strategy._stop_flag


@pytest.mark.parametrize("strategy", LOOPS, ids=lambda cls: cls.__name__)
def test_a_cut_holding_someone_else_is_refused(strategy):
    run = cut_mid_enqueue(strategy)
    stray = dict(run.db.individuals[-1])
    stray.update(_id=None, individual_id="not-in-this-population")
    run.db.individuals.append(stray)

    with pytest.raises(RuntimeError, match="cannot be finished"):
        run.restart()


@pytest.mark.parametrize("strategy", LOOPS, ids=lambda cls: cls.__name__)
def test_finishing_the_first_generation_keeps_the_start_time(strategy):
    run = Run("continuous", strategy=strategy).until(0, "waiting")
    run.db.generations[0]["enqueue_complete"] = False
    started = run.db.experiment["start_time"]

    run.restart()

    assert run.db.experiment["start_time"] == started
