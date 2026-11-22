"""A resumed random search is the run that was never interrupted.

Random search used to resume with an empty population, an empty archive, the
index of the last generation instead of the next one, and the random stream of
generation 0. Every test here kills the engine at one point of the loop, starts
a new strategy over what was persisted, and compares the result with a run that
nobody killed.

What is real and what is replaced is in ``tests/resume_harness.py``.
"""
from __future__ import annotations

import math

import pytest
from bson import ObjectId

from lib.strategy import random_search
from lib.strategy.random_search import WORST_OBJECTIVE, RandomSearchStrategy
from pylib.db.models.enums import EnumStatus
from tests.resume_harness import GENERATIONS, KEYS, PROBLEMS, Killed, Run, stored


@pytest.mark.parametrize("point", ["waiting", "closed", "evaluating", "evaluated"])
@pytest.mark.parametrize("checkpoint", [0, 2, GENERATIONS])
@pytest.mark.parametrize("problem", list(PROBLEMS))
def test_resumed_run_matches_the_uninterrupted_one(problem, checkpoint, point):
    straight = Run(problem).finish()
    interrupted = Run(problem).until(checkpoint, point)

    interrupted.restart()

    assert interrupted.finish().result() == straight.result()


def test_scenarios_exercise_penalty_and_reuse():
    """Without this the parametrised test could compare trivial runs."""
    infeasible = Run("infeasible").finish().db.individuals
    states = {(i.get("completeness") or {}).get("state") for i in infeasible}
    assert {"infeasible", "complete"} <= states

    repeated = Run("repeated").finish().db.individuals
    assert any((i.get("completeness") or {}).get("reused") for i in repeated)


def test_several_restarts_in_one_run():
    straight = Run("infeasible").finish()
    run = Run("infeasible")
    for checkpoint in range(GENERATIONS + 1):
        while run.last["index"] < checkpoint:
            run.work()
            run.handle()
        run.restart()

    assert run.finish().result() == straight.result()


@pytest.mark.parametrize("point", ["waiting", "closed", "evaluating", "evaluated"])
@pytest.mark.parametrize("outcome", ["failed_replica", "not_finite"])
def test_a_refused_evaluation_stays_refused(outcome, point):
    """A placeholder of 1e12 must not come back from a restart as a measurement."""
    def deliver(digest, metrics):
        if digest[2] < 90:
            if outcome == "failed_replica":
                return EnumStatus.ERROR, {}
            return EnumStatus.DONE, {**metrics, KEYS[0]: math.nan}
        return EnumStatus.DONE, metrics

    straight = Run("repeated", outcome=deliver).finish()
    refused = [
        i for i in straight.db.individuals
        if WORST_OBJECTIVE in [abs(v) for v in i["objectives"]]
    ]
    assert refused
    assert all(not i["completeness"]["objective_used"] for i in refused)

    run = Run("repeated", outcome=deliver).until(0, point)
    run.restart()

    assert run.finish().result() == straight.result()


def test_a_generation_refused_for_a_missing_aggregate_stays_refused():
    """Accepted replicas with no finite value: refused by the strategy itself."""
    def deliver(digest, metrics):
        return EnumStatus.DONE, {**metrics, KEYS[0]: math.nan}

    simulation = {"incomplete_objective": "accept"}
    straight = Run("repeated", simulation=simulation, outcome=deliver).finish()
    run = Run("repeated", simulation=simulation, outcome=deliver).until(0, "evaluated")
    run.restart()

    assert run.finish().result() == straight.result()


def test_drawn_simulation_seeds_are_the_same_after_a_restart():
    simulation = {"random_seeds": [], "random_seeds_count": 3}
    straight = Run("repeated", simulation=simulation).finish()
    run = Run("repeated", simulation=simulation).until(1, "waiting")
    run.restart()

    assert run.finish().result() == straight.result()


@pytest.mark.parametrize("checkpoint", [0, 2])
def test_checkpoint_without_a_snapshot_continues_the_stream(checkpoint, caplog):
    """Written before the fix: no RNG state and no population index."""
    straight = Run("infeasible").finish()
    run = Run("infeasible").until(checkpoint, "closed")
    for generation in run.db.generations:
        generation.pop("rng_state", None)
    for individual in run.db.individuals:
        individual.pop("index", None)

    with caplog.at_level("WARNING"):
        run.restart()

    assert "RNG snapshot" in caplog.text
    assert run.finish().result() == straight.result()


def test_generation_that_closes_while_the_watcher_subscribes_is_processed():
    run = Run("repeated").until(1, "waiting")
    strategy = RandomSearchStrategy(stored(run.db.experiment), run.db)
    strategy._upload_topology_async = lambda *_a, **_kw: None
    strategy._fire_generation_done = lambda *_a, **_kw: None
    strategy._start_generation_poll = lambda: None
    # It ends after the generations were read and before anyone is listening.
    strategy._start_watcher = run.work

    strategy.start()

    assert run.last["index"] == 2


def test_resume_refuses_a_generation_without_individuals():
    run = Run("repeated").until(1, "waiting")
    last = run.last["_id"]
    run.db.individuals = [i for i in run.db.individuals if i["generation_id"] != last]

    with pytest.raises(RuntimeError, match="no individuals"):
        run.restart()


def test_a_firmware_run_queues_the_restored_genomes_firmware(monkeypatch):
    """The restored population is what maps a genome to its firmware."""
    monkeypatch.setattr(random_search, "create_files", lambda *_a, **_kw: {
        "csc_file_id": ObjectId(), "pos_file_id": ObjectId(),
    })
    simulation = {"synthetic": {"enabled": False}}
    run = Run("repeated", simulation=simulation).until(1, "waiting")
    run.restart()
    run.finish()

    source = run.db.experiment["source_repository_options"]["csma"]
    assert {s["source_repository_id"] for s in run.db.simulations} == {source}


def test_a_generation_with_nothing_to_simulate_is_closed_on_resume():
    """Enqueued whole, killed before the engine could close it: no worker will."""
    straight = Run("analytical").finish()
    run = Run("analytical")
    run.db.die_while_enqueueing = (2, "close", 0)
    with pytest.raises(Killed):
        run.finish()

    run.restart()

    # Closed and processed by the engine: nothing else would have closed it.
    assert run.last["index"] == 3
    assert run.finish().result() == straight.result()
