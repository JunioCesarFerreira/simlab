"""Cancelled stays cancelled, and stops what was running.

The scalability benchmark never read the status: it ran every wave and then
wrote ``Done`` over ``Cancelled``. The other strategies read it once per
generation, and a cancellation that arrived after that read was overwritten
the same way.
"""
from __future__ import annotations

import random
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.problem.resolve import build_adapter
from lib.strategy.batch import BatchStrategy
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.random_search import RandomSearchStrategy
from lib.strategy.usl_benchmark import UslBenchmarkStrategy
from pylib.db.models.enums import EnumStatus
from tests.resume_harness import PROBLEMS, Run

LOOPS = [NSGA2LoopStrategy, NSGA3LoopStrategy, RandomSearchStrategy]


def batch():
    adapter = build_adapter(
        PROBLEMS["repeated"], {"apply_coverage_repair": False}, random.Random(1))
    return {"chromosomes": [c.to_dict() for c in adapter.random_individual_generator(3)]}


@pytest.mark.parametrize("cls", [*LOOPS, BatchStrategy], ids=lambda cls: cls.__name__)
def test_a_cancel_after_the_handler_read_the_status_is_not_overwritten(cls):
    run = Run("repeated", strategy=cls, parameters=batch() if cls is BatchStrategy else None)
    if cls is not BatchStrategy:
        run.until(4, "waiting")
    run.work()
    read = run.db.experiment_repo.is_cancelled

    def read_then_cancelled(exp_id):
        answer = read(exp_id)
        run.db.experiment["status"] = EnumStatus.CANCELLED
        return answer

    run.db.experiment_repo.is_cancelled = read_then_cancelled
    run.handle()

    assert run.db.experiment["status"] == EnumStatus.CANCELLED


@pytest.mark.parametrize("cls", LOOPS, ids=lambda cls: cls.__name__)
def test_a_resume_does_not_undo_a_cancel(cls):
    run = Run("repeated", strategy=cls).until(1, "waiting")
    run.db.experiment["status"] = EnumStatus.CANCELLED

    run.boot()      # it was Running when the engine listed it

    assert run.db.experiment["status"] == EnumStatus.CANCELLED


@pytest.mark.parametrize(
    "cls", [*LOOPS, BatchStrategy, UslBenchmarkStrategy], ids=lambda cls: cls.__name__)
def test_polling_lets_go_of_a_cancelled_or_deleted_experiment(cls):
    """Without waiting for a generation that, deleted, will never close."""
    strategy = cls.__new__(cls)
    strategy.mongo = MagicMock()
    strategy.mongo.experiment_repo.is_cancelled.return_value = True
    strategy._exp_id = ObjectId()
    strategy._stop_flag = False
    strategy._watch_thread = strategy._sim_watch_thread = None
    strategy._threads = []

    assert strategy._stop_if_cancelled() is True
    assert strategy._stop_flag
