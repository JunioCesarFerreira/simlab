"""Pictures are drawn by a few threads, however many are asked for.

There was one thread per individual: 80 experiments put some 1,500 renders in
memory at once and took the engine to 4 GB, whatever it released afterwards.
"""
import threading
import time

import pytest

from lib.strategy.batch import BatchStrategy
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.random_search import RandomSearchStrategy
from lib.util import background


def test_a_burst_runs_a_few_at_a_time_and_all_of_it():
    lock, finished = threading.Lock(), threading.Event()
    running, most, done = [0], [0], [0]

    def work():
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.002)
        with lock:
            running[0] -= 1
            done[0] += 1
            if done[0] == 200:
                finished.set()

    for _ in range(200):
        background.submit(work)

    assert finished.wait(10)
    assert most[0] <= background.WORKERS


def test_a_failure_does_not_take_its_worker_with_it():
    for _ in range(background.WORKERS * 2):
        background.submit(lambda: 1 / 0)
    after = threading.Event()

    background.submit(after.set)

    assert after.wait(5)


@pytest.mark.parametrize(
    "cls", [NSGA2LoopStrategy, NSGA3LoopStrategy, RandomSearchStrategy, BatchStrategy],
    ids=lambda cls: cls.__name__)
def test_every_strategy_queues_its_pictures(cls, monkeypatch):
    queued = []
    monkeypatch.setattr(background, "submit", queued.append)
    strategy = cls.__new__(cls)
    before = threading.active_count()

    for index in range(50):
        strategy._upload_topology_async(None, None, 0, index, f"genome-{index}", {})

    assert len(queued) == 50
    assert threading.active_count() <= before
