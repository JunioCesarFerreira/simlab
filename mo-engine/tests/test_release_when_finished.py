"""A strategy that stopped gives back its threads and its change streams.

Nothing ended them: ``watch_collection`` loops for ever, so every finished
experiment left about 15 threads, 7 connections and 50–100 MB in the engine,
which ran out of 4 GB after some 70 experiments.
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

import pylib.db
from lib.strategy.batch import BatchStrategy
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.random_search import RandomSearchStrategy
from lib.strategy.usl_benchmark import UslBenchmarkStrategy
from tests.resume_harness import Run

STRATEGIES = [NSGA2LoopStrategy, NSGA3LoopStrategy, RandomSearchStrategy,
              BatchStrategy, UslBenchmarkStrategy]


class _Streams:
    """A change stream with no events: it only ends if it is told to."""

    def __init__(self):
        self.open = 0
        self._lock = threading.Lock()
        self.released = threading.Event()

    def watch_status_terminal(self, _on_change, should_stop=None):
        with self._lock:
            self.open += 1
        while not (should_stop and should_stop()) and not self.released.is_set():
            time.sleep(0.01)
        with self._lock:
            self.open -= 1


@pytest.fixture
def streams():
    fake = _Streams()
    yield fake
    fake.released.set()


def watching(cls, streams):
    strategy = cls.__new__(cls)
    strategy.mongo = SimpleNamespace(generation_repo=streams, simulation_repo=streams)
    strategy._exp_id = ObjectId()
    strategy._stop_flag = False
    strategy._watch_thread = strategy._sim_watch_thread = None
    strategy._threads = []
    strategy._current_generation = None
    strategy._generation_id = None
    start = getattr(strategy, "_start_watcher", None) or strategy._start_watchers
    return strategy, start


def eventually(condition, seconds=3.0):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


@pytest.mark.parametrize("cls", STRATEGIES, ids=lambda cls: cls.__name__)
def test_stop_closes_the_change_streams(cls, streams):
    strategy, start = watching(cls, streams)
    start()
    assert eventually(lambda: streams.open > 0)

    strategy.stop()

    assert eventually(lambda: streams.open == 0), f"{streams.open} stream(s) left open"


@pytest.mark.parametrize("cls", STRATEGIES, ids=lambda cls: cls.__name__)
def test_a_run_that_finished_before_watching_stays_stopped(cls, streams):
    """With nothing to simulate, a run can end before ``start`` gets to watch."""
    strategy, start = watching(cls, streams)
    strategy.stop()

    start()

    assert strategy._stop_flag
    assert eventually(lambda: streams.open == 0)


@pytest.mark.parametrize("cls", STRATEGIES[:3], ids=lambda cls: cls.__name__)
def test_finishing_stops_the_strategy(cls):
    assert Run("analytical", strategy=cls).finish().strategy._stop_flag


def test_a_start_that_failed_is_stopped(monkeypatch):
    monkeypatch.setattr(pylib.db, "create_mongo_repository_factory", lambda *_: MagicMock())
    monkeypatch.delitem(sys.modules, "engine", raising=False)
    engine = importlib.import_module("engine")
    strategy = MagicMock()
    strategy.start.side_effect = RuntimeError("after its watchers had started")
    monkeypatch.setattr(engine, "capture_firmware_snapshot", lambda *_: None)
    monkeypatch.setattr(engine, "select_strategy", lambda _doc: strategy)

    assert engine.process_experiment({"_id": ObjectId()}) is False

    strategy.stop.assert_called_once()
