"""At boot, each experiment gets exactly one strategy, and none is missed.

The engine used to claim what was ``Waiting`` (which makes it ``Running``) and
only then look for what a previous process had left ``Running``: it found what
it had just started and resumed it, so two strategies drove one experiment. It
also scanned before listening, so an experiment created in between was seen by
neither and waited until the next restart.

The real ``main()`` runs. The database and the change stream are faked, and
threads run inline so that every interleaving here is deterministic.
"""
from __future__ import annotations

import importlib
import sys
from collections import Counter
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

import pylib.db
from pylib.db.models.enums import EnumStatus


class Booted(Exception):
    """Where the main thread would go to sleep for ever."""


class _Experiments:
    """The repository methods the engine calls, plus what MongoDB delivers."""

    def __init__(self):
        self.connection = MagicMock()
        self.documents: dict[ObjectId, dict] = {}
        self.events: list[dict] = []
        self.resumed: list[str] = []
        self.listening = False
        self.streams = 1
        #: Name of a moment at boot -> what another client does right then.
        self.meanwhile: dict[str, object] = {}

    def _moment(self, name):
        action = self.meanwhile.pop(name, None)
        if action:
            action()

    def add(self, status=EnumStatus.WAITING) -> str:
        document = {"_id": ObjectId(), "status": status, "parameters": {}}
        self.documents[document["_id"]] = document
        if self.listening and status == EnumStatus.WAITING:
            self.events.append({"fullDocument": dict(document)})
        return str(document["_id"])

    def status(self, exp_id: str) -> str:
        return self.documents[ObjectId(exp_id)]["status"]

    def find_startable_by_status(self, status):
        found = [dict(d) for d in self.documents.values() if d["status"] == status]
        self._moment(f"after scanning {status}")
        return found

    def update_starting(self, exp_id):
        document = self.documents[ObjectId(exp_id)]
        if document["status"] != EnumStatus.WAITING:
            return False
        document["status"] = EnumStatus.RUNNING
        return True

    def claim_resume(self, exp_id):
        if self.status(exp_id) != EnumStatus.RUNNING:
            return False
        self.resumed.append(exp_id)
        return True

    def update_unless_cancelled(self, exp_id, fields):
        if self.status(exp_id) == EnumStatus.CANCELLED:
            return False
        self.documents[ObjectId(exp_id)].update(fields)
        return True

    def update_status(self, exp_id, status):
        self.documents[ObjectId(exp_id)]["status"] = status
        return True

    def watch_status_waiting(self, on_change, on_open=None):
        for _ in range(self.streams):
            self._moment("before the stream opens")
            # Like MongoDB, a stream delivers only what happens after it opens.
            self.listening = True
            if on_open:
                on_open()
            self._moment("stream open")
            while self.events:
                on_change(self.events.pop(0))
            self.listening = False
            self._moment("stream down")


class _InlineThread:
    def __init__(self, target, args=(), kwargs=None, **_options):
        self._run = lambda: target(*args, **(kwargs or {}))

    def start(self):
        self._run()


@pytest.fixture
def engine(monkeypatch):
    # engine.py connects when imported.
    monkeypatch.setattr(pylib.db, "create_mongo_repository_factory", lambda *_: MagicMock())
    monkeypatch.delitem(sys.modules, "engine", raising=False)
    module = importlib.import_module("engine")

    mongo = MagicMock()
    mongo.experiment_repo = _Experiments()
    monkeypatch.setattr(module, "mongo", mongo)
    monkeypatch.setattr(module, "Thread", _InlineThread)
    monkeypatch.setattr(module, "start_runtime_metrics_watcher", lambda _mongo: None)
    monkeypatch.setattr(module, "capture_firmware_snapshot", lambda *_: None)

    def sleep(_seconds):
        raise Booted

    monkeypatch.setattr(module.time, "sleep", sleep)

    module.started = Counter()
    module.failing = set()

    def select_strategy(document):
        exp_id = str(document["_id"])
        module.started[exp_id] += 1
        if exp_id in module.failing:
            raise RuntimeError("the strategy cannot start")
        return MagicMock()

    monkeypatch.setattr(module, "select_strategy", select_strategy)

    def boot():
        with pytest.raises(Booted):
            module.main()

    module.boot = boot
    module.experiments = mongo.experiment_repo
    return module


def test_waiting_at_boot_is_started_once_and_never_resumed(engine):
    waiting = [engine.experiments.add() for _ in range(3)]

    engine.boot()

    assert engine.started == Counter(waiting)
    assert engine.experiments.resumed == []


def test_running_at_boot_is_resumed_once(engine):
    running = [engine.experiments.add(EnumStatus.RUNNING) for _ in range(3)]

    engine.boot()

    assert engine.started == Counter(running)
    assert sorted(engine.experiments.resumed) == sorted(running)


def test_only_what_was_running_before_boot_is_resumed(engine):
    waiting = engine.experiments.add()
    running = engine.experiments.add(EnumStatus.RUNNING)
    for status in (EnumStatus.DONE, EnumStatus.ERROR, EnumStatus.CANCELLED):
        engine.experiments.add(status)

    engine.boot()

    assert engine.started == Counter([waiting, running])
    assert engine.experiments.resumed == [running]


@pytest.mark.parametrize("moment", [
    f"after scanning {EnumStatus.RUNNING}",
    "before the stream opens",
    "stream open",
])
def test_created_while_the_engine_boots_is_started_once(engine, moment):
    created = []
    engine.experiments.meanwhile[moment] = lambda: created.append(engine.experiments.add())

    engine.boot()

    assert created
    assert engine.started == Counter(created)
    assert engine.experiments.resumed == []


def test_created_during_the_scan_is_started_once_by_its_event(engine):
    """Seen by the stream that was already open, and possibly by the scan too."""
    created = []
    engine.experiments.meanwhile[f"after scanning {EnumStatus.WAITING}"] = (
        lambda: created.append(engine.experiments.add()))

    engine.boot()

    assert engine.started == Counter(created)


def test_seen_by_the_scan_and_by_the_stream_is_started_once(engine):
    waiting = engine.experiments.add()
    document = engine.experiments.documents[ObjectId(waiting)]
    engine.experiments.events.append({"fullDocument": dict(document)})

    engine.boot()

    assert engine.started == Counter([waiting])


def test_created_while_the_stream_was_down_is_started_when_it_reopens(engine):
    created = []
    engine.experiments.streams = 2
    engine.experiments.meanwhile["stream down"] = (
        lambda: created.append(engine.experiments.add()))

    engine.boot()

    assert engine.started == Counter(created)


def test_a_reopened_stream_does_not_resume_what_this_engine_runs(engine):
    waiting = engine.experiments.add()
    running = engine.experiments.add(EnumStatus.RUNNING)
    engine.experiments.streams = 3

    engine.boot()

    assert engine.started == Counter([waiting, running])
    assert engine.experiments.resumed == [running]


def test_cancelled_between_the_scan_and_the_claim_is_not_started(engine):
    waiting = engine.experiments.add()
    engine.experiments.meanwhile[f"after scanning {EnumStatus.WAITING}"] = (
        lambda: engine.experiments.update_status(waiting, EnumStatus.CANCELLED))

    engine.boot()

    assert not engine.started


@pytest.mark.parametrize("status", [EnumStatus.WAITING, EnumStatus.RUNNING])
def test_a_failed_start_ends_in_error_without_stopping_the_others(engine, status):
    failing = engine.experiments.add(status)
    other = engine.experiments.add(status)
    engine.failing.add(failing)

    engine.boot()

    assert engine.started == Counter([failing, other])
    assert engine.experiments.status(failing) == EnumStatus.ERROR
    assert engine.experiments.status(other) == EnumStatus.RUNNING
