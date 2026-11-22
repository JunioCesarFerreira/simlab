"""``on_open`` runs once a change stream is listening, and before its events.

That order is what lets a caller scan for what was already there without a
gap: whatever the scan is too early to see, the stream delivers.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from pymongo.errors import PyMongoError

from pylib.db import connection as module
from pylib.db.connection import MongoDBConnection
from pylib.db.repositories.experiment import ExperimentRepository


class Stop(Exception):
    """Ends the endless watch loop."""


def _watching(monkeypatch, streams):
    """A connection whose successive ``watch()`` calls yield ``streams``."""
    log, pending = [], list(streams)

    class _Collection:
        def watch(self, _pipeline, **options):
            if not pending:
                raise Stop
            log.append(("open", options.get("resume_after")))
            events = pending.pop(0)

            @contextmanager
            def stream():
                yield deliver(events)

            return stream()

    def deliver(events):
        for event in events:
            if isinstance(event, Exception):
                raise event
            yield event

    @contextmanager
    def connect():
        yield {"things": _Collection()}

    conn = MongoDBConnection("mongodb://unused", "test")
    monkeypatch.setattr(conn, "connect", connect)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    return conn, log


def test_on_open_runs_after_the_stream_opens_and_before_its_events(monkeypatch):
    conn, log = _watching(monkeypatch, [[{"_id": "a"}, PyMongoError("lost")]])

    with pytest.raises(Stop):
        conn.watch_collection(
            "things", [], lambda change: log.append(("event", change["_id"])),
            on_open=lambda: log.append(("scan", None)))

    assert log == [("open", None), ("scan", None), ("event", "a")]


def test_on_open_runs_again_on_every_reconnection(monkeypatch):
    conn, log = _watching(
        monkeypatch, [[PyMongoError("lost")], [{"_id": "b"}, PyMongoError("lost")]])

    with pytest.raises(Stop):
        conn.watch_collection(
            "things", [], lambda change: log.append(("event", change["_id"])),
            on_open=lambda: log.append(("scan", None)))

    assert log == [
        ("open", None), ("scan", None),
        ("open", None), ("scan", None), ("event", "b"),
    ]


def test_a_database_error_in_on_open_reopens_the_stream(monkeypatch):
    conn, log = _watching(monkeypatch, [[], [{"_id": "c"}, PyMongoError("lost")]])
    scans = []

    def scan():
        scans.append(1)
        if len(scans) == 1:
            raise PyMongoError("not reachable")

    with pytest.raises(Stop):
        conn.watch_collection(
            "things", [], lambda change: log.append(("event", change["_id"])),
            on_open=scan)

    assert log == [("open", None), ("open", None), ("event", "c")]
    assert len(scans) == 2


def test_without_on_open_nothing_changes(monkeypatch):
    conn, log = _watching(monkeypatch, [[{"_id": "d"}, PyMongoError("lost")]])

    with pytest.raises(Stop):
        conn.watch_collection(
            "things", [], lambda change: log.append(("event", change["_id"])))

    assert log == [("open", None), ("event", "d")]


def test_the_experiment_watch_hands_on_open_to_the_stream():
    repository = ExperimentRepository(MagicMock())
    on_open = MagicMock()

    repository.watch_status_waiting(MagicMock(), on_open=on_open)

    assert repository.connection.watch_collection.call_args.kwargs["on_open"] is on_open


class _IdleStream:
    """An open stream: ``try_next`` gives ``None`` while nothing happens."""

    alive = True

    def __init__(self, events=()):
        self.events = list(events)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def try_next(self):
        return self.events.pop(0) if self.events else None


def _idle(monkeypatch, stream, failure=None):
    options = []

    class _Collection:
        def watch(self, _pipeline, **kwargs):
            options.append(kwargs)
            if failure:
                raise failure
            return stream

    @contextmanager
    def connect():
        yield {"things": _Collection()}

    conn = MongoDBConnection("mongodb://unused", "test")
    monkeypatch.setattr(conn, "connect", connect)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    return conn, options


def test_an_idle_stream_ends_when_told_to(monkeypatch):
    stream = _IdleStream()
    conn, options = _idle(monkeypatch, stream)
    asked = []

    conn.watch_collection(
        "things", [], MagicMock(), should_stop=lambda: asked.append(1) or len(asked) > 3)

    assert stream.closed
    # The server answers an idle getMore within this, so the question is asked.
    assert 0 < options[0]["max_await_time_ms"] <= 5000


def test_events_are_delivered_until_then(monkeypatch):
    stream = _IdleStream([{"_id": "a"}, {"_id": "b"}])
    conn, _ = _idle(monkeypatch, stream)
    seen = []

    conn.watch_collection("things", [], seen.append, should_stop=lambda: len(seen) == 2)

    assert [change["_id"] for change in seen] == ["a", "b"]


def test_a_stream_already_stopped_is_not_opened(monkeypatch):
    conn, options = _idle(monkeypatch, _IdleStream())

    conn.watch_collection("things", [], MagicMock(), should_stop=lambda: True)

    assert options == []


def test_stopping_ends_the_retries_too(monkeypatch):
    conn, options = _idle(monkeypatch, None, failure=PyMongoError("no primary"))

    conn.watch_collection("things", [], MagicMock(), should_stop=lambda: len(options) >= 3)

    assert len(options) == 3
