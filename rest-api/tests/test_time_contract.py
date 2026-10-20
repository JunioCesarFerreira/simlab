"""Timestamps leave the API with their timezone, or they are not timestamps.

R25. BSON stores an instant, and PyMongo was handing it back with
``tzinfo=None``. Nothing downstream could then tell UTC from a local reading
of it: the API serialised ``2026-10-15T15:00:00`` with no ``Z``, a browser in
``America/Sao_Paulo`` read that as 18:00 UTC, and a measurement one minute old
displayed as being in the future.

Writing aware UTC was only half of it. These tests cover the other half — what
comes **out** — and the half-fix that made the first one look done:

* the client is built ``tz_aware``, so a read carries its offset;
* the response carries it too, as ``Z`` or ``+00:00``;
* a value stored before any of this existed is read as UTC, which is the
  documented interpretation and not a guess: every process that wrote one ran
  with its clock on UTC.
"""
import json
from datetime import datetime, timedelta, timezone

from fastapi.encoders import jsonable_encoder

from pylib.db.connection import TZ_AWARE
from pylib.timeutil import as_aware, elapsed_since, utc_now


class TestTheClientDeclaresItsContract:
    def test_the_connection_is_tz_aware(self):
        """A read-side setting, and the one that makes every collection's
        timestamps unambiguous at once."""
        assert TZ_AWARE is True

    def test_the_client_is_constructed_with_it(self):
        from unittest.mock import patch

        from pylib.db.connection import MongoDBConnection

        with patch("pylib.db.connection.MongoClient") as constructor:
            connection = MongoDBConnection("mongodb://unused", "db")
            with connection.connect():
                pass

        assert constructor.call_args.kwargs["tz_aware"] is True


class TestWhatTheResponseCarries:
    """FastAPI's encoder is the thing that actually writes the string."""

    def test_an_aware_value_serialises_with_its_offset(self):
        measured = datetime(2026, 10, 15, 15, 0, 0, tzinfo=timezone.utc)

        encoded = json.loads(json.dumps(jsonable_encoder({"checked_at": measured})))

        assert encoded["checked_at"].endswith("+00:00") or \
            encoded["checked_at"].endswith("Z"), encoded
        # The property that matters: a reader in any timezone resolves it to
        # the same instant.
        assert datetime.fromisoformat(
            encoded["checked_at"].replace("Z", "+00:00")) == measured

    def test_a_naive_value_serialises_without_one(self):
        """The defect, stated as a test. This is what the API used to send,
        and why a browser three hours behind read it as being in the future.
        """
        encoded = jsonable_encoder({"checked_at": datetime(2026, 10, 15, 15, 0)})

        assert not encoded["checked_at"].endswith("+00:00")
        assert not encoded["checked_at"].endswith("Z")

    def test_what_the_api_writes_now_is_aware(self):
        assert utc_now().tzinfo is not None
        assert utc_now().utcoffset() == timedelta(0)


class TestHistoricalValues:
    """The documented interpretation, not a guess.

    Every process that wrote a naive timestamp ran in a container with its
    clock on UTC, so reading one as UTC is correct for the data that exists.
    It is **not** correct for a value written by a process on another clock,
    and nothing can recover that — which is why the interpretation is written
    down rather than inferred.
    """

    def test_a_naive_value_is_read_as_utc(self):
        naive = datetime(2026, 10, 15, 15, 0, 0)

        assert as_aware(naive) == datetime(
            2026, 10, 15, 15, 0, 0, tzinfo=timezone.utc)

    def test_an_aware_value_is_left_alone(self):
        aware = datetime(2026, 10, 15, 15, 0, 0, tzinfo=timezone.utc)

        assert as_aware(aware) is aware

    def test_an_age_can_be_taken_across_the_two(self):
        """Mixed documents are the normal case during a migration, and a
        ``TypeError`` here would land in the health cycle."""
        naive = datetime(2026, 10, 15, 15, 0, 0)
        now = datetime(2026, 10, 15, 15, 1, 0, tzinfo=timezone.utc)

        assert elapsed_since(naive, now=now) == 60.0
