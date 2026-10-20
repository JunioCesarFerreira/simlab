"""Timestamps that survive a partial migration.

The collections this feature introduced store aware UTC; the rest of the
repository still writes naive values, and documents written before the change
hold naive ones too. Arithmetic mixing the two raises ``TypeError`` — on paths
that usually only run when something has already gone wrong, which is the
worst place to find out.
"""
from datetime import datetime, timedelta, timezone

from pylib.timeutil import as_aware, elapsed_since, utc_now


class TestUtcNow:
    def test_it_is_aware(self):
        assert utc_now().tzinfo is not None

    def test_it_is_utc(self):
        assert utc_now().utcoffset() == timedelta(0)


class TestAsAware:
    def test_a_naive_value_is_read_as_utc(self):
        """What it already was wherever these services run.

        Recording the assumption is what keeps documents written before the
        migration comparable with the ones written after.
        """
        naive = datetime(2026, 10, 15, 12, 0, 0)

        aware = as_aware(naive)

        assert aware.tzinfo is not None
        assert aware == datetime(2026, 10, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_an_aware_value_is_left_alone(self):
        original = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
        assert as_aware(original) is original


class TestElapsedSince:
    def test_it_measures_a_naive_timestamp(self):
        """A naive value as this system stores one: UTC, without saying so.

        Deliberately not `datetime.now()`, which on a machine outside UTC is
        local time — and a naive local timestamp read as UTC is wrong by the
        offset. That ambiguity is what the migration removes; here it is
        pinned so the assumption is visible rather than assumed.
        """
        then = utc_now().replace(tzinfo=None) - timedelta(seconds=30)

        elapsed = elapsed_since(then)

        assert elapsed is not None and 29 <= elapsed <= 31

    def test_a_naive_local_timestamp_is_off_by_the_offset(self):
        """The limit of what can be recovered from data already written.

        Stated as a test so nobody later reads `as_aware` as a conversion: it
        is an assumption about who wrote the value, and it only holds where
        the writer's clock is UTC.
        """
        local_naive = datetime.now()
        offset = datetime.now(timezone.utc).astimezone().utcoffset()
        assert offset is not None

        elapsed = elapsed_since(local_naive)

        assert elapsed is not None
        assert abs(elapsed + offset.total_seconds()) < 2

    def test_it_measures_an_aware_timestamp(self):
        then = utc_now() - timedelta(seconds=30)
        elapsed = elapsed_since(then)
        assert elapsed is not None and 29 <= elapsed <= 31

    def test_the_two_kinds_agree(self):
        """The whole point: a caller must not have to know which it got."""
        instant = datetime(2026, 10, 15, 12, 0, 0)
        reference = datetime(2026, 10, 15, 12, 1, 0, tzinfo=timezone.utc)

        from_naive = elapsed_since(instant, now=reference)
        from_aware = elapsed_since(
            instant.replace(tzinfo=timezone.utc), now=reference)

        assert from_naive == from_aware == 60.0

    def test_something_that_is_not_a_timestamp_is_not_a_crash(self):
        # These paths run when something has already gone wrong; raising here
        # would replace a diagnosable state with a stack trace.
        for value in (None, "", 0, "2026-10-15"):
            assert elapsed_since(value) is None

    def test_a_future_timestamp_reads_negative_rather_than_absolute(self):
        # A clock that moved should be visible as such, not hidden.
        ahead = utc_now() + timedelta(seconds=60)
        elapsed = elapsed_since(ahead)
        assert elapsed is not None and elapsed < 0
