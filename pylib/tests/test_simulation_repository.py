"""Tests for SimulationRepository query scoping.

``individual_id`` is a chromosome hash, not an identity. The same chromosome
is produced again by later experiments, and every run of it accumulates under
that one id, so a lookup by individual alone answers with simulations the
caller never asked about. One pilot's individual panel showed nine simulations
where two belonged to the experiment on screen.

The connection is mocked in the style of ``test_infra_repository.py``: what
these assert is the *filter sent to MongoDB*, which is where the scoping
actually lives.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.simulation import SimulationRepository

EXP_ID = ObjectId("507f1f77bcf86cd799440001")
SIM_OID = ObjectId("507f1f77bcf86cd799440002")
ATTEMPT = "8f14e45fceea167a5a36dedd4bea2543"
IND_HASH = "aa48834d584df6e66a035e491c5ad291136c25fc"


class FakeDb:
    """Minimal ``db[name]`` mapping over per-collection MagicMocks."""

    def __init__(self):
        self.collections: dict[str, MagicMock] = {}

    def __getitem__(self, name: str) -> MagicMock:
        return self.collections.setdefault(name, MagicMock())


def make_connection() -> tuple[MagicMock, FakeDb]:
    db = FakeDb()
    connection = MagicMock()

    @contextmanager
    def connect():
        yield db

    connection.connect = connect
    return connection, db


@pytest.fixture
def simulations() -> tuple[SimulationRepository, MagicMock]:
    connection, db = make_connection()
    repo = SimulationRepository(connection)
    coll = db["simulations"]
    coll.reset_mock()                       # drop the index calls from __init__
    return repo, coll


class TestFindByIndividual:
    def test_scopes_by_experiment_when_given(self, simulations):
        repo, coll = simulations
        coll.find.return_value = []

        repo.find_by_individual(IND_HASH, EXP_ID)

        coll.find.assert_called_once_with(
            {"individual_id": IND_HASH, "experiment_id": EXP_ID}
        )

    def test_experiment_id_is_matched_as_objectid(self, simulations):
        """A string would silently match nothing: the field is an ObjectId."""
        repo, coll = simulations
        coll.find.return_value = []

        repo.find_by_individual(IND_HASH, EXP_ID)

        query = coll.find.call_args.args[0]
        assert isinstance(query["experiment_id"], ObjectId)

    def test_without_experiment_the_filter_is_the_hash_alone(self, simulations):
        """Kept working so an existing caller cannot break silently.

        It is the unscoped behaviour that produced the duplicated listing, so
        the omission has to be visible in the filter rather than guessed at.
        """
        repo, coll = simulations
        coll.find.return_value = []

        repo.find_by_individual(IND_HASH)

        coll.find.assert_called_once_with({"individual_id": IND_HASH})

    def test_returns_the_documents_found(self, simulations):
        repo, coll = simulations
        docs = [{"_id": ObjectId(), "individual_id": IND_HASH, "random_seed": 123456},
                {"_id": ObjectId(), "individual_id": IND_HASH, "random_seed": 336157}]
        coll.find.return_value = iter(docs)

        assert repo.find_by_individual(IND_HASH, EXP_ID) == docs

    def test_seeds_of_the_same_experiment_are_both_kept(self, simulations):
        """Two runs of one individual under different seeds are not duplicates.

        The pilot ran each individual twice by design. Scoping must cut the
        other experiments' runs and leave these.
        """
        repo, coll = simulations
        coll.find.return_value = [
            {"individual_id": IND_HASH, "experiment_id": EXP_ID, "random_seed": 123456},
            {"individual_id": IND_HASH, "experiment_id": EXP_ID, "random_seed": 336157},
        ]

        found = repo.find_by_individual(IND_HASH, EXP_ID)

        assert [d["random_seed"] for d in found] == [123456, 336157]


class TestAttemptGuards:
    """Every write about a simulation says which attempt it belongs to.

    An attempt can lose its container without knowing: a recovery pass or the
    stale-reservation sweep may have handed it on. Only ``set_remote_pid`` was
    guarded, so the thread that lost the race still marked the run started,
    published its result over the newer attempt, or failed a simulation that
    had already moved on. The keyword is required of every caller — ``None``
    is how a recovery path says "unconditional, and I mean it".
    """

    def test_running_is_guarded(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        assert repo.mark_running(SIM_OID, attempt_id=ATTEMPT) is True

        assert coll.update_one.call_args.args[0] == {
            "_id": SIM_OID, "attempt_id": ATTEMPT}

    def test_done_is_guarded(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_done(SIM_OID, None, None, {}, attempt_id=ATTEMPT)

        assert coll.update_one.call_args.args[0]["attempt_id"] == ATTEMPT

    def test_error_is_guarded(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_error(SIM_OID, "boom", attempt_id=ATTEMPT)

        assert coll.update_one.call_args.args[0]["attempt_id"] == ATTEMPT

    def test_a_refused_write_is_reported(self, simulations):
        """False means "not yours any more", and the caller has to act on it."""
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=0)

        assert repo.mark_done(SIM_OID, None, None, {}, attempt_id=ATTEMPT) is False
        assert repo.mark_error(SIM_OID, "boom", attempt_id=ATTEMPT) is False
        assert repo.mark_running(SIM_OID, attempt_id=ATTEMPT) is False

    def test_none_is_unconditional_for_recovery(self, simulations):
        # Recovery and the synthetic path pass None deliberately: there is no
        # attempt, or the one that existed is known to be gone.
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_error(SIM_OID, "recovered", attempt_id=None)

        assert coll.update_one.call_args.args[0] == {"_id": SIM_OID}


class TestRequeue:
    """Losing the container is not the simulation's fault.

    The work was sound, so the simulation goes back to Waiting and the
    dispatcher places it again — rather than being marked in error for a race
    it had no part in.
    """

    def test_it_returns_to_waiting_and_loses_its_placement(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.requeue(SIM_OID, ATTEMPT, "taken over") is True

        sets = coll.update_one.call_args.args[1]["$set"]
        assert sets["status"] == EnumStatus.WAITING
        for field in ("attempt_id", "remote_pid", "assigned_resource_id",
                      "assigned_host_id", "start_time", "end_time"):
            assert sets[field] is None, field

    def test_it_is_guarded_by_the_attempt(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.requeue(SIM_OID, ATTEMPT, "taken over")

        assert coll.update_one.call_args.args[0] == {
            "_id": SIM_OID, "attempt_id": ATTEMPT}

    def test_a_simulation_a_newer_attempt_owns_is_not_dragged_back(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.requeue(SIM_OID, ATTEMPT, "taken over") is False

    def test_the_reason_is_kept_for_the_operator(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.requeue(SIM_OID, ATTEMPT, "the reservation is gone")

        assert "reservation" in \
            coll.update_one.call_args.args[1]["$set"]["system_message"]
