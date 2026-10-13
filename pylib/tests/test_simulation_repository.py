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

from pylib.db.repositories.simulation import SimulationRepository

EXP_ID = ObjectId("507f1f77bcf86cd799440001")
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
