from unittest.mock import MagicMock, patch
from bson import ObjectId
import pytest

from pylib.db.connection import MongoDBConnection
from pylib.db.repositories.individual import IndividualRepository


def test_pooled_client_survives_queries_and_query_errors_until_shutdown():
    with patch("pylib.db.connection.MongoClient") as constructor:
        connection = MongoDBConnection("mongodb://unused", "test", reuse_client=True)
        with connection.connect() as db:
            assert db is constructor.return_value.__getitem__.return_value
        with pytest.raises(RuntimeError), connection.connect():
            raise RuntimeError("query failure")
        constructor.assert_called_once_with("mongodb://unused")
        constructor.return_value.close.assert_not_called()
        connection.close()
        constructor.return_value.close.assert_called_once()


def test_default_client_lifecycle_is_preserved_for_other_services():
    with patch("pylib.db.connection.MongoClient") as constructor:
        connection = MongoDBConnection("mongodb://unused", "test")
        for _ in range(2):
            with connection.connect():
                pass
        assert constructor.call_count == 2
        assert constructor.return_value.close.call_count == 2


@pytest.mark.parametrize("objectives_only", [False, True])
def test_bulk_query_groups_interleaved_generations_and_projects_metrics(objectives_only):
    connection = MagicMock()
    collection = connection.connect.return_value.__enter__.return_value.__getitem__.return_value
    a, b, exp = ObjectId(), ObjectId(), ObjectId()
    records = [{"generation_id": a, "objectives": [1, 2]},
               {"generation_id": b, "objectives": [3, 4]},
               {"generation_id": a, "objectives": [5, 6]}]
    collection.find.return_value = iter(records)
    repository = IndividualRepository(connection)
    grouped = repository.find_grouped_by_experiment(exp, objectives_only=objectives_only)
    assert grouped == {a: [records[0], records[2]], b: [records[1]]}
    projection = {"_id": 0, "generation_id": 1, "individual_id": 1, "objectives": 1} if objectives_only else None
    collection.find.assert_called_once_with({"experiment_id": exp}, projection)
