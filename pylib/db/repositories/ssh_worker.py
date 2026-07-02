import logging
from bson import ObjectId

from pylib.db.models.ssh_worker import SshWorker
from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)

COLLECTION = "ssh_workers"


class SshWorkerRepository:
    """
    Manages the list of SSH worker connections used by master-node to dispatch simulations.
    Each worker represents one Cooja container (local or remote machine).
    """

    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def find_enabled(self) -> list[SshWorker]:
        """Return all enabled worker connections."""
        with self.connection.connect() as db:
            return list(db[COLLECTION].find({"enabled": True}))

    def count(self) -> int:
        with self.connection.connect() as db:
            return db[COLLECTION].count_documents({})

    def insert_many(self, workers: list[SshWorker]) -> list[ObjectId]:
        with self.connection.connect() as db:
            result = db[COLLECTION].insert_many(workers)
            return result.inserted_ids
