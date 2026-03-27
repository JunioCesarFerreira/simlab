import logging
from typing import Any
from bson import ObjectId

from pylib.db.models.genome_cache import GenomeCache
from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)


class GenomeCacheRepository:
    """
    Persists the set of genomes (chromosomes) that have already been evaluated
    for a given experiment, along with their computed objectives.

    Responsibilities:
    - Prevent re-simulation of identical chromosomes across generations and
      across mo-engine restarts (deterministic get_hash required on Chromosome).
    - Allow immediate objective reuse when the same genome re-appears in a
      new generation (e.g. elite individuals surviving selection).
    """

    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["genome_cache"].create_index(
                [("experiment_id", 1), ("genome_hash", 1)],
                unique=True,
                name="idx_genome_cache_experiment_hash",
            )

    def insert(
        self,
        experiment_id: ObjectId,
        genome_hash: str,
        chromosome: dict[str, Any],
    ) -> ObjectId:
        """Register a new genome. objectives starts as None (not yet evaluated)."""
        doc: GenomeCache = {
            "experiment_id": experiment_id,
            "genome_hash": genome_hash,
            "chromosome": chromosome,
            "objectives": None,
        }
        with self.connection.connect() as db:
            return db["genome_cache"].insert_one(doc).inserted_id

    def set_objectives(
        self,
        experiment_id: ObjectId,
        genome_hash: str,
        objectives: list[float],
    ) -> bool:
        """Persist computed objectives for a previously registered genome."""
        with self.connection.connect() as db:
            result = db["genome_cache"].update_one(
                {"experiment_id": experiment_id, "genome_hash": genome_hash},
                {"$set": {"objectives": objectives}},
            )
            return result.modified_count > 0

    def get_all_by_experiment(self, experiment_id: ObjectId) -> list[dict]:
        """
        Return all cache entries for an experiment.
        Each entry has at least: genome_hash (str), objectives (list|None).
        """
        with self.connection.connect() as db:
            return list(db["genome_cache"].find(
                {"experiment_id": experiment_id},
                {"_id": 0, "genome_hash": 1, "objectives": 1},
            ))

    def delete_by_experiment(self, experiment_id: ObjectId) -> int:
        with self.connection.connect() as db:
            result = db["genome_cache"].delete_many({"experiment_id": experiment_id})
            return result.deleted_count
