import logging
from bson import ObjectId, errors

from pylib.db.models.individual import Individual
from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)


class IndividualRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["individuals"].create_index([("generation_id", 1)], name="idx_individuals_generation_id")
            db["individuals"].create_index([("experiment_id", 1)], name="idx_individuals_experiment_id")
            db["individuals"].create_index(
                [("generation_id", 1), ("individual_id", 1)],
                unique=True,
                name="idx_individuals_generation_individual"
            )

    def insert(self, individual: Individual) -> ObjectId:
        with self.connection.connect() as db:
            return db["individuals"].insert_one(individual).inserted_id

    def get(self, individual_oid: str) -> Individual:
        try:
            oid = ObjectId(individual_oid)
        except errors.InvalidId:
            log.error("Invalid ID: %s", individual_oid)
            return None
        with self.connection.connect() as db:
            return db["individuals"].find_one({"_id": oid})

    def find_by_generation(self, generation_id: ObjectId) -> list[Individual]:
        with self.connection.connect() as db:
            return list(db["individuals"].find({"generation_id": generation_id}))

    def find_by_experiment_and_ids(
        self,
        experiment_id: ObjectId,
        individual_ids: list[str],
    ) -> list[Individual]:
        """Individuals of one experiment, looked up by chromosome hash.

        Survivor sets are hashes and a survivor may have been evaluated several
        generations earlier, so they cannot be resolved against a single
        generation's documents. The same hash can appear in more than one
        generation; callers that need one document per hash keep the first.
        """
        if not individual_ids:
            return []
        with self.connection.connect() as db:
            return list(db["individuals"].find({
                "experiment_id": experiment_id,
                "individual_id": {"$in": list(individual_ids)},
            }))

    def update_objectives(
        self,
        individual_id: str,
        generation_id: ObjectId,
        objectives: list[float]
    ) -> bool:
        with self.connection.connect() as db:
            result = db["individuals"].update_one(
                {"generation_id": generation_id, "individual_id": individual_id},
                {"$set": {"objectives": objectives}}
            )
            return result.modified_count > 0

    def update_topology_picture(
        self,
        individual_id: str,
        generation_id: ObjectId,
        topology_picture_id: ObjectId
    ) -> bool:
        with self.connection.connect() as db:
            result = db["individuals"].update_one(
                {"generation_id": generation_id, "individual_id": individual_id},
                {"$set": {"topology_picture_id": topology_picture_id}}
            )
            return result.modified_count > 0

    def delete_by_experiment(self, experiment_id: ObjectId) -> int:
        with self.connection.connect() as db:
            result = db["individuals"].delete_many({"experiment_id": experiment_id})
            return result.deleted_count

    def delete_by_generation(self, generation_id: ObjectId) -> int:
        with self.connection.connect() as db:
            result = db["individuals"].delete_many({"generation_id": generation_id})
            return result.deleted_count
