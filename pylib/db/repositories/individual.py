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
