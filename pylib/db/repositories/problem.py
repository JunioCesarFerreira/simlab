import logging
from datetime import datetime, timezone
from typing import Optional, Any

from bson import ObjectId, errors

from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)

COLLECTION = "problems"


class ProblemRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def insert(
        self,
        name: str,
        draft: dict,
        image_world_bounds: Optional[list] = None,
    ) -> ObjectId:
        now = datetime.now(timezone.utc)
        doc = {
            "name": name,
            "created_time": now,
            "updated_time": now,
            "draft": draft,
            "background_image_id": None,
            "image_world_bounds": image_world_bounds,
        }
        with self.connection.connect() as db:
            return db[COLLECTION].insert_one(doc).inserted_id

    def get_all(self) -> list[dict[str, Any]]:
        with self.connection.connect() as db:
            return list(db[COLLECTION].find(
                {},
                {"_id": 1, "name": 1, "created_time": 1, "updated_time": 1},
            ).sort("updated_time", -1))

    def get(self, problem_id: str) -> Optional[dict[str, Any]]:
        try:
            oid = ObjectId(problem_id)
        except errors.InvalidId:
            log.error("Invalid problem ID: %s", problem_id)
            return None
        with self.connection.connect() as db:
            return db[COLLECTION].find_one({"_id": oid})

    def update(self, problem_id: str, updates: dict) -> bool:
        try:
            oid = ObjectId(problem_id)
        except errors.InvalidId:
            return False
        updates["updated_time"] = datetime.now(timezone.utc)
        with self.connection.connect() as db:
            result = db[COLLECTION].update_one({"_id": oid}, {"$set": updates})
            return result.matched_count > 0

    def set_background(self, problem_id: str, image_id: ObjectId) -> bool:
        return self.update(problem_id, {"background_image_id": image_id})

    def delete(self, problem_id: str) -> Optional[ObjectId]:
        """Delete the document and return its background_image_id for GridFS cleanup."""
        try:
            oid = ObjectId(problem_id)
        except errors.InvalidId:
            return None
        with self.connection.connect() as db:
            doc = db[COLLECTION].find_one({"_id": oid}, {"background_image_id": 1})
            if not doc:
                return None
            db[COLLECTION].delete_one({"_id": oid})
            return doc.get("background_image_id")
