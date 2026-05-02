import logging
from datetime import datetime
from typing import Any
from bson import ObjectId, errors

from pylib.db.models.campaign import Campaign
from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)


class CampaignRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def insert(self, campaign: Campaign) -> ObjectId:
        with self.connection.connect() as db:
            return db["campaigns"].insert_one(campaign).inserted_id

    def find_all(self) -> list[dict[str, Any]]:
        with self.connection.connect() as db:
            return list(db["campaigns"].find({}))

    def get(self, campaign_id: str) -> Campaign:
        try:
            oid = ObjectId(campaign_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", campaign_id)
            return None
        with self.connection.connect() as db:
            return db["campaigns"].find_one({"_id": oid})

    def update(self, campaign_id: str, updates: dict) -> bool:
        try:
            oid = ObjectId(campaign_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", campaign_id)
            return False
        with self.connection.connect() as db:
            result = db["campaigns"].update_one({"_id": oid}, {"$set": updates})
            return result.modified_count > 0

    def delete(self, campaign_id: str) -> bool:
        try:
            oid = ObjectId(campaign_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", campaign_id)
            return False
        with self.connection.connect() as db:
            result = db["campaigns"].delete_one({"_id": oid})
            return result.deleted_count == 1

    def add_experiment(self, campaign_id: str, experiment_id: str) -> bool:
        try:
            camp_oid = ObjectId(campaign_id)
            exp_oid = ObjectId(experiment_id)
        except errors.InvalidId as e:
            log.error("Invalid ID: %s", e)
            return False
        with self.connection.connect() as db:
            result = db["campaigns"].update_one(
                {"_id": camp_oid},
                {"$addToSet": {"experiment_ids": exp_oid}}
            )
            return result.matched_count == 1

    def remove_experiment(self, campaign_id: str, experiment_id: str) -> bool:
        try:
            camp_oid = ObjectId(campaign_id)
            exp_oid = ObjectId(experiment_id)
        except errors.InvalidId as e:
            log.error("Invalid ID: %s", e)
            return False
        with self.connection.connect() as db:
            result = db["campaigns"].update_one(
                {"_id": camp_oid},
                {"$pull": {"experiment_ids": exp_oid}}
            )
            return result.matched_count == 1
