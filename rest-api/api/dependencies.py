import os
from functools import lru_cache

from pylib.db import create_mongo_repository_factory, MongoRepository


@lru_cache(maxsize=1)
def get_factory() -> MongoRepository:
    """
    Single MongoRepository factory instance shared across all endpoints.
    Resolved once and cached for the process lifetime.
    """
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
    db_name = os.getenv("DB_NAME", "simlab")
    return create_mongo_repository_factory(mongo_uri, db_name)
