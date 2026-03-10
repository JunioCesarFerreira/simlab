from functools import lru_cache

from pylib import mongo_db
from pylib.service_settings import MongoServiceSettings


@lru_cache(maxsize=1)
def get_mongo_factory() -> mongo_db.MongoRepository:
    settings = MongoServiceSettings.from_env()
    return mongo_db.create_mongo_repository_factory(
        settings.mongo_uri,
        settings.db_name,
    )
