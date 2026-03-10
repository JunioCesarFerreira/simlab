import logging
import time
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Generator

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


class EnumStatus(str, Enum):
    BUILDING = "Building"
    WAITING = "Waiting"
    RUNNING = "Running"
    DONE = "Done"
    ERROR = "Error"


logger = logging.getLogger(__name__)


class MongoDBConnection:
    def __init__(self, uri: str, db_name: str):
        logger.info("[MongoDBConnection] uri:%s db_name:%s", uri, db_name)
        self.uri = uri
        self.db_name = db_name

    @contextmanager
    def connect(self) -> Generator:
        client = MongoClient(self.uri)
        try:
            yield client[self.db_name]
        finally:
            client.close()

    def waiting_ping(self) -> None:
        while True:
            try:
                with MongoClient(self.uri) as client:
                    client.admin.command("ping")
                break
            except pymongo.errors.ConnectionFailure:
                logger.error("[WorkGenerator] Waiting for MongoDB connection...")
                time.sleep(3)

    def watch_collection(
        self,
        collection_name: str,
        pipeline: list[dict],
        on_change: Callable[[dict], None],
        full_document: str = "default",
        retry_delay_sec: int = 3,
    ) -> None:
        """
        Observe changes in a collection and reconnect automatically on transient
        Mongo errors so service pipelines do not stop silently.
        """
        while True:
            try:
                with self.connect() as db:
                    collection: Collection = db[collection_name]
                    logger.info(
                        "[watch_collection] Watching '%s' with automatic reconnect.",
                        collection_name,
                    )
                    with collection.watch(pipeline, full_document=full_document) as stream:
                        for change in stream:
                            try:
                                on_change(change)
                            except Exception:
                                logger.exception(
                                    "[watch_collection] Callback error in collection '%s'.",
                                    collection_name,
                                )
            except PyMongoError as exc:
                logger.error(
                    "[watch_collection] Error watching '%s': %s. Retrying in %ss.",
                    collection_name,
                    exc,
                    retry_delay_sec,
                )
                time.sleep(retry_delay_sec)
