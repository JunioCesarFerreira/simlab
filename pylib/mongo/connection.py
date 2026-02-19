import logging
import time
import pymongo
from typing import Generator, Callable
from enum import Enum
from pymongo import MongoClient
from contextlib import contextmanager
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
        logger.info(f"[MongoDBConnection] uri:{uri} db_name:{db_name}")
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
                logger.error("[WorkGenerator] Aguardando conexão com MongoDB...")
                time.sleep(3)
    
    
    def watch_collection(self,
        collection_name: str,
        pipeline: list[dict],
        on_change: Callable[[dict], None],
        full_document: str = "default"
        ) -> None:
        """
        Observes changes in a specific collection using a given pipeline.

        Args:
            collection_name (str): MongoDB collection name.
            pipeline (list): Aggregation pipeline (e.g., [$match]).
            on_change (Callable): Callback function called on each event.
            full_document (str): Full document retrieval mode.
        """
        with self.connect() as db:
            collection: Collection = db[collection_name]
            try:
                with collection.watch(pipeline, full_document=full_document) as stream:
                    for change in stream:
                        on_change(change)
            except PyMongoError as e:
                logger.error(f"[watch_collection] Erro ao observar coleção '{collection_name}': {e}")
