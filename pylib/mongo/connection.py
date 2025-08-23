import logging
import time
import pymongo
from typing import Generator, Callable
from enum import Enum
from pymongo import MongoClient
from contextlib import contextmanager
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

# Constantes de status
class EnumStatus(str, Enum):
    BUILDING = "Building"
    WAITING = "Waiting"
    RUNNING = "Running"
    DONE = "Done"
    ERROR = "Error"

logger = logging.getLogger(__name__)

class MongoDBConnection:
    def __init__(self, uri: str, db_name: str):
        print(f"[MongoDBConnection] uri:{uri} db_name:{db_name}")
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
                print("[WorkGenerator] Aguardando conexão com MongoDB...")
                time.sleep(3)
    
    def watch_collection(self,
        collection_name: str,
        pipeline: list[dict],
        on_change: Callable[[dict], None],
        full_document: str = "default"
        ) -> None:
        """
        Observa alterações em uma coleção específica com um pipeline dado.

        Args:
            collection_name (str): Nome da coleção MongoDB.
            pipeline (list): Pipeline de agregação (ex: [$match]).
            on_change (Callable): Função de callback chamada a cada evento.
            full_document (str): Modo de recuperação do documento completo.
        """
        with self.connect() as db:
            collection: Collection = db[collection_name]
            try:
                with collection.watch(pipeline, full_document=full_document) as stream:
                    for change in stream:
                        on_change(change)
            except PyMongoError as e:
                print(f"[watch_collection] Erro ao observar coleção '{collection_name}': {e}")
