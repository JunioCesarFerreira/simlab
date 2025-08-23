from typing import Optional, Any
from bson import ObjectId

from dto import SourceFile, SourceRepository
from mongo.connection import MongoDBConnection

class SourceRepositoryAccess:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def insert(self, source: SourceRepository) -> ObjectId:
        with self.connection.connect() as db:
            return db["sources"].insert_one(source).inserted_id

    def get_all(self) -> list[SourceRepository]:
        with self.connection.connect() as db:
            return list(db["sources"].find())
    
    # Adiciona um novo arquivo (SourceFile) à lista source_ids de um SourceRepository.
    def append_source_file(self, repository_id: str, new_source_file: SourceFile) -> bool:
        with self.connection.connect() as db:
            result = db["sources"].update_one(
                {"_id": ObjectId(repository_id)},
                {"$addToSet": {"source_ids": new_source_file}}  # evita duplicatas exatas
            )
            return result.modified_count > 0

    # Atualiza os campos de metadados de um SourceRepository (exceto 'source_ids').
    def update_metadata(self, repository_id: str, updates: dict[str, Any]) -> bool:
        allowed_keys = {"name", "description"}  # adicione mais campos permitidos se necessário
        filtered_updates = {k: v for k, v in updates.items() if k in allowed_keys}

        if not filtered_updates:
            return False

        with self.connection.connect() as db:
            result = db["sources"].update_one(
                {"_id": ObjectId(repository_id)},
                {"$set": filtered_updates}
            )
            return result.modified_count > 0
    
    # Recupera um SourceRepository pelo seu ID.
    def get_by_id(self, repository_id: str) -> Optional[SourceRepository]:
        with self.connection.connect() as db:
            return db["sources"].find_one({"_id": ObjectId(repository_id)})