import logging
import gridfs
from bson import ObjectId
from mongo.connection import MongoDBConnection

logger = logging.getLogger(__name__)

class MongoGridFSHandler:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def upload_file(self, path: str, name: str) -> ObjectId:
        with self.connection.connect() as db:
            fs = gridfs.GridFS(db)
            with open(path, "rb") as f:
                file_id = fs.put(f, filename=name)
        return ObjectId(file_id)

    def download_file(self, file_id: str, local_path: str):
        try:
            with self.connection.connect() as db:
                fs = gridfs.GridFS(db)
                grid_out = fs.get(ObjectId(file_id))
                with open(local_path, 'wb') as f:
                    f.write(grid_out.read())
                logger.info(f"File {local_path} saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save file {file_id}: {e}")