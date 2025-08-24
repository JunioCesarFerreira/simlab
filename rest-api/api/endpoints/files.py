from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from bson import ObjectId, errors as bson_errors
from gridfs.errors import NoFile
import os, sys, tempfile

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()

@router.get("/{file_id}/as/{extension}", response_class=FileResponse)
def download_file(file_id: str, extension: str, background_tasks: BackgroundTasks):
    # valida ObjectId
    try:
        oid = ObjectId(file_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid file_id")

    # cria arquivo temporário .txt
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
    tmp_path = tmp.name
    tmp.close()

    try:
        # baixa o conteúdo do GridFS diretamente para o arquivo temporário
        factory.fs_handler.download_file(oid, tmp_path)
    except NoFile:
        os.remove(tmp_path)
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise HTTPException(status_code=500, detail=f"Erro ao baixar arquivo: {e}")

    # remove o temp após enviar a resposta
    background_tasks.add_task(os.remove, tmp_path)

    return FileResponse(
        tmp_path,
        filename=f"{file_id}.{extension}",
        media_type="text/plain; charset=utf-8",
        background=background_tasks,
    )
