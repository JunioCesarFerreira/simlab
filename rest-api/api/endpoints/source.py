from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from bson import ObjectId
import os, sys, tempfile
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from dto import SourceRepository, SourceFile, source_repository_from_mongo
from pylib import mongo_db

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()


@router.get("/", response_model=list[SourceRepository])
def list_sources() -> list[SourceRepository]:
    try:
        docs = factory.source_repo.get_all()
        return [source_repository_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repository_id}", response_model=SourceRepository)
def get_source_repository(repository_id: str) -> SourceRepository:
    try:
        doc = factory.source_repo.get_by_id(repository_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Source repository not found")
        return source_repository_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=str)
async def create_source_repository(
    name: str = Form(...),
    description: str = Form(""),
    files: list[UploadFile] = File(...)
) -> str:
    source_files: list[SourceFile] = []
    try:
        for upload in files:
            # Salva temporário apenas para subir via GridFS e depois apagar
            with NamedTemporaryFile(delete=False) as tmp:
                content = await upload.read()
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name
            file_id = factory.fs_handler.upload_file(tmp_path, name=upload.filename)
            source_files.append({"id": str(file_id), "file_name": upload.filename})
            os.remove(tmp_path)

        source: SourceRepository = {
            "id": "",
            "name": name,
            "description": description,
            "source_files": source_files,
        }
        inserted_id = factory.source_repo.insert(source)
        return str(inserted_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repository_id}/download")
def download_source_repository(repository_id: str):
    try:
        repo = factory.source_repo.get_by_id(repository_id)
        if not repo:
            raise HTTPException(status_code=404, detail="Repositório não encontrado")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip_file:
            with ZipFile(tmp_zip_file.name, 'w') as zipf:
                for file_info in (repo.get("source_files") or []):
                    file_id = file_info.get("id")
                    file_name = file_info.get("file_name", str(file_id))
                    with NamedTemporaryFile(delete=False) as tmp_file:
                        factory.fs_handler.download_file(ObjectId(file_id), tmp_file.name)
                        zipf.write(tmp_file.name, arcname=file_name)
                        tmp_file.close()
                        os.remove(tmp_file.name)

            tmp_zip_path = tmp_zip_file.name

        return FileResponse(
            tmp_zip_path,
            filename=f"repository_{repository_id}.zip",
            media_type="application/zip",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao baixar arquivos do repositório: {e}")
