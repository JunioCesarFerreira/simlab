import os
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from bson import ObjectId

from pylib.db import MongoRepository
from pylib.db.models import SourceFile
from api.dependencies import get_factory
from api.domain.source import SourceRepositoryDto
from api.mappers.source import source_repository_from_mongo

router = APIRouter()


@router.get("/", response_model=list[SourceRepositoryDto])
def list_sources(factory: MongoRepository = Depends(get_factory)) -> list[SourceRepositoryDto]:
    """List all registered source repositories."""
    try:
        return [source_repository_from_mongo(d) for d in factory.source_repo.get_all()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repository_id}", response_model=SourceRepositoryDto)
def get_source_repository(
    repository_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> SourceRepositoryDto:
    """Retrieve a single source repository by its ObjectId."""
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
    files: list[UploadFile] = File(...),
    factory: MongoRepository = Depends(get_factory)
) -> str:
    """Create a source repository and upload its files to GridFS. Returns the repository_id."""
    source_files: list[SourceFile] = []
    try:
        for upload in files:
            with NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await upload.read())
                tmp.flush()
                tmp_path = tmp.name
            file_id = factory.fs_handler.upload_file(tmp_path, name=upload.filename)
            source_files.append({"id": str(file_id), "file_name": upload.filename})
            os.remove(tmp_path)

        inserted_id = factory.source_repo.insert({
            "id": "",
            "name": name,
            "description": description,
            "source_files": source_files,
        })
        return str(inserted_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{repository_id}/download")
def download_source_repository(
    repository_id: str,
    factory: MongoRepository = Depends(get_factory)
):
    """Download all files of a source repository as a ZIP archive."""
    try:
        repo = factory.source_repo.get_by_id(repository_id)
        if not repo:
            raise HTTPException(status_code=404, detail="Source repository not found")

        with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with ZipFile(tmp_zip.name, "w") as zipf:
                for file_info in (repo.get("source_files") or []):
                    file_id = file_info.get("id")
                    file_name = file_info.get("file_name", str(file_id))
                    with NamedTemporaryFile(delete=False) as tmp:
                        factory.fs_handler.download_file(ObjectId(file_id), tmp.name)
                        zipf.write(tmp.name, arcname=file_name)
                        tmp.close()
                        os.remove(tmp.name)
            zip_path = tmp_zip.name

        return FileResponse(zip_path, filename=f"repository_{repository_id}.zip", media_type="application/zip")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
