import logging
import mimetypes
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import Response

from pylib.db import MongoRepository
from api.dependencies import get_factory
from api.domain.problem import ProblemInfoDto, ProblemDto, ProblemCreateDto, ProblemUpdateDto
from api.mappers.problem import problem_to_info_dto, problem_to_dto

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=list[ProblemInfoDto])
def list_problems(factory: MongoRepository = Depends(get_factory)) -> list[ProblemInfoDto]:
    """List all saved problems (without draft or background, for index views)."""
    try:
        return [problem_to_info_dto(d) for d in factory.problem_repo.get_all()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=str)
def create_problem(
    body: ProblemCreateDto,
    factory: MongoRepository = Depends(get_factory),
) -> str:
    """Create a new saved problem. Returns the new problem ID."""
    try:
        inserted_id = factory.problem_repo.insert(
            name=body.name,
            draft=body.draft,
            image_world_bounds=body.image_world_bounds,
        )
        return str(inserted_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{problem_id}", response_model=ProblemDto)
def get_problem(
    problem_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> ProblemDto:
    """Retrieve a saved problem including its draft and metadata."""
    try:
        doc = factory.problem_repo.get(problem_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Problem not found")
        return problem_to_dto(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{problem_id}", response_model=bool)
def update_problem(
    problem_id: str,
    body: ProblemUpdateDto,
    factory: MongoRepository = Depends(get_factory),
) -> bool:
    """Update name, draft and/or image_world_bounds of a saved problem."""
    try:
        updates: dict = {}
        if body.name is not None:
            updates["name"] = body.name
        if body.draft is not None:
            updates["draft"] = body.draft
        if body.image_world_bounds is not None:
            updates["image_world_bounds"] = body.image_world_bounds
        if not updates:
            return True
        result = factory.problem_repo.update(problem_id, updates)
        if not result:
            raise HTTPException(status_code=404, detail="Problem not found")
        return True
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{problem_id}", response_model=bool)
def delete_problem(
    problem_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> bool:
    """Delete a saved problem and remove its background image from GridFS."""
    try:
        bg_id = factory.problem_repo.delete(problem_id)
        if bg_id is None:
            raise HTTPException(status_code=404, detail="Problem not found")
        if bg_id:
            try:
                factory.fs_handler.delete_file(str(bg_id))
            except Exception:
                pass  # GridFS cleanup is best-effort
        return True
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{problem_id}/background", response_model=str)
async def upload_background(
    problem_id: str,
    file: UploadFile = File(...),
    factory: MongoRepository = Depends(get_factory),
) -> str:
    """Upload (or replace) the background image for a saved problem. Returns the new GridFS file ID."""
    try:
        doc = factory.problem_repo.get(problem_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Problem not found")

        # Remove old background from GridFS if one exists
        old_bg = doc.get("background_image_id")
        if old_bg:
            try:
                factory.fs_handler.delete_file(str(old_bg))
            except Exception:
                # Best-effort cleanup of the previous image — a stale GridFS
                # file must not block uploading the new one.
                log.warning("Failed to delete old background %s for problem %s",
                            old_bg, problem_id, exc_info=True)

        data = await file.read()
        filename = file.filename or "background.png"
        image_id = factory.fs_handler.upload_bytes(data, name=filename)
        factory.problem_repo.set_background(problem_id, image_id)
        return str(image_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{problem_id}/background")
def get_background(
    problem_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> Response:
    """Download the background image of a saved problem."""
    try:
        doc = factory.problem_repo.get(problem_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Problem not found")
        bg_id = doc.get("background_image_id")
        if not bg_id:
            raise HTTPException(status_code=404, detail="No background image")

        data = factory.fs_handler.read_file_content(str(bg_id))

        # Infer MIME type from GridFS filename when available
        content_type = "image/png"
        try:
            from bson import ObjectId
            import gridfs as _gfs
            with factory.fs_handler.connection.connect() as db:
                fs = _gfs.GridFS(db)
                grid_out = fs.get(ObjectId(str(bg_id)))
                fname = grid_out.filename or ""
                guessed, _ = mimetypes.guess_type(fname)
                if guessed and guessed.startswith("image/"):
                    content_type = guessed
        except Exception:
            # Non-fatal: fall back to the default image/png content type.
            log.warning("Could not infer MIME type for background %s; using %s",
                        bg_id, content_type, exc_info=True)

        return Response(content=data, media_type=content_type)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
