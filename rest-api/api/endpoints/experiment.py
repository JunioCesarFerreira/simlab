from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from bson import errors as bson_errors
from tempfile import NamedTemporaryFile

from pylib.db import MongoRepository
from api.dependencies import get_factory
from api.domain.experiment import ExperimentDto, ExperimentInfoDto
from api.mappers.experiment import experiment_from_mongo, experiment_info_from_mongo, experiment_to_mongo

router = APIRouter()


@router.post("/", response_model=str)
def create_experiment(
    experiment: ExperimentDto,
    factory: MongoRepository = Depends(get_factory)
) -> str:
    """Create a new experiment. Returns the generated experiment_id."""
    try:
        doc = experiment_to_mongo(experiment)
        return str(factory.experiment_repo.insert(doc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[ExperimentInfoDto])
def get_experiments_by_status(
    status: str,
    factory: MongoRepository = Depends(get_factory)
) -> list[ExperimentInfoDto]:
    """Retrieve all experiments with a given status."""
    try:
        docs = factory.experiment_repo.find_by_status(status)
        return [experiment_info_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentDto)
def get_experiment(
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> ExperimentDto:
    """Retrieve a single experiment by its ObjectId."""
    try:
        doc = factory.experiment_repo.get(experiment_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{experiment_id}", response_model=bool)
def update_experiment(
    experiment_id: str,
    updates: dict,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Partially update an experiment using $set semantics."""
    try:
        return factory.experiment_repo.update(experiment_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{experiment_id}", response_model=bool)
def delete_experiment(
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Delete an experiment and all its associated data (cascade)."""
    try:
        res = factory.experiment_repo.delete(experiment_id)
        if isinstance(res, dict):
            return res.get("deleted_experiments", 0) == 1
        return bool(res)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{experiment_id}/status", response_model=bool)
def update_experiment_status(
    experiment_id: str,
    new_status: str,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Update only the status field of an experiment."""
    try:
        factory.experiment_repo.update_status(experiment_id, new_status)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{experiment_id}/analysis-file", response_model=str)
async def attach_analysis_file(
    experiment_id: str,
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    factory: MongoRepository = Depends(get_factory)
) -> str:
    """Upload and attach an analysis file to an experiment. Returns the GridFS file_id."""
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            tmp_path = tmp.name
        oid = factory.experiment_repo.add_analysis_file_to_experiment(
            experiment_id, description, tmp_path, name
        )
        return str(oid)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
