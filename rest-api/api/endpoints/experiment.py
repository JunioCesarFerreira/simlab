# api/endpoints/experiment.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import os, sys
from bson import errors
from tempfile import NamedTemporaryFile

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
import sys
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db

from api.dto import (
    ExperimentDto,
    ExperimentInfoDto,
    experiment_to_mongo, 
    experiment_from_mongo
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()


@router.post("/", response_model=str)
def create_experiment(experiment: ExperimentDto) -> str:
    """
    Create a new experiment.
    
    - Receives a validated ExperimentDto
    - Converts it to MongoDB-compatible format
    - Persists it in the experiments collection
    
    Returns the generated experiment_id as string.
    """
    try:
        doc = experiment_to_mongo(experiment)
        exp_id = factory.experiment_repo.insert(doc)  # returns ObjectId
        return str(exp_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentDto)
def get_experiment(experiment_id: str) -> ExperimentDto:
    """
    Retrieve a single experiment by its identifier.
    
    - experiment_id must be a valid ObjectId string
    - Returns the experiment mapped to ExperimentDto
    - Returns 404 if the experiment does not exist
    """
    try:
        doc = factory.experiment_repo.get_by_id(experiment_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[ExperimentInfoDto])
def get_experiments_by_status(status: str) -> list[ExperimentInfoDto]:
    """
    Retrieve all experiments with a given status.
    
    - Status is matched exactly as stored in the database
    - Returns a list of ExperimentDto objects
    - Empty list is returned if no experiment matches
    """
    try:
        docs = factory.experiment_repo.find_by_status(status)
        return [experiment_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{experiment_id}", response_model=bool)
def update_experiment(experiment_id: str, updates: dict) -> bool:
    """
    Update arbitrary fields of an experiment.
    
    - experiment_id must be a valid ObjectId string
    - updates is a partial dictionary of fields to be updated
    - Uses a $set-like semantic in the repository layer
    
    Returns True if the update operation succeeds.
    """
    try:
        return factory.experiment_repo.update(experiment_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{experiment_id}", response_model=bool)
def delete_experiment(experiment_id: str) -> bool:
    """
    Delete an experiment by its identifier.
    
    - experiment_id must be a valid ObjectId string
    - Deletes only the experiment document
    - Associated generations, simulations or GridFS data
      must be handled explicitly elsewhere
    
    Returns True if exactly one experiment was deleted.
    """
    try:
        res = factory.experiment_repo.delete_by_id(experiment_id)
        if isinstance(res, dict):
            return res.get("deleted_experiments", 0) == 1
        return bool(res)
    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{experiment_id}/status", response_model=bool)
def update_experiment_status(experiment_id: str, new_status: str) -> bool:
    """
    Update only the status field of an experiment.
    
    - Designed for state-machine transitions (e.g., Waiting → Running)
    - Avoids full document updates
    - experiment_id must be a valid ObjectId string
    
    Returns True if the operation succeeds.
    """
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
    file: UploadFile = File(...)
) -> str:
    """
    Attach an analysis file to an experiment.
    
    - File is stored in GridFS (or files collection)
    - Experiment stores only metadata + file_id
    
    Returns the file_id as string.
    """
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            tmp_path = tmp.name
        oid = factory.experiment_repo.add_analysis_file_to_experiment(
            experiment_id,
            description,
            tmp_path,
            name
            )
        return str(oid)

    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
