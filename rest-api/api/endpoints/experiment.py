# api/endpoints/experiment.py
from fastapi import APIRouter, HTTPException
import os
from bson import errors

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
import sys
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib.dto.experiment import ExperimentDto, experiment_to_mongo, experiment_from_mongo
from pylib import mongo_db

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()


@router.post("/", response_model=str)
def create_experiment(experiment: ExperimentDto) -> str:
    try:
        doc = experiment_to_mongo(experiment)
        exp_id = factory.experiment_repo.insert(doc)  # returns ObjectId
        return str(exp_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentDto)
def get_experiment(experiment_id: str) -> ExperimentDto:
    try:
        doc = factory.experiment_repo.get_by_id(experiment_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[ExperimentDto])
def get_experiments_by_status(status: str) -> list[ExperimentDto]:
    try:
        docs = factory.experiment_repo.find_by_status(status)
        return [experiment_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{experiment_id}", response_model=bool)
def update_experiment(experiment_id: str, updates: dict) -> bool:
    try:
        return factory.experiment_repo.update(experiment_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{experiment_id}", response_model=bool)
def delete_experiment(experiment_id: str) -> bool:
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
    try:
        factory.experiment_repo.update_status(experiment_id, new_status)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
