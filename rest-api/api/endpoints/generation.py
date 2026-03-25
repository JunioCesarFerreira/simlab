# api/endpoints/generation.py
from fastapi import APIRouter, HTTPException
from bson import ObjectId
import os, sys

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib.db import create_mongo_repository_factory
from api.dto import GenerationDto
from api.dto_conversor import generation_from_mongo

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()


@router.get("/{generation_id}", response_model=GenerationDto)
def get_generation(generation_id: str) -> GenerationDto:
    """
    Retrieve a single generation by its identifier.
    Includes the list of individuals with their chromosome, objectives and topology.
    """
    try:
        doc = factory.generation_repo.get(generation_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Generation not found")
        individuals = factory.individual_repo.find_by_generation(ObjectId(generation_id))
        return generation_from_mongo(doc, individuals)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-experiment/{experiment_id}", response_model=list[GenerationDto])
def get_generations_by_experiment(experiment_id: str) -> list[GenerationDto]:
    """
    Retrieve all generations belonging to an experiment, ordered by index.
    Each generation includes its individuals.
    """
    try:
        gens = factory.generation_repo.find_by_experiment(ObjectId(experiment_id))
        result = []
        for gen in gens:
            individuals = factory.individual_repo.find_by_generation(gen["_id"])
            result.append(generation_from_mongo(gen, individuals))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[GenerationDto])
def get_generations_by_status(status: str) -> list[GenerationDto]:
    """
    Retrieve all generations with a given status.
    """
    try:
        gens = factory.generation_repo.find_by_status(status)
        result = []
        for gen in gens:
            individuals = factory.individual_repo.find_by_generation(gen["_id"])
            result.append(generation_from_mongo(gen, individuals))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{generation_id}/status", response_model=bool)
def update_generation_status(generation_id: str, new_status: str) -> bool:
    """Update only the status field of a generation."""
    try:
        return factory.generation_repo.update(ObjectId(generation_id), {"status": new_status})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
