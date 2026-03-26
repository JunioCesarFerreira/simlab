from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId

from pylib.db import MongoRepository
from api.dependencies import get_factory
from api.domain.generation import GenerationDto
from api.mappers.generation import generation_from_mongo

router = APIRouter()


def _build_generation(gen: dict, factory: MongoRepository) -> GenerationDto:
    individuals = factory.individual_repo.find_by_generation(gen["_id"])
    return generation_from_mongo(gen, individuals)


@router.get("/by-experiment/{experiment_id}", response_model=list[GenerationDto])
def get_generations_by_experiment(
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> list[GenerationDto]:
    """Retrieve all generations of an experiment ordered by index, each with its individuals."""
    try:
        gens = factory.generation_repo.find_by_experiment(ObjectId(experiment_id))
        return [_build_generation(g, factory) for g in gens]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[GenerationDto])
def get_generations_by_status(
    status: str,
    factory: MongoRepository = Depends(get_factory)
) -> list[GenerationDto]:
    """Retrieve all generations with a given status."""
    try:
        gens = factory.generation_repo.find_by_status(status)
        return [_build_generation(g, factory) for g in gens]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{generation_id}", response_model=GenerationDto)
def get_generation(
    generation_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> GenerationDto:
    """Retrieve a single generation by its ObjectId, including its individuals."""
    try:
        doc = factory.generation_repo.get(generation_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Generation not found")
        return _build_generation(doc, factory)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{generation_id}/status", response_model=bool)
def update_generation_status(
    generation_id: str,
    new_status: str,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Update only the status field of a generation."""
    try:
        return factory.generation_repo.update(ObjectId(generation_id), {"status": new_status})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
