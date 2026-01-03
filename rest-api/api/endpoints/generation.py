# api/endpoints/generation.py
from fastapi import APIRouter, HTTPException
from bson import ObjectId, errors
import os, sys

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db

from api.dto import GenerationDto, generation_to_mongo, generation_from_mongo

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()


@router.post("/", response_model=str)
def create_generation(generation: GenerationDto) -> str:
    """
    Create a new generation.
    
    - Receives a validated GenerationDto
    - Converts it to MongoDB-compatible format
    - Persists it in the generations collection
    
    Returns the generated generation_id as string.
    """
    try:
        doc = generation_to_mongo(generation)
        gen_id = factory.generation_repo.insert(doc)
        return str(gen_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{gen_id}", response_model=GenerationDto)
def get_generation(gen_id: str) -> GenerationDto:
    """
    Retrieve a single generation by its identifier.
    
    - gen_id must be a valid ObjectId string
    - Returns the generation mapped to GenerationDto
    - Returns 404 if the generation does not exist
    """
    try:
        doc = factory.generation_repo.get_by_id(gen_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Generation not found")
        return generation_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{gen_id}", response_model=bool)
def update_generation(gen_id: str, updates: dict) -> bool:
    """
    Update arbitrary fields of a generation.
    
    - gen_id must be a valid ObjectId string
    - updates is a partial dictionary of fields to be updated
    - Uses a $set-like semantic in the repository layer
    
    Returns True if the update operation succeeds.
    """
    try:
        return factory.generation_repo.update(ObjectId(gen_id), updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{gen_id}", response_model=bool)
def delete_generation(gen_id: str) -> bool:
    """
    Delete a generation by its identifier.
    
    - gen_id must be a valid ObjectId string
    - Deletes only the generation document
    - Associated simulations or GridFS data
      must be handled explicitly elsewhere
    
    Returns True if exactly one generation was deleted.
    """
    try:
        res = factory.generation_repo.delete_by_id(ObjectId(gen_id))
        if isinstance(res, dict):
            return res.get("deleted_experiments", 0) == 1
        return bool(res)
    except errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/by-status/{status}", response_model=list[GenerationDto])
def get_generations_by_status(status: str) -> list[GenerationDto]:
    """
    Retrieve all generations with a given status.
    
    - Status is matched exactly as stored in the database
    - Returns a list of GenerationDto objects
    - Empty list is returned if no generation matches
    """
    try:
        docs = factory.generation_repo.find_by_status(status)
        return [generation_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{gen_id}/status", response_model=bool)
def update_generation_status(gen_id: str, new_status: str) -> bool:
    """
    Update only the status field of a generation.
    
    - Intended for controlled lifecycle transitions
      (e.g., Created → Ready → Completed)
    - Avoids full document updates
    - gen_id must be a valid ObjectId string
    
    Returns True if the operation succeeds.
    """
    try:
        factory.generation_repo.update_status(gen_id, new_status)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
