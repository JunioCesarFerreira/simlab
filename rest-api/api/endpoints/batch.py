# api/endpoints/batch.py
from fastapi import APIRouter, HTTPException
from bson import ObjectId, errors
import os, sys

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db

from api.dto import BatchDto
from api.dto_conversor import batch_to_mongo, batch_from_mongo

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()


@router.post("/", response_model=str)
def create_batch(batch: BatchDto) -> str:
    """
    Create a new batch.
    
    - Receives a validated BatchDto
    - Converts it to MongoDB-compatible format
    - Persists it in the batchs collection
    
    Returns the generated batch_id as string.
    """
    try:
        doc = batch_to_mongo(batch)
        batch_id = factory.batch_repo.insert(doc)
        return str(batch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{batch_id}", response_model=BatchDto)
def get_batch(batch_id: str) -> BatchDto:
    """
    Retrieve a single batch by its identifier.
    
    - batch_id must be a valid ObjectId string
    - Returns the batch mapped to BatchDto
    - Returns 404 if the batch does not exist
    """
    try:
        doc = factory.batch_repo.get_by_id(batch_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Batch not found")
        return batch_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{batch_id}", response_model=bool)
def update_batch(batch_id: str, updates: dict) -> bool:
    """
    Update arbitrary fields of a batch.
    
    - batch_id must be a valid ObjectId string
    - updates is a partial dictionary of fields to be updated
    - Uses a $set-like semantic in the repository layer
    
    Returns True if the update operation succeeds.
    """
    try:
        return factory.batch_repo.update(ObjectId(batch_id), updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[BatchDto])
def get_batchs_by_status(status: str) -> list[BatchDto]:
    """
    Retrieve all batchs with a given status.
    
    - Status is matched exactly as stored in the database
    - Returns a list of BatchDto objects
    - Empty list is returned if no batch matches
    """
    try:
        docs = factory.batch_repo.find_by_status(status)
        return [batch_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{batch_id}/status", response_model=bool)
def update_batch_status(batch_id: str, new_status: str) -> bool:
    """
    Update only the status field of a batch.
    
    - Intended for controlled lifecycle transitions
      (e.g., Created → Ready → Completed)
    - Avoids full document updates
    - batch_id must be a valid ObjectId string
    
    Returns True if the operation succeeds.
    """
    try:
        return factory.batch_repo.update(ObjectId(batch_id), {"status": new_status})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
