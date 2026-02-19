# api/endpoints/simulation.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from bson import ObjectId
import tempfile
import os, sys

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db

from api.dto import SimulationDto, simulation_to_mongo, simulation_from_mongo

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()


@router.post("/", response_model=str)
def create_simulation(simulation: SimulationDto) -> str:
    """
    Create a new simulation.
    
    - Receives a validated SimulationDto
    - Converts it to MongoDB-compatible format
    - Persists it in the simulations collection
    
    Returns the generated simulation_id as string.
    """
    try:
        doc = simulation_to_mongo(simulation)
        sim_id = factory.simulation_repo.insert(doc)
        return str(sim_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{sim_id}", response_model=SimulationDto)
def get_simulation(sim_id: str) -> SimulationDto:
    """
    Create a new simulation.
    
    - Receives a validated SimulationDto
    - Converts it to MongoDB-compatible format
    - Persists it in the simulations collection
    
    Returns the generated simulation_id as string.
    """
    try:
        doc = factory.simulation_repo.get(sim_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Simulation not found")
        return simulation_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{sim_id}", response_model=bool)
def update_simulation(sim_id: str, updates: dict) -> bool:
    """
    Update arbitrary fields of a simulation.
    
    - sim_id must be a valid ObjectId string
    - updates is a partial dictionary of fields to be updated
    - Uses a $set-like semantic in the repository layer
    
    Returns True if the update operation succeeds.
    """
    try:
        return factory.simulation_repo.update(ObjectId(sim_id), updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{sim_id}", response_model=bool)
def delete_simulation(sim_id: str) -> bool:
    """
    Delete a simulation by its identifier.
    
    - sim_id must be a valid ObjectId string
    - Deletes only the simulation document
    - Associated GridFS files must be handled explicitly elsewhere
    
    Returns True if the delete operation succeeds.
    """
    try:
        return factory.simulation_repo.delete_by_id(ObjectId(sim_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[SimulationDto])
def get_simulations_by_status(status: str) -> list[SimulationDto]:
    """
    Retrieve all simulations with a given status.
    
    - Status is matched exactly as stored in the database
    - Returns a list of SimulationDto objects
    - Empty list is returned if no simulation matches
    """
    try:
        docs = factory.simulation_repo.find_by_status(status)
        return [simulation_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{sim_id}/file/{field_name}")
def download_simulation_file(sim_id: str, field_name: str):
    """
    Download a file associated with a simulation.
    
    - sim_id must be a valid ObjectId string
    - field_name refers to a file reference field stored in the simulation document
      (e.g., pos_file_id, log_cooja_id, topology_picture_id)
    - The file is retrieved from GridFS and returned as a binary download
    
    Returns the file as an attachment.
    """
    try:
        # search sim and file id
        sim = factory.simulation_repo.get(sim_id)
        if not sim:
            raise HTTPException(status_code=404, detail="Simulation not found")

        file_id = sim.get(field_name)
        if not file_id:
            raise HTTPException(status_code=404, detail=f"Field '{field_name}' is empty")

        # Baixa arquivo para um tmp local e devolve como download
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        factory.fs_handler.download_file(ObjectId(file_id), tmp_path)

        return FileResponse(
            tmp_path,
            filename=f"{field_name}_{sim_id}",
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao baixar arquivo '{field_name}': {e}")


@router.patch("/{sim_id}/status", response_model=bool)
def update_simulation_status(sim_id: str, new_status: str) -> bool:
    """
    Update only the status field of a simulation.
    
    - Intended for controlled lifecycle transitions
      (e.g., Queued → Running → Finished → Failed)
    - Avoids full document updates
    - sim_id must be a valid ObjectId string
    
    Returns True if the operation succeeds.
    """
    try:
        factory.simulation_repo.update_status(sim_id, new_status)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
