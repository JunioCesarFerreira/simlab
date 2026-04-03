from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from bson import ObjectId
import tempfile

from pylib.db import MongoRepository
from api.dependencies import get_factory
from api.domain.simulation import SimulationDto
from api.mappers.simulation import simulation_from_mongo, simulation_to_mongo

router = APIRouter()


@router.post("/", response_model=str)
def create_simulation(
    simulation: SimulationDto,
    factory: MongoRepository = Depends(get_factory)
) -> str:
    """Create a new simulation. Returns the generated simulation_id."""
    try:
        doc = simulation_to_mongo(simulation)
        return str(factory.simulation_repo.insert(doc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-individual/{individual_id}", response_model=list[SimulationDto])
def get_simulations_by_individual(
    individual_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> list[SimulationDto]:
    """Retrieve all simulations belonging to a given individual (by chromosome hash)."""
    try:
        docs = factory.simulation_repo.find_by_individual(individual_id)
        return [simulation_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[SimulationDto])
def get_simulations_by_status(
    status: str,
    factory: MongoRepository = Depends(get_factory)
) -> list[SimulationDto]:
    """Retrieve all simulations with a given status."""
    try:
        docs = factory.simulation_repo.find_by_status(status)
        return [simulation_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{sim_id}", response_model=SimulationDto)
def get_simulation(
    sim_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> SimulationDto:
    """Retrieve a simulation by its ObjectId."""
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
def update_simulation(
    sim_id: str,
    updates: dict,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Partially update a simulation using $set semantics."""
    try:
        return factory.simulation_repo.update(ObjectId(sim_id), updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{sim_id}", response_model=bool)
def delete_simulation(
    sim_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Delete a simulation by its ObjectId."""
    try:
        return factory.simulation_repo.delete_by_id(ObjectId(sim_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{sim_id}/file/{field_name}")
def download_simulation_file(
    sim_id: str,
    field_name: str,
    factory: MongoRepository = Depends(get_factory)
):
    """Download a GridFS file referenced by a simulation field (e.g. log_cooja_id)."""
    try:
        sim = factory.simulation_repo.get(sim_id)
        if not sim:
            raise HTTPException(status_code=404, detail="Simulation not found")
        file_id = sim.get(field_name)
        if not file_id:
            raise HTTPException(status_code=404, detail=f"Field '{field_name}' is empty")
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        factory.fs_handler.download_file(ObjectId(file_id), tmp_path)
        return FileResponse(tmp_path, filename=f"{field_name}_{sim_id}", media_type="application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{sim_id}/status", response_model=bool)
def update_simulation_status(
    sim_id: str,
    new_status: str,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Update only the status field of a simulation."""
    try:
        factory.simulation_repo.update_status(sim_id, new_status)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
