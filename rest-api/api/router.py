from fastapi import APIRouter, Depends
from api.endpoints import experiment, simulation, generation, source, files
from api.auth import get_api_key

api_router = APIRouter(dependencies=[Depends(get_api_key)])
api_router.include_router(source.router, prefix="/sources", tags=["sources"])
api_router.include_router(experiment.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(generation.router, prefix="/generations", tags=["generations"])
api_router.include_router(simulation.router, prefix="/simulations", tags=["simulations"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
