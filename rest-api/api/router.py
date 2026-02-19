from fastapi import APIRouter, Depends
from api.endpoints import experiment, simulation, batch, source, files, analysis
from api.auth import get_api_key

api_router = APIRouter(dependencies=[Depends(get_api_key)])
api_router.include_router(source.router, prefix="/sources", tags=["sources"])
api_router.include_router(experiment.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(batch.router, prefix="/batch", tags=["batch"])
api_router.include_router(simulation.router, prefix="/simulations", tags=["simulations"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
