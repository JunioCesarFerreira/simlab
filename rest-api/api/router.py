from fastapi import APIRouter, Depends
from api.endpoints import experiment, simulation, generation, source, files, campaign, problems, synthetic, infra, scalability
from api.auth import get_api_key

api_router = APIRouter(dependencies=[Depends(get_api_key)])
api_router.include_router(source.router, prefix="/sources", tags=["sources"])
api_router.include_router(experiment.router, prefix="/experiments", tags=["experiments"])
api_router.include_router(campaign.router, prefix="/campaigns", tags=["campaigns"])
api_router.include_router(generation.router, prefix="/generations", tags=["generations"])
api_router.include_router(simulation.router, prefix="/simulations", tags=["simulations"])
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(problems.router, prefix="/problems", tags=["problems"])
api_router.include_router(synthetic.router, prefix="/synthetic", tags=["synthetic"])
api_router.include_router(infra.router, prefix="/infra", tags=["infrastructure"])
api_router.include_router(
    scalability.router, prefix="/scalability", tags=["scalability"])
