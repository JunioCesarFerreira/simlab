# api/endpoints/analysis.py
from fastapi import APIRouter, HTTPException
import os, sys

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()


@router.get("/experiments/{experiment_id}/paretos")
def get_paretos(experiment_id: str):
    """
    Retrieve Pareto fronts for an experiment.
    
    - Returns Pareto fronts per generation
    - Only simulations with status == "Done" are considered
    - Dominance is evaluated according to experiment objective goals
    
    Output format:
    {
        generation_index: [
            {
                "simulation_id": ObjectId,
                "objectives": { ... }
            },
            ...
        ],
        ...
    }
    """
    try:
        return factory.analytics_repo.get_pareto_per_generation(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/individuals")
def get_individuals(experiment_id: str):
    """
    Retrieve all individuals per generation for an experiment.
    
    - Includes all simulations (independent of status)
    - Intended for convergence plots, diversity analysis,
      and offline post-processing
    
    Output format:
    {
        generation_index: [
            {
                "simulation_id": ObjectId,
                "status": "...",
                "objectives": { ... }
            },
            ...
        ],
        ...
    }
    """
    try:
        return factory.analytics_repo.get_individuals_per_generation(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
