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


@router.get("/experiments/{experiment_id}/paretos_to_min")
def get_paretos(experiment_id: str):
    """
    Retrieve Pareto fronts per generation for a given experiment,
    with all objective values expressed in a minimization space.

    This endpoint returns the non-dominated solutions (Pareto fronts)
    computed independently for each generation, considering only
    simulations with status == "Done".

    Objective handling:
    - All objectives are converted to a minimization formulation.
    - Objectives originally defined as "max" in the experiment
      configuration are sign-inverted prior to dominance evaluation.
    - Objectives defined as "min" are returned unchanged.

    This guarantees that the returned objective vectors are directly
    compatible with standard multi-objective performance indicators
    such as hypervolume (HV), generational distance (GD), and IGD,
    without requiring any additional transformations on the client side.
    
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
        return factory.analytics_repo.get_pareto_per_generation_only_min(experiment_id)
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
