import gridfs
from pymongo import MongoClient
from bson import ObjectId
from typing import Any

from .pareto_methods import pareto_front


def get_pareto_per_generation(
    mongo_uri: str,
    db_name: str,
    experiment_id: str
) -> dict[int, list[dict[str, Any]]]:

    client = MongoClient(mongo_uri)
    db = client[db_name]

    experiment = db.experiments.find_one({"_id": ObjectId(experiment_id)})
    
    print("\nExperiment data:")
    print(experiment)
    print()
    
    if experiment is None:
        raise ValueError("Experiment not found")

    # Map objective -> goal (min/max)
    goals = {
        obj["name"]: obj["goal"]
        for obj in experiment["transform_config"]["objectives"]
    }
    
    print('Objectives and goals:')
    print(goals)
    print()

    pareto_by_generation = {}

    generations = db.generations.find(
        {"_id": {"$in": experiment["generations"]}},
        sort=[("index", 1)]
    )
    
    
    print("Generations found:")
    gen_list = list(generations)
    print(gen_list)
    print()

    for gen in gen_list:
        print(f"Processing Generation {gen['index']} (ID: {gen['_id']})")
        simulations = list(
            db.simulations.find(
                {
                    "_id": {"$in": [ObjectId(oid) for oid in list(gen["simulations_ids"])]},
                    "status": "Done"
                },
                {
                    "_id": 1,
                    "objectives": 1
                }
            )
        )
        
        print(f"Generation {gen['index']} - Simulations found:")
        print(simulations)  
        print()

        candidates = []
        for sim in simulations:
            candidates.append({
                "simulation_id": sim["_id"],
                "objectives": sim["objectives"],
                # Ajuste conforme sua arquitetura real
                "chromosome": sim.get("parameters", {}).get("problem")
            })

        pareto = pareto_front(candidates, goals)

        pareto_by_generation[gen["index"]] = pareto

    return pareto_by_generation



def upload_file(
    mongo_uri: str,
    db_name: str,
    path: str, 
    name: str
) -> ObjectId:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    with open(path, "rb") as f:
        file_id = fs.put(f, filename=name)
        
    return ObjectId(file_id)



def add_analysis_file_to_experiment(
    mongo_uri: str,
    db_name: str,
    experiment_id: str,
    description: str,
    file_id: ObjectId
) -> None:
    """
    Register an analysis artifact in the experiment document.

    Parameters
    ----------
    description : str
        Human-readable description of the analysis artifact (English).
    file_id : ObjectId
        GridFS file ObjectId.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]

    result = db.experiments.update_one(
        {"_id": ObjectId(experiment_id)},
        {
            "$set": {
                f"analysis_files.{description}": file_id
            }
        }
    )

    if result.matched_count == 0:
        raise ValueError("Experiment not found")