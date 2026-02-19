import logging
from typing import Any
from bson import ObjectId, errors

from mongo.connection import MongoDBConnection

log = logging.getLogger(__name__)

class Pareto:
    @staticmethod
    def dominates(
            a: dict[str, float], 
            b: dict[str, float], 
            goals: dict[str, str]
    ) -> bool:
        """
        Returns True if a dominates b.
        
        goals: {objective_name: "min" | "max"}
        """
        better_or_equal = True
        strictly_better = False

        for k, goal in goals.items():
            if goal == "min":
                if a[k] > b[k]:
                    better_or_equal = False
                if a[k] < b[k]:
                    strictly_better = True
            else:
                if a[k] < b[k]:
                    better_or_equal = False
                if a[k] > b[k]:
                    strictly_better = True

        return better_or_equal and strictly_better


    @staticmethod
    def pareto_front(
        items: list[dict[str, Any]],
        goals: dict[str, str]
    ) -> list[dict[str, Any]]:
        front = []

        for i, a in enumerate(items):
            dominated = False
            for j, b in enumerate(items):
                if i == j:
                    continue
                if Pareto.dominates(b["objectives"], a["objectives"], goals):
                    dominated = True
                    break
            if not dominated:
                front.append(a)

        return front
    
    
    @staticmethod
    def to_minimization(
        objectives: dict[str, float],
        goals: dict[str, str]
    ) -> dict[str, float]:
        """
        Convert objectives to minimization space.
        - min objectives: unchanged
        - max objectives: sign-inverted
        """
        out = {}
        for k, v in objectives.items():
            goal = goals.get(k, "min")
            if goal == "max":
                out[k] = -v
            else:
                out[k] = v
        return out
    
    
    @staticmethod
    def dominates_min(a: dict[str, float], b: dict[str, float]) -> bool:
        better_or_equal = True
        strictly_better = False

        for k in a.keys():
            if a[k] > b[k]:
                better_or_equal = False
            if a[k] < b[k]:
                strictly_better = True

        return better_or_equal and strictly_better


    @staticmethod
    def pareto_front_min(
        items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        front = []

        for i, a in enumerate(items):
            dominated = False
            for j, b in enumerate(items):
                if i == j:
                    continue
                if Pareto.dominates_min(b["objectives"], a["objectives"]):
                    dominated = True
                    break
            if not dominated:
                front.append(a)

        return front

#------------------------------------------------------------------------------------------------------


class AnalyticsRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def get_pareto_front(
        self,
        experiment_id: str
    ) -> list[dict[str, Any]]:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return []

        with self.connection.connect() as db:
            experiment = db["experiments"].find_one({"_id": oid})
            if not experiment:
                log.error("Experiment not found")
                return []

            goals = experiment.get("objectives_goals", {})
            runs = list(db["runs"].find({"experiment_id": oid}))

            items = []
            for run in runs:
                items.append({
                    "run_id": str(run["_id"]),
                    "objectives": run.get("objectives", {})
                })

            pareto_front = Pareto.pareto_front(items, goals)
            return pareto_front
        
        
    def get_pareto_per_generation(
        self,
        experiment_id: str
    ) -> dict[int, list[dict[str, Any]]]:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return []
        with self.connection.connect() as db:
            experiment = db.experiments.find_one({"_id": oid})
                        
            if experiment is None:
                raise ValueError("Experiment not found")

            # Map objective -> goal (min/max)
            goals = {
                obj["name"]: obj["goal"]
                for obj in experiment["data_conversion_config"]["objectives"]
            }
            
            pareto_by_generation = {}

            generations = db.generations.find(
                {"_id": {"$in": experiment["generations"]}},
                sort=[("index", 1)]
            )
            
            gen_list = list(generations)

            for gen in gen_list:
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
                
                candidates = []
                for sim in simulations:
                    candidates.append({
                        "simulation_id": str(sim["_id"]),
                        "objectives": sim["objectives"]
                    })

                pareto = Pareto.pareto_front(candidates, goals)

                pareto_by_generation[gen["index"]] = pareto

            return pareto_by_generation
        
        
    def get_pareto_per_generation_only_min(
        self,
        experiment_id: str
    ) -> dict[int, list[dict[str, Any]]]:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return []
        with self.connection.connect() as db:
            experiment = db.experiments.find_one({"_id": oid})
                        
            if experiment is None:
                raise ValueError("Experiment not found")

            # Map objective -> goal (min/max)
            goals = {
                obj["name"]: obj["goal"]
                for obj in experiment["data_conversion_config"]["objectives"]
            }
            
            pareto_by_generation = {}

            generations = db.generations.find(
                {"_id": {"$in": experiment["generations"]}},
                sort=[("index", 1)]
            )
            
            gen_list = list(generations)

            for gen in gen_list:
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
                
                candidates = []
                for sim in simulations:
                    candidates.append({
                        "simulation_id": str(sim["_id"]),
                        "objectives": Pareto.to_minimization(sim["objectives"], goals)
                    })

                pareto = Pareto.pareto_front_min(candidates)

                pareto_by_generation[gen["index"]] = pareto

            return pareto_by_generation
        
        
    def get_individuals_per_generation(
        self,
        experiment_id: str
    ) -> dict[int, list[dict[str, Any]]]:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return []
        with self.connection.connect() as db:
            experiment = db.experiments.find_one({"_id": oid})
                        
            if experiment is None:
                raise ValueError("Experiment not found")
            
            individuals_per_gen = {}

            generations = db.generations.find(
                {"_id": {"$in": experiment["generations"]}},
                sort=[("index", 1)]
            )
            
            gen_list = list(generations)

            for gen in gen_list:
                simulations = list(
                    db.simulations.find(
                        {
                            "_id": {"$in": [ObjectId(oid) for oid in list(gen["simulations_ids"])]}
                        },
                        {
                            "_id": 1,
                            "objectives": 1,
                            "status": 1,
                            "start_time": 1,
                            "end_time": 1,
                            "topology_picture_id": 1
                        }
                    )
                )
                
                candidates = []
                for sim in simulations:
                    candidates.append({
                        "simulation_id": str(sim["_id"]),
                        "status": sim["status"],
                        "start_time": sim["start_time"],
                        "end_time": sim["end_time"],
                        "objectives": sim["objectives"],
                        "topology_picture_id": str(sim["topology_picture_id"])
                    })

                individuals_per_gen[gen["index"]] = candidates

            return individuals_per_gen