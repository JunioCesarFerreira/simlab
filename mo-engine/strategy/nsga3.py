from strategy.base import EngineStrategy
from lib.random_network_methods import network_gen
from pylib.dto import Simulation, SimulationConfig, Generation
from datetime import datetime
import random
from bson import ObjectId


class NSGALoopStrategy(EngineStrategy):
    def __init__(self, experiment, mongo):
        super().__init__(experiment, mongo)
        self.results_buffer = {}
        self.generation = 0
        self.sim_id_to_solution = {}
        self.current_population = []
        self.max_generations = int(experiment["parameters"].get("generations", 3))
        self.population_size = int(experiment["parameters"].get("population_size", 10))
        self.region = tuple(experiment["parameters"].get("region", (0, 0, 100, 100)))
        self.radius = float(experiment["parameters"].get("radius", 25.0))
        self.exp_id = str(experiment["_id"])

    def start(self):
        print(f"[NSGA-III] Iniciando experimento {self.exp_id}")
        self._initialize_population()
        self._generate_and_queue_simulations()

    def _initialize_population(self):
        print(f"[NSGA-III] Inicializando população com {self.population_size} indivíduos.")
        self.current_population = [
            network_gen(amount=self.population_size, region=self.region, radius=self.radius)
            for _ in range(self.population_size)
        ]

    def _generate_and_queue_simulations(self):
        simulation_ids = []

        for i, individual in enumerate(self.current_population):
            fixed = [
                {
                    "name": f"m{j}",
                    "position": [x, y],
                    "sourceCode": "default",
                    "radiusOfReach": self.radius,
                    "radiusOfInter": self.radius
                }
                for j, (x, y) in enumerate(individual)
            ]

            config: SimulationConfig = {
                "name": f"gen{self.generation}-ind{i}",
                "duration": 300.0,
                "radiusOfReach": self.radius,
                "radiusOfInter": self.radius,
                "region": self.region,
                "simulationElements": {
                    "fixedMotes": fixed,
                    "mobileMotes": []
                }
            }

            sim_doc: Simulation = {
                "id": "",
                "status": "Waiting",
                "start_time": None,
                "end_time": None,
                "parameters": config,
                "pos_file_id": "",
                "csc_file_id": "",
                "log_cooja_id": "",
                "runtime_log_id": "",
                "csv_log_id": "",
                "experiment_id": ObjectId(self.exp_id)  # necessário para rastrear
            }

            sim_id = self.mongo.simulation_repo.insert(sim_doc)
            self.sim_id_to_solution[str(sim_id)] = individual
            simulation_ids.append(str(sim_id))

        queue: Generation = {
            "id": "",
            "status": "Waiting",
            "start_time": datetime.now(),
            "end_time": None,
            "simulations_ids": simulation_ids
        }

        self.mongo.generation_repo.insert(queue)
        print(f"[NSGA-III] Geração {self.generation} enfileirada com {len(simulation_ids)} simulações.")

    def on_simulation_result(self, result_doc: dict):
        sim_id = str(result_doc.get("simulation_id") or result_doc.get("simulationId"))
        if not sim_id or sim_id not in self.sim_id_to_solution:
            return

        self.results_buffer[sim_id] = result_doc

        if len(self.results_buffer) == self.population_size:
            print(f"[NSGA-III] Todos os resultados da geração {self.generation} recebidos.")
            self._process_generation_results()
            self.generation += 1
            self.results_buffer.clear()
            if self.generation < self.max_generations:
                self._generate_and_queue_simulations()
            else:
                self._finalize()

    def _process_generation_results(self):
        print(f"[NSGA-III] Processando resultados da geração {self.generation}...")

        # 🔧 MOCK: Seleciona nova população aleatória a partir dos resultados atuais
        selected = random.sample(list(self.sim_id_to_solution.values()), k=self.population_size)
        self.current_population = selected
        print(f"[NSGA-III] Nova população gerada (mock de seleção).")

    def _finalize(self):
        print(f"[NSGA-III] Experimento {self.exp_id} finalizado.")
        self.mongo.experiment_repo.update(self.exp_id, {
            "status": "Done",
            "end_time": datetime.now()
        })
