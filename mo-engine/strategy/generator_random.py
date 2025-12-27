import os
import logging
from threading import Thread
from datetime import datetime
from pathlib import Path
from bson import ObjectId

from strategy.base import EngineStrategy
from lib.build_input_sim_cooja import create_files
from lib.random_network_methods import network_gen
from pylib.dto.database import Simulation, Generation, SimulationConfig
from pylib.mongo_db import EnumStatus
from pylib import plot_network

log = logging.getLogger(__name__)

class GeneratorRandomStrategy(EngineStrategy):
    """
    Generates random topologies, creates simulations, and monitors them via the Change Stream
    (using SimulationRepository.watch_simulations). When all simulations
    in the generation are complete (DONE or ERROR), mark the generation and experiment
    as DONE.
    """

    def __init__(self, experiment, mongo):
        super().__init__(experiment, mongo)
        self.counter: int = 0
        self.number_of_simulations: int = 0
        self.pending: set[ObjectId] = set()

        self._watch_thread: Thread | None = None
        self._stop_flag: bool = False

        self._exp_id: ObjectId | None = None
        self._gen_id: ObjectId | None = None

    # ---------------------------------
    # Starts a random generations of simulations
    # ---------------------------------
    def start(self):
        exp_oid: ObjectId = self.experiment["_id"]
        self._exp_id = exp_oid

        params = self.experiment.get("parameters", {})
        num_of_gen = int(params.get("number_of_generations", 10))
        num_of_motes = int(params.get("number_of_fixed_motes", 10))
        region = tuple(params.get("region", (-100, -100, 100, 100)))
        radius = float(params.get("radius_of_reach", 50))
        interf = float(params.get("radius_of_interference", 60))
        mobile_motes = params.get("mobileMotes", [])

        gen: Generation = {
            "index": 1,
            "experiment_id": exp_oid,
            "status": EnumStatus.BUILDING,
            "start_time": datetime.now(),
            "end_time": None,
            "simulations_ids": []
        }

        gen_oid: ObjectId = self.mongo.generation_repo.insert(gen)
        self._gen_id = gen_oid

        simulation_ids: list[ObjectId] = []
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_of_gen):
            # 1. Generates topology
            points = network_gen(amount=num_of_motes, region=region, radius=radius)
            fixed = [
                {
                    "name": f"m{j}",
                    "position": [x, y],
                    "sourceCode": "default",
                    "radiusOfReach": radius,
                    "radiusOfInter": interf
                }
                for j, (x, y) in enumerate(points)
            ]

            # 2. Assembly simulation config
            config: SimulationConfig = {
                "name": f"auto-{i}",
                "duration": 120,
                "radiusOfReach": radius,
                "radiusOfInter": interf,
                "region": region,
                "simulationElements": {
                    "fixedMotes": fixed,
                    "mobileMotes": mobile_motes
                }
            }

            # 3. Create and registry files
            files_ids = create_files(config, self.mongo.fs_handler)
            image_tmp_path = tmp_dir / f"{exp_oid}-{gen_oid}-{i}.png"
            plot_network.plot_network_save_from_sim(str(image_tmp_path), config)
            topology_picture_id = self.mongo.fs_handler.upload_file(
                str(image_tmp_path),
                f"topology-{exp_oid}-{gen_oid}-{i}"
            )
            os.remove(image_tmp_path)

            # 4. Insert Simulation
            sim_doc: Simulation = {
                "id": i, # sequential index
                "experiment_id": exp_oid,
                "generation_id": gen_oid,
                "status": EnumStatus.WAITING,
                "start_time": None,
                "end_time": None,
                "parameters": config,
                "pos_file_id": files_ids["pos_file_id"],
                "csc_file_id": files_ids["csc_file_id"],
                "topology_picture_id": topology_picture_id,
                "log_cooja_id": "",
                "runtime_log_id": "",
                "csv_log_id": ""
            }

            sim_oid = self.mongo.simulation_repo.insert(sim_doc)
            log.info(f"sim_oid={sim_oid}")
            simulation_ids.append(sim_oid)
        
        # 5. Update generation and experiment
        self.mongo.generation_repo.update(gen_oid, {
            "simulations_ids": [str(_id) for _id in simulation_ids],
            "status": EnumStatus.WAITING
        })
        self.mongo.generation_repo.mark_waiting(gen_oid)

        self.mongo.experiment_repo.update(str(exp_oid), {
            "status": EnumStatus.RUNNING,
            "start_time": datetime.now(),
            "generations_ids": [str(gen_oid)]
        })

        # 6. Internal state for monitoring
        self.number_of_simulations = len(simulation_ids)
        self.counter = 0
        self.pending = set(simulation_ids)

        # 7. Starts watcher (listens for DONE) and filters by generation in callback
        self._stop_flag = False # closes previous watcher, if any
        if self._watch_thread and self._watch_thread.is_alive():
            self._stop_flag = True
            self._watch_thread.join(timeout=1.0)
            self._stop_flag = False

        def _run():
            self.mongo.simulation_repo.watch_status_done(self.event_simulation_done)

        self._watch_thread = Thread(target=_run, name="simulations-watcher", daemon=True)
        self._watch_thread.start()
            
    # ---------------------------------
    # Event: Simulations DONE
    # ---------------------------------
    def event_simulation_done(self, sim_doc: dict):
        """Callback for each Change Stream event (status == DONE)."""
        if self._stop_flag:
            return

        full = sim_doc.get("fullDocument") or {}
        gen_id = full.get("generation_id")
        sim_oid = full.get("_id")
        status = full.get("status")

        if self._gen_id is None or gen_id != self._gen_id:
            return

        log.info(f"[Watcher] sim {sim_oid} -> {status}")

        # 1. update local counters
        if isinstance(sim_oid, ObjectId) and sim_oid in self.pending:
            self.pending.remove(sim_oid)
            self.counter += 1

        if self._gen_id is None or self._exp_id is None:
            return

        # 2. any status other than DONE/ERROR counts as pending
        with self.mongo.simulation_repo.connection.connect() as db:
            remaining = db["simulations"].count_documents({
                "generation_id": self._gen_id,
                "status": {"$nin": [EnumStatus.DONE, EnumStatus.ERROR]}
            })

        if remaining == 0:
            # 3. change generation status DONE
            self.mongo.generation_repo.update(str(self._gen_id), {
                "status": EnumStatus.DONE,
                "end_time": datetime.now()
            })
            # 4. change experiment status DONE
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.DONE,
                "end_time": datetime.now()
            })
            # 5. signals watcher stop
            self._stop_flag = True
            log.info("[Watcher] generation/experiment completed (DONE).")

    # ---------------------------------
    # End watch thread
    # ---------------------------------
    def stop(self):
        self._stop_flag = True
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=1.0)
