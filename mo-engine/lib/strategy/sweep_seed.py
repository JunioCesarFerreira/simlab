import os
import random
import logging
import threading
from threading import Thread
from datetime import datetime
from pathlib import Path
from bson import ObjectId

from .base import EngineStrategy
from .simulation_seeds import resolve_simulation_seeds

from pylib.mongo_db import EnumStatus
from pylib.dto.database import Batch, Simulation, SimulationConfig
from pylib import plot_network

from lib.util.build_input_sim_cooja import create_files

# Problem Adapter
from lib.problem.adapter import ProblemAdapter, Chromosome
from lib.problem.resolve import build_adapter

logger = logging.getLogger(__name__)

class SweepSeedStrategy(EngineStrategy):
    def __init__(self, experiment: dict, mongo):
        super().__init__(experiment, mongo)
        self._watch_thread: Thread | None = None
        self._stop_flag: bool = False

        # --- experiment parameters ---
        params = experiment.get("parameters", {}) or {}
        algorithm_config = params.get("algorithm", {}) or {}
        problem_config = params.get("problem", {}) or {}
        simulation_config = params.get("simulation", {}) or {}
        
        src_repo_opts = experiment.get("source_repository_options", {}) or {}
        
        self.source_repository_options: dict[str, ObjectId] = {
            str(k): ObjectId(v) if isinstance(v, (str, bytes)) else v
            for k, v in src_repo_opts.items()
        }
        
        # --- simulation and algorithm parameters ---
        self._sim_duration: int = int(simulation_config.get("duration", 120))
    
        ga_random_seed: int = int(algorithm_config.get("random_seed", 42))
        self._ga_rng = random.Random(ga_random_seed)                        
        self._sim_rand_seeds = resolve_simulation_seeds(
            simulation_config=simulation_config,
            rng=self._ga_rng,
            default_count=100,
        )
        self._problem_adapter: ProblemAdapter = build_adapter(
            problem_config, 
            algorithm_config, 
            self._ga_rng
            )
                
        # prepare network metric conversor
        self._metric_conv_config = experiment.get("data_conversion_config", {}) or {}
        
        # prepare objective keys and goals
        obj = params.get("objectives", []) or []
        self._objective_keys: list[str] = [o["metric_name"] for o in obj]
        self._objective_goals: list[int] = [1 if o["goal"]=='min' else -1 for o in obj]
        if len(self._objective_keys) != len(self._objective_goals):
            raise ValueError(
                f"objective_keys ({len(self._objective_keys)}) and "
                f"objective_goals ({len(self._objective_goals)}) length mismatch"
            )
        logger.info(f"Objective keys: {self._objective_keys} with goals: {self._objective_goals}")
                    
        # --- loop state ---
        self._exp_id: ObjectId | None = None
        self._batch_id: ObjectId | None = None
        
        
# ------------------------------
# Interface EngineStrategy
# ------------------------------
    # START implementation
    def start(self):
        """
        Initializes the population and creates Generation 1 with `population_size` simulations.
        """
        self._exp_id = ObjectId(self.experiment["_id"]) if isinstance(self.experiment.get("_id"), (str, bytes)) else self.experiment.get("_id")
        if not isinstance(self._exp_id, ObjectId):
            self._exp_id = ObjectId(str(self.experiment.get("_id")))

        # Generates one individual with random genome (random topology) to extract the number of nodes and fixed positions for all generations (sweep seed strategy)
        self._model = self._problem_adapter.random_individual_generator(1)[0]

        # Enqueue Simulations
        self._batch_enqueue()

        # Starts watcher for receive results from this generation
        self._start_watcher()


    # EVENT_SIMULATION_DONE implementation
    def event_simulation_done(self, result_doc: dict):
        if self._stop_flag or self._batch_id is None:
            return

        logger.info("EVENT SIMULATION BATCH RESULT DONE")
        batch = result_doc.get("fullDocument")

        # Ensures that this result is from the current batch
        if ObjectId(batch.get("_id")) != self._batch_id:
            return

        logger.info(f"batch_id={self._batch_id} DONE")
        
        logger.info(f"[Sweep-Seed] Experiment {self._exp_id} completed.")
        self.mongo.experiment_repo.update(str(self._exp_id), {
            "status": EnumStatus.DONE,
            "end_time": datetime.now(),
            "system_message": f"Experiment {self._exp_id} completed."
        })
        # finish watcher
        self.stop()

    # STOP implementation
    def stop(self):
        self._stop_flag = True
        t = self._watch_thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._watch_thread = None

# ------------------------------
# Watcher (Change Stream)
# ------------------------------
    def _start_watcher(self):
        # encerra watcher anterior, se houver
        self._stop_flag = False
        if self._watch_thread and self._watch_thread.is_alive():
            self._stop_flag = True
            self._watch_thread.join(timeout=1.0)
            self._stop_flag = False

        def _run():
            # This method already opens the Change Stream with a fixed pipeline. (status == DONE)
            def _callback(result_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_simulation_done(result_doc)
                except Exception:
                    logger.exception(f"[Sweep-Seed] Watcher Callback Error.")

            logger.info("[Sweep-Seed] Starting Batch watcher (DONE).")
            self.mongo.batch_repo.watch_status_done(_callback)

        self._watch_thread = Thread(target=_run, daemon=True)
        self._watch_thread.start()

# ------------------------------
# Generation / Queuing
# ------------------------------
    def _batch_enqueue(self) -> None:
        assert self._exp_id is not None
        exp_oid = self._exp_id
        
        batch_simulation_ids: list[ObjectId] = [] # collect simulation ids for this generation batch
        
        # convert genome to simulation config
        simulationElements = self._problem_adapter.encode_simulation_input(self._model)        
        config: SimulationConfig = {
                "name": f"sweep-seed",
                "duration": self._sim_duration,
                "randomSeed": 0, # will be set later per simulation
                "radiusOfReach": self._problem_adapter.radius_of_reach,
                "radiusOfInter": self._problem_adapter.radius_of_inter,
                "region": self._problem_adapter.bounds,
                "simulationElements": simulationElements
            }
        
        # plot topology and upload to GridFS
        topology_picture_id = self._plot_topology(exp_oid, config)        
            
        # build and insert simulation per seed, and collect simulation ids for this genome
        simulation_ids_for_genome: list[ObjectId] = []
        for seed in self._sim_rand_seeds:
            config["randomSeed"] = seed
            sim_oid = self._insert_simulation_db(self._model, exp_oid, config, topology_picture_id)
            simulation_ids_for_genome.append(sim_oid)
            logger.info(f"SIM_OID={sim_oid} SEED={seed} for genome {self._model.get_hash()}")
            
        batch_simulation_ids.extend(simulation_ids_for_genome)
        
        # create batch for this generation
        batch_doc: Batch = {
            "index": 0,
            "status": EnumStatus.WAITING,
            "start_time": datetime.now(),
            "end_time": None,
            "simulations_ids": batch_simulation_ids
        }
        self._batch_id: ObjectId = self.mongo.batch_repo.insert(batch_doc)

        # update experiment
        self.mongo.experiment_repo.update(str(exp_oid), {
            "status": EnumStatus.RUNNING,
            "start_time": datetime.now()
        })
            
        logger.info(f"[Sweep-Seed] Batch enqueued with {len(batch_simulation_ids)} Simulations.")
                
                
    def _insert_simulation_db(self, 
            genome: Chromosome,
            exp_oid: ObjectId,
            config: SimulationConfig,
            topology_picture_id: ObjectId
            )-> ObjectId:
        files_ids = create_files(config, self.mongo.fs_handler)                     
        _, src_id = genome.get_source_by_mac_protocol(self.source_repository_options)
             
        sim_doc: Simulation = {
            "experiment_id": exp_oid,
            "individual_id": genome.get_hash(),
            "status": EnumStatus.WAITING,
            "random_seed": config.get("randomSeed", 0),
            "start_time": None,
            "end_time": None,
            "parameters": config,
            "pos_file_id": files_ids.get("pos_file_id", ""),
            "csc_file_id": files_ids.get("csc_file_id", ""),
            "source_repository_id": src_id,
            "topology_picture_id": topology_picture_id,
            "log_cooja_id": "",
            "runtime_log_id": "",
            "csv_log_id": ""
        }
        return self.mongo.simulation_repo.insert(sim_doc)


    def _plot_topology(self, exp_oid: ObjectId, config: SimulationConfig)->ObjectId:
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        image_tmp_path = tmp_dir / f"topology-{exp_oid}-sweep-seed.png"
        
        plot_network.plot_network_save_from_sim(str(image_tmp_path), config)
        
        topology_picture_id = self.mongo.fs_handler.upload_file(
            str(image_tmp_path),
            f"topology-{exp_oid}-sweep-seed"
        )
        
        if os.path.exists(image_tmp_path):
            os.remove(image_tmp_path)
            
        return topology_picture_id

