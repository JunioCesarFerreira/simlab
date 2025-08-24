import os
from threading import Thread
from datetime import datetime
from pathlib import Path
from bson import ObjectId

from strategy.base import EngineStrategy
from lib.build_input_sim_cooja import create_files
from lib.random_network_methods import network_gen
from pylib.dto import Simulation, SimulationConfig, Generation
from pylib.mongo_db import EnumStatus
from pylib import plot_network


class GeneratorRandomStrategy(EngineStrategy):
    """
    Gera topologias aleatórias, cria simulações e monitora via Change Stream
    (usando SimulationRepository.watch_simulations). Quando todas as simulações
    da geração estiverem finalizadas (DONE ou ERROR), marca geração e experimento
    como DONE.
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
    # Criação da geração e das simulações
    # ---------------------------------
    def start(self):
        exp_oid: ObjectId = self.experiment["_id"]
        self._exp_id = exp_oid

        params = self.experiment.get("parameters", {})
        num = int(params.get("number", 10))
        size = int(params.get("size", 10))
        region = tuple(params.get("region", (-100, -100, 100, 100)))
        radius = float(params.get("radius", 50))
        interf = float(params.get("interf", 60))

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

        for i in range(num):
            # 1) Gera topologia
            points = network_gen(amount=size, region=region, radius=radius)
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

            # 2) Monta config da simulação
            config: SimulationConfig = {
                "name": f"auto-{i}",
                "duration": 120,
                "radiusOfReach": radius,
                "radiusOfInter": interf,
                "region": region,
                "simulationElements": {
                    "fixedMotes": fixed,
                    "mobileMotes": []
                }
            }

            # 3) Gera arquivos + imagem da topologia
            files_ids = create_files(config, self.mongo.fs_handler)
            image_tmp_path = tmp_dir / f"{exp_oid}-{gen_oid}-{i}.png"
            plot_network.plot_network_save_from_sim(str(image_tmp_path), config)
            topology_picture_id = self.mongo.fs_handler.upload_file(
                str(image_tmp_path),
                f"topology-{exp_oid}-{gen_oid}-{i}"
            )
            os.remove(image_tmp_path)

            # 4) Insere simulação
            sim_doc: Simulation = {
                "id": i,  # índice sequencial local
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
            simulation_ids.append(sim_oid)
        
        # 5) Atualiza geração e experimento
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

        # 6) Estado interno para monitoramento
        self.number_of_simulations = len(simulation_ids)
        self.counter = 0
        self.pending = set(simulation_ids)

        # 7) Inicia watcher (escuta DONE) e filtra por geração no callback
        self._start_watcher()

    # ---------------------------------
    # Watcher: usa SimulationRepository.watch_simulations
    # ---------------------------------
    def _start_watcher(self):
        # encerra watcher anterior, se houver
        self._stop_flag = False
        if self._watch_thread and self._watch_thread.is_alive():
            self._stop_flag = True
            self._watch_thread.join(timeout=1.0)
            self._stop_flag = False

        def _run():
            # Este método do repo já abre o Change Stream com pipeline fixo (status == DONE)
            # Chamará self._on_sim_change(change) para cada alteração.
            self.mongo.simulation_repo.watch_simulations(self._on_sim_change)

        self._watch_thread = Thread(target=_run, name="simulations-watcher", daemon=True)
        self._watch_thread.start()

    def _on_sim_change(self, change: dict):
        """Callback para cada evento do Change Stream (status == DONE)."""
        if self._stop_flag:
            return

        full = change.get("fullDocument") or {}
        gen_id = full.get("generation_id")
        sim_oid = full.get("_id")
        status = full.get("status")

        # filtra para a geração atual
        if self._gen_id is None or gen_id != self._gen_id:
            return

        # telemetria opcional
        print(f"[Watcher] sim {sim_oid} -> {status}")

        # atualiza contadores locais
        if isinstance(sim_oid, ObjectId) and sim_oid in self.pending:
            self.pending.remove(sim_oid)
            self.counter += 1

        # tenta finalizar se tudo terminou (DONE/ERROR)
        self._try_finalize()

    # ---------------------------------
    # Reconciliação e finalização
    # ---------------------------------
    def _try_finalize(self):
        """Verifica no banco se restam simulações não-finalizadas desta geração."""
        if self._gen_id is None or self._exp_id is None:
            return

        # qualquer status que não seja DONE/ERROR conta como pendente
        with self.mongo.simulation_repo.connection.connect() as db:
            remaining = db["simulations"].count_documents({
                "generation_id": self._gen_id,
                "status": {"$nin": [EnumStatus.DONE, EnumStatus.ERROR]}
            })

        if remaining == 0:
            # marca geração como DONE
            self.mongo.generation_repo.update(str(self._gen_id), {
                "status": EnumStatus.DONE,
                "end_time": datetime.now()
            })
            # marca experimento como DONE
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.DONE,
                "end_time": datetime.now()
            })
            # sinaliza parada do watcher
            self._stop_flag = True
            print("[Watcher] geração/experimento finalizados (DONE).")

    def on_simulation_result(self, result_doc: dict):
        # não fará callback
        pass

    # ---------------------------------
    # Encerramento explícito
    # ---------------------------------
    def stop(self):
        self._stop_flag = True
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=1.0)
