import os
import random
from threading import Thread
from datetime import datetime
from pathlib import Path
from bson import ObjectId

from strategy.base import EngineStrategy
from pylib.mongo_db import EnumStatus
from pylib.dto import Simulation, SimulationConfig, Generation
from pylib import plot_network

from lib.random_network_methods import network_gen
from lib.build_input_sim_cooja import create_files

# NSGA utils
from .util.nsga.fast_nondominated_sort import fast_nondominated_sort
from .util.nsga.niching_selection import generate_reference_points, niching_selection
# Variation operators (real-coded)
from .util.genetic_operators.crossover.simulated_binary_crossover import make_sbx_crossover
from .util.genetic_operators.mutation.polynomial_mutation import make_polynomial_mutation


class NSGA3LoopStrategy(EngineStrategy):
    """
    Estratégia de loop NSGA-III integrada ao SimLab.

    - Gera uma população inicial (posições dos motes fixos em 2D).
    - Enfileira simulações no MongoDB (Change Streams disparam execução no master-node).
    - Ao receber os resultados (status DONE), calcula os objetivos,
      executa seleção NSGA-III com niching por pontos de referência,
      cria a próxima geração (offspring via SBX + Polynomial Mutation),
      e repete até atingir `number_of_generations`.
    """

    def __init__(self, experiment: dict, mongo):
        super().__init__(experiment, mongo)
        self._watch_thread: Thread | None = None
        self._stop_flag: bool = False

        # --- parâmetros do experimento ---
        params = experiment.get("parameters", {}) or {}
        self.population_size: int = int(params.get("population_size", params.get("number_of_generations", 10)))
        self.max_generations: int = int(params.get("number_of_generations", 5))
        self.num_of_motes: int = int(params.get("number_of_fixed_motes", 10))
        self.region: tuple[float, float, float, float] = tuple(params.get("region", (-100.0, -100.0, 100.0, 100.0)))  # (x1,y1,x2,y2)
        self.radius: float = float(params.get("radius_of_reach", params.get("radiusOfReach", 50.0)))
        self.interf: float = float(params.get("radius_of_interference", params.get("radiusOfInter", 60.0)))
        self.mobile_motes = params.get("mobileMotes", [])
        self.duration: int = int(params.get("duration", 120))

        # objetivos: chaves a extrair do documento de simulação concluída
        # Se não informado explicitamente, tentaremos algumas chaves comuns na análise (ajuste conforme seu pipeline)
        self.objective_keys: list[str] = list(params.get("objective_keys", [])) or [
            # Ajuste para seu pipeline de métricas:
            # Ex.: "avg_latency_ms", "energy_mJ", "packet_loss"
        ]

        # --- estado do loop ---
        self._exp_id: ObjectId | None = None
        self._gen_index: int = 0
        self._gen_id: ObjectId | None = None

        # população corrente como lista de indivíduos (cada indivíduo é um vetor [x0,y0,x1,y1,...])
        self.current_population: list[list[float]] = []
        # mapeia simulation_id(str) -> índice do indivíduo na população
        self.sim_id_to_index: dict[str, int] = {}
        # resultados coletados da geração atual: idx -> lista[float] objetivos (minimização)
        self.objectives_buffer: dict[int, list[float]] = {}

        # operadores
        bounds = self._gene_bounds()
        self._sbx = make_sbx_crossover(eta=20.0, bounds=bounds)
        self._poly = make_polynomial_mutation(eta=25.0, bounds=bounds, per_gene_prob=1.0 / (2 * self.num_of_motes))

    # ------------------------------
    # API do EngineStrategy
    # ------------------------------
    def start(self):
        """
        Inicializa a população e cria Geração 1 com `population_size` simulações.
        """
        self._exp_id = ObjectId(self.experiment["_id"]) if isinstance(self.experiment.get("_id"), (str, bytes)) else self.experiment.get("_id")
        if not isinstance(self._exp_id, ObjectId):
            self._exp_id = ObjectId(str(self.experiment.get("_id")))
        self._gen_index = 1

        # População inicial
        self.current_population = [self._random_individual() for _ in range(self.population_size)]

        # Enfileira simulações para a geração 1
        self._spawn_generation(self.current_population, self._gen_index)

        # Inicia watcher para receber os resultados desta geração
        self._start_watcher()

    def on_simulation_result(self, result_doc: dict):
        """
        Recebe um documento de simulação (status DONE) do Change Stream.
        """
        if self._stop_flag or self._gen_id is None:
            return

        # Garante que este resultado é da geração atual
        if str(result_doc.get("generation_id")) != str(self._gen_id):
            return

        sim_id = str(result_doc.get("_id"))
        idx = self.sim_id_to_index.get(sim_id)
        if idx is None:
            return

        obj = self._extract_objectives(result_doc)
        if obj is None:
            # Não há objetivos — ignore ou considere erro
            return

        self.objectives_buffer[idx] = obj

        # Checa conclusão da geração
        if len(self.objectives_buffer) >= len(self.current_population):
            self._on_generation_completed()

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
            # Este método do repo já abre o Change Stream com pipeline fixo (status == DONE)
            def _callback(result_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.on_simulation_result(result_doc)
                except Exception as e:
                    print(f"[NSGA-III] Erro no callback do watcher: {e}")

            print("[NSGA-III] Iniciando watcher de simulações (DONE).")
            self.mongo.simulation_repo.watch_simulations(_callback)

        self._watch_thread = Thread(target=_run, daemon=True)
        self._watch_thread.start()

    # ------------------------------
    # Geração / Enfileiramento
    # ------------------------------
    def _spawn_generation(self, population: list[list[float]], gen_index: int):
        """
        Cria a `Generation` e insere `population_size` simulações no MongoDB.
        """
        assert self._exp_id is not None
        exp_oid = self._exp_id

        gen_doc: Generation = {
            "index": gen_index,
            "experiment_id": exp_oid,
            "status": EnumStatus.BUILDING,
            "start_time": datetime.now(),
            "end_time": None,
            "simulations_ids": []
        }
        gen_oid: ObjectId = self.mongo.generation_repo.insert(gen_doc)
        self._gen_id = gen_oid

        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        simulation_ids: list[ObjectId] = []
        self.sim_id_to_index.clear()
        self.objectives_buffer.clear()

        for i, genome in enumerate(population):
            fixed = self._genome_to_fixed_motes(genome)
            config: SimulationConfig = {
                "name": f"nsga3-g{gen_index}-{i}",
                "duration": self.duration,
                "radiusOfReach": self.radius,
                "radiusOfInter": self.interf,
                "region": self.region,
                "simulationElements": {
                    "fixedMotes": fixed,
                    "mobileMotes": self.mobile_motes
                }
            }

            files_ids = create_files(config, self.mongo.fs_handler)
            image_tmp_path = tmp_dir / f"topology-{exp_oid}-{gen_oid}-{i}.png"
            plot_network.plot_network_save_from_sim(str(image_tmp_path), config)
            topology_picture_id = self.mongo.fs_handler.upload_file(
                str(image_tmp_path),
                f"topology-{exp_oid}-{gen_oid}-{i}"
            )
            if os.path.exists(image_tmp_path):
                os.remove(image_tmp_path)

            sim_doc: Simulation = {
                "id": i,
                "experiment_id": exp_oid,
                "generation_id": gen_oid,
                "status": EnumStatus.WAITING,
                "start_time": None,
                "end_time": None,
                "parameters": config,
                "pos_file_id": files_ids.get("pos_file_id", ""),
                "csc_file_id": files_ids.get("csc_file_id", ""),
                "topology_picture_id": topology_picture_id,
                "log_cooja_id": "",
                "runtime_log_id": "",
                "csv_log_id": ""
            }
            sim_oid = self.mongo.simulation_repo.insert(sim_doc)
            simulation_ids.append(sim_oid)
            self.sim_id_to_index[str(sim_oid)] = i

        # Atualiza geração/experimento
        self.mongo.generation_repo.update(gen_oid, {
            "simulations_ids": [str(_id) for _id in simulation_ids],
            "status": EnumStatus.WAITING
        })
        self.mongo.generation_repo.mark_waiting(gen_oid)

        self.mongo.experiment_repo.update(str(exp_oid), {
            "status": EnumStatus.RUNNING,
            "start_time": datetime.now() if self._gen_index == 1 else None,
            "generations_ids": [str(gen_oid)] if self._gen_index == 1 else None
        })

        print(f"[NSGA-III] Geração {gen_index} enfileirada com {len(population)} simulações.")

    # ------------------------------
    # Conclusão de geração e evolução
    # ------------------------------
    def _on_generation_completed(self):
        """
        Ao concluir todas as simulações da geração corrente, decide próxima população
        via NSGA-III (niching) e cria a próxima geração, ou finaliza o experimento.
        """
        assert self._gen_id is not None
        print(f"[NSGA-III] Geração {self._gen_index} concluída. Selecionando próxima população...")

        # Marca geração DONE
        self.mongo.generation_repo.update(str(self._gen_id), {
            "status": EnumStatus.DONE,
            "end_time": datetime.now()
        })

        # Constrói matriz de objetivos alinhada à população
        # Assumimos minimização em todos os objetivos
        indices = list(range(len(self.current_population)))
        objectives: list[list[float]] = [self.objectives_buffer[i] for i in indices]

        import numpy as np
        F = np.array(objectives, dtype=float)
        M = F.shape[1]

        # 1) Ordenação por frentes
        fronts = fast_nondominated_sort(objectives)

        # 2) Monta população escolhida (elitismo)
        selected_idx: list[int] = []
        niche_count: dict[int, int] = {}

        # Gera pontos de referência (escolhe p mínimo tal que |H| >= pop_size)
        p = 1
        while True:
            H = generate_reference_points(M, p)
            if len(H) >= self.population_size or p > 10:
                break
            p += 1

        for front in fronts:
            if len(selected_idx) + len(front) < self.population_size:
                selected_idx.extend(front)
                # atualiza contagem de nichos para os selecionados deste front
                # (associação apenas para estatística; a regra canônica considera associação na fronteira parcial)
                # aqui associamos todos para manter equilíbrio nas próximas iterações
                # Obs: operação barata para tamanhos moderados
                assoc_idx, _ = self._associate_indices(F, H, front)
                for nid in assoc_idx:
                    niche_count[nid] = niche_count.get(nid, 0) + 1
            else:
                remaining = self.population_size - len(selected_idx)
                if remaining > 0:
                    # Aplicar niching selection nesta última fronteira parcial
                    partial_selected = niching_selection(front, F, remaining, H, dict(niche_count))
                    selected_idx.extend(partial_selected)
                break

        # 3) Gera offspring por variação (SBX + Polynomial Mutation)
        parents = [self.current_population[i] for i in selected_idx]
        offspring = self._produce_offspring(parents, self.population_size)

        # Próxima geração
        if self._gen_index >= self.max_generations:
            # Finaliza experimento
            self._finalize()
            return

        self._gen_index += 1
        self._spawn_generation(offspring, self._gen_index)
        # watcher já está ativo; limpar buffers para nova geração
        self.objectives_buffer.clear()
        self.sim_id_to_index.clear()
        self.current_population = offspring

    # ------------------------------
    # Helpers de variação e extração
    # ------------------------------
    def _produce_offspring(self, parents: list[list[float]], n_children: int) -> list[list[float]]:
        rng = random.Random()
        children: list[list[float]] = []
        while len(children) < n_children:
            i, j = rng.randrange(len(parents)), rng.randrange(len(parents))
            mom, dad = parents[i], parents[j]
            # Crossover (90%) + Mutation
            if rng.random() < 0.9:
                c1, c2 = self._sbx(mom, dad, rng)
            else:
                c1, c2 = list(mom), list(dad)
            # Mutation
            if rng.random() < 0.2:
                c1 = self._poly(c1, rng)
            if rng.random() < 0.2:
                c2 = self._poly(c2, rng)
            children.append(c1)
            if len(children) < n_children:
                children.append(c2)
        return children[:n_children]

    def _random_individual(self) -> list[float]:
        pts = network_gen(amount=self.num_of_motes, region=self.region, radius=self.radius)
        # Flatten [ (x,y), ... ] -> [x0,y0,x1,y1,...]
        genome: list[float] = []
        for (x, y) in pts:
            genome.extend([float(x), float(y)])
        return genome

    def _genome_to_fixed_motes(self, genome: list[float]) -> list[dict]:
        fixed: list[dict] = []
        for j in range(self.num_of_motes):
            x = float(genome[2*j])
            y = float(genome[2*j + 1])
            fixed.append({
                "name": f"m{j}",
                "position": [x, y],
                "sourceCode": "default",
                "radiusOfReach": self.radius,
                "radiusOfInter": self.interf
            })
        return fixed

    def _gene_bounds(self) -> list[tuple[float, float]]:
        x1, y1, x2, y2 = self.region
        bounds: list[tuple[float, float]] = []
        for _ in range(self.num_of_motes):
            bounds.append((x1, x2))  # x
            bounds.append((y1, y2))  # y
        return bounds

    def _extract_objectives(self, result_doc: dict) -> list[float] | None:
        """
        Extracts objective vector (minimization) from DONE simulation doc.
        Priority:
        1) 'objectives' as list[float]
        2) 'objectives' as dict -> follow self.objective_keys
        3) 'metrics'    as dict -> follow self.objective_keys
        """
        obj = result_doc.get("objectives")
        # list already in order
        if isinstance(obj, (list, tuple)) and all(isinstance(v, (int, float)) for v in obj):
            return [float(v) for v in obj]

        # try dict by keys (from transform_config)
        if isinstance(obj, dict) and self.objective_keys:
            try:
                return [float(obj[k]) for k in self.objective_keys]
            except Exception:
                pass

        metrics = result_doc.get("metrics") or {}
        if isinstance(metrics, dict) and self.objective_keys:
            try:
                return [float(metrics[k]) for k in self.objective_keys]
            except Exception:
                pass

        print("[NSGA-III] Warning: objectives not found. Set 'objective_keys' via TransformConfig.")
        return None

    def _associate_indices(self, F, H, indices: list[int]) -> tuple[list[int], list[float]]:
        """
        Associa `indices` (índices absolutos na população) a niches de H.
        Retorna (niche_ids, distances) alinhados à ordem de `indices`.
        """
        import numpy as np
        sub = F[indices, :]
        from .util.nsga.niching_selection import associate_to_niches
        niche_idx, niche_dist = associate_to_niches(sub, H)
        return list(map(int, niche_idx)), list(map(float, niche_dist))

    def _finalize(self):
        assert self._exp_id is not None
        print(f"[NSGA-III] Experimento {self._exp_id} finalizado.")
        self.mongo.experiment_repo.update(str(self._exp_id), {
            "status": EnumStatus.DONE,
            "end_time": datetime.now()
        })
        # encerra watcher
        self._stop_flag = True
        if self._watch_thread and self._watch_thread.is_alive():
            self._watch_thread.join(timeout=1.0)
