# Plano de Integração dos Modelos MILP no SimLab

> Proposta: novo item no menu lateral da GUI para visualizar modelos MILP (P2 mobile e P3 target),
> gerar experimentos em lote a partir de varreduras de parâmetros do MILP, e um novo módulo de
> execução (`milp-engine`) com suporte a Gurobi licenciado para containers.
>
> Repositórios analisados: `simlab`, `wsn-milp`, `wsn-milp-nsga3-p2`.
> Data: 2026-07-12

---

## 1. Análise dos repositórios

### 1.1 `wsn-milp` (modelos standalone)

| Arquivo | Modelo | Formato de entrada |
|---|---|---|
| `wsn-mobile/mobile.py` | **P2 mobile**: fluxo multi-período, instala candidatos `y_j`, ativa enlaces `z_ij(t)`, fluxo `x_ij(t)` com capacidade `C0(1 − k_decay·d)²` | **Já consome o formato `ProblemP2` do SimLab** (`problem.name=problem2`, `radius_of_reach`, `sink`, `candidates`, `mobile_nodes[].path_segments/is_closed/is_round_trip/speed/time_step`) |
| `wsn-target/target.py` | **P3 target**: k-cobertura (`Σ a_hj y_j ≥ k`), g-conectividade (`Σ A_ij y_i ≥ g·y_j`), fluxo com demanda `B·y_j` e big-M `M_max` | Formato Cooja legado (`simulationElements.fixedMotes`, `radiusOfCov`) — precisa adaptação para `ProblemP3` |

Parâmetros dos modelos (candidatos naturais à varredura):

- **P2 mobile**: `C0`, `kdecay`, `B`, `w_install`, `duration/T`
- **P3 target**: `w_install`, `k_cov`, `g_conn`, `B`, `M_max`

Ambos são scripts monolíticos (constantes hard-coded no topo, `gurobipy` obrigatório, plots
matplotlib no final). O núcleo de modelagem (~100 linhas cada) é diretamente portável.

### 1.2 `wsn-milp-nsga3-p2` (pipeline de varredura + lote)

É o protótipo exato do que se quer integrar, porém fora do SimLab:

- `milp/runner.py` — **varredura de parâmetros** (grade `C0 × kdecay × B`), **deduplicação por
  genótipo** (string binária dos `y_j`), **checkpoint incremental** (`checkpoint.json`) para
  retomada, aceita solução inteira em time-limit (`SolCount > 0`), gera um JSON por topologia única.
- `batch_runner/batch_runner.py` — executa as topologias em 6 containers Cooja via SSH/SCP com
  fila própria (paramiko), coleta `objectives.json` (latência RTT, energia mJ, throughput).
- `Dockerfile.milp-p2` — `python:3.12-slim` + `gurobipy`; licença via
  `GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic` com bind-mount `./gurobi.lic:/opt/gurobi/gurobi.lic:ro`.
- Gurobi: `TimeLimit=300s`, `MIPGap=1%`, `OutputFlag=0`.

**O que aproveitar**: `solve_p2()` (núcleo do modelo, quase idêntico ao `mobile.py`), o padrão
genótipo/dedup/checkpoint, e a parametrização Gurobi.
**O que descartar**: todo o `batch_runner/` — o SimLab já possui essa infraestrutura de forma
superior (master-node + change streams + GridFS + 16 workers Cooja).

### 1.3 `simlab` — pontos de encaixe existentes

A arquitetura atual já cobre ~70% da proposta:

1. **`BatchStrategy`** (`mo-engine/lib/strategy/batch.py`): estratégia que recebe cromossomos
   prontos em `parameters.chromosomes`, executa todos como uma única geração, aguarda término,
   consolida objetivos e calcula o front de Pareto. **É exatamente o executor de lote de que a
   varredura MILP precisa** — a saída do MILP (máscaras binárias) é um `ChromosomeP2`/`ChromosomeP3`
   (`{mac_protocol, mask}`) válido.
2. **Problemas P2/P3 já modelados** (`pylib/config/problems.py`, `mo-engine/lib/problem/`):
   `ProblemP2` (sink, candidates, mobile_nodes) e `ProblemP3` (sink, candidates, targets,
   radius_of_cover, k_required, g_required) correspondem 1:1 aos dois modelos MILP. Os drafts
   criados no **Problem Editor** já são a instância de entrada do MILP.
3. **Padrão de orquestração event-driven**: mo-engine observa a coleção `experiments` via change
   streams (`engine.py`); master-node observa `generations`. Um novo módulo MILP se encaixa no
   mesmo padrão observando uma nova coleção.
4. **Campaigns**: agrupam experimentos — mapeamento natural para "uma varredura MILP → N lotes".
5. **GUI**: sidebar (`components/layout/Sidebar.vue`), roteamento lazy (`app/router/index.ts`),
   clients REST por recurso (`src/api/*.ts`), wizard de lançamento (`problem-editor/launch/steps/`),
   páginas de detalhe de experimento com gráficos Pareto reutilizáveis.

**Lacunas** (o que precisa ser construído):

- Nenhum solver MILP no sistema (nenhuma dependência gurobipy/highspy).
- Nenhum conceito de "modelo" exposto na API/GUI (só "problema" e "experimento").
- A estratégia `batch` não é exposta no wizard da GUI (só NSGA-II/III e random_search).
- Nenhum mecanismo de licença Gurobi.

### 1.4 Ponto crítico de correção: ordem do genótipo

No `runner.py` vizinho, o bit *i* do genótipo segue `sorted(J, key=nome)`. No SimLab, o
`ChromosomeP2.mask[i]` refere-se a `problem.candidates[i]` (ordem da lista). **A convenção do
SimLab deve mandar**: o solver deve produzir a máscara na ordem de `candidates`, nunca ordenar por
nome. Erro aqui produziria topologias silenciosamente erradas.

Outro detalhe: o gene `mac_protocol` não existe no MILP. A varredura deve fixá-lo (default
`csma=0`) ou, opcionalmente, duplicar cada topologia para os dois MACs (checkbox na GUI).

---

## 2. Arquitetura proposta

```
GUI ──POST /milp/sweeps──▶ REST API ──insere──▶ MongoDB (milp_sweeps)
                                                    │ change stream
                                                    ▼
                                              milp-engine  (novo container)
                                              │  varre grade de parâmetros
                                              │  resolve MILP (Gurobi/HiGHS)
                                              │  deduplica genótipos + checkpoint
                                              │  progresso incremental → milp_sweeps
                                              ▼
                                    cria experiment {strategy: "batch",
                                                     parameters.chromosomes: [...]}
                                                    │ change stream (fluxo já existente)
                                                    ▼
                              mo-engine (BatchStrategy) ─▶ master-node ─▶ cooja1..16
                                                    │
                                                    ▼
                              métricas, Pareto, GUI de análise (tudo já existente)
```

Decisões de projeto:

- **Módulo separado (`milp-engine/`), não uma estratégia do mo-engine.** Motivos: isolamento da
  dependência gurobipy e da licença; solve MILP é CPU-bound e não deve competir com o loop
  evolutivo; ciclo de vida independente (uma varredura pode alimentar vários experimentos).
- **Handoff via experimento `batch`.** O milp-engine não fala com Cooja nem com o master-node;
  ele apenas materializa cromossomos e cria o documento de experimento. Todo o restante do
  pipeline (execução, persistência, análise, GUI) é reuso puro.
- **Abstração de solver.** Interface `SolverBackend` com `GurobiBackend` (padrão) e
  `HighsBackend` (fallback open-source, via `highspy`). P2/P3 são MILPs puros (binárias +
  contínuas lineares) — HiGHS resolve ambos. Permite CI sem licença e uso do lab sem Gurobi.
- **Uma varredura → um experimento batch por instância** (e opcionalmente uma Campaign agrupando,
  quando a varredura referenciar múltiplos problemas).

### 2.1 Licenciamento Gurobi em containers

Suporte em ordem de prioridade (todos via configuração, sem rebuild):

| Esquema | Mecanismo | Uso |
|---|---|---|
| **WLS (Web License Service)** — recomendado para containers | `gurobi.lic` com `WLSACCESSID/WLSSECRET/LICENSEID` montado read-only (padrão já usado no `wsn-milp-nsga3-p2`), ou env vars `GRB_WLSACCESSID/GRB_WLSSECRET/GRB_LICENSEID` | Licença acadêmica WLS é gratuita; requer acesso à internet do container |
| **Compute Server** | `GRB_LICENSE_FILE` apontando `COMPUTESERVER=host` | Solve remoto; container cliente não precisa de licença |
| **Token Server** | lic com `TOKENSERVER=host:porta` | Licenças flutuantes on-premise |
| **Trial embutido do pip `gurobipy`** | nada a configurar | Limite ~2000 vars/constraints — suficiente para instâncias pequenas e para testes |
| **HiGHS** (`solver: "highs"`) | sem licença | Fallback completo open-source |

Implementação: no startup o milp-engine valida a licença (cria um `gp.Model` de 1 variável),
grava o resultado em um documento de status (`milp_engine_status`) e a GUI exibe o estado da
licença/solver na página de Models. `docker-compose.yaml` ganha:

```yaml
  milpengine:
    build: { context: ., dockerfile: Dockerfile.milp-engine }
    environment:
      - MONGO_URI=mongodb://mongodb:27017/?replicaSet=rs0
      - MILP_SOLVER=gurobi            # gurobi | highs
      - GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
      # alternativa WLS por env: GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID
    volumes:
      - ./gurobi.lic:/opt/gurobi/gurobi.lic:ro   # opcional
    networks: [mongo-net]
```

### 2.2 Modelo de dados (MongoDB)

**Registry de modelos** — estático em código (como `PROBLEM_REGISTRY`), exposto via API:

```python
MILP_MODEL_REGISTRY = {
  "milp_p2_mobile": {
    "problem_key": "problem2",
    "title": "P2 — Mobile Coverage MILP",
    "formulation_md": "...",       # LaTeX do README do wsn-milp
    "parameters": [                 # esquema para o builder de varredura na GUI
      {"name": "C0",        "type": "float", "default": 310,  "sweepable": True},
      {"name": "kdecay",    "type": "float", "default": 0.25, "sweepable": True},
      {"name": "B",         "type": "float", "default": 25,   "sweepable": True},
      {"name": "w_install", "type": "float", "default": 1e6,  "sweepable": True},
    ],
    "solver_defaults": {"time_limit_s": 300, "mip_gap": 0.01},
  },
  "milp_p3_target": {
    "problem_key": "problem3",
    "parameters": ["w_install", "k_cov", "g_conn", "B", "M_max"],
    ...
  },
}
```

**Coleção `milp_sweeps`** (documento de trabalho do milp-engine):

```jsonc
{
  "name": "P2 sweep — instancia 40 candidatos",
  "model_key": "milp_p2_mobile",
  "problem_id": "<ObjectId de problems>",        // draft do Problem Editor
  "parameter_grid": { "C0": [10,110,310,610,1010], "kdecay": [0.9,0.5,0.25,0.1], "B": [1,25,50,75,100] },
  "fixed_parameters": { "w_install": 1e6 },
  "solver": { "backend": "gurobi", "time_limit_s": 300, "mip_gap": 0.01 },
  "batch_options": { "mac_protocols": [0], "duration": 120, "source_repository_options": {...}, "data_conversion_config": {...} },
  "status": "CREATED | RUNNING | SOLVED | ERROR | CANCELLED",
  "progress": { "total_combos": 100, "solved": 37, "infeasible": 2, "unique_genotypes": 12 },
  "checkpoint": { "combo_index": 37, "genotypes": ["01101...", ...] },   // retomada
  "solutions": [ { "params": {"C0":10,"kdecay":0.9,"B":1}, "genotype": "01101...", "n_installed": 5, "obj_value": 5000123.4, "mip_gap": 0.004, "runtime_s": 12.3, "solver_status": "OPTIMAL" } ],
  "experiment_id": "<criado ao final>",
  "campaign_id": null,
  "system_message": null
}
```

Cada solve individual fica registrado em `solutions` (inclusive duplicados, com flag), pois a
relação parâmetros→topologia é em si um resultado científico da varredura.

### 2.3 REST API

Novo router `api/endpoints/milp.py`:

| Método | Rota | Função |
|---|---|---|
| GET | `/milp/models` | Lista o registry (título, problema associado, parâmetros, formulação) |
| GET | `/milp/models/{key}` | Detalhe de um modelo |
| GET | `/milp/status` | Estado do engine/licença/solvers disponíveis |
| POST | `/milp/sweeps` | Cria varredura (valida grade, estima nº de combinações) |
| GET | `/milp/sweeps` | Lista com progresso resumido |
| GET | `/milp/sweeps/{id}` | Detalhe: progresso, soluções, link p/ experimento |
| PATCH | `/milp/sweeps/{id}/cancel` | Cancelamento cooperativo |
| DELETE | `/milp/sweeps/{id}` | Remove (se não RUNNING) |

### 2.4 Novo módulo `milp-engine/`

```
milp-engine/
├── engine.py                    # main: change stream em milp_sweeps + retomada de pendentes
├── requirements.txt             # pymongo, numpy, gurobipy, highspy
├── lib/
│   ├── models/
│   │   ├── base.py              # MilpModel: build(problem, params) -> ModelIR; genotype order = candidates
│   │   ├── p2_mobile.py         # port de solve_p2 (runner.py) lendo ProblemP2 + trajectory utils do pylib
│   │   └── p3_target.py         # port de target.py lendo ProblemP3
│   ├── solver/
│   │   ├── base.py              # SolverBackend.solve(ir, time_limit, gap) -> Solution
│   │   ├── gurobi_backend.py    # + validação de licença (WLS/lic file/compute server)
│   │   └── highs_backend.py
│   ├── sweep.py                 # produto cartesiano da grade, dedup por genótipo, checkpoint em Mongo
│   └── handoff.py               # monta experiment batch (chromosomes, problem, batch_options) e insere
└── tests/                       # instâncias mínimas (≤10 candidatos) resolvíveis pelo trial/HiGHS
```

Nota sobre trajetórias: o P2 precisa amostrar posições dos móveis em t=1..T. Reutilizar
`pylib`/`mo-engine/lib/util/trajectory_sampling.py` (ou mover a função para `pylib` se hoje só
existir no mo-engine) em vez de portar `make_mobile_trajectory_fn` do vizinho — garante que o MILP
e o simulador vejam as MESMAS trajetórias.

### 2.5 GUI

- **Sidebar**: novo item `Models` (rota `/models`) entre `Problems` e `Synthetic`.
- **`pages/ModelsList.vue`**: cards dos modelos do registry (título, problema associado, nº de
  parâmetros, badge do estado da licença Gurobi/HiGHS vindo de `/milp/status`); formulação
  matemática renderizada (KaTeX — única dependência nova do front, ou imagem estática como
  primeira versão).
- **`pages/ModelDetail.vue`** com wizard de varredura (padrão dos steps existentes):
  1. Seleção do problema (drafts compatíveis com `problem_key` via `/problems`);
  2. Grade de parâmetros — por parâmetro: valor fixo, lista explícita ou range/step, com contador
     "N combinações, ~T tempo estimado" em tempo real;
  3. Config do solver (backend, time limit, MIPGap) + opções de lote (MACs, duração da simulação,
     source repositories, data conversion — reuso dos componentes do launch existente);
  4. Confirmação → `POST /milp/sweeps`.
- **`pages/SweepDetail.vue`** (rota `/models/sweeps/:id`): barra de progresso (polling, padrão das
  páginas atuais), tabela de soluções (params → genótipo, nós instalados, obj, gap, tempo),
  visualização da topologia selecionada (reuso dos componentes de rede do Problem Editor), link
  para o experimento batch gerado e sua análise Pareto.
- **`api/milp.ts`** + tipos em `types/`.

---

## 3. Plano de ação faseado

### Fase 0 — Preparação (½ dia)
- [ ] Definir esquema de licença disponível (WLS acadêmica? trial?) e obter `gurobi.lic` de teste.
- [ ] Decidir escopo v1: sugerido **P2 primeiro** (modelo já validado no pipeline vizinho), P3 na sequência.
- [ ] Branch `feature/milp-models-integration`.

### Fase 1 — Núcleo MILP e solvers (2–3 dias)
- [ ] `milp-engine/lib/models/p2_mobile.py`: port de `solve_p2` consumindo `ProblemP2` (formato do
      `wsn-milp/wsn-mobile/mobile.py`, que já é o do SimLab); máscara na ordem de `candidates`.
- [ ] `milp-engine/lib/solver/`: `GurobiBackend` + `HighsBackend` + seleção por env/config.
- [ ] `milp-engine/lib/sweep.py`: grade cartesiana, dedup por genótipo, checkpoint.
- [ ] Testes unitários com instância mínima (grafo de 5–10 candidatos, solução conhecida) rodando
      com HiGHS e com trial gurobipy; teste de consistência da ordem do genótipo.
- **Critério de aceite**: dado um draft P2 do Problem Editor, a varredura local (sem Docker) produz
  o mesmo conjunto de genótipos com Gurobi e HiGHS na instância pequena.

### Fase 2 — Persistência e REST API (1–2 dias)
- [ ] `pylib/db`: repositório `milp_sweep_repo` (CRUD, update de progresso atômico, change stream helper).
- [ ] `rest-api/api/endpoints/milp.py` + mappers + domain DTOs; registrar no `router.py`.
- [ ] Testes de API (padrão de `tests/test_experiment.py`).
- **Critério de aceite**: CRUD completo de sweeps via Swagger, validação de grade (limite de
  combinações, parâmetros conhecidos do modelo).

### Fase 3 — Orquestração do milp-engine (2 dias)
- [ ] `milp-engine/engine.py`: change stream em `milp_sweeps` + varredura de pendentes no startup
      (mesmo padrão do `mo-engine/engine.py`), atualização incremental de progresso, cancelamento.
- [ ] `lib/handoff.py`: ao concluir, montar experimento `{strategy:"batch",
      parameters:{problem, chromosomes, simulation}, source_repository_options,
      data_conversion_config}` e inserir — mo-engine assume dali em diante.
- [ ] Tratamento de erro: licença indisponível → sweep em ERROR com `system_message` claro.
- **Critério de aceite**: sweep criado via API termina em SOLVED com `experiment_id` preenchido e
  o experimento batch roda até DONE no pipeline existente (pode ser em modo synthetic para CI).

### Fase 4 — Docker e licenciamento (1 dia)
- [ ] `Dockerfile.milp-engine` (python:3.12-slim + gurobipy + highspy + pylib, padrão do
      `Dockerfile.mo-engine`).
- [ ] Serviço `milpengine` no `docker-compose.yaml` (mongo-net, healthcheck, labels simlab.*).
- [ ] Suporte aos 4 esquemas de licença via env/volume; validação no startup publicada em
      `/milp/status`.
- [ ] `docs/markdown/MILP_MODULE.md`: como obter licença WLS acadêmica, montar `gurobi.lic`,
      variáveis, fallback HiGHS (espelhar o estilo do `SYNTHETIC_MODE.md`).
- **Critério de aceite**: `docker compose up` com e sem `gurobi.lic` — sem licença o engine sobe
  saudável reportando `solver: highs (fallback)`.

### Fase 5 — GUI (3–4 dias)
- [ ] Item `Models` no `Sidebar.vue` + rotas no router.
- [ ] `api/milp.ts` + tipos.
- [ ] `ModelsList.vue` (cards + status de licença + formulação).
- [ ] Wizard de varredura em `ModelDetail.vue` (reuso de Step3Simulation/Step5DataConversion do
      launch existente para as opções de lote).
- [ ] `SweepDetail.vue` (progresso com polling, tabela de soluções, preview de topologia, link para
      o experimento).
- [ ] Exibir badge "origem: MILP sweep" no `ExperimentDetail` quando o experimento vier de um sweep.
- **Critério de aceite**: fluxo completo pela GUI — escolher modelo → problema → grade → lançar →
  acompanhar → abrir Pareto do experimento gerado.

### Fase 6 — P3 target + validação E2E + docs (2 dias)
- [ ] `p3_target.py` (port do `wsn-target/target.py` para `ProblemP3`; params `w, k, g, B, M_max`).
- [ ] Rodada E2E real com instância P2 pequena (Cooja de verdade, poucos genótipos) comparando com
      resultados do `wsn-milp-nsga3-p2` como sanity check.
- [ ] Atualizar `README.md`/`README_pt.md` (topologia com o novo container) e `CHANGELOG.md`.

**Estimativa total: ~11–14 dias úteis** (P2 utilizável ao fim da Fase 5).

---

## 4. Riscos e mitigações

| Risco | Impacto | Mitigação |
|---|---|---|
| Licença Gurobi indisponível no container (acadêmica named-user não funciona em Docker) | bloqueia solver principal | WLS acadêmica (gratuita) como caminho recomendado; HiGHS como fallback de 1ª classe; trial p/ instâncias pequenas |
| Ordem do genótipo ≠ ordem de `candidates` | topologias erradas silenciosamente | convenção única (índice da lista `candidates`), teste dedicado round-trip MILP→chromosome→plot |
| Trajetórias do MILP ≠ trajetórias do simulador | métricas inconsistentes | usar o MESMO utilitário de amostragem (`trajectory_sampling`) nos dois lados |
| Explosão combinatória da grade (ex.: 8.250 combos × 300s) | sweeps de dias | contador/estimativa no wizard, limite configurável de combinações, checkpoint p/ retomada, cancelamento |
| Time-limit sem incumbente (`SolCount=0`) | buracos na varredura | registrar como `TIMEOUT_NO_SOLUTION` em `solutions` (padrão do runner.py vizinho) |
| gurobipy no CI | testes quebrando | testes parametrizados por backend; CI roda só HiGHS |

---

## 5. Resumo do reuso

| Origem | Artefato | Destino |
|---|---|---|
| `wsn-milp/wsn-mobile/mobile.py` | modelo P2 (já no formato ProblemP2) | `milp-engine/lib/models/p2_mobile.py` |
| `wsn-milp/wsn-target/target.py` | modelo P3 | `milp-engine/lib/models/p3_target.py` |
| `wsn-milp-nsga3-p2/milp/runner.py` | dedup por genótipo, checkpoint, params Gurobi | `milp-engine/lib/sweep.py` |
| `wsn-milp-nsga3-p2/Dockerfile.milp-p2` | padrão de licença em container | `Dockerfile.milp-engine` |
| `wsn-milp-nsga3-p2/batch_runner/` | — descartado — | equivalente já existe (master-node) |
| simlab `BatchStrategy` | execução de lote completa | reuso sem alteração |
| simlab Problem Editor / problems API | instâncias de entrada | reuso sem alteração |
| simlab Campaigns / ExperimentDetail / Pareto | análise de resultados | reuso sem alteração |
