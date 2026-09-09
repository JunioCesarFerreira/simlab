# Plano de ação — correções de NSGA / HV / GD

Origem: auditoria em `experiments/nsga-metrics-audit/README.md` (SimLab `d4c639f`,
09/09/2026). Os nove achados foram reconfirmados no código antes deste plano.
Nenhuma correção foi aplicada ainda.

Princípio que organiza as fases: **a oscilação relatada tem várias causas
candidatas somadas**. Corrigir tudo de uma vez torna impossível atribuir a
melhora. Cada fase abaixo termina com uma medição comparável à anterior.

---

## Fase 0 — Rede de segurança (pré-requisito de todas as demais) · **concluída**

Sem baseline congelado, cada correção muda os números e não se separa efeito de
regressão.

Entregue em [`mo-engine/tests/regression/`](../../mo-engine/tests/regression/README.md):

- `kernel.py` — harness síncrono sobre os métodos reais de reprodução e seleção
  ambiental, sem MongoDB e **sem depender do notebook externo** do NSGA-Studies.
  Registra HV/GD/IGD/IGD+ por geração para os três conjuntos (descendentes,
  sobreviventes, arquivo), mais `radial_error` analítico no DTLZ2.
- `baseline_pre_fix.json` — 30 execuções (NSGA-II e NSGA-III nativos × DTLZ2
  M=3/n=10, ZDT1 M=2/n=10, SCH1 M=2/n=1 × sementes 1/2/3/5/7), marcado
  `stage: "pre-fix"`. É referência de comparação, **não** meta a preservar.
- `baseline.py` — `--write` / `--check` / `--summary`.
- `test_baseline.py` — reprodutibilidade, sensibilidade à semente, comparação
  com o baseline, e o achado 1 como invariante permanente.

Critério de saída atingido: `--check` reproduz o baseline em execuções repetidas,
e DTLZ2/NSGA-III reproduz os números da auditoria (HV final 0,581202 nos
descendentes contra 0,620339 nos sobreviventes; maior queda por passo 0,042602
contra 0,015225).

Observação nova, vinda do baseline: `nsga3-sch1-m2-n1` é o único caso em que os
**sobreviventes** também caem muito (0,569 contra 0,024 do NSGA-II no mesmo
problema). Com M=2 e `divisions=10` são 11 direções para 50 indivíduos — o
cenário em que os defeitos de niching do achado 2 mais pesam. Adotar como
critério de saída da Fase 2.2.

---

## Fase 1 — Definir e persistir o conjunto medido (achados 1 e 8) · **concluída**

É a causa mais provável da oscilação visível nos gráficos: hoje a curva mede
`ND(Q_t)`, a frente dos **descendentes**, não a população preservada.

### 1.1 Persistir sobreviventes

- Adicionar `survivors: list[str]` ao `Generation`
  ([pylib/db/models/generation.py](../../pylib/db/models/generation.py)) — lista
  de `individual_id` (hash do cromossomo).
  Preferir campo na geração a uma flag em `Individual`: é uma escrita por
  geração, não mexe no índice único `(generation_id, individual_id)`, e permite
  referenciar sobreviventes vindos de gerações mais antigas.
- Novo método `GenerationRepository.set_survivors(gen_oid, hashes)`.
- Em `_evolution` ([nsga3.py:836](../../mo-engine/lib/strategy/nsga3.py#L836) e o
  equivalente em `nsga2.py`), logo após `self._parents = self._select_next_parents(...)`,
  gravar os hashes em `self._generation_id` — que nesse instante ainda aponta
  para a geração recém-concluída, antes do próximo `_generation_enqueue`
  incrementar `_gen_index`.
- Para a geração 0, gravar a própria população inicial como sobreviventes.

### 1.2 Corrigir a finalização

Hoje o teste de parada roda **antes** da última seleção ambiental e
`_final_pareto_front` devolve `ND(pais ∪ últimos_filhos)` — até 2N candidatos
contra uma população selecionada de N.

- Reordenar `_evolution`: seleção ambiental sobre `R_t` → grava sobreviventes →
  **então** testa parada → finaliza com `ND(P_final)`.
- Ajustar `_final_pareto_front` para operar sobre `self._parents` pós-seleção.
- Alinhar o orçamento de avaliações ao protocolo do notebook e documentar
  explicitamente a convenção de "geração 0 = população inicial".

### 1.3 Expor as séries no endpoint

- `GET /{id}/hv-gd`: novo parâmetro `population=survivors|offspring|archive`,
  default `survivors`.
- Resolver `survivors` exige um mapa `individual_id → objectives` no escopo do
  **experimento** (um sobrevivente pode ser de geração anterior); adicionar
  `IndividualRepository.find_by_experiment` ou uma projeção agregada.
- Fallback para `offspring` em experimentos antigos sem o campo `survivors`,
  sinalizado na resposta (`population_source`).
- GUI: rotular a série explicitamente. "Cumulative" hoje só altera o HV — GD/IGD
  continuam sendo dos descendentes; ou estender ao arquivo, ou desabilitar a
  opção para GD/IGD.

Critério de saída atingido. O que foi entregue:

- `Generation.survivors` + `GenerationRepository.set_survivors`, escritos por
  `_persist_survivors` em ambas as estratégias — inclusive na geração 0, onde
  P_0 sobrevive trivialmente. Escrita best-effort: sobreviventes são metadado de
  análise e uma falha não pode abortar um experimento em andamento.
- `_evolution` reordenado nas duas estratégias: seleção → grava sobreviventes →
  testa parada → finaliza com `ND(P_final)`. Orçamento de avaliações inalterado.
- `_final_pareto_front` passa a usar só `self._parents`.
- Endpoint com `population=survivors|offspring|archive` (default `survivors`),
  resolvendo hashes no escopo do experimento e devolvendo `population_source`
  com o fallback para `offspring` em runs antigos. `population=archive` estende
  o acumulado a GD/IGD/IGD+, não só ao HV.
- GUI: seletor **Measured set** substitui o toggle Per generation/Cumulative em
  [HvGdChart.vue](../../gui/simlab/src/components/charts/HvGdChart.vue) e em
  [ExperimentsComparison.vue](../../gui/simlab/src/pages/ExperimentsComparison.vue),
  governando os três gráficos; a legenda diz qual conjunto foi medido e avisa
  quando houve fallback.
- Testes: [test_survivor_persistence.py](../../mo-engine/tests/test_survivor_persistence.py)
  roda o `_evolution` real de forma síncrona e trava os achados 1 e 8;
  `TestHvGdMeasuredPopulation` em `rest-api/tests/test_experiment.py` cobre o
  endpoint, incluindo a resolução de sobreviventes entre gerações e o fallback.
- [test_final_pareto_front.py](../../mo-engine/tests/test_final_pareto_front.py)
  atualizado: o contrato agora é ND(P_final), e um filho não selecionado não
  pode vazar para a frente reportada.

O baseline da Fase 0 permanece idêntico — o kernel já media os três conjuntos e
já aplicava a seleção ao último lote, então a Fase 1 alinhou a produção ao que o
baseline media, sem mover os números.

Pendências herdadas, tratadas nas fases seguintes: a retomada continua carregando
os filhos como pais (Fase 3.2, agora trivial com `survivors` persistido), e o
endpoint continua projetando objetivos por posição (Fase 5).

---

## Fase 2 — Operadores genéticos (achados 3 e 2) · alta

### 2.1 SBX limitado — segundo filho (achado 3)

[simulated_binary_crossover.py:34-47](../../mo-engine/lib/genetic_operators/crossover/simulated_binary_crossover.py#L34-L47)
calcula um único `beta/alpha/betaq` a partir da distância ao limite **inferior**
e o reutiliza nos dois filhos.

- Calcular `betaq` separadamente: filho 1 com `beta = 1 + 2(y1-xl)/(y2-y1)`,
  filho 2 com `beta = 1 + 2(xu-y2)/(y2-y1)`, cada um com seu `alpha`, ambos
  consumindo o **mesmo** sorteio `rand` (como `cxSimulatedBinaryBounded` da DEAP).
- O clipping posterior fica como guarda numérica, não como substituto do cálculo.
- Teste de regressão do caso da auditoria: pais 0,8 e 0,99, limites [0,1],
  eta=20, `rand=0,99` → filho superior `0,999287945` (hoje `1,0`).
- Teste de paridade estatística contra DEAP com RNG determinística.

Impacto: todos os adaptadores que usam este operador, inclusive `*_deap` e
`*_pymoo` (que só substituem a sobrevivência).

### 2.2 Niching NSGA-III nativo (achado 2)

[niching_selection.py:123-175](../../mo-engine/lib/nsga/niching_selection.py#L123-L175)
tem quatro desvios do canônico. Reescrever a função:

- Normalização por ponto ideal + extremos/interceptos do hiperplano (com
  fallbacks para casos degenerados), considerando os pontos das frentes **já
  aceitas**, não só a frente a truncar.
- Associação por **distância perpendicular** à direção de referência —
  `associate_to_niches`, já presente no arquivo e hoje não usada nessa seleção.
- `niche_count` inicializado com a ocupação dos indivíduos já aceitos: a
  assinatura passa a receber `selected_idx` (ou os objetivos aceitos), e o
  chamador em [nsga3.py:875-885](../../mo-engine/lib/strategy/nsga3.py#L875-L885)
  passa a informá-los.
- Remover da busca por ocupação mínima os nichos sem candidatos, eliminando o
  fallback aleatório sobre todos os restantes (que hoje nem atualiza a ocupação).
- Em nicho já ocupado, escolher aleatoriamente entre os candidatos, não o mais
  próximo.
- Teste de paridade contra `deap.tools.selNSGA3` em matrizes de objetivos fixas.

### 2.3 Torneio do NSGA-II

O torneio de reprodução desempata ranks aleatoriamente. Aplicar o desempate
canônico por crowding distance em
[tournament_selection.py](../../mo-engine/lib/genetic_operators/selection/tournament_selection.py),
passando as distâncias já calculadas na seleção ambiental.

Critério de saída: paridade com DEAP dentro da tolerância definida nos testes, e
comparação Fase 1 vs. Fase 2 no baseline multi-semente.

---

## Fase 3 — Determinismo e retomada (achado 7) · alta quando esses caminhos são usados

### 3.1 Ligar os RNGs de biblioteca à semente

- [nsga3_deap.py](../../mo-engine/lib/strategy/nsga3_deap.py) (e o par NSGA-II):
  semear `random` e `numpy.random` a partir de `algorithm.random_seed` antes de
  `selNSGA3`.
- [nsga3_pymoo.py:69-71](../../mo-engine/lib/strategy/nsga3_pymoo.py#L69-L71):
  passar `random_state` explícito para `survival.do`. Na versão instalada
  (pymoo 0.6.1.6), omitir o argumento cria um `default_rng()` sem semente —
  `np.random.seed` **não** controla esse gerador.
- Manter um gerador NumPy por estratégia, derivado da semente, em vez de semear
  o estado global a cada chamada.
- Teste: cinco instâncias da mesma estratégia com a mesma matriz de objetivos e
  seed 42 devem selecionar o mesmo conjunto (hoje: cinco conjuntos distintos nos
  adaptadores).

### 3.2 Retomada fiel

`_restore_population_state` ([nsga2.py:354](../../mo-engine/lib/strategy/nsga2.py#L354),
[nsga3.py:360](../../mo-engine/lib/strategy/nsga3.py#L360)) carrega a geração
anterior como pais — esses documentos são os **filhos** anteriores, não os
sobreviventes. Na reprodução da auditoria, perdeu 4 de 10 pais.

- Depende da Fase 1: carregar `survivors` da geração anterior como `_parents`.
- Persistir o estado dos RNGs no checkpoint: `random.Random.getstate()`
  serializado, estado do gerador NumPy, e o estado de normalização da biblioteca
  quando aplicável (pymoo mantém ideal/nadir entre gerações).
- Teste: N gerações contínuas vs. execução interrompida e retomada no meio →
  mesma população final.

---

## Fase 4 — Referências e definição das métricas (achados 5 e 6)

### 4.1 Fixar uma definição de GD e documentá-la

O SimLab usa `sum(d_i)/N` normalizado ([pylib/moo_metrics.py:147](../../pylib/moo_metrics.py#L147));
o notebook SCH1 usa `sqrt(sum(d_i²))/N` cru — que também não é RMS.

- Manter a definição do SimLab (coincide com pymoo/moocore) e **corrigir o
  notebook**, não o contrário.
- Registrar a fórmula exata, a normalização e o tamanho da referência em
  [RUNTIME_METRICS.md](RUNTIME_METRICS.md), e devolvê-los na resposta da API
  (`gd_formula` além de `reference_kind`/`reference_size`/`normalized` já
  existentes) para que os gráficos sejam autoexplicativos.

### 4.2 Referência analítica densa (erro de discretização)

[benchmarks.true_front](../../pylib/benchmarks.py#L161) usa 500 pontos aleatórios
para DTLZ2. Pontos exatamente sobre a hiperesfera acusam GD de 0,0016 (M=2),
0,0296 (M=3) e 0,1938 (M=6) — discretização, não falta de convergência.

- Trocar a amostra aleatória por grade determinística (direções Das–Dennis
  projetadas na esfera), com densidade função de M e default bem maior.
- Normalizar pelos extremos **teóricos** do benchmark (ideal e nadir conhecidos),
  não pelos extremos da amostra.
- Critério de saída: GD de pontos exatamente sobre a frente < 1e-3 para M=2, 3 e 6.
- Adicional (diagnóstico, não substituto): expor para DTLZ2 o erro radial
  `mean(| ||f||₂ − 1 |)`, que é analítico. Ele não mede cobertura — IGD/IGD+
  continuam exigindo referência densa.

### 4.3 Unificar as referências entre API, CLI e plots

[compute_hv_gd.py:120-122](../../pareto-analysis/compute_hv_gd.py#L120-L122) usa
`pior_observado + 5% + 1` mesmo quando recebe `--true-front-bench`, enquanto a
API e `plot_pareto_results.py` usam `1,1 × nadir_teórico`. O comentário de
paridade com a API não corresponde ao comportamento.

- Extrair a construção de referência (frente de referência + ponto de HV) para
  uma função única em `pylib` e consumi-la nos três caminhos.
- No modo real, deixar explícito na resposta que HV vem do pior observado da
  própria execução — números de runs diferentes não são comparáveis entre si.

---

## Fase 5 — Robustez do endpoint (achado 9)

Em [rest-api/api/endpoints/experiment.py](../../rest-api/api/endpoints/experiment.py):

- **Objetivos por posição** (linha ~573): `objs = [float(raw[i]) for i in range(n_obj)]`
  ignora os nomes, enquanto a referência é montada por nome. Reordenar `[f1,f2]`
  para `[f2,f1]` mudou GD de 0 para 11,31. Mapear nome → índice usando
  `parameters.algorithm.objectives[].metric_name`, retornar 422 para nome
  desconhecido, e recusar projeções de subconjunto contra uma frente analítica
  de outro M.
- **`all_min` vazio** (linha ~615): com todos os indivíduos penalizados, `max()`
  lança `ValueError`. Retornar `_empty_hv_gd()`.
- **Sintético sem `pareto_front`** (linha ~559): hoje retorna séries vazias antes
  de tentar a referência analítica, que está disponível. Mover o *early return*
  para depois da resolução da referência.
- **Testes HTTP**: `rest-api/tests/test_experiment.py` travou na inicialização do
  `TestClient`/AnyIO no ambiente da auditoria. Destravar (versões de
  anyio/httpx/starlette) — sem isso essas correções ficam sem cobertura de
  integração.

---

## Fase 6 — Alinhamento de protocolo com os notebooks (achado 4)

Deliberadamente por último: a ablação da auditoria mostra que mexer só na
mutação **não** melhora tudo (NSGA-III piorou: HV 0,620 → 0,605). Isto é
alinhamento de protocolo para comparação, não correção de defeito — e os efeitos
das fases 1–3 precisam ser medidos antes.

Em [SyntheticLaunchWizard.vue:143-153](../../gui/simlab/src/components/synthetic-editor/launch/SyntheticLaunchWizard.vue#L143-L153):

- `probMt=0.1` × `perGeneProb=0.05` dá 0,5% de chance efetiva por variável. O
  adapter tem fallback `1/n`, mas a interface sempre envia 0,05. Propor
  `probMt=1.0` e `perGeneProb` derivado de `1/nVars` (ou omitido, ativando o
  fallback), e explicitar no wizard a semântica composta das duas probabilidades.
- `divisions=10` fixo dá 11 direções em M=2 (notebook usa 49 → 50). Derivar o
  default de M.
- Documentar o domínio SCH1 ([-5,5] mapeado de [0,1] vs. [-10,10] do notebook), a
  referência de HV por benchmark e o orçamento de avaliações.

Só depois desta fase: comparação multi-semente entre SimLab e notebooks com os
mesmos problemas, populações iniciais e orçamento — com análise de significância,
que a auditoria explicitamente não fez.

---

## Ordem e dependências

```
Fase 0 ──> Fase 1 ──> Fase 3.2 (retomada depende de survivors)
       ├─> Fase 2 (independente)
       ├─> Fase 3.1 (independente)
       ├─> Fase 4 (independente)
       └─> Fase 5 (independente)
                 └──> Fase 6 (só após 1–3 medidas)
```

Fases 2, 3.1, 4 e 5 podem seguir em paralelo depois da Fase 0. A Fase 1 é a que
mais muda as curvas e deve ser medida isolada.

## Riscos

- **Fases 1, 2 e 3 mudam os resultados numéricos de experimentos já rodados.**
  Experimentos antigos no MongoDB não têm `survivors`; o endpoint precisa
  degradar para `offspring` e dizer isso na resposta.
- A reescrita do niching pode alterar a distribuição da frente de forma
  perceptível na GUI antes de qualquer ganho de HV — esperado, não regressão.
- A Fase 4.2 muda os valores absolutos de GD/IGD de todos os benchmarks
  sintéticos; qualquer número publicado antes precisa ser recalculado.
- A auditoria não consultou experimentos no MongoDB nem os containers de
  produção. Convém validar a Fase 1 contra um experimento real armazenado antes
  de considerar a hipótese confirmada em produção.
