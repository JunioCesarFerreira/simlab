# Baseline de regressão NSGA / HV / GD

Fase 0 de [`docs/markdown/NSGA_METRICS_FIX_PLAN.md`](../../../docs/markdown/NSGA_METRICS_FIX_PLAN.md).

As fases 1–4 do plano mudam os números das métricas **de propósito**. Sem um
baseline congelado não há como separar a correção pretendida de uma regressão
acidental. Este diretório existe para isso.

## O que é medido

[`kernel.py`](kernel.py) executa os métodos reais de reprodução e seleção
ambiental do SimLab sobre um benchmark analítico, de forma síncrona: sem
MongoDB, sem change streams, sem workers. Uma execução é função pura de
`(config, seed)`.

Grade congelada: NSGA-II e NSGA-III nativos × DTLZ2 (M=3, n=10), ZDT1 (M=2,
n=10) e SCH1 (M=2, n=1) × sementes 1, 2, 3, 5, 7 — população 50, 20 gerações,
ruído zero, parâmetros padrão da GUI (`prob_mt=0.1`, `per_gene_prob=0.05`).

Três conjuntos por geração, porque a auditoria mostrou que a plataforma plota um
e os notebooks plotam outro:

| Conjunto | Definição | Quem usa hoje |
| --- | --- | --- |
| `offspring` | `ND(Q_t)`, os filhos recém-gerados | endpoint `/hv-gd` |
| `survivors` | `ND(P_t)`, a população após a seleção | notebooks NSGA-Studies |
| `archive` | `ND` de tudo já avaliado | opção "Cumulative" (só HV) |

HV, GD, IGD e IGD+ são calculados **exatamente como o endpoint** os calcula
(HV com `1,1 × nadir` e só pontos que dominam estritamente a referência; GD/IGD
normalizados pela faixa ideal-nadir da frente de referência), para que um número
do baseline seja comparável com a série de um experimento real.

Para DTLZ2 há também `radial_error` = `mean(|‖f‖₂ − 1|)`. É analítico e imune ao
erro de discretização da frente de referência (achado 5) — é o único número que
continua comparável depois que a Fase 4.2 mudar `benchmarks.true_front`.

Diferença deliberada em relação à produção: o kernel aplica seleção ambiental
também ao último lote de filhos. A produção pula essa etapa (achado 8); a Fase
1.2 vai corrigi-la, e o kernel já codifica o comportamento pretendido para o
baseline não precisar ser recongelado por causa disso.

## Uso

```bash
cd mo-engine
../.venv/bin/python -m tests.regression.baseline --summary   # tabela do baseline
../.venv/bin/python -m tests.regression.baseline --check     # comparar (exit 1 se divergir)
../.venv/bin/python -m tests.regression.baseline --write     # recongelar
../.venv/bin/python -m pytest tests/regression -q
```

Fluxo ao aplicar uma fase do plano: rodar `--check`, **ler o diff e confirmar
que é a mudança pretendida**, então `--write` e subir `BASELINE_STAGE` em
[`baseline.py`](baseline.py). Recongelar sem ler o diff anula o propósito do
diretório.

## Baseline `pre-fix` (30 execuções)

`drop` é a maior queda de HV em um único passo, sobre todas as sementes.

```
config                     HV off    HV surv   drop off  drop surv    GD surv
-----------------------------------------------------------------------------
nsga2-dtlz2-m3-n10       0.498542   0.533516   0.039925   0.017707   0.099148
nsga3-dtlz2-m3-n10       0.581202   0.620339   0.042602   0.015225   0.054280
nsga2-zdt1-m2-n10        0.456881   0.464750   0.095775   0.000000   0.253981
nsga3-zdt1-m2-n10        0.417986   0.432678   0.055778   0.000000   0.295333
nsga2-sch1-m2-n1        16.339305  16.449792   0.443291   0.023870   0.000827
nsga3-sch1-m2-n1        15.653757  15.741825   1.316102   0.569466   0.000964
```

Duas leituras imediatas:

- **DTLZ2/NSGA-III reproduz a auditoria exatamente** (HV 0,581202 → 0,620339,
  queda 0,042602 → 0,015225). O harness é um porte fiel do script de auditoria.
- **`nsga3-sch1` é o único caso em que os sobreviventes também caem muito**
  (0,569 contra 0,024 do NSGA-II no mesmo problema). Com M=2 e `divisions=10`
  são 11 direções de referência para 50 indivíduos: é o cenário em que os
  defeitos de niching do achado 2 mais pesam. Candidato natural a critério de
  saída da Fase 2.2.

## Versões

O baseline registra `numpy`, `pymoo`, `deap` e `moocore`. `--check` avisa quando
divergem: uma diferença numérica sob outras versões não é necessariamente
regressão do SimLab.
