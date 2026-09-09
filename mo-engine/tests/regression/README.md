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

## Baseline `phase-2` (30 execuções)

`drop` é a maior queda de HV em um único passo, sobre todas as sementes.

```
config                     HV off    HV surv   drop off  drop surv    GD surv
-----------------------------------------------------------------------------
nsga2-dtlz2-m3-n10       0.442195   0.495511   0.111201   0.031077   0.149428
nsga3-dtlz2-m3-n10       0.530671   0.569152   0.050310   0.010402   0.085026
nsga2-zdt1-m2-n10        0.497700   0.509439   0.032654   0.000000   0.265797
nsga3-zdt1-m2-n10        0.439273   0.447513   0.088306   0.000000   0.315556
nsga2-sch1-m2-n1        16.390396  16.518083   0.340590   0.031402   0.000835
nsga3-sch1-m2-n1        16.057579  16.110428   0.404301   0.229464   0.001048
```

### `pre-fix` (Fase 0), para comparação

```
nsga2-dtlz2-m3-n10       0.498542   0.533516   0.039925   0.017707   0.099148
nsga3-dtlz2-m3-n10       0.581202   0.620339   0.042602   0.015225   0.054280
nsga2-zdt1-m2-n10        0.456881   0.464750   0.095775   0.000000   0.253981
nsga3-zdt1-m2-n10        0.417986   0.432678   0.055778   0.000000   0.295333
nsga2-sch1-m2-n1        16.339305  16.449792   0.443291   0.023870   0.000827
nsga3-sch1-m2-n1        15.653757  15.741825   1.316102   0.569466   0.000964
```

O `pre-fix` reproduzia a auditoria exatamente (NSGA-III/DTLZ2: HV 0,581202 →
0,620339; queda 0,042602 → 0,015225), o que valida o harness como porte fiel do
script de auditoria.

### O que a Fase 2 mudou, por operador

Ablação com cada comportamento antigo restaurado isoladamente, NSGA-III,
HV dos sobreviventes, média de 5 sementes:

| SBX | niching | DTLZ2 M=3 | SCH1 M=2 |
| --- | --- | ---: | ---: |
| antigo | antigo | 0,620339 | 15,741825 |
| antigo | novo | 0,580534 | **16,148035** |
| novo | antigo | 0,575054 | 15,741825 |
| novo | novo | 0,569152 | 16,110428 |

- **Critério de saída da Fase 2.2 atingido**: `nsga3-sch1` era o caso em que os
  sobreviventes também caíam muito. A queda máxima por passo cai de 0,569 para
  0,229, e o ganho é atribuível ao niching (o SBX quase não move SCH1 com n=1).
- **DTLZ2 piora, e isso é real** — persiste em 60 e 150 gerações e sob mutação
  1/n. Ver a seção seguinte.

### Por que o DTLZ2 piora com operadores corretos

O SBX antigo usava a distância ao limite **inferior** para os dois filhos. Para
pais próximos do limite superior isso super-espalha o filho de cima, que era
então grampeado exatamente no limite:

| Pais amostrados | filhos grampeados no limite (antes) | (depois) |
| --- | ---: | ---: |
| uniformes em [0,1] | 0,543% | 0,000% |
| em [0,85, 1,0] | 0,597% | 0,000% |
| em [0,0, 0,15] | 0,000% | 0,000% |

O defeito era assimétrico: fabricava soluções exatamente sobre o limite
**superior** e nunca sobre o inferior. No DTLZ2 as variáveis de posição precisam
chegar a 0 ou 1 para produzir as soluções de canto da frente, que são as de maior
contribuição para o HV. O bug entregava metade desses cantos de graça. O SBX
corrigido é o da DEAP — filhos nunca saem da caixa — e chega aos cantos só
assintoticamente.

Isto não é motivo para reverter: o operador antigo é comprovadamente a
distribuição errada (paridade exata com a DEAP em `test_sbx_bounded.py`) e a
vantagem que dava é um artefato de um único benchmark. Mas os números de DTLZ2
publicados antes da Fase 2 dependiam desse artefato.

### Torneio com crowding (Fase 2.3)

Efeito dentro do ruído a 5 sementes; o que ele faz de forma clara é aumentar a
variância no DTLZ2 (M=3), coerente com a fraqueza conhecida da crowding distance
em três ou mais objetivos — a premissa que motivou o NSGA-III:

| desempate | média HV | desvio | por semente |
| --- | ---: | ---: | --- |
| aleatório | 0,522031 | 0,017574 | 0,508 0,531 0,549 0,509 0,514 |
| crowding | 0,495511 | 0,082897 | 0,484 **0,361** 0,518 0,537 0,579 |

Uma comparação conclusiva exige mais sementes e análise de significância, que o
plano posterga para depois da Fase 6.

## Versões

O baseline registra `numpy`, `pymoo`, `deap` e `moocore`. `--check` avisa quando
divergem: uma diferença numérica sob outras versões não é necessariamente
regressão do SimLab.
