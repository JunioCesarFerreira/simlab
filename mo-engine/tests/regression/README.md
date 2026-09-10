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
ruído zero, **parâmetros padrão do assistente de lançamento**. Os defaults
acompanham o que a interface realmente envia, para o baseline continuar medindo
a plataforma como ela é entregue; a Fase 6 os moveu para `prob_mt=1.0`,
`per_gene_prob=1/n` e `divisions` derivado de M e da população.

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

**GD é a distância exata** à frente analítica (`benchmarks.front_distance`), não
a distância média ao vizinho mais próximo numa referência amostrada. A Fase 0
registrava isso numa coluna separada, `radial_error`; a Fase 4 promoveu
exatamente essa quantidade a GD, então a coluna extra virou duplicata e saiu.
IGD e IGD+ continuam sobre a referência amostrada — é o que os faz medir
cobertura — e por isso carregam o piso de discretização dela.

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

## Baseline `phase-6` (30 execuções)

`drop` é a maior queda de HV em um único passo, sobre todas as sementes.

```
config                     HV off    HV surv   drop off  drop surv    GD surv
-----------------------------------------------------------------------------
nsga2-dtlz2-m3-n10       0.415902   0.475401   0.095484   0.025915   0.170337
nsga3-dtlz2-m3-n10       0.546410   0.568859   0.048615   0.007782   0.078157
nsga2-zdt1-m2-n10        0.560509   0.570737   0.093423   0.000000   0.239160
nsga3-zdt1-m2-n10        0.528164   0.544913   0.016820   0.000000   0.244904
nsga2-sch1-m2-n1        16.395027  16.550427   1.252112   0.047228   0.000109
nsga3-sch1-m2-n1        16.400878  16.521825   1.252112   0.040700   0.000063
```

### O que a Fase 6 mudou (parâmetros padrão)

| config | HV surv Fase 4 | HV surv Fase 6 | Δ |
| --- | ---: | ---: | ---: |
| nsga2-dtlz2-m3-n10 | 0,495511 | 0,475401 | −0,020 |
| nsga3-dtlz2-m3-n10 | 0,569152 | 0,568859 | −0,000 |
| nsga2-zdt1-m2-n10 | 0,509439 | **0,570737** | +0,061 |
| nsga3-zdt1-m2-n10 | 0,447513 | **0,544913** | +0,097 |
| nsga2-sch1-m2-n1 | 16,518083 | 16,550427 | +0,032 |
| **nsga3-sch1-m2-n1** | 16,110428 | **16,521825** | **+0,411** |

O salto de `nsga3-sch1` é o `divisions`: em M=2 o valor fixo 10 dava 11 direções
de referência para 50 indivíduos. Derivado, dá 49 divisões (50 direções, uma por
slot). Era exatamente o caso que a Fase 0 apontou como pior queda de HV entre
sobreviventes (0,569) e que a Fase 2 reduziu a 0,229 — agora está em 0,041.

O −0,020 do `nsga2-dtlz2` está dentro do desvio entre sementes (0,039 medido com
15 sementes). Numa comparação com 15 sementes e 40 gerações, a mutação de livro
é neutra no DTLZ2 e claramente melhor no ZDT1:

| problema | algoritmo | 0,1 × 0,05 | 1,0 × 1/n |
| --- | --- | ---: | ---: |
| DTLZ2 M=3 | NSGA-II | 0,545873 ± 0,039 | 0,550428 ± 0,034 |
| DTLZ2 M=3 | NSGA-III | 0,633169 ± 0,023 | 0,635412 ± 0,021 |
| ZDT1 M=2 | NSGA-II | 0,619231 ± 0,089 | **0,801769 ± 0,026** |
| ZDT1 M=2 | NSGA-III | 0,632825 ± 0,082 | **0,805805 ± 0,017** |

Além da média, o desvio cai por um fator de três no ZDT1 — a taxa antiga
deixava a busca à mercê da população inicial.

A Fase 4 não mexeu no HV; só o GD mudou, ao trocar a referência amostrada pela
distância exata. Quanto a coluna `GD surv` caiu é exatamente o erro de
discretização que estava sendo lido como falta de convergência:

| config | GD antes (Fase 2) | GD depois (Fase 4) | erro removido |
| --- | ---: | ---: | ---: |
| nsga2-dtlz2-m3-n10 | 0,149428 | 0,141762 | 0,0077 |
| nsga3-dtlz2-m3-n10 | 0,085026 | 0,075072 | 0,0100 |
| nsga2-zdt1-m2-n10 | 0,265797 | 0,265773 | 0,00002 |
| nsga3-zdt1-m2-n10 | 0,315556 | 0,315554 | 0,00000 |
| **nsga2-sch1-m2-n1** | 0,000835 | **0,000034** | 25× |
| **nsga3-sch1-m2-n1** | 0,001048 | **0,000270** | 4× |

O ZDT1 mal se move porque aquelas execuções param longe da frente: o erro de
discretização é irrelevante ao lado da distância real. O SCH1 é o oposto — a
população chega tão perto que quase todo o GD relatado era da referência.

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
