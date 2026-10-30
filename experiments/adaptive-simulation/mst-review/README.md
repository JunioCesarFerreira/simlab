# Evidências locais da revisão MST — 20–21/10/2026

Complemento de [PROPOSTA_MST_APRENDIZADO.md](../../../docs/markdown/PROPOSTA_MST_APRENDIZADO.md).
Todos os resultados deste diretório usam `TopologyEvaluator`, uma função
determinística auxiliar. Nenhuma simulação Cooja ou banco externo foi acionado.

Comandos a partir da raiz do repositório (Python 3.12):

```powershell
.\.venv\Scripts\python.exe experiments/adaptive-simulation/run_encoding_comparison.py --population 20 --generations 8 --seed 42 --repeats 3 --json experiments/adaptive-simulation/mst-review/encoding.json
.\.venv\Scripts\python.exe experiments/adaptive-simulation/run_comparison.py --population 20 --generations 8 --seed 42 --kappa 1.96 1.0 0.5 --json experiments/adaptive-simulation/mst-review/adaptive.json
```

`encoding.json` conserva resultados individuais por semente, fronteiras e
comparações MST/SPT retrospectivas das mesmas máscaras. MST não é um quarto
braço de evolução: a comparação só mede soma de custos geométricos e
profundidade do esqueleto, sem novas avaliações. Esses custos não são energia.

`adaptive.json` compara estratégias sobre o mesmo cenário auxiliar. A função de
avaliação depende principalmente da quantidade de relays e da dispersão; não
modela contenção, perdas e formação real do RPL. Seus números não estimam a
economia de uma campanha Cooja.

## Comparação de representações

Médias de três sementes evolutivas (42, 43 e 44), população 20 e oito passos de
evolução além da geração inicial. Cada execução persistiu 180 indivíduos.

| Representação | Avaliações novas | Relays médios | Tamanho da frente | HV aproximado | Desconectados |
|---|---:|---:|---:|---:|---:|
| P2 clássico | 166,7 | 23,9 | 11,7 | 14625,4 | 0 |
| Reparo estrutural | 166,0 | 23,7 | 12,0 | 14843,6 | 0 |
| Árvore admissível | 129,7 | 17,2 | 15,3 | 18517,7 | 0 |

O HV usa Monte Carlo (200 mil pontos, RNG fixo) e o mesmo ponto de referência
calculado a partir das frentes deste piloto. Não é a referência pré-fixada
recomendada para uma campanha confirmatória. Três sementes não estabelecem
significância nem generalização. O avaliador também favorece certas faixas de
quantidade de relays; não atribuir o resultado exclusivamente à representação.

A comparação retrospectiva dos esqueletos sobre os indivíduos da variante em
árvore produziu:

| Esqueleto | Soma média dos custos geométricos normalizados | Média da profundidade máxima |
|---|---:|---:|
| SPT | 0,3678 | 3,1259 |
| MST | 0,2560 | 5,6796 |

Essas médias incluem ocorrências repetidas de indivíduos nas gerações. A MST
reduziu o custo das arestas, mas aumentou a profundidade; a implantação simulada
é exatamente a mesma nos dois cálculos.

## Triagem adaptativa

Piloto de uma semente evolutiva (42); baseline com reparo estrutural: 167
avaliações. Mesma quantidade de gerações, não mesmo orçamento de simulações.

| κ | Avaliações | Economia vs. execução baseline | Economia interna da política | HV / HV baseline | Auditorias | Promoções |
|---|---:|---:|---:|---:|---:|---:|
| 1,96 | 152 | 9,0% | 4,4% | 0,874 | 0 | 0 |
| 1,00 | 142 | 15,0% | 13,9% | 0,717 | 1 | 0 |
| 0,50 | 118 | 29,3% | 26,7% | 0,668 | 2 | 1 |

A economia vs. baseline é `1 − avaliações/167`. A métrica interna usa como
denominador os indivíduos novos elegíveis da **própria trajetória adaptativa**;
não é o mesmo denominador, pois seleção e acertos de cache mudam entre execuções.

A perda de HV neste piloto é substancial, mesmo com κ=1,96. Isso contraria uma
interpretação de que a promoção garantiria preservação da qualidade. Auditoria
zero não permite avaliar risco. O `false_skip_rate` salvo para κ=0,50 é 1/3,
mas seu denominador inclui amostras além das duas auditorias; não o interpretar
como uma estimativa baseada exclusivamente em sorteio aleatório. A nova proposta
explica como separar essas estatísticas.

O estudo não testa o aprendizado de operadores proposto, a triagem calibrada ou
redução sequencial de sementes. Esses mecanismos continuam como trabalho futuro.

## Ambiente e testes

Windows, Python 3.12.1, NumPy 1.26.3, pytest 9.0.2. Ambiente `.venv` local com
acesso aos pacotes preexistentes; não reproduz integralmente as versões de CI.
Foram adicionados zstandard 0.25.0 e moocore 0.3.1 para viabilizar a suíte.

A primeira execução ampla encontrou 26 falhas de reprodutibilidade nos adapters
pymoo usando a versão local antiga 0.6.1.3. O ambiente de teste foi ajustado para
pymoo 0.6.1.6 e deap 1.4.4, já fixados no `mo-engine/requirements.txt`, sem alterar
essas estratégias. O log da validação final está em [tests.txt](tests.txt).

**Resultado final: 566 testes passaram**, com um aviso de depreciação de
`python-dateutil`, em 125,93 s. Comando: a partir de `mo-engine`,
`..\.venv\Scripts\python.exe -m pytest -q --tb=short`.

Os harnesses acima usam o NSGA-III nativo; foram executados antes dessa troca
de pymoo/deap. Demais versões relevantes: pymongo 4.11, pandas 2.2.1,
matplotlib 3.8.2. Horários/durações são locais à execução e não constituem
benchmark de desempenho de Cooja. O frame de apresentação está em
[frame_mst_aprendizado.tex](../../../docs/latex/frame_mst_aprendizado.tex).
