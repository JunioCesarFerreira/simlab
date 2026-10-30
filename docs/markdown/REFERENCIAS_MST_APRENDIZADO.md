# Leituras para revisão: MST e aprendizado entre gerações

Consulta em 20–21/10/2026. Associado à
[análise da proposta](PROPOSTA_MST_APRENDIZADO.md). As relações com o SimLab são
interpretações de projeto; nenhum trabalho abaixo valida diretamente os nossos
cenários, firmware ou percentuais de economia.

## Representação e distinção entre árvores

**1. Lima, T. W.; Rothlauf, F.; Delbem, A. C. B. (2008).**
*The Node-Depth Encoding: Analysis and Application to the Bounded-Diameter
Minimum Spanning Tree Problem*. GECCO 2008, pp. 969–976.
[PDF do artigo](https://gpbib.pmacs.upenn.edu/gecco2008/docs/p969.pdf).

Consultados resumo e trechos do PDF. Investiga representação node-depth,
operadores e vieses de inicialização. É especialmente pertinente para distinguir
qualidade da codificação, viabilidade e efeito do viés de geração. Revisar as
condições dos operadores e a análise de estrelas/profundidade. Não transferir
automaticamente resultados de MST com diâmetro limitado para cobertura móvel e
multiobjetivos simulados.

**2. *Node-depth phylogenetic-based encoding, a spanning-tree representation for
evolutionary algorithms. Part I: Proposal and properties analysis* (2016).**
*Swarm and Evolutionary Computation* 31, pp. 1–10.
[Página editorial](https://www.sciencedirect.com/science/article/pii/S2210650216300232),
[DOI 10.1016/j.swevo.2016.05.001](https://doi.org/10.1016/j.swevo.2016.05.001).

Consultado o resumo editorial indexado e trechos; acesso direto posterior foi
bloqueado pelo provedor. Não houve leitura integral. Propõe representação
node-depth e rearranjos inspirados em árvores filogenéticas. Serve como leitura
para localidade e operações de poda/enxerto. Conferir autores/metadados no
editor ao importar a referência e conferir hipóteses de viabilidade antes de
atribuir aos nossos operadores as propriedades demonstradas no artigo.

**3. Chow, C. K.; Liu, C. N. (1968).** *Approximating Discrete Probability
Distributions with Dependence Trees*. *IEEE Transactions on Information Theory*
14(3), pp. 462–467. DOI 10.1109/TIT.1968.1054142.
[Registro e resumo dos autores na IBM Research](https://research.ibm.com/publications/approximating-discrete-probability-distributions-with-dependence-trees).

Consultado registro/resumo institucional. Aproxima uma distribuição discreta
por dependências em árvore; é a referência fundamental para separar árvore de
dependência estatística e árvore física. A árvore ponderada por informação mútua
é de peso máximo. Pode motivar uma futura distribuição sobre genes, mas não
garante que uma máscara amostrada conecte relays ao sink. Revisar estimação com
amostras pequenas, suavização e viés de selecionar apenas elites.

**4. Winter, T. et al. (2012).** *RPL: IPv6 Routing Protocol for Low-Power and
Lossy Networks*. RFC 6550.
[Texto oficial, §3.2](https://datatracker.ietf.org/doc/html/rfc6550#section-3.2),
[§14: Objective Function](https://datatracker.ietf.org/doc/html/rfc6550#section-14).

Consultadas definições e seções sobre seleção de pais/rank. Sustenta a distinção
entre um esqueleto geométrico do otimizador e o DODAG formado pelo protocolo.
O RFC possui atualizações listadas no registro oficial; esta análise usa apenas
a distinção conceitual, não faz auditoria de conformidade do firmware.

## Avaliações caras e modelos auxiliares

**5. Tong, H.; Huang, C.; Liu, J.; Yao, X. (2019).** *Voronoi-based Efficient
Surrogate-assisted Evolutionary Algorithm for Very Expensive Problems*.
[Preprint dos autores, arXiv:1901.05755](https://arxiv.org/abs/1901.05755).

Consultado resumo. Combina busca local/global e avaliação de modelo em cenário
de poucas avaliações caras. É precedente para tratar previsão e gerenciamento
de avaliações como partes do algoritmo. Não usa nossa representação em árvore
e não demonstra controle do falso descarte em redes RPL. Revisar protocolo
experimental e custo do treinamento; não importar seus ganhos numéricos.

**6. Jin, Y. (2011).** *Surrogate-assisted evolutionary computation: Recent
advances and future challenges*. *Swarm and Evolutionary Computation* 1(2),
pp. 61–70.
[DOI 10.1016/j.swevo.2011.05.001](https://doi.org/10.1016/j.swevo.2011.05.001).

Leitura complementar sugerida, **texto não acessado nesta revisão**. Metadados
conferidos na bibliografia editorial de
[Li, Zhan e Zhang (2022)](https://link.springer.com/article/10.1007/s11633-022-1317-4).
O DOI retornou bloqueio de acesso. Não foi usado como comprovação de resultados
específicos ou como se suas conclusões tivessem sido integralmente verificadas.

## Incerteza e economia de réplicas

**7. Tibshirani, R. J.; Barber, R. F.; Candès, E. J.; Ramdas, A. (2019).**
*Conformal Prediction Under Covariate Shift*.
[Preprint dos autores, arXiv:1904.06019](https://arxiv.org/abs/1904.06019).

Consultado resumo e registro de versões. Propõe predição conformal ponderada
quando a distribuição das entradas muda, sob condições sobre a razão de
densidades. Interessa porque a evolução altera as regiões amostradas. Não
justifica chamar `média ± 1,96 × desvio dos vizinhos` de intervalo de 95%.
Revisar hipóteses e disponibilidade de dados para estimar os pesos antes de
incorporar o método.

**8. Barber, R. F.; Candès, E. J.; Ramdas, A.; Tibshirani, R. J. (2023).**
*Conformal prediction beyond exchangeability*.
[Preprint dos autores, arXiv:2202.13415, versão de 2023](https://arxiv.org/abs/2202.13415).

Consultado resumo/registro. Discute perda de permutabilidade, ponderação de
observações e modelos que tratam dados recentes de modo diferente. Relevante
para validação temporal e deriva ao longo de gerações. Não estabelece por si
só uma garantia para nossa seleção adaptativa de amostras. Rever teoremas e
medidas de desvio da distribuição antes de alegar cobertura formal.

**9. Howard, S. R.; Ramdas, A.; McAuliffe, J.; Sekhon, J. (2021).**
*Time-uniform, nonparametric, nonasymptotic confidence sequences*.
*The Annals of Statistics* 49(2), pp. 1055–1080.
[Preprint dos autores](https://arxiv.org/abs/1810.08240),
[DOI 10.1214/20-AOS1991](https://doi.org/10.1214/20-AOS1991).

Consultado resumo e referência editorial no registro. Fornece ferramentas para
intervalos válidos ao longo do tempo sob hipóteses explícitas, relevantes para
decidir quando parar de adicionar sementes. Não equivale a fornecer limites
para um indivíduo nunca simulado. Antes de implementar, escolher uma construção
compatível com métricas limitadas/caudas, dependência e comparações múltiplas.

**10. Kone, C.; Kaufmann, E.; Richert, L. (2024).** *Bandit Pareto Set
Identification: the Fixed Budget Setting*. AISTATS, PMLR 238, pp. 2548–2556.
[Artigo e PDF no PMLR](https://proceedings.mlr.press/v238/kone24a.html).

Consultados resumo e metadados oficiais. Estuda identificação de Pareto com
observações ruidosas sob orçamento fixo; referência direta para alocar réplicas
entre alternativas. O conjunto finito de braços da formulação não é automaticamente
equivalente a um espaço de cromossomos que cresce por mutação. Revisar hipóteses
de ruído, critério de sucesso e regra de alocação antes de adaptar.

## Ordem sugerida para a revisão

1. Ler o contraexemplo MST/SPT e a auditoria do código na proposta.
2. Conferir Lima et al. para representação e vieses; RFC para o significado do
   que o simulador realmente executa.
3. Revisar Chow–Liu somente se a ideia também envolver modelar dependência entre
   genes; isso é uma extensão diferente de usar uma árvore física.
4. Comparar aprendizado de operadores com uma triagem simples bem auditada.
5. Estudar calibração e alocação sequencial antes de prometer descarte seguro.

Esta é uma seleção temática, não uma revisão sistemática. Não foram copiados
artigos integrais para o repositório; os links preservam a fonte para conferência.
