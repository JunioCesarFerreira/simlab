# util NSGA

## NSGA

## GA

### Exemple 1
```py
# Suppose each individual is list[float]
pop: List[List[float]] = init_population(...)
fit: List[float] = evaluate(pop)  # higher is better; if minimizing, set maximize=False

ga = GeneticAlgorithm[List[float]](
    GAConfig(n_offspring=len(pop), crossover_prob=0.9, mutation_prob=0.2, maximize=True, seed=42),
    selection=lambda P, S, k, R: tournament_selection(P, S, k, R, tournament_size=3),
    crossover=lambda a, b, R: one_point_crossover(a, b, R),
    mutation=lambda x, R: gaussian_mutation(x, R, sigma=0.05, per_gene_prob=0.1, bounds=None),
)

# External loop (elitism/combination/stop criteria are up to you)
for gen in range(G):
    fit = evaluate(pop)
    offspring = ga.step(pop, fit)
    # e.g., generational replacement:
    pop = offspring
```

### Example 2
```py
# Exemplo: indivíduos são vetores reais (list[float]) com bounds conhecidos
bounds = [(-5.0, 5.0)] * 10  # 10 genes

# Crossover SBX com η_c = 15 e respeitando bounds
sbx = make_sbx_crossover(eta=15.0, bounds=bounds)

# Polynomial Mutation com η_m = 20 e prob. por gene = 1/n (default)
pm = make_polynomial_mutation(eta=20.0, per_gene_prob=None, bounds=bounds)

ga = GeneticAlgorithm[List[float]](
    GAConfig(n_offspring=100, crossover_prob=0.9, mutation_prob=0.2, maximize=True, seed=7),
    selection=lambda P, S, k, R: tournament_selection(P, S, k, R, tournament_size=3),
    crossover=lambda a, b, R: sbx(a, b, R),
    mutation=lambda x, R: pm(x, R),
)
```