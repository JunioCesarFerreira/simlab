/**
 * How many generations and simulations an experiment plans.
 *
 * `number_of_generations` counts evolution steps: every strategy that reads it
 * evaluates generation 0 as well, so a run has `number_of_generations + 1`
 * generations (mo-engine/README.md, "Configuration").
 */
export interface PlannedAlgorithm {
  population_size?: number;
  number_of_generations?: number;
}

/** Generations the run will have, or what exists so far when the strategy plans none (batch, benchmark). */
export function plannedGenerations(algorithm: PlannedAlgorithm | undefined, existing: number): number {
  const steps = algorithm?.number_of_generations;
  return typeof steps === "number" && steps >= 0 ? steps + 1 : existing;
}

/** Upper bound of the simulations: genomes found in the cache are not run again. */
export function plannedSimulations(algorithm: PlannedAlgorithm | undefined, seeds: number): number | null {
  const pop = algorithm?.population_size ?? 0;
  const steps = algorithm?.number_of_generations;
  if (!pop || typeof steps !== "number" || steps < 0 || !seeds) return null;
  return pop * (steps + 1) * seeds;
}
