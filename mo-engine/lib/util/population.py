from bson import ObjectId
# Problem Adapter
from lib.problem.adapter import Chromosome

class PopulationSnapshot:
    def __init__(self):
        self._genomes: list[Chromosome] = []   # wait P_t (genomes)
        self._simulations: list[ObjectId] = []  # wait P_t (simulations)
        self._objectives: list[list[float]] = []  # wait F(P_t)
        
    def set(self, 
            genomes: list[Chromosome], 
            sim_oid_to_idx: dict[str, int], 
            objectives: list[list[float]]
        ) -> None:
        if len(genomes) != len(objectives):
            raise ValueError(
                "genomes and objectives must have the same length"
            )

        simulations: list[ObjectId | None] = [None] * len(genomes)

        for sim_oid, idx in sim_oid_to_idx.items():
            if idx < 0 or idx >= len(genomes):
                raise IndexError(f"Invalid index {idx} for simulation {sim_oid}")
            simulations[idx] = ObjectId(sim_oid)

        if any(s is None for s in simulations):
            raise ValueError("Some simulations are missing in the mapping")

        self._genomes = list(genomes)
        self._simulations = simulations
        self._objectives = [row[:] for row in objectives]
                 
    def __len__(self) -> int:
        return len(self._genomes)
    
    def get_genomes(self) -> list[Chromosome]:
        return list(self._genomes)
    
    def get_objectives(self) -> list[list[float]]:
        return [row[:] for row in self._objectives]       

    def genome(self, idx: int) -> Chromosome:
        return self._genomes[idx]

    def simulation(self, idx: int) -> ObjectId:
        return self._simulations[idx]

    def objectives(self, idx: int) -> list[float]:
        return self._objectives[idx]
    
    def dont_have_objectives(self) -> bool:
        return not self._objectives

def select_next_population(
    selected_idxs: list[int],
    pop_size: int,
    P: PopulationSnapshot,
    Q: PopulationSnapshot,
) -> PopulationSnapshot:
    """
    Builds the next parent population after environmental selection
    from R = P ∪ Q.

    Assumes:
      - P is fully evaluated
      - Q is fully evaluated
      - selected_idxs refer to indices in R
    """

    if pop_size <= 0:
        raise ValueError("pop_size must be positive")

    if len(selected_idxs) < pop_size:
        raise ValueError("selected_idxs smaller than pop_size")

    # Build union R
    P_genomes = P.get_genomes()
    Q_genomes = Q.get_genomes()
    R_genomes = P_genomes + Q_genomes

    P_objectives = P.get_objectives()
    Q_objectives = Q.get_objectives()
    R_objectives = P_objectives + Q_objectives

    if len(R_genomes) != len(R_objectives):
        raise ValueError("R_genomes and R_objectives size mismatch")

    genomes: list[Chromosome] = []
    objectives: list[list[float]] = []
    sim_oid_to_idx: dict[str, int] = {}

    total_size = len(R_genomes)
    p_size = len(P)

    for i, idx in enumerate(selected_idxs[:pop_size]):
        if idx < 0 or idx >= total_size:
            raise IndexError(f"Invalid selected index {idx}")

        genomes.append(R_genomes[idx])
        objectives.append(R_objectives[idx])

        if idx < p_size:
            sim_oid = P.simulation(idx)
        else:
            sim_oid = Q.simulation(idx - p_size)

        sim_oid_to_idx[str(sim_oid)] = i

    snapshot = PopulationSnapshot()
    snapshot.set(
        genomes=genomes,
        sim_oid_to_idx=sim_oid_to_idx,
        objectives=objectives,
    )
    
    print(f"\nSelected genomes for next population: {genomes}")

    return snapshot