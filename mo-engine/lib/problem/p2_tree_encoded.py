from typing import Any, Mapping, Sequence
import logging

from pylib.config.algorithm import GeneticAlgorithmConfigDto

from .adapter import Random
from .chromosomes import ChromosomeP2, ChromosomeP2Tree
from .p2_topology_aware import Problem2TopologyAwareAdapter
from .topology import ParentArrayTree, TwoLevelTree
from .topology.tree_operators import CoverageModel, TreeOperators, mask_from_tree

log = logging.getLogger(__name__)

#: Selectable rooted-tree backends. Both satisfy ``RootedTreeBackend``; the
#: two-level one is the default because its splices stay O(sqrt(n)).
TREE_BACKENDS = {
    "two_level": TwoLevelTree,
    "parent_array": ParentArrayTree,
}


# ============================================================
# Problem 2 (tree-encoded variant)
# Chromosome: the same binary mask, plus the sink-rooted tree that produced it.
# Operators: PAO / CAO / grow / prune on a two-level rooted forest.
# Repair: none - every operator is feasibility-preserving by construction.
# ============================================================

class Problem2TreeEncodedAdapter(Problem2TopologyAwareAdapter):
    """Tree-encoded variant of Problem 2.

    Subclasses :class:`Problem2TopologyAwareAdapter`, so it inherits the
    scenario caches, the descriptors, the scenario fingerprint and the routing
    knowledge — everything the adaptive-simulation strategy needs. What it
    replaces is the **variation pipeline**.

    ### Why there is no repair

    The mask-encoded variants let an operator produce a disconnected
    chromosome and then fix it (``repair_connectivity_to_sink`` globally, or
    :class:`TopologyRepair` structurally). Here the operators work on the
    sink-rooted tree itself and only ever

    * link a node under a parent within ``R_com`` of it, and
    * link it under a parent that already reaches the sink, and
    * remove relays by cutting a subtree,

    all three of which map feasible trees to feasible trees. The induced mask
    is therefore connected to the sink by construction, and no repair function
    is called anywhere in this adapter. Coverage is handled the same way:
    :meth:`TreeOperators.grow_to_coverage` runs the same greedy set-cover as
    the mask variant, but restricted to the admissible frontier, so it can
    never need a follow-up connectivity fix either.

    ### Genotype vs phenotype

    ``ChromosomeP2Tree`` carries ``tree_parents`` alongside ``mask``, and the
    tree is genotype only: equality and hashing ignore it, because two trees
    over the same relay set deploy identical motes and would run identical
    simulations. Concretely this means a tree-encoded experiment shares
    ``genome_cache`` entries with a mask-encoded one, and the descriptors
    ``phi(x)`` keep using the **canonical** shortest-path tree of the mask
    (inherited ``build_tree``) so they stay a deterministic function of
    ``scenario + chromosome``, as the estimator requires. The genotype tree is
    reachable through :meth:`genotype_tree`.
    """

    CONSUMED_GA_KEYS = Problem2TopologyAwareAdapter.CONSUMED_GA_KEYS | frozenset({
        "tree_mutation_moves",
    })

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def assert_problem(self, problem: Mapping[str, Any]) -> None:
        super().assert_problem(problem)

        cfg = dict(problem.get("tree_encoding") or {})
        backend_name = str(cfg.get("backend", "two_level"))
        if backend_name not in TREE_BACKENDS:
            raise ValueError(
                f"Unknown tree backend {backend_name!r}; known: {', '.join(sorted(TREE_BACKENDS))}."
            )
        self._tree_backend_name = backend_name
        self._tree_mutation_moves = int(cfg.get("mutation_moves", 2))
        self._max_relays: int | None = (
            int(cfg["max_relays"]) if cfg.get("max_relays") is not None else None
        )
        self._operators: TreeOperators | None = None
        log.info(
            "[P2-tree] Tree encoding enabled: backend=%s, mutation_moves=%d.",
            backend_name, self._tree_mutation_moves,
        )

    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto) -> None:
        super().set_ga_operator_configs(rng, parameters)
        moves = parameters.get("tree_mutation_moves")
        if moves is not None:
            self._tree_mutation_moves = int(moves)
        self._operators = self._build_operators(rng)

    def _build_operators(self, rng: Random) -> TreeOperators:
        coverage = CoverageModel(
            cover_bits=self._candidate_cover_bits,
            n_points=len(self._coverage_matrix),
            threshold=self.problem.min_coverage_percentage,
        )
        backend = TREE_BACKENDS[self._tree_backend_name]
        return TreeOperators(
            scenario=self._scenario,
            rng=rng,
            coverage=coverage,
            tree_factory=lambda root: backend(root),
        )

    @property
    def operators(self) -> TreeOperators:
        """Feasibility-preserving operator set (built on first GA configuration)."""
        if self._operators is None:
            self._operators = self._build_operators(getattr(self, "_rng", Random()))
        return self._operators

    # ------------------------------------------------------------------
    # Genotype access
    # ------------------------------------------------------------------
    def genotype_tree(self, chromosome: ChromosomeP2):
        """The tree the operators work on — the chromosome's own, when it has one.

        A plain :class:`ChromosomeP2` (seeded from a mask-encoded run, or
        restored from a document written before this variant existed) falls
        back to the canonical shortest-path tree, which is always feasible for
        a P2-valid mask. That is what makes the two encodings interchangeable
        as starting points.
        """
        parents = getattr(chromosome, "tree_parents", None)
        return self.operators.tree_from_mask(chromosome.mask, parents or None)

    def _chromosome_from_tree(self, tree, mac_protocol: int) -> ChromosomeP2Tree:
        n = len(self.problem.candidates)
        return ChromosomeP2Tree(
            mac_protocol=mac_protocol,
            mask=mask_from_tree(tree, n),
            tree_parents=tuple(self.operators.parent_array(tree, n)),
        )

    def _finish(self, tree, mac_protocol: int) -> ChromosomeP2Tree:
        """Apply coverage growth (if enabled) and freeze the tree into a genome."""
        if self._apply_coverage_repair:
            self.operators.grow_to_coverage(tree, budget=self._repair_coverage_budget)
        return self._chromosome_from_tree(tree, mac_protocol)

    # ------------------------------------------------------------------
    # Genetic operators - no repair anywhere below this line
    # ------------------------------------------------------------------
    def random_individual_generator(self, size: int) -> list[ChromosomeP2Tree]:
        """Grow ``size`` random trees outwards from the sink.

        Replaces ``stochastic_reachability_mask`` + repair: growth starts at
        the sink and only ever adds admissible links, so every individual is
        feasible the moment it exists.
        """
        pop: list[ChromosomeP2Tree] = []
        seen: set[str] = set()
        max_attempts = 20

        for _ in range(size):
            chosen: ChromosomeP2Tree | None = None
            candidate: ChromosomeP2Tree | None = None
            for _attempt in range(max_attempts):
                tree = self.operators.random_tree(max_relays=self._max_relays)
                candidate = self._finish(tree, self._rng.randint(0, 1))
                if candidate.get_hash() not in seen:
                    chosen = candidate
                    break
            if chosen is None:
                if candidate is None:
                    raise RuntimeError(
                        "[P2-tree] Could not grow any tree from the sink; check "
                        "the candidate set and radius_of_reach."
                    )
                log.warning(
                    "[P2-tree] random_individual_generator: no unique genome after "
                    "%d attempts; accepting a duplicate.", max_attempts,
                )
                chosen = candidate
            seen.add(chosen.get_hash())
            pop.append(chosen)
        return pop

    def crossover(self, parents: Sequence[ChromosomeP2]) -> list[ChromosomeP2Tree]:
        """Subtree transplant in both directions.

        A whole structural block of one parent is grafted into the other,
        keeping its internal shape. The graft point is chosen among admissible
        parents only, so the child is a valid tree without any repair; if no
        admissible graft point exists the child is simply the unchanged parent.
        """
        p1, p2 = parents[0], parents[1]

        child1 = self.genotype_tree(p1)
        self.operators.transplant(child1, self.genotype_tree(p2))
        child2 = self.genotype_tree(p2)
        self.operators.transplant(child2, self.genotype_tree(p1))

        mac1 = p1.mac_protocol if self._rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if self._rng.random() < 0.5 else p1.mac_protocol
        return [self._finish(child1, mac1), self._finish(child2, mac2)]

    def mutate(self, chromosome: ChromosomeP2) -> ChromosomeP2Tree:
        """Apply a few random structural moves, then flip the MAC gene.

        Moves are drawn from PAO (re-hang a subtree), CAO (re-root then
        re-hang), grow (activate a relay) and prune (deactivate a subtree).
        Each is a no-op when nothing admissible exists, so mutation can never
        produce an infeasible individual.
        """
        tree = self.genotype_tree(chromosome)
        moves = (
            self.operators.pao,
            self.operators.cao,
            lambda t: self.operators.grow(t, 1) > 0,
            lambda t: self.operators.prune(t, 1) > 0,
        )
        for _ in range(max(1, self._tree_mutation_moves)):
            self._rng.choice(moves)(tree)

        mac = chromosome.mac_protocol
        if self._rng.random() < self._p_bit_mut:
            mac = 1 - mac
        return self._finish(tree, mac)

    # ------------------------------------------------------------------
    # Guardrail
    # ------------------------------------------------------------------
    def _structural_repair_mask(self, mask: list[int], context: str) -> list[int] | None:
        """Not part of this variant's pipeline.

        Kept as an explicit failure so that a future edit which reintroduces a
        repair call is caught immediately rather than silently re-enabling the
        behaviour this encoding exists to remove.
        """
        raise AssertionError(
            f"[P2-tree] Connectivity repair was invoked ({context}), but the "
            "tree encoding is supposed to make it unnecessary. Use the "
            "'problem2_topology_aware' variant if repair is wanted."
        )
