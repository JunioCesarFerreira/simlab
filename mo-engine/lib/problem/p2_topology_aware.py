from typing import Any, Mapping, Sequence
import logging

from pylib.config.algorithm import GeneticAlgorithmConfigDto

from lib.genetic_operators.crossover.uniform_crossover_mask import uniform_crossover_mask
from lib.genetic_operators.mutation.bitflip_mutation import bitflip_mutation
from lib.util.random_network import stochastic_reachability_mask

from .adapter import ChromosomeP2, Random
from .p2_discrete_mobility import Problem2DiscreteMobilityAdapter
from .topology import (
    ParentArrayTree,
    RepairResult,
    RepairWeights,
    RoutingKnowledge,
    ScenarioTopology,
    TopologyDescriptorExtractor,
    TopologyDescriptors,
    TopologyRepair,
    TreeCostWeights,
)

log = logging.getLogger(__name__)

# Penalty applied to a chromosome the structural repair could not make valid.
# Must dominate the coverage penalty of the base adapter (1e9 * (1 + deficit),
# deficit <= 1) so "structurally broken" ranks worse than "under-covered",
# while staying below the WORST_OBJECTIVE sentinel (1e12) used for
# "no metrics at all".
_STRUCTURAL_PENALTY = 5e9


# ============================================================
# Problem 2 (topology-aware variant)
# Chromosome: identical binary mask over candidate positions Q.
# Added: sink-rooted structural tree, structure-driven repair and
#        cheap descriptors phi(x) for adaptive simulation strategies.
# ============================================================

class Problem2TopologyAwareAdapter(Problem2DiscreteMobilityAdapter):
    """Topology-aware variant of Problem 2.

    Keeps the semantics **and the binary chromosome** of
    :class:`Problem2DiscreteMobilityAdapter` — same candidate set, same sink,
    same mobile fleet, same coverage constraint, same Cooja encoding — so
    experiments can be compared one-to-one against the original P2.

    What changes is everything *around* the chromosome:

    * a :class:`ScenarioTopology` cache (adjacency, distances, temporal
      coverage) computed once per scenario;
    * a sink-rooted :class:`ParentArrayTree` derived from each individual;
    * a structure-driven :class:`TopologyRepair` replacing the global BFS
      repair after crossover/mutation;
    * cheap, deterministic descriptors ``phi(x)`` and a scenario fingerprint,
      both consumed by the adaptive-simulation strategy;
    * an optional :class:`RoutingKnowledge` matrix of observed link importance.
    """

    # Inherited: per_gene_prob, apply_coverage_repair, repair_coverage_budget.
    CONSUMED_GA_KEYS = Problem2DiscreteMobilityAdapter.CONSUMED_GA_KEYS | frozenset({
        "topology_repair_budget",
    })

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def assert_problem(self, problem: Mapping[str, Any]) -> None:
        super().assert_problem(problem)

        cfg = dict(problem.get("topology_heuristic") or {})
        self._topology_enabled: bool = bool(cfg.get("enabled", True))

        self._scenario = ScenarioTopology(
            candidates=self.problem.candidates,
            sink=self.problem.sink,
            radius=self.problem.radius_of_reach,
            mobile_nodes=self.problem.mobile_nodes,
        )
        self._routing = RoutingKnowledge()
        self._repair_weights = RepairWeights(
            routing_importance_weight=float(cfg.get("routing_importance_weight", 1.0)),
            distance_weight=float(cfg.get("distance_weight", 1.0)),
            structural_quality_weight=float(cfg.get("structural_quality_weight", 1.0)),
            relay_cost_weight=float(cfg.get("relay_cost_weight", 1.0)),
        )
        self._tree_cost = TreeCostWeights(
            distance_weight=float(cfg.get("tree_distance_weight", 1.0)),
            routing_importance_weight=float(cfg.get("tree_routing_importance_weight", 0.0)),
        )
        self._repair_budget = int(
            cfg.get("repair_budget", max(16, len(self.problem.candidates)))
        )
        self._build_topology_components()

        # Scenario-level context folded into the fingerprint by the strategy.
        self._fingerprint_extra: dict[str, Any] = {
            "radius_of_inter": self.problem.radius_of_inter,
            "region": list(self.problem.region),
            "min_coverage_percentage": self.problem.min_coverage_percentage,
        }
        log.info(
            "[P2-topology] Scenario cached: %d candidates, %d time slices, "
            "max degree %d.",
            self._scenario.n_candidates,
            len(self._scenario.time_slices),
            self._scenario.max_degree,
        )

    def _build_topology_components(self) -> None:
        self._repairer = TopologyRepair(
            scenario=self._scenario,
            weights=self._repair_weights,
            tree_cost=self._tree_cost,
            routing=self._routing,
            max_iterations=self._repair_budget,
        )
        self._descriptor_extractor = TopologyDescriptorExtractor(
            scenario=self._scenario,
            routing=self._routing,
        )

    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto) -> None:
        super().set_ga_operator_configs(rng, parameters)
        budget = parameters.get("topology_repair_budget")
        if budget is not None:
            self._repair_budget = int(budget)
            self._build_topology_components()

    # ------------------------------------------------------------------
    # Structural API consumed by the adaptive strategy
    # ------------------------------------------------------------------
    @property
    def scenario(self) -> ScenarioTopology:
        return self._scenario

    @property
    def routing_knowledge(self) -> RoutingKnowledge:
        return self._routing

    @property
    def descriptor_extractor(self) -> TopologyDescriptorExtractor:
        return self._descriptor_extractor

    def scenario_fingerprint(self, extra: Mapping[str, Any] | None = None) -> str:
        """Digest identifying the scenario a piece of knowledge belongs to."""
        context = dict(self._fingerprint_extra)
        if extra:
            context.update(extra)
        return self._scenario.fingerprint(context)

    def build_tree(self, chromosome: ChromosomeP2) -> ParentArrayTree:
        """Sink-rooted structural tree ``T_x`` of ``chromosome``."""
        return self._repairer.build_tree(chromosome.mask)

    def structural_repair(self, mask: Sequence[int]) -> RepairResult:
        """Run the structure-driven repair on a raw mask."""
        return self._repairer.repair(mask)

    def descriptors(self, chromosome: ChromosomeP2) -> TopologyDescriptors:
        """Cheap descriptors ``phi(x)``; deterministic for scenario+chromosome."""
        tree = self.build_tree(chromosome)
        return self._descriptor_extractor.extract(chromosome.mask, tree)

    def observe_simulated(self, chromosome: ChromosomeP2, key: str | None = None) -> bool:
        """Fold a really-simulated individual into the routing knowledge."""
        return self._routing.observe_tree(self.build_tree(chromosome), key=key)

    def is_structurally_feasible(self, chromosome: ChromosomeP2) -> bool:
        """Whether every active relay of ``chromosome`` reaches the sink."""
        return not self.build_tree(chromosome).detached_nodes()

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------
    def penalty_objectives(self, chromosome: ChromosomeP2, n_objectives: int) -> list[float] | None:
        """Coverage penalty (inherited) plus a structural-connectivity penalty.

        A chromosome whose active relays cannot all reach the sink is not worth
        simulating at all: RPL would never build a DODAG over the orphan
        component.  Such individuals are ranked below every under-covered but
        connected one.
        """
        if self._topology_enabled and not self.is_structurally_feasible(chromosome):
            return [_STRUCTURAL_PENALTY] * n_objectives
        return super().penalty_objectives(chromosome, n_objectives)

    # ------------------------------------------------------------------
    # Repair pipeline
    # ------------------------------------------------------------------
    def _structural_repair_mask(self, mask: list[int], context: str) -> list[int] | None:
        """Structure-driven repair followed by the inherited coverage repair.

        Returns ``None`` when the individual could not be repaired, so the
        caller can fall back to the parent genome exactly like the base P2 does
        when ``repair_connectivity_to_sink`` fails.
        """
        if not self._topology_enabled:
            return super()._repair_mask(mask)

        result = self._repairer.repair(mask)
        if not result.feasible:
            log.warning(
                "[P2-topology] Structural repair failed (%s): %s.",
                context,
                result.reason or "unknown",
            )
            return None

        repaired = super()._repair_mask(result.mask)
        if repaired == result.mask:
            return repaired

        # Coverage repair activates candidates anywhere in Q, so it can break
        # the sink-rooted structure again: re-run the structural pass and keep
        # the coverage gain only when the result stays connected.
        second = self._repairer.repair(repaired)
        if not second.feasible:
            log.warning("[P2-topology] Coverage repair discarded (%s): broke connectivity.", context)
            return result.mask
        return second.mask

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def random_individual_generator(self, size: int) -> list[ChromosomeP2]:
        """Same stochastic growth as the base P2, with structural repair."""
        Q = self.problem.candidates
        S = self.problem.sink
        R = self.problem.radius_of_reach

        pop: list[ChromosomeP2] = []
        seen: set[str] = set()
        max_attempts = 20

        for _ in range(size):
            chrm: ChromosomeP2 | None = None
            candidate: ChromosomeP2 | None = None
            for _attempt in range(max_attempts):
                raw = stochastic_reachability_mask(Q, S, R, self._rng)
                mask = self._structural_repair_mask(raw, "random")
                if mask is None:
                    continue
                candidate = ChromosomeP2(
                    mac_protocol=self._rng.randint(0, 1),
                    mask=mask,
                )
                if candidate.get_hash() not in seen:
                    chrm = candidate
                    break
            if chrm is None:
                if candidate is None:
                    raise RuntimeError(
                        "[P2-topology] Could not generate any structurally valid "
                        "individual; check the candidate set and radius_of_reach."
                    )
                log.warning(
                    "[P2-topology] random_individual_generator: no unique genome "
                    "after %d attempts; accepting a duplicate.", max_attempts,
                )
                chrm = candidate
            seen.add(chrm.get_hash())
            pop.append(chrm)
        return pop

    def crossover(self, parents: Sequence[ChromosomeP2]) -> list[ChromosomeP2]:
        p1: ChromosomeP2 = parents[0]
        p2: ChromosomeP2 = parents[1]
        c1, c2 = uniform_crossover_mask(p1.mask, p2.mask, self._rng)

        m1 = self._structural_repair_mask(c1, "crossover-c1")
        if m1 is None:
            m1 = p1.mask
        m2 = self._structural_repair_mask(c2, "crossover-c2")
        if m2 is None:
            m2 = p2.mask

        mac1 = p1.mac_protocol if self._rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if self._rng.random() < 0.5 else p1.mac_protocol

        return [
            ChromosomeP2(mac_protocol=mac1, mask=m1),
            ChromosomeP2(mac_protocol=mac2, mask=m2),
        ]

    def mutate(self, chromosome: ChromosomeP2) -> ChromosomeP2:
        mutated = bitflip_mutation(chromosome.mask, self._p_bit_mut, self._rng)
        out = self._structural_repair_mask(mutated, "mutation")
        if out is None:
            out = chromosome.mask

        mac = chromosome.mac_protocol
        if self._rng.random() < self._p_bit_mut:
            mac = 1 - mac

        return ChromosomeP2(mac_protocol=mac, mask=out)
