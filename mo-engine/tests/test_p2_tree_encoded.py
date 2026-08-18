"""The tree-encoded P2 variant: feasibility without a repair pass.

The central claim under test is that the connectivity repair is not merely
*unused* but *unnecessary*: the operators are closed over feasible trees, so
no output of crossover, mutation or initialisation can be disconnected.
"""
import random

import pytest

from lib.problem.chromosomes import ChromosomeP2, ChromosomeP2Tree, chromosome_from_dict
from lib.problem.p2_topology_aware import Problem2TopologyAwareAdapter
from lib.problem.p2_tree_encoded import Problem2TreeEncodedAdapter
from lib.problem.resolve import build_adapter, build_test_adapter
from lib.problem.topology import SINK_NODE, ParentArrayTree, TwoLevelTree, build_sink_rooted_tree
from lib.problem.topology.tree_operators import INACTIVE, CoverageModel, TreeOperators


def _problem(name: str = "problem2_tree_encoded", **overrides) -> dict:
    problem = {
        "name": name,
        "region": [-100.0, -100.0, 100.0, 100.0],
        "sink": (0.0, 0.0),
        "candidates": [
            (float(x * 20 - 60), float(y * 20 - 60))
            for x in range(7) for y in range(7)
        ],
        "mobile_nodes": [
            {
                "path_segments": [("-60 + 120*t", "40")],
                "is_closed": False,
                "is_round_trip": True,
                "speed": 5.0,
                "time_step": 1.0,
            }
        ],
        "radius_of_reach": 30.0,
        "radius_of_inter": 60.0,
        "min_coverage_percentage": 80.0,
    }
    problem.update(overrides)
    return problem


def _adapter(seed: int = 7, **overrides) -> Problem2TreeEncodedAdapter:
    return build_adapter(
        _problem(**overrides),
        {"per_gene_prob": 0.1, "tree_mutation_moves": 3},
        random.Random(seed),
    )


def _assert_feasible(adapter, chromosome, label: str) -> None:
    """The mask is sink-connected and every genotype edge respects R_com."""
    canonical = build_sink_rooted_tree(adapter.scenario, chromosome.mask)
    assert canonical.detached_nodes() == [], f"{label}: mask is not connected to the sink"

    tree = adapter.genotype_tree(chromosome)
    assert tree.detached_nodes() == [] or all(
        chromosome.mask[n] == 0 for n in tree.detached_nodes()
    ), f"{label}: an active relay is detached in the genotype tree"

    for node in tree.attached_nodes():
        if node == SINK_NODE:
            continue
        parent = tree.parent(node)
        distance = (
            adapter.scenario.distance_to_sink(node) if parent == SINK_NODE
            else adapter.scenario.distance(node, parent)
        )
        assert distance <= adapter.scenario.radius + 1e-9, (
            f"{label}: edge {node}->{parent} is {distance:.2f} > R_com"
        )


# ---------------------------------------------------------------------------
# Registry and compatibility
# ---------------------------------------------------------------------------
def test_registry_exposes_three_independent_p2_variants():
    classic = build_test_adapter(_problem("problem2"))
    aware = build_test_adapter(_problem("problem2_topology_aware"))
    encoded = build_test_adapter(_problem("problem2_tree_encoded"))

    assert type(classic).__name__ == "Problem2DiscreteMobilityAdapter"
    assert type(aware) is Problem2TopologyAwareAdapter
    assert type(encoded) is Problem2TreeEncodedAdapter


def test_cooja_encoding_is_unchanged():
    adapter = _adapter()
    chromosome = ChromosomeP2Tree(
        mac_protocol=0, mask=[1] + [0] * 48, tree_parents=(SINK_NODE,) + (INACTIVE,) * 48
    )
    encoded = adapter.encode_simulation_input(chromosome)

    assert encoded["fixedMotes"][0]["name"] == "sink"
    assert len(encoded["fixedMotes"]) == 2
    assert encoded["fixedMotes"][1]["name"] == "relay_0"


class TestChromosomeInteroperability:
    def test_tree_is_genotype_and_never_changes_identity(self):
        plain = ChromosomeP2(mac_protocol=0, mask=[1, 0, 1])
        one = ChromosomeP2Tree(mac_protocol=0, mask=[1, 0, 1], tree_parents=(-1, -2, 0))
        other = ChromosomeP2Tree(mac_protocol=0, mask=[1, 0, 1], tree_parents=(-1, -2, -1))

        # Same deployment => same individual => one genome-cache entry.
        assert plain.get_hash() == one.get_hash() == other.get_hash()
        assert plain == one and one == other
        assert len({plain, one, other}) == 1

    def test_tree_survives_persistence(self):
        chromosome = ChromosomeP2Tree(
            mac_protocol=1, mask=[1, 1, 0], tree_parents=(-1, 0, -2)
        )
        restored = chromosome_from_dict("problem2_tree_encoded", chromosome.to_dict())

        assert restored == chromosome
        assert restored.tree_parents == (-1, 0, -2)
        assert restored.mac_protocol == 1

    def test_document_without_a_tree_still_restores(self):
        restored = chromosome_from_dict(
            "problem2_tree_encoded", {"mac_protocol": 0, "mask": [1, 0, 1]}
        )
        assert restored.tree_parents == ()
        assert restored.mask == [1, 0, 1]

    def test_a_mask_only_chromosome_is_accepted_by_the_operators(self):
        """Seeding a tree-encoded run from a mask-encoded one must work."""
        adapter = _adapter()
        seeded = Problem2TopologyAwareAdapter(_problem("problem2_topology_aware"))
        seeded.set_ga_operator_configs(random.Random(3), {"per_gene_prob": 0.1})
        plain = seeded.random_individual_generator(1)[0]

        tree = adapter.genotype_tree(plain)
        assert tree.detached_nodes() == []
        child = adapter.mutate(plain)
        _assert_feasible(adapter, child, "mutated-from-mask")

    def test_inconsistent_parent_array_falls_back_to_the_canonical_tree(self):
        adapter = _adapter()
        chromosome = adapter.random_individual_generator(1)[0]
        corrupted = ChromosomeP2Tree(
            mac_protocol=chromosome.mac_protocol,
            mask=chromosome.mask,
            tree_parents=tuple(99 for _ in chromosome.tree_parents),
        )
        tree = adapter.genotype_tree(corrupted)

        assert tree.detached_nodes() == []
        assert set(n for n in tree.attached_nodes() if n != SINK_NODE) == {
            i for i, bit in enumerate(chromosome.mask) if bit
        }


# ---------------------------------------------------------------------------
# Feasibility closure — the point of the encoding
# ---------------------------------------------------------------------------
class TestNoRepairNeeded:
    def test_initial_population_is_feasible(self):
        adapter = _adapter()
        population = adapter.random_individual_generator(15)

        assert len(population) == 15
        for chromosome in population:
            assert isinstance(chromosome, ChromosomeP2Tree)
            _assert_feasible(adapter, chromosome, "init")

    def test_offspring_stay_feasible_over_many_generations(self):
        adapter = _adapter()
        parents = adapter.random_individual_generator(2)

        for generation in range(40):
            children = adapter.crossover(parents)
            assert len(children) == 2
            for child in children:
                _assert_feasible(adapter, child, f"crossover-g{generation}")
            mutated = adapter.mutate(children[0])
            _assert_feasible(adapter, mutated, f"mutation-g{generation}")
            parents = [children[0], mutated]

    def test_connectivity_repair_is_never_invoked(self, monkeypatch):
        """Hard proof: make every repair entry point explode, then evolve."""
        import lib.util.connectivity as connectivity
        from lib.problem.topology.repair import TopologyRepair

        def _boom(*args, **kwargs):
            raise AssertionError("a connectivity repair function was called")

        monkeypatch.setattr(connectivity, "repair_connectivity_to_sink", _boom)
        monkeypatch.setattr(TopologyRepair, "repair", _boom)

        adapter = _adapter(seed=21)
        parents = adapter.random_individual_generator(6)
        for _ in range(25):
            children = adapter.crossover(parents[:2])
            mutated = adapter.mutate(children[0])
            parents = [mutated, children[1]]
            _assert_feasible(adapter, mutated, "no-repair")

    def test_the_repair_hook_is_wired_shut(self):
        adapter = _adapter()
        with pytest.raises(AssertionError, match="tree encoding"):
            adapter._structural_repair_mask([0] * 49, "manual")

    def test_structural_feasibility_always_holds(self):
        adapter = _adapter()
        for chromosome in adapter.random_individual_generator(10):
            assert adapter.is_structurally_feasible(chromosome)
            penalty = adapter.penalty_objectives(chromosome, 3)
            # Only the coverage constraint can ever penalise this variant.
            assert penalty is None or penalty[0] < 5e9


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class TestOperators:
    def _ops(self, seed: int = 5, backend=TwoLevelTree) -> TreeOperators:
        adapter = _adapter(seed)
        return TreeOperators(
            scenario=adapter.scenario,
            rng=random.Random(seed),
            coverage=CoverageModel(
                cover_bits=adapter._candidate_cover_bits,
                n_points=len(adapter._coverage_matrix),
                threshold=80.0,
            ),
            tree_factory=lambda root: backend(root),
        )

    def test_grow_only_links_admissible_parents(self):
        ops = self._ops()
        tree = ops.new_tree()
        assert ops.grow(tree, count=20) > 0

        for node in tree.attached_nodes():
            if node == SINK_NODE:
                continue
            parent = tree.parent(node)
            assert parent in ops.admissible_parents(tree, node) or parent == SINK_NODE

    def test_prune_deactivates_a_whole_subtree(self):
        ops = self._ops()
        tree = ops.random_tree()
        before = set(tree.attached_nodes())
        assert ops.prune(tree, count=1) == 1
        after = set(tree.attached_nodes())

        assert after < before
        assert tree.root in after

    def test_pao_moves_a_subtree_without_breaking_the_tree(self):
        ops = self._ops(seed=8)
        tree = ops.random_tree()
        before = set(tree.attached_nodes())

        if ops.pao(tree):
            assert set(tree.attached_nodes()) == before, "PAO must preserve the relay set"
        assert all(tree.is_connected_to_root(n) for n in tree.attached_nodes())

    def test_cao_reroots_and_preserves_the_relay_set(self):
        ops = self._ops(seed=17)
        tree = ops.random_tree()
        before = set(tree.attached_nodes())

        if ops.cao(tree):
            assert set(tree.attached_nodes()) == before, "CAO must preserve the relay set"
        assert all(tree.is_connected_to_root(n) for n in tree.attached_nodes())

    def test_transplant_grafts_the_donor_block_intact(self):
        ops = self._ops(seed=4)
        host = ops.random_tree()
        donor = ops.random_tree()

        if ops.transplant(host, donor):
            assert all(host.is_connected_to_root(n) for n in host.attached_nodes())
            for node in host.attached_nodes():
                if node == SINK_NODE:
                    continue
                parent = host.parent(node)
                assert parent == SINK_NODE or parent in ops.scenario.adjacency[node]

    def test_grow_to_coverage_reaches_the_threshold_when_possible(self):
        ops = self._ops(seed=12)
        tree = ops.new_tree()
        ops.grow(tree, count=1)
        ops.grow_to_coverage(tree, budget=49)

        active = [n for n in tree.attached_nodes() if n != SINK_NODE]
        assert ops.coverage.score(active) >= 80.0
        assert all(tree.is_connected_to_root(n) for n in tree.attached_nodes())

    def test_parent_array_round_trips_through_a_mask(self):
        ops = self._ops()
        tree = ops.random_tree()
        n = ops.scenario.n_candidates
        from lib.problem.topology.tree_operators import mask_from_tree

        mask = mask_from_tree(tree, n)
        parents = ops.parent_array(tree, n)
        rebuilt = ops.tree_from_mask(mask, parents)

        assert mask_from_tree(rebuilt, n) == mask
        for node in rebuilt.attached_nodes():
            if node != SINK_NODE:
                assert rebuilt.parent(node) == parents[node]

    def test_inactive_candidates_are_marked_distinctly_from_sink_children(self):
        ops = self._ops()
        tree = ops.random_tree()
        parents = ops.parent_array(tree, ops.scenario.n_candidates)

        active = {n for n in tree.attached_nodes() if n != SINK_NODE}
        for index, parent in enumerate(parents):
            assert (parent == INACTIVE) == (index not in active)

    @pytest.mark.parametrize("backend", [TwoLevelTree, ParentArrayTree])
    def test_operators_run_on_either_backend(self, backend):
        ops = self._ops(seed=6, backend=backend)
        tree = ops.random_tree()
        for _ in range(20):
            ops.pao(tree)
            ops.cao(tree)
            ops.grow(tree, 1)
            ops.prune(tree, 1)
            assert all(tree.is_connected_to_root(n) for n in tree.attached_nodes())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class TestConfiguration:
    def test_backend_is_selectable(self):
        two_level = _adapter(tree_encoding={"backend": "two_level"})
        flat = _adapter(tree_encoding={"backend": "parent_array"})

        assert isinstance(two_level.operators.new_tree(), TwoLevelTree)
        assert isinstance(flat.operators.new_tree(), ParentArrayTree)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown tree backend"):
            build_test_adapter(_problem(tree_encoding={"backend": "quantum"}))

    def test_mutation_move_count_is_configurable(self):
        adapter = build_adapter(
            _problem(), {"per_gene_prob": 0.1, "tree_mutation_moves": 7}, random.Random(1)
        )
        assert adapter._tree_mutation_moves == 7

    def test_coverage_growth_can_be_disabled(self):
        adapter = build_adapter(
            _problem(), {"per_gene_prob": 0.1, "apply_coverage_repair": False}, random.Random(1)
        )
        population = adapter.random_individual_generator(5)
        for chromosome in population:
            _assert_feasible(adapter, chromosome, "no-coverage-growth")
