"""Regression: the generation document must be inserted BEFORE its simulations.

Fast workers (synthetic mode) can complete every simulation in milliseconds; if
the generation document did not exist yet, master-node's generation mark_done()
would target a missing document (no-op) and the generation would hang at index 0.
This test locks the ordering invariant for every strategy that enqueues work.
"""
from unittest.mock import MagicMock

from bson import ObjectId
import pytest

from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.random_search import RandomSearchStrategy
from lib.strategy.batch import BatchStrategy


def _make_strategy(cls, enqueue_name):
    """Build a strategy instance bypassing __init__, wired with mocks and one
    fake genome, and return (strategy, insert_order_list, enqueue_callable)."""
    strat = cls.__new__(cls)
    strat._is_synthetic = True          # skip CSC/source-repo lookups
    strat._exp_id = ObjectId()
    strat._gen_index = 0
    strat._sim_rand_seeds = [42]
    strat._genome_objectives_cache = {}
    strat._inserted_genomes = set()
    strat._sim_done_count = 0
    strat._count_sims_inserted = 0
    strat._objective_keys = ["f1"]
    strat._map_genome_objectives = {}
    # Feasible genome (no penalty) so the enqueue reaches the simulation path.
    strat._problem_adapter = MagicMock()
    strat._problem_adapter.penalty_objectives.return_value = None
    # Simulation-based problem (P1–P4): NOT analytical, so the enqueue keeps
    # inserting Simulation documents (a bare MagicMock would be truthy here).
    strat._problem_adapter.is_analytical = False

    order: list[str] = []
    mongo = MagicMock()
    mongo.generation_repo.insert.side_effect = lambda doc: (order.append("generation"), doc["_id"])[1]
    mongo.simulation_repo.insert.side_effect = lambda doc: (order.append("simulation"), ObjectId())[1]
    strat.mongo = mongo

    genome = MagicMock()
    genome.get_hash.return_value = "hash-1"
    genome.to_dict.return_value = {"relays": []}
    strat._current_population = [genome]
    strat._chromosomes = [genome]  # batch uses this name

    # Stub heavy helpers (config building / topology upload)
    strat._convert_genome_to_sim_config = MagicMock(
        return_value={"randomSeed": 0, "simulationElements": {"fixedMotes": []}}
    )
    strat._upload_topology_async = MagicMock()

    return strat, order, getattr(strat, enqueue_name)


@pytest.mark.parametrize("cls,enqueue_name", [
    (NSGA3LoopStrategy, "_generation_enqueue"),
    (NSGA2LoopStrategy, "_generation_enqueue"),
    (RandomSearchStrategy, "_generation_enqueue"),
    (BatchStrategy, "_enqueue_batch"),
])
def test_generation_inserted_before_simulations(cls, enqueue_name):
    strat, order, enqueue = _make_strategy(cls, enqueue_name)

    enqueue()

    assert "generation" in order, "generation document was never inserted"
    assert "simulation" in order, "no simulation was inserted"
    # The generation must be created before any simulation.
    assert order.index("generation") < order.index("simulation"), (
        f"{cls.__name__}: generation inserted after simulations — "
        f"master-node could mark_done() a non-existent generation. order={order}"
    )
    # And _generation_id must be set so the reconcile/close path can find it.
    assert strat._generation_id is not None


@pytest.mark.parametrize("cls", [NSGA3LoopStrategy, NSGA2LoopStrategy, RandomSearchStrategy])
def test_analytical_enqueue_skips_simulations(cls):
    """Analytical (P0) problems evaluate in-process: the enqueue inserts
    Individuals with objectives but NO Simulation documents, and marks the
    generation DONE directly (no master-node round-trip)."""
    strat = cls.__new__(cls)
    strat._is_synthetic = True
    strat._exp_id = ObjectId()
    strat._gen_index = 0
    strat._sim_rand_seeds = [42]
    strat._genome_objectives_cache = {}
    strat._inserted_genomes = set()
    strat._count_sims_inserted = 0
    strat._objective_keys = ["f1", "f2"]
    strat._objective_goals = [1, 1]
    strat._map_genome_objectives = {}

    strat._problem_adapter = MagicMock()
    strat._problem_adapter.is_analytical = True
    strat._problem_adapter.decision_vector.return_value = [0.5, 0.5]
    strat._bench = "ZDT1"
    strat._noise_std = 0.0
    strat._sch1_domain = (-5.0, 5.0)

    order: list[str] = []
    mongo = MagicMock()
    mongo.generation_repo.insert.side_effect = lambda doc: (order.append("generation"), doc["_id"])[1]
    mongo.simulation_repo.insert.side_effect = lambda doc: (order.append("simulation"), ObjectId())[1]
    strat.mongo = mongo

    genome = MagicMock()
    genome.get_hash.return_value = "hash-1"
    genome.to_dict.return_value = {"x": [0.5, 0.5]}
    strat._current_population = [genome]

    strat._convert_genome_to_sim_config = MagicMock(return_value={"randomSeed": 0, "simulationElements": {}})
    strat._upload_topology_async = MagicMock()
    strat._fire_generation_done = MagicMock()   # don't spawn the evolution loop

    strat._generation_enqueue()

    assert "generation" in order
    assert "simulation" not in order, f"{cls.__name__}: analytical problems must not enqueue simulations"
    mongo.individual_repo.insert.assert_called_once()
    assert genome in strat._map_genome_objectives
    mongo.generation_repo.mark_done.assert_called_once()
    assert strat._generation_id is not None
