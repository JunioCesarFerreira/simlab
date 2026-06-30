"""Tests for uniform-rank fallback when all parents are infeasible.

When every parent in a generation is infeasible (penalty_objectives returns
non-None for all), _run_genetic_algorithm must skip NDS-based ranking and
assign rank=0 to every individual so that tournament selection is purely
random. This prevents gradient-penalty magnitudes from biasing exploration
when there is no feasible reference to guide convergence.
"""
import random
from unittest.mock import MagicMock, patch

import pytest

from lib.problem.chromosomes import Chromosome


# ---------------------------------------------------------------------------
# Minimal chromosome stub
# ---------------------------------------------------------------------------

class _FakeChromosome(Chromosome):
    def __init__(self, tag: str):
        self._tag = tag

    def to_dict(self):
        return {"tag": self._tag}

    def get_source_by_mac_protocol(self, options):
        return self, None

    def __repr__(self):
        return f"Fake({self._tag})"

    def __eq__(self, other):
        return isinstance(other, _FakeChromosome) and self._tag == other._tag

    def __hash__(self):
        return hash(self._tag)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_strategy(cls_path: str, algo_label: str, parents, objectives_map, penalty_map):
    """
    Build a minimal strategy instance (bypassing __init__) ready for
    _run_genetic_algorithm().

    penalty_map: dict[Chromosome, list[float] | None]
      None  → feasible
      list  → infeasible penalty vector
    """
    module_path, class_name = cls_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    StratCls = getattr(mod, class_name)

    s = StratCls.__new__(StratCls)
    s._objective_keys = ["m0", "m1"]
    s._pop_size = len(parents)
    s._prob_cx = 0.9
    s._prob_mt = 0.1
    s._ga_rng = random.Random(42)
    s._parents = parents
    s._map_genome_objectives = objectives_map

    adapter = MagicMock()
    adapter.penalty_objectives.side_effect = lambda g, n: penalty_map.get(g)
    adapter.crossover.side_effect = lambda pair: (pair[0], pair[1])
    adapter.mutate.side_effect = lambda g: g
    s._problem_adapter = adapter

    return s


# ---------------------------------------------------------------------------
# NSGA-III
# ---------------------------------------------------------------------------

class TestNSGA3InfeasibleUniformRank:
    _CLS = "lib.strategy.nsga3.NSGA3LoopStrategy"

    def _make(self, penalty_map):
        parents = [_FakeChromosome(f"g{i}") for i in range(len(penalty_map))]
        pmap = {p: penalty_map.get(_FakeChromosome(p._tag)) for p in parents}
        obj_map = {
            p: pmap[p] if pmap[p] is not None else [float(i), float(i + 0.5)]
            for i, p in enumerate(parents)
        }
        return _make_strategy(self._CLS, "NSGA-III", parents, obj_map, pmap)

    def test_all_infeasible_does_not_call_compute_ranks(self):
        """When all parents are infeasible, compute_individual_ranks must be bypassed."""
        penalty_map = {_FakeChromosome(f"g{i}"): [1e9 * (1 + i * 0.1)] * 2 for i in range(4)}
        s = self._make(penalty_map)
        with patch("lib.strategy.nsga3.compute_individual_ranks") as mock_ranks:
            s._run_genetic_algorithm()
        mock_ranks.assert_not_called()

    def test_all_infeasible_returns_pop_size_children(self):
        """Offspring count must equal pop_size even when all parents are infeasible."""
        penalty_map = {_FakeChromosome(f"g{i}"): [1e9 * (1 + i * 0.1)] * 2 for i in range(4)}
        s = self._make(penalty_map)
        children = s._run_genetic_algorithm()
        assert len(children) == s._pop_size

    def test_all_infeasible_children_from_parents(self):
        """All offspring must come from the existing parent pool."""
        penalty_map = {_FakeChromosome(f"g{i}"): [1e9 * (1 + i * 0.1)] * 2 for i in range(4)}
        s = self._make(penalty_map)
        children = s._run_genetic_algorithm()
        parent_set = set(s._parents)
        assert all(c in parent_set for c in children)

    def test_mixed_feasible_calls_compute_ranks(self):
        """When at least one parent is feasible, NDS-based ranking must be used."""
        penalty_map = {
            _FakeChromosome("g0"): [1e9, 1e9],
            _FakeChromosome("g1"): [1e9, 1e9],
            _FakeChromosome("g2"): None,          # feasible
            _FakeChromosome("g3"): [1e9, 1e9],
        }
        s = self._make(penalty_map)
        with patch("lib.strategy.nsga3.compute_individual_ranks") as mock_ranks:
            mock_ranks.return_value = {i: 0 for i in range(4)}
            s._run_genetic_algorithm()
        mock_ranks.assert_called_once()

    def test_all_feasible_calls_compute_ranks(self):
        """When all parents are feasible, NDS-based ranking must always be used."""
        penalty_map = {_FakeChromosome(f"g{i}"): None for i in range(4)}
        s = self._make(penalty_map)
        with patch("lib.strategy.nsga3.compute_individual_ranks") as mock_ranks:
            mock_ranks.return_value = {i: 0 for i in range(4)}
            s._run_genetic_algorithm()
        mock_ranks.assert_called_once()


# ---------------------------------------------------------------------------
# NSGA-II (same behaviour, same contract)
# ---------------------------------------------------------------------------

class TestNSGA2InfeasibleUniformRank:
    _CLS = "lib.strategy.nsga2.NSGA2LoopStrategy"

    def _make(self, penalty_map):
        parents = [_FakeChromosome(f"g{i}") for i in range(4)]
        pmap = {p: penalty_map.get(_FakeChromosome(p._tag)) for p in parents}
        obj_map = {
            p: pmap[p] if pmap[p] is not None else [float(i), float(i + 0.5)]
            for i, p in enumerate(parents)
        }
        return _make_strategy(self._CLS, "NSGA-II", parents, obj_map, pmap)

    def test_all_infeasible_does_not_call_compute_ranks(self):
        penalty_map = {_FakeChromosome(f"g{i}"): [1e9 * (1 + i * 0.1)] * 2 for i in range(4)}
        s = self._make(penalty_map)
        with patch("lib.strategy.nsga2.compute_individual_ranks") as mock_ranks:
            s._run_genetic_algorithm()
        mock_ranks.assert_not_called()

    def test_all_infeasible_returns_pop_size_children(self):
        penalty_map = {_FakeChromosome(f"g{i}"): [1e9 * (1 + i * 0.1)] * 2 for i in range(4)}
        s = self._make(penalty_map)
        children = s._run_genetic_algorithm()
        assert len(children) == s._pop_size

    def test_mixed_feasible_calls_compute_ranks(self):
        penalty_map = {
            _FakeChromosome("g0"): [1e9, 1e9],
            _FakeChromosome("g1"): None,
            _FakeChromosome("g2"): [1e9, 1e9],
            _FakeChromosome("g3"): [1e9, 1e9],
        }
        s = self._make(penalty_map)
        with patch("lib.strategy.nsga2.compute_individual_ranks") as mock_ranks:
            mock_ranks.return_value = {i: 0 for i in range(4)}
            s._run_genetic_algorithm()
        mock_ranks.assert_called_once()
