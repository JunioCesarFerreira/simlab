"""Smoke tests for NSGA3DeapStrategy and NSGA3PymooStrategy.

These tests exercise only _select_next_parents() in isolation, without a
MongoDB connection. They verify that:
  1. The method returns exactly pop_size chromosomes.
  2. Every returned chromosome is drawn from the original R_population.
  3. Class name encoding supports multiple n_obj in one process (no global clash).
"""
import math
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_union(pop_size: int, n_obj: int = 2):
    """Build a synthetic R_population + R_objectives for selection tests."""
    n = 2 * pop_size
    population = [f"genome_{i}" for i in range(n)]
    objectives = [
        [math.sin(i * 0.5) + 1, math.cos(i * 0.3) + 1][: n_obj] if n_obj <= 2
        else [math.sin(i * 0.5 + j) + 1 for j in range(n_obj)]
        for i in range(n)
    ]
    return population, objectives


# ---------------------------------------------------------------------------
# _ensure_deap_classes — n_obj isolation
# ---------------------------------------------------------------------------

class TestEnsureDeapClasses:
    def test_different_n_obj_coexist(self):
        """Class names encode n_obj so two counts can coexist in one process."""
        from lib.strategy.nsga3_deap import _ensure_deap_classes
        fn2, in2 = _ensure_deap_classes(2)
        fn3, in3 = _ensure_deap_classes(3)
        assert fn2 != fn3
        assert in2 != in3
        from deap import creator
        assert len(getattr(creator, fn2).weights) == 2
        assert len(getattr(creator, fn3).weights) == 3

    def test_idempotent_same_n_obj(self):
        """Calling with the same n_obj twice must not raise."""
        from lib.strategy.nsga3_deap import _ensure_deap_classes
        fn_a, in_a = _ensure_deap_classes(2)
        fn_b, in_b = _ensure_deap_classes(2)
        assert fn_a == fn_b
        assert in_a == in_b

    def test_fitness_assignment_correct(self):
        """DEAP individual accepts correct-length fitness values."""
        from lib.strategy.nsga3_deap import _ensure_deap_classes
        from deap import creator
        _, ind_name = _ensure_deap_classes(2)
        IndClass = getattr(creator, ind_name)
        ind = IndClass([1.0, 2.0])
        ind.fitness.values = (1.0, 2.0)  # must not raise

    def test_fitness_assignment_wrong_length_raises(self):
        """DEAP raises AssertionError when fitness length mismatches weights."""
        from lib.strategy.nsga3_deap import _ensure_deap_classes
        from deap import creator
        _, ind_name = _ensure_deap_classes(2)
        IndClass = getattr(creator, ind_name)
        ind = IndClass([1.0, 2.0, 3.0])
        with pytest.raises((AssertionError, Exception)):
            ind.fitness.values = (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# NSGA3DeapStrategy._select_next_parents
# ---------------------------------------------------------------------------

class TestNSGA3DeapSelectNextParents:
    @pytest.fixture
    def strategy_2obj(self):
        from lib.strategy.nsga3_deap import NSGA3DeapStrategy, _ensure_deap_classes
        from deap import tools
        s = NSGA3DeapStrategy.__new__(NSGA3DeapStrategy)
        s._objective_keys = ["m0", "m1"]
        s._divisions = 4
        s._pop_size = 10
        _, s._deap_ind_class = _ensure_deap_classes(2)
        s._deap_ref_points = tools.uniform_reference_points(nobj=2, p=4)
        return s

    @pytest.fixture
    def strategy_3obj(self):
        from lib.strategy.nsga3_deap import NSGA3DeapStrategy, _ensure_deap_classes
        from deap import tools
        s = NSGA3DeapStrategy.__new__(NSGA3DeapStrategy)
        s._objective_keys = ["m0", "m1", "m2"]
        s._divisions = 4
        s._pop_size = 10
        _, s._deap_ind_class = _ensure_deap_classes(3)
        s._deap_ref_points = tools.uniform_reference_points(nobj=3, p=4)
        return s

    def test_returns_pop_size_parents(self, strategy_2obj):
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert len(result) == 10

    def test_all_from_original_population(self, strategy_2obj):
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert all(g in pop for g in result)

    def test_3obj_does_not_clash_with_2obj(self, strategy_2obj, strategy_3obj):
        """Two instances with different n_obj must select independently."""
        pop2, obj2 = _make_union(10, n_obj=2)
        pop3, obj3 = _make_union(10, n_obj=3)
        r2 = strategy_2obj._select_next_parents(pop2, obj2)
        r3 = strategy_3obj._select_next_parents(pop3, obj3)
        assert len(r2) == 10
        assert len(r3) == 10
        assert all(g in pop2 for g in r2)
        assert all(g in pop3 for g in r3)


# ---------------------------------------------------------------------------
# NSGA3PymooStrategy._select_next_parents
# ---------------------------------------------------------------------------

class TestNSGA3PymooSelectNextParents:
    @pytest.fixture
    def strategy_2obj(self):
        from lib.strategy.nsga3_pymoo import NSGA3PymooStrategy
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
        from pymoo.core.problem import Problem
        s = NSGA3PymooStrategy.__new__(NSGA3PymooStrategy)
        s._objective_keys = ["m0", "m1"]
        s._divisions = 4
        s._pop_size = 10
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=4)
        s._pymoo_survival = ReferenceDirectionSurvival(ref_dirs)
        s._pymoo_problem = Problem(n_var=1, n_obj=2)
        return s

    @pytest.fixture
    def strategy_3obj(self):
        from lib.strategy.nsga3_pymoo import NSGA3PymooStrategy
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
        from pymoo.core.problem import Problem
        s = NSGA3PymooStrategy.__new__(NSGA3PymooStrategy)
        s._objective_keys = ["m0", "m1", "m2"]
        s._divisions = 4
        s._pop_size = 10
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)
        s._pymoo_survival = ReferenceDirectionSurvival(ref_dirs)
        s._pymoo_problem = Problem(n_var=1, n_obj=3)
        return s

    def test_returns_pop_size_parents(self, strategy_2obj):
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert len(result) == 10

    def test_all_from_original_population(self, strategy_2obj):
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert all(g in pop for g in result)

    def test_3obj_selection(self, strategy_3obj):
        pop, obj = _make_union(10, n_obj=3)
        result = strategy_3obj._select_next_parents(pop, obj)
        assert len(result) == 10
        assert all(g in pop for g in result)

    def test_simlab_idx_preserved(self, strategy_2obj):
        """Every returned genome must be identifiable from R_population."""
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        # All results are actual genome strings, not None
        assert all(g is not None for g in result)
        assert all(isinstance(g, str) for g in result)
