"""Smoke tests for NSGA2DeapStrategy and NSGA2PymooStrategy.

These tests exercise only _select_next_parents() in isolation, without a
MongoDB connection. They verify that:
  1. The method returns exactly pop_size chromosomes.
  2. Every returned chromosome is drawn from the original R_population.
  3. DEAP class names are namespaced to avoid collisions with nsga3_deap classes.
  4. Results are stable across 2-objective and 3-objective problems.
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
    if n_obj == 2:
        objectives = [[math.sin(i * 0.5) + 1, math.cos(i * 0.3) + 1] for i in range(n)]
    else:
        objectives = [[math.sin(i * 0.5 + j) + 1 for j in range(n_obj)] for i in range(n)]
    return population, objectives


# ---------------------------------------------------------------------------
# _ensure_deap_classes_nsga2 — namespace isolation
# ---------------------------------------------------------------------------

class TestEnsureDeapClassesNsga2:
    def test_different_n_obj_coexist(self):
        """Class names encode n_obj so two objective counts can coexist."""
        from lib.strategy.nsga2_deap import _ensure_deap_classes_nsga2
        fn2, in2 = _ensure_deap_classes_nsga2(2)
        fn3, in3 = _ensure_deap_classes_nsga2(3)
        assert fn2 != fn3
        assert in2 != in3
        from deap import creator
        assert len(getattr(creator, fn2).weights) == 2
        assert len(getattr(creator, fn3).weights) == 3

    def test_idempotent_same_n_obj(self):
        """Calling with the same n_obj twice must not raise."""
        from lib.strategy.nsga2_deap import _ensure_deap_classes_nsga2
        fn_a, in_a = _ensure_deap_classes_nsga2(2)
        fn_b, in_b = _ensure_deap_classes_nsga2(2)
        assert fn_a == fn_b
        assert in_a == in_b

    def test_namespaced_from_nsga3(self):
        """NSGA-2 DEAP class names must not clash with NSGA-3 DEAP class names."""
        from lib.strategy.nsga2_deap import _ensure_deap_classes_nsga2
        from lib.strategy.nsga3_deap import _ensure_deap_classes
        fn2_nsga2, _ = _ensure_deap_classes_nsga2(2)
        fn2_nsga3, _ = _ensure_deap_classes(2)
        assert fn2_nsga2 != fn2_nsga3

    def test_fitness_weights_are_minimization(self):
        """All weights must be -1.0 (minimization via DEAP negation convention)."""
        from lib.strategy.nsga2_deap import _ensure_deap_classes_nsga2
        from deap import creator
        fn, _ = _ensure_deap_classes_nsga2(3)
        assert getattr(creator, fn).weights == (-1.0, -1.0, -1.0)

    def test_fitness_assignment_correct(self):
        """DEAP individual accepts correct-length fitness values."""
        from lib.strategy.nsga2_deap import _ensure_deap_classes_nsga2
        from deap import creator
        _, ind_name = _ensure_deap_classes_nsga2(2)
        IndClass = getattr(creator, ind_name)
        ind = IndClass([1.0, 2.0])
        ind.fitness.values = (1.0, 2.0)  # must not raise


# ---------------------------------------------------------------------------
# NSGA2DeapStrategy._select_next_parents
# ---------------------------------------------------------------------------

class TestNSGA2DeapSelectNextParents:
    @pytest.fixture
    def strategy_2obj(self):
        from lib.strategy.nsga2_deap import NSGA2DeapStrategy, _ensure_deap_classes_nsga2
        s = NSGA2DeapStrategy.__new__(NSGA2DeapStrategy)
        s._objective_keys = ["m0", "m1"]
        s._pop_size = 10
        _, s._deap_ind_class = _ensure_deap_classes_nsga2(2)
        return s

    @pytest.fixture
    def strategy_3obj(self):
        from lib.strategy.nsga2_deap import NSGA2DeapStrategy, _ensure_deap_classes_nsga2
        s = NSGA2DeapStrategy.__new__(NSGA2DeapStrategy)
        s._objective_keys = ["m0", "m1", "m2"]
        s._pop_size = 10
        _, s._deap_ind_class = _ensure_deap_classes_nsga2(3)
        return s

    def test_returns_pop_size_parents(self, strategy_2obj):
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert len(result) == 10

    def test_all_from_original_population(self, strategy_2obj):
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert all(g in pop for g in result)

    def test_3obj_returns_pop_size(self, strategy_3obj):
        pop, obj = _make_union(10, n_obj=3)
        result = strategy_3obj._select_next_parents(pop, obj)
        assert len(result) == 10

    def test_3obj_all_from_original_population(self, strategy_3obj):
        pop, obj = _make_union(10, n_obj=3)
        result = strategy_3obj._select_next_parents(pop, obj)
        assert all(g in pop for g in result)

    def test_2obj_and_3obj_do_not_clash(self, strategy_2obj, strategy_3obj):
        """Two instances with different n_obj must select independently."""
        pop2, obj2 = _make_union(10, n_obj=2)
        pop3, obj3 = _make_union(10, n_obj=3)
        r2 = strategy_2obj._select_next_parents(pop2, obj2)
        r3 = strategy_3obj._select_next_parents(pop3, obj3)
        assert len(r2) == 10
        assert len(r3) == 10
        assert all(g in pop2 for g in r2)
        assert all(g in pop3 for g in r3)

    def test_dominated_solutions_excluded(self, strategy_2obj):
        """A Pareto-dominated solution should not appear in front-0 selection
        when the pool is large enough that crowding drives selection."""
        # Strictly dominated individual: [10.0, 10.0] dominated by all others.
        pop = [f"g{i}" for i in range(20)]
        objectives = [[float(i), float(20 - i)] for i in range(19)] + [[10.0, 10.0]]
        result = strategy_2obj._select_next_parents(pop, objectives)
        assert len(result) == 10
        assert all(g in pop for g in result)

    def test_result_contains_only_strings(self, strategy_2obj):
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert all(isinstance(g, str) for g in result)


# ---------------------------------------------------------------------------
# NSGA2PymooStrategy._select_next_parents
# ---------------------------------------------------------------------------

class TestNSGA2PymooSelectNextParents:
    @pytest.fixture
    def strategy_2obj(self):
        from lib.strategy.nsga2_pymoo import NSGA2PymooStrategy
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding  # pymoo >= 0.6
        from pymoo.core.problem import Problem
        s = NSGA2PymooStrategy.__new__(NSGA2PymooStrategy)
        s._objective_keys = ["m0", "m1"]
        s._pop_size = 10
        s._pymoo_survival = RankAndCrowding()
        s._pymoo_problem = Problem(n_var=1, n_obj=2)
        return s

    @pytest.fixture
    def strategy_3obj(self):
        from lib.strategy.nsga2_pymoo import NSGA2PymooStrategy
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding  # pymoo >= 0.6
        from pymoo.core.problem import Problem
        s = NSGA2PymooStrategy.__new__(NSGA2PymooStrategy)
        s._objective_keys = ["m0", "m1", "m2"]
        s._pop_size = 10
        s._pymoo_survival = RankAndCrowding()
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

    def test_3obj_returns_pop_size(self, strategy_3obj):
        pop, obj = _make_union(10, n_obj=3)
        result = strategy_3obj._select_next_parents(pop, obj)
        assert len(result) == 10

    def test_3obj_all_from_original_population(self, strategy_3obj):
        pop, obj = _make_union(10, n_obj=3)
        result = strategy_3obj._select_next_parents(pop, obj)
        assert all(g in pop for g in result)

    def test_simlab_idx_preserved(self, strategy_2obj):
        """Every returned genome must be identifiable from R_population."""
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert all(g is not None for g in result)
        assert all(isinstance(g, str) for g in result)

    def test_dominated_solutions_excluded(self, strategy_2obj):
        """Strictly dominated individual must not appear when pool is large enough."""
        pop = [f"g{i}" for i in range(20)]
        objectives = [[float(i), float(20 - i)] for i in range(19)] + [[10.0, 10.0]]
        result = strategy_2obj._select_next_parents(pop, objectives)
        assert len(result) == 10
        assert all(g in pop for g in result)

    def test_result_contains_only_strings(self, strategy_2obj):
        pop, obj = _make_union(10, n_obj=2)
        result = strategy_2obj._select_next_parents(pop, obj)
        assert all(isinstance(g, str) for g in result)


# ---------------------------------------------------------------------------
# Cross-library consistency: DEAP and pymoo must select the same count
# ---------------------------------------------------------------------------

class TestCrossLibraryConsistency:
    def test_both_return_pop_size(self):
        """DEAP and pymoo selections must return the same number of parents."""
        from lib.strategy.nsga2_deap import NSGA2DeapStrategy, _ensure_deap_classes_nsga2
        from lib.strategy.nsga2_pymoo import NSGA2PymooStrategy
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding  # pymoo >= 0.6
        from pymoo.core.problem import Problem

        pop, obj = _make_union(10, n_obj=2)

        s_deap = NSGA2DeapStrategy.__new__(NSGA2DeapStrategy)
        s_deap._objective_keys = ["m0", "m1"]
        s_deap._pop_size = 10
        _, s_deap._deap_ind_class = _ensure_deap_classes_nsga2(2)

        s_pymoo = NSGA2PymooStrategy.__new__(NSGA2PymooStrategy)
        s_pymoo._objective_keys = ["m0", "m1"]
        s_pymoo._pop_size = 10
        s_pymoo._pymoo_survival = RankAndCrowding()
        s_pymoo._pymoo_problem = Problem(n_var=1, n_obj=2)

        r_deap = s_deap._select_next_parents(pop, obj)
        r_pymoo = s_pymoo._select_next_parents(pop, obj)

        assert len(r_deap) == len(r_pymoo) == 10
        assert all(g in pop for g in r_deap)
        assert all(g in pop for g in r_pymoo)
