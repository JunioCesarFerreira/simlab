"""Finding 2 (second half) of the NSGA metrics audit: the mating tournament
broke rank ties at random, dropping NSGA-II's crowded-comparison operator.

Rank alone ties every member of a front with every other, and the first front is
exactly where mating pressure matters. Without the crowding tie-break the
tournament applies no pressure at all towards the sparse regions of the front.
"""
from __future__ import annotations

import random
from math import inf

from lib.genetic_operators.selection import (
    compute_crowding_distances,
    compute_individual_ranks,
    tournament_selection,
)
from lib.nsga import fast_nondominated_sort


class _ScriptedRng(random.Random):
    """Forces the tournament to draw one chosen pair, then a scripted coin flip."""

    def __init__(self, pair, coin=0):
        super().__init__(0)
        self._pair = pair
        self._coin = coin

    def sample(self, population, k):
        assert k == 2
        return list(self._pair)

    def choice(self, seq):
        return seq[self._coin]


def test_lower_rank_wins_regardless_of_crowding():
    population = ["a", "b"]
    ranks = {0: 0, 1: 1}
    crowding = {0: 0.0, 1: inf}     # b is far less crowded but ranks worse
    assert tournament_selection(population, ranks, _ScriptedRng((0, 1)), crowding) == "a"
    assert tournament_selection(population, ranks, _ScriptedRng((1, 0)), crowding) == "a"


def test_rank_tie_goes_to_the_less_crowded_individual():
    population = ["crowded", "sparse"]
    ranks = {0: 0, 1: 0}
    crowding = {0: 0.2, 1: 1.5}
    # Both draw orders, and a coin flip that would have picked the crowded one.
    assert tournament_selection(population, ranks, _ScriptedRng((0, 1), coin=0), crowding) == "sparse"
    assert tournament_selection(population, ranks, _ScriptedRng((1, 0), coin=0), crowding) == "sparse"


def test_boundary_individuals_win_their_ties():
    """Crowding distance is +inf on the front's boundary solutions, which is how
    NSGA-II keeps the extremes alive through mating."""
    population = ["interior", "boundary"]
    ranks = {0: 0, 1: 0}
    crowding = {0: 3.0, 1: inf}
    assert tournament_selection(population, ranks, _ScriptedRng((0, 1)), crowding) == "boundary"


def test_equal_crowding_falls_back_to_the_coin_flip():
    population = ["a", "b"]
    ranks = {0: 0, 1: 0}
    crowding = {0: 1.0, 1: 1.0}
    assert tournament_selection(population, ranks, _ScriptedRng((0, 1), coin=0), crowding) == "a"
    assert tournament_selection(population, ranks, _ScriptedRng((0, 1), coin=1), crowding) == "b"


def test_omitting_crowding_keeps_the_random_tie_break():
    """NSGA-III passes nothing: Deb & Jain mate without a crowded tournament and
    enforce diversity through reference-point niching instead."""
    population = ["a", "b"]
    ranks = {0: 0, 1: 0}
    assert tournament_selection(population, ranks, _ScriptedRng((0, 1), coin=0)) == "a"
    assert tournament_selection(population, ranks, _ScriptedRng((0, 1), coin=1)) == "b"


def test_crowding_distances_are_keyed_by_population_index():
    objectives = [[0.0, 3.0], [1.0, 1.0], [3.0, 0.0], [2.0, 2.0], [5.0, 5.0]]
    fronts = fast_nondominated_sort(objectives)
    ranks = compute_individual_ranks(fronts)
    distances = compute_crowding_distances(fronts, objectives)

    assert set(distances) == set(range(len(objectives)))
    # Front 0 is {0, 1, 2}: the two extremes are boundaries, index 1 interior.
    assert ranks[0] == ranks[1] == ranks[2] == 0
    assert distances[0] == inf and distances[2] == inf
    assert distances[1] < inf
