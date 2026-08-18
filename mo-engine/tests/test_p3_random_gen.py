"""
Regression test for P3's initial-population uniqueness.

`random_individual_generator` rejection-samples masks until one is feasible
(k-coverage + connectivity). When the feasible subspace is small, independent
draws can repeatedly land on the same mask, producing duplicate chromosomes.
NSGA-II/III and random-search de-duplicate genomes within a generation (see
`_generation_enqueue`'s `seen_generation_hashes`), so undetected duplicates
here silently shrink generation 0 below `population_size`.
"""
import random

from lib.problem.resolve import build_test_adapter


def _grid_candidates():
    return [
        (float(x), float(y))
        for x in range(-100, 101, 40)
        for y in range(-100, 101, 40)
        if (x, y) != (0, 0)
    ]


def _p3_problem():
    return {
        "name": "problem3",
        "region": [-150.0, -150.0, 150.0, 150.0],
        "sink": (0.0, 0.0),
        "candidates": _grid_candidates(),
        "targets": [(20.0, 20.0), (-40.0, 10.0), (60.0, -30.0)],
        "k_required": 1,
        "radius_of_reach": 45.0,
        "radius_of_inter": 55.0,
        "radius_of_cover": 45.0,
    }


def test_p3_random_individuals_are_unique_when_search_space_allows():
    adapter = build_test_adapter(_p3_problem())
    adapter.set_ga_operator_configs(random.Random(3), {"per_gene_prob": 0.1})

    pop = adapter.random_individual_generator(20)
    hashes = [chrm.get_hash() for chrm in pop]

    assert len(pop) == 20
    assert len(set(hashes)) == 20


def test_p3_random_individual_generator_falls_back_gracefully_on_tiny_search_space():
    # Degenerate instance: only two masks in the whole reachable space satisfy
    # k-coverage + connectivity. The generator must still return exactly
    # `size` individuals (accepting duplicates with a warning) instead of
    # raising or returning fewer than requested.
    problem = {
        "name": "problem3",
        "region": [-10.0, -10.0, 100.0, 100.0],
        "sink": (0.0, 0.0),
        "candidates": [(10.0 * i, 0.0) for i in range(1, 6)],
        "targets": [(25.0, 5.0)],
        "k_required": 2,
        "radius_of_reach": 12.0,
        "radius_of_inter": 15.0,
        "radius_of_cover": 20.0,
    }
    adapter = build_test_adapter(problem)
    adapter.set_ga_operator_configs(random.Random(11), {"per_gene_prob": 0.1})

    pop = adapter.random_individual_generator(20)

    assert len(pop) == 20
