"""Unit tests for the P0 pure-synthetic problem adapter (lib/problem/p0_synthetic.py).

P0 replaces the WSN P1 base for algorithm-validation runs: the genome is a flat
real vector x ∈ [0,1]^n driven by textbook SBX + polynomial mutation, with no
relays/sink/MAC and no connectivity/coverage repair.
"""
import random

import pytest

from lib.problem.resolve import build_adapter, build_test_adapter
from lib.problem.chromosomes import ChromosomeP0, chromosome_from_dict


def _adapter(n: int = 6, seed: int = 0):
    return build_adapter(
        {"name": "problem0", "n": n},
        {"eta_cx": 20.0, "eta_mt": 20.0, "per_gene_prob": 1.0 / n},
        random.Random(seed),
    )


class TestResolveP0:
    def test_resolves_problem0(self):
        ad = build_test_adapter({"name": "problem0", "n": 4})
        assert ad.__class__.__name__ == "Problem0SyntheticAdapter"

    def test_requires_n(self):
        with pytest.raises(KeyError):
            build_test_adapter({"name": "problem0"})

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError):
            build_test_adapter({"name": "problem0", "n": 0})


class TestInit:
    def test_dimension_and_bounds(self):
        pop = _adapter(n=8).random_individual_generator(20)
        assert all(len(c.x) == 8 for c in pop)
        assert all(0.0 <= v <= 1.0 for c in pop for v in c.x)

    def test_uniform_spread_not_center_biased(self):
        # The whole point of P0: uniform coverage of [0,1], unlike P1's radial
        # init that starved the extremes and collapsed benchmark fronts.
        xs = [c.x[0] for c in _adapter(n=1, seed=1).random_individual_generator(400)]
        lo = sum(1 for v in xs if v < 0.25) / len(xs)
        hi = sum(1 for v in xs if v > 0.75) / len(xs)
        assert lo > 0.15 and hi > 0.15

    def test_determinism(self):
        a = [c.x for c in _adapter(seed=7).random_individual_generator(10)]
        b = [c.x for c in _adapter(seed=7).random_individual_generator(10)]
        assert a == b


class TestOperators:
    def test_crossover_two_children_in_bounds(self):
        ad = _adapter(n=6)
        p1, p2 = ad.random_individual_generator(2)
        c1, c2 = ad.crossover([p1, p2])
        assert len(c1.x) == 6 and len(c2.x) == 6
        assert all(0.0 <= v <= 1.0 for v in c1.x + c2.x)

    def test_mutate_in_bounds(self):
        ad = _adapter(n=6)
        m = ad.mutate(ad.random_individual_generator(1)[0])
        assert len(m.x) == 6
        assert all(0.0 <= v <= 1.0 for v in m.x)


class TestEncode:
    def test_decision_vector_no_motes(self):
        ad = _adapter(n=5)
        ind = ad.random_individual_generator(1)[0]
        enc = ad.encode_simulation_input(ind)
        assert enc["fixedMotes"] == [] and enc["mobileMotes"] == []
        assert enc["decisionVector"] == list(ind.x)

    def test_structural_properties(self):
        ad = _adapter()
        assert ad.bounds == [0.0, 0.0, 1.0, 1.0]
        assert ad.radius_of_reach == 1.0 and ad.radius_of_inter == 1.0


class TestChromosome:
    def test_round_trip(self):
        c = ChromosomeP0(x=[0.1, 0.2, 0.3])
        r = chromosome_from_dict("problem0", c.to_dict())
        assert r == c and r.get_hash() == c.get_hash()

    def test_quantized_equality(self):
        a = ChromosomeP0(x=[0.1, 0.2, 0.3])
        b = ChromosomeP0(x=[0.1, 0.2, 0.3 + 1e-9])
        assert a == b and hash(a) == hash(b) and a.get_hash() == b.get_hash()
