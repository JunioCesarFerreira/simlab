"""Parity guard: the offline true fronts (lib/true_fronts.py) must stay byte-for-byte
identical to the canonical pylib.benchmarks fronts used by the runtime evaluator
and the live quality-indicator endpoint.

pareto-analysis keeps its own standalone copy on purpose (the CLIs must run with
no pylib on the path), so this test — not a shared import — is what prevents the
two from drifting apart (review finding #9).
"""
import numpy as np

from lib.true_fronts import (
    dtlz2_front,
    front_distance,
    sample_true_front,
    sch1_front,
    true_ideal,
    true_nadir,
    zdt1_front,
)
from pylib import benchmarks as bm


def test_dtlz2_front_matches_canonical():
    for m in (2, 3, 5):
        assert np.array_equal(dtlz2_front(m), bm.true_front("DTLZ2", m))


def test_zdt1_front_matches_canonical():
    assert np.array_equal(zdt1_front(), bm.true_front("ZDT1", 2))


def test_sch1_front_matches_canonical():
    assert np.array_equal(sch1_front(), bm.true_front("SCH1", 2))


def test_sample_dispatch_matches_canonical():
    assert np.array_equal(sample_true_front("dtlz2", 3), bm.true_front("DTLZ2", 3))
    assert np.array_equal(sample_true_front("zdt1", 2), bm.true_front("ZDT1", 2))
    assert np.array_equal(sample_true_front("sch1", 2), bm.true_front("SCH1", 2))


def test_true_nadir_matches_canonical():
    assert true_nadir("DTLZ2", 3) == bm.nadir("DTLZ2", 3)
    assert true_nadir("DTLZ2", 5) == bm.nadir("DTLZ2", 5)
    assert true_nadir("ZDT1", 2) == bm.nadir("ZDT1", 2)
    assert true_nadir("SCH1", 2) == bm.nadir("SCH1", 2)


def test_true_ideal_matches_canonical():
    for bench, m in (("DTLZ2", 2), ("DTLZ2", 3), ("DTLZ2", 6), ("ZDT1", 2), ("SCH1", 2)):
        assert true_ideal(bench, m) == bm.ideal(bench, m)


def test_front_distance_matches_canonical():
    rng = np.random.default_rng(4)
    for bench, m in (("DTLZ2", 2), ("DTLZ2", 3), ("DTLZ2", 6), ("ZDT1", 2), ("SCH1", 2)):
        points = rng.random((40, m)) * 3.0
        assert np.array_equal(
            front_distance(bench, points, m), bm.front_distance(bench, points, m)
        )
