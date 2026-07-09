"""Parity guard: the offline true fronts (lib/true_fronts.py) must stay byte-for-byte
identical to the canonical pylib.benchmarks fronts used by the runtime evaluator
and the live quality-indicator endpoint.

pareto-analysis keeps its own standalone copy on purpose (the CLIs must run with
no pylib on the path), so this test — not a shared import — is what prevents the
two from drifting apart (review finding #9).
"""
import numpy as np

from lib.true_fronts import dtlz2_front, zdt1_front, sch1_front, sample_true_front
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
