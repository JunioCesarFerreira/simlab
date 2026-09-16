"""Phase 0 guard rails: the kernel is reproducible and its numbers are frozen.

Two distinct guarantees:

  * ``test_kernel_is_bit_for_bit_reproducible`` — a run is a pure function of
    (config, seed). Without this, every later phase's diff is noise.
  * ``test_matches_frozen_baseline`` — the frozen numbers still hold. This test
    is EXPECTED to fail while a fix-plan phase is applied; the workflow is to
    read the diff, confirm it is the intended change, then re-freeze with
    ``python -m tests.regression.baseline --write`` and bump ``BASELINE_STAGE``.
"""
from __future__ import annotations

import pytest

from tests.regression import baseline
from tests.regression.kernel import BASELINE_CONFIGS, KernelConfig, run_kernel


def test_kernel_is_bit_for_bit_reproducible():
    config = KernelConfig(
        algorithm="nsga3", bench="DTLZ2", m=3, n=10, divisions=4,
        pop_size=20, generations=5
    )
    assert run_kernel(config, 1) == run_kernel(config, 1)


def test_distinct_seeds_produce_distinct_runs():
    """Guards against a harness that is reproducible because it is constant."""
    config = KernelConfig(
        algorithm="nsga3", bench="DTLZ2", m=3, n=10, divisions=4,
        pop_size=20, generations=5
    )
    assert run_kernel(config, 1) != run_kernel(config, 2)


def test_matches_frozen_baseline():
    expected = baseline.load()
    actual = baseline.build()
    differences = baseline.compare(actual, expected)
    assert not differences, (
        f"{len(differences)} difference(s) against the {expected['stage']} baseline. "
        "If this is an intended fix-plan change, re-freeze with "
        "`python -m tests.regression.baseline --write`.\n"
        + "\n".join(differences[:25])
    )


@pytest.mark.parametrize(
    "config", [c for c in BASELINE_CONFIGS if c.bench == "DTLZ2"], ids=lambda c: c.label
)
def test_survivor_series_is_smoother_than_offspring_series(config: KernelConfig):
    """The audit's finding 1, as a permanent invariant.

    Environmental selection is elitist over the union, so the surviving
    population's front moves less between steps than the raw offspring front
    does. The ``/hv-gd`` endpoint currently plots the offspring series, which is
    why its curves look like they regress. Phase 1 makes survivors the default;
    this test pins the reason.
    """
    offspring_drop = 0.0
    survivor_drop = 0.0
    for seed in (1, 2, 3):
        history = run_kernel(config, seed)
        for previous, current in zip(history, history[1:]):
            offspring_drop = max(
                offspring_drop, previous["offspring"]["hv"] - current["offspring"]["hv"]
            )
            survivor_drop = max(
                survivor_drop, previous["survivors"]["hv"] - current["survivors"]["hv"]
            )
    assert survivor_drop < offspring_drop
