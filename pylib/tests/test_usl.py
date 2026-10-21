"""The USL fit, tested against curves whose parameters are known.

A fit that returns *a* curve is easy; one that returns the right curve is the
point. Every case here generates throughput from known ``λ, σ, κ`` and
checks the fit gives them back — exactly when the data is exact, within a
tolerance when it is noisy — plus the boundaries where the model changes
character: linear, Amdahl (no peak) and retrograde (a peak and a decline).
"""
import math

import numpy as np
import pytest

from pylib import usl

LEVELS = [1, 2, 4, 6, 8, 10, 12, 14]


def measured(lam, sigma, kappa, levels=LEVELS, repeat=1):
    ns = [n for n in levels for _ in range(repeat)]
    return ns, usl.throughput(ns, lam, sigma, kappa).tolist()


class TestTheModel:
    def test_one_unit_of_concurrency_is_lambda(self):
        assert usl.throughput(1, 3.0, 0.4, 0.01) == pytest.approx(3.0)

    def test_no_contention_and_no_coherency_is_linear(self):
        assert usl.throughput([1, 2, 8], 2.0, 0.0, 0.0).tolist() == [2.0, 4.0, 16.0]

    def test_pure_contention_is_amdahl(self):
        """κ = 0 reduces the USL to Amdahl's law, whose limit is λ/σ."""
        assert usl.throughput(10_000, 1.0, 0.25, 0.0) == pytest.approx(4.0, rel=1e-3)


class TestThePeak:
    def test_it_is_where_the_derivative_vanishes(self):
        lam, sigma, kappa = 1.0, 0.1, 0.01
        n_star = usl.peak(sigma, kappa)

        assert n_star == pytest.approx(math.sqrt(0.9 / 0.01))
        left, at, right = usl.throughput([n_star - 0.5, n_star, n_star + 0.5],
                                         lam, sigma, kappa)
        assert at > left and at > right

    def test_without_coherency_there_is_none(self):
        """Amdahl only approaches its limit; reporting a peak would invent
        one."""
        assert usl.peak(0.3, 0.0) is None

    def test_a_negligible_coherency_is_not_reported_as_a_peak(self):
        assert usl.peak(0.1, 1e-12) is None

    def test_the_integer_peak_is_the_better_neighbour(self):
        """Not necessarily the nearest one: the curve is asymmetric."""
        lam, sigma, kappa = 1.0, 0.05, 0.02
        n_star = usl.peak(sigma, kappa)
        chosen = usl.best_integer_peak(lam, sigma, kappa)

        neighbours = [math.floor(n_star), math.ceil(n_star)]
        assert chosen in neighbours
        assert usl.throughput(chosen, lam, sigma, kappa) == max(
            usl.throughput(neighbours, lam, sigma, kappa))


class TestFittingRecoversKnownParameters:
    @pytest.mark.parametrize("lam,sigma,kappa", [
        (0.5, 0.0, 0.0),        # perfectly linear
        (0.5, 0.2, 0.0),        # Amdahl: contention only
        (0.5, 0.05, 0.004),     # retrograde, peak inside the tested range
        (2.0, 0.3, 0.001),      # strong contention, mild coherency
    ])
    def test_exact_data_gives_the_parameters_back(self, lam, sigma, kappa):
        ns, xs = measured(lam, sigma, kappa)

        fitted = usl.fit(ns, xs)

        assert fitted.status == usl.FIT_OK
        assert fitted.lam == pytest.approx(lam, rel=1e-3)
        assert fitted.sigma == pytest.approx(sigma, abs=2e-3)
        assert fitted.kappa == pytest.approx(kappa, rel=2e-2, abs=1e-6)
        assert fitted.r2 == pytest.approx(1.0, abs=1e-6)

    def test_noisy_data_lands_close(self):
        """3% multiplicative noise, three repetitions per level — the shape of
        a real benchmark."""
        rng = np.random.default_rng(7)
        lam, sigma, kappa = 0.55, 0.08, 0.003
        ns, xs = measured(lam, sigma, kappa, repeat=3)
        noisy = (np.asarray(xs) * rng.normal(1.0, 0.03, len(xs))).tolist()

        fitted = usl.fit(ns, noisy)

        assert fitted.lam == pytest.approx(lam, rel=0.08)
        assert fitted.sigma == pytest.approx(sigma, abs=0.05)
        assert fitted.kappa == pytest.approx(kappa, rel=0.5)
        assert fitted.r2 > 0.95

    def test_a_retrograde_system_reports_its_peak(self):
        lam, sigma, kappa = 0.5, 0.05, 0.004
        ns, xs = measured(lam, sigma, kappa)

        fitted = usl.fit(ns, xs)

        assert fitted.n_peak == pytest.approx(math.sqrt(0.95 / 0.004), rel=2e-2)
        assert fitted.n_peak_int in (15, 16)
        assert fitted.x_peak == pytest.approx(
            float(usl.throughput(fitted.n_peak_int, lam, sigma, kappa)), rel=1e-2)
        assert fitted.x_asymptote is None

    def test_an_amdahl_system_reports_its_ceiling_instead(self):
        ns, xs = measured(1.0, 0.25, 0.0)

        fitted = usl.fit(ns, xs)

        assert fitted.n_peak is None
        assert fitted.x_asymptote == pytest.approx(4.0, rel=1e-2)


class TestTheBounds:
    def test_sigma_never_leaves_the_unit_interval(self):
        """Super-linear data — faster per unit as concurrency grows — would
        pull an unbounded fit to a negative σ, which has no meaning here."""
        ns = [1, 2, 4, 8]
        xs = [1.0, 2.2, 4.8, 10.0]

        fitted = usl.fit(ns, xs)

        assert 0.0 <= fitted.sigma <= 1.0
        assert fitted.kappa >= 0.0

    def test_lambda_is_never_negative(self):
        fitted = usl.fit([1, 2, 3], [0.1, 0.05, 0.01])

        assert fitted.lam >= 0.0


class TestWhatTheDataAllows:
    def test_one_level_cannot_be_fitted(self):
        fitted = usl.fit([4, 4, 4], [1.0, 1.1, 0.9])

        assert fitted.status == usl.FIT_INSUFFICIENT
        assert usl.curve(fitted, 10) == []

    def test_two_levels_fit_contention_only(self):
        """Two abscissae cannot identify three parameters; κ needs curvature,
        so it is the one held at zero — and the status says so."""
        ns, xs = measured(1.0, 0.2, 0.0, levels=[1, 4])

        fitted = usl.fit(ns, xs)

        assert fitted.status == usl.FIT_NO_COHERENCY
        assert fitted.kappa == 0.0
        assert fitted.sigma == pytest.approx(0.2, abs=1e-2)

    def test_failed_measurements_are_dropped_not_fitted(self):
        """A zero is a run that produced nothing, not a system with no
        throughput; fitting it would drag the whole curve down."""
        ns, xs = measured(0.5, 0.1, 0.002)
        ns, xs = ns + [8, 8], xs + [0.0, float("nan")]

        fitted = usl.fit(ns, xs)

        assert fitted.n_points == len(LEVELS)
        assert fitted.sigma == pytest.approx(0.1, abs=5e-3)

    def test_mismatched_inputs_are_refused(self):
        with pytest.raises(ValueError):
            usl.fit([1, 2], [1.0])

    def test_repetitions_count_as_separate_points(self):
        ns, xs = measured(0.5, 0.1, 0.002, repeat=3)

        fitted = usl.fit(ns, xs)

        assert fitted.n_points == 3 * len(LEVELS)
        assert fitted.n_levels == len(LEVELS)


class TestTheDrawnCurve:
    def test_it_covers_every_integer_level_with_the_ideal_line(self):
        fitted = usl.fit(*measured(0.5, 0.1, 0.002))

        rows = usl.curve(fitted, 5)

        assert [r["n"] for r in rows] == [1, 2, 3, 4, 5]
        assert rows[0]["ideal"] == pytest.approx(fitted.lam)
        assert rows[4]["ideal"] == pytest.approx(5 * fitted.lam)
        assert all(r["x"] <= r["ideal"] + 1e-9 for r in rows)

    def test_the_range_extends_past_the_measurements(self):
        fitted = usl.fit(*measured(0.5, 0.1, 0.0))

        assert usl.display_range(LEVELS, fitted) >= 2 * max(LEVELS)

    def test_the_range_shows_the_decline_after_a_peak(self):
        fitted = usl.fit(*measured(0.5, 0.05, 0.004))

        upper = usl.display_range(LEVELS, fitted)

        assert upper > fitted.n_peak, "the peak must be on the chart"
