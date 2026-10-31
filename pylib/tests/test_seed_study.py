"""The seed study: its seed lists, and the statistics behind "how many seeds"."""
import math

import numpy as np
import pytest
from scipy import stats

from pylib.seed_aggregation import AGGREGATOR_DISPATCH, aggregate_seed_values
from pylib.seed_study import (
    DEFAULT_GUARD,
    LOW16,
    MAX_SEEDS,
    SEED_LIMIT,
    aggregate_rows,
    bca_interval,
    boxplot,
    ci_mean_t,
    clopper_pearson,
    describe,
    diagnose_seeds,
    generate_seeds,
    histogram,
    holm,
    median_interval,
    n_star,
    outlier_flags,
    precision_curve,
    qq_points,
    quantile_upper_limit,
    relative_half_width,
    running_interval,
    shapiro_p,
    stein_n,
    study_rng,
    validate_seeds,
)


class TestGenerateSeeds:
    def test_the_same_master_seed_gives_the_same_list(self):
        assert generate_seeds(30, 7) == generate_seeds(30, 7)

    def test_a_shorter_list_is_a_prefix_of_a_longer_one(self):
        """What lets a study be extended, and "the first m seeds" be used."""
        assert generate_seeds(40, 7)[:25] == generate_seeds(25, 7)

    def test_a_different_master_seed_gives_a_different_list(self):
        assert generate_seeds(10, 1) != generate_seeds(10, 2)

    def test_values_stay_in_the_range_cooja_can_add_a_mote_id_to(self):
        seeds = generate_seeds(MAX_SEEDS, 3)

        assert all(1 <= s < SEED_LIMIT for s in seeds)
        assert SEED_LIMIT + LOW16 == 2**31

    def test_the_largest_list_keeps_its_distance_in_the_low_16_bits(self):
        report = diagnose_seeds(generate_seeds(MAX_SEEDS, 3))

        assert report["low16_collisions"] == 0
        assert report["consecutive_pairs"] == 0

    def test_a_list_that_cannot_fit_says_so(self):
        with pytest.raises(ValueError, match="could not place"):
            generate_seeds(100, 1, guard=LOW16 // 8)

    def test_a_count_out_of_range_is_refused(self):
        with pytest.raises(ValueError, match="count"):
            generate_seeds(MAX_SEEDS + 1, 1)


class TestValidateSeeds:
    def test_an_imported_list_keeps_its_order(self):
        assert validate_seeds([9, 3.0, 7]) == [9, 3, 7]

    @pytest.mark.parametrize("values, mentions", [
        ([], "non-empty"),
        ("123", "non-empty"),
        ([1, 2, 2.5], "whole numbers"),
        ([1, 2, True], "whole numbers"),
        ([1, 2, 0], "must be in"),
        ([1, 2, SEED_LIMIT], "must be in"),
        ([1, 2], "between"),
        ([1, 2, 2], "duplicate"),
    ])
    def test_what_is_refused(self, values, mentions):
        with pytest.raises(ValueError, match=mentions):
            validate_seeds(values)


class TestDiagnoseSeeds:
    def test_seeds_with_the_same_low_bits_collide(self):
        report = diagnose_seeds([5, 5 + LOW16, 20000])

        assert report["low16_collisions"] == 1

    def test_consecutive_seeds_are_counted(self):
        """The wizard's 1, 2, 3 preset: three pairs inside the guard."""
        report = diagnose_seeds([1, 2, 3])

        assert report == {"low16_collisions": 0, "consecutive_pairs": 3,
                          "guard": DEFAULT_GUARD}

    def test_the_distance_wraps_around_the_low_16_bits(self):
        assert diagnose_seeds([1, LOW16 - 1, 30000])["consecutive_pairs"] == 1


class TestStudyRng:
    def test_it_is_reproducible(self):
        assert (study_rng([1, 2], "latency").random()
                == study_rng([1, 2], "latency").random())

    def test_labels_keep_streams_apart(self):
        assert (study_rng([1, 2], "latency").random()
                != study_rng([1, 2], "energy").random())


class TestDescribe:
    def test_the_usual_summary(self):
        out = describe(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        assert out["n"] == 5 and out["mean"] == 3 and out["median"] == 3
        assert out["sd"] == pytest.approx(math.sqrt(2.5))
        assert out["cv"] == pytest.approx(math.sqrt(2.5) / 3)
        assert out["iqr"] == 2

    def test_a_single_value_has_no_dispersion_to_report(self):
        out = describe(np.array([7.0]))

        assert out["mean"] == 7 and out["sd"] is None and out["cv"] is None

    def test_no_values_at_all(self):
        assert describe(np.array([]))["mean"] is None

    def test_a_zero_mean_has_no_coefficient_of_variation(self):
        assert describe(np.array([-1.0, 1.0]))["cv"] is None


class TestIntervals:
    def test_the_t_interval_by_hand(self):
        values = np.array([10.0, 12.0, 14.0, 16.0, 18.0])

        low, high = ci_mean_t(values, 0.95)

        half = 2.776445 * math.sqrt(10) / math.sqrt(5)
        assert low == pytest.approx(14 - half, abs=1e-4)
        assert high == pytest.approx(14 + half, abs=1e-4)

    def test_bca_of_constant_data_is_that_constant(self):
        rng = np.random.default_rng(1)

        assert bca_interval(np.full(10, 4.0), "mean", 0.95, rng) == [4.0, 4.0]
        assert bca_interval(np.full(10, 4.0), "sd", 0.95, rng) == [0.0, 0.0]

    def test_bca_contains_the_sample_mean(self):
        rng = np.random.default_rng(1)
        values = rng.lognormal(0, 0.5, 40)

        low, high = bca_interval(values, "mean", 0.95, rng, resamples=2000)

        assert low < values.mean() < high

    def test_the_median_interval_needs_six_values_at_95(self):
        """Six is the first size with 1 - 2 × 0.5⁶ ≥ 0.95."""
        assert median_interval(np.arange(5.0), 0.95) is None
        assert median_interval(np.arange(6.0), 0.95) == [0.0, 5.0]

    @pytest.mark.parametrize("q, enough", [(0.90, 29), (0.95, 59)])
    def test_a_tail_limit_starts_at_the_size_the_maximum_qualifies(self, q, enough):
        assert quantile_upper_limit(np.arange(float(enough - 1)), q, 0.95) is None
        assert quantile_upper_limit(np.arange(float(enough)), q, 0.95) == enough - 1

    def test_no_failures_still_leave_an_upper_limit(self):
        low, high = clopper_pearson(0, 30, 0.95)

        assert low == 0
        assert high == pytest.approx(0.1157, abs=1e-3)

    def test_an_interval_with_no_trials_does_not_exist(self):
        assert clopper_pearson(0, 0, 0.95) is None

    def test_holm_by_hand_and_with_gaps(self):
        assert holm([0.01, None, 0.04, 0.03]) == pytest.approx(
            [0.03, None, 0.06, 0.06])

    def test_shapiro_has_nothing_to_say_about_a_constant(self):
        assert shapiro_p(np.full(10, 2.0)) is None


class TestNStar:
    def test_a_case_worked_by_hand(self):
        """Mean 100, sd 20, ±5%: γ' = 0.05/1.05 and h = 4.7619.

        n = 70: 1.99495 × 20 / √70 = 4.769 > h.
        n = 71: 1.99444 × 20 / √71 = 4.734 ≤ h.
        """
        half_width = relative_half_width(100, 0.05)

        assert half_width == pytest.approx(4.7619, abs=1e-4)
        assert n_star(20, half_width, 0.95) == 71

    def test_the_answer_is_the_smallest_that_works(self):
        for sd, h in [(1.0, 0.5), (3.0, 0.1), (80.0, 10.0)]:
            n = n_star(sd, h, 0.95)
            t = stats.t.ppf(0.975, n - 1)
            assert t * sd / math.sqrt(n) <= h
            if n > 2:
                assert stats.t.ppf(0.975, n - 2) * sd / math.sqrt(n - 1) > h

    def test_no_dispersion_needs_the_minimum(self):
        assert n_star(0.0, 1.0, 0.95) == 2

    def test_a_zero_target_has_no_answer(self):
        assert n_star(1.0, 0.0, 0.95) is None

    def test_stein_never_asks_for_less_than_the_pilot(self):
        assert stein_n(0.1, 10.0, 0.95, 30) == 30

    def test_stein_pays_for_the_small_pilot(self):
        assert stein_n(20, 4.7619, 0.95, 10) > n_star(20, 4.7619, 0.95)


class TestAggregateRows:
    @pytest.mark.parametrize("spec", [
        "mean", "median", "min", "max",
        {"kind": "trimmed_mean", "trim": 0.1},
        {"kind": "trimmed_mean", "trim": 0.25},
        {"kind": "trimmed_mean", "trim": 0.49},
    ])
    def test_it_computes_what_the_engine_computes(self, spec):
        """The study has to evaluate the estimator the optimisation will use."""
        rng = np.random.default_rng(3)
        for size in (1, 2, 3, 4, 7, 10, 30):
            matrix = rng.normal(0, 1, (5, size))
            expected = [aggregate_seed_values(list(row), spec) for row in matrix]
            assert aggregate_rows(matrix, spec) == pytest.approx(expected)

    def test_every_aggregator_is_covered(self):
        assert set(AGGREGATOR_DISPATCH) == {
            "mean", "median", "trimmed_mean", "min", "max"}


class TestPrecisionCurve:
    def test_it_reaches_certainty_when_the_tolerance_is_generous(self):
        values = np.random.default_rng(1).normal(100, 1, 30)

        out = precision_curve(values, "mean", 50.0, 0.95, np.random.default_rng(2),
                              resamples=200)

        assert out["n_emp"] == 2
        assert out["curve"][-1]["k"] == 30

    def test_it_says_when_the_sample_does_not_reach_the_target(self):
        values = np.random.default_rng(1).normal(100, 30, 10)

        out = precision_curve(values, "mean", 0.001, 0.95, np.random.default_rng(2),
                              resamples=200)

        assert out["n_emp"] is None

    def test_the_target_of_a_maximum_moves_with_the_number_of_seeds(self):
        """Why a maximum has no n*: more seeds change what is being estimated."""
        values = np.random.default_rng(1).normal(100, 10, 40)

        curve = precision_curve(values, "max", 1.0, 0.95, np.random.default_rng(2),
                                resamples=500)["curve"]

        assert curve[-1]["expected"] > curve[0]["expected"] + 5


class TestShapeHelpers:
    def test_integer_metrics_get_unit_bins(self):
        bins = histogram(np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0]))

        assert all(b["to"] - b["from"] == pytest.approx(1.0) for b in bins)
        assert sum(b["count"] for b in bins) == 8

    def test_bins_hold_every_value(self):
        values = np.random.default_rng(1).lognormal(0, 1, 200)

        assert sum(b["count"] for b in histogram(values)) == 200

    def test_a_constant_is_one_bin(self):
        assert histogram(np.full(5, 2.0)) == [{"from": 2.0, "to": 2.0, "count": 5}]

    def test_whiskers_stop_at_the_last_value_inside_the_fences(self):
        box = boxplot(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0]))

        assert box["high"] == 5

    def test_an_outlier_is_flagged_by_both_rules(self):
        values = np.array([10.0, 11, 12, 13, 14, 15, 16, 100])

        assert outlier_flags(values) == [None] * 7 + ["both"]

    def test_qq_points_are_sorted_and_symmetric(self):
        points = qq_points(np.array([3.0, 1.0, 2.0]))

        assert [p[1] for p in points] == [1, 2, 3]
        assert points[0][0] == pytest.approx(-points[2][0])

    def test_the_running_mean_ends_at_the_sample_mean(self):
        values = np.array([1.0, 2.0, 6.0])

        points = running_interval(values, 0.95)

        assert [p["n"] for p in points] == [2, 3]
        assert points[-1]["mean"] == 3


class TestCoverage:
    """The method validated against a known truth, before trusting it on data."""

    @pytest.mark.parametrize("sampler", [
        lambda rng, n: rng.normal(100, 15, n),
        lambda rng, n: rng.uniform(50, 150, n),
        lambda rng, n: rng.lognormal(4.6, 0.25, n),
    ], ids=["normal", "uniform", "lognormal"])
    def test_the_t_interval_keeps_its_coverage(self, sampler):
        rng = np.random.default_rng(11)
        truth = sampler(np.random.default_rng(0), 2_000_000).mean()
        hits = 0
        for _ in range(4000):
            low, high = ci_mean_t(sampler(rng, 30), 0.95)
            hits += low <= truth <= high

        # Binomial tolerance for 4000 trials, three standard errors.
        assert hits / 4000 == pytest.approx(0.95, abs=0.012)

    def test_n_star_delivers_the_precision_it_promises(self):
        rng = np.random.default_rng(12)
        n = n_star(15.0, relative_half_width(100.0, 0.05), 0.95)
        means = rng.normal(100, 15, (4000, n)).mean(axis=1)

        within = np.abs(means - 100) / 100 <= 0.05

        assert within.mean() >= 0.94

    def test_bca_keeps_its_coverage_on_skewed_data(self):
        rng = np.random.default_rng(13)
        truth = math.exp(0.5 ** 2 / 2)
        hits = 0
        for _ in range(300):
            low, high = bca_interval(rng.lognormal(0, 0.5, 30), "mean", 0.95, rng,
                                     resamples=999)
            hits += low <= truth <= high

        assert hits / 300 == pytest.approx(0.95, abs=0.06)
