"""The seed study: its seed lists, and the statistics behind "how many seeds"."""
import math

import numpy as np
import pytest
from scipy import stats

from pylib.seed_aggregation import AGGREGATOR_DISPATCH, aggregate_seed_values
from pylib.seed_study import (
    _estimates_diverge,
    DEFAULT_GUARD,
    LOW16,
    MAX_SEEDS,
    SEED_LIMIT,
    aggregate_rows,
    analyse_panel,
    analyse_study,
    balanced_matrix,
    bca_interval,
    boxplot,
    ci_mean_t,
    clopper_pearson,
    compare_control,
    describe,
    diagnose_seeds,
    failure_kind,
    failure_summary,
    first_reliable,
    front_stability,
    generate_seeds,
    group_aliases,
    histogram,
    holm,
    m_star,
    median_interval,
    metric_detail,
    metric_status,
    n_star,
    outlier_flags,
    pigeonhole_interval,
    precision_curve,
    qq_points,
    quantile_upper_limit,
    rank_stability,
    relative_half_width,
    reliability_at,
    running_interval,
    shapiro_p,
    stein_n,
    study_rng,
    validate_seeds,
    variance_components,
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


def crossed(rng, K, n, individual=1.0, seed=0.0, residual=1.0):
    """Data from the model of the study, with known components."""
    return (10.0
            + rng.normal(0, math.sqrt(individual), (K, 1))
            + rng.normal(0, math.sqrt(seed), (1, n))
            + rng.normal(0, math.sqrt(residual), (K, n)))


class TestBalancedMatrix:
    def test_a_complete_matrix_is_left_alone(self):
        values = np.arange(12.0).reshape(3, 4)

        block, rows, columns = balanced_matrix(values)

        assert block.shape == (3, 4) and rows == [0, 1, 2] and columns == [0, 1, 2, 3]

    def test_a_single_hole_costs_one_seed(self):
        values = np.arange(12.0).reshape(3, 4)
        values[1, 2] = np.nan

        block, rows, columns = balanced_matrix(values)

        assert block.shape == (3, 3) and columns == [0, 1, 3]

    def test_a_chromosome_that_failed_everywhere_costs_only_itself(self):
        values = np.arange(20.0).reshape(4, 5)
        values[2, :4] = np.nan

        block, rows, columns = balanced_matrix(values)

        assert rows == [0, 1, 3] and columns == [0, 1, 2, 3, 4]


class TestVarianceComponents:
    def test_a_table_worked_by_hand(self):
        """Grand mean 5, chromosome means 3 and 7, seed means 4 and 6.

        SS chromosome = 2·(4 + 4) = 16 and SS seed = 2·(1 + 1) = 4, out of a
        total of 20: an additive table leaves no residual.
        """
        values = np.array([[2.0, 4.0], [6.0, 8.0]])

        out = variance_components(values)

        assert out["residual"] == pytest.approx(0.0)
        assert out["individual"] == pytest.approx(16 / 2)
        assert out["seed"] == pytest.approx(4 / 2)
        assert out["shares"]["individual"] == pytest.approx(0.8)

    def test_the_model_recovers_what_was_simulated(self):
        rng = np.random.default_rng(21)
        runs = [variance_components(crossed(rng, 40, 40, 4.0, 0.25, 9.0))
                for _ in range(200)]

        assert np.mean([r["individual"] for r in runs]) == pytest.approx(4.0, rel=0.1)
        assert np.mean([r["seed"] for r in runs]) == pytest.approx(0.25, rel=0.2)
        assert np.mean([r["residual"] for r in runs]) == pytest.approx(9.0, rel=0.02)

    def test_a_negative_estimate_is_zero(self):
        rng = np.random.default_rng(1)
        out = variance_components(rng.normal(0, 1, (2, 200)))

        assert out["individual"] >= 0 and out["seed"] >= 0


class TestReliability:
    def test_the_projection_by_hand(self):
        components = {"individual": 4.0, "seed": 1.0, "residual": 2.0}

        e_rho2, phi = reliability_at(components, 2)

        assert e_rho2 == pytest.approx(4 / 5)
        assert phi == pytest.approx(4 / 5.5)

    def test_m_star_is_the_smallest_that_reaches_the_target(self):
        components = {"individual": 1.0, "seed": 0.0, "residual": 1.0}

        needed = m_star(components, 0.9)

        assert needed == 9
        assert reliability_at(components, 9)[0] == pytest.approx(0.9)
        assert reliability_at(components, 8)[0] < 0.9

    def test_no_signal_has_no_answer(self):
        components = {"individual": 0.0, "seed": 0.0, "residual": 1.0}

        assert m_star(components, 0.9) is None
        assert reliability_at(components, 3) == (None, None)

    def test_m_star_reproduces_the_proposal_from_its_shares(self):
        """Eρ²(1) of 0.64 means a ratio of 0.5625, so 0.90 takes 6 seeds."""
        components = {"individual": 0.64, "seed": 0.0, "residual": 0.36}

        assert m_star(components, 0.90) == 6


class TestPigeonhole:
    def test_the_interval_contains_the_truth_most_of_the_time(self):
        rng = np.random.default_rng(31)
        truth = 9
        hits = 0
        for _ in range(40):
            low, high = pigeonhole_interval(crossed(rng, 30, 30), 0.9, 0.95, rng,
                                            resamples=200)
            hits += low <= truth and (high is None or truth <= high)

        assert hits >= 34


class TestRankStability:
    def test_rankings_agree_more_with_more_seeds(self):
        rng = np.random.default_rng(41)
        values = crossed(rng, 30, 40)

        rows = rank_stability(values, "mean", rng)

        assert [r["k"] for r in rows] == [1, 2, 3, 5, 8, 13, 20]
        assert rows[-1]["spearman"] > rows[0]["spearman"] + 0.3

    def test_it_agrees_with_the_model(self):
        """Spearman between sets of k seeds estimates Eρ²(k).

        Averaged over panels: one panel's spread is itself a draw, and it moves
        both numbers together.
        """
        rng = np.random.default_rng(42)
        fifth = [next(r["spearman"] for r in rank_stability(crossed(rng, 60, 40), "mean", rng)
                      if r["k"] == 5)
                 for _ in range(30)]

        assert np.mean(fifth) == pytest.approx(5 / 6, abs=0.03)

    def test_first_reliable(self):
        rows = [{"k": 1, "spearman": 0.5}, {"k": 2, "spearman": 0.95}]

        assert first_reliable(rows, 0.9) == 2
        assert first_reliable(rows, 0.99) is None


class TestFrontStability:
    def test_a_noiseless_panel_has_one_front(self):
        rng = np.random.default_rng(51)
        first = np.array([[1.0], [2.0], [3.0]]) * np.ones((3, 6))
        second = np.array([[3.0], [2.0], [1.0]]) * np.ones((3, 6))

        rows = front_stability(np.stack([first, second]), ["mean", "mean"], [1, 1], rng,
                               draws=5)

        assert all(r["jaccard"] == 1 for r in rows)

    def test_the_goal_decides_which_side_is_better(self):
        rng = np.random.default_rng(52)
        layer = np.array([[1.0], [2.0]]) * np.ones((2, 4))

        rows = front_stability(np.stack([layer, layer]), ["mean", "mean"], [-1, 1], rng,
                               draws=3)

        assert rows[0]["jaccard"] == 1


class TestAnalysePanel:
    def _run(self, values, spec="mean", with_interval=False):
        return analyse_panel(values, reliability=0.9, confidence=0.95, spec=spec,
                             rng=np.random.default_rng(1), with_interval=with_interval)

    def test_it_answers_with_the_model_for_the_mean(self):
        out = self._run(crossed(np.random.default_rng(61), 40, 40), with_interval=True)

        assert out["m_star"]["method"] == "generalizability"
        assert out["m_star"]["value"] == pytest.approx(9, abs=4)
        assert out["signal"] is True
        assert out["reliability"][0]["m"] == 1

    def test_another_aggregator_is_answered_by_the_ranking(self):
        out = self._run(crossed(np.random.default_rng(62), 40, 40), spec="median")

        assert out["m_star"]["method"] == "rank_stability"
        assert out["m_star"]["ci"] is None

    def test_holes_are_reported(self):
        values = crossed(np.random.default_rng(63), 12, 12)
        values[3, 4] = np.nan

        out = self._run(values)

        assert out["used"]["dropped_seeds"] == [4] or out["used"]["dropped_chromosomes"] == [3]

    def test_a_panel_with_no_signal_has_no_m_star(self):
        values = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

        out = self._run(values)

        assert out["signal"] is False and out["m_star"]["value"] is None

    def test_one_chromosome_is_not_a_panel(self):
        assert self._run(np.ones((1, 5))) is None


SEEDS = list(range(100, 130))
OBJECTIVES = [{"metric_name": "latency", "goal": "min", "aggregator": "mean"},
              {"metric_name": "throughput", "goal": "max", "aggregator": "mean"}]


def study(K=1, n=30, **over):
    rng = np.random.default_rng(71)
    values = {
        "latency": 200 + 20 * rng.normal(0, 1, (K, 1)) + 30 * rng.normal(0, 1, (K, n)),
        "throughput": 3000 + 100 * rng.normal(0, 1, (K, 1)) + 50 * rng.normal(0, 1, (K, n)),
    }
    values.update(over)
    return values


def analysed(values, K=1, n=30, objectives=OBJECTIVES, **kwargs):
    return analyse_study(values, seeds=list(range(100, 100 + n)),
                         chromosomes=[f"c{i}" for i in range(K)],
                         objectives=objectives, **kwargs)


def named(result, name):
    return next(m for m in result["metrics"] if m["name"] == name)


class TestMetricStatus:
    def test_a_metric_that_varies_is_usable(self):
        assert metric_status(np.array([[1.0, 2.0, 3.0]])) == "ok"

    def test_a_constant_metric(self):
        assert metric_status(np.zeros((2, 5))) == "constant"

    def test_constant_means_with_each_solution_not_between_them(self):
        """Different per chromosome and equal per seed: the seed changes nothing."""
        assert metric_status(np.array([[1.0, 1.0], [2.0, 2.0]])) == "constant"

    def test_a_wrapped_subtraction_is_not_a_measurement(self):
        assert metric_status(np.array([[1.0, 1.8e19, 3.0]])) == "overflow_suspect"

    def test_nothing_measured_yet(self):
        assert metric_status(np.full((2, 3), np.nan)) == "no_data"

    def test_one_seed_each_is_not_a_constant(self):
        """A study that has just started must not read as "the seed changes nothing"."""
        assert metric_status(np.array([[1.0, np.nan], [np.nan, 2.0]])) == "no_data"


class TestAliases:
    def test_copies_go_under_the_first_name(self):
        same = np.array([[1.0, 2.0, 3.0]])
        groups = group_aliases({"a": same, "b": same.copy(), "c": same + 1}, ["a", "b", "c"])

        assert groups == {"a": ["b"], "c": []}

    def test_metrics_that_are_always_zero_are_not_copies(self):
        groups = group_aliases({"a": np.zeros((1, 3)), "b": np.zeros((1, 3))}, ["a", "b"])

        assert groups == {}


class TestAnalyseStudy:
    def test_one_chromosome_answers_precision_and_not_ranking(self):
        result = analysed(study())

        report = result["recommendation"]["report"]["by_objective"]
        optimise = result["recommendation"]["optimise"]
        assert report["latency"]["value"] > 1
        assert optimise["by_objective"]["latency"]["value"] is None
        assert optimise["max"] is None
        assert named(result, "latency")["panel"] is None

    def test_a_panel_answers_both(self):
        result = analysed(study(K=20), K=20)

        optimise = result["recommendation"]["optimise"]
        assert optimise["by_objective"]["latency"]["method"] == "generalizability"
        assert optimise["max"] == max(
            row["value"] for row in optimise["by_objective"].values())
        assert result["front_stability"]

    def test_objectives_come_first(self):
        values = {"hops": np.random.default_rng(1).normal(3, 1, (1, 30)), **study()}

        result = analysed(values)

        assert [m["name"] for m in result["metrics"]] == ["latency", "throughput", "hops"]

    def test_only_objectives_get_a_recommendation(self):
        values = {"hops": np.random.default_rng(1).normal(3, 1, (1, 30)), **study()}

        result = analysed(values)

        assert set(result["recommendation"]["report"]["by_objective"]) == {
            "latency", "throughput"}
        assert named(result, "hops")["is_objective"] is False

    def test_a_copy_of_an_objective_is_counted_once(self):
        values = study()
        values["rtt_latency"] = values["latency"].copy()

        result = analysed(values)

        assert named(result, "latency")["aliases"] == ["rtt_latency"]
        assert [m["name"] for m in result["metrics"]].count("rtt_latency") == 0

    def test_a_defective_metric_stays_visible_with_no_recommendation(self):
        """The proposal's point: 1,409 seeds is what ignoring this would recommend."""
        values = study()
        values["latency"][0, 3] = 1.8e19

        result = analysed(values)

        assert named(result, "latency")["status"] == "overflow_suspect"
        assert result["recommendation"]["report"]["by_objective"]["latency"]["value"] is None
        assert "overflow_suspect" in {w["code"] for w in result["warnings"]}

    def test_a_metric_that_is_always_zero_says_so(self):
        values = {**study(), "bytes_tx": np.zeros((1, 30))}

        metric = named(analysed(values), "bytes_tx")

        assert metric["status"] == "constant" and metric["all_zero"] is True

    def test_an_extrapolation_is_said(self):
        values = study()
        values["latency"] = 200 + 80 * (values["latency"] - values["latency"].mean()) / 30

        result = analysed(values)

        codes = {w["code"]: w for w in result["warnings"]}
        assert "extrapolated" in codes
        assert codes["extrapolated"]["metric"] == "latency"

    def test_normal_data_does_not_read_as_a_disagreement(self):
        rows = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            values = {"latency": rng.normal(200, 24, (1, 60))}
            result = analysed(values, n=60, objectives=OBJECTIVES[:1])
            rows.append("divergent_estimates" in {w["code"] for w in result["warnings"]})

        assert sum(rows) <= 2

    @pytest.mark.parametrize("value, empirical, expected", [
        (40, 20, True),
        (40, 36, False),
        (3, 2, False),
        (100, 20, False),
    ], ids=["far apart", "close", "a seed apart", "past the sample"])
    def test_when_the_two_estimates_disagree(self, value, empirical, expected):
        rows = [{"n_star": {"value": value, "empirical": empirical}}]

        assert _estimates_diverge(rows, 60) is expected

    def test_a_mean_near_zero_needs_an_absolute_target(self):
        values = {"delta": np.random.default_rng(2).normal(0, 1, (1, 30))}
        objectives = [{"metric_name": "delta", "goal": "min"}]

        relative = analysed(values, objectives=objectives)
        given = analysed(values, objectives=objectives, absolute={"delta": 0.5})

        row = named(relative, "delta")["by_chromosome"][0]["n_star"]
        assert row["status"] == "near_zero_mean" and row["value"] is None
        assert "near_zero_mean" in {w["code"] for w in relative["warnings"]}
        assert named(given, "delta")["by_chromosome"][0]["n_star"]["value"] > 1

    def test_a_maximum_has_no_precision_target(self):
        objectives = [{"metric_name": "latency", "goal": "min", "aggregator": "max"}]

        result = analysed({"latency": study()["latency"]}, objectives=objectives)

        report = result["recommendation"]["report"]["by_objective"]["latency"]
        assert report["value"] is None and report["method"] == "none"
        assert "aggregator_no_precision" in {w["code"] for w in result["warnings"]}

    def test_a_median_is_answered_by_resampling(self):
        objectives = [{"metric_name": "throughput", "goal": "max", "aggregator": "median"}]

        result = analysed({"throughput": study()["throughput"]}, objectives=objectives)

        report = result["recommendation"]["report"]["by_objective"]["throughput"]
        assert report["method"] == "empirical" and report["value"] >= 2

    def test_several_objectives_ask_for_more_together(self):
        result = analysed(study())

        report = result["recommendation"]["report"]
        assert report["joint"]["confidence"] == pytest.approx(0.975)
        assert report["joint"]["value"] > max(
            row["value"] for row in report["by_objective"].values())

    def test_a_study_still_running_can_be_read(self):
        values = study()
        for matrix in values.values():
            matrix[:, 5:] = np.nan

        result = analysed(values)

        row = named(result, "latency")["by_chromosome"][0]
        assert row["n"] == 5 and row["ci_mean_t"] is None
        assert row["n_star"]["status"] == "indicative"
        assert named(result, "latency")["observed"] == 5

    def test_the_same_data_gives_the_same_numbers(self):
        first = analysed(study(K=5), K=5)
        second = analysed(study(K=5), K=5)

        assert first == second

    def test_a_small_panel_and_few_seeds_are_warned_about(self):
        codes = {w["code"] for w in analysed(study(K=4, n=12), K=4, n=12)["warnings"]}

        assert {"small_panel", "few_seeds"} <= codes

    def test_a_panel_with_holes_says_what_it_left_out(self):
        values = study(K=12)
        for matrix in values.values():
            matrix[2, 7] = np.nan

        result = analysed(values, K=12)

        assert "panel_unbalanced" in {w["code"] for w in result["warnings"]}

    def test_it_can_be_serialised(self):
        import json

        json.dumps(analysed(study(K=3), K=3), allow_nan=False)


class TestMetricDetail:
    def _detail(self, values, spec="mean"):
        seeds = list(range(100, 100 + len(values)))
        return metric_detail(values, seeds=seeds, simulation_ids=[f"s{s}" for s in seeds],
                             genome_hash="c0", metric="latency", spec=spec)

    def test_every_chart_has_its_data(self):
        values = list(np.random.default_rng(1).normal(200, 30, 30))

        out = self._detail(values)

        assert len(out["values"]) == 30 and len(out["qq"]) == 30
        assert sum(b["count"] for b in out["histogram"]) == 30
        assert len(out["running"]) == 29
        assert out["precision"]["curve"][0]["k"] == 2
        assert out["box"]["q1"] < out["box"]["q3"]

    def test_intervals_appear_as_the_sample_allows(self):
        values = list(np.random.default_rng(1).normal(200, 30, 30))

        intervals = self._detail(values)["intervals"]

        assert intervals["mean_t"] and intervals["mean_bca"] and intervals["sd_bca"]
        assert intervals["p90_upper"] is not None
        assert intervals["p95_upper"] is None, "59 seeds are needed for it"

    def test_n_star_carries_the_range_from_the_interval_of_sd(self):
        values = list(np.random.default_rng(1).normal(200, 30, 30))

        n_star_row = self._detail(values)["n_star"]

        assert n_star_row["range"][0] <= n_star_row["value"] <= n_star_row["range"][1]

    def test_seeds_with_no_result_keep_their_place_in_the_order(self):
        values = [1.0, None, 3.0, 4.0, 2.0]

        out = self._detail(values)

        assert [v["order"] for v in out["values"]] == [0, 2, 3, 4]
        assert [v["seed"] for v in out["values"]] == [100, 102, 103, 104]
        assert out["summary"]["n"] == 4

    def test_a_value_links_back_to_its_simulation(self):
        assert self._detail([1.0, 2.0, 3.0])["values"][1]["simulation_id"] == "s101"

    def test_nothing_measured(self):
        out = self._detail([None, None, None])

        assert out["values"] == [] and out["summary"]["mean"] is None


class TestCompareControl:
    def test_identical_repeats(self):
        out = compare_control({"c": {"m": 1.5}}, {"c": [{"m": 1.5}, {"m": 1.5}]}, ["m"])

        assert out["checked"] and out["identical"]
        assert out["replicates"] == 2 and out["compared"] == 1

    def test_a_divergence_names_where(self):
        out = compare_control({"c": {"m": 1.5}}, {"c": [{"m": 1.5}, {"m": 1.6}]}, ["m"])

        assert out["identical"] is False
        assert out["divergent"] == [
            {"genome_hash": "c", "metric": "m", "values": [1.5, 1.5, 1.6]}]

    def test_the_repeats_are_compared_when_the_main_run_failed(self):
        out = compare_control({}, {"c": [{"m": 1.5}, {"m": 1.5}]}, ["m"])

        assert out["checked"] and out["identical"]

    def test_no_control_yet(self):
        out = compare_control({"c": {"m": 1.5}}, {}, ["m"])

        assert out["checked"] is False and out["identical"] is False


class TestFailures:
    @pytest.mark.parametrize("message, kind", [
        ("Could not reach vm8/vm8-cooja1: host", "infrastructure"),
        ("Recovered: the master-node restarted", "infrastructure"),
        ("could not fetch /opt/contiki-ng/tools/cooja/COOJA.testlog", "infrastructure"),
        ("the run process disappeared without recording an exit code", "infrastructure"),
        ("Cooja exited with code 1", "execution"),
        ("", "execution"),
        (None, "execution"),
    ])
    def test_where_a_failure_comes_from(self, message, kind):
        assert failure_kind(message) == kind

    def test_a_rate_over_every_attempt(self):
        attempts = [{"status": "Done"}] * 28 + [
            {"status": "Error", "system_message": "Could not reach vm"},
            {"status": "Error", "system_message": "boom"},
            {"status": "Running"},
        ]

        out = failure_summary(attempts, 0.95)

        assert out["attempts"] == 30 and out["failed"] == 2
        assert out["by_kind"] == {"infrastructure": 1, "execution": 1}
        assert out["rate_ci"][0] < 2 / 30 < out["rate_ci"][1]
