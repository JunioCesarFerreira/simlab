"""Unit tests for pylib.statistics — the layer that turns raw simulation logs
into objective values. Errors here silently corrupt every experiment result,
so the aggregation registry is covered directly. The cross-seed aggregators
live in pylib.seed_aggregation (test_seed_aggregation.py).
"""
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylib.statistics import (
    quantile,
    mean,
    median,
    sum_last_minus_first,
    sum_rate,
    inverse_of,
    sum_all,
    sum_max,
    ratio_sum,
    ratio_sum_max,
    node_coverage,
    evaluate_config,
    STAT_DISPATCH,
)

log = logging.getLogger("test")


# ── Column statistics ────────────────────────────────────────────────────────

class TestColumnStats:
    def test_quantile_median(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert quantile(s, 0.5) == pytest.approx(2.5)

    def test_quantile_empty_is_nan(self):
        assert math.isnan(quantile(pd.Series([], dtype=float), 0.5))

    def test_quantile_ignores_nan(self):
        s = pd.Series([1.0, np.nan, 3.0])
        assert quantile(s, 0.5) == pytest.approx(2.0)

    def test_mean_basic_and_dropna(self):
        assert mean(pd.Series([2.0, 4.0, np.nan])) == pytest.approx(3.0)

    def test_mean_empty_is_nan(self):
        assert math.isnan(mean(pd.Series([], dtype=float)))

    def test_median_odd_even(self):
        assert median(pd.Series([3.0, 1.0, 2.0])) == pytest.approx(2.0)
        assert median(pd.Series([1.0, 2.0, 3.0, 4.0])) == pytest.approx(2.5)

    def test_median_empty_is_nan(self):
        assert math.isnan(median(pd.Series([], dtype=float)))

    def test_inverse_of(self):
        assert inverse_of(4.0, scale=2.0) == pytest.approx(0.5, rel=1e-6)

    def test_inverse_of_zero_is_finite(self):
        # 1e-9 epsilon prevents division by zero
        assert math.isfinite(inverse_of(0.0))


# ── sum_all ──────────────────────────────────────────────────────────────────

class TestSumAll:
    def test_sums_numeric(self):
        df = pd.DataFrame({"v": [1, 2, 3]})
        assert sum_all(df, "v") == pytest.approx(6.0)

    def test_coerces_non_numeric_to_nan(self):
        df = pd.DataFrame({"v": [1, "x", 3]})
        assert sum_all(df, "v") == pytest.approx(4.0)

    def test_missing_column_is_nan(self):
        assert math.isnan(sum_all(pd.DataFrame({"a": [1]}), "v"))


# ── sum_last_minus_first ─────────────────────────────────────────────────────

class TestSumLastMinusFirst:
    def test_per_node_growth(self):
        df = pd.DataFrame({
            "node": ["a", "a", "b", "b"],
            "root_time_now": [1, 2, 1, 2],
            "total": [10, 15, 100, 130],
        })
        # (15-10) + (130-100) = 5 + 30 = 35
        assert sum_last_minus_first(df, "total") == pytest.approx(35.0)

    def test_negative_variation_clipped(self):
        df = pd.DataFrame({
            "node": ["a", "a"],
            "root_time_now": [1, 2],
            "total": [50, 20],  # decreasing → clipped to 0
        })
        assert sum_last_minus_first(df, "total") == pytest.approx(0.0)

    def test_missing_column_is_nan(self):
        df = pd.DataFrame({"node": ["a"], "root_time_now": [1]})
        assert math.isnan(sum_last_minus_first(df, "total"))

    def test_ordering_respected(self):
        # rows out of time order must still compute last-first correctly
        df = pd.DataFrame({
            "node": ["a", "a", "a"],
            "root_time_now": [3, 1, 2],
            "total": [30, 10, 20],
        })
        assert sum_last_minus_first(df, "total") == pytest.approx(20.0)


# ── sum_rate ─────────────────────────────────────────────────────────────────

class TestSumRate:
    def test_volume_per_second(self):
        # duration = (3000 - 1000)/1000 = 2s ; volume = 10+20+30 = 60 → 30/s
        df = pd.DataFrame({
            "root_time_now": [1000, 2000, 3000],
            "vol": [10, 20, 30],
        })
        assert sum_rate(df, "vol") == pytest.approx(30.0)

    def test_zero_duration_is_nan(self):
        df = pd.DataFrame({"root_time_now": [1000, 1000], "vol": [5, 5]})
        assert math.isnan(sum_rate(df, "vol"))

    def test_missing_column_is_nan(self):
        assert math.isnan(sum_rate(pd.DataFrame({"root_time_now": [1]}), "vol"))


# ── evaluate_config (dispatch integration) ───────────────────────────────────

class TestEvaluateConfig:
    def test_dispatch_multiple_metrics(self):
        df = pd.DataFrame({
            "node": ["a", "a"],
            "root_time_now": [1, 2],
            "latency": [10.0, 20.0],
            "sent": [100, 140],
        })
        cfg = {
            "node_col": "node",
            "time_col": "root_time_now",
            "metrics": [
                {"name": "lat_mean", "kind": "mean", "column": "latency"},
                {"name": "sent_delta", "kind": "sum_last_minus_first", "column": "sent"},
            ],
        }
        out = evaluate_config(df, cfg, log)
        assert out["lat_mean"] == pytest.approx(15.0)
        assert out["sent_delta"] == pytest.approx(40.0)

    def test_unknown_kind_yields_nan(self):
        df = pd.DataFrame({"x": [1.0]})
        cfg = {"metrics": [{"name": "bad", "kind": "nope", "column": "x"}]}
        out = evaluate_config(df, cfg, log)
        assert math.isnan(out["bad"])

    def test_missing_column_yields_nan(self):
        df = pd.DataFrame({"x": [1.0]})
        cfg = {"metrics": [{"name": "m", "kind": "mean", "column": "absent"}]}
        out = evaluate_config(df, cfg, log)
        assert math.isnan(out["m"])


# ── sum_max ──────────────────────────────────────────────────────────────────

class TestSumMax:
    def test_total_of_a_counter_counts_every_report(self):
        # a: reports 1..3 arrived, b: 1..2. last-minus-first would give 2 + 1.
        df = pd.DataFrame({"node": ["a", "a", "a", "b", "b"], "total": [1, 2, 3, 1, 2]})
        assert sum_max(df, "total") == pytest.approx(5.0)

    def test_a_late_report_does_not_lower_the_total(self):
        # TSCH: report 3 reaches the root after report 4.
        df = pd.DataFrame({
            "node": ["a"] * 4, "root_time_now": [1, 2, 3, 4], "total": [1, 2, 4, 3],
        })
        assert sum_max(df, "total") == pytest.approx(4.0)

    def test_missing_column_is_nan(self):
        assert math.isnan(sum_max(pd.DataFrame({"node": ["a"]}), "total"))


# ── ratio kinds ──────────────────────────────────────────────────────────────

class TestRatios:
    def test_ratio_sum_is_a_ratio_of_totals_not_a_mean_of_ratios(self):
        # 10 mJ in 1 s and 0 mJ in 9 s: 1 mW over the 10 s, not the 5 mW mean of the two rows.
        df = pd.DataFrame({"e": [10.0, 0.0], "t": [1000.0, 9000.0]})
        assert ratio_sum(df, "e", "t", scale=1000) == pytest.approx(1.0)

    def test_ratio_sum_max_divides_counter_totals(self):
        df = pd.DataFrame({
            "node": ["a", "a", "b"], "rx": [1, 2, 1], "sent": [1, 4, 4],
        })
        assert ratio_sum_max(df, "rx", "sent") == pytest.approx(3 / 8)

    def test_zero_denominator_is_nan_not_infinity(self):
        df = pd.DataFrame({"node": ["a"], "e": [1.0], "t": [0.0]})
        assert math.isnan(ratio_sum(df, "e", "t"))
        assert math.isnan(ratio_sum_max(df, "e", "t"))

    def test_missing_denominator_column_is_nan(self):
        df = pd.DataFrame({"node": ["a"], "e": [1.0]})
        assert math.isnan(ratio_sum(df, "e", "t"))


# ── node_coverage ────────────────────────────────────────────────────────────

class TestNodeCoverage:
    def test_fraction_of_deployed_nodes_heard_from(self):
        df = pd.DataFrame({"node": ["a", "a", "b"]})
        assert node_coverage(df, "node", 4) == pytest.approx(0.5)

    def test_unknown_deployment_is_nan(self):
        df = pd.DataFrame({"node": ["a"]})
        assert math.isnan(node_coverage(df, "node", None))
        assert math.isnan(node_coverage(df, "node", 0))


class TestEvaluateConfigNewKinds:
    df = pd.DataFrame({
        "node": ["a", "a", "b"],
        "root_time_now": [1, 2, 1],
        "e": [3.0, 1.0, 2.0],
        "t": [1000.0, 1000.0, 2000.0],
        "rx": [1, 2, 1],
        "sent": [1, 2, 2],
        "lat": [10.0, 20.0, 90.0],
    })

    def test_all_kinds_through_the_dispatcher(self):
        cfg = {"metrics": [
            {"name": "p", "kind": "ratio_sum", "column": "e", "denominator": "t", "scale": 1000},
            {"name": "d", "kind": "ratio_sum_max", "column": "rx", "denominator": "sent"},
            {"name": "n", "kind": "node_coverage", "column": "node"},
            {"name": "s", "kind": "sum_max", "column": "rx"},
            {"name": "m", "kind": "median", "column": "lat"},
            {"name": "q", "kind": "quantile", "column": "lat", "q": 0.5},
        ]}
        out = evaluate_config(self.df, cfg, log, deployed_nodes=3)
        assert out == pytest.approx({"p": 1.5, "d": 0.75, "n": 2 / 3, "s": 3.0, "m": 20.0, "q": 20.0})

    def test_ratio_without_a_denominator_is_nan(self):
        cfg = {"metrics": [{"name": "p", "kind": "ratio_sum", "column": "e"}]}
        assert math.isnan(evaluate_config(self.df, cfg, log)["p"])

    def test_ratio_with_an_absent_denominator_column_is_nan(self):
        cfg = {"metrics": [{"name": "p", "kind": "ratio_sum", "column": "e", "denominator": "nope"}]}
        assert math.isnan(evaluate_config(self.df, cfg, log)["p"])

    def test_node_coverage_without_deployment_is_nan(self):
        cfg = {"metrics": [{"name": "n", "kind": "node_coverage", "column": "node"}]}
        assert math.isnan(evaluate_config(self.df, cfg, log)["n"])


# ── the GUI offers what the evaluator knows ─────────────────────────────────

def test_the_gui_offers_exactly_the_kinds_the_evaluator_knows():
    """The wizard once offered max, min, last and percentile, which were not
    kinds here: a metric that used one became NaN without failing the run."""
    ts = Path(__file__).resolve().parents[2] / "gui" / "simlab" / "src" / "services" / "metricKinds.ts"
    offered = set(re.findall(r'\{ value: "(\w+)"', ts.read_text()))
    assert offered, "could not read the kinds from metricKinds.ts"
    assert offered == set(STAT_DISPATCH)
