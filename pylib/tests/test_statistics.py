"""Unit tests for pylib.statistics — the layer that turns raw simulation logs
into objective values. Errors here silently corrupt every experiment result,
so the aggregation registry and cross-seed aggregators are covered directly.
"""
import logging
import math

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
    resolve_aggregator,
    aggregate_seed_values,
    evaluate_config,
    _agg_mean,
    _agg_median,
    _agg_trimmed_mean,
    _agg_min,
    _agg_max,
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


# ── Cross-seed aggregators ───────────────────────────────────────────────────

class TestAggregators:
    def test_mean(self):
        assert _agg_mean([1.0, 2.0, 3.0], {}) == pytest.approx(2.0)

    def test_median_odd_even(self):
        assert _agg_median([3.0, 1.0, 2.0], {}) == pytest.approx(2.0)
        assert _agg_median([1.0, 2.0, 3.0, 4.0], {}) == pytest.approx(2.5)

    def test_trimmed_mean_removes_extremes(self):
        # trim=0.2 of 5 → k=floor(1.0)=1 → drop 1 each side → mean(2,3,4)=3
        vals = [1.0, 2.0, 3.0, 4.0, 100.0]
        assert _agg_trimmed_mean(vals, {"trim": 0.2}) == pytest.approx(3.0)

    def test_trimmed_mean_invalid_trim_raises(self):
        with pytest.raises(ValueError):
            _agg_trimmed_mean([1.0, 2.0], {"trim": 0.7})

    def test_min_max(self):
        assert _agg_min([3.0, 1.0, 2.0], {}) == 1.0
        assert _agg_max([3.0, 1.0, 2.0], {}) == 3.0

    @pytest.mark.parametrize("fn", [_agg_mean, _agg_median, _agg_min, _agg_max])
    def test_empty_raises(self, fn):
        with pytest.raises(ValueError):
            fn([], {})


# ── resolve_aggregator / aggregate_seed_values ───────────────────────────────

class TestResolveAggregator:
    def test_string_spec(self):
        assert resolve_aggregator("mean") == ("mean", {})

    def test_dict_spec(self):
        kind, params = resolve_aggregator({"kind": "trimmed_mean", "trim": 0.1})
        assert kind == "trimmed_mean"
        assert params == {"trim": 0.1}

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            resolve_aggregator("does_not_exist")

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            resolve_aggregator(42)  # type: ignore[arg-type]

    def test_aggregate_seed_values_end_to_end(self):
        assert aggregate_seed_values([2.0, 4.0, 6.0], "mean") == pytest.approx(4.0)
        assert aggregate_seed_values([1.0, 5.0], "max") == 5.0


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
