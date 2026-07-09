"""Tests for the multi-run statistical tooling (lib/stats.py)."""
import math

import numpy as np
import pytest

from lib.stats import summary, wilcoxon_pvalue, compare_to_baseline


class TestSummary:
    def test_mean_std_ci(self):
        s = summary([1, 2, 3, 4, 5])
        assert s["n"] == 5
        assert s["mean"] == pytest.approx(3.0)
        assert s["std"] == pytest.approx(float(np.std([1, 2, 3, 4, 5], ddof=1)))
        assert s["ci95"] > 0

    def test_empty_is_nan(self):
        s = summary([])
        assert s["n"] == 0 and math.isnan(s["mean"])

    def test_single_has_zero_dispersion(self):
        s = summary([7.0])
        assert s["mean"] == 7.0 and s["std"] == 0.0 and s["ci95"] == 0.0

    def test_ignores_non_finite(self):
        assert summary([1.0, 2.0, float("nan"), float("inf")])["n"] == 2


class TestWilcoxon:
    def test_identical_samples_is_nan(self):
        assert math.isnan(wilcoxon_pvalue([1, 2, 3], [1, 2, 3]))

    def test_clearly_separated_is_significant(self):
        a = list(range(1, 13))
        b = [x + 5 for x in a]        # b strictly greater at every position
        p = wilcoxon_pvalue(a, b)
        assert 0.0 <= p < 0.05

    def test_shape_mismatch_is_nan(self):
        assert math.isnan(wilcoxon_pvalue([1, 2], [1, 2, 3]))


class TestCompareToBaseline:
    def test_baseline_and_others(self):
        data = {"base": [1, 2, 3, 4, 5], "other": [3, 4, 5, 6, 7]}
        r = compare_to_baseline(data, "base")
        assert math.isnan(r["base"]["p_vs_baseline"])
        assert r["other"]["mean"] == pytest.approx(5.0)
        assert not math.isnan(r["other"]["p_vs_baseline"])

    def test_missing_baseline_raises(self):
        with pytest.raises(KeyError):
            compare_to_baseline({"a": [1, 2]}, "nope")
