"""Tests for AGGREGATOR_DISPATCH and aggregate_seed_values (C4 — Ψa plural)."""
import pytest
from pylib.statistics import (
    AGGREGATOR_DISPATCH,
    aggregate_seed_values,
    resolve_aggregator,
)


# ---------------------------------------------------------------------------
# resolve_aggregator
# ---------------------------------------------------------------------------

def test_resolve_string_mean():
    kind, params = resolve_aggregator("mean")
    assert kind == "mean"
    assert params == {}


def test_resolve_string_median():
    kind, params = resolve_aggregator("median")
    assert kind == "median"
    assert params == {}


def test_resolve_dict_trimmed_mean():
    kind, params = resolve_aggregator({"kind": "trimmed_mean", "trim": 0.2})
    assert kind == "trimmed_mean"
    assert params == {"trim": 0.2}


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown aggregator"):
        resolve_aggregator("xyz")


def test_resolve_bad_type_raises():
    with pytest.raises(TypeError):
        resolve_aggregator(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# aggregate_seed_values — mean
# ---------------------------------------------------------------------------

def test_mean_basic():
    assert aggregate_seed_values([1.0, 2.0, 3.0], "mean") == pytest.approx(2.0)


def test_mean_single():
    assert aggregate_seed_values([5.0], "mean") == pytest.approx(5.0)


def test_mean_empty_raises():
    with pytest.raises(ValueError):
        aggregate_seed_values([], "mean")


# ---------------------------------------------------------------------------
# aggregate_seed_values — median
# ---------------------------------------------------------------------------

def test_median_odd():
    assert aggregate_seed_values([3.0, 1.0, 2.0], "median") == pytest.approx(2.0)


def test_median_even():
    assert aggregate_seed_values([1.0, 2.0, 3.0, 4.0], "median") == pytest.approx(2.5)


def test_median_empty_raises():
    with pytest.raises(ValueError):
        aggregate_seed_values([], "median")


# ---------------------------------------------------------------------------
# aggregate_seed_values — trimmed_mean
# ---------------------------------------------------------------------------

def test_trimmed_mean_zero_is_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert aggregate_seed_values(values, {"kind": "trimmed_mean", "trim": 0.0}) == pytest.approx(3.0)


def test_trimmed_mean_removes_extremes():
    # 10 values: [1..10], trim=0.1 → floor(0.1*10)=1 removed each side → [2..9]
    values = list(range(1, 11))  # [1,2,...,10]
    result = aggregate_seed_values(values, {"kind": "trimmed_mean", "trim": 0.1})
    expected = sum(range(2, 10)) / 8  # [2..9]
    assert result == pytest.approx(expected)


def test_trimmed_mean_invalid_trim_raises():
    with pytest.raises(ValueError, match="trim must be"):
        aggregate_seed_values([1.0, 2.0], {"kind": "trimmed_mean", "trim": 0.5})


def test_trimmed_mean_empty_raises():
    with pytest.raises(ValueError):
        aggregate_seed_values([], {"kind": "trimmed_mean", "trim": 0.1})


# ---------------------------------------------------------------------------
# aggregate_seed_values — min / max
# ---------------------------------------------------------------------------

def test_min():
    assert aggregate_seed_values([5.0, 1.0, 3.0], "min") == pytest.approx(1.0)


def test_max():
    assert aggregate_seed_values([5.0, 1.0, 3.0], "max") == pytest.approx(5.0)


def test_min_empty_raises():
    with pytest.raises(ValueError):
        aggregate_seed_values([], "min")


def test_max_empty_raises():
    with pytest.raises(ValueError):
        aggregate_seed_values([], "max")


# ---------------------------------------------------------------------------
# Regression: callers without aggregator arg still get mean behaviour
# ---------------------------------------------------------------------------

def test_all_known_aggregators_present():
    for key in ("mean", "median", "trimmed_mean", "min", "max"):
        assert key in AGGREGATOR_DISPATCH, f"Missing aggregator: {key}"
