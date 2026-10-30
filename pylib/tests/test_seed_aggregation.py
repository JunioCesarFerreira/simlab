"""Unit tests for pylib.seed_aggregation — the cross-seed aggregator (Ψa) that
turns the replicas of one individual into its objective values."""
import pytest

from pylib.seed_aggregation import (
    resolve_aggregator,
    aggregate_seed_values,
    objective_aggregators,
    _agg_mean,
    _agg_median,
    _agg_trimmed_mean,
    _agg_min,
    _agg_max,
)


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

    def test_trimmed_mean_below_one_value_per_side_is_mean(self):
        # trim=0.1 of 8 → k=floor(0.8)=0 → nothing dropped
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0]
        assert _agg_trimmed_mean(vals, {"trim": 0.1}) == pytest.approx(_agg_mean(vals, {}))

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


class TestResolveAggregator:
    def test_string_spec(self):
        assert resolve_aggregator("mean") == ("mean", {})

    def test_dict_spec(self):
        kind, params = resolve_aggregator({"kind": "trimmed_mean", "trim": 0.1})
        assert kind == "trimmed_mean"
        assert params == {"trim": 0.1}

    def test_dict_without_kind_is_mean(self):
        assert resolve_aggregator({}) == ("mean", {})

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            resolve_aggregator("does_not_exist")

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            resolve_aggregator(42)  # type: ignore[arg-type]

    @pytest.mark.parametrize("trim", [-0.1, 0.5, 0.9])
    def test_trim_out_of_range_raises(self, trim):
        with pytest.raises(ValueError, match="trim"):
            resolve_aggregator({"kind": "trimmed_mean", "trim": trim})

    @pytest.mark.parametrize("trim", ["0.1", None, True])
    def test_trim_not_a_number_raises(self, trim):
        with pytest.raises(ValueError, match="trim"):
            resolve_aggregator({"kind": "trimmed_mean", "trim": trim})

    def test_misspelt_parameter_raises(self):
        with pytest.raises(ValueError, match="trimm"):
            resolve_aggregator({"kind": "trimmed_mean", "trimm": 0.1})

    def test_parameter_on_kind_without_parameters_raises(self):
        with pytest.raises(ValueError, match="median"):
            resolve_aggregator({"kind": "median", "trim": 0.1})

    def test_aggregate_seed_values_end_to_end(self):
        assert aggregate_seed_values([2.0, 4.0, 6.0], "mean") == pytest.approx(4.0)
        assert aggregate_seed_values([1.0, 5.0], "max") == 5.0


class TestObjectiveAggregators:
    def test_objective_overrides_default(self):
        objectives = [
            {"metric_name": "latency", "goal": "min", "aggregator": "max"},
            {"metric_name": "energy", "goal": "min"},
            {"metric_name": "throughput", "goal": "max",
             "aggregator": {"kind": "trimmed_mean", "trim": 0.2}},
        ]
        assert objective_aggregators(objectives, "median") == [
            "max", "median", {"kind": "trimmed_mean", "trim": 0.2}]

    def test_default_is_mean(self):
        assert objective_aggregators([{"metric_name": "latency", "goal": "min"}]) == ["mean"]

    def test_invalid_objective_names_its_position(self):
        objectives = [
            {"metric_name": "latency", "goal": "min"},
            {"metric_name": "energy", "goal": "min", "aggregator": "average"},
        ]
        with pytest.raises(ValueError, match=r"^objectives\[1\]\.aggregator"):
            objective_aggregators(objectives)

    def test_wrong_type_is_reported_as_value_error(self):
        with pytest.raises(ValueError, match=r"^objectives\[0\]\.aggregator"):
            objective_aggregators([{"metric_name": "latency", "goal": "min", "aggregator": None}])

    def test_invalid_default_is_reported_even_if_unused(self):
        with pytest.raises(ValueError, match=r"^simulation\.aggregator"):
            objective_aggregators([{"metric_name": "latency", "goal": "min", "aggregator": "max"}], "xyz")

    def test_same_metric_two_ways_raises(self):
        objectives = [
            {"metric_name": "latency", "goal": "min", "aggregator": "max"},
            {"metric_name": "latency", "goal": "max", "aggregator": "min"},
        ]
        with pytest.raises(ValueError, match="latency"):
            objective_aggregators(objectives)

    def test_same_metric_same_aggregator_is_fine(self):
        objectives = [
            {"metric_name": "latency", "goal": "min", "aggregator": "median"},
            {"metric_name": "latency", "goal": "max", "aggregator": {"kind": "median"}},
        ]
        assert len(objective_aggregators(objectives)) == 2
