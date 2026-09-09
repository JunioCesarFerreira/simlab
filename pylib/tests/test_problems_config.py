"""Tests for the problem deserialization layer (pylib/config/problems.py).

Focused on alpha (`min_coverage_percentage`), the trajectory coverage level of
Problems P1 and P2: it is optional, defaults to full coverage, and is rejected
at the cast boundary when out of range — before any simulation is spent.
"""
import pytest

from pylib.config.problems import (
    DEFAULT_MIN_COVERAGE_PERCENTAGE,
    ProblemP1,
    ProblemP2,
    parse_min_coverage_percentage,
)


def _p1(**overrides) -> dict:
    base = {
        "name": "problem1",
        "radius_of_reach": 100,
        "radius_of_inter": 200,
        "region": [-100, -100, 100, 100],
        "sink": [0, 0],
        "number_of_relays": 4,
        "mobile_nodes": [],
    }
    base.update(overrides)
    return base


def _p2(**overrides) -> dict:
    base = {
        "name": "problem2",
        "radius_of_reach": 100,
        "radius_of_inter": 200,
        "region": [-100, -100, 100, 100],
        "sink": [0, 0],
        "candidates": [[10, 10], [20, 20]],
        "mobile_nodes": [],
    }
    base.update(overrides)
    return base


class TestParseMinCoveragePercentage:
    def test_defaults_to_full_coverage(self):
        assert parse_min_coverage_percentage({}) == DEFAULT_MIN_COVERAGE_PERCENTAGE
        assert DEFAULT_MIN_COVERAGE_PERCENTAGE == 100.0

    @pytest.mark.parametrize("raw, expected", [(0, 0.0), (85, 85.0), ("90.5", 90.5), (100, 100.0)])
    def test_accepts_the_closed_percent_range(self, raw, expected):
        assert parse_min_coverage_percentage({"min_coverage_percentage": raw}) == expected

    @pytest.mark.parametrize("raw", [-0.1, 100.1, 101, -1])
    def test_rejects_values_outside_the_percent_range(self, raw):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            parse_min_coverage_percentage({"min_coverage_percentage": raw})

    @pytest.mark.parametrize("raw", ["full", None, [90]])
    def test_rejects_non_numeric_values(self, raw):
        with pytest.raises(ValueError, match="must be a number"):
            parse_min_coverage_percentage({"min_coverage_percentage": raw})


class TestProblemCast:
    def test_p1_reads_alpha(self):
        assert ProblemP1.cast(_p1(min_coverage_percentage=80)).min_coverage_percentage == 80.0

    def test_p2_reads_alpha(self):
        assert ProblemP2.cast(_p2(min_coverage_percentage=80)).min_coverage_percentage == 80.0

    def test_alpha_is_optional_on_both(self):
        assert ProblemP1.cast(_p1()).min_coverage_percentage == 100.0
        assert ProblemP2.cast(_p2()).min_coverage_percentage == 100.0

    def test_out_of_range_alpha_fails_the_cast(self):
        with pytest.raises(ValueError):
            ProblemP1.cast(_p1(min_coverage_percentage=150))
        with pytest.raises(ValueError):
            ProblemP2.cast(_p2(min_coverage_percentage=-5))
