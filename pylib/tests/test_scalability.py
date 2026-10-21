"""From benchmark documents to measurements, and from those to a curve.

The fit itself is tested in ``test_usl.py``. What is tested here is the part
that decides what a measurement *is*: which timestamps bound a wave, which
waves count, and how two definitions of throughput differ.
"""
from datetime import datetime, timedelta, timezone

import pytest
from bson import ObjectId

from pylib import scalability, usl
from pylib.db.models.enums import EnumStatus

T0 = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)


def wave(level, repetition=0, index=0, status=EnumStatus.DONE):
    return {"_id": ObjectId(), "index": index, "status": status,
            "benchmark": {"level": level, "repetition": repetition, "wave": index}}


def run(assigned_after, held, *, status=EnumStatus.DONE, host="local"):
    """One simulation, placed ``assigned_after`` s into the wave, held
    ``held`` s."""
    assigned = T0 + timedelta(seconds=assigned_after)
    return {"status": status, "assigned_at": assigned,
            "start_time": assigned + timedelta(seconds=2),
            "end_time": assigned + timedelta(seconds=held),
            "assigned_host_name": host}


class TestMeasuringOneWave:
    def test_both_throughputs_from_the_right_timestamps(self):
        """Four simulations, placed over 3 s, each held 100–110 s."""
        result = scalability.measure_wave(wave(4), [
            run(0, 100), run(1, 104), run(2, 106), run(3, 110),
        ])

        assert result["valid"] is True
        assert result["makespan_sec"] == pytest.approx(113.0), "0 → 3 + 110"
        assert result["mean_held_sec"] == pytest.approx(105.0)
        assert result["throughput"]["batch"] == pytest.approx(60 * 4 / 113.0)
        assert result["throughput"]["per_simulation"] == pytest.approx(60 * 4 / 105.0)

    def test_the_held_time_starts_at_the_reservation_not_at_start_time(self):
        """``start_time`` is written after the upload, which leaves out the
        part of the cost that differs most between local and remote."""
        sim = run(0, 100)
        sim["start_time"] = T0 + timedelta(seconds=40)

        result = scalability.measure_wave(wave(1), [sim])

        assert result["mean_held_sec"] == pytest.approx(100.0)

    def test_a_straggler_moves_the_batch_and_barely_the_mean(self):
        """The difference between the two definitions, as a number."""
        result = scalability.measure_wave(wave(4), [
            run(0, 100), run(0, 100), run(0, 100), run(0, 200),
        ])

        assert result["makespan_sec"] == pytest.approx(200.0)
        assert result["mean_held_sec"] == pytest.approx(125.0)
        assert result["throughput"]["batch"] < result["throughput"]["per_simulation"]

    def test_a_wave_with_a_failure_is_not_a_measurement(self):
        """Fewer than N finished, so N is not the concurrency it measured."""
        result = scalability.measure_wave(wave(3), [
            run(0, 100), run(0, 100), run(0, 30, status=EnumStatus.ERROR),
        ])

        assert result["valid"] is False
        assert result["failed"] == 1
        assert "did not produce a result" in result["reason"]
        assert result["throughput"]["batch"] is None

    def test_a_wave_still_running_is_not_measured_yet(self):
        sims = [run(0, 100), run(0, 100)]
        sims[1]["status"] = EnumStatus.RUNNING

        result = scalability.measure_wave(wave(2), sims)

        assert result["valid"] is False and result["reason"] == "still running"

    def test_a_wave_whose_simulations_are_not_all_inserted_yet(self):
        result = scalability.measure_wave(wave(4), [run(0, 100)])

        assert result["valid"] is False
        assert "only 1 of 4" in result["reason"]

    def test_a_simulation_without_a_placement_time_invalidates_the_wave(self):
        """An older document, or one written by another path, has no
        ``assigned_at``. Guessing it would invent the held time."""
        sim = run(0, 100)
        sim["assigned_at"] = None

        result = scalability.measure_wave(wave(1), [sim])

        assert result["valid"] is False
        assert "placement" in result["reason"]

    def test_the_hosts_of_the_wave_are_recorded(self):
        """A single-container wave lands wherever the policy puts it, so the
        level-1 baseline can come from a fast host one repetition and a slow
        one the next. Recording it is what makes that spread explicable."""
        result = scalability.measure_wave(wave(3), [
            run(0, 100, host="local"), run(0, 100, host="local"),
            run(0, 140, host="vm7"),
        ])

        assert result["hosts"] == {"local": 2, "vm7": 1}

    def test_naive_timestamps_are_read_as_utc(self):
        sim = run(0, 100)
        sim["assigned_at"] = sim["assigned_at"].replace(tzinfo=None)

        result = scalability.measure_wave(wave(1), [sim])

        assert result["valid"] is True
        assert result["mean_held_sec"] == pytest.approx(100.0)


class TestSummarisingLevels:
    def test_repetitions_are_averaged_with_their_spread(self):
        waves = [
            {"level": 2, "valid": True, "throughput": {"batch": 1.0}},
            {"level": 2, "valid": True, "throughput": {"batch": 1.2}},
            {"level": 2, "valid": False, "throughput": {"batch": None}},
            {"level": 1, "valid": True, "throughput": {"batch": 0.6}},
        ]

        rows = scalability.summarize_levels(waves, "batch")

        assert [r["level"] for r in rows] == [1, 2]
        assert rows[1]["mean"] == pytest.approx(1.1)
        assert rows[1]["count"] == 2, "the invalid wave is not counted"
        assert rows[1]["std"] == pytest.approx(0.1414, rel=1e-3)
        assert rows[0]["std"] == 0.0


def benchmark(lam, sigma, kappa, levels=(1, 2, 4, 8), repetitions=2):
    """Generations and simulations whose batch throughput follows a known
    USL exactly — every simulation of a wave held for the same time."""
    generations, by_gen = [], {}
    index = 0
    for rep in range(repetitions):
        for level in levels:
            gen = wave(level, repetition=rep, index=index)
            x_per_min = float(usl.throughput(level, lam, sigma, kappa))
            held = 60.0 * level / x_per_min
            generations.append(gen)
            by_gen[str(gen["_id"])] = [run(0, held) for _ in range(level)]
            index += 1
    return generations, by_gen


class TestAnalysingABenchmark:
    def test_the_known_parameters_come_back(self):
        generations, by_gen = benchmark(0.5, 0.1, 0.004)

        result = scalability.analyse(generations, by_gen)

        fitted = result["fits"]["batch"]
        assert fitted["status"] == usl.FIT_OK
        assert fitted["lam"] == pytest.approx(0.5, rel=1e-2)
        assert fitted["sigma"] == pytest.approx(0.1, abs=5e-3)
        assert fitted["kappa"] == pytest.approx(0.004, rel=5e-2)
        assert result["valid_waves"] == 8

    def test_both_definitions_are_fitted(self):
        result = scalability.analyse(*benchmark(0.5, 0.1, 0.0))

        assert set(result["fits"]) == {"batch", "per_simulation"}
        assert result["default_metric"] == "batch"

    def test_a_peak_beyond_the_tested_range_is_flagged(self):
        """Inside the range it was measured; outside it is extrapolated, and
        the chart has to say which."""
        result = scalability.analyse(*benchmark(0.5, 0.05, 0.004))

        fitted = result["fits"]["batch"]
        assert fitted["n_peak"] > 8
        assert fitted["peak_within_tested"] is False

    def test_a_peak_inside_the_tested_range_is_flagged_as_such(self):
        result = scalability.analyse(*benchmark(0.5, 0.05, 0.05))

        fitted = result["fits"]["batch"]
        assert fitted["n_peak"] < 8
        assert fitted["peak_within_tested"] is True

    def test_the_curve_reaches_past_the_tested_range(self):
        result = scalability.analyse(*benchmark(0.5, 0.1, 0.0))

        assert result["fits"]["batch"]["curve"][-1]["n"] >= 16

    def test_generations_that_are_not_benchmark_waves_are_ignored(self):
        generations, by_gen = benchmark(0.5, 0.1, 0.0)
        stray = {"_id": ObjectId(), "index": 99, "status": EnumStatus.DONE}

        result = scalability.analyse(generations + [stray], by_gen)

        assert len(result["waves"]) == 8

    def test_an_empty_benchmark_is_an_empty_result_not_an_error(self):
        result = scalability.analyse([], {}, planned_levels=[1, 2, 4])

        assert result["waves"] == []
        assert result["fits"]["batch"]["status"] == usl.FIT_INSUFFICIENT
        assert result["planned_levels"] == [1, 2, 4]
