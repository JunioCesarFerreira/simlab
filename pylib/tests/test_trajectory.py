"""A closed path returns to its start, and every mobile mote moves for the
whole simulation at its own period.

``is_closed`` was read by nobody, and Cooja's Mobility plugin replays the
positions file with the period of its longest trajectory, so a mote with a
shorter path stood at its end until the longest one finished.
"""
import math

import pytest

from pylib.cooja_builder.parse_json_pos_dat import generate_positions_from_json
from pylib.trajectory import closing_segment, path_segments, point_at

LINE = [("0 + 10 * t", "0")]                                  # (0,0) → (10,0)
L_SHAPE = [("10 * t", "0"), ("10", "10 * t")]                 # (0,0) → (10,0) → (10,10)
ELLIPSE = [("5 * np.cos(2 * np.pi * t)", "3 * np.sin(2 * np.pi * t)")]


class TestClosingSegment:
    def test_an_open_path_gets_the_way_back(self):
        back = closing_segment(L_SHAPE)
        assert point_at(back, 0.0) == pytest.approx((10.0, 10.0))
        assert point_at(back, 1.0) == pytest.approx((0.0, 0.0))

    def test_a_path_that_ends_where_it_starts_gets_none(self):
        assert closing_segment(ELLIPSE) is None

    @pytest.mark.parametrize("closed, round_trip, n", [
        (True, False, 3), (False, False, 2), (True, True, 2), (False, True, 2)])
    def test_only_a_closed_one_way_path_is_extended(self, closed, round_trip, n):
        assert len(path_segments(L_SHAPE, closed, round_trip)) == n


def mote(path, speed, closed=False, round_trip=False, time_step=1.0):
    return {"name": "m", "sourceCode": "node.c", "functionPath": path, "speed": speed,
            "timeStep": time_step, "isClosed": closed, "isRoundTrip": round_trip}


def trajectories(tmp_path, motes, duration_sec):
    out = tmp_path / "pos.dat"
    generate_positions_from_json({"fixedMotes": [], "mobileMotes": motes}, str(out),
                                 duration_sec=duration_sec)
    rows: dict[int, list[tuple[float, float, float]]] = {}
    for line in out.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and not line.startswith("#"):
            rows.setdefault(int(parts[0]), []).append(tuple(map(float, parts[1:])))
    return rows


class TestPositionsFile:
    def test_every_mote_moves_until_the_end_of_the_simulation(self, tmp_path):
        # Periods of 10 and 31 steps: no common cycle within the 60 s.
        rows = trajectories(tmp_path, [mote(LINE, 1.0), mote([("31 * t", "5")], 1.0)], 60.0)

        for samples in rows.values():
            assert samples[-1][0] == pytest.approx(60.0)
        short = rows[0]
        period = len([s for s in short if s[0] < 10.0])
        # The short mote starts its path again instead of standing at its end.
        assert short[period][1:] == short[0][1:]
        assert len({s[1] for s in short[: 3 * period]}) > period // 2

    def test_a_closed_path_comes_back_to_its_start(self, tmp_path):
        rows = trajectories(tmp_path, [mote(L_SHAPE, 1.0, closed=True)], 40.0)[0]
        x_max = max(r[1] for r in rows)
        # Out along the L and back along the diagonal: it passes near (5, 5).
        assert x_max == pytest.approx(10.0, abs=0.5)
        assert any(math.dist(r[1:], (5.0, 5.0)) < 1.0 for r in rows)

    def test_without_a_duration_the_file_holds_one_pass(self, tmp_path):
        rows = trajectories(tmp_path, [mote(LINE, 1.0)], None)[0]
        assert rows[-1][0] < 11.0


def test_commensurate_periods_give_one_common_cycle(tmp_path):
    # A single mote: its own period is the cycle, plus its first point again.
    rows = trajectories(tmp_path, [mote(LINE, 1.0)], 3600.0)[0]
    assert len(rows) < 30 and rows[-1][1:] == rows[0][1:]


def test_two_motes_share_the_least_common_multiple_of_their_periods(tmp_path):
    rows = trajectories(tmp_path, [mote(LINE, 1.0), mote([("30 * t", "5")], 1.0)], 3600.0)
    # Periods of 10 and 30 steps: 30 steps and the first point again, for both.
    assert {len(r) for r in rows.values()} == {31}
    assert all(r[-1][1:] == r[0][1:] for r in rows.values())
