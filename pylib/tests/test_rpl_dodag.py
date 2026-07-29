"""Tests for pylib.rpl_dodag — retrospective RPL DODAG analysis (Phase 1).

These lock in the first-contact join-time and approximate-depth logic derived
from the root's per-node JSON metrics stream, and the parsing of a raw COOJA
testlog through read_root_records.
"""
import pandas as pd

from pylib.rpl_dodag import (
    analyze_dodag,
    analyze_dodag_exact,
    analyze_dodag_full,
    analyze_log,
    build_dodag_tree,
    count_deployed_nodes,
    first_contact_times,
    load_root_metrics,
    node_depths,
    read_log_end_ms,
    read_rpl_events,
    stable_convergence,
)


def _frame() -> pd.DataFrame:
    # Two nodes, several reports each; root_time_now is the root clock (ms).
    return pd.DataFrame(
        [
            {"node": "fd00::202", "root_time_now": 12000, "hops": 1},
            {"node": "fd00::202", "root_time_now": 22000, "hops": 1},
            {"node": "fd00::202", "root_time_now": 32000, "hops": 2},  # noisy
            {"node": "fd00::203", "root_time_now": 45000, "hops": 2},
            {"node": "fd00::203", "root_time_now": 55000, "hops": 2},
        ]
    )


def test_first_contact_takes_earliest_time_per_node():
    joins = first_contact_times(_frame())
    assert joins == {"fd00::202": 12000.0, "fd00::203": 45000.0}


def test_node_depths_uses_mode():
    # Node .202 reports hops 1,1,2 -> mode 1; .203 reports 2,2 -> 2.
    assert node_depths(_frame()) == {"fd00::202": 1, "fd00::203": 2}


def test_analyze_dodag_summary():
    res = analyze_dodag(_frame(), expected_nodes=2)
    assert res["n_nodes_joined"] == 2
    assert res["all_joined"] is True
    assert res["approx_first_join_ms"] == 12000.0
    assert res["approx_last_join_ms"] == 45000.0
    assert res["approx_formation_time_ms"] == 33000.0
    assert res["approx_max_depth"] == 2
    assert res["approx_mean_depth"] == 1.5
    assert res["per_node"]["fd00::203"] == {"approx_join_ms": 45000.0, "approx_depth": 2}


def test_all_joined_false_when_nodes_missing():
    res = analyze_dodag(_frame(), expected_nodes=5)
    assert res["all_joined"] is False
    assert res["n_nodes_joined"] == 2


def test_empty_frame_is_safe():
    res = analyze_dodag(pd.DataFrame())
    assert res["n_nodes_joined"] == 0
    assert res["approx_formation_time_ms"] is None
    assert res["all_joined"] is True  # no expectation given


def test_analyze_log_reads_raw_testlog(tmp_path):
    log = tmp_path / "COOJA.testlog"
    log.write_text(
        'Initializing simulation script\n'
        '[Mote:2] Not reachable yet\n'
        '[Mote:1] {"node":"fd00::202", "hops":1, "root_time_now":12000}\n'
        '[Mote:2] Sensor IPv6 = fd00::202\n'
        '[Mote:1] {"node":"fd00::203", "hops":2, "root_time_now":45000}\n'
        '[Mote:1] garbage {not json}\n'
        'Final simulation time: 60000 ms\n',
        encoding="utf-8",
    )
    df = load_root_metrics(log)
    assert list(df["node"]) == ["fd00::202", "fd00::203"]

    res = analyze_log(log, expected_nodes=2)
    assert res["n_nodes_joined"] == 2
    assert res["approx_last_join_ms"] == 45000.0


# ── Phase 2/3: exact tree + stable convergence from [RPL] events ──────────────

def _rpl_log(tmp_path):
    # Chain: .203 -> .202 -> root(.201). .203 switches parent once (.204 -> .202).
    log = tmp_path / "COOJA.testlog"
    log.write_text(
        '[t_us=1000000] [Mote:2] [RPL] t=1000 node=fd00::202 parent=fd00::201\n'
        '[t_us=2000000] [Mote:3] [RPL] t=2000 node=fd00::203 parent=fd00::204\n'
        '[t_us=5000000] [Mote:3] [RPL] t=5000 node=fd00::203 parent=fd00::202\n'
        '[t_us=1000000] [Mote:1] {"node":"fd00::202", "hops":1, "root_time_now":9000}\n'
        'Final simulation time: 120000 ms\n',
        encoding="utf-8",
    )
    return log


def test_read_rpl_events_captures_global_and_node_clock(tmp_path):
    events = read_rpl_events(_rpl_log(tmp_path))
    assert len(events) == 3
    assert events[0] == {
        "node": "fd00::202",
        "parent": "fd00::201",
        "node_t": 1000,
        "global_t_us": 1000000,
    }


def test_build_dodag_tree_final_parent_and_depth(tmp_path):
    events = read_rpl_events(_rpl_log(tmp_path))
    tree = build_dodag_tree(events)
    assert tree["root"] == "fd00::201"
    # Final parent of .203 is .202 (last event wins), not the earlier .204.
    assert tree["edges"] == {"fd00::202": "fd00::201", "fd00::203": "fd00::202"}
    assert tree["depth"] == {"fd00::202": 1, "fd00::203": 2}
    assert tree["disconnected"] == []


def test_read_log_end_prefers_final_time(tmp_path):
    assert read_log_end_ms(_rpl_log(tmp_path)) == 120000.0


def test_stable_convergence_last_switch_and_window(tmp_path):
    events = read_rpl_events(_rpl_log(tmp_path))
    res = stable_convergence(
        events, stability_window_ms=60_000, expected_nodes=2,
        observed_end_ms=120_000,
    )
    assert res["uses_global_clock"] is True
    # Last switch at t_us=5_000_000 -> 5000 ms; quiet window 120000-5000 > 60000.
    assert res["convergence_time_ms"] == 5000.0
    assert res["total_parent_switches"] == 3
    assert res["parent_switches_per_node"] == {"fd00::202": 1, "fd00::203": 2}
    assert res["stability_verified"] is True
    assert res["converged"] is True


def test_stable_convergence_not_converged_when_window_too_short(tmp_path):
    events = read_rpl_events(_rpl_log(tmp_path))
    res = stable_convergence(
        events, stability_window_ms=60_000, expected_nodes=2,
        observed_end_ms=40_000,  # only 35s of quiet after last switch
    )
    assert res["converged"] is False


def test_stable_convergence_not_converged_when_nodes_missing(tmp_path):
    events = read_rpl_events(_rpl_log(tmp_path))
    res = stable_convergence(
        events, stability_window_ms=60_000, expected_nodes=5,
        observed_end_ms=120_000,
    )
    assert res["converged"] is False


def test_analyze_dodag_exact_end_to_end(tmp_path):
    out = analyze_dodag_exact(_rpl_log(tmp_path), expected_nodes=2)
    assert out["tree"]["root"] == "fd00::201"
    assert out["convergence"]["convergence_time_ms"] == 5000.0
    assert out["convergence"]["converged"] is True


def test_count_deployed_nodes_excludes_root(tmp_path):
    # Motes 2 and 3 are sensors; Mote 1 is the root and must not be counted.
    assert count_deployed_nodes(_rpl_log(tmp_path)) == 2


def test_analyze_dodag_full_bundles_everything(tmp_path):
    full = analyze_dodag_full(_rpl_log(tmp_path))
    assert full["deployed_nodes"] == 2
    assert full["tree"]["root"] == "fd00::201"
    assert full["convergence"]["converged"] is True
    assert "approx" in full and full["approx"]["n_nodes_joined"] == 1


def test_empty_events_are_safe(tmp_path):
    empty = tmp_path / "e.testlog"
    empty.write_text("nothing here\n", encoding="utf-8")
    assert read_rpl_events(empty) == []
    assert build_dodag_tree([])["root"] is None
    assert stable_convergence([])["converged"] is False
