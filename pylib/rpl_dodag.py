"""Retrospective RPL DODAG analysis from Cooja simulation logs (Phase 1).

The current firmware/Cooja template does **not** timestamp per-node serial lines
(``Not reachable yet`` / ``Sensor IPv6 = ...``), so the only time-stamped signal
available in existing logs is the root's per-node JSON metrics stream. Every such
record carries the reporting node's IPv6 (``node``), the root clock at reception
(``root_time_now``, ms) and the reported hop distance (``hops``).

From that stream we approximate, per node, the *first-contact* time — the first
moment the node successfully reached the root, i.e. right after it joined the
DODAG and sent its first packet. This gives:

* per-node join time (approx., resolution ~ one send interval, typically 10 s);
* an approximate network formation time (last node to make first contact);
* an approximate tree *depth* per node (from ``hops``).

The exact parent/child topology and a *stable-convergence* time (last parent
switch, with a stability window) require firmware instrumentation (Phase 2): the
node firmware emits ``[RPL] t=<node_clock> node=<ip> parent=<ip>`` on every
parent switch, and the Cooja template prefixes each line with a global clock
``[t_us=<sim_time>]``. Those events are parsed here by :func:`read_rpl_events`
and turned into the exact tree (:func:`build_dodag_tree`) and a stable
convergence time (:func:`stable_convergence`). Approximate, first-contact
outputs are deliberately labelled ``approx_*``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from pylib.cooja_builder.csv_converter import read_root_records

NODE_COL = "node"
TIME_COL = "root_time_now"
HOPS_COL = "hops"


def load_root_metrics(log_path: str | Path) -> pd.DataFrame:
    """Load the root JSON metrics stream from a raw COOJA testlog into a frame."""
    df = pd.DataFrame(read_root_records(Path(log_path)))
    if not df.empty and {NODE_COL, TIME_COL} <= set(df.columns):
        df = df.sort_values([NODE_COL, TIME_COL]).reset_index(drop=True)
    return df


def first_contact_times(
    df: pd.DataFrame,
    node_col: str = NODE_COL,
    time_col: str = TIME_COL,
) -> dict[str, float]:
    """Return, per node, the earliest ``time_col`` at which the root heard from it.

    This is the approximate join time of each node (first successful report).
    """
    if df.empty or node_col not in df.columns or time_col not in df.columns:
        return {}
    firsts = df.groupby(node_col)[time_col].min()
    return {str(node): float(t) for node, t in firsts.items()}


def node_depths(
    df: pd.DataFrame,
    node_col: str = NODE_COL,
    hops_col: str = HOPS_COL,
) -> dict[str, int]:
    """Return an approximate tree depth per node, as the most frequent hop count.

    Hop distance from the root's TTL accounting is a noisy proxy for depth; the
    mode across a node's reports is more stable than any single reading.
    """
    if df.empty or node_col not in df.columns or hops_col not in df.columns:
        return {}
    out: dict[str, int] = {}
    for node, group in df.groupby(node_col):
        hops = pd.to_numeric(group[hops_col], errors="coerce").dropna()
        if hops.empty:
            continue
        out[str(node)] = int(hops.mode().iloc[0])
    return out


def analyze_dodag(
    df: pd.DataFrame,
    node_col: str = NODE_COL,
    time_col: str = TIME_COL,
    hops_col: str = HOPS_COL,
    expected_nodes: Optional[int] = None,
) -> dict:
    """Summarize approximate DODAG formation from the root metrics stream.

    Args:
        df: Root metrics frame (see :func:`load_root_metrics`).
        expected_nodes: If given, the number of sensor nodes that *should* join;
            enables an ``all_joined`` completeness flag. When fewer nodes made
            contact than expected, the formation time is only a lower bound.

    Returns a dict of ``approx_*`` metrics plus per-node detail. All times share
    the root clock unit of ``time_col`` (milliseconds).
    """
    joins = first_contact_times(df, node_col, time_col)
    depths = node_depths(df, node_col, hops_col)

    n_joined = len(joins)
    result: dict = {
        "n_nodes_joined": n_joined,
        "expected_nodes": expected_nodes,
        "all_joined": (expected_nodes is None) or (n_joined >= expected_nodes),
        "approx_first_join_ms": None,
        "approx_last_join_ms": None,
        "approx_formation_time_ms": None,
        "approx_max_depth": None,
        "approx_mean_depth": None,
        "per_node": {},
    }

    if joins:
        times = list(joins.values())
        first, last = min(times), max(times)
        result["approx_first_join_ms"] = first
        result["approx_last_join_ms"] = last
        # Formation time relative to the earliest observed join. Root_time_now
        # is ~0 at boot, so `last` alone is also a reasonable estimate; we report
        # the span to be robust to a non-zero clock offset of the first report.
        result["approx_formation_time_ms"] = last - first

    if depths:
        dvals = list(depths.values())
        result["approx_max_depth"] = max(dvals)
        result["approx_mean_depth"] = sum(dvals) / len(dvals)

    for node in sorted(joins):
        result["per_node"][node] = {
            "approx_join_ms": joins[node],
            "approx_depth": depths.get(node),
        }
    return result


def analyze_log(
    log_path: str | Path,
    expected_nodes: Optional[int] = None,
) -> dict:
    """Convenience: read a raw COOJA testlog and return the DODAG summary."""
    df = load_root_metrics(log_path)
    return analyze_dodag(df, expected_nodes=expected_nodes)


# --------------------------------------------------------------------------- #
# Phase 2/3 — exact tree and stable convergence from firmware [RPL] events     #
# --------------------------------------------------------------------------- #

# Optional global-clock prefix injected by the Cooja template: "[t_us=12345] ".
_GLOBAL_T_PATTERN = re.compile(r"\[t_us=(\d+)\]")
# Firmware parent-switch event: "[RPL] t=<clock> node=<ip> parent=<ip>".
_RPL_EVENT_PATTERN = re.compile(
    r"\[RPL\]\s+t=(\d+)\s+node=(\S+)\s+parent=(\S+)"
)


def read_rpl_events(log_path: str | Path) -> list[dict]:
    """Parse firmware parent-switch events from a raw COOJA testlog.

    Each returned event has:

    * ``node`` / ``parent``: link-local IPv6 addresses forming a DODAG edge;
    * ``node_t``: the emitting node's own clock (unit is firmware-specific —
      milliseconds for CSMA, TSCH network uptime ticks for TSCH; not comparable
      across nodes for CSMA);
    * ``global_t_us``: the global simulation time in microseconds if the template
      injected the ``[t_us=...]`` prefix, else ``None``. This is the clock to use
      when comparing events *across* nodes.

    Every emitted event is, by construction, an actual parent change.
    """
    events: list[dict] = []
    with Path(log_path).open(encoding="utf-8") as f:
        for line in f:
            m = _RPL_EVENT_PATTERN.search(line)
            if not m:
                continue
            gt = _GLOBAL_T_PATTERN.search(line)
            events.append(
                {
                    "node": m.group(2),
                    "parent": m.group(3),
                    "node_t": int(m.group(1)),
                    "global_t_us": int(gt.group(1)) if gt else None,
                }
            )
    return events


def _event_time(ev: dict) -> float:
    """Common-clock time for cross-node ordering: global if present, else node."""
    return ev["global_t_us"] if ev["global_t_us"] is not None else ev["node_t"]


def build_dodag_tree(events: list[dict]) -> dict:
    """Reconstruct the final DODAG tree from parent-switch events.

    The final parent of each node is the parent from its last event. Returns the
    edge map, per-node depth (BFS hop distance to the root — the node that never
    appears as a child), the root address, and any nodes left disconnected.
    """
    if not events:
        return {"root": None, "edges": {}, "depth": {}, "disconnected": []}

    # Last parent wins (events are appended in log/time order).
    final_parent: dict[str, str] = {}
    for ev in sorted(events, key=_event_time):
        final_parent[ev["node"]] = ev["parent"]

    children = set(final_parent.keys())
    parents = set(final_parent.values())
    # The root is a parent that is never a child.
    roots = parents - children
    root = sorted(roots)[0] if roots else None

    depth: dict[str, int] = {}
    disconnected: list[str] = []
    for node in final_parent:
        d, cur, seen = 0, node, set()
        while cur in final_parent and cur not in seen:
            seen.add(cur)
            cur = final_parent[cur]
            d += 1
            if cur == root:
                break
        if root is not None and cur == root:
            depth[node] = d
        else:
            disconnected.append(node)

    return {
        "root": root,
        "edges": final_parent,
        "depth": depth,
        "disconnected": sorted(disconnected),
    }


_FINAL_TIME_PATTERN = re.compile(r"Final simulation time:\s*(\d+)\s*ms")


def read_log_end_ms(log_path: str | Path) -> Optional[float]:
    """Return the observed end-of-simulation time in milliseconds, if derivable.

    Prefers the "Final simulation time: N ms" line emitted by the Cooja template;
    falls back to the largest ``[t_us=...]`` global prefix (microseconds → ms).
    Returns ``None`` when neither marker is present.
    """
    end_ms: Optional[float] = None
    max_t_us: Optional[int] = None
    with Path(log_path).open(encoding="utf-8") as f:
        for line in f:
            fm = _FINAL_TIME_PATTERN.search(line)
            if fm:
                end_ms = float(fm.group(1))
            gm = _GLOBAL_T_PATTERN.search(line)
            if gm:
                v = int(gm.group(1))
                if max_t_us is None or v > max_t_us:
                    max_t_us = v
    if end_ms is not None:
        return end_ms
    if max_t_us is not None:
        return max_t_us / 1000.0
    return None


def stable_convergence(
    events: list[dict],
    stability_window_ms: float = 60_000.0,
    expected_nodes: Optional[int] = None,
    observed_end_ms: Optional[float] = None,
) -> dict:
    """Compute the stable-convergence time of the DODAG from parent-switch events.

    Convergence is the instant of the *last* parent switch in the network, after
    which the topology stayed unchanged for at least ``stability_window_ms``. The
    network counts as converged only when every expected node has joined **and**
    that quiet window fully elapsed before the observation ended.

    All times are reported in milliseconds. Cross-node ordering uses the global
    ``[t_us=...]`` clock when every event carries it (the normal case, since the
    firmware ``[RPL]`` events and the template prefix were introduced together);
    otherwise it falls back to per-node clocks, reliable only for TSCH's
    network-synchronized uptime — this is flagged via ``uses_global_clock``.

    Args:
        observed_end_ms: End-of-observation time (see :func:`read_log_end_ms`).
            Required to verify the stability window; without it the window cannot
            be confirmed and ``stability_verified`` is False.
    """
    uses_global = bool(events) and all(e["global_t_us"] is not None for e in events)

    def to_ms(t: float) -> float:
        # Global clock is microseconds; per-node clock is taken as-is (ms for
        # CSMA firmware; TSCH ticks are not converted — see docstring).
        return t / 1000.0 if uses_global else float(t)

    switches_per_node: dict[str, int] = {}
    for ev in events:
        switches_per_node[ev["node"]] = switches_per_node.get(ev["node"], 0) + 1

    n_nodes = len(switches_per_node)
    result: dict = {
        "converged": False,
        "convergence_time_ms": None,
        "n_nodes": n_nodes,
        "expected_nodes": expected_nodes,
        "total_parent_switches": len(events),
        "parent_switches_per_node": switches_per_node,
        "uses_global_clock": uses_global,
        "stability_verified": False,
    }

    if not events:
        return result

    last_switch_ms = to_ms(max(_event_time(e) for e in events))
    result["convergence_time_ms"] = last_switch_ms

    all_joined = (expected_nodes is None) or (n_nodes >= expected_nodes)

    if observed_end_ms is not None:
        result["stability_verified"] = True
        window_elapsed = (observed_end_ms - last_switch_ms) >= stability_window_ms
    else:
        window_elapsed = stability_window_ms == 0

    result["converged"] = bool(all_joined and window_elapsed)
    return result


_MOTE_ID_PATTERN = re.compile(r"\[Mote:(\d+)\]")


def count_deployed_nodes(log_path: str | Path, root_id: int = 1) -> int:
    """Count distinct sensor motes (id != root_id) that appear in the testlog.

    This reflects how many nodes were *deployed* (produced any serial output),
    regardless of whether they joined the DODAG, giving a denominator for the
    ``all_joined`` / convergence completeness checks.
    """
    ids: set[int] = set()
    with Path(log_path).open(encoding="utf-8") as f:
        for line in f:
            m = _MOTE_ID_PATTERN.search(line)
            if m:
                mote_id = int(m.group(1))
                if mote_id != root_id:
                    ids.add(mote_id)
    return len(ids)


def analyze_dodag_exact(
    log_path: str | Path,
    stability_window_ms: float = 60_000.0,
    expected_nodes: Optional[int] = None,
) -> dict:
    """Full Phase 2/3 analysis: exact tree + stable convergence from a testlog."""
    events = read_rpl_events(log_path)
    return {
        "tree": build_dodag_tree(events),
        "convergence": stable_convergence(
            events,
            stability_window_ms,
            expected_nodes,
            observed_end_ms=read_log_end_ms(log_path),
        ),
    }


def analyze_dodag_full(
    log_path: str | Path,
    stability_window_ms: float = 60_000.0,
) -> dict:
    """One-shot DODAG summary for persistence: approximate + exact, self-contained.

    Derives the deployed-node count from the log itself and returns a compact
    dict suitable to store on a simulation document:

    * ``deployed_nodes`` — sensor motes seen in the log;
    * ``approx`` — first-contact based summary (works even on legacy logs);
    * ``tree`` / ``convergence`` — exact results when firmware ``[RPL]`` events
      are present (empty/False otherwise).
    """
    deployed = count_deployed_nodes(log_path)
    expected = deployed or None
    exact = analyze_dodag_exact(log_path, stability_window_ms, expected)
    return {
        "deployed_nodes": deployed,
        "approx": analyze_log(log_path, expected_nodes=expected),
        "tree": exact["tree"],
        "convergence": exact["convergence"],
    }
