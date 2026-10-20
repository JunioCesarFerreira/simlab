#!/usr/bin/env python3
"""Backfill RPL DODAG analysis onto already-stored simulations in MongoDB.

For every completed simulation that has a raw Cooja log in GridFS
(``log_cooja_id``), this downloads the log, runs the retrospective DODAG
analysis (:func:`pylib.rpl_dodag.analyze_dodag_full`) and writes the result
back to the simulation document under the ``dodag`` field.

What can be inferred from *legacy* logs (recorded before the firmware was
instrumented with ``[RPL]`` parent-switch events):

* ``dodag.approx`` — per-node first-contact join time, an approximate network
  formation time (~ one send interval, typically 10 s, of resolution) and an
  approximate depth per node (from the root's ``hops`` field);
* ``dodag.deployed_nodes`` — how many sensor motes appear in the log.

What is *not* recoverable from legacy logs (never logged): the exact
parent/child tree and the stable-convergence / parent-switch counts. For those
runs ``dodag.tree.root`` is ``None`` and ``dodag.convergence.converged`` is
False — only newly instrumented simulations carry the exact tree.

Usage:
    python util/backfill_dodag.py                 # dry-run, prints a summary
    python util/backfill_dodag.py --apply         # write results back
    python util/backfill_dodag.py --apply --force # recompute even if present
    MONGO_URI=mongodb://remote:27017/ python util/backfill_dodag.py --apply
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Make the repo root importable so `import pylib...` works when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pylib.db import create_mongo_repository_factory  # noqa: E402
from pylib import rpl_dodag  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mongo-uri",
                   default=os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0"),
                   help="MongoDB connection URI  [env: MONGO_URI]")
    p.add_argument("--db-name", default=os.getenv("DB_NAME", "simlab"),
                   help="MongoDB database name  [env: DB_NAME]")
    p.add_argument("--status", default="Done",
                   help="Only process simulations in this status (default: Done)")
    p.add_argument("--stability-window-ms", type=float, default=60_000.0,
                   help="Quiet window used for the stable-convergence check")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a 'dodag' field is already present")
    p.add_argument("--apply", action="store_true",
                   help="Persist results (default is a dry-run that only prints)")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most N simulations (0 = no limit)")
    return p.parse_args()


def _analyze_log_bytes(data: bytes, stability_window_ms: float) -> dict:
    """Run the DODAG analysis on raw log bytes via a temp file."""
    with tempfile.NamedTemporaryFile("wb", suffix=".log", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return rpl_dodag.analyze_dodag_full(tmp_path, stability_window_ms)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    factory = create_mongo_repository_factory(args.mongo_uri, args.db_name)

    sims = factory.simulation_repo.find_by_status(args.status)
    if args.limit > 0:
        sims = sims[: args.limit]

    print(f"Found {len(sims)} simulation(s) in status '{args.status}'"
          f"  (mode: {'APPLY' if args.apply else 'DRY-RUN'})")

    n_ok = n_skipped = n_nolog = n_err = 0
    for sim in sims:
        sim_id = sim["_id"]
        log_id = sim.get("log_cooja_id")

        if sim.get("dodag") is not None and not args.force:
            n_skipped += 1
            continue
        if not log_id:
            n_nolog += 1
            continue

        try:
            data = factory.fs_handler.read_file_content(str(log_id))
            dodag = _analyze_log_bytes(data, args.stability_window_ms)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [ERR ] {sim_id}: {exc}")
            n_err += 1
            continue

        conv = dodag["convergence"]
        approx = dodag["approx"]
        tree_root = dodag["tree"]["root"]
        if tree_root:  # exact tree from [RPL] events
            joined, time_ms, kind = conv["n_nodes"], conv["convergence_time_ms"], "exact-tree"
        else:          # legacy log: only approximate first-contact formation
            joined = approx["n_nodes_joined"]
            time_ms = approx["approx_formation_time_ms"]
            kind = "approx-only"
        print(f"  [{'DONE' if args.apply else 'PLAN'}] {sim_id}: "
              f"{joined}/{dodag['deployed_nodes']} joined, "
              f"formation~{time_ms} ms, {kind}")

        if args.apply:
            factory.simulation_repo.update(sim_id, {"dodag": dodag})
        n_ok += 1

    print(f"\nSummary: analyzed={n_ok}  skipped(existing)={n_skipped}  "
          f"no-log={n_nolog}  errors={n_err}")
    if not args.apply and n_ok > 0:
        print("Dry-run only — re-run with --apply to persist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
