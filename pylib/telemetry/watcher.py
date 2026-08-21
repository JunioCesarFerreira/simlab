"""Background watcher that triggers telemetry collection on experiment finish.

Runs inside the mo-engine process (the component that owns the experiment
lifecycle). Two complementary triggers:

- a Change Stream on the ``experiments`` collection firing when a document
  transitions to Done/Error without a ``runtime_metrics`` block;
- a *periodic* backfill sweep (also runs once at startup) that reprocesses
  experiments which finished while the watcher was down, or which were left in
  a transient ``unavailable``/``failed`` state (e.g. Prometheus was down at
  finish time and has since come back). Bounded by TELEMETRY_BACKFILL_HOURS so
  long-gone runs, for which Prometheus no longer retains data, are not churned.

Collection waits TELEMETRY_COLLECTION_DELAY_SECONDS after the finish event so
Prometheus scrapes the last samples of the execution window before the range
query runs.
"""
import logging
import os
import time
from datetime import datetime, timedelta
from threading import Thread

from pylib.telemetry.collector import collect_and_store

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _collect_later(mongo, exp_doc: dict, delay: float, force: bool = False) -> None:
    exp_id = str(exp_doc["_id"])
    started_at = exp_doc.get("start_time")
    finished_at = exp_doc.get("end_time")
    if not started_at or not finished_at:
        logger.warning(
            "[telemetry] experiment %s finished without start/end time — skipping", exp_id
        )
        return

    def _do() -> None:
        if delay > 0:
            time.sleep(delay)
        collect_and_store(mongo, exp_id, started_at, finished_at, force=force)

    Thread(target=_do, daemon=True, name=f"telemetry-{exp_id}").start()


def _run_sweep(mongo, backfill_hours: float, stale_claim_seconds: float) -> None:
    """One backfill pass: release stale claims, then (re)collect pending runs."""
    cutoff = datetime.now() - timedelta(hours=backfill_hours)
    claimed_before = datetime.now() - timedelta(seconds=stale_claim_seconds)
    try:
        released = mongo.experiment_repo.release_stale_runtime_metrics_claims(claimed_before)
        if released:
            logger.info("[telemetry] released %d stale 'collecting' claims", released)
        pending = mongo.experiment_repo.find_finished_runtime_metrics_to_collect(cutoff)
    except Exception:
        logger.exception("[telemetry] backfill sweep failed")
        return
    for doc in pending:
        # Fresh runs (no block yet) go through the atomic claim; runs already
        # carrying a retryable block must be overwritten, so force those.
        force = bool(doc.get("runtime_metrics"))
        logger.info(
            "[telemetry] backfilling runtime metrics for %s%s",
            doc["_id"], " (retry)" if force else "",
        )
        _collect_later(mongo, doc, delay=0.0, force=force)


def start_runtime_metrics_watcher(mongo) -> bool:
    """Start the watcher threads. Returns False when disabled via env."""
    if os.getenv("TELEMETRY_ENABLED", "True").lower() in ("false", "0", "no"):
        logger.info("[telemetry] runtime metrics collection disabled (TELEMETRY_ENABLED)")
        return False

    delay = _env_float("TELEMETRY_COLLECTION_DELAY_SECONDS", 30.0)
    backfill_hours = _env_float("TELEMETRY_BACKFILL_HOURS", 6.0)
    sweep_interval = _env_float("TELEMETRY_SWEEP_INTERVAL_SECONDS", 600.0)
    stale_claim_seconds = _env_float("TELEMETRY_STALE_CLAIM_SECONDS", 900.0)

    def _sweep_loop() -> None:
        while True:
            _run_sweep(mongo, backfill_hours, stale_claim_seconds)
            if sweep_interval <= 0:
                return  # single startup pass when periodic sweeping is disabled
            time.sleep(sweep_interval)

    def _on_finish(change: dict) -> None:
        exp_doc = change.get("fullDocument")
        if not exp_doc:
            return
        _collect_later(mongo, exp_doc, delay)

    Thread(target=_sweep_loop, daemon=True, name="telemetry-sweep").start()
    Thread(
        target=mongo.experiment_repo.watch_status_finished,
        args=(_on_finish,),
        daemon=True,
        name="telemetry-watcher",
    ).start()
    logger.info(
        "[telemetry] runtime metrics watcher started "
        "(delay=%.0fs, backfill=%.1fh, sweep=%.0fs)",
        delay, backfill_hours, sweep_interval,
    )
    return True
