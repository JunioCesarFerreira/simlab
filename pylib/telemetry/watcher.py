"""Background watcher that triggers telemetry collection on experiment finish.

Runs inside the mo-engine process (the component that owns the experiment
lifecycle). Two complementary triggers:

- a Change Stream on the ``experiments`` collection firing when a document
  transitions to Done/Error without a ``runtime_metrics`` block;
- a startup sweep for experiments that finished while the watcher was down
  (bounded by TELEMETRY_BACKFILL_HOURS so long-gone runs, for which Prometheus
  no longer retains data, are not churned).

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


def _collect_later(mongo, exp_doc: dict, delay: float) -> None:
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
        collect_and_store(mongo, exp_id, started_at, finished_at)

    Thread(target=_do, daemon=True, name=f"telemetry-{exp_id}").start()


def start_runtime_metrics_watcher(mongo) -> bool:
    """Start the watcher threads. Returns False when disabled via env."""
    if os.getenv("TELEMETRY_ENABLED", "True").lower() in ("false", "0", "no"):
        logger.info("[telemetry] runtime metrics collection disabled (TELEMETRY_ENABLED)")
        return False

    delay = _env_float("TELEMETRY_COLLECTION_DELAY_SECONDS", 30.0)
    backfill_hours = _env_float("TELEMETRY_BACKFILL_HOURS", 6.0)

    def _sweep() -> None:
        cutoff = datetime.now() - timedelta(hours=backfill_hours)
        try:
            released = mongo.experiment_repo.release_stale_runtime_metrics_claims()
            if released:
                logger.info("[telemetry] released %d stale 'collecting' claims", released)
            pending = mongo.experiment_repo.find_finished_missing_runtime_metrics(cutoff)
        except Exception:
            logger.exception("[telemetry] startup sweep failed")
            return
        for doc in pending:
            logger.info("[telemetry] backfilling runtime metrics for %s", doc["_id"])
            _collect_later(mongo, doc, delay=0.0)

    def _on_finish(change: dict) -> None:
        exp_doc = change.get("fullDocument")
        if not exp_doc:
            return
        _collect_later(mongo, exp_doc, delay)

    Thread(target=_sweep, daemon=True, name="telemetry-sweep").start()
    Thread(
        target=mongo.experiment_repo.watch_status_finished,
        args=(_on_finish,),
        daemon=True,
        name="telemetry-watcher",
    ).start()
    logger.info(
        "[telemetry] runtime metrics watcher started (delay=%.0fs, backfill=%.1fh)",
        delay, backfill_hours,
    )
    return True
