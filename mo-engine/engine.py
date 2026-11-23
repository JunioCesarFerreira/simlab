import os, sys, time, logging
from threading import Thread

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib.db import create_mongo_repository_factory, EnumStatus
from pylib.firmware_snapshot import capture_firmware_snapshot
from pylib.telemetry import start_runtime_metrics_watcher
from pylib.timeutil import utc_now
from lib.strategy.base import EngineStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.batch import BatchStrategy
from lib.strategy.random_search import RandomSearchStrategy
from lib.strategy.nsga3_deap import NSGA3DeapStrategy
from lib.strategy.nsga3_pymoo import NSGA3PymooStrategy
from lib.strategy.nsga2_deap import NSGA2DeapStrategy
from lib.strategy.nsga2_pymoo import NSGA2PymooStrategy
from lib.strategy.usl_benchmark import STRATEGY_NAME as USL_BENCHMARK, UslBenchmarkStrategy

# --------------------------- Logging --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [mo-engine] %(message)s",
)
log = logging.getLogger("mo-engine")
# --------------------------------------------------------------------

SimStatus = EnumStatus
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
IS_DOCKER = os.getenv("IS_DOCKER", False)
DB_NAME = os.getenv("DB_NAME", "simlab")

mongo = create_mongo_repository_factory(MONGO_URI, DB_NAME)


def select_strategy(exp_doc: dict) -> EngineStrategy:
    log.info("select strategy")
    exp_type = exp_doc.get("parameters", {}).get("strategy", "simple")
    log.info(f"selected: {exp_type}")
    # When adding new strategies, remember to update the front-end: Step2Experiment.vue
    if exp_type == "nsga3":
        return NSGA3LoopStrategy(exp_doc, mongo)
    if exp_type == "nsga2":
        return NSGA2LoopStrategy(exp_doc, mongo)
    if exp_type == "batch":
        return BatchStrategy(exp_doc, mongo)
    if exp_type == "random_search":
        return RandomSearchStrategy(exp_doc, mongo)
    if exp_type == "nsga3_deap":
        return NSGA3DeapStrategy(exp_doc, mongo)
    if exp_type == "nsga3_pymoo":
        return NSGA3PymooStrategy(exp_doc, mongo)
    if exp_type == "nsga2_deap":
        return NSGA2DeapStrategy(exp_doc, mongo)
    if exp_type == "nsga2_pymoo":
        return NSGA2PymooStrategy(exp_doc, mongo)
    # Launched from the Scalability page, not from the experiment wizard: it
    # measures the infrastructure rather than optimising anything, so it is
    # deliberately absent from Step2Experiment.vue.
    if exp_type == USL_BENCHMARK:
        return UslBenchmarkStrategy(exp_doc, mongo)
    raise ValueError(f"[mo-engine] Experiment type unknown: {exp_type}")


def _record_start_failure(exp_id: str, error: BaseException) -> None:
    """Error, with what went wrong: the status alone left the operator to the
    engine's log to find out why an experiment never ran."""
    mongo.experiment_repo.update_unless_cancelled(exp_id, {
        "status": EnumStatus.ERROR,
        "system_message": f"The strategy could not start: {type(error).__name__}: {error}",
        "end_time": utc_now(),
    })


def process_experiment(exp_doc: dict) -> bool:
    exp_id = str(exp_doc["_id"])
    log.info(f"Processing experiment id: {exp_id}")
    # Freeze the firmware before any simulation is queued: source repositories
    # stay editable, so this copy is the only evidence of what actually ran.
    # Best-effort by design — it records its own failures and never raises.
    capture_firmware_snapshot(mongo, exp_doc)
    strategy = None
    try:
        strategy = select_strategy(exp_doc)
        strategy.start()
        return True
    except Exception as error:
        log.exception("Failed to start strategy for experiment %s", exp_id)
        if strategy is not None:
            # It may have started its watchers, which would outlive it.
            try:
                strategy.stop()
            except Exception:
                log.exception("Failed to stop strategy for experiment %s", exp_id)
        _record_start_failure(exp_id, error)
        return False


def on_experiment_event(change: dict):
    log.info("on experiment event...")
    log.info(f"change: {change}")

    exp_doc = change.get("fullDocument")
    if not exp_doc:
        log.warning("Document missing from the event.")
        return
    exp_id = str(exp_doc["_id"])
        
    if mongo.experiment_repo.update_starting(exp_id):
        process_experiment(exp_doc)


def resume_experiment(exp_doc: dict) -> None:
    """Pick up an experiment that was already running.

    Separate from ``run_pending_experiment`` because the claim is different:
    there is no ``Waiting`` to transition from, and the start time must not be
    rewritten — it is the same run, continuing.
    """
    exp_id = str(exp_doc["_id"])
    if not mongo.experiment_repo.claim_resume(exp_id):
        log.info("Experiment %s is no longer running; not resuming it", exp_id)
        return
    process_experiment(exp_doc)


def run_pending_experiment(change: dict):
    log.info("run pending experiment...")
    log.info(f"change: {change}")

    exp_id = str(change["_id"])
    
    if mongo.experiment_repo.update_starting(exp_id):
        process_experiment(change)
       
       
def start_waiting_experiments() -> None:
    """Start whatever is ``Waiting`` now.

    Runs each time the change stream opens, so what was created while nothing
    was listening is found here and what comes later by the stream. The claim
    is atomic, so one seen by both is started once.
    """
    pending = mongo.experiment_repo.find_startable_by_status(EnumStatus.WAITING)
    while pending:
        run_pending_experiment(pending.pop())


def main() -> None:
    log.info("service started.")
    log.info(f"env:\n\tMONGO_URI: {MONGO_URI}\n\tDB_NAME: {DB_NAME}")
    exp_repo = mongo.experiment_repo
    exp_repo.connection.waiting_ping()

    # Collects Prometheus telemetry for each experiment as it finishes and
    # persists the raw series (GridFS) + summary on the experiment document.
    start_runtime_metrics_watcher(mongo)

    # First the ones a previous process left in flight, and only then the new
    # ones: starting one makes it `Running`, and read afterwards it would be
    # resumed as well, with two strategies driving one experiment.
    #
    # Only `Waiting` used to be picked up, so an experiment interrupted
    # mid-generation stayed `Running` for ever: its strategy state died with
    # the process and the change stream only fires on new ones. Observed in
    # practice — an engine restart left a finished generation with no evolution
    # after it, and the experiment never completed.
    #
    # The strategies already know how to come back: `start()` resumes the
    # latest persisted generation, and processing a generation that is already
    # terminal is a path they handle. The firmware capture is idempotent, so
    # re-entering it costs nothing.
    resumable = exp_repo.find_startable_by_status(EnumStatus.RUNNING)
    if resumable:
        log.info(
            "Resuming %d experiment(s) left running by a previous process",
            len(resumable))
    for exp_doc in resumable:
        resume_experiment(exp_doc)

    # `Waiting` is read only once the stream is listening: read before, an
    # experiment created in between was seen by neither.
    Thread(
        target=exp_repo.watch_status_waiting,
        args=(on_experiment_event, start_waiting_experiments),
        daemon=True).start()

    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
