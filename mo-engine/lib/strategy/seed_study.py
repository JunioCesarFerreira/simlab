"""A seed study: one simulation per seed, with everything else fixed.

Not an optimisation. A panel of chromosomes runs under the same list of seeds,
and :mod:`pylib.seed_study` turns the results into the distribution of each
metric and the number of seeds each objective needs. Built on the batch
strategy because that is already "one generation, fixed chromosomes, several
seeds", with the replica gate, the recovery and the resume that come with it.

What this adds to a batch, and why:

* **The panel.** Chosen here and written on the experiment before anything
  runs, so a restart reads it back instead of drawing again: regenerating gives
  the same panel only while the problem's generator draws the same numbers in
  the same order, which is a property of code that may change.
* **The chromosome as it runs.** Resolving the firmware can rewrite the MAC gene,
  so the panel keeps the resolved chromosomes. It also compares them after
  resolving: two draws can become one chromosome, and a batch skips a duplicate
  without an error, which would shrink the panel in silence.
* **The control.** After the main generation, the first seed is repeated on
  every chromosome. If the numbers change, something other than the seed adds
  noise and the premise of the study does not hold. The repeats live in a
  generation of their own, because in the main one they would be collapsed:
  reading keeps one result per individual and seed, so that a repetition does
  not count twice. It records no plan, for the same reason.
"""
import logging
from typing import Callable, Optional

from bson import ObjectId

from .batch import BatchStrategy
from .usl_benchmark import FEASIBLE_ATTEMPTS, choose_workload

from pylib.db import EnumStatus
from pylib.db.models import Generation
from pylib.timeutil import utc_now

from lib.problem.adapter import Chromosome, ProblemAdapter
from lib.problem.chromosomes import chromosome_from_dict

logger = logging.getLogger(__name__)

STRATEGY_NAME = "seed_study"
CONTROL_ROLE = "control"
DEFAULT_CONTROL_REPLICATES = 2


def choose_panel(
    adapter: ProblemAdapter, n_objectives: int, size: int,
    resolve: Callable[[Chromosome], Chromosome],
    *, attempts: int = FEASIBLE_ATTEMPTS,
) -> tuple[list[Chromosome], int]:
    """``size`` feasible and distinct chromosomes as they will run, and the draws it took.

    Distinct after ``resolve``, which can make two chromosomes equal. A small
    discrete problem may not have that many: after ``attempts`` duplicates in a
    row the study stops and says how far it got, rather than running a smaller
    panel than the one it was asked for.
    """
    panel: list[Chromosome] = []
    seen: set[str] = set()
    draws = 0
    repeated = 0
    while len(panel) < size:
        chromosome, tries = choose_workload(adapter, None, n_objectives, attempts=attempts)
        draws += tries
        resolved = resolve(chromosome)
        genome_hash = resolved.get_hash()
        if genome_hash in seen:
            repeated += 1
            if repeated >= attempts:
                raise ValueError(
                    f"only {len(panel)} distinct chromosomes of the {size} asked for: "
                    f"{attempts} draws in a row repeated one already in the panel")
            continue
        repeated = 0
        seen.add(genome_hash)
        panel.append(resolved)
    return panel, draws


class SeedStudyStrategy(BatchStrategy):
    """Runs a panel under a list of seeds, then repeats a seed as a control."""

    def __init__(self, experiment: dict, mongo):
        super().__init__(experiment, mongo)
        params = experiment.get("parameters", {}) or {}
        study = params.get("seed_study", {}) or {}
        self._panel_config: dict = study.get("panel", {}) or {}
        self._provided: list[dict] = study.get("chromosomes") or []
        self._control_replicates = int(
            study.get("control_replicates", DEFAULT_CONTROL_REPLICATES))
        # `build_adapter` hands the RNG over only together with GA operator
        # settings, and a study has none. Every adapter reads those settings
        # with defaults, so an empty mapping gives it the RNG and nothing else.
        self._problem_adapter.set_ga_operator_configs(self._rng, {})
        self._control_generation_id: Optional[ObjectId] = None

    def _input_chromosomes(self, experiment: dict) -> list[Chromosome]:
        # Chosen in `start`, which is where the experiment's id is known.
        return []

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        raw_id = self.experiment.get("_id")
        self._exp_id = raw_id if isinstance(raw_id, ObjectId) else ObjectId(str(raw_id))
        generations = self.mongo.generation_repo.find_by_experiment(self._exp_id)
        try:
            self._chromosomes = self._load_or_choose_panel(resume=bool(generations))
        except Exception as error:
            logger.exception("[SeedStudy] Cannot build the panel of %s", self._exp_id)
            super()._finalize_experiment(system_msg=str(error))
            return

        control = next(
            (g for g in generations
             if (g.get("seed_study") or {}).get("role") == CONTROL_ROLE), None)
        if control is None:
            super().start()
            return

        # Resuming with the main generation already over: only the control
        # is left, and the objectives were written before it started.
        self._control_generation_id = control["_id"]
        self._generation_id = control["_id"]
        logger.info("[SeedStudy] Resuming the control of %s", self._exp_id)
        self._start_watcher()
        self._start_generation_poll()
        if control.get("status") in (EnumStatus.DONE, EnumStatus.ERROR):
            with self._lock:
                self._handle_generation_done(control["_id"])

    def _handle_generation_done(self, gen_oid: ObjectId):
        """Core handler — must be called while holding self._lock."""
        if self._control_generation_id is None:
            super()._handle_generation_done(gen_oid)
            return
        if self._stop_flag or gen_oid != self._control_generation_id:
            return
        if self.mongo.experiment_repo.is_cancelled(str(self._exp_id)):
            logger.info("[SeedStudy] %s cancelled; stopping.", self._exp_id)
            self.stop()
            return
        # The first simulations can finish and close the generation while the
        # rest of the control is still being inserted.
        if self.mongo.generation_repo.any_simulation_active(gen_oid):
            return
        self._close_study()

    def _finalize_experiment(self, system_msg: Optional[str] = None,
                             pareto_front: Optional[list[dict]] = None) -> None:
        """The main generation is over: the control comes before the end."""
        if pareto_front is None or self._control_generation_id is not None:
            super()._finalize_experiment(system_msg, pareto_front)
            return
        if not self._count_done(self._generation_id):
            super()._finalize_experiment(
                "no simulation of the study produced a result, so there is nothing "
                "to analyse")
            return
        # Written now because the status only changes after the control, and a
        # restart in between has no other way back to it.
        self.mongo.experiment_repo.update(str(self._exp_id), {"pareto_front": pareto_front})
        if self._control_replicates < 1:
            self._close_study()
            return
        try:
            self._enqueue_control()
        except Exception:
            # The measurements are good without it. The analysis reports that
            # determinism was not checked.
            logger.exception("[SeedStudy] Could not enqueue the control of %s", self._exp_id)
            self._close_study("the control of determinism could not be enqueued")

    def _close_study(self, note: Optional[str] = None) -> None:
        doc = self.mongo.experiment_repo.get(str(self._exp_id)) or {}
        message = f"Seed study completed: {len(self._chromosomes)} chromosome(s), " \
                  f"{len(self._sim_rand_seeds)} seed(s)."
        if note:
            message += f" Note: {note}."
        super()._finalize_experiment(message, doc.get("pareto_front") or [])

    def _count_done(self, gen_oid: ObjectId) -> int:
        with self.mongo.simulation_repo.connection.connect() as db:
            return db["simulations"].count_documents(
                {"generation_id": gen_oid, "status": EnumStatus.DONE})

    # ── the panel ──────────────────────────────────────────────────────────

    def _resolved(self, chromosome: Chromosome) -> Chromosome:
        """The chromosome as it will run. A synthetic run has no firmware to pick."""
        if self._is_synthetic:
            return chromosome
        return chromosome.get_source_by_mac_protocol(self.source_repository_options)[0]

    def _load_or_choose_panel(self, *, resume: bool) -> list[Chromosome]:
        doc = self.mongo.experiment_repo.get(str(self._exp_id)) or {}
        stored = doc.get("seed_study_panel") or {}
        if stored.get("chromosomes"):
            return [self._resolved(chromosome_from_dict(self._problem_name, c))
                    for c in stored["chromosomes"]]
        if resume:
            raise ValueError(
                "this study has generations but no recorded panel; resuming with "
                "different chromosomes would mix two measurements")

        n_objectives = max(1, len(self._objective_keys))
        if self._provided:
            source, draws = "provided", 0
            panel = self._provided_panel(n_objectives)
        else:
            source = "generated"
            size = int(self._panel_config.get("size", 1))
            panel, draws = choose_panel(
                self._problem_adapter, n_objectives, size, self._resolved)
        self.mongo.experiment_repo.update(str(self._exp_id), {"seed_study_panel": {
            "source": source,
            "size": len(panel),
            "draws": draws,
            "random_seed": self._panel_config.get("random_seed"),
            "genome_hashes": [c.get_hash() for c in panel],
            "chromosomes": [c.to_dict() for c in panel],
        }})
        logger.info("[SeedStudy] Panel of %s: %d %s chromosome(s)",
                    self._exp_id, len(panel), source)
        return panel

    def _provided_panel(self, n_objectives: int) -> list[Chromosome]:
        panel: list[Chromosome] = []
        seen: set[str] = set()
        for index, raw in enumerate(self._provided):
            chromosome = self._resolved(chromosome_from_dict(self._problem_name, raw))
            if self._problem_adapter.penalty_objectives(chromosome, n_objectives) is not None:
                raise ValueError(
                    f"chromosome {index} is infeasible, so the adapter would not simulate it")
            genome_hash = chromosome.get_hash()
            if genome_hash in seen:
                raise ValueError(
                    f"chromosome {index} is the same as an earlier one once its "
                    "firmware is resolved")
            seen.add(genome_hash)
            panel.append(chromosome)
        return panel

    # ── the control ────────────────────────────────────────────────────────

    def _enqueue_control(self) -> None:
        """Repeat the first seed on every chromosome, in a generation of its own."""
        exp_oid = self._exp_id
        gen_oid = ObjectId()
        seed = self._sim_rand_seeds[0]
        generation: Generation = {
            "_id": gen_oid,
            "experiment_id": exp_oid,
            "index": 1,
            "status": EnumStatus.WAITING,
            "start_time": utc_now(),
            "end_time": None,
            "seed_study": {"role": CONTROL_ROLE},
        }
        self.mongo.generation_repo.insert(generation)
        self._control_generation_id = gen_oid
        self._generation_id = gen_oid
        self._sim_done_count = 0
        self._count_sims_inserted = len(self._chromosomes) * self._control_replicates

        for index, genome in enumerate(self._chromosomes):
            config = self._convert_genome_to_sim_config(
                genome=genome, gen_index=1, ind_idx=index, seed=seed)
            for replicate in range(self._control_replicates):
                config["name"] = f"control-{index}-{seed}-{replicate}"
                self._insert_simulation_db(genome.get_hash(), exp_oid, gen_oid, config)
        logger.info("[SeedStudy] Control of %s: %d simulation(s) on seed %d",
                    exp_oid, self._count_sims_inserted, seed)
