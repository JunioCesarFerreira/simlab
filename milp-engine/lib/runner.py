"""
SweepRunner: executes one milp_sweeps document end to end.

Lifecycle (assumes the caller already claimed the sweep via update_starting):
  1. resolve model and solver backend (with optional fallback);
  2. resume from the persisted checkpoint (next_index + known genotypes);
  3. run the sweep, persisting every combination atomically;
  4. on completion, hand the unique topologies off as a batch experiment
     (and link it to the sweep's campaign, when one is set);
  5. finish the sweep as Done / Cancelled / Error.
"""
import dataclasses
import logging
import threading
from typing import Any

from pylib.db import EnumStatus

from .handoff import build_batch_experiment
from .models import get_model
from .solver import resolve_backend
from .sweep import SolveRecord, expand_grid, run_sweep

logger = logging.getLogger(__name__)

# Poll the cancel flag at most once every N records to bound Mongo chatter
_CANCEL_POLL_EVERY = 1


class SweepRunner:
    def __init__(self, sweep_doc: dict[str, Any], mongo):
        self.sweep = sweep_doc
        self.sweep_id = str(sweep_doc["_id"])
        self.mongo = mongo
        self._thread: threading.Thread | None = None

        progress = sweep_doc.get("progress") or {}
        self._progress = {
            "total_combos": int(progress.get("total_combos", 0)),
            "done": int(progress.get("done", 0)),
            "solved": int(progress.get("solved", 0)),
            "infeasible": int(progress.get("infeasible", 0)),
            "unique_genotypes": int(progress.get("unique_genotypes", 0)),
        }
        checkpoint = sweep_doc.get("checkpoint") or {}
        self._start_index = int(checkpoint.get("next_index", 0))
        self._genotypes: list[str] = list(checkpoint.get("genotypes", []))
        self._records_since_poll = 0
        self._cancelled_seen = False

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_safe, daemon=True, name=f"milp-sweep-{self.sweep_id}"
        )
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # ----------------------------------------------------------------- internal

    def _run_safe(self) -> None:
        try:
            self._run()
        except Exception as exc:
            logger.exception("[SweepRunner] Sweep %s failed", self.sweep_id)
            self.mongo.milp_sweep_repo.finish(
                self.sweep_id, EnumStatus.ERROR, message=str(exc)
            )

    def _run(self) -> None:
        model = get_model(self.sweep["model_key"])
        solver_cfg = dict(self.sweep.get("solver") or {})
        backend = resolve_backend(
            solver_cfg.get("backend", "gurobi"),
            allow_fallback=bool(solver_cfg.get("allow_fallback", True)),
        )
        if backend.name != solver_cfg.get("backend"):
            logger.warning(
                "[SweepRunner] Sweep %s: backend '%s' unavailable, using '%s'",
                self.sweep_id, solver_cfg.get("backend"), backend.name,
            )
            self.mongo.milp_sweep_repo.update(
                self.sweep_id, {"solver.effective_backend": backend.name}
            )

        grid = self.sweep.get("parameter_grid") or {}
        fixed = self.sweep.get("fixed_parameters") or {}
        self._progress["total_combos"] = len(expand_grid(grid, fixed))

        result = run_sweep(
            model,
            self.sweep["problem"],
            grid,
            backend,
            fixed_params=fixed,
            time_limit_s=solver_cfg.get("time_limit_s"),
            mip_gap=solver_cfg.get("mip_gap"),
            start_index=self._start_index,
            seen_genotypes=set(self._genotypes),
            on_record=self._on_record,
            should_stop=self._should_stop,
        )

        if result.cancelled:
            self.mongo.milp_sweep_repo.finish(
                self.sweep_id, EnumStatus.CANCELLED, message="Cancelled by user request."
            )
            return

        # Recover masks of genotypes found before a resume (checkpoint only
        # stores the genotype strings — mask == genotype bits by construction).
        all_masks: dict[str, list[int]] = {
            g: [int(bit) for bit in g] for g in self._genotypes
        }
        all_masks.update(result.unique_masks)

        if not all_masks:
            self.mongo.milp_sweep_repo.finish(
                self.sweep_id,
                EnumStatus.DONE,
                message="Sweep finished without feasible topologies; no experiment created.",
            )
            return

        try:
            experiment = build_batch_experiment(self.sweep, all_masks)
        except ValueError as exc:
            self.mongo.milp_sweep_repo.finish(
                self.sweep_id, EnumStatus.ERROR, message=str(exc)
            )
            return

        experiment["status"] = EnumStatus.WAITING
        exp_id = self.mongo.experiment_repo.insert(experiment)
        logger.info(
            "[SweepRunner] Sweep %s handed off %d chromosomes to experiment %s",
            self.sweep_id, len(experiment["parameters"]["chromosomes"]), exp_id,
        )

        campaign_id = self.sweep.get("campaign_id")
        if campaign_id:
            self.mongo.campaign_repo.add_experiment(str(campaign_id), str(exp_id))

        self.mongo.milp_sweep_repo.finish(
            self.sweep_id, EnumStatus.DONE, experiment_id=exp_id
        )

    def _on_record(self, record: SolveRecord) -> None:
        self._progress["done"] += 1
        if record.genotype is not None:
            self._progress["solved"] += 1
            if not record.is_duplicate:
                self._genotypes.append(record.genotype)
                self._progress["unique_genotypes"] += 1
        elif record.status == "INFEASIBLE":
            self._progress["infeasible"] += 1

        self.mongo.milp_sweep_repo.append_solution(
            self.sweep_id,
            dataclasses.asdict(record),
            dict(self._progress),
            {"next_index": record.index + 1, "genotypes": list(self._genotypes)},
        )

    def _should_stop(self) -> bool:
        if self._cancelled_seen:
            return True
        self._records_since_poll += 1
        if self._records_since_poll >= _CANCEL_POLL_EVERY:
            self._records_since_poll = 0
            self._cancelled_seen = self.mongo.milp_sweep_repo.is_cancel_requested(
                self.sweep_id
            )
        return self._cancelled_seen
