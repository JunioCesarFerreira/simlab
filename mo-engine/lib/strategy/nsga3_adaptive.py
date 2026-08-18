"""NSGA-III with adaptive allocation of the simulation budget.

The evolutionary machinery is *unchanged*: population handling, crossover,
mutation, non-dominated sorting, reference points, niching, persistence and
restart/resume all come from :class:`NSGA3LoopStrategy`.  What this subclass
adds is a decision layer in front of the simulator.

Per generation, in two phases:

**Phase A - screening.**  Each new offspring is classified: an exact genome
cache hit is REUSEd, a hard-constraint violation is penalised as usual, and
everything else goes through :class:`AdaptiveEvaluationPolicy`.  Individuals
whose *optimistic* estimate ``L(x) = f(x) - kappa sigma(x)`` is already clearly
dominated by a really-evaluated solution are marked ESTIMATE_ONLY and enter the
generation with their *conservative* estimate ``U(x) = f(x) + kappa sigma(x)``;
everyone else is simulated.

**Phase B - promotion.**  Once the screening simulations finish, a provisional
NSGA-III selection runs over ``P_{t-1} U P_t``.  Any ESTIMATE_ONLY individual
that lands in the first front or in the survivor set is PROMOTEd: its real
simulation is queued into the same generation, and the definitive selection
only runs once those results are in.

The invariant that follows — and the reason the estimator can never corrupt the
search — is that **every individual selected into the next population carries
simulated or exact-cache objectives**.  Estimated objectives are always tagged
``evaluation_source="estimated"``, never written to the genome cache, never fed
back to the estimator and never published in the final Pareto front.
"""
from __future__ import annotations

import logging
import random
from enum import Enum
from typing import Any, Optional, Sequence

from bson import ObjectId

from pylib.db import EnumStatus

from lib.adaptive.dominance import dominated_by_any
from lib.adaptive.knowledge_base import SENTINEL_THRESHOLD
from lib.adaptive import (
    AdaptiveEvaluationConfig,
    AdaptiveEvaluationPolicy,
    AdaptiveMetricsTracker,
    DecisionReason,
    DecisionRecord,
    EvaluationDecision,
    EvaluationKnowledgeBase,
    EvaluationRecord,
    GenerationAdaptiveMetrics,
    PredictionSample,
)
from lib.nsga import fast_nondominated_sort, niching_selection
from lib.problem.adapter import Chromosome
from lib.problem.chromosomes import chromosome_from_dict

from .nsga3 import NSGA3LoopStrategy

logger = logging.getLogger(__name__)

# Offset applied to the GA seed to derive the policy's RNG.  A separate stream
# keeps audit draws from shifting the crossover/mutation sequence, so a baseline
# and an adaptive run with the same seed explore the same genomes.
_POLICY_SEED_OFFSET = 0x5A17

#: Attributes a problem adapter must expose to drive this strategy.
_REQUIRED_ADAPTER_API = ("descriptors", "scenario_fingerprint", "observe_simulated")


def _evaluation_type_of(
    decision: Optional[DecisionRecord], objectives: Sequence[float]
) -> str:
    """Provenance label of a set of ground-truth objectives.

    A hard-constraint penalty never reaches the policy, so it has no decision
    record; it is recognised by its sentinel magnitude and labelled as such
    rather than being passed off as a measurement.
    """
    if decision is not None and decision.decision == EvaluationDecision.REUSE:
        return "cached"
    if any(abs(float(v)) >= SENTINEL_THRESHOLD for v in objectives):
        return "penalty"
    return "simulated"


class GenerationPhase(str, Enum):
    """Where a generation is inside the two-phase evaluation cycle."""

    SCREENING = "screening"
    PROMOTION = "promotion"


class NSGA3AdaptiveSimulationStrategy(NSGA3LoopStrategy):
    """NSGA-III that decides, per individual, whether a simulation is worth it."""

    def __init__(self, experiment: dict, mongo):
        super().__init__(experiment, mongo)

        params = experiment.get("parameters", {}) or {}
        algorithm_config = params.get("algorithm", {}) or {}
        self._adaptive_config = AdaptiveEvaluationConfig.from_mapping(
            algorithm_config.get("adaptive_evaluation")
            or params.get("adaptive_evaluation")
        )

        self._require_topology_adapter()
        self._scenario_fingerprint: str = self._problem_adapter.scenario_fingerprint(
            self._fingerprint_context(experiment)
        )
        logger.info(
            "[adaptive-eval] Scenario fingerprint %s; warm-up needs %d samples.",
            self._scenario_fingerprint[:12],
            self._adaptive_config.min_training_samples,
        )

        seed = int(algorithm_config.get("random_seed", 42))
        self._policy_rng = random.Random(seed ^ _POLICY_SEED_OFFSET)

        self._knowledge_base = EvaluationKnowledgeBase(self._scenario_fingerprint)
        self._policy = AdaptiveEvaluationPolicy(
            config=self._adaptive_config,
            knowledge_base=self._knowledge_base,
            rng=self._policy_rng,
        )
        self._metrics_tracker = AdaptiveMetricsTracker()

        # --- per-generation adaptive state ---
        self._phase: GenerationPhase = GenerationPhase.SCREENING
        self._gen_decisions: dict[str, DecisionRecord] = {}
        self._gen_metrics: Optional[GenerationAdaptiveMetrics] = None
        self._gen_samples: list[PredictionSample] = []
        self._estimated_hashes: set[str] = set()

    @property
    def _current_gen_index(self) -> int:
        """Index of the generation currently being evaluated.

        Derived from ``_gen_index`` (which the base class advances when a
        generation document is created, and restores on resume) so it stays
        correct after a restart, when no enqueue has run in this process.
        """
        return max(0, self._gen_index - 1)

    def _ensure_generation_metrics(self) -> GenerationAdaptiveMetrics:
        """Metrics bucket of the current generation, creating it if needed.

        A resumed run re-enters an already-enqueued generation without having
        gone through ``_generation_enqueue``, so the bucket may not exist yet.
        """
        if self._gen_metrics is None or self._gen_metrics.generation != self._current_gen_index:
            self._gen_metrics = self._metrics_tracker.start_generation(self._current_gen_index)
        return self._gen_metrics

    # ------------------------------------------------------------------
    # Wiring / validation
    # ------------------------------------------------------------------
    def _require_topology_adapter(self) -> None:
        """Fail fast when the problem cannot describe its individuals."""
        missing = [
            name for name in _REQUIRED_ADAPTER_API
            if not callable(getattr(self._problem_adapter, name, None))
        ]
        if missing:
            raise ValueError(
                f"{type(self._problem_adapter).__name__} cannot drive "
                f"nsga3_adaptive_simulation: missing {', '.join(missing)}. "
                "Use a topology-aware problem (e.g. 'problem2_topology_aware')."
            )

    def _fingerprint_context(self, experiment: dict) -> dict[str, Any]:
        """Everything outside the geometry that changes an objective's meaning."""
        params = experiment.get("parameters", {}) or {}
        simulation_config = params.get("simulation", {}) or {}
        return {
            "problem_name": self._problem_name,
            "objectives": [
                {"metric_name": k, "goal": "min" if g == 1 else "max"}
                for k, g in zip(self._objective_keys, self._objective_goals)
            ],
            "transform_config": experiment.get("data_conversion_config", {}) or {},
            "aggregator": self._aggregator,
            "duration": self._sim_duration,
            "random_seeds": list(self._sim_rand_seeds),
            "synthetic": simulation_config.get("synthetic", {}) or {},
            "source_repository_options": sorted(
                str(v) for v in (experiment.get("source_repository_options") or {}).values()
            ),
        }

    # ------------------------------------------------------------------
    # Knowledge base bootstrap / resume
    # ------------------------------------------------------------------
    def _load_genome_cache_from_db(self) -> None:
        """Restore the genome cache, then rebuild the derived adaptive state."""
        super()._load_genome_cache_from_db()
        self._rebuild_knowledge_base()
        self._restore_decisions()

    def _rebuild_knowledge_base(self) -> None:
        """Rebuild ``D`` from the genome cache — no second source of truth.

        Descriptors are recomputed from the chromosome (they are deterministic
        for a fixed scenario), and every rebuilt individual also feeds the
        routing-importance matrix, so a resumed run reasons exactly like an
        uninterrupted one.
        """
        assert self._exp_id is not None
        try:
            entries = self.mongo.genome_cache_repo.get_all_full_by_experiment(self._exp_id)
        except Exception:
            logger.exception("[adaptive-eval] Could not read the genome cache; starting cold.")
            return

        restored = 0
        for entry in entries or []:
            objectives = entry.get("objectives")
            chromosome_dict = entry.get("chromosome")
            genome_hash = str(entry.get("genome_hash", ""))
            if not objectives or not chromosome_dict or not genome_hash:
                continue
            try:
                genome = chromosome_from_dict(self._problem_name, chromosome_dict)
                descriptors = self._problem_adapter.descriptors(genome)
            except Exception:
                logger.exception("[adaptive-eval] Skipping unreadable cache entry %s.", genome_hash)
                continue

            record = EvaluationRecord(
                scenario_fingerprint=self._scenario_fingerprint,
                chromosome_hash=genome_hash,
                chromosome=dict(chromosome_dict),
                descriptors=descriptors.as_dict(),
                descriptor_vector=tuple(float(v) for v in descriptors.vector()),
                objectives=tuple(float(v) for v in objectives),
                evaluation_type=str(entry.get("evaluation_source") or "cached"),
            )
            if self._knowledge_base.add(record):
                restored += 1
                if record.is_measurement:
                    self._problem_adapter.observe_simulated(genome, key=genome_hash)

        if restored:
            logger.info(
                "[adaptive-eval] Knowledge base restored: %d records (%d usable for training).",
                restored, self._knowledge_base.training_size,
            )
        self._policy.refit(force=True)

    def _restore_decisions(self) -> None:
        """Re-read the persisted decision log so nothing is decided twice."""
        assert self._exp_id is not None
        try:
            documents = self.mongo.adaptive_repo.find_by_experiment(self._exp_id)
        except Exception:
            logger.exception("[adaptive-eval] Could not read the decision log; continuing.")
            return

        for doc in documents or []:
            # An individual is still "only estimated" when no ground truth was
            # ever attached to its decision.
            if doc.get("decision") == EvaluationDecision.ESTIMATE_ONLY.value and not doc.get(
                "actual_objectives"
            ):
                self._estimated_hashes.add(str(doc.get("individual_id", "")))
            else:
                self._estimated_hashes.discard(str(doc.get("individual_id", "")))
        if documents:
            logger.info(
                "[adaptive-eval] Decision log restored: %d entries, %d still estimated.",
                len(documents), len(self._estimated_hashes),
            )

    # ------------------------------------------------------------------
    # Phase A - screening
    # ------------------------------------------------------------------
    def _generation_enqueue(self) -> None:
        assert self._exp_id is not None
        exp_oid = self._exp_id
        population = self._current_population
        self._sim_done_count = 0

        gen_oid, gen_index = self._create_generation_document()
        self._phase = GenerationPhase.SCREENING
        self._gen_decisions = {}
        self._gen_samples = []
        self._gen_metrics = self._metrics_tracker.start_generation(gen_index)

        first_seed = self._sim_rand_seeds[0] if self._sim_rand_seeds else 123456
        self._count_sims_inserted = 0
        sims_inserted = 0
        seen: set[str] = set()
        pending: list[tuple[int, Chromosome, str]] = []

        # --- triage: cache hits and hard-constraint violations never reach the policy
        for idx, genome in enumerate(population):
            genome_hash = genome.get_hash()
            self._gen_metrics.generated_individuals += 1
            if genome_hash in seen:
                logger.info(
                    "Genome %s already present in generation %d; skipping duplicate individual.",
                    genome_hash, gen_index,
                )
                continue
            seen.add(genome_hash)
            self._gen_metrics.unique_individuals += 1

            if genome_hash in self._genome_objectives_cache:
                self._handle_cache_hit(genome, genome_hash, exp_oid, gen_oid, gen_index, idx, first_seed)
                continue

            if genome_hash in self._inserted_genomes:
                logger.info("Genome %s already inserted this SESSION; skipping.", genome_hash)
                continue

            penalty = self._problem_adapter.penalty_objectives(genome, len(self._objective_keys))
            if penalty is not None:
                self._handle_penalised(
                    genome, genome_hash, penalty, exp_oid, gen_oid, gen_index, idx, first_seed
                )
                continue

            pending.append((idx, genome, genome_hash))

        # --- policy: every remaining individual is a real simulation candidate
        self._gen_metrics.baseline_simulations = len(pending)
        decisions = [self._decide(genome, genome_hash, gen_index) for _, genome, genome_hash in pending]
        decisions = self._policy.apply_budget(decisions)

        for (idx, genome, genome_hash), decision in zip(pending, decisions):
            self._gen_decisions[genome_hash] = decision
            sims_inserted += self._apply_decision(
                decision, genome, genome_hash, exp_oid, gen_oid, gen_index, idx, first_seed
            )

        self._gen_metrics.simulation_documents += sims_inserted
        logger.info(
            "[adaptive-eval] generation=%d screened=%d simulate=%d estimate=%d "
            "cache=%d penalised=%d",
            gen_index,
            self._gen_metrics.unique_individuals,
            self._gen_metrics.initial_simulations,
            self._gen_metrics.estimated_only,
            self._gen_metrics.exact_cache_hits,
            self._gen_metrics.penalized_individuals,
        )
        self._close_generation_enqueue(gen_oid, gen_index, sims_inserted, len(population))

    def _decide(self, genome: Chromosome, genome_hash: str, gen_index: int) -> DecisionRecord:
        """Run the policy on one individual (descriptors computed here)."""
        descriptors = self._problem_adapter.descriptors(genome)
        return self._policy.decide(
            individual_id=genome_hash,
            generation=gen_index,
            descriptor_vector=descriptors.vector(),
            mask=self._chromosome_bits(genome),
            descriptors=descriptors.as_dict(),
        )

    def _apply_decision(
        self,
        decision: DecisionRecord,
        genome: Chromosome,
        genome_hash: str,
        exp_oid: ObjectId,
        gen_oid: ObjectId,
        gen_index: int,
        ind_idx: int,
        first_seed: int,
    ) -> int:
        """Materialise one Phase-A decision; returns simulations inserted."""
        assert self._gen_metrics is not None
        logger.info(decision.log_line())

        if decision.decision == EvaluationDecision.ESTIMATE_ONLY:
            conservative = decision.conservative_objectives
            if conservative is None:  # defensive: the policy never emits this
                logger.warning(
                    "[adaptive-eval] ESTIMATE_ONLY without a bound for %s; simulating instead.",
                    genome_hash,
                )
                decision.decision = EvaluationDecision.SIMULATE
                decision.reason = DecisionReason.NO_PREDICTION
            else:
                # Provisional, conservative value: good enough to triage the
                # individual, explicitly flagged so it can never be mistaken
                # for a measurement.
                self._map_genome_objectives[genome] = [float(v) for v in conservative]
                self._estimated_hashes.add(genome_hash)
                decision.evaluation_source = "estimated"
                self._insert_individual_document(
                    exp_oid, gen_oid, genome, genome_hash,
                    objectives_min=[float(v) for v in conservative],
                    evaluation_source="estimated",
                )
                self._upload_topology_for(
                    exp_oid, gen_oid, gen_index, ind_idx, genome, genome_hash, first_seed
                )
                self._gen_metrics.estimated_only += 1
                self._persist_decision(decision)
                return 0

        # SIMULATE (or a demoted ESTIMATE_ONLY that had no usable bound)
        self._estimated_hashes.discard(genome_hash)
        decision.evaluation_source = "simulated"
        self._insert_individual_document(
            exp_oid, gen_oid, genome, genome_hash, evaluation_source="simulated"
        )
        self._inserted_genomes.add(genome_hash)
        self.mongo.genome_cache_repo.insert(exp_oid, genome_hash, genome.to_dict())
        self._gen_metrics.initial_simulations += 1
        if decision.audit_selected:
            self._gen_metrics.audit_simulations += 1
        self._persist_decision(decision)
        return self._enqueue_genome_simulations(
            genome, genome_hash, exp_oid, gen_oid, gen_index, ind_idx, first_seed
        )

    def _handle_cache_hit(
        self,
        genome: Chromosome,
        genome_hash: str,
        exp_oid: ObjectId,
        gen_oid: ObjectId,
        gen_index: int,
        ind_idx: int,
        first_seed: int,
    ) -> None:
        """Exact genome match: replay the stored ground truth, simulate nothing."""
        assert self._gen_metrics is not None
        cached = self._genome_objectives_cache[genome_hash]
        self._map_genome_objectives[genome] = cached
        self._estimated_hashes.discard(genome_hash)
        self._insert_individual_document(
            exp_oid, gen_oid, genome, genome_hash,
            objectives_min=cached, evaluation_source="cache",
        )
        self._upload_topology_for(
            exp_oid, gen_oid, gen_index, ind_idx, genome, genome_hash, first_seed
        )
        self._gen_metrics.exact_cache_hits += 1
        decision = DecisionRecord(
            individual_id=genome_hash,
            generation=gen_index,
            decision=EvaluationDecision.REUSE,
            reason=DecisionReason.EXACT_CACHE_HIT,
            actual_objectives=[float(v) for v in cached],
            evaluation_source="cache",
        )
        self._gen_decisions[genome_hash] = decision
        self._persist_decision(decision)
        logger.info(decision.log_line())

    def _handle_penalised(
        self,
        genome: Chromosome,
        genome_hash: str,
        penalty: list[float],
        exp_oid: ObjectId,
        gen_oid: ObjectId,
        gen_index: int,
        ind_idx: int,
        first_seed: int,
    ) -> None:
        """Hard-constraint violation: same gradient penalty as the base strategy."""
        assert self._gen_metrics is not None
        self._map_genome_objectives[genome] = penalty
        self._estimated_hashes.discard(genome_hash)
        self._insert_individual_document(
            exp_oid, gen_oid, genome, genome_hash,
            objectives_min=penalty, evaluation_source="penalty",
        )
        self._inserted_genomes.add(genome_hash)
        self.mongo.genome_cache_repo.insert(exp_oid, genome_hash, genome.to_dict())
        self.mongo.genome_cache_repo.set_objectives(exp_oid, genome_hash, penalty)
        self._genome_objectives_cache[genome_hash] = penalty
        self._upload_topology_for(
            exp_oid, gen_oid, gen_index, ind_idx, genome, genome_hash, first_seed
        )
        self._gen_metrics.penalized_individuals += 1
        logger.info(
            "Genome %s is infeasible (penalty=%.2e); skipping simulation.", genome_hash, penalty[0]
        )

    # ------------------------------------------------------------------
    # Phase B - promotion and generation completion
    # ------------------------------------------------------------------
    def _handle_generation_done(self, gen_oid: ObjectId) -> None:
        if self._stop_flag or self._generation_id is None:
            return
        if gen_oid != self._generation_id:
            return
        if self.mongo.experiment_repo.is_cancelled(str(self._exp_id)):
            logger.info("[adaptive-eval] Experiment %s cancelled; stopping.", self._exp_id)
            self.stop()
            return
        if self._has_active_simulations(gen_oid):
            logger.info(
                "[adaptive-eval] Generation %s reported terminal while simulations are "
                "still active; deferring to the next event.", gen_oid,
            )
            return

        logger.info("EVENT GENERATION TERMINAL gen_id=%s phase=%s", gen_oid, self._phase.value)

        self._collect_generation_objectives()
        self._ingest_ground_truth()

        if self._phase == GenerationPhase.SCREENING:
            promoted = self._select_promotions()
            if promoted:
                self._enqueue_promotions(promoted, gen_oid)
                return

        self._finalize_generation_metrics()
        self._update_individual_objectives()
        self._evolution()

    def _has_active_simulations(self, gen_oid: ObjectId) -> bool:
        """Guard against a stale DONE event while work is still queued."""
        try:
            return self.mongo.generation_repo.any_simulation_active(gen_oid) is True
        except Exception:
            logger.exception("[adaptive-eval] Could not check for active simulations.")
            return False

    def _ingest_ground_truth(self) -> None:
        """Fold the freshly measured individuals into the knowledge base.

        Only real measurements are ingested — never an estimate — so the
        estimator can never train on its own output.
        """
        for genome in self._current_population:
            genome_hash = genome.get_hash()
            if genome_hash in self._estimated_hashes:
                continue
            objectives = self._map_genome_objectives.get(genome)
            if objectives is None or genome_hash in self._knowledge_base:
                continue

            decision = self._gen_decisions.get(genome_hash)
            evaluation_type = _evaluation_type_of(decision, objectives)
            try:
                descriptors = self._problem_adapter.descriptors(genome)
            except Exception:
                logger.exception("[adaptive-eval] Could not describe genome %s.", genome_hash)
                continue

            record = EvaluationRecord(
                scenario_fingerprint=self._scenario_fingerprint,
                chromosome_hash=genome_hash,
                chromosome=genome.to_dict(),
                descriptors=descriptors.as_dict(),
                descriptor_vector=tuple(float(v) for v in descriptors.vector()),
                objectives=tuple(float(v) for v in objectives),
                evaluation_type=evaluation_type,
                seed_count=len(self._sim_rand_seeds),
                generation=self._current_gen_index,
            )
            if not self._knowledge_base.add(record):
                continue
            if record.is_measurement:
                self._problem_adapter.observe_simulated(genome, key=genome_hash)
            self._persist_ground_truth(genome_hash, objectives, evaluation_type)

            if decision is not None:
                self._score_prediction(decision, objectives)
                decision.actual_objectives = [float(v) for v in objectives]
                decision.evaluation_source = evaluation_type if evaluation_type == "cached" else "simulated"
                self._persist_decision(decision)

        self._policy.refit()

    def _persist_ground_truth(
        self, genome_hash: str, objectives: list[float], evaluation_type: str
    ) -> None:
        """Cache a measurement as soon as it exists, not at generation end.

        A generation that enters a promotion round returns before
        ``_update_individual_objectives`` runs, so persisting here is what makes
        the knowledge base restorable at *any* interruption point — including
        between the screening and the promotion simulations.
        """
        if genome_hash in self._genome_objectives_cache:
            return
        try:
            self.mongo.genome_cache_repo.set_objectives(
                self._exp_id, genome_hash, objectives, evaluation_type
            )
        except Exception:
            logger.warning("genome_cache write failed for %s; continuing.", genome_hash)
            return
        self._genome_objectives_cache[genome_hash] = objectives

    def _score_prediction(self, decision: DecisionRecord, actual: Sequence[float]) -> None:
        """Record the (prediction, ground truth) pair of an audit or promotion."""
        if decision.predicted_objectives is None:
            return
        wanted_skip = decision.dominance_result is True
        became_relevant = not self._is_dominated_by_front(actual)
        sample = PredictionSample(
            individual_id=decision.individual_id,
            predicted=tuple(float(v) for v in decision.predicted_objectives),
            actual=tuple(float(v) for v in actual),
            was_skipped=wanted_skip,
            became_relevant=wanted_skip and became_relevant,
        )
        self._gen_samples.append(sample)
        self._metrics_tracker.add_prediction_sample(sample)

    def _is_dominated_by_front(self, objectives: Sequence[float]) -> bool:
        return dominated_by_any(objectives, self._knowledge_base.known_front())

    # ------------------------------------------------------------------
    def _select_promotions(self) -> list[Chromosome]:
        """Estimated individuals that a provisional NSGA-III would keep.

        They are exactly the ones whose approximation could steer the search,
        so they must be measured for real before the definitive selection runs.
        """
        estimated = [
            genome for genome in self._current_population
            if genome.get_hash() in self._estimated_hashes
        ]
        if not estimated:
            return []

        union = list(self._parents) + list(self._current_population)
        objectives: list[list[float]] = []
        members: list[Chromosome] = []
        for genome in union:
            values = self._map_genome_objectives.get(genome)
            if values is None:
                continue
            members.append(genome)
            objectives.append([float(v) for v in values])
        if not objectives:
            return []

        survivors = self._provisional_survivors(objectives)
        relevant_hashes = {members[i].get_hash() for i in survivors}

        promoted: list[Chromosome] = []
        for genome in estimated:
            if genome.get_hash() in relevant_hashes:
                promoted.append(genome)
        return promoted

    def _provisional_survivors(self, objectives: list[list[float]]) -> set[int]:
        """Indices NSGA-III would keep: full fronts + niching on the split one.

        Mirrors :meth:`NSGA3LoopStrategy._select_next_parents`, but returns
        indices and never finalises the experiment — it is a *what-if* pass.
        The whole first front is always included, so a promising estimate is
        promoted even when niching would have dropped it.
        """
        fronts = fast_nondominated_sort(objectives)
        if not fronts:
            return set()

        selected: list[int] = []
        for front in fronts:
            if len(selected) + len(front) <= self._pop_size:
                selected.extend(front)
            else:
                remaining = self._pop_size - len(selected)
                if remaining > 0:
                    selected.extend(
                        niching_selection(
                            front, objectives, self._ref_points, remaining, self._ga_rng
                        )
                    )
                break
        return set(selected) | set(fronts[0])

    def _enqueue_promotions(self, promoted: Sequence[Chromosome], gen_oid: ObjectId) -> None:
        """Queue the real simulations of the promoted individuals.

        The generation is reopened (status RUNNING) *before* any simulation is
        inserted, so master-node closes it again only once the promotion round
        is complete; that second DONE event drives Phase B to its conclusion.
        """
        assert self._exp_id is not None
        exp_oid = self._exp_id
        gen_index = self._current_gen_index
        metrics = self._ensure_generation_metrics()
        self._phase = GenerationPhase.PROMOTION

        budget = self._adaptive_config.budget
        selection = list(promoted)
        if budget.enabled and budget.promotion_reserve > 0 and len(selection) > budget.promotion_reserve:
            # Keep the reserve meaningful: promote the individuals whose
            # estimate is least trustworthy first.
            selection.sort(
                key=lambda g: -(
                    (self._gen_decisions.get(g.get_hash()) or DecisionRecord(
                        individual_id="", generation=gen_index,
                        decision=EvaluationDecision.ESTIMATE_ONLY,
                        reason=DecisionReason.OPTIMISTIC_DOMINATED,
                    )).normalized_uncertainty or 0.0
                )
            )
            dropped = selection[budget.promotion_reserve:]
            selection = selection[: budget.promotion_reserve]
            logger.warning(
                "[adaptive-eval] Promotion reserve (%d) smaller than the %d relevant "
                "estimates; %d individual(s) stay estimated this generation.",
                budget.promotion_reserve, len(promoted), len(dropped),
            )

        self.mongo.generation_repo.update(gen_oid, {"status": EnumStatus.RUNNING})

        first_seed = self._sim_rand_seeds[0] if self._sim_rand_seeds else 123456
        sims_inserted = 0
        for genome in selection:
            genome_hash = genome.get_hash()
            decision = self._gen_decisions.get(genome_hash) or DecisionRecord(
                individual_id=genome_hash,
                generation=gen_index,
                decision=EvaluationDecision.ESTIMATE_ONLY,
                reason=DecisionReason.OPTIMISTIC_DOMINATED,
            )
            self._gen_decisions[genome_hash] = decision
            decision.decision = EvaluationDecision.PROMOTE_TO_SIMULATION
            decision.reason = DecisionReason.PROVISIONAL_SURVIVOR
            decision.promotion_selected = True
            decision.evaluation_source = "simulated"

            # Drop the provisional value so the objective collector picks up the
            # real metrics once the simulations finish.
            self._map_genome_objectives.pop(genome, None)
            self._estimated_hashes.discard(genome_hash)
            self._inserted_genomes.add(genome_hash)
            self.mongo.individual_repo.update_fields(
                genome_hash, gen_oid, {"objectives": [], "evaluation_source": "simulated"}
            )
            self.mongo.genome_cache_repo.insert(exp_oid, genome_hash, genome.to_dict())
            self._persist_decision(decision)
            logger.info(decision.log_line())

            index = self._population_index(genome)
            sims_inserted += self._enqueue_genome_simulations(
                genome, genome_hash, exp_oid, gen_oid, gen_index, index, first_seed
            )
            metrics.promotions += 1

        metrics.simulation_documents += sims_inserted
        logger.info(
            "[adaptive-eval] generation=%d promotion round: %d individual(s), %d simulation(s).",
            gen_index, len(selection), sims_inserted,
        )
        self._close_generation_enqueue(gen_oid, gen_index, sims_inserted, len(self._current_population))

    def _population_index(self, genome: Chromosome) -> int:
        for i, candidate in enumerate(self._current_population):
            if candidate is genome or candidate == genome:
                return i
        return 0

    # ------------------------------------------------------------------
    def _finalize_generation_metrics(self) -> None:
        """Close the generation's accounting and persist it."""
        metrics_bucket = self._ensure_generation_metrics()
        still_estimated = [
            d for d in self._gen_decisions.values()
            if d.decision == EvaluationDecision.ESTIMATE_ONLY
        ]
        metrics_bucket.estimated_only = len(still_estimated)
        uncertainties = [
            d.normalized_uncertainty for d in self._gen_decisions.values()
            if d.normalized_uncertainty is not None
        ]
        novelties = [
            d.novelty for d in self._gen_decisions.values() if d.novelty is not None
        ]
        metrics = self._metrics_tracker.score_generation(
            metrics_bucket, self._gen_samples, uncertainties, novelties
        )
        logger.info(
            "[adaptive-eval] generation=%d simulated=%d promoted=%d estimated=%d "
            "baseline=%d avoided=%d reduction=%.2f%%",
            metrics.generation,
            metrics.initial_simulations,
            metrics.promotions,
            metrics.estimated_only,
            metrics.baseline_simulations,
            metrics.avoided_simulations,
            100.0 * metrics.simulation_reduction_ratio,
        )
        try:
            self.mongo.adaptive_repo.upsert_metrics(
                self._exp_id, self._scenario_fingerprint, metrics.generation, metrics.to_dict()
            )
        except Exception:
            logger.exception("[adaptive-eval] Could not persist generation metrics.")

    def _persist_decision(self, decision: DecisionRecord) -> None:
        if self._exp_id is None:
            return
        try:
            self.mongo.adaptive_repo.upsert_decision(
                self._exp_id, self._scenario_fingerprint, decision.generation, decision.to_dict()
            )
        except Exception:
            logger.exception("[adaptive-eval] Could not persist decision for %s.", decision.individual_id)

    # ------------------------------------------------------------------
    # Guarantees on the evolutionary state
    # ------------------------------------------------------------------
    def _is_ground_truth(self, genome: Chromosome) -> bool:
        """Estimated objectives never reach the persistent genome cache."""
        return genome.get_hash() not in self._estimated_hashes

    def _select_next_parents(self, R_population: list, R_objectives: "list[list[float]]") -> "list | None":
        """Environmental selection restricted to really-evaluated individuals.

        With ``require_simulated_survivors`` (the default), any individual that
        is still only estimated is excluded from ``R_t`` — promotion already
        measured every estimate that could have survived, so what remains is
        provably outside the survivor set.  The filter is skipped when it would
        starve the selection, which can only happen in a degenerate run.
        """
        if not self._adaptive_config.require_simulated_survivors:
            return super()._select_next_parents(R_population, R_objectives)

        keep = [
            i for i, genome in enumerate(R_population)
            if genome.get_hash() not in self._estimated_hashes
        ]
        if len(keep) < self._pop_size:
            logger.warning(
                "[adaptive-eval] Only %d simulated individuals for a population of %d; "
                "admitting estimated ones this round.", len(keep), self._pop_size,
            )
            return super()._select_next_parents(R_population, R_objectives)

        selected = super()._select_next_parents(
            [R_population[i] for i in keep], [R_objectives[i] for i in keep]
        )
        return selected

    def _final_pareto_front(self) -> list[dict]:
        """Published front — estimated individuals are excluded by construction."""
        estimated = self._estimated_hashes
        original_population = self._current_population
        self._current_population = [
            g for g in original_population if g.get_hash() not in estimated
        ]
        try:
            return super()._final_pareto_front()
        finally:
            self._current_population = original_population

    def _finalize_experiment(
        self,
        system_msg: Optional[str] = None,
        pareto_front: Optional[list[dict]] = None,
    ) -> None:
        summary = self._metrics_tracker.experiment_summary()
        logger.info(
            "[adaptive-eval] Experiment totals: %d simulated of %d baseline "
            "(%d avoided, reduction %.2f%%).",
            summary.get("total_actual_simulations", 0),
            summary.get("baseline_simulations", 0),
            summary.get("avoided_simulations", 0),
            100.0 * float(summary.get("simulation_reduction_ratio", 0.0)),
        )
        try:
            self.mongo.adaptive_repo.upsert_metrics(
                self._exp_id, self._scenario_fingerprint, -1, summary
            )
        except Exception:
            logger.exception("[adaptive-eval] Could not persist the experiment summary.")
        super()._finalize_experiment(system_msg=system_msg, pareto_front=pareto_front)

    # ------------------------------------------------------------------
    @staticmethod
    def _chromosome_bits(genome: Chromosome) -> list[int]:
        """Binary view of a chromosome, used by the Hamming novelty term."""
        mask = getattr(genome, "mask", None)
        if mask is not None:
            return [int(b) for b in mask]
        raw = genome.to_dict().get("mask", [])
        return [int(b) for b in raw]

    # ------------------------------------------------------------------
    @property
    def adaptive_metrics(self) -> AdaptiveMetricsTracker:
        """Exposed for tests and for post-hoc analysis of a finished run."""
        return self._metrics_tracker

    @property
    def knowledge_base(self) -> EvaluationKnowledgeBase:
        return self._knowledge_base
