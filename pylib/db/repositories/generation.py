import logging
from datetime import datetime
from typing import Callable, Optional, TypedDict
from bson import ObjectId, errors
from pymongo import ReturnDocument

from pylib.db.models.generation import Generation
from pylib.db.models.enums import EnumStatus
from pylib.db.connection import MongoDBConnection
from pylib.timeutil import as_aware, utc_now

log = logging.getLogger(__name__)


#: An evaluation whose objectives rest on every replica that was planned, and
#: on every metric those replicas were supposed to carry.
STATE_COMPLETE = "complete"
#: Some replicas finished and some did not. The objectives *can* be computed;
#: they are a mean over fewer runs than the design called for, and saying so
#: is the whole point of this record.
STATE_PARTIAL = "partial"
#: Replicas are still in flight. Not a verdict, a timestamp.
STATE_PENDING = "pending"
#: Nothing finished. There is no mean to take.
STATE_INVALID = "invalid"
#: The generation never recorded what it planned, so completeness cannot be
#: decided — only observed. Experiments created before this existed are in
#: this state for ever, and claiming they were complete would be a guess.
STATE_UNKNOWN = "unknown"


class ReplicaCompleteness(TypedDict):
    """What stands behind one individual's objectives.

    ``expected`` is what the generation *planned* — one simulation per
    configured seed — and ``observed`` how many of those seeds produced a
    result. They differ when a replica failed, and the difference used to be
    invisible: the objective was the mean of whatever survived.

    Counted by **seed**, not by document. A replica that was repeated has two
    documents for one seed, and counting documents would make a retried
    individual look over-replicated and let one seed contribute twice.
    """
    expected: int
    observed: int
    #: Seeds whose simulation ended without a result. These have something to
    #: run again.
    failed: int
    #: Seeds whose simulation exists and has not finished.
    pending: int
    #: Planned seeds with no simulation at all — a creation that failed, or a
    #: generation still being enqueued. Nothing arrives on its own and there
    #: is nothing to repeat.
    absent: int
    expected_seeds: list[int]
    observed_seeds: list[int]
    missing_seeds: list[int]
    failed_seeds: list[int]
    #: The subset of ``missing_seeds`` with no simulation at all. Listed and
    #: not only counted because it is the one hole nothing can repair: the
    #: repeat path clones a failed document, and here there is none. An
    #: operator reading a ``partial`` evaluation needs to know which of its
    #: holes are waiting on a decision rather than on a run.
    absent_seeds: list[int]
    #: Metrics carried by fewer seeds than were planned. Without this a
    #: finished generation could read ``partial`` with ``missing_seeds: []``
    #: and nothing explaining it — every planned replica reached ``Done``,
    #: and one of them came back without this metric.
    thin_metrics: list[str]
    #: Why the state is what it is, in one phrase. Derived, never a decision:
    #: it names the hole so that the policy's destination for it is legible
    #: in the document itself.
    reason: str
    #: Per metric, because a run can finish without carrying every one of
    #: them — two objectives of the same individual then rest on different
    #: numbers of replicas.
    per_metric: dict[str, int]
    #: How many simulation documents exist, which is how a repeat shows up.
    attempts: int
    #: One of the ``STATE_*`` constants above. A description, not a decision:
    #: what to *do* about a partial evaluation is a question about the
    #: experiment, and it is answered by the engine's policy.
    state: str


def _ended_after(candidate: dict, kept: dict) -> bool:
    """Whether ``candidate`` is the later of two results for one seed.

    Missing end times compare as oldest, so a document without one never
    displaces a document that has one.
    """
    new_end, old_end = candidate.get("end_time"), kept.get("end_time")
    if new_end is None:
        return False
    if old_end is None:
        return True
    try:
        return as_aware(new_end) > as_aware(old_end)
    except TypeError:
        return False


def _reason_for(
    state: str,
    pending_seeds: list[int],
    failed_seeds: list[int],
    absent_seeds: list[int],
    thin_metrics: list[str],
) -> str:
    """Name the hole behind a state, in the order a reader needs it.

    Two states are ambiguous without this. ``partial`` says a mean rests on
    fewer runs than planned but not whether a run can still repair it, and a
    generation whose every planned seed reached ``Done`` can be ``partial``
    purely because one result came back without a metric — with
    ``missing_seeds`` empty, nothing in the record said why.

    The phrases distinguish what the policy can act on (a failed replica has
    a document to clone) from what it cannot (a seed that was never created,
    a metric absent from a finished run). Both of the latter reach the
    exhaustion branch directly, and this is what makes that visible instead
    of inferred.
    """
    if state == STATE_UNKNOWN:
        return "the generation did not record which seeds it planned"
    if state == STATE_PENDING:
        return f"{len(pending_seeds)} replica(s) still running"
    if state == STATE_COMPLETE:
        return "every planned replica finished with every metric"
    parts: list[str] = []
    if failed_seeds:
        parts.append(f"{len(failed_seeds)} replica(s) failed (repeatable)")
    if absent_seeds:
        parts.append(
            f"{len(absent_seeds)} planned replica(s) were never created "
            "(nothing to repeat)")
    if thin_metrics:
        parts.append(
            "finished replicas are missing "
            + ", ".join(thin_metrics)
            + " (nothing to repeat)")
    if not parts:
        # STATE_INVALID with no failures recorded: every planned seed is
        # unaccounted for and none of them left a document behind.
        parts.append("no replica produced a result")
    return "; ".join(parts)


def _completeness_of(
    runs: list[dict], metrics: list[str], planned_seeds: Optional[list[int]]
) -> ReplicaCompleteness:
    """One individual's completeness, counted by seed.

    Pure, so the counting rules can be tested without a collection — and they
    need testing, because "how many replicas does this objective rest on?" has
    three plausible wrong answers: the number of documents, the number of
    finished documents, and the number of finished documents carrying the
    metric. Only the last is right, and only per metric.
    """
    done_seeds: set[int] = set()
    terminal_seeds: set[int] = set()
    active_seeds: set[int] = set()
    seen_seeds: set[int] = set()
    per_metric: dict[str, int] = {}
    metric_seeds: dict[str, set[int]] = {m: set() for m in metrics}

    for run in runs:
        seed = run.get("random_seed")
        seed = int(seed) if seed is not None else None
        if seed is not None:
            seen_seeds.add(seed)
        status = run.get("status")
        if status == EnumStatus.DONE:
            if seed is not None:
                done_seeds.add(seed)
            measured = run.get("network_metrics") or {}
            for metric in metrics:
                if metric in measured and seed is not None:
                    metric_seeds[metric].add(seed)
        elif status in (EnumStatus.ERROR, EnumStatus.CANCELLED):
            if seed is not None:
                terminal_seeds.add(seed)
        elif seed is not None:
            # A seed is in flight if **any** of its documents is non-terminal.
            # Deriving that from "neither done nor terminal" was wrong once a
            # repeat existed: the failed attempt put the seed in
            # `terminal_seeds`, so its waiting clone disappeared from the
            # count and the evaluation read `invalid` while it was running.
            active_seeds.add(seed)

    expected_seeds = planned_seeds if planned_seeds is not None else sorted(seen_seeds)
    expected = len(expected_seeds)
    missing = sorted(set(expected_seeds) - done_seeds)
    # Three reasons a planned seed has no result, and they are not
    # interchangeable:
    #
    # * ``failed``  — it ran and ended without one. There is a document to
    #   run again, which is what makes a repeat possible.
    # * ``pending`` — a document exists and has not finished. Not a verdict.
    # * ``absent``  — no document at all. Either the generation is still being
    #   enqueued, or the creation of that replica failed; nothing will arrive
    #   on its own, and there is nothing to repeat either.
    #
    # Collapsing ``absent`` into ``pending`` made a generation with a replica
    # that was never created look like one that was merely slow, which is the
    # state that waits for ever.
    # `active_seeds` takes precedence over `terminal_seeds`: a seed whose
    # earlier attempt failed and whose repeat is in flight is pending, not
    # failed, and repeating it again would run the same replica twice.
    pending_seeds = sorted(active_seeds - done_seeds)
    failed_seeds = sorted(terminal_seeds - done_seeds - active_seeds)
    absent_seeds = sorted(set(expected_seeds) - seen_seeds)
    for metric in metrics:
        per_metric[metric] = len(metric_seeds[metric])

    thin = sorted(m for m in metrics if per_metric[m] != expected)
    if planned_seeds is None:
        state = STATE_UNKNOWN
    elif pending_seeds:
        state = STATE_PENDING
    elif not done_seeds:
        state = STATE_INVALID
    elif missing or thin:
        state = STATE_PARTIAL
    else:
        state = STATE_COMPLETE
    reason = _reason_for(state, pending_seeds, failed_seeds, absent_seeds, thin)

    return {
        "expected": expected,
        "observed": len(done_seeds),
        "failed": len(failed_seeds),
        "pending": len(pending_seeds),
        "absent": len(absent_seeds),
        "expected_seeds": list(expected_seeds),
        "observed_seeds": sorted(done_seeds),
        "missing_seeds": missing,
        #: The subset of ``missing_seeds`` that has a simulation to run again.
        "failed_seeds": failed_seeds,
        "absent_seeds": absent_seeds,
        "thin_metrics": thin,
        "per_metric": per_metric,
        "attempts": len(runs),
        "state": state,
        "reason": reason,
    }


class GenerationRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection
        with self.connection.connect() as db:
            db["generations"].create_index([("experiment_id", 1)], name="idx_generations_experiment_id")
            db["generations"].create_index([("experiment_id", 1), ("index", 1)], unique=True, name="idx_generations_experiment_index")
            db["generations"].create_index([("status", 1)], name="idx_generations_status")

    def insert(self, gen: Generation) -> ObjectId:
        with self.connection.connect() as db:
            return db["generations"].insert_one(gen).inserted_id

    def get(self, generation_id: str) -> Generation:
        try:
            oid = ObjectId(generation_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", generation_id)
            return None
        with self.connection.connect() as db:
            return db["generations"].find_one({"_id": oid})

    def find_by_experiment(self, experiment_id: ObjectId) -> list[Generation]:
        with self.connection.connect() as db:
            return list(db["generations"].find(
                {"experiment_id": experiment_id},
                sort=[("index", 1)]
            ))

    def find_by_status(self, status: EnumStatus) -> list[Generation]:
        with self.connection.connect() as db:
            return list(db["generations"].find({"status": status}))

    def find_pending(self) -> list[Generation]:
        return self.find_by_status(EnumStatus.WAITING)

    def update(self, generation_id: ObjectId, updates: dict) -> bool:
        with self.connection.connect() as db:
            result = db["generations"].update_one({"_id": generation_id}, {"$set": updates})
            return result.modified_count > 0

    def set_survivors(self, generation_id: ObjectId, individual_ids: list[str]) -> bool:
        """Record the population environmental selection kept at this generation.

        Stored on the generation rather than as a flag on each individual: it is
        one write per generation, it leaves the unique
        ``(generation_id, individual_id)`` index alone, and a survivor carried
        over from an older generation has no document in this generation to flag.
        """
        return self.update(generation_id, {"survivors": list(individual_ids)})

    def mark_waiting(self, generation_id: ObjectId):
        self.update(generation_id, {"status": EnumStatus.WAITING})

    def mark_running(self, generation_id: ObjectId):
        self.update(generation_id, {
            "status": EnumStatus.RUNNING,
            "start_time": utc_now()
        })

    def mark_done(self, generation_id: ObjectId):
        self.update(generation_id, {
            "status": EnumStatus.DONE,
            "end_time": utc_now()
        })

    def mark_error(self, generation_id: ObjectId):
        self.update(generation_id, {
            "status": EnumStatus.ERROR,
            "end_time": utc_now()
        })

    def close_if_finished(
        self, generation_id: ObjectId, *, require_simulations: bool = False
    ) -> Optional[str]:
        """Close a generation once nothing in it is still running.

        ``Done`` when every simulation finished well, ``Error`` when at least
        one did not, and ``None`` when there is still work in progress.
        Returns what it decided so a caller can log or act on it.

        **Idempotent**: it re-reads the generation's simulations every time, so
        repeating it decides the same thing. That is what makes the close
        recoverable — a failure of this write after a simulation was published
        must not fall back on the simulation, so the caller isolates it and a
        periodic pass repeats it (R24).

        This exists as one method because the same five lines were written out
        in four places — the entrypoint, the runner, the recovery pass and the
        synthetic path — and **missing from a fifth**: the stale-reservation
        sweep failed the last active simulation of a generation without ever
        re-evaluating it, so the generation stayed pending for ever and the
        experiment waited on a strategy event that would never come.

        ``require_simulations`` refuses to close one that has none. A
        generation is inserted *before* its simulations, so "no simulations"
        means "not enqueued yet" as often as it means "nothing to wait for" —
        and only the caller knows which. The engine's all-cached path means
        the second and passes the default; a periodic reconciliation cannot
        know, and passes true.
        """
        if self.any_simulation_active(generation_id):
            return None
        if require_simulations and not self.count_simulations(generation_id):
            return None
        delivered = self.all_planned_replicas_done(generation_id)
        if delivered if delivered is not None else self.all_simulations_done(
                generation_id):
            self.mark_done(generation_id)
            return EnumStatus.DONE
        self.mark_error(generation_id)
        return EnumStatus.ERROR

    def count_simulations(self, generation_id: ObjectId) -> int:
        with self.connection.connect() as db:
            return db["simulations"].count_documents(
                {"generation_id": generation_id})

    def find_open_before(self, cutoff: datetime) -> list[Generation]:
        """Generations still open, old enough that nothing is mid-enqueue.

        The input to the periodic close. The grace is what keeps it off a
        generation whose simulations are still being created: the documents go
        in one at a time, and a fast worker can finish the first before the
        second exists — so "one simulation, none active, all done" is a state
        an enqueue in progress passes through.
        """
        with self.connection.connect() as db:
            return list(db["generations"].find({
                "status": {"$in": [EnumStatus.WAITING, EnumStatus.RUNNING]},
                "start_time": {"$lt": cutoff},
            }))

    def record_plan(
        self, generation_id: ObjectId, expected_seeds: list[int]
    ) -> bool:
        """Write down what this generation intends to run.

        Completeness has to be measured against the plan, not against the
        documents that exist: a generation whose second simulation failed to
        be created looked complete with one replica, because one replica was
        all there was to count.
        """
        return self.update(
            generation_id, {"expected_seeds": [int(s) for s in expected_seeds]})

    def claim_retry_round(
        self, generation_id: ObjectId, max_retries: int,
        replicas: list[dict],
    ) -> Optional[int]:
        """Atomically reserve the budget AND the work needed to use it.

        An interrupted enqueue resumes this manifest before considering a new
        round, even if some of its clones have already started or finished.
        An empty plan must not consume a round.
        """
        if not replicas or max_retries <= 0:
            return None
        with self.connection.connect() as db:
            doc = db["generations"].find_one_and_update(
                {
                    "_id": generation_id,
                    "status": {"$ne": EnumStatus.CANCELLED},
                    "retry_plan": {"$exists": False},
                    "$or": [
                        {"replica_retries": {"$lt": int(max_retries)}},
                        {"replica_retries": {"$exists": False}},
                    ],
                },
                {"$inc": {"replica_retries": 1},
                 "$set": {"retry_plan": replicas}},
                return_document=ReturnDocument.AFTER,
                projection={"replica_retries": 1},
            )
            if doc is None:
                return None
            return int(doc.get("replica_retries") or 0)

    def pending_retry_round(self, generation_id: ObjectId) -> Optional[dict]:
        """Return the unfinished enqueue, including its already claimed round."""
        with self.connection.connect() as db:
            return db["generations"].find_one({
                "_id": generation_id, "retry_plan": {"$exists": True},
                "status": {"$ne": EnumStatus.CANCELLED},
            }, {"retry_plan": 1, "replica_retries": 1})

    def finish_retry_round(
        self, generation_id: ObjectId, round_number: int
    ) -> bool:
        """Reopen and retire the manifest in one guarded write.

        A delayed caller must not clear a newer round's plan or reopen a
        cancelled generation. If this write fails, replaying the same manifest
        finds the existing clones instead of creating them again.
        """
        with self.connection.connect() as db:
            res = db["generations"].update_one(
                {"_id": generation_id, "replica_retries": round_number,
                 "retry_plan": {"$exists": True},
                 "status": {"$ne": EnumStatus.CANCELLED}},
                {"$unset": {"retry_plan": ""},
                 "$set": {"status": EnumStatus.RUNNING, "end_time": None}},
            )
            return res.matched_count > 0

    def retries_used(self, generation_id: ObjectId) -> int:
        with self.connection.connect() as db:
            doc = db["generations"].find_one(
                {"_id": generation_id}, {"replica_retries": 1}) or {}
            return int(doc.get("replica_retries") or 0)

    def all_simulations_done(self, generation_id: ObjectId) -> bool:
        """Returns True if every simulation in this generation has status DONE."""
        with self.connection.connect() as db:
            total = db["simulations"].count_documents({"generation_id": generation_id})
            if total == 0:
                log.warning("Generation %s has no simulations; treating as done.", generation_id)
                return True
            done = db["simulations"].count_documents({
                "generation_id": generation_id,
                "status": EnumStatus.DONE
            })
            return done == total

    def all_planned_replicas_done(
        self, generation_id: ObjectId
    ) -> Optional[bool]:
        """Whether every planned replica produced a result, counted by seed.

        ``None`` when the generation recorded no plan, which is every
        generation created before ``expected_seeds`` existed: there the
        caller keeps the document-based rule, because changing how a
        historical generation reads would rewrite its meaning.

        Why this is not ``all_simulations_done``: once a failed replica can
        be **repeated**, a generation holding a terminal document is no
        longer necessarily a generation that failed. The repeat is a second
        document for the same seed, and the first one stays on the record
        with its message on purpose — so counting documents reported
        ``Error`` for a generation whose every planned seed had a result and
        whose every individual read ``complete``. Observed in a pilot: one
        JVM killed mid-run, the replica repeated and finished, the
        generation still ``Error`` while both of its individuals were
        ``complete`` with three attempts for two seeds.

        Counted by seed, like completeness, so the status and the record
        cannot disagree. Per-metric holes are deliberately *not* part of
        this: a thin metric is reported in the completeness record, and a
        status is too coarse a place to say it.
        """
        with self.connection.connect() as db:
            generation = db["generations"].find_one(
                {"_id": generation_id}, {"expected_seeds": 1}) or {}
            planned = generation.get("expected_seeds")
            if not planned:
                return None
            expected = {int(seed) for seed in planned}
            runs = list(db["simulations"].find(
                {"generation_id": generation_id},
                {"individual_id": 1, "random_seed": 1, "status": 1}))
        if not runs:
            # Same answer as the document rule, and for the same reason: an
            # empty generation is not evidence of a failure. ``close_if_
            # finished`` decides whether it may be closed at all.
            return True
        done: dict[str, set[int]] = {}
        seen: set[str] = set()
        for run in runs:
            individual = str(run.get("individual_id") or "")
            seen.add(individual)
            seed = run.get("random_seed")
            if seed is None or run.get("status") != EnumStatus.DONE:
                continue
            done.setdefault(individual, set()).add(int(seed))
        return all(expected <= done.get(individual, set())
                   for individual in seen)

    def any_simulation_active(self, generation_id: ObjectId) -> bool:
        """Returns True if any simulation is still WAITING or RUNNING."""
        with self.connection.connect() as db:
            count = db["simulations"].count_documents({
                "generation_id": generation_id,
                "status": {"$in": [EnumStatus.WAITING, EnumStatus.RUNNING]}
            })
            return count > 0

    def get_simulations_metrics_by_individual(
        self,
        generation_id: ObjectId | str,
        metrics: list[str],
        aggregator: "str | dict" = "mean",
    ) -> dict[str, dict[str, float]]:
        """
        Returns a map: individual_id_hash → {metric_name → aggregated value}
        Aggregates over all DONE simulations belonging to the same individual
        (multiple seeds per individual).

        aggregator: str or dict following resolve_aggregator() convention.
        Default "mean" preserves historical behaviour.
        """
        if isinstance(generation_id, str):
            try:
                generation_id = ObjectId(generation_id)
            except Exception:
                raise ValueError(f"Invalid generation_id: {generation_id!r}")

        with self.connection.connect() as db:
            sims = list(db["simulations"].find(
                {"generation_id": generation_id, "status": EnumStatus.DONE},
                {"_id": 0, "individual_id": 1, "network_metrics": 1,
                 "random_seed": 1, "end_time": 1}
            ))

        # One result per seed. A replica that was repeated can leave two
        # ``Done`` documents for one seed — the collection of a run that was
        # adopted and then published again, say — and appending both would
        # weigh that seed twice in the mean while the completeness record,
        # which counts seeds, still said two of two. The later result wins:
        # it is the one the repetition was for.
        latest: dict[tuple[str, Any], dict] = {}
        for sim in sims:
            key = (sim.get("individual_id", ""), sim.get("random_seed"))
            if key[1] is None:
                # No seed to deduplicate by — legacy documents. Kept, under a
                # key of their own, so nothing is silently dropped.
                latest[(key[0], id(sim))] = sim
                continue
            kept = latest.get(key)
            if kept is None or _ended_after(sim, kept):
                latest[key] = sim

        # Accumulate per individual
        accumulated: dict[str, dict[str, list[float]]] = {}
        for sim in latest.values():
            ind_id = sim.get("individual_id", "")
            metrics_map = sim.get("network_metrics") or {}
            if ind_id not in accumulated:
                accumulated[ind_id] = {}
            for m in metrics:
                if m in metrics_map:
                    accumulated[ind_id].setdefault(m, []).append(float(metrics_map[m]))

        # Aggregate per seed using Ψa
        from pylib.statistics import resolve_aggregator, AGGREGATOR_DISPATCH
        kind, params = resolve_aggregator(aggregator)
        agg_fn = AGGREGATOR_DISPATCH[kind]

        result: dict[str, dict[str, float]] = {}
        for ind_id, metrics_vals in accumulated.items():
            if kind == "mean" and not params:
                # Preserve original literal expression for bit-identical results.
                result[ind_id] = {m: sum(v) / len(v) for m, v in metrics_vals.items() if v}
            else:
                result[ind_id] = {m: agg_fn(v, params) for m, v in metrics_vals.items() if v}
        return result

    def replica_completeness(
        self,
        generation_id: ObjectId | str,
        metrics: list[str],
        expected_seeds: Optional[list[int]] = None,
    ) -> dict[str, "ReplicaCompleteness"]:
        """How much evidence stands behind each individual's objectives.

        The aggregation above averages whatever finished and says nothing
        about what did not. An individual evaluated from one replicate is then
        indistinguishable from one evaluated from two — and that happened in
        practice: a pilot lost one simulation of a two-seed individual, and
        the objective published for it was a single run's, with no record of
        the difference.

        Reported rather than decided. What to do about an incomplete
        evaluation — accept it, repeat the replica, reject the individual — is
        a question about the experiment, and it is answered by the engine.

        ``expected_seeds`` is **the plan**, read from the generation when the
        caller does not supply it. Deriving it from the documents that happen
        to exist cannot see a replica whose creation failed, so a generation
        that enqueued one simulation instead of two looked complete; and after
        a repeat there are more documents than seeds, which would make the
        same individual look over-replicated. A generation written before this
        was recorded reports :data:`STATE_UNKNOWN` rather than a guess.

        ``per_metric`` is separate from ``observed`` because a run can finish
        and still not carry every metric: two objectives of the *same*
        individual can then rest on different numbers of replicas.
        """
        if isinstance(generation_id, str):
            generation_id = ObjectId(generation_id)
        with self.connection.connect() as db:
            if expected_seeds is None:
                generation = db["generations"].find_one(
                    {"_id": generation_id}, {"expected_seeds": 1}) or {}
                planned = generation.get("expected_seeds")
                expected_seeds = (
                    [int(s) for s in planned] if planned else None)
            sims = list(db["simulations"].find(
                {"generation_id": generation_id},
                {"_id": 0, "individual_id": 1, "status": 1,
                 "random_seed": 1, "network_metrics": 1},
            ))

        planned_seeds = sorted(set(expected_seeds)) if expected_seeds else None
        grouped: dict[str, list[dict]] = {}
        for sim in sims:
            ind_id = str(sim.get("individual_id") or "")
            if ind_id:
                grouped.setdefault(ind_id, []).append(sim)

        out: dict[str, ReplicaCompleteness] = {}
        for ind_id, runs in grouped.items():
            out[ind_id] = _completeness_of(runs, metrics, planned_seeds)
        return out

    def watch_status_waiting(self, on_change: Callable[[dict], None]):
        log.info("[GenerationRepository] Watching for WAITING generations...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": EnumStatus.WAITING
                }
            }
        ]
        self.connection.watch_collection(
            "generations",
            pipeline,
            on_change,
            full_document="updateLookup"
        )

    def watch_status_terminal(self, on_change: Callable[[dict], None]):
        """Watches for generations reaching any terminal state (DONE or ERROR)."""
        log.info("[GenerationRepository] Watching for terminal generations (DONE or ERROR)...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": {"$in": [EnumStatus.DONE, EnumStatus.ERROR]}
                }
            }
        ]
        self.connection.watch_collection(
            "generations",
            pipeline,
            on_change,
            full_document="updateLookup"
        )
