"""What an objective vector rests on, and what to do when it rests on less.

SimLab evaluates an individual with several simulation seeds and aggregates
what comes back. The aggregation reads the simulations that reached ``Done``,
which means a lost replica is invisible in the result: an individual evaluated
from one run and one evaluated from two produce the same shape of answer, and
the optimiser cannot tell them apart. It happened in a pilot — one simulation
of a two-seed individual was lost, and the objective published for it was a
single run's.

Three things are needed to fix that, and they are different things:

1. **A record.** ``GenerationRepository.replica_completeness`` counts what was
   planned against what was observed, per seed and per metric. Descriptive —
   it decides nothing.
2. **A policy.** Whether an incomplete evaluation may be used at all is a
   question about the experiment, not about the data, so it is configured and
   stated rather than assumed. That is :class:`ReplicaPolicy`.
3. **A repair.** A failed replica can be run again with the same seed, which
   is the only way to turn an incomplete evaluation into a complete one. It
   needs a limit, because a seed that always fails would otherwise keep a
   generation open for ever.

The default is to repeat once and then refuse the individual. That is a
**change in behaviour**: before, a partial mean was used silently. It is the
conservative reading — an under-replicated individual should not compete on
equal terms with a fully evaluated one — and it is configurable for anyone who
wants the previous numbers back.
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

from pylib.db.repositories.generation import (
    STATE_COMPLETE,
    STATE_PENDING,
    STATE_UNKNOWN,
)

logger = logging.getLogger(__name__)

#: Run the missing replicas again, up to ``max_retries`` times per generation.
MODE_REPEAT = "repeat"
#: Do not repeat; apply ``on_exhausted`` straight away.
MODE_NONE = "none"

#: Refuse the individual: it is given the worst objective in every dimension,
#: which is what already happens to an individual with no metrics at all.
EXHAUSTED_WORST = "worst"
#: Use the partial mean, with the completeness recorded alongside it. The
#: previous behaviour, minus the silence.
EXHAUSTED_ACCEPT = "accept"

DEFAULT_MAX_RETRIES = 1


@dataclass(frozen=True)
class ReplicaPolicy:
    """What to do about an evaluation that rests on fewer replicas than planned."""

    mode: str = MODE_REPEAT
    max_retries: int = DEFAULT_MAX_RETRIES
    on_exhausted: str = EXHAUSTED_WORST

    @staticmethod
    def from_config(simulation_config: Optional[dict]) -> "ReplicaPolicy":
        """Read the policy from an experiment's simulation configuration.

        Unknown values fall back to the default *and say so*: a typo in a
        policy name must not silently change what an experiment measures.
        """
        config = simulation_config or {}
        mode = str(config.get("replica_policy") or MODE_REPEAT)
        if mode not in (MODE_REPEAT, MODE_NONE):
            logger.warning(
                "Unknown replica_policy %r; using %r", mode, MODE_REPEAT)
            mode = MODE_REPEAT
        on_exhausted = str(config.get("incomplete_objective") or EXHAUSTED_WORST)
        if on_exhausted not in (EXHAUSTED_WORST, EXHAUSTED_ACCEPT):
            logger.warning(
                "Unknown incomplete_objective %r; using %r",
                on_exhausted, EXHAUSTED_WORST)
            on_exhausted = EXHAUSTED_WORST
        try:
            retries = int(config.get("max_replica_retries", DEFAULT_MAX_RETRIES))
        except (TypeError, ValueError):
            logger.warning(
                "max_replica_retries is not a number; using %d",
                DEFAULT_MAX_RETRIES)
            retries = DEFAULT_MAX_RETRIES
        return ReplicaPolicy(
            mode=mode, max_retries=max(0, retries), on_exhausted=on_exhausted)

    def describe(self) -> str:
        if self.mode == MODE_REPEAT:
            return (
                f"repeat missing replicas up to {self.max_retries} time(s) per "
                f"generation, then {self.on_exhausted}"
            )
        return f"do not repeat; {self.on_exhausted}"


def is_usable(record: Optional[dict], policy: ReplicaPolicy) -> bool:
    """Whether this evaluation may be handed to the algorithm as it is.

    ``None`` — no record at all — is usable: it means the objectives did not
    come from simulations in this generation (a genome whose evaluation was
    reused from the cache, say), and nothing here knows better.

    An evaluation in :data:`STATE_UNKNOWN` is also usable, with a different
    reason: the generation never recorded what it planned, so incompleteness
    cannot be *demonstrated*. Refusing individuals on a suspicion would
    silently invalidate every experiment created before the plan was recorded.

    A **reused** record keeps the decision that was made when the vector was
    produced. The same numbers must not be refused in one generation and
    accepted in the next because the second one had nothing to count.
    """
    if record is None:
        return True
    if record.get("reused"):
        # Carried over, not re-measured: the decision is read rather than
        # re-derived.
        #
        # Defensive, and worth saying so plainly: with the policy fixed for
        # the whole experiment, re-deriving reaches the same answer in every
        # configuration a single experiment can have, so **no test
        # distinguishes this branch from its absence**. It is here because
        # the alternative is a rule that happens to agree; if the policy ever
        # becomes per-generation, agreeing by coincidence stops being enough.
        # The part that *is* observable is in `annotate`, which must not
        # overwrite a recorded decision.
        return bool(record.get("objective_used", True))
    state = record.get("state")
    if state in (STATE_COMPLETE, STATE_UNKNOWN,
                 STATE_ANALYTICAL, STATE_INFEASIBLE):
        return True
    if policy.on_exhausted == EXHAUSTED_ACCEPT:
        # Accepted, but only when there is something to accept.
        return int(record.get("observed") or 0) > 0
    return False


def repeatable(
    completeness: dict[str, dict], policy: ReplicaPolicy, retries_used: int
) -> dict[str, list[int]]:
    """The seeds worth running again, per individual.

    Empty when the policy says not to repeat, when the limit is spent, or
    when nothing *failed*. Two exclusions, for different reasons: a
    ``pending`` evaluation has replicas still running, and an ``absent`` seed
    has no simulation to repeat — only a seed whose run ended without a
    result can be run again.
    """
    if policy.mode != MODE_REPEAT or retries_used >= policy.max_retries:
        return {}
    out: dict[str, list[int]] = {}
    for individual_id, record in completeness.items():
        if record.get("state") == STATE_PENDING:
            continue
        # ``failed_seeds``, not ``missing_seeds``: a planned seed with no
        # simulation at all has nothing to run again, and reopening the
        # generation for it would leave one nothing will ever close.
        failed = [int(s) for s in record.get("failed_seeds") or []]
        if failed:
            out[individual_id] = failed
    return out


def summarize(completeness: dict[str, dict]) -> str:
    """One line for the log: how many evaluations rest on what."""
    if not completeness:
        return "no evaluations"
    counts: dict[str, int] = {}
    for record in completeness.values():
        state = str(record.get("state") or STATE_UNKNOWN)
        counts[state] = counts.get(state, 0) + 1
    return ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))


#: An objective vector carried over from an earlier evaluation of the same
#: genome. Not a state of the replicas — the state of the replicas is whatever
#: the original evaluation recorded, which is exactly what this preserves.
SOURCE_CACHE = "genome_cache"
#: Evaluated in-process, with no simulations at all (an analytical problem).
STATE_ANALYTICAL = "analytical"
#: Rejected before simulating: the individual violates a constraint.
STATE_INFEASIBLE = "infeasible"


def reused(record: Optional[dict]) -> dict:
    """The evidence of an evaluation being reused, kept rather than re-derived.

    A genome that reappears in a later generation has no simulations *there*,
    so nothing computes a completeness for it. The record was therefore absent
    and the individual was written as ``unknown`` with ``objective_used``
    true — even when the original evaluation was a partial mean the policy had
    **refused**. The evidence was known and then thrown away, and the
    displayed meaning changed with it.

    Measured in a 344-simulation pilot: 38 of 210 individuals reported
    ``unknown`` purely through reuse.

    What this returns is the original record, marked as carried over. The
    decision travels with it, because the vector is the same vector: nothing
    about it was re-measured, so nothing about it should be re-decided.
    """
    if not record:
        return {"state": STATE_UNKNOWN, "source": SOURCE_CACHE, "reused": True}
    carried = dict(record)
    carried["source"] = SOURCE_CACHE
    carried["reused"] = True
    return carried


def without_simulations(state: str, reason: str) -> dict:
    """The record for an evaluation that never had replicas to count.

    An analytical problem and an individual rejected by a constraint both
    produce an objective vector with no simulation behind it. ``unknown`` is
    the wrong word for either: it means "the plan was not recorded", and here
    there was no plan to record.
    """
    return {"state": state, "reason": reason, "expected": 0, "observed": 0}


def annotate(record: Optional[dict], *, usable: bool, policy: ReplicaPolicy) -> dict:
    """The completeness record as it is stored on the individual.

    Carries the decision next to the evidence, so a reader of the document
    does not have to know the policy to understand why an objective looks the
    way it does.
    """
    base: dict[str, Any] = dict(record or {"state": STATE_UNKNOWN})
    # A carried-over record already holds the decision that produced its
    # vector; overwriting it with this generation's verdict is what changed
    # the meaning of the evidence.
    base.setdefault("objective_used", bool(usable))
    if not base.get("reused"):
        base["objective_used"] = bool(usable)
    base["policy"] = policy.describe()
    return base


@dataclass
class GateResult:
    """What the gate decided about one generation.

    ``proceed`` is the only thing a strategy has to act on: false means stop
    here, because either replicas are still running or a repeat has just been
    queued and the generation will close again.
    """
    proceed: bool
    completeness: dict[str, dict]
    repeated: int = 0
    waiting: bool = False


class ReplicaGate:
    """Stands between a closed generation and the objectives drawn from it.

    Composed rather than inherited. It needs the repositories, the metric
    names and the policy, and nothing else about the strategy that uses it —
    so the four strategies pass what it needs instead of sharing a base class
    that reaches into their internals.
    """

    def __init__(self, mongo: Any, *, objective_keys: list[str],
                 policy: ReplicaPolicy, label: str = "engine"):
        self._mongo = mongo
        self._keys = list(objective_keys)
        self._policy = policy
        self._label = label

    @property
    def policy(self) -> ReplicaPolicy:
        return self._policy

    def record_plan(self, generation_id: Any, seeds: list[int]) -> None:
        """Write down the seeds this generation intends to run.

        Completeness is measured against this. Derived from the documents
        instead, it cannot see a replica whose *creation* failed — a
        generation that enqueued one simulation where two were configured
        looked complete, because one was all there was to count.
        """
        try:
            self._mongo.generation_repo.record_plan(generation_id, seeds)
        except Exception:
            logger.exception(
                "[%s] Could not record the replica plan of generation %s; "
                "its completeness will read as unknown",
                self._label, generation_id,
            )

    def evaluate(self, generation_id: Any) -> GateResult:
        """Decide whether this generation's objectives may be computed."""
        if self._mongo.generation_repo.any_simulation_active(generation_id):
            # A redelivered terminal event, or one that arrived for a
            # generation that has since been reopened for a repeat. Computing
            # objectives now would read a half-finished generation.
            logger.info(
                "[%s] Generation %s still has simulations running; waiting "
                "for it to close again", self._label, generation_id,
            )
            return GateResult(proceed=False, completeness={}, waiting=True)

        completeness = self._mongo.generation_repo.replica_completeness(
            generation_id, self._keys)
        logger.info(
            "[%s] Generation %s completeness: %s (%s)",
            self._label, generation_id, summarize(completeness),
            self._policy.describe(),
        )

        retries_used = self._mongo.generation_repo.retries_used(generation_id)
        missing = repeatable(completeness, self._policy, retries_used)
        if not missing:
            if self._policy.mode == MODE_REPEAT and retries_used:
                logger.info(
                    "[%s] Generation %s was repeated %d time(s)",
                    self._label, generation_id, retries_used,
                )
            return GateResult(proceed=True, completeness=completeness)

        # The budget is taken **before** anything is created, and the
        # condition is in the write. The counter used to be incremented after
        # the clones, and a failure of that write left clones that would run
        # with a count that had not moved — so the next evaluation created
        # another round. Reproduced with `max_retries=1`: two repetitions.
        #
        # Failing this way round costs a wasted round instead of an extra one,
        # and `release_retry_round` gives it back when nothing was created.
        round_number = self._mongo.generation_repo.claim_retry_round(
            generation_id, self._policy.max_retries)
        if round_number is None:
            logger.info(
                "[%s] Generation %s has spent its repetition budget",
                self._label, generation_id)
            return GateResult(proceed=True, completeness=completeness)

        queued = self._queue_repeats(generation_id, missing, round_number)
        if not queued:
            # Nothing could be repeated: the holes are seeds whose simulation
            # was never created, so there is no document to run again. The
            # round goes back, and the policy decides from here — reopening a
            # generation that nothing will ever close would be worse.
            logger.warning(
                "[%s] Generation %s is missing replicas that have no "
                "simulation to repeat: %s", self._label, generation_id,
                {k: v for k, v in missing.items()},
            )
            try:
                self._mongo.generation_repo.release_retry_round(generation_id)
            except Exception:
                logger.exception(
                    "[%s] Could not give back the unused repetition round of "
                    "%s", self._label, generation_id)
            return GateResult(proceed=True, completeness=completeness)

        # Reopened after the clones exist. The other order has a hole: a
        # worker can finish a clone before the reopen lands, closing the
        # generation again and leaving the reopen to set a finished generation
        # back to Running with nothing to close it. The master-node's periodic
        # pass covers that; this order makes it rare rather than routine.
        self._mongo.generation_repo.reopen(generation_id)
        logger.warning(
            "[%s] Generation %s: queued %d replica(s) again (round %d of %d)",
            self._label, generation_id, queued, round_number,
            self._policy.max_retries,
        )
        return GateResult(
            proceed=False, completeness=completeness, repeated=queued)

    def _queue_repeats(
        self, generation_id: Any, missing: dict[str, list[int]], round_number: int
    ) -> int:
        """Run the failed replicas again, with their own seeds.

        Each repeat is a **new** simulation document cloned from the one that
        failed, so the failure stays on the record with its message and the
        repeat has its own placement and outcome. Only ``Done`` documents are
        aggregated and completeness counts seeds, so a seed cannot contribute
        twice.

        ``round_number`` makes the creation idempotent: a pass that claimed a
        round and failed part-way through can be run again without duplicating
        the clones it already made.
        """
        try:
            failed = self._mongo.simulation_repo.find_terminal_by_generation(
                generation_id)
        except Exception:
            logger.exception(
                "[%s] Could not read the failed replicas of generation %s",
                self._label, generation_id)
            return 0

        queued = 0
        for simulation in failed:
            individual = str(simulation.get("individual_id") or "")
            seed = simulation.get("random_seed")
            if individual not in missing or seed is None:
                continue
            if int(seed) not in missing[individual]:
                continue
            try:
                clone = self._mongo.simulation_repo.clone_for_retry(
                    simulation["_id"],
                    reason=(
                        f"replica {seed} of {individual} failed: "
                        f"{simulation.get('system_message') or 'no message'}"
                    ),
                    retry_round=round_number,
                )
            except Exception:
                logger.exception(
                    "[%s] Could not repeat replica %s of %s",
                    self._label, seed, individual)
                continue
            if clone is not None:
                queued += 1
                # One seed, one repeat per pass: a second clone of the same
                # seed in one pass would run the same replica twice.
                missing[individual] = [
                    s for s in missing[individual] if s != int(seed)]
        return queued
