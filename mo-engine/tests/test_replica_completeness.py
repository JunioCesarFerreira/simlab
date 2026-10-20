"""The policy that stands between a closed generation and its objectives.

R19. ``replica_completeness`` existed in the persistence library and nothing
called it: the strategies aggregated whatever had reached ``Done`` and
published the mean, so an individual evaluated from one replica of two was
indistinguishable from one evaluated from both. The completeness was
computable and never computed.

What is asserted here is the decision, not the counting — the counting is
``pylib/tests/test_generation_closing.py``. The repositories are stand-ins,
but strict ones: the gate's whole job is the order and the conditions of the
calls it makes, so the fake records them.
"""
from typing import Any
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.strategy.completeness import (
    EXHAUSTED_ACCEPT,
    EXHAUSTED_WORST,
    MODE_NONE,
    MODE_REPEAT,
    ReplicaGate,
    ReplicaPolicy,
    annotate,
    is_usable,
    repeatable,
    reused,
    summarize,
    unrepairable,
)

GEN_ID = ObjectId("507f1f77bcf86cd799450001")
SIM_A1 = ObjectId("507f1f77bcf86cd799450011")
SIM_A2 = ObjectId("507f1f77bcf86cd799450012")


def record(**over) -> dict:
    base = {
        "expected": 2, "observed": 2, "failed": 0, "pending": 0, "absent": 0,
        "expected_seeds": [1, 2], "observed_seeds": [1, 2],
        "missing_seeds": [], "failed_seeds": [],
        # Kept in step with ``_completeness_of`` on purpose: a stand-in whose
        # shape has drifted from the real record is how a field stops being
        # read without any test noticing.
        "absent_seeds": [], "thin_metrics": [],
        "per_metric": {"latency": 2}, "attempts": 2, "state": "complete",
        "reason": "every planned replica finished with every metric",
    }
    base.update(over)
    return base


PARTIAL = record(
    observed=1, failed=1, observed_seeds=[1], missing_seeds=[2],
    failed_seeds=[2], per_metric={"latency": 1}, state="partial",
    reason="1 replica(s) failed (repeatable)")

#: The hole a repeat cannot fill: the seed was planned and no simulation was
#: ever created for it, so there is no document to clone.
ABSENT = record(
    observed=1, absent=1, observed_seeds=[1], missing_seeds=[2],
    absent_seeds=[2], per_metric={"latency": 1}, state="partial",
    reason="1 planned replica(s) were never created (nothing to repeat)")

#: The other one: every planned seed finished, and one came back without a
#: metric. ``missing_seeds`` is empty, so nothing else in the record explains
#: why this evaluation is partial.
THIN = record(
    per_metric={"latency": 1}, thin_metrics=["latency"], state="partial",
    reason="finished replicas are missing latency (nothing to repeat)")


# ── The policy as configuration ─────────────────────────────────────────────

class TestPolicyFromConfiguration:
    def test_the_default_repeats_once_then_refuses(self):
        policy = ReplicaPolicy.from_config({})

        assert policy.mode == MODE_REPEAT
        assert policy.max_retries == 1
        assert policy.on_exhausted == EXHAUSTED_WORST

    def test_an_experiment_can_ask_for_the_previous_behaviour(self):
        policy = ReplicaPolicy.from_config({
            "replica_policy": MODE_NONE,
            "incomplete_objective": EXHAUSTED_ACCEPT,
        })

        assert policy.mode == MODE_NONE
        assert policy.on_exhausted == EXHAUSTED_ACCEPT

    def test_a_typo_falls_back_to_the_default(self):
        """And must not silently change what the experiment measures."""
        policy = ReplicaPolicy.from_config({"replica_policy": "repaet"})

        assert policy.mode == MODE_REPEAT

    def test_a_non_numeric_limit_falls_back(self):
        policy = ReplicaPolicy.from_config({"max_replica_retries": "many"})

        assert policy.max_retries == 1

    def test_the_limit_can_be_zero(self):
        policy = ReplicaPolicy.from_config({"max_replica_retries": 0})

        assert policy.max_retries == 0


# ── Which evaluations the algorithm may see ────────────────────────────────

class TestUsability:
    def test_a_complete_evaluation_is_usable(self):
        assert is_usable(record(), ReplicaPolicy()) is True

    def test_a_partial_one_is_not_usable_by_default(self):
        """The default refuses it: an under-replicated individual should not
        compete as though it were fully evaluated."""
        assert is_usable(PARTIAL, ReplicaPolicy()) is False

    def test_a_partial_one_is_usable_when_the_policy_says_so(self):
        policy = ReplicaPolicy(on_exhausted=EXHAUSTED_ACCEPT)

        assert is_usable(PARTIAL, policy) is True

    def test_accepting_still_needs_something_to_accept(self):
        empty = record(observed=0, observed_seeds=[], missing_seeds=[1, 2],
                       failed_seeds=[1, 2], failed=2, state="invalid")
        policy = ReplicaPolicy(on_exhausted=EXHAUSTED_ACCEPT)

        assert is_usable(empty, policy) is False

    def test_no_record_at_all_is_usable(self):
        """A genome whose objectives were reused from the cache has no
        simulations in this generation, and nothing here knows better."""
        assert is_usable(None, ReplicaPolicy()) is True

    def test_an_unknown_state_is_usable(self):
        """Experiments created before the plan was recorded. Refusing them on
        a suspicion would silently invalidate every one of them."""
        legacy = record(state="unknown")

        assert is_usable(legacy, ReplicaPolicy()) is True

    def test_the_decision_travels_with_the_evidence(self):
        annotated = annotate(PARTIAL, usable=False, policy=ReplicaPolicy())

        assert annotated["objective_used"] is False
        assert "repeat" in annotated["policy"]
        assert annotated["missing_seeds"] == [2]


class TestWhatIsWorthRepeating:
    def test_a_failed_seed_is_repeatable(self):
        assert repeatable({"a": PARTIAL}, ReplicaPolicy(), 0) == {"a": [2]}

    def test_nothing_is_repeated_once_the_limit_is_spent(self):
        assert repeatable({"a": PARTIAL}, ReplicaPolicy(max_retries=1), 1) == {}

    def test_nothing_is_repeated_when_the_policy_says_not_to(self):
        assert repeatable({"a": PARTIAL}, ReplicaPolicy(mode=MODE_NONE), 0) == {}

    def test_a_pending_evaluation_is_left_alone(self):
        pending = record(
            observed=1, pending=1, state="pending", missing_seeds=[2],
            failed_seeds=[])

        assert repeatable({"a": pending}, ReplicaPolicy(), 0) == {}

    def test_an_absent_seed_is_not_repeatable(self):
        """There is no simulation to run again, and reopening the generation
        for it would leave one nothing will ever close."""
        absent = record(
            observed=1, absent=1, state="partial", missing_seeds=[2],
            failed_seeds=[])

        assert repeatable({"a": absent}, ReplicaPolicy(), 0) == {}

    def test_the_summary_names_the_states(self):
        assert summarize({"a": record(), "b": PARTIAL}) == "1 complete, 1 partial"


# ── The gate, against repositories that record what it did ────────────────

class FakeMongo:
    def __init__(self, *, completeness, active=False, retries=0, failed=(),
                 max_retries=1, claim=True):
        self.generation_repo = MagicMock()
        self.simulation_repo = MagicMock()
        self.generation_repo.any_simulation_active.return_value = active
        self.generation_repo.replica_completeness.return_value = completeness
        self.generation_repo.retries_used.return_value = retries
        # The budget is taken before anything is created, and the condition is
        # in the write: `None` is how the repository says the limit is spent.
        self.generation_repo.claim_retry_round.return_value = (
            retries + 1 if claim and retries < max_retries else None)
        self.generation_repo.pending_retry_round.return_value = None
        self.generation_repo.finish_retry_round.return_value = True
        self.simulation_repo.find_terminal_by_generation.return_value = list(failed)
        self.cloned: list[ObjectId] = []
        self.rounds: list[Any] = []

        def clone(sim_id, reason="", retry_round=None):
            self.cloned.append(sim_id)
            self.rounds.append(retry_round)
            return ObjectId()

        self.simulation_repo.clone_for_retry.side_effect = clone


def failed_doc(sim_id, individual, seed, message="boom"):
    return {"_id": sim_id, "individual_id": individual, "random_seed": seed,
            "system_message": message}


def gate(mongo, policy=None) -> ReplicaGate:
    return ReplicaGate(
        mongo, objective_keys=["latency"],
        policy=policy or ReplicaPolicy(), label="test")


class TestTheGate:
    def test_a_complete_generation_proceeds(self):
        mongo = FakeMongo(completeness={"a": record()})

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is True
        mongo.generation_repo.finish_retry_round.assert_not_called()

    def test_a_failed_replica_is_queued_again_with_its_own_seed(self):
        mongo = FakeMongo(
            completeness={"a": PARTIAL},
            failed=[failed_doc(SIM_A2, "a", 2)])

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is False, "evolution waits for the repeat"
        assert result.repeated == 1
        assert mongo.cloned == [SIM_A2]
        mongo.generation_repo.finish_retry_round.assert_called_once_with(GEN_ID, 1)

    def test_the_clones_are_queued_before_the_generation_reopens(self):
        """The other order has a hole: a worker can finish a clone before the
        reopen lands, closing the generation again and leaving the reopen to
        set a finished generation back to Running."""
        order: list[str] = []
        mongo = FakeMongo(
            completeness={"a": PARTIAL},
            failed=[failed_doc(SIM_A2, "a", 2)])
        mongo.simulation_repo.clone_for_retry.side_effect = (
            lambda sim_id, reason="", retry_round=None:
            order.append("clone") or ObjectId())
        mongo.generation_repo.finish_retry_round.side_effect = (
            lambda gid, round_number: order.append("reopen") or True)
        mongo.generation_repo.claim_retry_round.side_effect = (
            lambda gid, limit, replicas: order.append("claim") or 1)

        gate(mongo).evaluate(GEN_ID)

        # The budget first, then the clones, then the reopen. Claiming after
        # the clones is what let a failed counter write produce a second round.
        assert order == ["claim", "clone", "reopen"]

    def test_only_the_missing_seed_is_repeated(self):
        mongo = FakeMongo(
            completeness={"a": PARTIAL},
            failed=[failed_doc(SIM_A1, "a", 1), failed_doc(SIM_A2, "a", 2)])

        gate(mongo).evaluate(GEN_ID)

        assert mongo.cloned == [SIM_A2], "seed 1 already produced a result"

    def test_one_seed_is_not_cloned_twice_in_one_pass(self):
        """Two failed documents for one seed — a seed that failed, was
        repeated, and failed again. Cloning both would run it twice."""
        mongo = FakeMongo(
            completeness={"a": PARTIAL},
            failed=[failed_doc(SIM_A1, "a", 2), failed_doc(SIM_A2, "a", 2)])

        gate(mongo).evaluate(GEN_ID)

        assert len(mongo.cloned) == 1

    def test_the_limit_stops_the_repeating(self):
        mongo = FakeMongo(
            completeness={"a": PARTIAL}, retries=1,
            failed=[failed_doc(SIM_A2, "a", 2)])

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is True, "the policy decides from here"
        assert mongo.cloned == []
        mongo.generation_repo.finish_retry_round.assert_not_called()

    def test_a_generation_still_running_is_not_evaluated(self):
        """A redelivered terminal event, or one that arrived for a generation
        since reopened. Computing objectives would read it half-finished."""
        mongo = FakeMongo(completeness={"a": record()}, active=True)

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is False and result.waiting is True
        mongo.generation_repo.replica_completeness.assert_not_called()

    def test_nothing_to_clone_proceeds_instead_of_reopening(self):
        """A hole with no document behind it. Reopening would leave a
        generation nothing will ever close."""
        mongo = FakeMongo(completeness={"a": PARTIAL}, failed=[])

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is True
        mongo.generation_repo.finish_retry_round.assert_not_called()

    def test_the_plan_is_recorded_with_the_generation(self):
        mongo = FakeMongo(completeness={})

        gate(mongo).record_plan(GEN_ID, [1, 2])

        mongo.generation_repo.record_plan.assert_called_once_with(
            GEN_ID, [1, 2])

    def test_a_plan_that_cannot_be_recorded_does_not_stop_the_run(self):
        mongo = FakeMongo(completeness={})
        mongo.generation_repo.record_plan.side_effect = RuntimeError("mongo gone")

        gate(mongo).record_plan(GEN_ID, [1, 2])   # must not raise


# ── The strategy, driving the real gate ────────────────────────────────────

class TestTheStrategyHonoursThePolicy:
    """Level 2: a real ``_handle_generation_done`` with a real gate.

    The strategy is built bypassing ``__init__`` — the same technique
    ``test_generation_ordering.py`` uses — so what is exercised is the handler
    and the gate, with the repositories and the evolution step as the only
    stand-ins. The point is the acceptance criterion: *the algorithm receives
    only the evaluations the configured policy considers valid.*
    """

    def _strategy(self, *, completeness, metrics, policy=None, active=False,
                  retries=0, failed=()):
        import random
        from lib.strategy.nsga2 import NSGA2LoopStrategy

        strat = NSGA2LoopStrategy.__new__(NSGA2LoopStrategy)
        strat._stop_flag = False
        strat._generation_id = GEN_ID
        strat._exp_id = ObjectId()
        strat._objective_keys = ["latency"]
        strat._objective_goals = [1]
        strat._aggregator = "mean"
        strat._ga_rng = random.Random(1)
        strat._genome_objectives_cache = {}
        strat._genome_completeness_cache = {}
        strat._generation_completeness = {}
        strat._map_genome_objectives = {}
        strat._replica_policy = policy or ReplicaPolicy()

        mongo = FakeMongo(
            completeness=completeness, active=active, retries=retries,
            failed=failed)
        mongo.experiment_repo = MagicMock()
        mongo.experiment_repo.is_cancelled.return_value = False
        mongo.generation_repo.get_simulations_metrics_by_individual.return_value = metrics
        mongo.individual_repo = MagicMock()
        mongo.genome_cache_repo = MagicMock()
        strat.mongo = mongo
        strat._replica_gate = ReplicaGate(
            mongo, objective_keys=["latency"],
            policy=strat._replica_policy, label="test")

        genome = MagicMock()
        genome.get_hash.return_value = "a"
        strat._current_population = [genome]
        strat._evolution = MagicMock()
        strat._extract_objectives_to_minimization = (
            lambda m: [float(m["latency"])])
        return strat, mongo, genome

    def test_a_complete_individual_reaches_the_algorithm(self):
        strat, mongo, genome = self._strategy(
            completeness={"a": record()}, metrics={"a": {"latency": 1.5}})

        strat._handle_generation_done(GEN_ID)

        assert strat._map_genome_objectives[genome] == [1.5]
        strat._evolution.assert_called_once()

    def test_an_incomplete_one_is_refused_by_default(self):
        from lib.strategy.nsga2 import WORST_OBJECTIVE

        strat, mongo, genome = self._strategy(
            completeness={"a": PARTIAL}, metrics={"a": {"latency": 1.5}},
            retries=1)   # the repeat has already been spent

        strat._handle_generation_done(GEN_ID)

        assert strat._map_genome_objectives[genome] == [WORST_OBJECTIVE]
        strat._evolution.assert_called_once()

    def test_an_incomplete_one_is_accepted_when_configured(self):
        strat, mongo, genome = self._strategy(
            completeness={"a": PARTIAL}, metrics={"a": {"latency": 1.5}},
            policy=ReplicaPolicy(
                mode=MODE_NONE, on_exhausted=EXHAUSTED_ACCEPT))

        strat._handle_generation_done(GEN_ID)

        assert strat._map_genome_objectives[genome] == [1.5]

    def test_the_completeness_is_persisted_with_the_objectives(self):
        strat, mongo, genome = self._strategy(
            completeness={"a": record()}, metrics={"a": {"latency": 1.5}})

        strat._handle_generation_done(GEN_ID)

        mongo.individual_repo.update_evaluation.assert_called_once()
        args = mongo.individual_repo.update_evaluation.call_args.args
        assert args[0] == "a"
        stored = args[3]
        assert stored["state"] == "complete"
        assert stored["objective_used"] is True

    def test_a_repeat_defers_the_evolution(self):
        strat, mongo, genome = self._strategy(
            completeness={"a": PARTIAL}, metrics={"a": {"latency": 1.5}},
            failed=[failed_doc(SIM_A2, "a", 2)])

        strat._handle_generation_done(GEN_ID)

        strat._evolution.assert_not_called()
        mongo.individual_repo.update_evaluation.assert_not_called()
        assert mongo.cloned == [SIM_A2]

    def test_a_generation_still_running_is_not_evaluated(self):
        strat, mongo, genome = self._strategy(
            completeness={"a": record()}, metrics={"a": {"latency": 1.5}},
            active=True)

        strat._handle_generation_done(GEN_ID)

        strat._evolution.assert_not_called()
        mongo.generation_repo.get_simulations_metrics_by_individual.assert_not_called()


# ── R26: the repetition budget must survive a failed write ────────────────

class TestTheRepetitionBudget:
    """Reproduces R26. The counter was incremented *after* the clones.

    A failure of that write — or a restart in the window — left clones that
    would run and a count that had not moved, so the next evaluation saw the
    old number and created another round. Measured with ``max_retries=1``:
    two effective repetitions.

    Budget and replica manifest are claimed together. An interrupted enqueue
    resumes the same round rather than wasting it or exceeding the limit.
    """

    def test_the_round_is_claimed_before_anything_is_created(self):
        mongo = FakeMongo(
            completeness={"a": PARTIAL}, failed=[failed_doc(SIM_A2, "a", 2)])

        gate(mongo).evaluate(GEN_ID)

        args = mongo.generation_repo.claim_retry_round.call_args.args
        assert args[:2] == (GEN_ID, 1)
        assert args[2][0]["simulation_id"] == SIM_A2

    def test_a_spent_budget_creates_nothing(self):
        """The limit is enforced by the repository refusing the claim, not by
        the caller comparing a number it read earlier."""
        mongo = FakeMongo(
            completeness={"a": PARTIAL}, failed=[failed_doc(SIM_A2, "a", 2)],
            claim=False)

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is True
        assert mongo.cloned == []
        mongo.generation_repo.finish_retry_round.assert_not_called()

    def test_the_clones_carry_their_round(self):
        """Which is what makes a repeated pass idempotent: the same round
        cannot queue the same replica twice."""
        mongo = FakeMongo(
            completeness={"a": PARTIAL}, failed=[failed_doc(SIM_A2, "a", 2)])

        gate(mongo).evaluate(GEN_ID)

        assert mongo.rounds == [1]

    def test_no_round_is_claimed_without_a_simulation_to_repeat(self):
        mongo = FakeMongo(completeness={"a": PARTIAL}, failed=[])

        assert gate(mongo).evaluate(GEN_ID).proceed is True
        mongo.generation_repo.claim_retry_round.assert_not_called()

    def test_a_read_failure_is_not_interpreted_as_no_work(self):
        mongo = FakeMongo(completeness={"a": PARTIAL})
        mongo.simulation_repo.find_terminal_by_generation.side_effect = RuntimeError("offline")

        with pytest.raises(RuntimeError, match="offline"):
            gate(mongo).evaluate(GEN_ID)
        mongo.generation_repo.claim_retry_round.assert_not_called()

    @pytest.mark.parametrize("active", [False, True])
    def test_an_interrupted_round_resumes_even_with_clones_already_running(self, active):
        mongo = FakeMongo(completeness={"a": PARTIAL}, retries=1, active=active)
        mongo.generation_repo.pending_retry_round.return_value = {
            "replica_retries": 1,
            "retry_plan": [{"simulation_id": SIM_A1, "reason": "failed"},
                           {"simulation_id": SIM_A2, "reason": "failed"}],
        }

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is False
        assert mongo.cloned == [SIM_A1, SIM_A2]
        assert mongo.rounds == [1, 1]
        mongo.generation_repo.claim_retry_round.assert_not_called()
        mongo.generation_repo.finish_retry_round.assert_called_once_with(GEN_ID, 1)
        mongo.generation_repo.close_if_finished.assert_called_once_with(GEN_ID)

    def test_a_partial_enqueue_keeps_its_manifest(self):
        mongo = FakeMongo(completeness={"a": PARTIAL})
        mongo.generation_repo.pending_retry_round.return_value = {
            "replica_retries": 1,
            "retry_plan": [{"simulation_id": SIM_A1, "reason": "failed"},
                           {"simulation_id": SIM_A2, "reason": "failed"}],
        }
        mongo.simulation_repo.clone_for_retry.side_effect = [ObjectId(), RuntimeError("offline")]

        with pytest.raises(RuntimeError, match="offline"):
            gate(mongo).evaluate(GEN_ID)
        mongo.generation_repo.finish_retry_round.assert_not_called()

    def test_another_evaluator_claiming_the_round_defers_objectives(self):
        mongo = FakeMongo(completeness={"a": PARTIAL}, claim=False,
                          failed=[failed_doc(SIM_A2, "a", 2)])
        mongo.generation_repo.pending_retry_round.side_effect = [None, {"retry_plan": []}]

        assert gate(mongo).evaluate(GEN_ID).waiting is True


# ── R19: a seed being repeated is pending, not failed ─────────────────────

class TestASeedBeingRepeated:
    """Counting has to agree with what is actually in flight.

    A seed whose first attempt failed is in ``terminal_seeds``; its waiting
    clone used to disappear from the count, because pending was derived as
    "neither done nor terminal". The evaluation then read ``invalid`` while
    the repeat was running, and the record described the wrong state.
    """

    def test_a_pending_seed_is_not_repeated_again(self):
        pending = record(
            observed=1, pending=1, state="pending", missing_seeds=[2],
            failed_seeds=[])

        assert repeatable({"a": pending}, ReplicaPolicy(), 0) == {}

    def test_the_gate_waits_for_a_generation_with_clones_running(self):
        mongo = FakeMongo(completeness={"a": record()}, active=True)

        assert gate(mongo).evaluate(GEN_ID).waiting is True


# ── R27: reuse must not change what the evidence says ─────────────────────

class TestReuseKeepsItsEvidence:
    """Reproduces R27, which the project's own pilot showed.

    A genome that reappears in a later generation has no simulations *there*,
    so nothing computes a completeness for it. The record was absent, and the
    individual was written as ``unknown`` with ``objective_used`` true — even
    when the original evaluation was a partial mean the policy had **refused**.
    The evidence was known and then discarded.

    In a 344-simulation pilot, 38 of 210 individuals reported ``unknown``
    purely through reuse, each with 0 simulations in its own generation and 2
    elsewhere in the experiment.
    """

    def test_a_carried_record_keeps_its_state(self):
        carried = reused(PARTIAL)

        assert carried["state"] == "partial"
        assert carried["observed"] == 1 and carried["expected"] == 2
        assert carried["reused"] is True
        assert carried["source"] == "genome_cache"

    def test_a_carried_refusal_stays_refused(self):
        """The vector is the same vector. Re-deciding it would let the same
        numbers be refused in one generation and accepted in the next."""
        carried = reused({**PARTIAL, "objective_used": False})

        assert is_usable(carried, ReplicaPolicy()) is False

    def test_a_carried_acceptance_stays_accepted(self):
        carried = reused({**record(), "objective_used": True})

        assert is_usable(carried, ReplicaPolicy()) is True

    def test_annotating_a_carried_record_does_not_overwrite_the_decision(self):
        carried = reused({**PARTIAL, "objective_used": False})

        annotated = annotate(carried, usable=True, policy=ReplicaPolicy())

        assert annotated["objective_used"] is False
        assert annotated["state"] == "partial"

    def test_nothing_cached_is_still_unknown_but_says_why(self):
        carried = reused(None)

        assert carried["state"] == "unknown"
        assert carried["source"] == "genome_cache"

    def test_an_analytical_evaluation_is_not_unknown(self):
        """`unknown` means "the plan was not recorded". An analytical problem
        had no plan to record, which is a different thing."""
        from lib.strategy.completeness import STATE_ANALYTICAL, without_simulations

        made = without_simulations(STATE_ANALYTICAL, "no simulation was run")

        assert made["state"] == STATE_ANALYTICAL
        assert made["expected"] == 0
        assert is_usable(made, ReplicaPolicy()) is True

    def test_an_infeasible_individual_is_named_as_such(self):
        from lib.strategy.completeness import STATE_INFEASIBLE, without_simulations

        made = without_simulations(STATE_INFEASIBLE, "violates a constraint")

        assert made["state"] == STATE_INFEASIBLE
        assert is_usable(made, ReplicaPolicy()) is True

    def test_the_cache_is_written_with_the_evidence(self):
        """The whole chain: evaluate, persist, and the cache keeps the record
        rather than only the numbers."""
        strat, mongo, genome = TestTheStrategyHonoursThePolicy()._strategy(
            completeness={"a": record()}, metrics={"a": {"latency": 1.5}})

        strat._handle_generation_done(GEN_ID)

        mongo.genome_cache_repo.set_objectives.assert_called_once()
        cached = mongo.genome_cache_repo.set_objectives.call_args.args[3]
        assert cached["state"] == "complete"
        assert cached["observed"] == 2


class TestTheReusePathItself:
    """The pilot's 38 individuals, as a test.

    Each had 0 simulations in its own generation and 2 elsewhere in the
    experiment: a genome reused from the cache. The individual was written
    with no completeness at all, which read as ``unknown`` and
    ``objective_used = true`` — losing a record that said ``complete, 2/2``.
    """

    def _enqueue_with_a_cached_genome(self, cached_completeness):
        import random

        from bson import ObjectId as OID
        from lib.strategy.nsga2 import NSGA2LoopStrategy

        strat = NSGA2LoopStrategy.__new__(NSGA2LoopStrategy)
        strat._is_synthetic = True
        strat._exp_id = OID()
        strat._gen_index = 0
        strat._sim_rand_seeds = [11111, 22222]
        strat._ga_rng = random.Random(1)
        strat._inserted_genomes = set()
        strat._sim_done_count = 0
        strat._count_sims_inserted = 0
        strat._objective_keys = ["latency"]
        strat._objective_goals = [1]
        strat._map_genome_objectives = {}
        strat._replica_policy = ReplicaPolicy()

        genome = MagicMock()
        genome.get_hash.return_value = "a"
        genome.to_dict.return_value = {"relays": []}
        strat._current_population = [genome]
        strat._genome_objectives_cache = {"a": [1.5]}
        strat._genome_completeness_cache = {"a": cached_completeness}

        mongo = MagicMock()
        mongo.generation_repo.insert.side_effect = lambda doc: doc["_id"]
        strat.mongo = mongo
        strat._replica_gate = ReplicaGate(
            mongo, objective_keys=["latency"], policy=strat._replica_policy)
        strat._problem_adapter = MagicMock()
        strat._problem_adapter.is_analytical = False
        strat._problem_adapter.penalty_objectives.return_value = None
        strat._convert_genome_to_sim_config = MagicMock(
            return_value={"randomSeed": 0, "simulationElements": {}})
        strat._upload_topology_async = MagicMock()
        strat._fire_generation_done = MagicMock()

        strat._generation_enqueue()
        return mongo.individual_repo.insert.call_args.args[0]

    def test_the_individual_carries_the_cached_evidence(self):
        doc = self._enqueue_with_a_cached_genome(record())

        carried = doc["completeness"]
        assert carried["state"] == "complete"
        assert carried["observed"] == 2 and carried["expected"] == 2
        assert carried["reused"] is True

    def test_a_refusal_is_carried_too(self):
        doc = self._enqueue_with_a_cached_genome(
            {**PARTIAL, "objective_used": False})

        carried = doc["completeness"]
        assert carried["state"] == "partial"
        assert carried["objective_used"] is False
        assert is_usable(carried, ReplicaPolicy()) is False

    def test_nothing_cached_still_says_where_it_came_from(self):
        doc = self._enqueue_with_a_cached_genome(None)

        assert doc["completeness"]["state"] == "unknown"
        assert doc["completeness"]["source"] == "genome_cache"


class TestTheHandlerDoesNotOverwriteWhatTheEnqueueWrote:
    """Found by running the code, not by reading the review.

    The enqueue writes the carried evidence onto a reused individual. The
    generation-done handler then writes objectives for **every** genome it
    holds a vector for — including the ones it skipped because their
    objectives were already known — and with no record for them in this
    generation it annotated ``None``: ``unknown``, ``objective_used: true``.
    The enqueue recorded *complete, 2 of 2* and the handler replaced it.

    Observed in a real 16-simulation run: 8 individuals complete, 8 unknown,
    with the genome cache correctly holding ``complete`` for all of them.
    """

    def _strategy_with_a_skipped_genome(self, enqueued):
        import random

        from lib.strategy.nsga2 import NSGA2LoopStrategy

        strat = NSGA2LoopStrategy.__new__(NSGA2LoopStrategy)
        strat._stop_flag = False
        strat._generation_id = GEN_ID
        strat._exp_id = ObjectId()
        strat._objective_keys = ["latency"]
        strat._objective_goals = [1]
        strat._aggregator = "mean"
        strat._ga_rng = random.Random(1)
        strat._genome_objectives_cache = {}
        strat._genome_completeness_cache = {}
        strat._replica_policy = ReplicaPolicy()
        strat._generation_completeness = dict(enqueued)

        genome = MagicMock()
        genome.get_hash.return_value = "a"
        # Already evaluated: the handler skips it and goes straight to
        # persisting, which is the path that overwrote the evidence.
        strat._map_genome_objectives = {genome: [1.5]}
        strat._current_population = [genome]

        mongo = FakeMongo(completeness={})
        mongo.experiment_repo = MagicMock()
        mongo.experiment_repo.is_cancelled.return_value = False
        mongo.generation_repo.get_simulations_metrics_by_individual.return_value = {}
        mongo.individual_repo = MagicMock()
        mongo.genome_cache_repo = MagicMock()
        strat.mongo = mongo
        strat._replica_gate = ReplicaGate(
            mongo, objective_keys=["latency"], policy=strat._replica_policy)
        strat._evolution = MagicMock()
        strat._extract_objectives_to_minimization = lambda m: [1.5]
        return strat, mongo

    def test_a_reused_record_survives_the_handler(self):
        carried = reused(record())
        strat, mongo = self._strategy_with_a_skipped_genome({"a": carried})

        strat._handle_generation_done(GEN_ID)

        stored = mongo.individual_repo.update_evaluation.call_args.args[3]
        assert stored["state"] == "complete"
        assert stored["reused"] is True

    def test_an_analytical_record_survives_it_too(self):
        from lib.strategy.completeness import STATE_ANALYTICAL, without_simulations

        made = without_simulations(STATE_ANALYTICAL, "no simulation was run")
        strat, mongo = self._strategy_with_a_skipped_genome({"a": made})

        strat._handle_generation_done(GEN_ID)

        stored = mongo.individual_repo.update_evaluation.call_args.args[3]
        assert stored["state"] == STATE_ANALYTICAL

    def test_with_nothing_written_either_it_is_still_unknown(self):
        """No record anywhere is a real state, and it says so."""
        strat, mongo = self._strategy_with_a_skipped_genome({})

        strat._handle_generation_done(GEN_ID)

        stored = mongo.individual_repo.update_evaluation.call_args.args[3]
        assert stored["state"] == "unknown"

    def test_this_generations_measurement_still_wins(self):
        """A genome that *was* simulated here must not read a stale record."""
        strat, mongo = self._strategy_with_a_skipped_genome(
            {"a": reused(PARTIAL)})
        mongo.generation_repo.replica_completeness.return_value = {
            "a": record()}

        strat._handle_generation_done(GEN_ID)

        stored = mongo.individual_repo.update_evaluation.call_args.args[3]
        assert stored["state"] == "complete"
        assert "reused" not in stored

    def test_the_in_memory_cache_is_filled_during_the_run(self):
        """The dict was only ever populated by `_load_genome_cache_from_db`.

        So within a single run, the reuse in the *next* generation found
        nothing and reported `unknown` — with the database holding the record
        all along. Found by running it: 8 of 16 individuals, with the genome
        cache correct for every one of them.
        """
        strat, mongo, genome = TestTheStrategyHonoursThePolicy()._strategy(
            completeness={"a": record()}, metrics={"a": {"latency": 1.5}})

        strat._handle_generation_done(GEN_ID)

        carried = strat._genome_completeness_cache["a"]
        assert carried["state"] == "complete"
        assert carried["observed"] == 2


class TestTheDestinationOfAHoleNoRepeatCanFill:
    """F3.4: an absent seed and a missing metric are decided, not awaited.

    Neither has a document to clone, so neither reaches ``_plan_repeats``,
    and the gate proceeds straight to the policy. That was already true. What
    was missing is that nothing said so: the log showed ``1 partial`` and a
    generation that carried on, which reads the same whether the repeat was
    tried and failed, was out of budget, or never applied at all.

    These assert the two properties the plan asks for — such an evaluation
    must not look complete, and must not wait for work nobody enqueued — and
    the statement that distinguishes them.
    """

    def test_an_absent_seed_does_not_spend_a_repetition_round(self):
        mongo = FakeMongo(completeness={"a": ABSENT})

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is True, "nothing will arrive on its own"
        assert result.repeated == 0
        mongo.generation_repo.claim_retry_round.assert_not_called()
        # Reopening for a seed with no simulation would leave a generation
        # nothing can close.
        mongo.generation_repo.finish_retry_round.assert_not_called()

    def test_a_missing_metric_does_not_spend_a_repetition_round(self):
        mongo = FakeMongo(completeness={"a": THIN})

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is True and result.repeated == 0
        mongo.generation_repo.claim_retry_round.assert_not_called()

    def test_neither_is_usable_under_the_default_policy(self):
        """The destination itself: refused, with the evidence kept."""
        policy = ReplicaPolicy()

        assert is_usable(ABSENT, policy) is False
        assert is_usable(THIN, policy) is False
        assert annotate(ABSENT, usable=False, policy=policy)[
            "objective_used"] is False

    def test_accepting_the_partial_mean_is_the_configured_alternative(self):
        policy = ReplicaPolicy(on_exhausted=EXHAUSTED_ACCEPT)

        assert is_usable(ABSENT, policy) is True, "one replica did finish"
        assert is_usable(THIN, policy) is True

    def test_the_gate_names_the_evaluations_it_decided_without_repeating(self):
        mongo = FakeMongo(completeness={"a": ABSENT, "b": THIN, "c": PARTIAL,
                                        "d": record()})

        holes = gate(mongo)._state_unrepairable(GEN_ID, mongo.
                                                generation_repo.
                                                replica_completeness(GEN_ID, []))

        assert set(holes) == {"a", "b"}, (
            "c has a document to clone; d is complete")
        assert "never created" in holes["a"]
        assert "latency" in holes["b"]

    def test_it_is_logged_once_with_the_policy_destination(self, caplog):
        mongo = FakeMongo(completeness={"a": ABSENT})

        with caplog.at_level("WARNING"):
            gate(mongo).evaluate(GEN_ID)

        messages = [r.getMessage() for r in caplog.records]
        unrepairable_lines = [m for m in messages if "cannot be repaired" in m]
        assert len(unrepairable_lines) == 1
        assert EXHAUSTED_WORST in unrepairable_lines[0], (
            "the destination, not only the diagnosis")

    def test_a_repeatable_hole_is_not_reported_as_decided(self):
        """It has a repair, so its destination is the repeat — reporting it
        here would say a decision had been made while a clone was running."""
        mongo = FakeMongo(completeness={"a": PARTIAL},
                          failed=[failed_doc(SIM_A2, "a", 2)])

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is False and result.repeated == 1
        assert unrepairable({"a": PARTIAL}) == {}

    def test_a_pending_evaluation_is_not_a_decided_one(self):
        pending = record(observed=1, pending=1, observed_seeds=[1],
                         missing_seeds=[2], state="pending",
                         reason="1 replica(s) still running")

        assert unrepairable({"a": pending}) == {}

    def test_an_exhausted_budget_still_reports_the_unrepairable_hole(self):
        """The two reasons an evaluation is decided without a repeat are
        different, and both have to be visible: one spent its budget, the
        other never had a repair available."""
        mongo = FakeMongo(completeness={"a": ABSENT, "b": PARTIAL}, retries=1)

        result = gate(mongo).evaluate(GEN_ID)

        assert result.proceed is True, "the budget is spent"
        assert unrepairable(result.completeness) == {
            "a": ABSENT["reason"]}, "b is repeatable in principle"
