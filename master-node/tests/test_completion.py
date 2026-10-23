"""Tests for planning completion, and for per-machine limits.

The numbers follow what was measured on the deployment that motivated this:
a local host whose workers take ~30 s on the benchmark simulation, VMs on one
machine that take ~60 s when five run at once, and a desktop that takes
~87 s. Locked in:

  * a slower free worker gets work only when it finishes the batch sooner —
    at the start of a long queue, never at its tail;
  * waiting is bounded: a run far past its expected time is not waited on;
  * a machine's limit holds across all of its hosts, for every policy;
  * a worker with no measurement is sampled rather than planned around.
"""
from datetime import datetime, timedelta, timezone

import pytest

from lib.scheduler.completion import OVERDUE_FACTOR, plan_completion
from lib.scheduler.inventory import apply_machine_limits, describe_loads, snapshot_loads
from lib.scheduler.policy import (
    EarliestCompletionPolicy,
    FastestThenLeastLoadedPolicy,
    HostLoad,
    LeastLoadedHostPolicy,
    build_policy,
)

LOCAL_SEC, VM_SEC, DESKTOP_SEC = 30.0, 60.0, 87.0


def host(name, *, free=(), busy=(), machine=None, schedulable=True, host_id=None):
    """A host with free workers of the given durations and busy ones given as
    ``(expected, running for)``."""
    return HostLoad(
        host_id=host_id or name, name=name, free=len(free), active=len(busy),
        schedulable=schedulable,
        fastest_free_sec=min(free) if free and None not in free else None,
        machine=machine or name, free_durations=tuple(free), occupied=tuple(busy))


def local(free=0, busy_running_for=()):
    return host("local", free=(LOCAL_SEC,) * free,
                busy=tuple((LOCAL_SEC, t) for t in busy_running_for))


def vm(i, *, free=True, running_for=None):
    return host(f"vm{i}", machine="andromeda",
                free=(VM_SEC,) if free else (),
                busy=((VM_SEC, running_for),) if running_for is not None else ())


class TestWhenToUseASlowerWorker:
    def test_the_fastest_free_worker_takes_the_head_of_the_queue(self):
        plan = plan_completion([local(free=6), vm(1)], waiting=10)

        assert plan.host.name == "local" and not plan.deferred

    def test_the_tail_waits_for_the_fast_host(self):
        """Three jobs left, the local workers 5 s from free: 35 s there beats
        60 s on the free VM. The deployment lost ~10 min per generation to
        exactly this."""
        plan = plan_completion(
            [local(busy_running_for=(25,) * 6), vm(1), vm(2)], waiting=3)

        assert plan.deferred and plan.host is None
        assert "faster worker" in plan.reason

    def test_a_long_queue_uses_the_slower_worker_now(self):
        # Forty jobs: six local workers cannot finish them before a VM run
        # would have, so the VM starts one now.
        plan = plan_completion(
            [local(busy_running_for=(25,) * 6), vm(1)], waiting=40)

        assert plan.host.name == "vm1" and not plan.deferred

    def test_a_much_slower_worker_is_not_worth_it_even_in_a_short_burst(self):
        # One round of local work (5 + 30 s) is sooner than one desktop run.
        desktop = host("desktop", free=(DESKTOP_SEC,))
        plan = plan_completion([local(busy_running_for=(25,) * 6), desktop], waiting=12)

        assert plan.deferred

    def test_the_plan_predicts_when_the_waiting_work_is_done(self):
        plan = plan_completion([local(free=2)], waiting=4)

        assert plan.makespan_sec == pytest.approx(2 * LOCAL_SEC)


class TestWaitingIsBounded:
    def test_a_run_far_past_its_estimate_is_not_waited_on(self):
        # Stuck or misestimated: holding work for it could stall the queue.
        stuck = local(busy_running_for=(LOCAL_SEC * OVERDUE_FACTOR + 1,) * 6)
        plan = plan_completion([stuck, vm(1)], waiting=2)

        assert plan.host.name == "vm1"

    def test_a_busy_worker_with_no_estimate_is_not_waited_on(self):
        unknown = host("local", busy=((None, 3.0),) * 6)
        plan = plan_completion([unknown, vm(1)], waiting=2)

        assert plan.host.name == "vm1"

    def test_an_unreachable_host_is_not_waited_on(self):
        offline = host("local", busy=((LOCAL_SEC, 25.0),) * 6, schedulable=False)
        plan = plan_completion([offline, vm(1)], waiting=2)

        assert plan.host.name == "vm1"


class TestUnmeasuredWorkers:
    def test_an_unmeasured_worker_is_sampled_first(self):
        new = host("new", free=(None,))
        plan = plan_completion([local(free=6), new], waiting=1)

        assert plan.host.name == "new"
        assert "no measurement" in plan.reason

    def test_a_snapshot_without_per_worker_detail_still_places(self):
        bare = HostLoad(host_id="h", name="h", free=2, active=0, schedulable=True)

        assert plan_completion([bare], waiting=1).host.name == "h"

    def test_nothing_free_is_no_capacity_not_a_deferral(self):
        plan = plan_completion([local(busy_running_for=(10,) * 6)], waiting=5)

        assert plan.host is None and not plan.deferred


class TestMachineLimits:
    def test_a_machine_at_its_limit_takes_no_more_work(self):
        """Eight VMs of one machine peaked at five simulations at once."""
        vms = [vm(i, free=False, running_for=10) for i in range(1, 6)] + \
              [vm(i) for i in range(6, 9)]

        capped = apply_machine_limits(vms, {"andromeda": 5})

        assert all(load.capped for load in capped)
        assert not any(load.eligible for load in capped)
        assert " cap]" in describe_loads(capped)

    def test_below_its_limit_it_still_takes_work(self):
        vms = [vm(1, free=False, running_for=10), vm(2)]

        assert apply_machine_limits(vms, {"andromeda": 5})[1].eligible

    def test_the_limit_counts_every_host_of_the_machine(self):
        # One VM busy on each of two hosts: a limit of two is reached, and the
        # third host — idle itself — is capped with them.
        vms = [vm(1, free=False, running_for=5), vm(2, free=False, running_for=5), vm(3)]

        capped = apply_machine_limits(vms, {"andromeda": 2})

        assert capped[2].capped

    def test_other_machines_are_unaffected(self):
        loads = apply_machine_limits(
            [local(free=6), vm(1, free=False, running_for=1)], {"andromeda": 1})

        assert loads[0].eligible and not loads[0].capped

    def test_no_limits_is_the_pool_as_it_was(self):
        loads = [local(free=6), vm(1)]

        assert apply_machine_limits(loads, {}) is loads

    @pytest.mark.parametrize("policy", [
        LeastLoadedHostPolicy(), FastestThenLeastLoadedPolicy(), EarliestCompletionPolicy()])
    def test_every_policy_respects_it(self, policy):
        vms = apply_machine_limits(
            [vm(1, free=False, running_for=1), vm(2)], {"andromeda": 1})

        assert policy.select(vms, 0) is None

    def test_the_plan_fits_only_the_fastest_free_workers_under_the_limit(self):
        # Room for one more on the machine: of two free VMs, one is counted.
        slow = host("vm2", machine="andromeda", free=(VM_SEC * 2,))
        loads = apply_machine_limits(
            [vm(1, free=False, running_for=10), vm(3), slow], {"andromeda": 2})

        plan = plan_completion(loads, waiting=10, limits={"andromeda": 2})

        assert plan.host.name == "vm3"


class TestThePolicy:
    def test_it_is_registered_under_its_name(self):
        assert isinstance(build_policy("earliest_completion"), EarliestCompletionPolicy)

    def test_it_orders_within_the_host_by_speed(self):
        assert EarliestCompletionPolicy.prefers_fastest is True

    def test_without_the_plan_it_behaves_like_the_ranked_policy(self):
        loads = [local(free=6), vm(1)]

        assert EarliestCompletionPolicy().select(loads, 0).name == \
            FastestThenLeastLoadedPolicy().select(loads, 0).name


class TestReadingTheSnapshot:
    NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)

    class Repo:
        def __init__(self, rows):
            self.rows = rows

        def inventory_snapshot(self):
            return self.rows

    def test_per_worker_detail_and_the_machine_are_read(self):
        row = {
            "_id": "vm1", "name": "vm1", "type": "remote", "free": 1, "active": 1,
            "schedulable": True,
            "access": {"address": "Andromeda.Example", "port": 2191},
            "free_durations": [61.5],
            "occupied_workers": [{"mean_duration_sec": 58.0,
                                  "started_at": self.NOW - timedelta(seconds=20)}],
        }

        [load] = snapshot_loads(self.Repo([row]), now=self.NOW)

        assert load.machine == "andromeda.example"
        assert load.free_durations == (61.5,)
        assert load.occupied == ((58.0, 20.0),)

    def test_a_reserved_worker_has_not_started(self):
        row = {"_id": "h", "name": "h", "free": 0, "active": 1, "schedulable": True,
               "occupied_workers": [{"mean_duration_sec": 30.0, "started_at": None}]}

        [load] = snapshot_loads(self.Repo([row]), now=self.NOW)

        assert load.occupied == ((30.0, None),)

    def test_an_explicit_machine_wins_over_the_address(self):
        row = {"_id": "h", "name": "h", "machine": "rack-2", "free": 0, "active": 0,
               "schedulable": True, "access": {"address": "gw.example"}}

        assert snapshot_loads(self.Repo([row]))[0].machine == "rack-2"
