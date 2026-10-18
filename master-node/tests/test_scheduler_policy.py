"""Tests for the placement policy.

The specification's own worked example is the headline test: three hosts with
8, 4 and 2 free containers, six simulations, and the answer must be 2/2/2. By
utilization it would be 3/2/1, so this file is what keeps the implementation
from drifting into favouring whichever machine has the most containers.

A pure function over a snapshot, so all of it runs without a database.
"""
import pytest

from lib.scheduler.inventory import describe_loads, find_load, total_capacity
from lib.scheduler.policy import (
    FastestThenLeastLoadedPolicy,
    HostLoad,
    LeastLoadedHostPolicy,
    build_policy,
    select_fastest_then_least_loaded,
    select_least_loaded_host,
)


def load(host_id, name, free, active=0, schedulable=True,
         fastest_free_sec=None) -> HostLoad:
    return HostLoad(
        host_id=host_id, name=name, free=free, active=active,
        schedulable=schedulable, fastest_free_sec=fastest_free_sec)


def place(loads: list[HostLoad], count: int) -> dict[str, int]:
    """Simulate ``count`` successive decisions over a mutable snapshot.

    Each placement takes a container and adds an active run on the chosen
    host, which is exactly what the atomic reservation does to the inventory
    the next decision reads.
    """
    state = {l.name: {"free": l.free, "active": l.active, "id": l.host_id,
                      "sched": l.schedulable} for l in loads}
    assigned: dict[str, int] = {name: 0 for name in state}
    cursor = 0
    for _ in range(count):
        snapshot = [
            load(v["id"], name, v["free"], v["active"], v["sched"])
            for name, v in state.items()
        ]
        chosen = select_least_loaded_host(snapshot, cursor)
        if chosen is None:
            break
        cursor += 1
        state[chosen.name]["free"] -= 1
        state[chosen.name]["active"] += 1
        assigned[chosen.name] += 1
    return assigned


# ── The specification's example ──────────────────────────────────────────────

class TestSpecifiedBehaviour:
    def test_six_simulations_across_8_4_2_give_two_each(self):
        hosts = [
            load("a", "host-a", free=8),   # local
            load("b", "host-b", free=4),   # remote
            load("c", "host-c", free=2),   # remote
        ]
        assert place(hosts, 6) == {"host-a": 2, "host-b": 2, "host-c": 2}

    def test_the_result_is_not_proportional_to_capacity(self):
        # The distinction that matters: 3/2/1 would be the utilization answer,
        # and would quietly hand most of every run to the biggest host.
        assert place([load("a", "a", 8), load("b", "b", 4), load("c", "c", 2)], 6) \
            != {"a": 3, "b": 2, "c": 1}

    def test_three_simulations_land_one_per_host(self):
        assert place([load("a", "a", 8), load("b", "b", 4), load("c", "c", 2)], 3) \
            == {"a": 1, "b": 1, "c": 1}

    def test_excess_spills_over_once_a_host_is_full(self):
        # Host c has two containers; the seventh and eighth simulations have to
        # go somewhere else, respecting its limit.
        assigned = place([load("a", "a", 8), load("b", "b", 4), load("c", "c", 2)], 9)
        assert assigned["c"] == 2
        assert sum(assigned.values()) == 9
        # And the remainder stays as even as the capacities allow.
        assert abs(assigned["a"] - assigned["b"]) <= 1

    def test_capacity_is_never_exceeded(self):
        assigned = place([load("a", "a", 2), load("b", "b", 1)], 10)
        assert assigned == {"a": 2, "b": 1}


# ── Eligibility ──────────────────────────────────────────────────────────────

class TestEligibility:
    def test_a_host_with_no_free_container_is_skipped(self):
        chosen = select_least_loaded_host(
            [load("a", "full", free=0, active=5), load("b", "free", free=1, active=9)], 0)
        assert chosen.name == "free"

    def test_an_unschedulable_host_is_excluded(self):
        # Offline or administratively disabled: it still reports its capacity,
        # but no new work goes there.
        chosen = select_least_loaded_host(
            [load("a", "offline", free=8, schedulable=False),
             load("b", "online", free=1, active=3)], 0)
        assert chosen.name == "online"

    def test_nothing_eligible_returns_none(self):
        # The dispatcher's cue to hold the job, not to fail it.
        assert select_least_loaded_host([load("a", "a", free=0)], 0) is None
        assert select_least_loaded_host([], 0) is None

    def test_eligibility_needs_both_conditions(self):
        assert load("a", "a", free=1, schedulable=True).eligible is True
        assert load("a", "a", free=0, schedulable=True).eligible is False
        assert load("a", "a", free=1, schedulable=False).eligible is False


# ── Determinism ──────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_ties_rotate_instead_of_repeating(self):
        hosts = [load("a", "a", 4), load("b", "b", 4), load("c", "c", 4)]
        picked = [select_least_loaded_host(hosts, i).name for i in range(6)]
        # With every host idle, always choosing the same one would queue work
        # behind one machine until its count rose.
        assert len(set(picked)) == 3
        assert picked == picked[:3] * 2

    def test_tie_order_follows_the_id_not_the_name(self):
        # A renamed host must not change where work goes.
        hosts = [load("id-a", "zebra", 4), load("id-b", "alpha", 4)]
        assert select_least_loaded_host(hosts, 0).host_id == "id-a"

    def test_same_snapshot_and_cursor_give_the_same_answer(self):
        hosts = [load("a", "a", 4, 1), load("b", "b", 4, 1), load("c", "c", 4, 0)]
        assert select_least_loaded_host(hosts, 7).name == \
            select_least_loaded_host(hosts, 7).name

    def test_input_order_does_not_matter(self):
        hosts = [load("a", "a", 4, 2), load("b", "b", 4, 1)]
        assert select_least_loaded_host(hosts, 0).name == \
            select_least_loaded_host(list(reversed(hosts)), 0).name


# ── Policy selection ─────────────────────────────────────────────────────────

class TestPolicyRegistry:
    def test_default_is_least_loaded_host(self):
        assert build_policy().name == "least_loaded_host"
        assert build_policy(None).name == "least_loaded_host"

    def test_named_policy_is_resolved(self):
        assert isinstance(build_policy("least_loaded_host"), LeastLoadedHostPolicy)

    def test_unknown_policy_falls_back_with_a_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="lib.scheduler.policy"):
            policy = build_policy("magic")
        # A typo in an env var must not stop the master-node from placing work.
        assert policy.name == "least_loaded_host"
        assert "magic" in caplog.text

    def test_policy_delegates_to_the_function(self):
        hosts = [load("a", "a", 1, 0), load("b", "b", 1, 5)]
        assert LeastLoadedHostPolicy().select(hosts, 0).name == "a"


# ── Inventory helpers ────────────────────────────────────────────────────────

class TestInventoryHelpers:
    def test_capacity_counts_only_schedulable_hosts(self):
        hosts = [load("a", "a", 4, 1), load("b", "b", 8, 2, schedulable=False)]
        assert total_capacity(hosts) == (4, 1)

    def test_describe_is_loggable_and_marks_offline_hosts(self):
        described = describe_loads(
            [load("a", "host-a", 4, 1), load("b", "host-b", 2, 0, schedulable=False)])
        assert "host-a[1a/4f]" in described
        assert "host-b[0a/2f off]" in described

    def test_describe_with_no_hosts(self):
        assert describe_loads([]) == "no hosts registered"

    def test_find_load_by_id(self):
        hosts = [load("a", "a", 1), load("b", "b", 1)]
        assert find_load(hosts, "b").name == "b"
        assert find_load(hosts, "zzz") is None



def place_with(select, loads: list[HostLoad], count: int) -> dict[str, int]:
    """``place`` for an arbitrary selector, keeping the speeds fixed.

    A host's measured speed does not change as work is placed on it — that is
    the point of measuring occupancy rather than queue length — so the loop
    only mutates the free/active pair.
    """
    state = {l.name: {"free": l.free, "active": l.active, "id": l.host_id,
                      "sched": l.schedulable, "sec": l.fastest_free_sec}
             for l in loads}
    assigned: dict[str, int] = {name: 0 for name in state}
    cursor = 0
    for _ in range(count):
        snapshot = [
            load(v["id"], name, v["free"], v["active"], v["sched"], v["sec"])
            for name, v in state.items()
        ]
        chosen = select(snapshot, cursor)
        if chosen is None:
            break
        cursor += 1
        assigned[chosen.name] += 1
        state[chosen.name]["free"] -= 1
        state[chosen.name]["active"] += 1
    return assigned


HETEROGENEOUS = [
    ("a", "local", 8, 111.0),
    ("b", "vm7", 2, 139.0),
    ("c", "vm8", 2, 141.0),
]


def pool(active: int = 0) -> list[HostLoad]:
    """The measured three-host pool: 8+2+2 containers, 111/139/141 s."""
    return [load(i, n, free, active=active, fastest_free_sec=sec)
            for i, n, free, sec in HETEROGENEOUS]


class TestSpeedRestrictsTheCandidates:
    """The rank narrows who may be chosen; the original rule decides inside.

    Breaking the *tie* instead — the obvious design — was implemented first
    and does nothing, which is why the first test asserts it rather than the
    prose describing it: after one placement the fast host is strictly more
    loaded, so the count sends the next job to a slow host before any tie can
    occur.

    The primary rule is left alone because it is already speed-aware in a way
    no statistic matches: a slower host holds its containers longer, so its
    count stays high and it stops being selected. On a three-host pool the
    resulting split matched measured throughput to within half a point. What
    it cannot do is leave a slow host **unused** while a fast one is free,
    and that is the whole cost of a batch's tail.
    """

    def test_breaking_the_tie_alone_would_not_have_worked(self):
        """Four jobs, twelve free containers. The count spreads them one per
        host before a tie can happen, so a tie-break never fires."""
        assert place_with(select_least_loaded_host, pool(), 4) == {
            "local": 2, "vm7": 1, "vm8": 1}

    def test_slack_keeps_the_batch_on_the_fast_host(self):
        """The same four jobs. The count makes the batch wait 141 s; the
        restriction makes it wait 111 s."""
        assert place_with(select_fastest_then_least_loaded, pool(), 4) == {
            "local": 4, "vm7": 0, "vm8": 0}

    def test_a_slower_host_loses_even_when_it_is_less_loaded(self):
        chosen = select_fastest_then_least_loaded([
            load("a", "local", free=4, active=5, fastest_free_sec=111.0),
            load("b", "vm8", free=4, active=1, fastest_free_sec=141.0),
        ], rr_cursor=0)

        assert chosen.name == "local", (
            "carrying four more runs and still finishing this one sooner")

    def test_a_host_within_the_tolerance_still_competes(self):
        """Below the tolerance the difference is inside the noise of a
        per-worker mean, and spreading the work is worth more than chasing
        it."""
        chosen = select_fastest_then_least_loaded([
            load("a", "one", free=4, active=1, fastest_free_sec=111.0),
            load("b", "two", free=4, active=0, fastest_free_sec=118.0),
        ], rr_cursor=0)

        assert chosen.name == "two", "7% apart, so the count still decides"

    def test_the_slow_hosts_take_over_once_the_fast_one_fills(self):
        """What makes the restriction safe: a host with no free container is
        not eligible, so it cannot hold work back."""
        assert place_with(select_fastest_then_least_loaded, pool(), 12) == {
            "local": 8, "vm7": 2, "vm8": 2}

    def test_a_queue_longer_than_the_pool_uses_all_of_it(self):
        assert place_with(select_fastest_then_least_loaded, [
            load("a", "local", free=8, fastest_free_sec=111.0),
            load("b", "vm8", free=4, fastest_free_sec=141.0),
        ], 50) == {"local": 8, "vm8": 4}

    def test_under_saturation_there_is_no_decision_to_change(self):
        """Why this is safe to turn on. With every container busy the
        dispatcher waits, and the host that releases one is the only
        candidate — so the throughput-proportional split the count produces
        is untouched."""
        chosen = select_fastest_then_least_loaded([
            load("a", "local", free=0, active=8, fastest_free_sec=111.0),
            load("b", "vm8", free=1, active=1, fastest_free_sec=141.0),
        ], rr_cursor=0)

        assert chosen.name == "vm8", "the slow host is the only one free"

    def test_an_unmeasured_host_is_sampled_first(self):
        """Optimistic and self-limiting: one run gives it a real number. The
        pessimistic reading would leave a newly registered host unused for as
        long as the pool has slack — exactly when it should be gathering its
        first samples."""
        chosen = select_fastest_then_least_loaded([
            load("a", "local", free=4, active=1, fastest_free_sec=111.0),
            load("b", "new-host", free=4, active=1, fastest_free_sec=None),
        ], rr_cursor=0)

        assert chosen.name == "new-host"

    def test_a_pool_with_no_measurements_behaves_exactly_as_before(self):
        """Every deployment is in this state on its first run."""
        loads = [load("a", "big", free=8), load("b", "mid", free=4),
                 load("c", "small", free=2)]

        assert place_with(select_fastest_then_least_loaded, loads, 6) == \
            place_with(select_least_loaded_host, loads, 6)

    def test_a_homogeneous_pool_keeps_the_specified_split(self):
        """8/4/2 containers, six simulations, still 2/2/2. Every host is
        inside the tolerance, so the candidate set is everything and the
        behaviour the feature exists for is untouched."""
        loads = [load("a", "big", free=8, fastest_free_sec=100.0),
                 load("b", "mid", free=4, fastest_free_sec=100.0),
                 load("c", "small", free=2, fastest_free_sec=100.0)]

        assert place_with(select_fastest_then_least_loaded, loads, 6) == {
            "big": 2, "mid": 2, "small": 2}

    def test_indistinguishable_hosts_still_rotate(self):
        """The original rule's rotation applies inside the shortlist, so two
        machines of equal speed do not queue behind whichever id sorts
        first."""
        loads = [load("a", "one", free=4, active=1, fastest_free_sec=120.0),
                 load("b", "two", free=4, active=1, fastest_free_sec=120.0)]

        assert {select_fastest_then_least_loaded(loads, c).name
                for c in range(4)} == {"one", "two"}

    def test_two_unmeasured_hosts_also_rotate(self):
        loads = [load("a", "one", free=4, active=1),
                 load("b", "two", free=4, active=1)]

        assert {select_fastest_then_least_loaded(loads, c).name
                for c in range(4)} == {"one", "two"}

    def test_it_is_deterministic_for_one_snapshot(self):
        loads = pool()

        assert len({select_fastest_then_least_loaded(loads, 0).name
                    for _ in range(10)}) == 1

    def test_nothing_eligible_is_still_no_capacity(self):
        """Held, not failed — the dispatcher's cue to wait."""
        assert select_fastest_then_least_loaded([
            load("a", "local", free=0, active=3, fastest_free_sec=111.0),
            load("b", "vm8", free=4, active=1, schedulable=False),
        ], rr_cursor=0) is None


class TestThePolicyIsSelectable:
    def test_it_is_registered_under_its_name(self):
        policy = build_policy("fastest_then_least_loaded")

        assert isinstance(policy, FastestThenLeastLoadedPolicy)
        assert policy.name == "fastest_then_least_loaded"

    def test_the_default_is_unchanged(self):
        assert isinstance(build_policy(), LeastLoadedHostPolicy)
        assert isinstance(build_policy(None), LeastLoadedHostPolicy)

    def test_only_the_ranked_policy_orders_within_the_host(self):
        """Two halves of one decision: picking the host that finishes soonest
        and then handing it the slowest container would undo half the
        choice."""
        assert FastestThenLeastLoadedPolicy().prefers_fastest is True
        assert LeastLoadedHostPolicy().prefers_fastest is False
