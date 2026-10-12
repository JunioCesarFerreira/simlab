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
    HostLoad,
    LeastLoadedHostPolicy,
    build_policy,
    select_least_loaded_host,
)


def load(host_id, name, free, active=0, schedulable=True) -> HostLoad:
    return HostLoad(
        host_id=host_id, name=name, free=free, active=active, schedulable=schedulable)


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
