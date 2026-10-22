"""The topology the graph page draws: master-node, hosts, workers, and runs.

Most of it is assembly, tested on documents directly. The two parts that
decide something are tested on their own: whether the master-node is alive —
nothing records that, it is inferred from its writes — and which worker is
running what.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from bson import ObjectId

from pylib.timeutil import utc_now

from api.mappers.topology import (
    MASTER_ALIVE,
    MASTER_STALE,
    MASTER_STALE_AFTER_SEC,
    MASTER_UNKNOWN,
    master_status,
    topology_from_documents,
)

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
LOCAL = ObjectId("507f1f77bcf86cd799470001")
VM7 = ObjectId("507f1f77bcf86cd799470002")
EXP = ObjectId("507f1f77bcf86cd799470010")
GEN = ObjectId("507f1f77bcf86cd799470020")
SIM = ObjectId("507f1f77bcf86cd799470030")


def host(_id, name, *, type_="remote", status="online", seconds_ago=10,
         enabled=True, **over):
    doc = {"_id": _id, "name": name, "type": type_, "status": status,
           "enabled": enabled, "last_check": NOW - timedelta(seconds=seconds_ago),
           "last_check_message": "ready"}
    doc.update(over)
    return doc


def worker(name, host_id, status="available", **over):
    doc = {"_id": ObjectId(), "name": name, "host_id": host_id, "status": status,
           "enabled": True, "last_check_message": "ready"}
    doc.update(over)
    return doc


def assemble(hosts, resources, **over):
    args = dict(
        hosts=hosts, schedulable={str(h["_id"]): True for h in hosts},
        resources=resources, simulations=[], experiments=[], generations=[],
        running_experiments=[], now=NOW)
    args.update(over)
    return topology_from_documents(**args)


class TestIsTheMasterAlive:
    """Nothing records the master-node's liveness. Its health cycle writes
    ``last_check`` on every enabled host every cycle — even an unreachable
    one — so silence across *all* of them means the cycle is not running."""

    def test_a_recent_check_on_any_host_means_alive(self):
        result = master_status([host(LOCAL, "local", seconds_ago=20)], NOW)

        assert result["status"] == MASTER_ALIVE
        assert result["age_sec"] == 20

    def test_no_check_for_longer_than_the_threshold_means_stale(self):
        result = master_status(
            [host(LOCAL, "local", seconds_ago=MASTER_STALE_AFTER_SEC + 5)], NOW)

        assert result["status"] == MASTER_STALE

    def test_an_unreachable_host_still_proves_the_cycle_runs(self):
        """A failed check is still a check: the write happens either way."""
        result = master_status(
            [host(VM7, "vm7", status="offline", seconds_ago=15)], NOW)

        assert result["status"] == MASTER_ALIVE

    def test_a_disabled_host_is_not_evidence(self):
        """Never checked, so old by design — counting it would make a healthy
        master look stopped."""
        result = master_status([
            host(LOCAL, "local", seconds_ago=10),
            host(VM7, "vm7", enabled=False, seconds_ago=99999),
        ], NOW)

        assert result["status"] == MASTER_ALIVE

    def test_nothing_to_observe_is_unknown_not_dead(self):
        assert master_status([], NOW)["status"] == MASTER_UNKNOWN
        assert master_status([host(VM7, "vm7", enabled=False)], NOW)["status"] == (
            MASTER_UNKNOWN)

    def test_naive_timestamps_are_read_as_utc(self):
        stamp = (NOW - timedelta(seconds=30)).replace(tzinfo=None)

        result = master_status([host(LOCAL, "local", last_check=stamp)], NOW)

        assert result["status"] == MASTER_ALIVE


class TestTheGraph:
    def test_workers_hang_under_their_host(self):
        result = assemble(
            [host(LOCAL, "local", type_="local"), host(VM7, "vm7")],
            [worker("cooja1", LOCAL), worker("vm7-cooja1", VM7),
             worker("cooja2", LOCAL)])

        by_name = {h["name"]: [w["name"] for w in h["workers"]] for h in result["hosts"]}
        assert by_name == {"local": ["cooja1", "cooja2"], "vm7": ["vm7-cooja1"]}

    def test_the_local_host_comes_first(self):
        result = assemble([host(VM7, "a-remote"), host(LOCAL, "local", type_="local")], [])

        assert [h["name"] for h in result["hosts"]] == ["local", "a-remote"]

    def test_schedulability_comes_from_the_snapshot(self):
        """Not re-derived: a second rule for "can this host take work" would
        disagree with the dispatcher's sooner or later."""
        result = assemble([host(LOCAL, "local")], [],
                          schedulable={str(LOCAL): False})

        assert result["hosts"][0]["schedulable"] is False

    def test_a_held_worker_says_so_and_why(self):
        result = assemble([host(LOCAL, "local")], [
            worker("cooja1", LOCAL, status="disabled", enabled=False,
                   held_for={"kind": "collection_pending"})])

        w = result["hosts"][0]["workers"][0]
        assert w["held"] is True and w["held_kind"] == "collection_pending"

    def test_a_host_being_removed_is_flagged(self):
        result = assemble([host(VM7, "vm7", withdrawn_at=NOW)], [])

        assert result["hosts"][0]["withdrawing"] is True


class TestWhatIsRunningWhere:
    def _busy(self, status="busy", **over):
        return worker("cooja1", LOCAL, status=status, current_simulation_id=SIM,
                      started_at=NOW - timedelta(seconds=40), **over)

    def _documents(self):
        return dict(
            simulations=[{"_id": SIM, "experiment_id": EXP, "generation_id": GEN,
                          "status": "Running", "random_seed": 4242}],
            experiments=[{"_id": EXP, "name": "pilot"}],
            generations=[{"_id": GEN, "index": 3}],
            running_experiments=[{"_id": EXP, "name": "pilot",
                                  "parameters": {"strategy": "nsga3"}}],
        )

    def test_a_busy_worker_carries_its_simulation_and_experiment(self):
        result = assemble([host(LOCAL, "local")], [self._busy()], **self._documents())

        run = result["hosts"][0]["workers"][0]["run"]
        assert run["experiment_name"] == "pilot"
        assert run["generation_index"] == 3
        assert run["random_seed"] == 4242
        assert run["started_at"] == NOW - timedelta(seconds=40)

    def test_a_reservation_is_also_a_run(self):
        """Reserved means the container is held for that simulation, even
        before the JVM starts."""
        result = assemble([host(LOCAL, "local")], [self._busy(status="reserved")],
                          **self._documents())

        assert result["hosts"][0]["workers"][0]["run"] is not None

    def test_a_free_worker_with_a_stale_pointer_runs_nothing(self):
        """The status is what the dispatcher acts on; a leftover
        ``current_simulation_id`` on an available worker is not a run."""
        result = assemble([host(LOCAL, "local")],
                          [self._busy(status="available")], **self._documents())

        assert result["hosts"][0]["workers"][0]["run"] is None

    def test_running_experiments_count_their_workers_per_host(self):
        sims = [{"_id": ObjectId(), "experiment_id": EXP, "generation_id": GEN}
                for _ in range(3)]
        resources = [
            worker("cooja1", LOCAL, status="busy", current_simulation_id=sims[0]["_id"]),
            worker("cooja2", LOCAL, status="busy", current_simulation_id=sims[1]["_id"]),
            worker("vm7-cooja1", VM7, status="busy", current_simulation_id=sims[2]["_id"]),
        ]

        result = assemble(
            [host(LOCAL, "local"), host(VM7, "vm7")], resources,
            simulations=sims, experiments=[{"_id": EXP, "name": "pilot"}],
            running_experiments=[{"_id": EXP, "name": "pilot"}])

        assert result["running_experiments"] == [{
            "id": str(EXP), "name": "pilot", "strategy": None,
            "on_workers": 3, "by_host": {"local": 2, "vm7": 1}}]

    def test_a_running_experiment_with_nothing_placed_is_still_listed(self):
        """Waiting in the queue is a state worth seeing: running, and on no
        worker."""
        result = assemble([host(LOCAL, "local")], [],
                          running_experiments=[{"_id": EXP, "name": "queued"}])

        assert result["running_experiments"][0]["on_workers"] == 0


def connection_returning(collections: dict):
    db = MagicMock()
    db.__getitem__.side_effect = lambda name: collections[name]

    @contextmanager
    def connect():
        yield db

    conn = MagicMock()
    conn.connect = connect
    return conn


class TestTheRoute:
    def test_it_answers_with_a_fixed_number_of_queries(self, client, mock_factory):
        """One ``$in`` per kind of document, not one request per worker."""
        busy = worker("cooja1", LOCAL, status="busy", current_simulation_id=SIM)
        # The route reads the real clock, so the check has to be recent by it.
        mock_factory.host_repo.find_all.return_value = [
            host(LOCAL, "local", last_check=utc_now())]
        mock_factory.resource_repo.find_all.return_value = [busy, worker("cooja2", LOCAL)]
        mock_factory.host_repo.inventory_snapshot.return_value = [
            {"_id": LOCAL, "schedulable": True}]
        sims, exps, gens = MagicMock(), MagicMock(), MagicMock()
        sims.find.return_value = [{"_id": SIM, "experiment_id": EXP, "generation_id": GEN}]
        exps.find.side_effect = [[{"_id": EXP, "name": "pilot", "status": "Running"}], []]
        gens.find.return_value = [{"_id": GEN, "index": 0}]
        mock_factory.simulation_repo.connection = connection_returning(
            {"simulations": sims, "experiments": exps, "generations": gens})

        resp = client.get("/api/v1/infra/topology")

        assert resp.status_code == 200
        body = resp.json()
        assert body["master"]["status"] == MASTER_ALIVE
        workers = body["hosts"][0]["workers"]
        assert workers[0]["run"]["experiment_name"] == "pilot"
        assert workers[1]["run"] is None
        assert sims.find.call_count == 1
        assert sims.find.call_args.args[0] == {"_id": {"$in": [SIM]}}

    def test_an_idle_pool_does_not_query_simulations(self, client, mock_factory):
        mock_factory.host_repo.find_all.return_value = [host(LOCAL, "local")]
        mock_factory.resource_repo.find_all.return_value = [worker("cooja1", LOCAL)]
        mock_factory.host_repo.inventory_snapshot.return_value = []
        sims, exps, gens = MagicMock(), MagicMock(), MagicMock()
        exps.find.return_value = []
        mock_factory.simulation_repo.connection = connection_returning(
            {"simulations": sims, "experiments": exps, "generations": gens})

        resp = client.get("/api/v1/infra/topology")

        assert resp.status_code == 200
        sims.find.assert_not_called()
