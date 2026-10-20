"""Kill/restart the replica gate against real MongoDB, in an owned scratch DB.

Run with mo-engine's Python: retry_restart.py --mongo-uri mongodb://.../?directConnection=true
The URI selects a server, never a database to modify. A unique validation DB
is created and removed; no existing experiment or collection is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "mo-engine"), str(ROOT)]
from bson import ObjectId
from pylib.db.connection import MongoDBConnection
from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.generation import GenerationRepository
from pylib.db.repositories.simulation import SimulationRepository
from pylib.db.repositories.infra import HostRepository
from lib.strategy.completeness import ReplicaGate, ReplicaPolicy


def repositories(uri, name):
    connection = MongoDBConnection(uri, name, reuse_client=True)
    return connection, SimpleNamespace(
        generation_repo=GenerationRepository(connection),
        simulation_repo=SimulationRepository(connection))


def child(args):
    connection, mongo = repositories(args.mongo_uri, args.database)
    if args.withdraw_host:
        hosts = HostRepository(connection)
        if args.crash == "withdrawal":
            begin = hosts.begin_withdrawal

            def interrupted_withdrawal(host_id):
                assert begin(host_id)
                os.kill(os.getpid(), signal.SIGKILL)

            hosts.begin_withdrawal = interrupted_withdrawal
        try:
            assert hosts.delete_if_drained(ObjectId(args.withdraw_host)) == (True, [])
        finally:
            connection.close()
        return
    if args.crash == "claim":
        claim = mongo.generation_repo.claim_retry_round

        def interrupted_claim(*a, **kw):
            result = claim(*a, **kw)
            assert result == 1
            os.kill(os.getpid(), signal.SIGKILL)

        mongo.generation_repo.claim_retry_round = interrupted_claim
    elif args.crash == "clone":
        clone = mongo.simulation_repo.clone_for_retry

        def interrupted_clone(*a, **kw):
            assert clone(*a, **kw) is not None
            os.kill(os.getpid(), signal.SIGKILL)

        mongo.simulation_repo.clone_for_retry = interrupted_clone
    try:
        result = ReplicaGate(mongo, objective_keys=["latency"], policy=ReplicaPolicy()).evaluate(ObjectId(args.generation))
        assert result.proceed is False
    finally:
        connection.close()


def main(args):
    name = "simlab_validation_retry_" + uuid.uuid4().hex
    connection, mongo = repositories(args.mongo_uri, name)
    results = []
    try:
        with connection.connect() as db:
            for checkpoint, expected_before in [("claim", 0), ("clone", 1)]:
                generation, experiment = ObjectId(), ObjectId()
                db.generations.insert_one({"_id": generation, "experiment_id": experiment,
                                           "status": EnumStatus.ERROR, "expected_seeds": [1, 2]})
                sources = []
                for seed in [1, 2]:
                    sources.append(db.simulations.insert_one({
                        "generation_id": generation, "experiment_id": experiment,
                        "individual_id": "validation", "random_seed": seed,
                        "status": EnumStatus.ERROR,
                    }).inserted_id)
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--mongo-uri", args.mongo_uri, "--database", name,
                           "--generation", str(generation)]
                killed = subprocess.run(command + ["--crash", checkpoint], timeout=30)
                assert killed.returncode == -signal.SIGKILL
                assert db.generations.find_one({"_id": generation})["replica_retries"] == 1
                assert db.simulations.count_documents({"generation_id": generation, "retry_round": 1}) == expected_before
                # Resume with the first clone RUNNING: that must not hide the
                # other replica still missing from the persisted manifest.
                db.simulations.update_many({"generation_id": generation, "retry_round": 1},
                                           {"$set": {"status": EnumStatus.RUNNING}})
                subprocess.run(command, timeout=30, check=True)
                clones = list(db.simulations.find({"generation_id": generation, "retry_round": 1}))
                assert len(clones) == 2 and {c["random_seed"] for c in clones} == {1, 2}
                assert db.generations.find_one({"_id": generation})["replica_retries"] == 1
                assert mongo.generation_repo.pending_retry_round(generation) is None
                # Concurrent replay must return the same id, without resetting
                # a clone that has already completed.
                db.simulations.update_many({"generation_id": generation, "retry_round": 1},
                                           {"$set": {"status": EnumStatus.DONE, "network_metrics": {"latency": 1}}})
                with ThreadPoolExecutor(max_workers=12) as pool:
                    ids = list(pool.map(lambda _: mongo.simulation_repo.clone_for_retry(sources[0], retry_round=1), range(24)))
                assert len(set(ids)) == 1
                assert db.simulations.count_documents({"generation_id": generation, "retry_round": 1}) == 2
                assert db.simulations.find_one({"_id": ids[0]})["status"] == EnumStatus.DONE
                assert mongo.generation_repo.close_if_finished(generation) == EnumStatus.DONE
                results.append({"checkpoint": checkpoint, "signal": "SIGKILL", "clones_before": expected_before,
                                "clones_after": 2, "rounds": 1, "concurrent_replays": 24, "status": "Done"})

            generation = ObjectId()
            db.generations.insert_one({"_id": generation, "status": EnumStatus.ERROR})
            replicas = [{"simulation_id": ObjectId(), "reason": "validation"}]
            with ThreadPoolExecutor(max_workers=12) as pool:
                claims = list(pool.map(lambda _: mongo.generation_repo.claim_retry_round(generation, 3, replicas), range(24)))
            assert claims.count(1) == 1 and claims.count(None) == 23
            assert mongo.generation_repo.finish_retry_round(generation, 1)
            assert mongo.generation_repo.claim_retry_round(generation, 3, replicas) == 2
            assert not mongo.generation_repo.finish_retry_round(generation, 1)
            assert mongo.generation_repo.pending_retry_round(generation)["replica_retries"] == 2
            results.append({"simultaneous_claims": 24, "accepted": 1, "stale_finish_refused": True})

            source = db.simulations.insert_one({
                "generation_id": ObjectId(), "experiment_id": ObjectId(),
                "individual_id": "concurrent-insert", "random_seed": 42,
                "status": EnumStatus.ERROR,
            }).inserted_id
            with ThreadPoolExecutor(max_workers=12) as pool:
                ids = list(pool.map(lambda _: mongo.simulation_repo.clone_for_retry(source, retry_round=1), range(24)))
            assert len(set(ids)) == 1
            assert db.simulations.count_documents({"retry_of": source}) == 1
            results.append({"simultaneous_first_clones": 24, "documents_created": 1})

            host_id = db.hosts.insert_one({"name": "withdrawal-validation", "enabled": True}).inserted_id
            db.cooja_resources.insert_many([
                {"host_id": host_id, "name": f"worker-{n}", "status": "available"}
                for n in range(2)
            ])
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--mongo-uri", args.mongo_uri, "--database", name,
                       "--withdraw-host", str(host_id)]
            killed = subprocess.run(command + ["--crash", "withdrawal"], timeout=30)
            assert killed.returncode == -signal.SIGKILL
            assert db.hosts.find_one({"_id": host_id}).get("withdrawn_at")
            assert db.cooja_resources.count_documents({"host_id": host_id}) == 2
            subprocess.run(command, timeout=30, check=True)
            assert db.hosts.find_one({"_id": host_id}) is None
            assert db.cooja_resources.count_documents({"host_id": host_id}) == 0
            results.append({"check": "P09", "withdrawal_interrupted_by": "SIGKILL",
                            "resumed_in_new_process": True, "orphan_workers": 0})
    finally:
        connection._client.drop_database(name)
        assert name not in connection._client.list_database_names()
        connection.close()
    print(json.dumps({"checks": results, "scratch_database_removed": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--database", help=argparse.SUPPRESS)
    parser.add_argument("--generation", help=argparse.SUPPRESS)
    parser.add_argument("--withdraw-host", help=argparse.SUPPRESS)
    parser.add_argument("--crash", choices=["claim", "clone", "withdrawal"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.database:
        if not args.database.startswith("simlab_validation_retry_"):
            parser.error("child requires a validation database")
        child(args)
    else:
        main(args)
