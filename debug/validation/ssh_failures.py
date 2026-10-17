"""Real SSH/JVM/SCP and MongoDB fault checks in disposable infrastructure.

Requires two disposable SSH servers with different keys and an isolated Cooja
container. Creates a unique MongoDB database and /tmp directory; removes both.
Use master-node's Python. See README.md in this directory for the fixture.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "master-node"), str(ROOT)]
from bson import ObjectId
import paramiko
from scp import SCPClient
from pylib.db.connection import MongoDBConnection
from pylib.db.repositories.generation import GenerationRepository
from pylib.db.repositories.simulation import SimulationRepository
from pylib.db.repositories.infra import HostRepository, CoojaResourceRepository
from pylib.db.models.enums import EnumStatus
from pylib.timeutil import utc_now
from lib.credentials import SSHCredential
from lib.detached import DetachedRun, STATUS_PREFIX, inspect_dir
from lib.health import HealthMonitor
from lib.scheduler.runner import LostOwnership, SimulationRunner
from lib.transport.base import TransportError
from lib.transport.hostkey import HostKeyMismatch
from lib.transport.plan import ConnectionPlan, SSHEndpoint
from lib.transport.session import connect_host
from lib.transport.ssh import SSHTransport

spec = importlib.util.spec_from_file_location("validation_master", ROOT / "master-node/master-node.py")
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)


def main(args):
    name = "simlab_validation_ssh_" + uuid.uuid4().hex
    remote = "/tmp/" + name
    connection = MongoDBConnection(args.mongo_uri, name, reuse_client=True)
    mongo = SimpleNamespace(generation_repo=GenerationRepository(connection),
                            simulation_repo=SimulationRepository(connection),
                            resource_repo=CoojaResourceRepository(connection),
                            host_repo=HostRepository(connection))
    host = {"_id": ObjectId(), "name": "validation", "type": "local", "enabled": True,
            "discovery": {"mode": "manual"}, "container_access": {"username": "root"}}
    credential = SSHCredential(username="root", password="root")
    plan = ConnectionPlan(SSHEndpoint("127.0.0.1", args.cooja_port), remote)
    results = []

    def transport():
        return SSHTransport(plan, credential)

    def worker(db):
        # Scenarios reuse one physical target, sequentially. Its previous
        # record has already been reconciled/released in this scratch DB.
        db.cooja_resources.delete_many({})
        sim_id, resource_id, generation_id = ObjectId(), ObjectId(), ObjectId()
        attempt = uuid.uuid4().hex
        sim = {"_id": sim_id, "generation_id": generation_id, "experiment_id": ObjectId(),
               "individual_id": "validation", "random_seed": 1, "status": EnumStatus.WAITING,
               "attempt_id": attempt}
        resource = {"_id": resource_id, "host_id": host["_id"], "name": str(resource_id),
                    "enabled": True, "status": "available", "origin": "manual",
                    "endpoint": {"address": "127.0.0.1", "port": args.cooja_port},
                    "remote_dir": remote}
        db.generations.insert_one({"_id": generation_id, "experiment_id": sim["experiment_id"], "index": 0, "status": EnumStatus.RUNNING, "expected_seeds": [1]})
        db.simulations.insert_one(sim)
        db.cooja_resources.insert_one(resource)
        assert mongo.resource_repo.reserve(resource_id, sim_id, attempt)
        return sim, resource, attempt

    try:
        with connection.connect() as db, tempfile.TemporaryDirectory(prefix="simlab-validation-") as local:
            db.hosts.insert_one(host)
            java = Path(local) / "ValidationSleep.java"
            java.write_text('public class ValidationSleep { public static void main(String[] args) throws Exception { System.out.println("validation JVM ready"); Thread.sleep(60000); } }')
            with transport() as tr:
                assert tr.exec_command(f"mkdir {shlex.quote(remote)}").ok
                tr.put_files([str(java)], [java.name])
            original_launch = DetachedRun.launch
            # Exercise the real executor and detached launcher with a small
            # real JVM, avoiding a Cooja experiment or user firmware as test data.
            def launch_jvm(run, command):
                return original_launch(run, f"/opt/java/openjdk/bin/java -Xms16m -Xmx64m {shlex.quote(remote + '/' + java.name)}")

            for failure in ["launch_ack", "pid_read"]:
                class DroppedConfirmation(SSHTransport):
                    launched = False

                    def exec_command(self, command, timeout=None):
                        if failure == "pid_read" and self.launched and "cat run.pid" in command:
                            self.close()
                        result = super().exec_command(command, timeout=timeout)
                        if f"{STATUS_PREFIX}-launched" in command:
                            self.launched = True
                        if failure == "launch_ack" and self.launched:
                            self.close()  # real socket close after sshd accepted the launch
                            raise TransportError("validation: launch confirmation lost")
                        return result

                sim, resource, attempt = worker(db)
                tr = DroppedConfirmation(plan, credential)
                runner = SimulationRunner(mongo, execute=entry.run_cooja_simulation,
                                          prepare=lambda *_: (True, [], []))
                with patch("lib.scheduler.runner.build_transport", return_value=tr), patch.object(DetachedRun, "launch", launch_jvm):
                    runner.run(sim, host, resource, attempt)
                held = mongo.resource_repo.get(resource["_id"])
                assert held["held_for"]["kind"] == "termination_unknown"
                assert mongo.resource_repo.reserve(resource["_id"], ObjectId(), "competitor") is None
                assert not mongo.resource_repo.delete_if_free(resource["_id"])
                db.cooja_resources.insert_one({
                    "host_id": host["_id"], "name": "free-validation", "enabled": True,
                    "status": "available", "remote_dir": remote + "/free",
                    "endpoint": {"address": "127.0.0.1", "port": args.cooja_port},
                })
                assert mongo.host_repo.delete_if_drained(host["_id"]) == (False, [resource["name"]])
                assert db.cooja_resources.count_documents({"host_id": host["_id"]}) == 2
                assert not mongo.host_repo.is_withdrawing(host["_id"])
                with transport() as fresh:
                    # Source-file startup may compile first. Confirm the real
                    # JVM remains alive independently of the dead SSH channel.
                    time.sleep(0.15)
                    state = inspect_dir(fresh, remote_dir=remote)
                    assert state.alive, state
                    command = fresh.exec_command(f"cd {shlex.quote(remote)} && tr '\\0' ' ' < /proc/$(cat run.pid)/cmdline")
                    assert "java" in command.stdout
                    monitor = HealthMonitor(mongo, inspect_run=lambda *_: inspect_dir(fresh, remote_dir=remote))
                    assert monitor.resolve_held_workers() == []
                    # Kill only the process group launched inside this owned directory.
                    fresh.exec_command(f"cd {shlex.quote(remote)} && kill -TERM -- -$(cat run.pgid)")
                    for _ in range(40):
                        if not inspect_dir(fresh, remote_dir=remote).alive:
                            break
                        time.sleep(0.05)
                    assert not inspect_dir(fresh, remote_dir=remote).alive
                    monitor.resolve_held_workers()
                    assert mongo.resource_repo.get(resource["_id"]).get("held_for") is None
                results.append({"check": "P01/P02", "failure": failure, "jvm_survived": True,
                                "competitor_refused": True, "hold_survived_reconciliation": True,
                                "hold_resolved_after_stop": True, "held_deletion_refused": True,
                                "host_removal_rolled_back": True})

            # Real SCP is throttled via its progress callback, so it exceeds a
            # short real TTL while MongoDB's sweep keeps running concurrently.
            payload = Path(local) / "payload"
            payload.write_bytes(b"x" * (512 * 1024))
            for budget in [5.0, 0.15]:
                sim, resource, attempt = worker(db)
                monitor = HealthMonitor(mongo, reservation_ttl_sec=0.3)
                stop = threading.Event()
                swept, competitors, sweep_errors = [], [], []

                def sweep():
                    while not stop.wait(0.03):
                        try:
                            swept.extend(monitor.sweep_stale_reservations())
                            competitors.append(mongo.resource_repo.reserve(resource["_id"], ObjectId(), "competitor"))
                        except Exception as error:
                            sweep_errors.append(str(error))

                watcher = threading.Thread(target=sweep)
                runner = SimulationRunner(mongo, execute=lambda *_: None, prepare=lambda *_: None,
                                          lease_heartbeat_sec=0.04, prepare_budget_sec=budget)
                watcher.start()
                cancelled = False
                started = time.monotonic()
                try:
                    with transport() as tr:
                        tr._scp = lambda: SCPClient(tr._active.get_transport(), socket_timeout=1,
                                                    progress=lambda *_: time.sleep(0.06))
                        try:
                            with runner._lease(resource, sim["_id"], attempt, abort=tr.close):
                                tr.put_files([str(payload)], ["slow-upload"])
                        except LostOwnership:
                            cancelled = True
                        assert mongo.resource_repo.is_held_by(resource["_id"], sim["_id"], attempt)
                finally:
                    stop.set()
                    watcher.join(timeout=5)
                    assert not watcher.is_alive()
                elapsed = time.monotonic() - started
                assert elapsed > 0.3
                assert cancelled == (budget < 1)
                assert not swept and not any(competitors) and not sweep_errors
                assert mongo.resource_repo.release(resource["_id"], sim["_id"], attempt, container_state_counts=False)
                results.append({"check": "P04", "ttl_seconds": 0.3, "budget_seconds": budget,
                                "elapsed_seconds": round(elapsed, 3), "cancelled": cancelled,
                                "sweeps": len(competitors), "premature_releases": len(swept)})

            # Real handshakes from different servers contend for ONE empty pin.
            pin_host = {"_id": ObjectId(), "name": "validation-pin", "access": {}}
            db.hosts.insert_one(pin_host)
            monitor = HealthMonitor(mongo)
            barrier = threading.Barrier(2)
            key = paramiko.Ed25519Key.from_private_key_file(args.private_key)
            clients = []
            real_client = paramiko.SSHClient

            def client_factory():
                client = real_client()
                clients.append(client)
                return client

            def contact(port):
                def learn(fp):
                    barrier.wait(timeout=10)
                    monitor._pin_host_key(pin_host, fp)
                try:
                    client = connect_host(SSHEndpoint("127.0.0.1", port), SSHCredential("root", pkey=key),
                                          on_learn_host_key=learn, timeout=10)
                    _, stdout, _ = client.exec_command("printf accepted")
                    assert stdout.read() == b"accepted"
                    client.close()
                    return "accepted"
                except HostKeyMismatch:
                    return "refused"

            with patch("lib.transport.session.paramiko.SSHClient", side_effect=client_factory), ThreadPoolExecutor(max_workers=2) as pool:
                contacts = list(pool.map(contact, [args.ssh_a_port, args.ssh_b_port]))
            assert sorted(contacts) == ["accepted", "refused"]
            assert all(c.get_transport() is None or not c.get_transport().is_active() for c in clients)
            assert mongo.host_repo.get(pin_host["_id"])["access"]["host_key_fingerprint"]
            results.append({"check": "P08", "real_handshakes": 2, "results": sorted(contacts), "sessions_closed": True})
    finally:
        with transport() as tr:
            # Path is generated here; never accepts a caller-supplied directory.
            tr.exec_command(f"cd {shlex.quote(remote)} 2>/dev/null && if [ -s run.pgid ]; then kill -TERM -- -$(cat run.pgid) 2>/dev/null || true; fi")
            tr.exec_command(f"rm -rf -- {shlex.quote(remote)}")
        connection._client.drop_database(name)
        assert name not in connection._client.list_database_names()
        connection.close()
    print(json.dumps({"checks": results, "scratch_database_removed": True, "remote_directory_removed": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--cooja-port", type=int, required=True)
    parser.add_argument("--ssh-a-port", type=int, required=True)
    parser.add_argument("--ssh-b-port", type=int, required=True)
    parser.add_argument("--private-key", required=True)
    main(parser.parse_args())
