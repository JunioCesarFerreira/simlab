#!/usr/bin/env python3
"""Recover a simulation result that was produced and never collected.

Why this exists
---------------
When a run finishes and its result cannot be fetched — twice, over two
sessions — the master-node stops rather than losing it. The testlog is copied
on the container to a name the next run's cleanup does not know, where the copy
could be made, and the simulation records ``unfetched_result``: which worker,
which directory, which file, which process.

Nothing comes back for it automatically, and the worker is **held** out of the
pool until something does. That is deliberate: deciding what a half-collected
result is worth is not a decision to automate quietly. This tool is the other
half of that contract — the procedure that makes the retention resolvable
rather than permanent.

What it does
------------
``list``    shows the simulations carrying an ``unfetched_result``, and the
            workers currently held, so the two can be matched.

``show``    prints one record in full: where the bytes are, which attempt
            produced them, and what the worker is holding.

``fetch``   opens a session to the worker, brings the preserved testlog back,
            and writes it next to the others — **without** touching the
            simulation. Use it to look at the bytes before deciding.

``publish`` fetches, converts, stores both artifacts in GridFS and records
            them on the simulation. Refuses a simulation that already carries
            a result. The worker is *not* released by this: run ``release``
            once you have confirmed the result is where you want it.

``discard`` records that the result was given up on, with a reason, and
            releases the worker. The remote bytes are left alone; removing
            them is a separate decision and a separate command.

``release`` clears the hold on the worker and lets the health cycle measure
            it again. Refuses while the simulation still carries an
            unresolved ``unfetched_result``, so the order cannot be reversed
            by accident.

Usage
-----
    python util/recover_result.py list
    python util/recover_result.py show <simulation_id>
    python util/recover_result.py fetch <simulation_id> [--out DIR]
    python util/recover_result.py publish <simulation_id>
    python util/recover_result.py discard <simulation_id> --reason "..."
    python util/recover_result.py release <worker_name>

Environment
-----------
    MONGO_URI    default mongodb://localhost:27017/?replicaSet=rs0
    DB_NAME      default simlab
    INFRA_KEY_PATH  the keystore that opens the host credential; the same one
                    the master-node uses, and it has to be readable here

Limits worth knowing
--------------------
This reads and writes MongoDB directly and connects to the worker over SSH,
which means it needs the private key the master-node holds. It is an operator
tool, not an API: run it where the master-node runs.

It cannot recover what was never preserved. If ``remote_file`` is null the
second session never opened, so no copy was made — the original may still be
under ``COOJA.testlog``, and ``fetch --original`` will try that, but any run
that has since used the worker will have deleted it. The worker being held is
what makes that unlikely.
"""
import argparse
import os
import sys
from pathlib import Path

def _make_importable() -> None:
    """Put ``lib`` (master-node) and ``pylib`` on the path, from either layout.

    Run from the repository, they are the siblings ``master-node/`` and
    ``pylib/``. Run inside the master-node image, the code is at
    ``/app/master-node`` and ``/app/pylib`` — and this file may have been
    copied anywhere in that container, so the image's layout is named
    explicitly instead of being derived from ``__file__``.
    """
    here = Path(__file__).resolve().parents[1]
    for candidate in (here, here / "master-node", Path("/app"),
                      Path("/app/master-node")):
        if (candidate / "lib").is_dir() or (candidate / "pylib").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))


_make_importable()

from bson import ObjectId                                      # noqa: E402

from lib.credentials import SecretOpener                        # noqa: E402
from lib.transport import build_transport                       # noqa: E402
from pylib import cooja_files                                   # noqa: E402
from pylib.crypto.sealed import DEFAULT_KEY_PATH, Keystore      # noqa: E402
from pylib.db import create_mongo_repository_factory            # noqa: E402
from pylib.db.models.enums import EnumStatus                    # noqa: E402
from pylib.timeutil import utc_now                              # noqa: E402

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
KEY_PATH = os.getenv("INFRA_KEY_PATH", DEFAULT_KEY_PATH)

#: What the runner names a preserved testlog. Kept in one place here as well,
#: because an operator reading a directory listing needs to recognise it.
PRESERVED_PREFIX = "simlab-unfetched"


def _mongo():
    return create_mongo_repository_factory(MONGO_URI, DB_NAME)


def _opener() -> SecretOpener:
    return SecretOpener(Keystore.load_or_create(KEY_PATH))


def _simulation(mongo, simulation_id: str) -> dict:
    sim = mongo.simulation_repo.get(simulation_id)
    if sim is None:
        raise SystemExit(f"simulation {simulation_id} not found")
    return sim


def _record(sim: dict) -> dict:
    record = sim.get("unfetched_result")
    if not record:
        raise SystemExit(
            f"simulation {sim['_id']} has no unfetched result recorded")
    return record


def _worker(mongo, record: dict) -> tuple[dict, dict]:
    resource = mongo.resource_repo.get(record.get("resource_id"))
    if resource is None:
        raise SystemExit(
            "the worker that produced this result is no longer registered; "
            "the bytes may still be on the container, but nothing here knows "
            "how to reach it")
    host = mongo.host_repo.get(resource.get("host_id"))
    if host is None:
        raise SystemExit("the host of that worker is no longer registered")
    return host, resource


# ── commands ───────────────────────────────────────────────────────────────

def cmd_list(args) -> None:
    mongo = _mongo()
    with mongo.simulation_repo.connection.connect() as db:
        pending = list(db["simulations"].find(
            {"unfetched_result": {"$exists": True}},
            {"status": 1, "individual_id": 1, "random_seed": 1,
             "unfetched_result": 1},
        ))
    if pending:
        print(f"{len(pending)} simulation(s) with an uncollected result:\n")
        for sim in pending:
            record = sim.get("unfetched_result") or {}
            print(f"  {sim['_id']}  {sim.get('status')}")
            print(f"      individual {sim.get('individual_id')}  "
                  f"seed {sim.get('random_seed')}")
            print(f"      on {record.get('host_name')}/"
                  f"{record.get('resource_name')}  pid {record.get('remote_pid')}")
            print(f"      {record.get('remote_dir')}/"
                  f"{record.get('remote_file') or '(no copy was made)'}")
            print(f"      stage {record.get('stage')!r}: {record.get('reason')}")
    else:
        print("no simulation carries an uncollected result")

    held = mongo.resource_repo.find_held()
    if held:
        print(f"\n{len(held)} worker(s) held:\n")
        for resource in held:
            record = resource.get("held_for") or {}
            print(f"  {resource.get('name')}  {record.get('kind')}  "
                  f"since {record.get('at')}")
            print(f"      {record.get('reason')}")
    else:
        print("\nno worker is held")


def cmd_show(args) -> None:
    mongo = _mongo()
    sim = _simulation(mongo, args.simulation_id)
    record = _record(sim)
    print(f"simulation {sim['_id']}")
    print(f"  status      {sim.get('status')}")
    print(f"  experiment  {sim.get('experiment_id')}")
    print(f"  generation  {sim.get('generation_id')}")
    print(f"  individual  {sim.get('individual_id')}  "
          f"seed {sim.get('random_seed')}")
    print(f"  artifacts   log={sim.get('log_cooja_id')} "
          f"csv={sim.get('csv_log_id')}")
    print("\nuncollected result:")
    for key in ("host_name", "resource_name", "container_id", "attempt_id",
                "remote_pid", "remote_dir", "remote_file", "stage", "reason",
                "recorded_at"):
        print(f"  {key:13} {record.get(key)}")
    resource = mongo.resource_repo.get(record.get("resource_id"))
    if resource is not None:
        print(f"\nworker {resource.get('name')}: status "
              f"{resource.get('status')}, enabled {resource.get('enabled')}")
        print(f"  held_for  {resource.get('held_for')}")


def _fetch(mongo, sim: dict, record: dict, out_dir: Path,
           original: bool) -> Path:
    host, resource = _worker(mongo, record)
    remote_name = record.get("remote_file")
    if original or not remote_name:
        if not original:
            print("no copy was made; falling back to the original testlog. "
                  "Any run that has used this worker since will have deleted "
                  "it.")
        remote_name = "COOJA.testlog"
    out_dir.mkdir(parents=True, exist_ok=True)
    local = out_dir / f"sim_{sim['_id']}.log"
    transport = build_transport(host, resource, _opener())
    with transport:
        print(f"fetching {transport.plan.remote_dir}/{remote_name} "
              f"from {host.get('name')}/{resource.get('name')}")
        transport.get_file(remote_name, str(local))
    size = local.stat().st_size
    print(f"wrote {local} ({size} bytes)")
    if size == 0:
        raise SystemExit("the file is empty; nothing to recover")
    return local


def cmd_fetch(args) -> None:
    mongo = _mongo()
    sim = _simulation(mongo, args.simulation_id)
    _fetch(mongo, sim, _record(sim), Path(args.out), args.original)
    print("\nnothing was changed in the database. Use `publish` to record it, "
          "or `discard` to give it up.")


def cmd_publish(args) -> None:
    mongo = _mongo()
    sim = _simulation(mongo, args.simulation_id)
    if sim.get("status") == EnumStatus.DONE:
        raise SystemExit(
            "this simulation already carries a published result; recovering "
            "over it would replace a measurement")
    record = _record(sim)
    out_dir = Path(args.out)
    local = _fetch(mongo, sim, record, out_dir, args.original)

    csv_path = out_dir / f"sim_{sim['_id']}.csv"
    cooja_files.convert_cooja_log_to_csv(str(local), str(csv_path))
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise SystemExit(
            "the testlog converted to an empty CSV; there are no metrics to "
            "publish. Keep the bytes and use `discard` if that is the "
            "decision.")

    log_id = mongo.fs_handler.upload_file(str(local), "sim_result.log")
    csv_id = mongo.fs_handler.upload_file(str(csv_path), "sim_result.csv")
    print(f"stored log={log_id} csv={csv_id}")

    # Recorded as a *recovered* result, not as an ordinary one: it was
    # collected by hand, after the fact, and that belongs in the document.
    mongo.simulation_repo.update(ObjectId(sim["_id"]), {
        "log_cooja_id": log_id,
        "csv_log_id": csv_id,
        "recovered_result": {
            "at": utc_now(),
            "from": f"{record.get('remote_dir')}/{record.get('remote_file')}",
            "by": "util/recover_result.py",
        },
    })
    print("recorded on the simulation.")
    print("\nThe metrics were **not** computed and the status was not changed: "
          "whether this result should feed the optimiser is a decision about "
          "the experiment. The worker is still held — run `release` once you "
          "are satisfied.")


def cmd_discard(args) -> None:
    mongo = _mongo()
    sim = _simulation(mongo, args.simulation_id)
    record = _record(sim)
    mongo.simulation_repo.update(ObjectId(sim["_id"]), {
        "unfetched_result.discarded": {
            "at": utc_now(), "reason": args.reason,
        },
    })
    print(f"recorded the decision to give up on {sim['_id']}: {args.reason}")
    print(f"the bytes are still at {record.get('remote_dir')}/"
          f"{record.get('remote_file')}; removing them is a separate step.")
    print("the worker is still held — run `release` to let it back in.")


def cmd_release(args) -> None:
    mongo = _mongo()
    held = [r for r in mongo.resource_repo.find_held()
            if r.get("name") == args.worker]
    if not held:
        raise SystemExit(f"no held worker named {args.worker!r}")
    resource = held[0]
    record = resource.get("held_for") or {}
    simulation_id = record.get("simulation_id")
    if simulation_id:
        sim = mongo.simulation_repo.get(str(simulation_id)) or {}
        unresolved = sim.get("unfetched_result") or {}
        if unresolved and not (unresolved.get("discarded")
                               or sim.get("recovered_result")):
            raise SystemExit(
                f"simulation {simulation_id} still has an unresolved result. "
                "Publish it or discard it first — releasing the worker lets "
                "the next run delete the bytes.")
    if mongo.resource_repo.clear_held_for(resource["_id"]):
        print(f"{args.worker} is back in service; the next health cycle will "
              "measure it.")
    else:
        print(f"{args.worker} was no longer held.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="simulations with an uncollected result, and "
                                "the workers held")

    show = sub.add_parser("show", help="one record in full")
    show.add_argument("simulation_id")

    for name, help_text in (("fetch", "bring the bytes back, change nothing"),
                            ("publish", "bring them back and record them")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("simulation_id")
        cmd.add_argument("--out", default="recovered",
                         help="where to write the files (default: recovered/)")
        cmd.add_argument(
            "--original", action="store_true",
            help="read COOJA.testlog instead of the preserved copy")

    discard = sub.add_parser("discard", help="record giving up on a result")
    discard.add_argument("simulation_id")
    discard.add_argument("--reason", required=True)

    release = sub.add_parser("release", help="let a held worker back in")
    release.add_argument("worker", help="the worker's name")

    args = parser.parse_args()
    {
        "list": cmd_list, "show": cmd_show, "fetch": cmd_fetch,
        "publish": cmd_publish, "discard": cmd_discard, "release": cmd_release,
    }[args.command](args)


if __name__ == "__main__":
    main()
