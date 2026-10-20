#!/usr/bin/env python3
"""Register a remote host for distributed simulation, from the command line.

Does exactly what the Infrastructure page will do once it exists, through the
same path: seals the SSH credential against the master-node's published public
key and stores only the ciphertext. The plaintext never reaches MongoDB and
never leaves this process.

Available before the REST API and the GUI so a multi-host setup can be
exercised early.

Usage
-----
    # register a host, reading its private key from a file
    python util/register_host.py add lab-01 \\
        --address 192.168.0.42 --user simlab --key ~/.ssh/simlab_lab01

    # or with a password (less preferred; key authentication is the default)
    python util/register_host.py add lab-01 \\
        --address 192.168.0.42 --user simlab --password-stdin

    # ask the master-node to probe it and discover its containers
    python util/register_host.py check lab-01
    python util/register_host.py discover lab-01

    # see the pool
    python util/register_host.py list

Environment
-----------
    MONGO_URI   default mongodb://localhost:27017/?replicaSet=rs0
    DB_NAME     default simlab

Notes
-----
The container credential is left unset by default, which makes the master-node
fall back to the Cooja image's built-in root login — the same thing it does for
the local containers. Pass ``--container-password-stdin`` only if the image on
that host was changed.
"""
import argparse
import getpass
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pylib.crypto.envelope import public_key_fingerprint  # noqa: E402
from pylib.crypto.sealed import seal_for  # noqa: E402
from pylib.db import create_mongo_repository_factory  # noqa: E402
from pylib.db.models.infra import (  # noqa: E402
    EnumAddressMode,
    EnumAuthMethod,
    EnumHostStatus,
    EnumHostType,
)

DEFAULT_LABEL_FILTER = "simlab.type=cooja"


def factory():
    return create_mongo_repository_factory(
        os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0"),
        os.getenv("DB_NAME", "simlab"),
    )


def published_key(mongo) -> dict:
    """The master-node's public key, or a clear explanation of its absence."""
    doc = mongo.infra_crypto_repo.get_active()
    if not doc or not doc.get("public_key"):
        raise SystemExit(
            "No infrastructure public key published yet.\n"
            "The master-node publishes it on startup. "
            "Start it once, then retry."
        )
    return doc


def read_secret(args) -> tuple[str, str]:
    """Return ``(auth_method, plaintext)`` from exactly one source."""
    sources = [bool(args.key), bool(args.password_stdin)]
    if sum(sources) != 1:
        raise SystemExit("Choose exactly one of --key or --password-stdin.")

    if args.key:
        path = Path(args.key).expanduser()
        if not path.is_file():
            raise SystemExit(f"No such key file: {path}")
        material = path.read_text()
        # Both PEM spellings put it in the header: "BEGIN ENCRYPTED PRIVATE
        # KEY" or "Proc-Type: 4,ENCRYPTED". Caught here so the failure is a
        # sentence instead of a paramiko format error at connect time.
        if "ENCRYPTED" in material[:200].upper():
            raise SystemExit(
                "That key is passphrase-protected. Store an unencrypted key: "
                "the sealed envelope is what protects it at rest."
            )
        return EnumAuthMethod.KEY.value, material

    # Read from a pipe when there is one, so this works in scripts; prompt
    # otherwise, so the password never lands in shell history.
    password = sys.stdin.readline().rstrip("\n") if not sys.stdin.isatty() \
        else getpass.getpass("Host SSH password: ")
    if not password:
        raise SystemExit("Empty password.")
    return EnumAuthMethod.PASSWORD.value, password


def cmd_add(args) -> int:
    mongo = factory()
    crypto = published_key(mongo)
    auth_method, plaintext = read_secret(args)

    envelope = seal_for(crypto["public_key"], plaintext, key_id=crypto["key_id"])
    del plaintext

    container_secret = None
    container_auth = EnumAuthMethod.PASSWORD.value
    if args.container_password_stdin:
        container_plain = sys.stdin.readline().rstrip("\n") if not sys.stdin.isatty() \
            else getpass.getpass("Container SSH password: ")
        container_secret = seal_for(
            crypto["public_key"], container_plain, key_id=crypto["key_id"])
        del container_plain

    host = {
        "name": args.name,
        "type": EnumHostType.REMOTE.value,
        "enabled": not args.disabled,
        "status": EnumHostStatus.OFFLINE.value,
        "access": {
            "transport": "ssh",
            "address": args.address,
            "port": args.port,
            "username": args.user,
            "auth_method": auth_method,
            "secret": envelope,
            "key_fingerprint": None,
            # Learned on first contact and pinned from then on.
            "host_key_fingerprint": None,
        },
        "container_access": {
            "transport": "ssh",
            "username": args.container_user,
            "port": args.container_port,
            "address_mode": args.address_mode,
            "auth_method": container_auth,
            "secret": container_secret,
        },
        "discovery": {
            "mode": "docker",
            "label_filter": args.label_filter,
            "name_prefix": args.name_prefix,
            "endpoint": "ssh-cli",
            "last_discovery": None,
        },
        "origin": "manual",
        "consecutive_failures": 0,
        "last_check": None,
        "last_check_message": "",
        "check_requested_at": None,
        "discovery_requested_at": None,
    }

    existing = mongo.host_repo.get_by_name(args.name)
    if existing is not None:
        if not args.replace:
            raise SystemExit(
                f"Host {args.name!r} already exists. Pass --replace to overwrite "
                "its configuration (its resources and their history are kept)."
            )
        mongo.host_repo.update(existing["_id"], {
            k: v for k, v in host.items() if k != "name"})
        host_id = existing["_id"]
        print(f"Updated host {args.name} ({host_id})")
    else:
        host_id = mongo.host_repo.insert(host)
        print(f"Registered host {args.name} ({host_id})")

    print(f"  sealed against key {crypto['key_id']} "
          f"({public_key_fingerprint(crypto['public_key'])})")
    print(f"  {args.user}@{args.address}:{args.port}, containers via {args.address_mode}")

    if not args.no_check:
        mongo.host_repo.request_check(host_id)
        mongo.host_repo.request_discovery(host_id)
        print("  asked the master-node to probe it and discover its containers")
    return 0


def cmd_check(args) -> int:
    mongo = factory()
    host = _require_host(mongo, args.name)
    mongo.host_repo.request_check(host["_id"])
    print(f"Requested a connectivity check for {args.name}; "
          "the master-node performs it (it holds the key, the API does not).")
    return 0


def cmd_discover(args) -> int:
    mongo = factory()
    host = _require_host(mongo, args.name)
    mongo.host_repo.request_discovery(host["_id"])
    print(f"Requested container discovery for {args.name}.")
    return 0


def cmd_enable(args) -> int:
    mongo = factory()
    host = _require_host(mongo, args.name)
    mongo.host_repo.set_enabled(host["_id"], not args.off)
    print(f"{args.name} is now {'disabled' if args.off else 'enabled'}.")
    if args.off:
        print("  Running simulations are left to finish; no new ones are placed.")
    return 0


def cmd_list(args) -> int:
    mongo = factory()
    snapshot = mongo.host_repo.inventory_snapshot()
    if not snapshot:
        print("No hosts registered.")
        return 0

    if args.json:
        print(json.dumps(snapshot, indent=2, default=str))
        return 0

    header = f"{'HOST':<16} {'TYPE':<7} {'STATUS':<14} {'FREE':>4} {'BUSY':>4} {'TOTAL':>5}  SCHED"
    print(header)
    print("-" * len(header))
    for row in snapshot:
        print(
            f"{str(row.get('name'))[:16]:<16} "
            f"{str(row.get('type'))[:7]:<7} "
            f"{str(row.get('status'))[:14]:<14} "
            f"{row.get('free', 0):>4} "
            f"{row.get('active', 0):>4} "
            f"{row.get('resource_total', 0):>5}  "
            f"{'yes' if row.get('schedulable') else 'no'}"
        )
        if args.verbose:
            for resource in mongo.resource_repo.find_by_host(row["_id"]):
                container = resource.get("container") or {}
                endpoint = resource.get("endpoint") or {}
                print(
                    f"    {resource['name']:<14} {resource.get('status', '?'):<10} "
                    f"container={container.get('state', '?'):<9} "
                    f"at={endpoint.get('address', '?')}:{endpoint.get('port', '?')}"
                )
            message = row.get("last_check_message")
            if message:
                print(f"    last check: {message}")
    return 0


def _require_host(mongo, name: str) -> dict:
    host = mongo.host_repo.get_by_name(name)
    if host is None:
        raise SystemExit(f"No host named {name!r}. Run `list` to see them.")
    return host


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The SSH credential is sealed in this process and stored as "
               "ciphertext; only the master-node can open it.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="register a remote host")
    add.add_argument("name")
    add.add_argument("--address", required=True, help="IP or DNS name of the host")
    add.add_argument("--port", type=int, default=22)
    add.add_argument("--user", required=True, help="SSH user on the host")
    add.add_argument("--key", help="path to an unencrypted private key")
    add.add_argument("--password-stdin", action="store_true",
                     help="read the host password from stdin (or prompt)")
    add.add_argument("--container-user", default="root")
    add.add_argument("--container-port", type=int, default=22)
    add.add_argument("--container-password-stdin", action="store_true",
                     help="only if the Cooja image on that host was changed")
    add.add_argument("--address-mode", default=EnumAddressMode.CONTAINER_IP.value,
                     choices=[m.value for m in EnumAddressMode],
                     help="how to reach the containers from the host")
    add.add_argument("--label-filter", default=DEFAULT_LABEL_FILTER)
    add.add_argument("--name-prefix", default=None)
    add.add_argument("--disabled", action="store_true",
                     help="register without adding it to the pool yet")
    add.add_argument("--replace", action="store_true")
    add.add_argument("--no-check", action="store_true",
                     help="skip the check and discovery requests")
    add.set_defaults(func=cmd_add)

    check = sub.add_parser("check", help="ask for a connectivity check")
    check.add_argument("name")
    check.set_defaults(func=cmd_check)

    discover = sub.add_parser("discover", help="ask for container discovery")
    discover.add_argument("name")
    discover.set_defaults(func=cmd_discover)

    enable = sub.add_parser("enable", help="add a host to the pool, or remove it")
    enable.add_argument("name")
    enable.add_argument("--off", action="store_true")
    enable.set_defaults(func=cmd_enable)

    listing = sub.add_parser("list", help="show hosts and their occupancy")
    listing.add_argument("-v", "--verbose", action="store_true",
                         help="also list each host's containers")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
