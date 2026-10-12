# Remote host fixture

Exercises the distributed execution path — SSH into a host, `docker ps` there,
tunnel into one of its containers — **on a single machine**, without installing
sshd on it and without a second computer.

```
master_node ──cooja_net──▶ remote-host ──remote_sim_net──▶ rcooja1, rcooja2
                           (sshd + docker CLI)
```

`remote-host` sits on both networks; the `rcooja*` containers sit only on the
private one. The master-node can therefore reach the host but **not** its
containers directly, which is the situation the two-hop tunnel exists for: if
the tunnel were broken, nothing here would run.

For development only — key authentication, no password, root login. The point
is to reproduce the production code path, not its hardening.

---

## 1. Generate the key pair

```bash
cd debug/docker/remote-host
mkdir -p keys
ssh-keygen -t ed25519 -N '' -C simlab-remote-fixture -f keys/simlab_remote
```

The private half is handed to `register_host.py`, which seals it against the
master-node's public key before storing it. The public half is mounted into the
fixture as `authorized_keys`.

> `keys/` is ignored by git — see `.gitignore` in this directory.

## 2. Start the main stack, then the fixture

```bash
# from the repository root
docker compose up -d
docker compose -f debug/docker/remote-host/docker-compose.yaml up -d
```

The order matters: the fixture joins `cooja_net`, which the main compose file
creates.

## 3. Register the host

```bash
python util/register_host.py add remote-sim \
    --address remote-host --port 22 --user root \
    --key debug/docker/remote-host/keys/simlab_remote \
    --label-filter simlab.type=cooja-remote \
    --address-mode container_ip
```

Three arguments deserve a word:

- `--address remote-host` — the container name, resolved by Docker's DNS on
  `cooja_net`. On a real host this would be an IP or a DNS name.
- `--label-filter simlab.type=cooja-remote` — both "hosts" here share one
  Docker daemon, so without a distinct label the local host's own filter
  (`simlab.type=cooja`) would discover these containers as well and count each
  one twice. A real remote host has its own daemon and no such collision.
- `--address-mode container_ip` — the realistic remote mode: the host routes to
  its containers' bridge addresses. `container_name` also works in this fixture,
  because the stand-in host happens to share a Docker network with them.

`add` also asks the master-node to probe the host and discover its containers.

## 4. Watch it come up

```bash
python util/register_host.py list -v
```

```
HOST             TYPE    STATUS         FREE BUSY TOTAL  SCHED
----------------------------------------------------------------
local            local   online            8    0     8  yes
remote-sim       remote  online            2    0     2  yes
    rcooja1        available  container=running  at=172.21.0.3:22
    rcooja2        available  container=running  at=172.21.0.4:22
```

A host stuck at `offline` or `misconfigured` is diagnosed by its
`last_check_message`, shown by `list -v`. The usual causes:

| Message mentions | Cause |
|---|---|
| `unreachable` | the fixture is not on `cooja_net`, or not running |
| `host key` | the fixture was rebuilt and its key changed — clear the pinned fingerprint |
| `credential cannot be opened` | the key was sealed against a key the master-node no longer holds (its keystore was recreated) |
| `docker ps` | the socket is not mounted, or the SSH user cannot use it |
| `cooja.jar not found` | SSH works but the container has no Cooja build |

## 5. Run the distribution test

With the local host's 8 containers and this host's 2, launch an experiment
whose generation holds 6 simulations and watch where they land:

```bash
watch -n 2 'python util/register_host.py list'
```

The expected outcome is **3 and 3** with two hosts, or **2/2/2** once a third
is registered — balanced by count of active simulations, not in proportion to
how many containers each host has. The master-node log states each decision:

```
Placed 65b0… on remote-sim/rcooja1 (attempt 7f3a…) — local[2a/6f] remote-sim[1a/1f]
```

## Tearing it down

```bash
docker compose -f debug/docker/remote-host/docker-compose.yaml down
python util/register_host.py enable remote-sim --off   # drains it first
```

Disabling stops new placements and lets anything still running finish.
Removing the containers while a simulation is on one of them is a good way to
exercise the recovery path on purpose.
