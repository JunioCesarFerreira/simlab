# SimLab REST API

Read and write access to SimLab's data: experiments, generations,
individuals, simulations, files, and the host/worker inventory the
master-node schedules on.

FastAPI, served under `/api/v1`. Swagger UI is at `/docs`, ReDoc at `/redoc` and
the OpenAPI document at `/openapi.json`. Collection routes end in a slash
(`/experiments/`): without it the API answers 307.

---

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=.. MONGO_URI='mongodb://localhost:27017/?directConnection=true' \
    uvicorn main:app --reload --port 8000
```

`directConnection` is needed against the compose MongoDB, whose replica set
advertises a name that only resolves inside Docker.

`PYTHONPATH=..` is how `pylib` is imported — it is shared by path rather than
installed. In the Compose deployment the API sits behind the reverse proxy and
is not published directly; see [nginx/README.md](../nginx/README.md).

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/?replicaSet=rs0` | The deployment is a replica set: the API uses transactions and the other services change streams |
| `DB_NAME` | `simlab` | |
| `SIMLAB_API_KEY` | `api-password` (compose: `simlab-api-key42`) | Required on every `/api/v1` request, as `X-API-Key`. `/health` and the documentation pages do not ask for it |
| `PROMETHEUS_URL` | `http://localhost:9090`, or `http://prometheus:9090` with `IS_DOCKER` | Used by the manual runtime-metrics collection |
| `SIMLAB_PARETO_SCRIPT` | `../pareto-analysis/plot_pareto_results.py` | The script behind `plot-pareto` |
| `SIMLAB_PARETO_PYTHON` | the API's interpreter | Runs that script |
| `UVICORN_RELOAD` | unset | `1` enables auto-reload under `python main.py` |
| `GRIDFS_COMPRESSION` / `GRIDFS_COMPRESSION_LEVEL` | `zstd` / `12` | Compression of uploads |

> The key is **not** user authentication. It is embedded in the GUI bundle, so
> anyone who can load the application can read it and call the administrative
> routes. Exposing the installation to users who should not administer it needs
> authentication and authorisation that do not exist yet; HTTPS and credential
> sealing do not substitute for them.

---

## What the routes are grouped into

| Prefix | What it covers |
|---|---|
| `/experiments` | Experiments, with their generations and individuals (`/{id}/full`), Pareto fronts, quality indicators (`/{id}/hv-gd`), Pareto plots, analysis files, firmware snapshots, runtime metrics and the distribution of work over hosts/workers |
| `/simulations` | One simulation's record, including **where it ran** — host, worker, container, attempt — and the files it references |
| `/generations` | Generations, each with its individuals. Individuals have no routes of their own |
| `/files` | GridFS downloads: one file, and ZIPs of an experiment's analysis, topologies or firmware. Uploads go through the routes that own each file |
| `/sources` | Source-code repositories and firmware |
| `/problems` | Saved problems and their background images |
| `/campaigns` | Named groups of experiments |
| `/synthetic` | The catalogue of synthetic benchmarks |
| `/scalability` | Creating, listing and analysing USL benchmarks |
| `/infra` | Hosts, workers, machine limits, the topology, the inventory overview, and the public key credentials are sealed against |

The complete list, with schemas, is the Swagger UI.

### What `POST /experiments/` keeps

`parameters` is a typed model with `strategy`, `algorithm`, `simulation`,
`problem` and `objectives`, all required. **Any other key in it is dropped
without an error.** The contents of `algorithm`, `simulation` and `problem` are
stored as sent. `simulation.duration` is in simulated minutes. Seed aggregators
are validated (422); the rest is validated by the engine when the experiment
starts.

---

## Two conventions worth knowing before using `/infra`

**Credentials arrive sealed, never in the clear.** A host's SSH secret is
encrypted in the browser against the master-node's public key
(`GET /infra/crypto/public-key`) and sent as
`{"alg": "crypto_box_seal", "key_id": "...", "ciphertext": "..."}`. A payload
that looks like plaintext is refused with **422** — including a short string
smuggled as base64, which the envelope validation catches because a sealed box
cannot be shorter than its own overhead. The API **cannot open** one: it has no
private key and no crypto library.

**Work the API cannot do itself is recorded as a request.** Checking a host or
discovering its containers needs a route to the container network and the key
that opens a credential, and the API has neither. `POST /infra/hosts/{id}/check`
and `/discover` therefore answer **202** with a timestamp, and the
master-node's health cycle carries them out — within one cycle, not instantly.

### Status codes these routes actually use

| Code | When |
|---|---|
| **202** | A check or discovery was *recorded*. It has not happened yet |
| **409** | The request conflicts with reality: a name already taken, a worker running a simulation, a discovered worker whose name identifies its container, a host being removed, a second registration pointing at a worker another one already names |
| **422** | The input was refused before anything was written: a credential in the clear, a port that is not a number or not whole, a boolean sent as something else, text sent as `null`, an unknown field at any level, a remote host with no address |

A 409 on a delete is worth reading closely: the condition travels with the
write, so it also means "it was free when you asked and taken by the time we
deleted" — the record is intact.

**Removing a host is transactional.** A refusal preserves every worker,
including free workers examined before the one that blocked the operation.

| Response | What happened |
|---|---|
| **204** | The host and every worker were removed together |
| **409** | Active work, a retained run/result, or a concurrent registration prevented removal. Nothing was removed; resolve the reported cause and retry |
| **404** | The host no longer exists |

The operation withdraws the host first, preventing new registrations and
placements. Worker deletions and the final host deletion share one transaction;
each delete also refuses `held_for`, even after quarantine cleared the active
reservation. A conflict rolls back all deletions and cancels withdrawal. If the
API process dies after marking withdrawal, repeating DELETE resumes it safely;
to keep that host instead, an administrator can clear `withdrawn_at` after
confirming no removal is still in progress. No `200/stranded` partial-success
response exists in this implementation.

**Two registrations may not point at one worker.** Address, port and directory
are compared before a registration is accepted, and on a single host a unique
index refuses a duplicate even if two requests interleave. The same container
reached through two different *host* registrations is only caught by the
check, because a host's address is not in the workers' collection.

---

## Updating a host or a worker

Updates are **partial by field**. Three intentions, and they are
distinguishable:

| You send | It means |
|---|---|
| the field, with a value | override it |
| nothing | leave what is stored |
| the field, as `null` | remove the override and inherit again |

The third one matters: without it there is no way to undo an override, and
changing a port used to replace the whole access block — taking the
credential and the pinned host key with it.

`null` is supported where inheriting is meaningful, which is not everywhere:

| Block | Fields that accept `null` |
|---|---|
| worker `connection` | `transport`, `username`, `port`, `auth_method`, `secret` |
| host `container_access` | `secret` (and `remote_dir` as an empty string) |
| host `access` | `secret`, `host_key_fingerprint`, `key_fingerprint` — a host with no address or user is not a host |

Two other behaviours of an update are worth knowing:

* **A worker's port.** Sending `endpoint.port` without a `connection.port`
  states the *effective* port, and a conflicting override left by an older
  version of the form is removed. That override wins in the connection plan,
  so without this an edit changed the endpoint, reported the new port, and
  went on connecting to the old one. A client that means to keep an override
  sends it in the same request.
* **Replacing a pinned host key.** `access.host_key_fingerprint: null` puts
  the host back to first contact, and a string pins one known out of band.
  This is the *only* route for a legitimate key change — a rebuilt machine, a
  rotated key — because the pin is written on first contact and never
  overwritten: a connection presenting a different key has its session refused
  instead. The field was accepted and then silently dropped, so the request
  answered 200 and changed nothing. A document or a boolean is refused.
* **Editing a host's access** clears what was measured about its workers,
  bumps the host's configuration version and takes the idle ones out of the
  pool until they are measured again. Workers holding a simulation are left
  alone: the run in progress reached its container by a path that worked — and
  when it is released it lands `offline`, because its measurement is about the
  previous configuration. The version is a **condition** of the reservation
  and of the promotion, not just a record, so a probe that started before the
  edit cannot leave a worker schedulable after it.

A malformed body is refused rather than coerced. The shape of each block is
checked before its fields: `{"endpoint": 123}`, `{"access": 42}` and
`{"endpoint": {"address": {"a": 1}}}` all answer 422, where the first two
raised `TypeError` behind a 500 and the third was stored as the text
`"{'a': 1}"`.

---

## Tests

```bash
pip install -r requirements.txt -r requirements-test.txt
PYTHONPATH=.. pytest -q
```

The suite mocks the repository layer, so it needs no MongoDB. What it does not
cover is the contract against a real database; the round-trip tests in
[`tests/test_infra_update_roundtrip.py`](tests/test_infra_update_roundtrip.py)
are the exception — they drive endpoint, mapper and repository over a store
that applies `$set`, because asserting on a 200 is how a defect that erased
credentials passed review once.

---

## Known issues

- **`GET /files/simulations/{id}/topology` answers 500.** It calls
  `simulation_repo.get_topology_pic_file_id`, which does not exist; the test
  mocks it. The picture is on the individual (`topology_picture_id`): use
  `GET /files/{file_id}/as/png`.
- **A `batch` experiment cannot be created.** `parameters.chromosomes` is one of
  the keys that are dropped (above). See
  [the strategy](../mo-engine/lib/strategy/README.md#batch).
- **Writes are not validated on every route.** `PUT /experiments/{id}` and
  `PUT /simulations/{id}` store any field, the firmware snapshot included, and
  `PATCH` of a generation's or simulation's status accepts any string. Only the
  experiment's status is checked.
