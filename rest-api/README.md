# SimLab REST API

Read and write access to SimLab's data: experiments, generations,
individuals, simulations, files, and the host/worker inventory the
master-node schedules on.

FastAPI, served under `/api/v1`. The OpenAPI document is at `/docs`.

---

## Running it

```bash
pip install -r requirements.txt
PYTHONPATH=.. uvicorn main:app --reload --port 8000
```

`PYTHONPATH=..` is how `pylib` is imported — it is shared by path rather than
installed. In the Compose deployment the API sits behind the reverse proxy and
is not published directly; see [nginx/README.md](../nginx/README.md).

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/?replicaSet=rs0` | Replica set required, for change streams |
| `DB_NAME` | `simlab` | |
| `SIMLAB_API_KEY` | `api-password` | Required on every request, as `X-API-Key` |

> The key is **not** user authentication. It is embedded in the GUI bundle, so
> anyone who can load the application can read it and call the administrative
> routes. Exposing the installation to users who should not administer it needs
> authentication and authorisation that do not exist yet; HTTPS and credential
> sealing do not substitute for them.

---

## What the routes are grouped into

| Prefix | What it covers |
|---|---|
| `/experiments` | Experiments, their generations and individuals, Pareto fronts, analysis files |
| `/simulations` | One simulation's record, including **where it ran** — host, worker, container, attempt |
| `/generations`, `/individuals` | Reading the optimisation's structure |
| `/files` | GridFS: upload, download, conversion to PNG |
| `/sources` | Source-code repositories and firmware |
| `/problems`, `/synthetic`, `/campaigns` | Problem definitions and synthetic instances |
| `/infra` | Hosts, workers, the inventory overview, and the public key credentials are sealed against |

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
and `/discovery` therefore answer **202** with a timestamp, and the
master-node's health cycle carries them out — within one cycle, not instantly.

### Status codes these routes actually use

| Code | When |
|---|---|
| **202** | A check or discovery was *recorded*. It has not happened yet |
| **409** | The request conflicts with reality: a name already taken, a worker running a simulation, a discovered worker whose name identifies its container |
| **422** | The input was refused before anything was written: a credential in the clear, a port that is not a number, an unknown field, a remote host with no address |

A 409 on a delete is worth reading closely: the condition travels with the
write, so it also means "it was free when you asked and taken by the time we
deleted" — the record is intact.

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
