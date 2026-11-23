# Debug

Development and validation aids. None of them is a deployment: for that, use
the [main stack](../README.md).

- [`requests`](requests): preformatted JSON bodies to test the API from its
  Swagger. See [Requests](#requests).
- [`docker`](docker/README.md): Compose files for local debugging
  environments: `minimal-infra`, `mongo-cooja` and the `remote-host` fixture.
- [`validation`](validation/README.md): scripts that validate failures of the
  distributed execution against real services, on disposable infrastructure.

## Usage

### Prerequisites

- Docker and Docker Compose installed on your machine
- The ports and container names of the debug files are free. `minimal-infra`
  and `mongo-cooja` reuse the main stack's names, so **do not run them while
  the main stack exists on the machine**.

### 1. Choose a debug compose setup

- `minimal-infra`: MongoDB, REST API, mo-engine, master-node and three Cooja
  containers. No GUI, proxy or monitoring.
- `mongo-cooja`: MongoDB, eight Cooja containers, cAdvisor and Prometheus, to
  run the Python components on the host.
- `remote-host`: a simulated second host. It is added to the main stack and has
  its own [procedure](docker/remote-host/README.md).

### 2. Bring up the services

From the repository root:

```bash
cd debug/docker/<minimal-infra|mongo-cooja>
docker compose up -d          # add --build to force rebuilding
```

Without `-d`, the logs are shown in real time.

### 3. Verify running containers

```bash
docker compose ps
```

`minimal-infra` runs `mongodb`, `rest_api`, `mo_engine`, `master_node` and
`cooja1`–`cooja3`. Check that their state is **Up**.

### 4. Interact with the system

* The API of `minimal-infra` is on `http://localhost:8000` (Swagger in `/docs`,
  key `simlab-api-key42`).
* With `ENABLE_DATA_SYNTHETIC=True` on the master-node, results are generated
  without launching real simulations
  ([SYNTHETIC_MODE.md](../docs/markdown/SYNTHETIC_MODE.md)).
* Monitor the logs by **service** name:

```bash
docker compose logs masternode
docker compose logs moengine
```

### 5. Tear down / cleanup

```bash
docker compose down
```

`docker compose down -v` also deletes the volumes, and with them the database.

## Requests

Every body needs its ids adjusted (`source_repository_options`). The
data conversions read firmware v1, so use the v1 repositories with them or
replace the conversion
([FIRMWARE_METRICS.md](../docs/markdown/FIRMWARE_METRICS.md)).

| File | State |
|---|---|
| `post-nsga3-experiment-p1.json`, `post-nsga3-experiment-p2.json` | NSGA-III examples for P1 and P2 |
| `post-random-search-experiment-p2.json` | random-search example for P2 |
| `import-problem-p2.json` | for the GUI's problem import, not for the API |
| `post-batch-experiment-p2.json` | [batch](../mo-engine/lib/strategy/README.md#batch) example for P2: 3 chromosomes × 3 seeds |
| `post-nsga3-experiment-p3.json` | NSGA-III example for P3 |
| `post-nsga3-experiment-p4.json` | NSGA-III example for P4: 12 motes in four groups, the base and six stops |

`simulation.duration` is in simulated minutes. `mo-engine/tests/test_request_examples.py` checks that every
`post-*` example builds, encodes, has its objectives in the conversion and its seeds under
`random_seeds`.
