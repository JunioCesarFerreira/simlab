# Distributed execution failure validation

These scripts complement the unit suites with real MongoDB, SSH, SCP, JVM and
Chromium. Use disposable infrastructure: the Python scripts create and remove
`simlab_validation_*` databases; the SSH script also creates a remote
directory with that prefix. Do not point them at workers from experiments in
progress.

## Setting up the isolated services

From the repository root, with Docker available:

```bash
VALIDATION_DIR=$(mktemp -d /tmp/simlab-validation.XXXXXX)
mkdir "$VALIDATION_DIR/keys"
ssh-keygen -q -t ed25519 -N '' -f "$VALIDATION_DIR/keys/id_ed25519"
cp "$VALIDATION_DIR/keys/id_ed25519.pub" "$VALIDATION_DIR/keys/authorized_keys"

docker build -t simlab/remote-host-sshd:dev -f debug/docker/remote-host/Dockerfile.sshd .
docker run -d --name simlab-validate-mongo -p 127.0.0.1::27017 mongo:latest --replSet validation --bind_ip_all
docker exec simlab-validate-mongo mongosh --quiet --eval 'rs.initiate({_id:"validation",members:[{_id:0,host:"localhost:27017"}]})'
docker run -d --name simlab-validate-ssh-a -p 127.0.0.1::22 -v "$VALIDATION_DIR/keys/authorized_keys:/keys/authorized_keys:ro" simlab/remote-host-sshd:dev
docker run -d --name simlab-validate-ssh-b -p 127.0.0.1::22 -v "$VALIDATION_DIR/keys/authorized_keys:/keys/authorized_keys:ro" simlab/remote-host-sshd:dev
docker exec simlab-validate-ssh-b sh -c 'rm /etc/ssh/ssh_host_* && ssh-keygen -A'
docker restart simlab-validate-ssh-b
docker run -d --name simlab-validate-cooja --init --memory 768m -p 127.0.0.1::22 juniocesarferreira/simulation-cooja:v1.1
```

Wait for MongoDB to elect a primary and for the SSH servers to start. Find the
assigned ports with `docker port CONTAINER_NAME`. The SSH containers get
neither the Docker socket nor any volume from the existing services. The key
used is exclusive to this validation; only the public half is mounted on the
servers.

## Resuming a repetition round

With the `mo-engine` dependency environment:

```bash
mo-engine/.venv/bin/python debug/validation/retry_restart.py \
  --mongo-uri 'mongodb://127.0.0.1:MONGO_PORT/?directConnection=true' > retry.json
```

The script kills a process with `SIGKILL` after persisting the round and
after creating the first clone. A new process must complete the same plan
without spending another round, including with the first clone already
`Running`. It also checks 24 concurrent claim requests, initial creation and
replay, as well as refusing the delayed completion of an earlier round. A
host withdrawal is also interrupted after writing `withdrawn_at`; a new
process repeats the removal without leaving orphan workers. Its own database
is removed in the `finally` block.

## SSH loss, upload deadline and pin dispute

With the `master-node` dependency environment:

```bash
master-node/.venv/bin/python debug/validation/ssh_failures.py \
  --mongo-uri 'mongodb://127.0.0.1:MONGO_PORT/?directConnection=true' \
  --cooja-port COOJA_PORT --ssh-a-port PORT_A --ssh-b-port PORT_B \
  --private-key "$VALIDATION_DIR/keys/id_ed25519" > ssh.json
```

- **P01/P02:** the real executor and runner launch a small JVM on the Cooja
  image; the transport is closed after the launch is accepted or before the
  PID is read. The JVM must survive, the concurrent reservation must be
  refused, and a reconciliation must keep the hold until the stop is
  confirmed. The Java application only waits: this case validates the
  process and SSH lifecycle, not a scientific Cooja simulation.
- **P04:** a real 512 KiB SCP transfer is slowed down by the progress
  callback. The TTL is 0.3 s, renewed every 0.04 s, with a concurrent sweep
  every 0.03 s. Both a completed upload and an abort on a 0.15 s budget are
  exercised. There must be no release while the upload is still returning.
- **P08:** two real connections present different keys for the same
  registration, initially with no pin. A barrier synchronizes the learning
  callbacks. Exactly one must be accepted, with both sessions closed at the
  end and only the winning key persisted.

Failures are injected at controlled points in the transport, without relying
on hitting a random millisecond-wide window. The shortened intervals test the
same expiration logic without occupying a worker for tens of minutes.

## Browser, keyboard and accessibility

The script's dependencies live outside the GUI project:

```bash
npm install --prefix "$VALIDATION_DIR/browser" --no-audit --no-fund playwright@1.61.1 axe-core@4.11.0
"$VALIDATION_DIR/browser/node_modules/.bin/playwright" install chromium
```

In another terminal, serve the GUI with the validation API:

```bash
cd gui/simlab
VITE_API_BASE_URL=/api VITE_API_KEY=validation npm run dev -- --host 127.0.0.1 --port 4179 --strictPort
```

From the repository root:

```bash
VALIDATION_NODE_MODULES="$VALIDATION_DIR/browser/node_modules" \
VALIDATION_OUTPUT="$VALIDATION_DIR/browser-results" \
node debug/validation/gui_smoke.mjs
```

The browser intercepts `/api/*` with test data; it never calls the
operational API. The script checks the port-editing payload, a 409 response,
Escape and focus return, Tab navigation, a 390 px viewport, light/dark themes,
and the presentation of a partial, reused and unrecorded evaluation. It
produces screenshots and JSON; it fails on any JavaScript error, lost focus
return, page overflow, or a violation found by axe in the six
infrastructure/form scenarios. The scan covers automatable WCAG 2 A/AA and
2.1 AA rules, not a full accessibility certification nor a screen-reader test.

## Cleanup

The Python scripts remove their own databases and remote files. After saving
the results, stop the Vite server and remove only this fixture's containers:

```bash
docker rm -f -v simlab-validate-mongo simlab-validate-ssh-a simlab-validate-ssh-b simlab-validate-cooja
```

The local directory at `$VALIDATION_DIR` holds the reports and the disposable
key; remove it once you no longer need the results.
