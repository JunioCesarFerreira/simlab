# Changelog

All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — Infrastructure topology page

A **Topology** page draws the pool as a graph — master-node, hosts, Cooja
workers — coloured by status and refreshed every 3 s, so a run can be watched
moving through it.

### Added

- **`GET /infra/topology`**: the graph and what runs on each worker — its
  simulation, experiment, generation and seed — in a fixed number of queries
  however large the pool is. The per-worker `/resources/{id}/simulation`
  route was one request per container per refresh.
- **Master-node liveness**, which no document records: its health cycle
  writes `last_check` on every enabled host every cycle, even an unreachable
  one, so when no host has been checked for 120 s the cycle is not running.
  The page says so in a banner.
- **The page**: master on the left, one line per host, each host's workers
  beside it. Status by colour (free, running, degraded/held/withdrawing,
  failing, disabled); each running experiment gets a ring colour, stable
  while it runs, so two experiments sharing the pool are told apart. A panel
  lists the running experiments with their workers per host, including those
  still waiting for one.

Three choices in how it is drawn:

- **Fixed positions**, not a force layout: a picture refreshed every few
  seconds whose nodes move each time cannot be followed.
- **Left to right**, because the pool grows by hosts. A row of hosts across the
  top ran out of width at ten, with every box overlapping its neighbours.
- **Hidden cartesian axes** rather than a fitted view, which scaled x and y
  independently and stretched the symbols into ellipses.

Names and messages in tooltips come from operators and remote machines and
are rendered as HTML by ECharts, so they are escaped.

---

## [Unreleased] — Scalability benchmark (USL)

A new **Scalability** page measures how the simulation infrastructure scales,
by running one simulation at increasing concurrency and fitting the
[Universal Scalability Law](https://www.perfdynamics.com/Manifesto/USLscalability.html)
to the time each wave takes.

### Added

- **A benchmark strategy, `usl_benchmark`.** One chromosome and one seed, run
  in waves of `N` identical simulations for each level `N`, one generation per
  wave. Reusing the generation means the master-node places, runs, recovers
  and closes a wave exactly as it does any optimisation, so what is measured
  is the real path, not a model of it. Repetitions alternate ascending and
  descending order, so a drift in the machines lands on every level instead of
  biasing one end. A wave in which nothing produced a result stops the
  benchmark: the workload does not run, and further waves would only measure
  failures.
- **The workload comes from a saved problem.** The chromosome defined in the
  problem editor when there is one, otherwise one feasible network drawn with
  a fixed seed. It is recorded on the experiment (`benchmark_workload`) after
  its firmware is resolved, so a restart resumes with the same network and the
  recorded chromosome is the one that runs.
- **Two throughputs, both fitted.** *Batch* is `N ÷ makespan`, from the first
  placement of a wave to its last result — what a batch actually costs.
  *Per simulation* is `N ÷ mean held time` (Little's law) — insensitive to one
  straggler, so it isolates contention. The gap between them is the cost of
  the slowest container in the wave.
- **The fit** (`pylib/usl.py`) estimates `λ`, contention `σ ∈ [0, 1]` and
  coherency `κ ≥ 0` by a bounded search with a closed-form `λ`, using NumPy
  only. It reports `N* = √((1 − σ)/κ)` and the throughput there when `κ > 0`,
  the ceiling `λ/σ` when there is no peak, and whether the peak lies inside
  the tested range or is extrapolated. With two levels only, `κ` is held at
  zero and the status says so.
- **API:** `POST /scalability` (validated body; answers with the number of
  waves and simulations before anything runs), `GET /scalability`,
  `GET /scalability/{id}` (waves, both fits and the sampled curve, computed on
  demand so a running benchmark already has one).
- **GUI:** the page proposes levels from the workers that can take work
  (powers of two, then the whole pool), warns when a level exceeds the pool,
  and shows the measured waves, the model, linear scaling and the peak, with
  the extrapolated range shaded. The pool counts only hosts the dispatcher
  will place on: the overview's totals include hosts that are disabled or out
  of contact, and proposing their containers would plan waves that queue.
  The y-axis follows the data rather than the linear reference, so a peak
  stays legible instead of being squashed against the floor.
- **A benchmark's own experiment page** points to its analysis instead of
  drawing Pareto and convergence charts for an experiment with no objectives.

### Changed

- The experiment wizard's metric conversion moved to
  `services/defaultDataConversion.ts`, shared with the benchmark: the
  master-node evaluates metrics after every run, and two copies of that list
  would drift.

---

## [Unreleased] — Distributed Cooja resources

Simulations can now run on Cooja containers hosted on other machines, with the
master-node balancing work across hosts. Everything below is behind the same
`docker compose up`: a single-machine deployment behaves as it did, with its
containers seeded as before.

### Added

- **Host and worker inventory.** A *host* is a machine, a *worker* is a Cooja
  container on it. Both are stored, measured, and managed from the new
  **Infrastructure** page. A host is registered with its SSH access; its
  workers are either discovered through Docker or listed by hand, which is
  what makes a shared machine usable where the SSH user cannot reach the
  Docker daemon.
- **An optional speed-aware placement policy**, `SCHED_POLICY=fastest_then_least_loaded`.
  It restricts the candidate hosts to those within 10% of the fastest measured
  free container and then applies the default rule inside that set, so it is
  identical under saturation and on a uniform pool and differs only where a
  batch loses time: slack on a pool whose hosts differ in speed. Each
  successful run records how long its container was *held* — connection,
  upload, run, collection — as an exponentially weighted mean on the worker.
  Measured on a three-host pool, the same genome with the same seed took
  1.22–1.25x longer on the remote hosts. The trade is stated rather than
  hidden: a genuinely slower host is left idle while a faster one has room,
  which is why the default is unchanged.
- **Placement by absolute load.** Each simulation goes to the host carrying
  the fewest active simulations — a count, not a utilisation ratio, so a host
  with eight containers is not kept saturated while one with two sits idle.
  The decision is serialised in a single dispatcher thread; the executions are
  not.
- **Two-hop SSH.** A container on another machine is reached through a
  `direct-tcpip` channel on that host's own SSH session, so no container port
  has to be published on the network.
- **Credentials sealed in the browser.** A host's SSH credential is encrypted
  against the master-node's public key before it leaves the page, stored
  sealed, and opened only by the master-node. The API cannot open one, and a
  test proves it by sabotaging the crypto library in its environment.
- **Availability measured at two levels.** Docker says whether a container
  exists and runs; an SSH probe says whether it can actually run a simulation
  — it answers, it has a Cooja build, and its JVM runs. Both must agree before
  a worker takes work, and the page says *which* level failed.
- **Detached execution.** Cooja is launched in its own session, so a run
  survives a dropped connection and a master-node restart. On start-up, a run
  still going is adopted and followed; one that finished while the process was
  down has its result collected.
- **Where each simulation ran.** Host, worker, container id, attempt and pid
  are recorded at reservation time and published with the simulation, so the
  chain experiment → generation → individual → simulation now reaches the
  machine — and the distribution of work can be audited from the results.
- **Work distribution on the experiment page.** A card after *Firmware* shows
  what share of a run each host and each Cooja container carried, with the
  outcome split out — a container that took work and returned nothing is the
  case a single count cannot show. Served by
  `GET /experiments/{experiment_id}/placement`, derived from the placement
  each simulation recorded when it was reserved, so it still answers after a
  worker is renamed or removed. The shares are of the **placed** simulations
  and anything never placed is reported separately, so a queue does not
  silently shrink every container's percentage.
- **Replica completeness, from the plan to the page.** Each generation records
  the seeds it intends to run, and the completeness of an individual's
  evaluation is measured against *that* — not against the simulation documents
  that happen to exist, which cannot see a replica whose creation failed.
  Counted by seed rather than by document, so a repeated replica neither looks
  like an extra one nor contributes twice. Published on the individual
  (`completeness`) and shown in its panel, with "not recorded" kept distinct
  from "complete".
- **A policy for an incomplete evaluation**, read from the experiment's
  simulation configuration rather than assumed: `replica_policy`
  (`repeat`/`none`), `max_replica_retries` and `incomplete_objective`
  (`worst`/`accept`). A failed replica is run again with its own seed, as a
  new simulation document so the failure stays on the record.
- **The completeness record names its own hole.** `reason` gives one phrase
  for every state, `absent_seeds` lists the planned replicas that were never
  created and `thin_metrics` the metrics a finished replica came back without.
  The distinction is what the policy can act on: a failed replica has a
  document to clone, while an absent seed and a missing metric have none and
  reach the exhaustion branch on the first pass. Without it, an evaluation
  could read `partial` with `missing_seeds: []` and 2 of 2 replicas, and
  nothing in the document explained why. The engine states the same thing in
  its log, once per generation, naming the evaluations it decided without ever
  repeating them and where they went.
- **A worker that may still be running something leaves the pool, and stays
  out until a container is asked.** Disabling it and letting go of the attempt
  is one write, so a failure of the protective step cannot be followed by a
  release that undoes it — and the reason is recorded on the worker
  (`held_for`), so `release`, `force_release`, `reserve` and every status
  write refuse while it is set. Being disabled kept the *dispatcher* off the
  worker but not the reconciliation, which found a terminal simulation,
  concluded nothing was running and freed it with no probe.
- **A pass that resolves those holds**, every health cycle: it asks the
  container and clears the hold only when there is nothing alive and nothing
  left behind. Without it every lost confirmation — most of them transient —
  would need a person.
- **Every lost launch confirmation is treated the same way.** The launch is
  two round trips, and a connection that drops around either of them says
  nothing about the JVM. Only one of the three failure modes was classified
  before; the others released the worker. A stop that is *confirmed* still
  makes it an ordinary failure.
- **A lease held for the whole of the preparation.** A heartbeat renews the
  reservation while the firmware is fetched and uploaded, so the sweep cannot
  expire it underneath the thread doing the work — renewing only at the two
  ends proved ownership at two instants, and the sweep acts on what is true in
  between. A refused renewal is a definite answer that the reservation is
  gone, and nothing further is sent.
- **A run directory is never cleaned over a live process.** The reservation is
  not evidence that the previous run is over, so the container is asked.
- **An uncollected result is preserved under a name the next run will not
  delete**, and where it is — worker, directory, pid — is written on the
  simulation. When the copy cannot be made, because the second session never
  opened, the worker is **held** instead of released: the original is still
  under the name the next run deletes.
- **A repetition round is claimed before any clone exists**, with the limit in
  the condition of the write, and each clone carries its round. The counter
  used to be written after the clones, so a failure of that write let the
  limit be exceeded.
- **Interrupted retries resume their persisted work.** A `retry_plan` is stored
  atomically with the budget claim. Restarting before the first clone or during
  a partial enqueue resumes that same round, even while earlier clones run.
  Atomic upserts and a unique partial index prevent concurrent duplicates;
  existing results are preserved. All strategies replay `Error` generations
  after restart so the replica policy can finish its work.
- **Preparation deadlines interrupt SCP before releasing the worker.** Renewal
  continues until the transfer unwinds. Ownership checks that fail stop remote
  writes, and short configured TTLs now get a shorter heartbeat as well.
- **Infrastructure forms use native modal dialogs**, with keyboard isolation
  and focus restoration. The layout fits narrow screens, and status text and
  buttons use accessible contrast in light and dark themes.
- **Repeatable fault pilots** in `debug/validation`: real MongoDB process kills,
  SSH/JVM confirmation loss, SCP expiry, conflicting SSH pins and a browser
  workflow with accessibility checks.
- **HTTPS through a reverse proxy**, with GUI, API and Grafana behind one
  entry point and the auxiliary ports bound to loopback.

### Changed

- `NUMBER_OF_CONTAINERS` now **seeds** the local host's workers on first start
  instead of sizing a thread pool. The pool comes from the inventory, and
  discovery maintains it.
- Simulation state is written by one owner. The function that executes a
  simulation reports what it produced; the runner decides what that means.
- Deleting a host or a worker carries its condition in the write, so a
  container reserved between the check and the delete is refused rather than
  removed.
- Host and worker updates are partial by field: a change of port no longer
  replaces the credential, the pinned host key or the second hop's settings.
- **Timestamps are aware UTC end to end.** The collections already wrote it;
  the Mongo client now *reads* it that way (`tz_aware=True`), so an API
  response carries an offset instead of an ISO string a browser resolves in
  its own timezone — a measurement one minute old displayed as being in the
  future. A naive value already stored is read as UTC, which is correct for
  the data that exists: every process that wrote one ran with its clock on
  UTC. No data was migrated.
- **An incomplete evaluation no longer reaches the optimiser as a complete
  one.** This changes results. Before, the objective was the mean of whichever
  replicas finished, with nothing recording the difference. The default now
  repeats the missing replica once and, if it is still incomplete, gives the
  individual the worst objective — an under-replicated individual should not
  compete on equal terms with a fully evaluated one. Set
  `incomplete_objective: accept` for the previous numbers, with the
  completeness recorded alongside.
- **A generation whose replica was repeated closes as `Done`.** Its status
  used to count simulation *documents*, and once a failed replica can be
  repeated a terminal document no longer means the generation failed — the
  original stays on the record with its message, by design, and the repeat is
  a second document. So a generation whose every planned seed had a result,
  and whose every individual read `complete`, still closed in error. The rule
  now counts **seeds**, the same unit completeness uses, so the status and the
  record cannot disagree. Two consequences worth knowing: a planned seed with
  no simulation at all now closes the generation in error, where counting
  documents reported `Done` because the one that existed had finished; and a
  generation that never recorded a plan — every one created before
  `expected_seeds` existed — keeps the old rule exactly, because
  reinterpreting a historical generation is a different decision.
- **A published result is never reclassified.** `mark_error` refuses to write
  over a simulation that reached `Done`: closing its generation is separate
  bookkeeping, and a failure of that write used to mark the finished
  simulation in error.
- **A reused evaluation keeps its evidence.** The genome cache stores the
  completeness next to the objectives, so a genome that reappears in a later
  generation — where it has no simulations of its own to count — is not
  reported as "unknown, used". Measured in a 344-simulation pilot: 38 of 210
  individuals had lost a record that said *complete, 2 of 2*. An analytical
  evaluation and an individual rejected by a constraint are now named as such
  rather than sharing the same word.
- **One result per seed, in the counting and in the mean.** The aggregation
  walked every `Done` document, so two results for one seed weighed it twice
  while the completeness record still said two of two. And a seed whose
  earlier attempt failed while its repeat is in flight is *pending*, not
  failed — deriving pending by exclusion lost the repeat entirely.
- **A measurement has to be about the configuration in force.** The probe's
  version is now a condition of the reservation and of the promotion, not just
  a record: a probe that started before an access edit and published after it
  could leave a worker schedulable on evidence about a path that no longer
  existed.
- **A conflicting host key refuses the session**, instead of being logged while
  the connection that accepted it carried on.
- **A resumed run is asked what became of it before any deadline applies.**
  The time the master-node spent down is not time the simulation spent
  executing.
- **The readiness probe takes Java's own exit status.** It was reading the
  status of a `head` at the end of a pipeline, which succeeds after printing
  the error message of a runtime that is not there.
- **Removing a host is a withdrawal first.** Nothing can be created on or
  placed on a host being removed, so a registration that had already read it
  cannot leave a worker behind without one.
- **Two registrations may not point at the same worker.** Address, port and
  directory are compared across every host before one is accepted. A host is
  identified by its SSH **address and port**, as the connection plan reaches
  it: two VMs behind one gateway (`gateway:2197`, `gateway:2198`) are two
  machines, and their containers may share an internal port. Keying the host
  on its address alone refused such an edit with a 409 naming the other VM's
  worker, and blocked any edit of two workers that already matched.
- **Validation is uniform across the infrastructure routes**: unknown fields
  at the root of a body, textual and fractional ports, booleans sent as
  strings and nulls sent where text is expected are all refused with a 422.
- **Editing a worker's port removes a conflicting override** left by an older
  version of the form, which used to win silently over the port the page
  showed.
- **A worker switched back on no longer reads as disabled.** The occupancy
  bar's administrative segment was keyed on the *status* as well as on the
  `enabled` flag, and the status only leaves `disabled` when the next health
  cycle measures the container — so the bar contradicted the toggle beside it
  for up to a cycle. The bucket now follows the flag, which is the operator's
  decision; `unavailable` became the complement rather than a list of
  statuses, so a state nobody enumerated still lands in a segment instead of
  vanishing from the total; and re-enabling a worker or a host moves it off
  `disabled` to `offline` with *"back in the pool; waiting to be measured"*
  instead of leaving a stale "ready". The bar also says so when its counts
  stop adding up to the inventory, rather than drawing a bar past 100%.
- **Editing a host's access takes its workers out of the pool** until they are
  measured again, and the measurement carries the configuration version it was
  taken under — so a probe that started before the edit cannot validate it.
- **A pinned host key can be replaced administratively.**
  `PUT /infra/hosts/{id}` with `access.host_key_fingerprint: null` puts the host
  back to first contact; a string pins one known out of band. The field was
  accepted, published as nullable and then silently dropped, so the request
  answered 200 and changed nothing — which left no route at all for a
  legitimate key change, because the pin is never overwritten by a connection.
  A document or a boolean is refused with a 422, and the rest of the access
  block, including the sealed credential, survives.

### Removed

- `INFRA_MODE=legacy`, the escape hatch kept during the transition. It was a
  second execution path with its own ownership and termination semantics, and
  keeping it meant every fix had to hold in both.

### Known limitations

- **One master-node.** There is no leader election and no atomic claim of a
  simulation itself, so two dispatchers could place the same work.
  Reservations are atomic per container; that is not the same guarantee.
- **The API key is in the GUI bundle.** Anyone who loads the application can
  read it and call the administrative routes. HTTPS and credential sealing do
  not substitute for user authentication.
- **Runtime metrics cover local containers only.** cAdvisor scrapes this
  machine, so a distributed generation under-reports CPU and memory.
- A failed result collection is retried over a new session. If that also
  fails, the artifacts are copied aside and their location recorded on the
  simulation — but **nothing goes back for them automatically**. The manual
  procedure is `util/recover_result.py`, documented in `util/README.md`: it
  locates the bytes, brings them back without touching the database, publishes
  or discards them, and only then lets the worker go. The order is enforced,
  not advisory.
- **No end-to-end *exactly once*,** and the earlier wording here was itself
  too strong. The attempt guard makes a publication by *another* attempt fail;
  it does not forbid the same attempt from publishing twice. What is
  implemented: a publication by an attempt that lost its container is refused
  and its blobs removed; a repeated publish reuses the same ids; a repeated
  conversion removes the previous blobs first, and leaves one unreferenced if
  that removal fails; a crash between the GridFS upload and the state write
  leaves blobs nothing references, and nothing collects them. The aggregation
  reads one result per seed, so a duplicate `Done` document cannot enter a
  mean twice — that is a property of the query, not of the writes.
- **Removing a host is transactional.** A refusal rolls back all worker
  deletions, cancels withdrawal and reports the blocking cause with 409.
  A retained run/result blocks deletion of both worker and host even after
  quarantine clears the reservation. Repeating removal after a process dies
  during withdrawal completes it without leaving orphan workers.
- **Two registrations for one worker on one host** are refused by a unique
  index as well as by the API check. The same container reached through two
  different *host* registrations is still only caught by the check, because
  the host's address is not in that collection.
- **Repeating a replica reopens a terminal generation**, which touches the
  optimisation loop. It has now run with Cooja: a JVM killed mid-run produced
  `Error` with the run-process diagnosis, the gate claimed round 1, cloned the
  same seed, the clone finished, the generation closed `Done` and the
  individual recorded three attempts for two seeds with one result per seed.
  The completion validation also kills a gate process against real MongoDB
  immediately after claiming and after the first clone, then resumes it in a
  new process without wasting or exceeding the round.
- **A stalled local preparation can still hold its worker.** The preparation
  budget interrupts SSH/SCP and keeps renewing until that operation unwinds.
  Python cannot forcibly stop arbitrary local preparation code; a function
  stuck outside SSH needs to return or the master needs to be restarted.
- **Nothing comes back for a preserved result on its own.**
  `unfetched_result` records the host, worker, container, attempt, directory,
  file and pid, and the worker is held until a person decides. Automating the
  collection is deliberately not done: deciding what a half-collected result
  is worth is not a decision to take quietly.
- **Two registrations for one physical worker are compared over configured
  address strings.** One machine registered twice under the *same* address is
  caught across hosts; the same machine under a DNS alias, through a tunnel,
  or behind a published port is not, and cannot be without resolving names at
  registration time — which can differ from the resolution at connection
  time. The boundary is tested rather than described, so it stays a known
  limit.

---

## [Unreleased] — Firmware traceability per experiment

### Added

- **Firmware snapshot at experiment start.** Source repositories are shared and
  mutable, so a finished experiment could reference firmware that is no longer
  the code it ran. The mo-engine now copies every referenced source file into
  GridFS — owned by that experiment alone — and records name, size and `sha256`
  in a `firmware_snapshot` block on the experiment document. The capture is
  idempotent (atomic claim) and best-effort: `partial`/`failed` states are
  persisted and shown instead of aborting the run. Experiments that reference
  no firmware (synthetic runs) are recorded as `skipped`.
- **API.** `GET /experiments/{id}/firmware` (the snapshot),
  `GET /experiments/{id}/firmware/files/{file_id}/content` (one file, scoped to
  the experiment) and `GET /files/experiments/{id}/firmware/zip` (all files as
  `{repository}/{file}` plus a `MANIFEST.json` with ids and digests). The block
  is also part of `GET /experiments/{id}`.
- **GUI.** A *Firmware* card on the experiment page lists each repository and
  its files with size and short digest, opens any file in the existing
  syntax-highlighted viewer, and downloads a single file or the whole snapshot.
- Documentation: [Firmware traceability](docs/markdown/FIRMWARE_TRACEABILITY.md).

### Changed

- `ExperimentRepository.delete` also removes the firmware copies owned by the
  experiment. The shared source repositories remain untouched.
- `MongoGridFSHandler.upload_file`/`upload_bytes` accept optional GridFS
  `metadata`, used to mark each copy with its provenance.
- `SourceFileViewer` now receives a `load` callback instead of a repository id,
  so the same viewer serves source repositories and firmware snapshots.

### Notes

- Snapshots are **not** backfilled: runs that predate the feature show
  "not recorded", since copying a repository's current state into a past run
  would fabricate provenance.
- The master-node still fetches firmware from the shared repository when
  dispatching simulations; the snapshot is the record of the state at start time.

### Tests

- `pylib`: capture contract (copies, digests, provenance metadata, repository
  deduplication), degraded paths (unreadable file, deleted repository),
  `skipped`/`failed` isolation, idempotency, and the delete cascade collecting
  only the copies — never the shared originals.
- `rest-api`: the three endpoints, including a file outside the snapshot being
  rejected, ZIP contents and manifest, name sanitization, and malformed input.
- GUI: presentation helpers and a render test of the *Firmware* card (listing,
  viewer wiring, downloads, partial/skipped/absent states).

---

## [Unreleased] — NSGA metrics: review corrections

### Fixed

- Restore benchmark axis order before analytical GD, so a ZDT1 request with
  permuted objectives has the same distance as the canonical request.
- Resolve missing survivor sets per generation. `population_sources` reports
  the set used for each generation; `population_source="mixed"` identifies a
  heterogeneous series. Charts label the fallback and each compared run.
- Serialize DEAP's global NumPy RNG context across experiments, including
  restoration on exceptions.
- Persist pymoo NSGA-III's hyperplane normalization in a versioned BSON-safe
  `selection_state` checkpoint. Legacy records remain readable with a warning
  when exact continuation cannot be guaranteed; malformed snapshots fail
  explicitly.

### Tests

- Regressions cover interior/on-front and off-front ZDT1 permutations, partial
  survivor histories with and without cumulative HV, empty survivor sets,
  concurrent RNG contexts and exceptions, and exact resume trajectories for
  all six backends across 72 seed/checkpoint/status combinations.
- Frontend tests cover mixed-population labels and legacy API responses.
- Validation and limitations are documented in
  [NSGA regression validation](mo-engine/tests/regression/README.md#validation-of-integration-fixes).

---

## [Unreleased] — NSGA metrics: endpoint robustness and launch defaults

### Fixed

- **`/hv-gd` read objectives by position while building the reference by name.**
  Individuals store their objectives as a positional list in the order the
  experiment declared them, so any request that reordered or subsetted the axes
  compared one objective against another's reference — asking for `[f2, f1]`
  turned a GD of 0 into 11.31. Requested names are now resolved against
  `parameters.objectives[].metric_name`, and an unknown name returns 422.
- **A subset request used the analytical front of a smaller benchmark.** DTLZ2
  with M=3 read on two axes is not DTLZ2 M=2; a subset now falls back to the
  empirical reference. A *permutation* keeps the analytical front, reordered to
  match — ZDT1 is not symmetric in its objectives.
- **All individuals penalised raised a 500.** `max()` over the empty set; it now
  returns the empty response shape.
- **A synthetic run with no stored `pareto_front` returned empty series** even
  though its analytical front needed none. The early return moved into the
  empirical branch, the only one that needs a stored front.
- **The launch wizard's mutation defaults left the search crossover-driven.**
  `prob_mt = 0.1` and `per_gene_prob = 0.05` compose: over 10 variables that is
  one variable mutated every twenty children. The defaults are now the textbook
  convention — every child mutated, each variable with probability `1/n`.
- **A fixed `divisions = 10` fitted no objective count.** It gives 11 reference
  directions in M=2 — a population of 50 competing for a tenth of the niches it
  could use — and 3003 in M=6. It is now derived from M and the population.

### Added

- `suggestedDivisions` and `expectedMutatedVariables` in `gui/src/lib/nsga3.ts`.
  The wizard shows the expected number of mutated variables per child, warning
  below 0.1, and both launch wizards now warn when the reference lattice is far
  *below* the population — the mirror of the existing H > population warning.
- `docs/markdown/SYNTHETIC_MODE.md`: a protocol-conventions table (HV reference,
  measured population, generation 0, evaluation budget, mutation composition,
  divisions) and a note that SimLab's default SCH1 domain is `[-5, 5]` against
  the `[-10, 10]` common in the literature — a different problem, not a harder
  or easier one in any simple sense.

### Notes

- The mutation default was chosen on measurement, not convention. Over 15 seeds
  and 40 generations the textbook rate is neutral on DTLZ2 (M=3) and large on
  ZDT1: HV 0.619 → 0.802 for NSGA-II and 0.633 → 0.806 for NSGA-III, with the
  seed-to-seed spread cut by a factor of three. **This corrects the phase-2
  reading**, which at 5 seeds and 20 generations suggested the opposite; that
  difference was noise.
- The two mutation probabilities are not redundant at equal products.
  Concentrating mutations in few children is a different search from spreading
  them thinly across all of them.
- The regression baseline is re-frozen at `phase-6`, its defaults tracking what
  the wizard sends. `nsga3-sch1` rises from 16.110 to 16.522 and its worst
  per-step HV drop among survivors falls from 0.229 to 0.041 — the case phase 0
  opened as the one where survivors also dropped sharply is now closed. It was
  the divisions.
- The audit reported `rest-api/tests/test_experiment.py` hanging on
  `TestClient`/AnyIO startup. That did not reproduce here; the suite runs.

---

## [Unreleased] — NSGA metrics: exact generational distance

### Fixed

- **GD reported discretisation error as lack of convergence.** Measured against
  a *sampled* reference front, GD cannot go below that sample's fill distance.
  Points lying exactly on the DTLZ2 front — true GD zero — scored 0.0016 at
  M=2, 0.030 at M=3 and 0.194 at M=6. Raising the sample to 200 000 points
  still left 0.051 at M=6, because the front is an (M−1)-dimensional manifold
  and fill distance shrinks only as `N^(-1/(M-1))`.
  GD now uses the **closed-form distance to the true front**
  (`pylib.benchmarks.front_distance`): the same points score 5·10⁻¹⁷ at every M.
  For DTLZ2 the front is the unit sphere, so the nearest point to any
  positive-orthant `f` is `f/‖f‖`; ZDT1 and SCH1 are plane curves solved to
  machine precision.
- **`compute_hv_gd.py` used a different hypervolume reference from everything
  else.** With `--true-front-bench` it still derived the reference point from
  the observed worst value, while `/hv-gd` and `plot_pareto_results.py` used the
  benchmark's fixed `1.1 × nadir` — so the same experiment reported one HV on
  the CLI and another in the GUI, and no two runs were comparable. It now uses
  the fixed nadir like the others.
- **Indicators were normalised by the sampled reference's extremes**, which fall
  short of the real corners and depend on how the sample was drawn. When the
  benchmark has a known range, normalisation now uses the theoretical
  ideal-nadir pair.

### Added

- `pylib.benchmarks.front_distance` and `pylib.benchmarks.ideal`;
  `moo_metrics.gd_analytical` and an explicit `bounds` argument on `gd`, `igd`
  and `igd_plus`. Mirrored into `pareto-analysis/lib/`, with the parity tests
  extended to the new surface — the CLIs must keep running with no `pylib` on
  the path, so the mirror stays a mirror.
- `/hv-gd` returns `gd_method` (`analytical` / `reference_front`), `gd_formula`
  and `normalization`; the chart caption says when GD escaped the reference
  front's discretisation error.
- `docs/markdown/SYNTHETIC_MODE.md` §7 documents the definitions, the measured
  discretisation table, and why IGD cannot take the same route.

### Notes

- **IGD and IGD+ keep the sampled reference and keep the floor.** They average
  over the reference set itself — that is what makes them measure coverage — so
  there is nothing to substitute. Read them as comparative numbers between runs
  sharing a reference, not as absolute distances.
- The regression baseline is re-frozen at `phase-4`. Hypervolume is unchanged;
  only GD moved, and the size of the move is the discretisation error that was
  being misread. It is 25× on `nsga2-sch1`, whose population gets close enough
  to the front that nearly all the reported GD came from the reference, and
  negligible on ZDT1, whose runs stop far enough away that it never mattered.
  The separate `radial_error` column added in phase 0 is gone: phase 4 promoted
  exactly that quantity to GD, so it became a duplicate.
- The plan's original exit criterion for this phase — "GD below 1e-3 for M=2, 3
  and 6, via a denser sample" — was unreachable and has been corrected in place.
  A Das–Dennis grid projected onto the sphere was also tried and rejected: at
  M=6 with 252 points it is *worse* than the random sample (0.272 vs 0.194),
  because the radial projection is not uniform on the sphere.
- `pareto-analysis` tests needed `requests`, missing from the local venv (CI
  already installed it). 76 tests now run where 24 did.

---

## [Unreleased] — NSGA metrics: determinism and resume

### Fixed

- **The DEAP and pymoo NSGA-III backends ignored the experiment seed.** Five
  identical calls produced five different selections. DEAP shuffles through the
  process-wide `numpy.random` module; pymoo builds
  `np.random.default_rng(None)` — OS entropy — whenever no `random_state` is
  passed, which `np.random.seed` cannot reach. Both are now driven from the
  experiment's `algorithm.random_seed`. The DEAP call runs inside a context that
  seeds the global numpy stream and **restores** it afterwards, so the engine
  does not disturb anything else sharing its interpreter.
  *Correction to the audit report: only the two NSGA-III adapters were affected.
  Rank-and-crowding survival has no random step, so the NSGA-II adapters were
  already deterministic; they are seeded anyway, against future library changes.*
- **Resuming a checkpoint restarted from the wrong population.**
  `_restore_population_state` loaded the previous generation's *offspring* as
  parents. Those documents are Q(t-1), not the survivors P(t-1), so every parent
  environmental selection had kept from an older generation was dropped: 4 to 7
  of 12 in the test scenario. Parents now come from that generation's
  `survivors`.
- **A resumed run drew from a fresh random stream.** Each generation document
  now carries an `rng_state` snapshot, taken at enqueue time — before any draw
  belonging to that generation. Because every library seed derives from the same
  generator, that one snapshot is the whole random state of a run.
- **Population order was lost on resume.** Individuals came back sorted by
  chromosome hash while the live population is in generation order, and the
  mating tournament draws by index — so a reordered population is a different
  search. Individuals now record their `index` within the generation, and the
  restore sorts by it.

### Added

- `mo-engine/lib/strategy/library_rng.py` — seed derivation, the numpy global
  seeding context, and BSON-safe snapshot/restore of the engine generator.
- `Generation.rng_state` and `Individual.index`.
- `IndividualRepository.find_by_experiment_and_ids` — survivor hashes cannot be
  resolved against one generation's documents.
- `mo-engine/tests/test_resume_reproducibility.py` — 4 continuous generations
  against 2 + resume + 2 must produce identical populations throughout, driving
  the real `_generation_enqueue` and `_evolution`.

### Notes

- Experiments started before this change still resume: missing `rng_state` or
  `survivors` logs a warning and falls back to the previous behaviour. A
  survivor with no matching individual document aborts the resume rather than
  continuing on a truncated population.
- `Generation.survivors` is now stored verbatim — in selection order, repeats
  included — because a resume rebuilds the population from it. De-duplicating
  would be harmless for the metrics but would shrink the restored population.

---

## [Unreleased] — NSGA metrics: genetic operators

### Fixed

- **Bounded SBX gave both children the same spread factor**, computed from the
  distance to the *lower* bound. The two children are pushed in opposite
  directions, so each needs the distance to the bound it is heading towards.
  Each child now derives its own factor from the same draw, matching DEAP's
  `cxSimulatedBinaryBounded` exactly. This affects every strategy, the `_deap`
  and `_pymoo` variants included — they replace only the survival step.
- **Native NSGA-III niching was not the canonical algorithm.** Four defects,
  all corrected: normalisation now uses ideal point, extreme points and
  hyperplane intercepts over `St = accepted ∪ truncated front` instead of the
  front's own min/range; association is by perpendicular distance to the
  reference *direction* rather than Euclidean distance to the reference
  *point*; niche occupancy is seeded from the already-accepted individuals
  instead of starting at zero; and niches with no remaining candidate leave the
  minimum-occupancy race, which removes a fallback that picked uniformly at
  random from everything left without even updating the occupancy.
  `associate_to_niches` was dead code in this module and is now the association
  the selection uses.
- **The NSGA-II mating tournament broke rank ties at random**, dropping the
  crowded-comparison operator. Ties now go to the larger crowding distance.
  NSGA-III deliberately keeps the random tie-break: Deb & Jain enforce diversity
  through reference-point niching, not through a crowded tournament.

### Notes

- **DTLZ2 results get worse with the corrected operators, and the cause is
  understood.** The old SBX clipped ~0.54% of children onto the *upper* bound
  (0% onto the lower one — the defect was asymmetric). DTLZ2's position
  variables must reach 0 or 1 to produce the front's corner solutions, which
  dominate hypervolume, so the defect manufactured half of those corners for
  free. The regression persists at 60 and 150 generations and under textbook
  mutation, so it is not a short-budget artifact. The fix stands — the old
  distribution is provably wrong — but **DTLZ2 numbers produced before this
  change depend on that artifact and need recomputing.**
- The crowded mating tournament's mean effect is within the 5-seed noise; what
  it measurably does is raise run-to-run variance on DTLZ2 (M=3), consistent
  with crowding distance being a weak diversity signal beyond two objectives.
- Intercepts follow **pymoo**, not DEAP. DEAP solves the hyperplane in the
  translated space and then divides by `intercepts - ideal`, subtracting the
  ideal point a second time; pymoo returns `ideal + 1/x`, which is what Deb &
  Jain's Eq. (4) describes. A test pins the choice.
- The selected population carries ~4% duplicate slots — a child that reproduces
  a surviving parent exactly enters the union twice. Pre-existing in both
  algorithms and untouched here.

### Added

- `mo-engine/tests/test_sbx_bounded.py`, `test_niching_selection.py` and
  `test_tournament_crowding.py` — parity against DEAP and pymoo, plus one test
  per audit defect.
- `compute_crowding_distances` in `lib.genetic_operators.selection`.
- The regression baseline is re-frozen at stage `phase-2`
  (`mo-engine/tests/regression/baseline.json`, renamed from
  `baseline_pre_fix.json`). Its
  [README](mo-engine/tests/regression/README.md) carries the per-operator
  ablation, the bound-clipping measurements and the pre-fix table.

---

## [Unreleased] — NSGA metrics: the measured population

Corrections derived from the [NSGA metrics audit](experiments/nsga-metrics-audit/README.md).

### Fixed

- **Quality indicators measured the wrong set.** HV, GD, IGD and IGD+ were
  computed over the *offspring* `Q_t` — the children evaluated in a generation —
  because those are the individuals a generation document stores. Environmental
  selection can keep an excellent parent that never reappears among the
  children, so the curves showed a regression while the search was in fact
  holding that parent. `mo-engine` now persists the surviving population `P_t`
  on each generation and `GET /experiments/{id}/hv-gd` measures it by default.
- **The last generation never got an environmental selection.** The stop
  condition in `_evolution` was tested *before* the selection, so the reported
  Pareto front was `ND(P_{t-1} ∪ Q_t)` — a union of up to `2·pop_size`
  candidates no selection had run on — instead of `ND(P_final)` of `pop_size`
  individuals. The selection now runs first for both NSGA-II and NSGA-III. The
  evaluation budget is unchanged: neither path enqueues another generation.
- **"Cumulative" only ever applied to HV.** GD and IGD stayed on the offspring
  whichever view was selected, so the two halves of the panel described
  different sets. All indicators now follow one selector.

### Added

- **`GET /experiments/{id}/hv-gd?population=`** — `survivors` (default),
  `offspring` or `archive`. The response reports `population_source`: runs
  recorded before survivor sets existed degrade to `offspring` and say so,
  rather than returning empty series. `hv_cumulative` is unchanged and stays
  independent of the selection.
- **`Generation.survivors`** — the chromosome hashes environmental selection
  kept, written by `GenerationRepository.set_survivors`. Stored on the
  generation rather than as a flag per individual: one write per generation, no
  change to the unique `(generation_id, individual_id)` index, and a survivor
  carried over from an older generation has no document in the current one to
  flag. Writes are best-effort — survivors are analysis metadata and a failed
  write must not abort a running experiment.
- **Web GUI** — the HV-only *Per generation / Cumulative* toggle is replaced, on
  both the experiment detail chart and the comparison page, by a **Measured
  set** selector (*Survivors / Offspring / Archive*) that drives HV, GD and IGD
  together. The caption states which set was measured and warns when a run fell
  back to the offspring.
- **`mo-engine/tests/regression/`** — a frozen pre-correction baseline (30 runs:
  NSGA-II and NSGA-III over DTLZ2, ZDT1 and SCH1, seeds 1/2/3/5/7) with a
  synchronous kernel harness that needs no MongoDB, so each later phase of the
  fix plan produces an inspectable, attributable diff. See its
  [README](mo-engine/tests/regression/README.md).

### Notes

- Reading survivors requires resolving chromosome hashes across the whole
  experiment, not within one generation: a survivor may have been evaluated
  several generations earlier.
- Experiments finished before this change keep their stored `pareto_front`,
  which was built from the unselected union. Re-running is the only way to get
  the `ND(P_final)` front for them.

---

## [Unreleased] — Coverage level α as an explicit problem parameter

### Added

- **Web GUI — problem editor**: new **Coverage Level α (%)** field for Problems
  P1 and P2, next to the communication radii. It edits the problem's
  `min_coverage_percentage` — the minimum share of the sampled mobile-node
  trajectory that must stay connected to the sink — and is validated to the
  closed range `[0, 100]`. The field is hidden for P3/P4, whose adapters have
  no trajectory coverage constraint.
- **Web GUI — launch wizard**: step 1 (*Problem*) summarizes the value as
  `Coverage level α  ≥ N%`, so the level in force is visible before an
  experiment is created; the value is exported in `parameters.problem` for P1/P2
  only.
- **Web GUI — visualization**: the problem topology modal on the experiment
  detail page shows an `α ≥ N%` pill, and the *Problem* parameter table lists
  `min_coverage_percentage` like any other problem field. Experiments launched
  before this change simply have no pill.
- **`pylib/config/problems`**: `parse_min_coverage_percentage()` centralizes the
  parsing for `ProblemP1`/`ProblemP2` and now **rejects** non-numeric values and
  values outside `[0, 100]` at the cast boundary, so a mis-configured experiment
  fails before any simulation is spent instead of silently penalizing the whole
  population. The field stays optional and still defaults to `100.0`.
- **mo-engine**: the P1 and P2 adapters log the α in force when the coverage
  structures are built, so a run's log records the level it was executed with.

### Notes

- α remains a **problem** parameter, not a GA knob: it defines what *feasible*
  means, while `apply_coverage_repair` / `repair_coverage_budget` control how
  hard the operators try to reach it. The serialized key is unchanged
  (`min_coverage_percentage`), so existing saved problems, stored experiments
  and request payloads keep working — α is presentation and validation on top
  of the field the engine has always read.

---

## [Unreleased] — Inverted Generational Distance (IGD / IGD+)

### Added

- **`pylib/moo_metrics`**: new canonical module for the convergence quality
  indicators — GD, IGD and IGD+ (Ishibuchi et al., 2015) — plus reference-front
  sanitation and ideal-nadir normalisation. Single source of truth shared by the
  REST API; `pareto-analysis/lib/metrics.py` mirrors it for the standalone CLIs
  (which must run with no `pylib` on the path) under a new parity test.
- **REST API**: `GET /experiments/{id}/hv-gd` now returns `igd_plus` alongside
  `igd`, plus `reference_size` and `normalized`, and accepts `?normalize=`.
- **Web GUI**: the experiment detail page shows **IGD and IGD+ side by side with
  GD** in a third chart panel, each with its own PNG export. A caption names the
  reference front and warns when it is the run's own final front, in which case
  GD and IGD are self-referential — progress towards this run's own result
  rather than convergence to the true optimum, and not comparable across runs.
- **`pareto-analysis`**: both CLIs report IGD/IGD+; the uploaded figure gained a
  third panel; `compute_hv_gd.py` now reuses `compute_convergence_metrics`
  instead of its own duplicate loop (and so gains the cumulative-HV curve).
  New `--raw-distances` flag on both.

### Fixed

- **IGD was unusable on non-synthetic experiments**: the stored Pareto front was
  used as the GD/IGD reference with no filtering, so a penalised individual
  (the engine writes 1e9+ for infeasible solutions) contaminated it. IGD
  averages *over* the reference, so one such row dragged the indicator to ~1e9.
  GD hid the same contamination because it minimises over the reference
  instead. Penalised, duplicate and dominated rows are now dropped, and a
  reference row missing an objective is skipped rather than defaulted to `0.0`
  (which read as optimal on a minimised axis and pulled the whole front towards
  the origin).
- **`docs/markdown/SYNTHETIC_MODE.md`** claimed GD was the RMS variant
  `sqrt((1/N)·Σ dᵢ²)`; the code has always used the `p = 1` arithmetic mean.
  The doc now matches the code, and a test locks the definition down.

### Changed

- **GD values move**: distances are now normalised by the reference front's
  ideal-nadir range by default, so objectives of different magnitudes weigh
  comparably. Pass `?normalize=false` (API) or `--raw-distances` (CLIs) for the
  previous raw-unit values. Hypervolume is unaffected — it keeps its own
  reference point in raw units.
- `compute_convergence_metrics` returns a named `ConvergenceMetrics` tuple
  (still positionally unpackable) rather than a bare 4-tuple.

---

## [Unreleased] — Runtime (computational) telemetry per experiment

### Added — runtime metrics collection, persistence and visualization

- **`pylib/telemetry`**: new module that, when an experiment finishes
  (Done/Error), queries Prometheus/cAdvisor over the exact
  `[start_time, end_time]` window, normalizes every time series
  (timestamp, metric, value, unit, scope, labels) and preserves them
  integrally as an **immutable GridFS artifact** — Parquet/snappy preferred,
  CSV.gz fallback when pyarrow is unavailable. The experiment document gains
  only a small `runtime_metrics` block: collection status, artifact reference
  (`file_id`, sha256, size, schema version) and a summary (duration, CPU
  average/peak %, memory average/peak bytes) computed over all aggregate
  samples of the window. Collection is idempotent (atomic claim) and new
  metric summaries can be added without breaking changes.
- **mo-engine**: starts a runtime-metrics watcher (Change Stream on finished
  experiments + startup backfill sweep bounded by `TELEMETRY_BACKFILL_HOURS`).
  Tunables: `PROMETHEUS_URL`, `TELEMETRY_ENABLED`, `TELEMETRY_QUERY_STEP`,
  `TELEMETRY_COLLECTION_DELAY_SECONDS`, `TELEMETRY_CONTAINER_FILTER`.
- **REST API**: `GET /experiments/{id}` now embeds the `runtime_metrics`
  summary; new `GET /experiments/{id}/runtime-metrics?max_points=N`
  reconstructs the full series from the GridFS artifact on demand, with
  bucket-average downsampling.
- **GUI**: new **Runtime Metrics** section on the experiment detail page —
  summary tiles (duration, CPU avg/peak, memory avg/peak) shown immediately,
  full CPU/memory time-series charts (aggregate + per-container) loaded only
  when the user clicks *Show charts*.
- **docker-compose**: `moengine` joins `monitoring-net` and receives
  `PROMETHEUS_URL` so it can reach Prometheus.
- Experiment cascade delete now also removes the telemetry artifact from
  GridFS.

---

## [Unreleased] — Phase 1 of the implementation plan

### Added — synthetic benchmark instances (GUI + per-experiment config)

- **Synthetic Instances GUI editor** (`gui/simlab/src/components/synthetic-editor/`,
  route `/synthetic`, sidebar entry): visual definition of DTLZ2 / ZDT1 / SCH1
  benchmark instances (objectives `M`, decision variables `n`, noise σ, region Ω)
  with a live theoretical Pareto-front preview — 2D SVG for `M = 2` and an
  interactive **3D quarter-sphere** (echarts-gl) for DTLZ2 `M = 3`. A dedicated
  4-step launch wizard creates the experiment directly.
- **Per-experiment synthetic config**: `parameters.simulation.synthetic =
  { enabled, bench, noise_std }` (`SyntheticConfig` in `pylib`), which takes
  precedence over the `ENABLE_DATA_SYNTHETIC` / `BENCH` / `NOISE_STD` env vars.
  Also exposed as an optional toggle in the existing Launch Wizard (step 3).
- **REST endpoint** `GET /synthetic/benchmarks` (benchmark catalogue) and
  `is_synthetic` / `synthetic_bench` fields on the experiment list response,
  driving an amber *Synthetic* badge in the experiment card and detail views.
- **mo-engine**: when `synthetic.enabled = true`, all strategies (NSGA-II/III,
  Random Search, Batch) skip CSC/firmware generation and source-repository
  lookups (no Cooja artifacts are produced).

### Fixed — synthetic evaluation correctness

- Objectives are now derived from `parameters.objectives[].metric_name` (in
  order) instead of the empty `data_conversion_config.objectives`, so results
  are stored under the exact keys the mo-engine reads back (previously the
  objectives map was written empty).
- Region normalization now reads `parameters.problem.region` (was reading a
  non-existent top-level `parameters.region`, always falling back to the default
  Ω and ignoring custom regions).
- The fixed **sink** is excluded from the genome, keeping the decision-variable
  count consistent with `n_relays` (genome length = `2 · n_relays`).
- SCH1 now maps its decision variable back to the region's x-axis scale
  consistently with DTLZ2/ZDT1; ZDT1 guards against `sqrt` of a negative value.

### Added — alignment with the SimLab paper (article reproducibility)

- **Random Search strategy** (`mo-engine/lib/strategy/random_search.py`):
  baseline algorithm-agnostic generator referenced in §3.4 / Table 2 of the
  paper, registered in `engine.py` under `parameters.strategy = "random_search"`.
  Reuses the same genome cache and penalty-objectives mechanisms as NSGA-III.
- **Example experiment payload** for the new strategy:
  `debug/requests/post-random-search-experiment-p2.json`.
- **Script-based baseline orchestrator** (`baseline/`): minimal shell + Docker
  Compose pipeline that mirrors what §5.6 / Table 6 of the paper compares
  against. Includes `run_baseline.sh` (SSH/SCP orchestration without state
  machine, persistence, or resume), `docker-compose.baseline.yaml`
  (10 standalone Cooja workers), and `prepare_workload.py` (generates the
  30 × 50-mote workload from §5.2 using the same Cooja template the mo-engine
  uses).
- **External NSGA-III benchmark** (`experiments/external-nsga3-benchmark/`):
  reproduction infrastructure for Table 3 of the paper, comparing the native
  SimLab NSGA-III against DEAP and pymoo implementations on DTLZ2. Scripts
  share a common DTLZ2 definition and HV / GD / IGD / Coverage metrics. DEAP
  and pymoo remain **off-path dependencies** (in `experiments/.../requirements.txt`)
  and are not pulled into the production Docker image.

---

## [v1.0.0] – Initial Release (2025-10-12)

### Overview
This is the **first public and functional release** of the **SimLab** project — a distributed framework for managing and executing large-scale multi-objective simulations using Dockerized environments and a REST API interface.

The current version provides a **fully operational system** capable of running and monitoring experiments end-to-end through container orchestration and MongoDB integration.

### Features
- ✅ **Functional base architecture**
  - Master-node orchestrator for simulation execution (via SSH/SCP)
  - MO-engine (multi-objective optimization loop)
  - REST API for experiment management
  - MongoDB integration for experiment and generation tracking
- 🐳 **Dockerized environment**
  - Ready-to-run Docker Compose configurations for local and distributed setups  
  - Debug environments under `debug/` for simple testing or small experiments
- ⚙️ **Synthetic data mode**
  - Built-in synthetic benchmark evaluation (`DTLZ2`, `ZDT1`, `SCH1`) for validation and algorithm testing without running Cooja simulations
- 📡 **Asynchronous orchestration**
  - Multi-threaded simulation queue, automatic enqueue of waiting experiments
- 📁 **GridFS-based file management**
  - Storage and retrieval of simulation inputs, outputs, logs, and CSV results

### Documentation and Improvements (Planned)
This initial release is functional but not yet fully documented.  
The following enhancements are planned for upcoming versions:

- Complete documentation of setup and deployment workflows
- Additional testing and CI automation  
- Extended examples of experiment submission and monitoring  
- Benchmark dataset publication and performance validation  
- Development of a graphical user interface (GUI) in Vue.js to simplify experiment configuration, execution monitoring, and result visualization  
- English and Portuguese documentation parity  
- More algorithms and analyzers

### Notes
This version establishes the operational baseline of SimLab.  
Subsequent versions will focus on documentation, reproducibility, and academic publication preparation.

---

**Author:**  
Junio Cesar Ferreira<br/>
Institute of Mathematical and Computer Sciences (ICMC), University of São Paulo (USP)
