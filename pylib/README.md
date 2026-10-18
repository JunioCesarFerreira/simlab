# pylib

Code shared by every SimLab service. It is imported by path rather than
installed: each service adds the repository root to `sys.path`, so a change
here reaches the master-node, the API, the mo-engine and the utilities without
a release step.

```python
import os, sys

root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from pylib.db import create_mongo_repository_factory
```

---

## What is in here

| Module | Responsibility |
|---|---|
| `db/models/` | The shape of each document, as `TypedDict`. Also `infra.py`'s `operator_configured`, the one rule that decides whose observation about a worker counts — shared because the master-node, the API and the GUI must agree on it |
| `db/repositories/` | One per collection. This is where correctness lives for anything concurrent: a reservation is won by exactly one caller, a release only works for the attempt that still holds the container, and a deletion carries its condition in the write. The same applies to the newer primitives — a lease is renewed only by its holder, a quarantine disables *and* lets go in one write, and a failure is never recorded over a published result |
| `db/gridfs.py` | File storage, with compression |
| `crypto/envelope.py` | The sealed-envelope *format* and its validation. **stdlib only**, deliberately: the API must be able to validate an envelope without being able to open one |
| `crypto/sealed.py` | The keystore that opens them (X25519, libsodium sealed boxes), used only by the master-node |
| `timeutil.py` | Aware-UTC timestamps, and arithmetic that tolerates the naive values written before that convention |
| `db/connection.py` | The Mongo client, built `tz_aware` — which is what makes a stored instant readable as one instead of a wall clock nobody can place |
| `statistics.py`, `moo_metrics.py` | Metric evaluation and multi-objective quality indicators |
| `cooja_files.py`, `cooja_builder/` | Reading Cooja's output and writing its inputs |
| `rpl_dodag.py` | DODAG formation and tree, from a testlog |
| `telemetry/` | Runtime metric collection |
| `firmware_snapshot.py` | Copying the firmware an experiment ran with |

---

## The crypto boundary

`envelope.py` and `sealed.py` are separate on purpose, and the separation is
the design:

- the **API** validates envelopes and never opens them. Its environment has no
  PyNaCl at all, and [a test](../pylib/tests/test_sealed_secrets.py) proves
  the validation path does not depend on it by sabotaging `nacl` in
  `sys.modules`;
- the **master-node** holds the private key, in its own volume, written with
  mode `0600`.

What this buys is that a compromised API cannot read a stored credential. What
it does not buy is protection against a compromised browser or a swapped
public key — the design assumes the integrity of what is served to the page.

---

## Running the tests

`pylib` declares its own test dependencies, because what its suite needs and
what any one service installs are different sets — the API image deliberately
has no PyNaCl, and the suite needs it:

```bash
pip install -r pylib/requirements-test.txt
cd pylib && pytest -q
```

The same file is what CI installs. A count of passing tests is only comparable
against the same command and environment: measured in a clean Python 3.12 with
that requirements file, the suite was 401 tests at `e0444bf`. The current
working tree measures **450 passing and 1 skipped** under the combined local
venvs and Python 3.13 — a different command and a different environment, so
the two numbers are not comparable, and only the first says anything about CI.

---

## Two invariants this library enforces rather than documents

Both were written as prose first and did not hold:

* **A published result is final.** `SimulationRepository.mark_error` refuses
  to write over a simulation in `Done`. Closing its generation is separate
  bookkeeping over a different document, and a failure of *that* write used to
  propagate back and mark the finished simulation in error — an experiment's
  outcome changed retroactively by a failure that had nothing to do with the
  run. The guard belongs here, and not only at the call site, because the call
  site is what got it wrong.
* **A worker holding something unresolved is released by nobody.** A
  quarantine is a decision, and a decision that lives only in a process does
  not survive it: being disabled kept the dispatcher off a worker and not the
  reconciliation, which freed it on the strength of a terminal simulation.
  With `held_for` set, `release`, `force_release`, `reserve` and every status
  write refuse, and `clear_held_for` is the only way out — for a caller that
  has asked the container.
* **One result per seed.** Completeness counts seeds and the aggregation
  walked documents, so two results for one seed weighed it twice in the mean
  while the record still said two of two. The later result wins; a seed whose
  repeat is in flight counts as pending rather than failed.
* **A repetition budget and its work are claimed atomically.** The generation's
  `retry_plan` records source simulations/reasons together with the increment
  of `replica_retries`. A restart resumes that manifest before checking active
  simulations or the remaining budget. Completing the enqueue removes the plan
  and reopens the generation in one write guarded by the round number.
  Clones use `$setOnInsert` and a unique partial index on generation, individual,
  seed and round: replay never resets an existing result. Historical simulations
  without `retry_round` are outside that index. If an earlier deployment wrote
  duplicate retry identities, index creation fails visibly; inspect and resolve
  those records before upgrading, preserving their results and audit history.
* **A worker's occupancy is measured, not assumed.** `record_duration` folds
  each completed run into an exponentially weighted mean on the worker, and
  `inventory_snapshot` publishes the fastest free container per host. Only a
  speed-aware scheduling policy reads it, so it is advisory: a stale or
  missing value costs a slightly worse placement and nothing else. A missing
  value is *unknown*, never "infinitely slow" — a pool that has just been
  registered has no measurements at all.
* **Completeness is measured against the plan.**
  `GenerationRepository.replica_completeness` compares the generation's
  recorded `expected_seeds` with the seeds that produced a result. Derived from
  the documents that exist, it cannot see a replica whose *creation* failed —
  a generation configured for two seeds that enqueued one looked complete, with
  one replica. Counted by seed rather than by document, so a repeat shows up as
  an extra attempt instead of an extra replica.
* **The record names its own hole.** `failed_seeds` can be repeated (there is a
  document to clone); `absent_seeds` and `thin_metrics` cannot, and `reason`
  says which of the three a state is. Without them an evaluation could read
  `partial` with `missing_seeds: []` and 2 of 2 replicas, because every planned
  seed finished and one came back without a metric, and nothing in the document
  explained it.
* **A generation's status uses the same unit as its completeness.**
  `close_if_finished` counts seeds through `all_planned_replicas_done`, so a
  generation whose failed replica was repeated closes `Done`: the failed
  document stays on the record by design, and counting documents made a
  repaired generation indistinguishable from a broken one. A generation with no
  recorded plan — every one created before `expected_seeds` existed — keeps the
  document rule, because reinterpreting a historical generation is a separate
  decision.
