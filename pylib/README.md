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
| `db/repositories/` | One per collection. This is where correctness lives for anything concurrent: a reservation is won by exactly one caller, a release only works for the attempt that still holds the container, and a deletion carries its condition in the write |
| `db/gridfs.py` | File storage, with compression |
| `crypto/envelope.py` | The sealed-envelope *format* and its validation. **stdlib only**, deliberately: the API must be able to validate an envelope without being able to open one |
| `crypto/sealed.py` | The keystore that opens them (X25519, libsodium sealed boxes), used only by the master-node |
| `timeutil.py` | Aware-UTC timestamps, and arithmetic that tolerates the naive values written before that convention |
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
that requirements file, the suite is 401 tests.
