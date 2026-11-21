# Provenance and citable releases — how to do it

Goal: every experiment says **which code, which images and which firmware** produced
it, and the code is citable by a **DOI** that resolves to exactly that code. Today
only the firmware is recorded (`firmware_snapshot`, sha256 per file). The rest is
missing:

| Missing | Effect measured on 2026-11-18 |
|---|---|
| Code version per experiment | The paper-1 run `s42` (2026-10-29) and the `s7` runs (2026-11-07) straddle commits `848cc38`/`8b3504c`. It cannot be shown that they ran the same code. |
| Image digests | `mongo:latest`, `grafana/grafana-oss` (no tag), `python:3.12-slim` (moving tag), `simulation-cooja:v1.1` (tag, not digest). A rebuild can change any of them silently. |
| Release | `CITATION.cff` says `v1.1.0`; there were 353 commits since that tag (2026-02-16). No DOI. |

Do the steps in order. Steps 1–4 are code; 5–7 are operations. Nothing here changes a
result; it makes a result traceable.

---

## 1. Bake the commit into every image

`Dockerfile.mo-engine`, `Dockerfile.master-node`, `Dockerfile.rest-api` (and
`Dockerfile.gui` if its version is shown in the UI), right after `FROM`:

```dockerfile
ARG SIMLAB_GIT_SHA=unknown
ARG SIMLAB_GIT_DIRTY=unknown
ARG SIMLAB_VERSION=unknown
ENV SIMLAB_GIT_SHA=${SIMLAB_GIT_SHA} \
    SIMLAB_GIT_DIRTY=${SIMLAB_GIT_DIRTY} \
    SIMLAB_VERSION=${SIMLAB_VERSION}
```

At the end of each Python image, freeze the full dependency set (top-level pins do not
fix transitive packages):

```dockerfile
RUN pip freeze --all > /app/pip-freeze.txt
```

`docker-compose.yaml`, in each `build:` block:

```yaml
    build:
      context: .
      dockerfile: Dockerfile.mo-engine
      args:
        SIMLAB_GIT_SHA: ${SIMLAB_GIT_SHA:-unknown}
        SIMLAB_GIT_DIRTY: ${SIMLAB_GIT_DIRTY:-unknown}
        SIMLAB_VERSION: ${SIMLAB_VERSION:-unknown}
```

Build only through a wrapper, so the values cannot be forgotten (`scripts/build.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail
export SIMLAB_GIT_SHA=$(git rev-parse HEAD)
export SIMLAB_GIT_DIRTY=$([ -z "$(git status --porcelain)" ] && echo false || echo true)
export SIMLAB_VERSION=$(git describe --tags --always --dirty)
docker compose build "$@"
```

## 2. Record it on the experiment

New `pylib/provenance.py`, called from `mo-engine/engine.py` right next to
`capture_firmware_snapshot(mongo, exp_doc)` (line 73), same best-effort contract
(a failure is recorded, never fatal):

```python
def capture_provenance(mongo, exp_doc) -> None:
    block = {
        "schema": 1,
        "captured_at": utc_now(),
        "engine": {
            "git_sha": os.getenv("SIMLAB_GIT_SHA", "unknown"),
            "git_dirty": os.getenv("SIMLAB_GIT_DIRTY", "unknown"),
            "version": os.getenv("SIMLAB_VERSION", "unknown"),
            "python": sys.version.split()[0],
            "pip_freeze_sha256": sha256_of("/app/pip-freeze.txt"),
        },
    }
    mongo.experiment_repo.update(str(exp_doc["_id"]), {"provenance": block})
```

Add `provenance` to `pylib/db/models/experiment.py`, to the REST DTO and mapper
(`rest-api/api/domain/experiment.py`, `rest-api/api/mappers/experiment.py`: the mapper
drops unknown fields), and show it next to the firmware snapshot in the GUI.

## 3. Record the worker side per simulation

The engine's commit does not say which master-node or which Cooja image ran a
simulation, and a distributed pool can mix hosts. In the master-node, where a
simulation is assigned (it already writes `assigned_host_name`):

- `runtime.master_git_sha` from `SIMLAB_GIT_SHA` of the master-node;
- `runtime.worker_image_id`: the **image ID** of the worker container. The Docker
  inspect document has it at the top level as `Image` (`sha256:…`);
  `master-node/lib/dockerctl/normalize.py` currently stores `Config.Image`, which is the
  *name* (`juniocesarferreira/simulation-cooja:v1.1`). Keep the name and add the ID.

## 4. Refuse unclean builds for campaigns

Add an engine setting `SIMLAB_REQUIRE_CLEAN_BUILD` (default `false`). When `true`, an
experiment whose `git_dirty` is not `false` or whose `git_sha` is `unknown` is finished
immediately as `Error` with a message saying why. Set it to `true` on the machine that
runs a thesis campaign, so a forgotten local edit cannot produce a result.

## 5. Pin every image by digest

Read the digest of what is running now, to pin **the same** version (not a newer one):

```bash
docker image inspect mongo:latest --format '{{index .RepoDigests 0}}'
docker image inspect juniocesarferreira/simulation-cooja:v1.1 --format '{{index .RepoDigests 0}}'
docker image inspect grafana/grafana-oss --format '{{index .RepoDigests 0}}'
docker image inspect python:3.12-slim --format '{{index .RepoDigests 0}}'
```

Write them as `name:tag@sha256:…` in `docker-compose.yaml` and the `FROM` lines, e.g.
`image: mongo:<version>@sha256:452bdaf2…` (the local `mongo:latest` today). Do the same
on every remote host's compose file (`debug/docker/remote-host/docker-compose.yaml` and
the real hosts), then confirm all hosts report the same Cooja image ID:

```bash
ssh <host> docker image inspect juniocesarferreira/simulation-cooja:v1.1 --format '{{.Id}}'
```

Also align `numpy` across services that compute the same numbers: `rest-api` pins
`2.5.0`, `mo-engine`/`master-node` pin `2.4.4`.

## 6. Release and DOI

1. Update `CITATION.cff` (`version`, `date-released`) and move the `[Unreleased]`
   entries of `CHANGELOG.md` under the new version.
2. Add `.zenodo.json` (title, creators with ORCID, license `MIT`, keywords, and a
   `related_identifiers` entry pointing to the paper when it exists).
3. Enable the repository in Zenodo (Zenodo → GitHub → toggle the repository) **before**
   creating the release; Zenodo only archives releases made after it is enabled.
4. Tag and release:

   ```bash
   git tag -a v1.2.0 -m "SimLab v1.2.0 — campaign v2 baseline"
   git push origin v1.2.0
   gh release create v1.2.0 --notes-file <notes.md>
   ```

5. Zenodo mints two DOIs: a *concept* DOI (all versions) and a *version* DOI. Papers
   cite the **version** DOI. Add it to `CITATION.cff` (`doi:`) and the README badge in
   the next commit.
6. Build the campaign images from the tag (`git checkout v1.2.0 && scripts/build.sh`),
   so `git_sha` on every experiment equals the tagged commit.

## 7. Verify, then cite

Before analysing a campaign, check that every experiment in its manifest ran the
release (`mongosh` or a short pymongo script):

```javascript
db.experiments.find(
  { _id: { $in: CAMPAIGN_IDS } },
  { name: 1, "provenance.engine.git_sha": 1, "provenance.engine.git_dirty": 1,
    "firmware_snapshot.status": 1 }
)
// every row: git_sha == <tag sha>, git_dirty == "false", firmware captured
db.simulations.distinct("runtime.worker_image_id", { experiment_id: { $in: CAMPAIGN_IDS } })
// exactly one ID
```

In the paper, one sentence carries all of it:

> All experiments were run with SimLab v1.2.0 (DOI 10.5281/zenodo.NNNNNNN, commit
> `abcdef1`), Cooja image `simulation-cooja@sha256:…`, firmware `rpl-udp-csma-v2`
> (sha256 recorded per experiment). Data and analysis scripts: DOI 10.5281/zenodo.MMMMMMM.

Deposit the campaign data (exported experiment documents, the campaign manifest and the
analysis scripts) as a **separate** Zenodo dataset record, linked to the software DOI
with `isSupplementTo`. `util/mongo_backup.py` produces the export.
