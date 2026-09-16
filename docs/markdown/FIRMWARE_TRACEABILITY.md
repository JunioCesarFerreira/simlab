# Firmware traceability — what each experiment actually ran

Source repositories (collection `sources`) are **shared and mutable**: the same
firmware can be selected by many experiments, and its files can be edited,
replaced or deleted long after a run has finished. Without further measures, a
finished experiment would only hold a *reference* to firmware that may no longer
be the code it executed — the results would not be reproducible.

To close that gap, SimLab copies every firmware file referenced by an
experiment into GridFS **when the experiment starts**, under the sole ownership
of that experiment, and records the copy in a `firmware_snapshot` block on the
experiment document.

## Data flow

```text
Experiment goes Waiting → Running   (mo-engine, update_starting)
Claim the snapshot                  (atomic: only when no block exists)
For each distinct source repository (source_repository_options)
    read every file from GridFS     (shared original, untouched)
    copy it to GridFS               (new file_id, metadata.kind=firmware_snapshot)
    record name, size and sha256
Persist firmware_snapshot           (status captured | partial | skipped | failed)
Strategy starts                     (simulations are queued only after this)
```

The capture is **best-effort**: errors are recorded in the block and surfaced
in the GUI, but never stop the experiment from running. It is also
**idempotent** — the atomic claim means a restarted engine reprocessing a
pending experiment does not duplicate copies in GridFS.

## Experiment document (`firmware_snapshot`)

```json
{
  "firmware_snapshot": {
    "status": "captured",          // captured | partial | skipped | failed
    "captured_at": "...",
    "schema_version": 1,
    "repositories": [
      {
        "option_keys": ["csma"],              // keys in source_repository_options
        "source_repository_id": "...",        // shared repo, may change or vanish
        "name": "rpl-udp-csma",
        "description": "...",
        "files": [
          {
            "file_name": "node.c",
            "file_id": "...",                 // the copy, owned by the experiment
            "origin_file_id": "...",          // file in the shared repository
            "size_bytes": 4213,
            "sha256": "..."
          }
        ],
        "missing_files": []                   // referenced but unreadable at capture
      }
    ]
  }
}
```

### Status values

| Status     | Meaning                                                                 |
|------------|-------------------------------------------------------------------------|
| `captured` | Every referenced file was copied. The record is complete.               |
| `partial`  | At least one file (or a whole repository) could not be copied — see `missing_files`. |
| `skipped`  | The experiment references no firmware (e.g. a synthetic run). `reason` explains it. |
| `failed`   | The capture itself errored; `error` carries the message. No firmware record exists. |
| `capturing`| Transient claim marker, written while the copy is in flight.             |

Several options routinely point at the same repository (one per MAC protocol);
such a repository is copied **once** and every option key that resolved to it is
kept in `option_keys`, so the resolution stays auditable.

## Lifecycle

- **Ownership.** The copies belong to the experiment. `ExperimentRepository.delete`
  removes them together with the experiment's other artifacts. The shared
  originals (`origin_file_id`) are never deleted by that cascade.
- **No backfill.** Experiments that ran before this feature show *"not recorded"*
  and stay that way. Copying the *current* state of a shared repository into a
  past run would fabricate provenance.
- **Scope.** The snapshot is a record, not the execution path: the master-node
  still fetches firmware from the shared repository when it dispatches each
  simulation. Editing a repository *while* an experiment runs is therefore not
  reflected in its snapshot, which captures the state at start time.

## API

| Endpoint | Purpose |
|---|---|
| `GET /experiments/{id}` | The `firmware_snapshot` block is part of the experiment document. |
| `GET /experiments/{id}/firmware` | The snapshot alone. `404` when the run has none. |
| `GET /experiments/{id}/firmware/files/{file_id}/content` | Raw text of one file, scoped to this experiment's snapshot. |
| `GET /files/experiments/{id}/firmware/zip` | Every captured file as `{repository}/{file}`, plus a `MANIFEST.json` with ids and digests. |
| `GET /files/{file_id}/as/{ext}` | Generic download of a single copy. |

## GUI

The experiment page shows a **Firmware** card below *Runtime Metrics*: capture
status, file count and total size, one collapsible section per repository (with
the option keys it served and a link to the original repository), and per file
the size, the short `sha256` and buttons to view the source (syntax-highlighted)
or download it. A *Download ZIP* button fetches the whole snapshot.
