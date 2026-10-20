# GridFS storage — how the Cooja output grew, and what to do about it

> **Status.** The database this analyses was reclaimed on 2026-09-25, after its
> experiment records were archived to
> [simlab-results](https://github.com/JunioCesarFerreira/simlab-results) and
> its simulation output distilled into a ~16.5 GB dataset for deposit. The
> measurements below are kept because they explain the compression now built
> into `pylib/db/gridfs.py`, and because the same growth will recur: the ratios
> are properties of Cooja's output, not of that particular database.

At the time of measurement, `fs.chunks` held 477 GB of *logical* data across
281 031 files. This document records where that volume was, and what can be
done about it.

All numbers below were measured on that database and on samples pulled from
it — none are estimates unless labelled as such.

---

## 1. First correction: 477 GB is not what is on disk

```
fs.chunks   logical (size)        477.0 GB
            on disk (storageSize) 118.6 GB      block_compressor=snappy
```

WiredTiger already compresses the collection with snappy, at **4.0×**. The real
footprint is 118.6 GB on a 850 GB volume (27 % used). So this is not an
emergency — but it is growing linearly with every experiment.

## 2. Where the volume is

| filename | files | logical | avg | share |
|---|---:|---:|---:|---:|
| `sim_result.log` | 53 834 | **389.2 GB** | 7.40 MB | 81.6 % |
| `sim_result.csv` | 53 834 | **77.0 GB** | 1.47 MB | 16.1 % |
| `simulation.xml` | 55 832 | 0.69 GB | 12.7 KB | 0.1 % |
| `positions.dat` | 55 387 | 0.60 GB | 11.1 KB | 0.1 % |
| analysis charts (PNG) | 144 | 0.13 GB | 0.87 MB | — |
| everything else | ~62 000 | ~9 GB | — | ~2 % |

Two files account for **97.7 %** of everything.

## 3. The core finding: the log is 85 % a copy of the CSV

Normalising the lines of a 41 MB `sim_result.log` and grouping by shape:

```
17.4 MB   48 299 lines   {"node":"IPV6","cpu_energy_mj":N,"lpm_energy_mj":N,...}
12.9 MB   35 807 lines   {"node":"IPV6","cpu_energy_mj":N,"lpm_energy_mj":N,...}
 1.6 MB   85 402 lines   Sensor IPv6 = IPV6
 0.5 MB   25 898 lines   Not reachable yet
 0.05MB    1 517 lines   [RPL] t=N node=IPV6 parent=IPV6
```

Those JSON lines are the per-node metric records — and they are **exactly** the
rows of `sim_result.csv`:

- JSON lines in the log: **84 106**. Data rows in the CSV: **84 106**. Identical.
- JSON keys: 21. CSV columns: 21. `set(keys) == set(columns)`, no field on
  either side that the other lacks.

So the largest asset in the database is a verbose re-serialisation — with the
key names repeated on every record — of the second largest asset.

Stripping those lines leaves **15 %** of the log: the boot banner, the RPL
parent changes, the `Sensor IPv6 =` announcements and the reachability probes.
That residue is the part that is genuinely only in the log.

> **One caveat, and it matters.** Each JSON line carries a prefix
> `[t_us=20613264] [Mote:1]` that the CSV does *not* have. To stay lossless, the
> `t_us` and `mote` values must be added as two columns when the records move to
> the columnar file. Delta-encoded, they cost almost nothing.

## 4. Measured compression ratios

### 4.1 `sim_result.log` — 8 random files from different experiments

| id | original | zstd-12 | strip JSON + zstd-12 | ratio |
|---|---:|---:|---:|---:|
| 6a46e196… | 1.87 MB | 0.11 MB | 0.002 MB | 1045× |
| 6a69ca8c… | 11.60 MB | 0.89 MB | 0.154 MB | 75× |
| 6a46fcc3… | 3.32 MB | 0.18 MB | 0.006 MB | 603× |
| 6a46be4b… | 2.58 MB | 0.16 MB | 0.004 MB | 597× |
| 6a8c3827… | 13.84 MB | 0.96 MB | 0.176 MB | 78× |
| 6a8bbeef… | 10.70 MB | 0.75 MB | 0.135 MB | 79× |
| 69f57f33… | 3.23 MB | 0.16 MB | 0.005 MB | 708× |
| 6aaac4b6… | 10.04 MB | 0.77 MB | 0.144 MB | 70× |
| **aggregate** | **57.2 MB** | **3.99 MB (14×)** | **0.63 MB (91×)** | |

Compression alone gives 14×. Removing the redundancy first gives **91×**.

Compressor choice on a single 41 MB log, for reference:

| | size | ratio | time |
|---|---:|---:|---:|
| gzip -6 | 4.18 MB | 9.9× | 0.6 s |
| zstd -3 | 3.68 MB | 11.2× | 0.04 s |
| **zstd -12** | **2.89 MB** | **14.3×** | **0.3 s** |
| xz -6 | 2.34 MB | 17.6× | 6.2 s |
| zstd -19 | 2.22 MB | 18.6× | 31.3 s |

`zstd -12` is the right trade: 80 % of xz's ratio at 20× the speed.

### 4.2 `sim_result.csv` — columnar beats text compression by 2.7×

On a 7.5 MB CSV (84 106 rows × 21 columns):

| encoding | size | ratio |
|---|---:|---:|
| gzip -6 | 1.60 MB | 4.7× |
| parquet + zstd-3 (default) | 1.00 MB | 7.5× |
| xz -6 | 0.89 MB | 8.4× |
| **parquet + DELTA_BINARY_PACKED + BYTE_STREAM_SPLIT + zstd-9** | **0.35 MB** | **21.8×** |
| same, zstd-19 | 0.33 MB | 22.8× |

The jump from 7.5× to 21.8× comes entirely from the encodings, not the codec:
`node` is a repeated IPv6 string (dictionary), the counters and timestamps are
monotonic (delta), and the latency columns are doubles (byte-stream split).
Five of the 21 columns have ≤ 2 distinct values in the whole file.

Round-trip verified lossless: `table.equals(read_back)` → `True`.

## 5. Projection

| asset | now | strategy | after |
|---|---:|---|---:|
| `sim_result.log` | 389.2 GB | drop JSON records (moved to Parquet) + zstd-12 | ~4.3 GB |
| `sim_result.csv` | 77.0 GB | Parquet, delta + byte-stream-split + zstd-9 | ~3.5 GB |
| `simulation.xml` | 0.69 GB | zstd + dedup by content hash | ~0.05 GB |
| `positions.dat` | 0.60 GB | zstd | ~0.05 GB |
| charts, images | 0.13 GB | keep | 0.13 GB |
| **total** | **477 GB** | | **~8 GB** |

Roughly **60×** logically, and on disk 118.6 GB → under 10 GB.

## 6. What to do, in order of effort

### Tier 0 — switch the block compressor — **not available**

The obvious move is to change `fs.chunks` from snappy to zstd. On **MongoDB
8.2 this cannot be done to an existing collection**: `collMod` rejects both
spellings —

```
db.runCommand({collMod: "fs.chunks", wiredTiger: {configString: "block_compressor=zstd"}})
  → BSON field 'collMod.wiredTiger' is an unknown field.
db.runCommand({collMod: "fs.chunks", storageEngine: {wiredTiger: {...}}})
  → BSON field 'collMod.storageEngine' is an unknown field.
```

What remains is `--wiredTigerCollectionBlockCompressor zstd` on mongod, which
applies to **newly created** collections only and needs a restart. Once Tier 1
is in place the payload is already compressed, so the block compressor has
little left to do; this is not worth a restart on its own.

### Tier 1 — compress on write — **implemented**

The payload is compressed before it reaches GridFS and decompressed on read,
inside `MongoGridFSHandler`. Every read and write in the stack already goes
through that class, so no caller changed.

What it does *not* do is rename anything. The stored `filename` stays
`sim_result.log`, so every existing query keeps matching; a
`metadata.compression` block records the codec, the level and the original
size, and reads dispatch on that block alone. A file stored before this
existed has no block and is returned verbatim.

| | |
|---|---|
| `GRIDFS_COMPRESSION` | `off`/`none`/`0`/`false` disables compression **on write**. Reads keep handling both forms, so it can be flipped at any time without stranding data. |
| `GRIDFS_COMPRESSION_LEVEL` | Default 12. Clamped to zstd's `[1, 22]` with a warning — a typo must not take down every upload in a running campaign. |

Two payloads are left alone: anything below 4 KB, where the frame header and
the metadata block cost more than they save, and anything whose extension
already implies a codec (`.gz`, `.zip`, `.parquet`, `.png`, …). For `upload_bytes`
there is a third guard: the frame is kept only when it is actually smaller than
the payload, which covers incompressible content the extension list cannot
recognise.

> The extension list is a heuristic, not a measurement. Matplotlib's PNGs, for
> instance, *do* compress ~23% because they are written at a low zlib level —
> and the per-individual topology images are stored without an extension, so
> they are compressed and save roughly 199 MB across a 54 k-image run. The
> `.png` entry only affects images uploaded with their extension intact.

### Tier 2 — kill the redundancy — **done, as an extraction**

Planned as an in-place migration; carried out instead as a one-off extraction,
because by the time it ran the database was being retired rather than kept. The
result and the pipeline are in
[simlab-results](https://github.com/JunioCesarFerreira/simlab-results); §7 has
the numbers.

The migration path is still the right shape if this ever needs to run against a
live database: convert experiment by experiment, and keep the source until the
rewritten pair has been verified.

### Tier 3 — retention policy (decide, then half a day)

Even at 8 GB the archive grows with every run. Worth deciding: are the full
logs of *dominated* individuals ever re-read? If not, keep the full residue
only for individuals on the final Pareto front and reduce the rest to the
Parquet metrics. That is another order of magnitude, at the cost of
irreversibility — so it needs an explicit decision, not a default.

## 7. What was actually built

Tier 1 lives here: `pylib/db/gridfs.py` compresses payloads with zstd before
they reach GridFS and decompresses on read, transparently to every caller, and
`util/gridfs_compact.py` applies the same encoding to files already stored —
reversibly, preserving each `_id`.

Tier 2 was carried out as a one-off extraction rather than a migration, since
the database was being retired. The pipeline and the result live in
[simlab-results](https://github.com/JunioCesarFerreira/simlab-results):
`tools/build_gridfs_dataset.py` read 390.8 GB of Cooja logs and wrote 16.5 GB
of Parquet plus compressed log residue — **24×**, with 1 095 per-simulation
checks confirming the discarded CSVs are reconstructible from what was kept.

One refinement the data forced, worth carrying forward: §3 says the log's JSON
lines *are* the CSV rows. They are the same **multiset**, but not in the same
order — the CSV is grouped by node, the log is chronological. Extracting the
samples from the log therefore preserves strictly more (the `t_us`/`mote`
prefix and the ordering), and makes the CSV redundant rather than the other way
around.

## 8. One thing compression changed beyond storage

`download_file` used to log its failures and return normally, leaving whatever
had been written — often nothing — on disk. master-node hands those files
straight to Cooja without checking, so a failed read produced an empty
`simulation.csc`, a run that looked successful, and metrics attributed to a
valid individual.

The failure mode predates compression, but decoding a frame is one more thing
that can fail, and streaming the output widened the window in which a partial
file exists. So the write now goes to a sibling `.part` and is renamed into
place only once complete, and failures propagate. Every REST endpoint already
had `except NoFile` / `except Exception` handlers for this that had been dead
code; in master-node, `prepare_simulation_files` now returns `False` on a
failed download, which marks the simulation `Error` instead of running it.

If a campaign starts reporting simulations in `Error` with
`Failed to prepare simulation files`, that is this path firing — look for the
`Failed to download …` line just before it.
