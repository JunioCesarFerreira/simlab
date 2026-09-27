#!/usr/bin/env python3
"""
GridFS compaction — re-encodes stored files with zstd, in place and reversibly.

New uploads are already compressed by ``pylib.db.gridfs.MongoGridFSHandler``;
this backfills the files that predate it.

Every file keeps its ``_id``, so no document that references it is ever
touched — which is what makes the operation reversible with ``--revert``.

Safety, per file:
  1. read the original bytes and hash them;
  2. compress, decompress the result and compare hashes in memory — the
     conversion is abandoned if they differ;
  3. store an untouched backup copy under a temporary id;
  4. delete and rewrite the file under the same ``_id``;
  5. read it back through the handler and compare hashes again;
  6. only then drop the backup.

A crash between 3 and 6 leaves the backup behind; ``--recover`` restores from
those. Interrupted runs can simply be restarted: files already carrying a
``metadata.compression`` block are skipped.

Usage:
    python gridfs_compact.py --dry-run
    python gridfs_compact.py --filename sim_result.log --limit 100
    python gridfs_compact.py                      # everything eligible
    python gridfs_compact.py --revert             # back to plain bytes
    python gridfs_compact.py --recover            # restore orphaned backups
"""

import argparse
import hashlib
import os
import sys
import time
from typing import Any, Iterator, Optional

import gridfs
import zstandard as zstd
from bson import ObjectId
from pymongo import MongoClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pylib.db.gridfs import (  # noqa: E402
    CODEC, DEFAULT_LEVEL, MIN_SIZE, SKIP_SUFFIXES, compression_metadata, decode,
)

BACKUP_PREFIX = "__compact_backup__"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def eligible(doc: dict[str, Any], revert: bool) -> bool:
    """Files worth touching in this direction."""
    compressed = bool((doc.get("metadata") or {}).get("compression"))
    if revert:
        return compressed
    if compressed:
        return False
    name = (doc.get("filename") or "").lower()
    if name.startswith(BACKUP_PREFIX):
        return False
    if name.endswith(SKIP_SUFFIXES):
        return False
    return doc.get("length", 0) >= MIN_SIZE


def candidates(db, filename: Optional[str], revert: bool, limit: Optional[int]) -> Iterator[dict]:
    query: dict[str, Any] = {} if filename is None else {"filename": filename}
    seen = 0
    # no_cursor_timeout keeps the cursor alive across a long conversion, which
    # also means it is not reaped for us: the context manager closes it even
    # when --limit stops the generator early.
    #
    # The scan runs while the collection is rewritten under it. Backups get new
    # ids and may therefore be handed back by the scan, and a converted file
    # could in principle be revisited; eligible() rejects both, so the pass is
    # self-protecting rather than relying on the scan's ordering.
    with db["fs.files"].find(query, no_cursor_timeout=True).sort("_id", 1) as cursor:
        for doc in cursor:
            if not eligible(doc, revert):
                continue
            yield doc
            seen += 1
            if limit and seen >= limit:
                return


def convert_one(fs: gridfs.GridFS, doc: dict, level: int, revert: bool) -> tuple[int, int]:
    """Re-encode one file in place. Returns (bytes_before, bytes_after)."""
    oid = doc["_id"]
    filename = doc.get("filename")
    metadata = dict(doc.get("metadata") or {})
    stored_before = doc.get("length", 0)

    grid_out = fs.get(oid)
    stored_bytes = grid_out.read()
    plain = decode(stored_bytes, metadata)
    digest = sha256(plain)

    if revert:
        payload = plain
        metadata.pop("compression", None)
    else:
        payload = zstd.ZstdCompressor(level=level).compress(plain)
        # Verify before anything is destroyed: if this frame does not decode
        # back to the exact original, the file is left untouched.
        if sha256(zstd.ZstdDecompressor().decompress(payload, max_output_size=0)) != digest:
            raise RuntimeError(f"{oid}: round-trip mismatch, skipped")
        metadata["compression"] = compression_metadata(len(plain), level)

    backup_id = fs.put(stored_bytes, filename=f"{BACKUP_PREFIX}{oid}",
                       metadata={"origin_id": oid,
                                 "origin_filename": filename,
                                 "origin_metadata": doc.get("metadata")})
    try:
        fs.delete(oid)
        extra = {"metadata": metadata} if metadata else {}
        fs.put(payload, _id=oid, filename=filename, **extra)

        written = fs.get(oid)
        if sha256(decode(written.read(), written.metadata)) != digest:
            raise RuntimeError(f"{oid}: read-back mismatch")
    except Exception:
        restore(fs, backup_id)
        raise
    fs.delete(backup_id)
    return stored_before, len(payload)


def restore(fs: gridfs.GridFS, backup_id: ObjectId) -> None:
    """Put an original back under its own id and drop the backup."""
    backup = fs.get(backup_id)
    meta = backup.metadata or {}
    oid = meta["origin_id"]
    data = backup.read()
    try:
        fs.delete(oid)
    except gridfs.NoFile:
        pass
    origin_metadata = meta.get("origin_metadata")
    extra = {"metadata": origin_metadata} if origin_metadata else {}
    fs.put(data, _id=oid, filename=meta.get("origin_filename"), **extra)
    fs.delete(backup_id)


def recover(db, fs: gridfs.GridFS) -> int:
    """Restore every backup left behind by an interrupted run."""
    found = 0
    for doc in db["fs.files"].find({"filename": {"$regex": f"^{BACKUP_PREFIX}"}}):
        restore(fs, doc["_id"])
        found += 1
        print(f"  restored {doc.get('metadata', {}).get('origin_id')}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=os.getenv("MONGO_URI",
                        "mongodb://localhost:27017/?directConnection=true"))
    parser.add_argument("--db", default=os.getenv("DB_NAME", "simlab"))
    parser.add_argument("--filename", help="Only files with this exact filename")
    parser.add_argument("--limit", type=int, help="Stop after N files")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL, help="zstd level")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be converted, change nothing")
    parser.add_argument("--revert", action="store_true",
                        help="Decompress back to plain bytes")
    parser.add_argument("--recover", action="store_true",
                        help="Restore backups left by an interrupted run")
    args = parser.parse_args()

    client = MongoClient(args.uri)
    db = client[args.db]
    fs = gridfs.GridFS(db)

    try:
        if args.recover:
            n = recover(db, fs)
            print(f"Restored {n} file(s) from backup.")
            return 0

        verb = "Reverting" if args.revert else f"Compressing ({CODEC}-{args.level})"
        print(f"{verb} in '{args.db}'"
              + (f", filename={args.filename}" if args.filename else "")
              + (" [DRY RUN]" if args.dry_run else ""))

        before_total = after_total = 0
        done = failed = 0
        t0 = time.time()

        for doc in candidates(db, args.filename, args.revert, args.limit):
            if args.dry_run:
                before_total += doc.get("length", 0)
                done += 1
                continue
            try:
                before, after = convert_one(fs, doc, args.level, args.revert)
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                failed += 1
                print(f"  ! {doc['_id']} ({doc.get('filename')}): {exc}", file=sys.stderr)
                continue
            before_total += before
            after_total += after
            done += 1
            if done % 100 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done} files | {human(before_total)} -> {human(after_total)}"
                      f" | {rate:.1f} files/s")

        elapsed = time.time() - t0
        print(f"\n{'Would convert' if args.dry_run else 'Converted'}: {done} file(s)"
              + (f", {failed} failed" if failed else ""))
        if args.dry_run:
            print(f"Current size: {human(before_total)}")
        else:
            ratio = before_total / after_total if after_total else 0
            print(f"{human(before_total)} -> {human(after_total)}"
                  f"  ({ratio:.1f}x, saved {human(before_total - after_total)})"
                  f" in {elapsed:.0f}s")
        return 1 if failed else 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
