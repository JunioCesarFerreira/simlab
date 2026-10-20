"""GridFS access with transparent zstd compression.

Cooja output dominates the database: ``sim_result.log`` and ``sim_result.csv``
together account for ~98% of the GridFS volume, and both are plain text that
compresses ~14x. Compressing the payload before it reaches GridFS keeps that
volume down without touching a single caller — every read and write in the
stack already goes through this handler.

The transformation is deliberately reversible:

* the stored ``filename`` never changes, so existing queries keep working;
* a ``metadata.compression`` block records how the bytes were encoded, and is
  the *only* thing reads dispatch on — a file without it is returned verbatim;
* ``GRIDFS_COMPRESSION=off`` disables compression on write while reads keep
  handling both forms, so the feature can be switched off at any moment.

``util/gridfs_compact.py`` applies the same encoding to files already stored,
and reverts it.
"""

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import gridfs
import zstandard as zstd
from bson import ObjectId

from pylib.db.connection import MongoDBConnection

logger = logging.getLogger(__name__)

CODEC = "zstd"

# Level 12 compresses the Cooja logs ~14x at ~0.3s for a 41MB file. Level 19
# only reaches ~18x and costs 30s, which is not a trade worth making on the
# simulation hot path.
DEFAULT_LEVEL = 12

# zstd rejects anything above 22; negative levels are its "fast" modes.
MAX_LEVEL = getattr(zstd, "MAX_COMPRESSION_LEVEL", 22)
MIN_LEVEL = 1

# Compressing these again only burns CPU: they already carry their own codec.
SKIP_SUFFIXES = (
    ".gz", ".zst", ".zip", ".xz", ".bz2", ".7z", ".rar",
    ".parquet", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf",
    ".mp4", ".webm", ".ico",
)

# Below this the zstd frame header eats most of the gain and the metadata
# block costs more than it saves.
MIN_SIZE = 4096


def compression_enabled() -> bool:
    """Writes compress unless ``GRIDFS_COMPRESSION`` says otherwise."""
    return os.getenv("GRIDFS_COMPRESSION", CODEC).strip().lower() not in ("off", "none", "0", "false")


def compression_level() -> int:
    """Clamped to what zstd accepts. A typo in the environment must not take
    down every upload in a running campaign — it degrades to a valid level and
    says so."""
    raw = os.getenv("GRIDFS_COMPRESSION_LEVEL")
    if raw is None:
        return DEFAULT_LEVEL
    try:
        level = int(raw)
    except ValueError:
        logger.warning("GRIDFS_COMPRESSION_LEVEL=%r is not an integer; using %d",
                       raw, DEFAULT_LEVEL)
        return DEFAULT_LEVEL
    clamped = max(MIN_LEVEL, min(level, MAX_LEVEL))
    if clamped != level:
        logger.warning("GRIDFS_COMPRESSION_LEVEL=%d is out of range [%d, %d]; using %d",
                       level, MIN_LEVEL, MAX_LEVEL, clamped)
    return clamped


def should_compress(name: Optional[str], size: Optional[int]) -> bool:
    """Whether a payload is worth compressing, by name and size alone."""
    if not compression_enabled():
        return False
    if size is not None and size < MIN_SIZE:
        return False
    if name and name.lower().endswith(SKIP_SUFFIXES):
        return False
    return True


def compression_metadata(original_size: int, level: int) -> dict[str, Any]:
    """The block that marks a stored file as compressed."""
    return {"codec": CODEC, "level": level, "original_size": original_size}


def decode(data: bytes, metadata: Optional[dict[str, Any]]) -> bytes:
    """Undo :func:`compression_metadata`-marked encoding; pass anything else through."""
    block = (metadata or {}).get("compression")
    if not block:
        return data
    codec = block.get("codec")
    if codec != CODEC:
        raise ValueError(f"Unsupported GridFS compression codec: {codec!r}")
    # A streamed frame does not declare its content size in the header, so the
    # one-shot ``decompress`` would reject it. The stream reader accepts both
    # that and the sized frames written by ``upload_bytes``.
    with zstd.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
        return reader.read()


class MongoGridFSHandler:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def upload_file(self, path: str, name: str,
                    metadata: Optional[dict[str, Any]] = None) -> ObjectId:
        size = os.path.getsize(path)
        level = compression_level()
        compress = should_compress(name, size)
        with self.connection.connect() as db:
            fs = gridfs.GridFS(db)
            with open(path, "rb") as f:
                if compress:
                    # stream_reader keeps peak memory flat: a 41MB log is never
                    # fully materialised on either side of the compressor.
                    #
                    # The content size is deliberately NOT declared in the frame
                    # header: it would make the upload fail outright if the file
                    # changed size between stat and read, and nothing reads it —
                    # ``decode`` streams, so it never needs to know up front.
                    reader = zstd.ZstdCompressor(level=level).stream_reader(f)
                    file_id = fs.put(reader, filename=name,
                                     **self._extra(metadata, size, level))
                else:
                    file_id = fs.put(f, filename=name, **self._extra(metadata))
        return ObjectId(file_id)

    def download_file(self, file_id: str, local_path: str):
        """Write a GridFS file to disk, decompressing it when it was stored
        compressed.

        The write goes to a sibling temporary file and is renamed into place
        only once it has completed, and failures are raised rather than logged.
        Both matter: master-node feeds these files straight to Cooja, so a
        truncated ``simulation.csc`` that arrives with no error attached runs a
        simulation against a broken input and records the result as valid."""
        target = Path(local_path)
        tmp = target.with_name(f".{target.name}.part")
        try:
            with self.connection.connect() as db:
                fs = gridfs.GridFS(db)
                grid_out = fs.get(ObjectId(file_id))
                with open(tmp, "wb") as f:
                    if (grid_out.metadata or {}).get("compression"):
                        self._stream_decompress(grid_out, f)
                    else:
                        shutil.copyfileobj(grid_out, f, length=1 << 20)
            os.replace(tmp, target)
            logger.info(f"File {local_path} saved successfully.")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            logger.error(f"Failed to save file {file_id}: {e}")
            raise

    def read_file_content(self, file_id: str) -> bytes:
        with self.connection.connect() as db:
            fs = gridfs.GridFS(db)
            grid_out = fs.get(ObjectId(file_id))
            return decode(grid_out.read(), grid_out.metadata)

    def upload_bytes(self, data: bytes, name: str,
                     metadata: Optional[dict[str, Any]] = None) -> ObjectId:
        level = compression_level()
        with self.connection.connect() as db:
            fs = gridfs.GridFS(db)
            payload = None
            if should_compress(name, len(data)):
                candidate = zstd.ZstdCompressor(level=level).compress(data)
                # The extension list catches the formats we know carry their own
                # codec; this catches the ones we do not. Storing the frame when
                # it is larger than the payload would be a pure loss.
                if len(candidate) < len(data):
                    payload = candidate
            if payload is not None:
                file_id = fs.put(payload, filename=name,
                                 **self._extra(metadata, len(data), level))
            else:
                file_id = fs.put(data, filename=name, **self._extra(metadata))
        return ObjectId(file_id)

    @staticmethod
    def _stream_decompress(grid_out, dest) -> None:
        """Inflate a compressed GridFS file straight to disk."""
        reader = zstd.ZstdDecompressor().stream_reader(grid_out)
        while True:
            block = reader.read(1 << 20)
            if not block:
                break
            dest.write(block)

    @staticmethod
    def _extra(metadata: Optional[dict[str, Any]],
               original_size: Optional[int] = None,
               level: Optional[int] = None) -> dict[str, Any]:
        """Optional GridFS ``metadata`` document, omitted when not provided so
        existing callers keep writing exactly the same file records. When the
        payload was compressed, the marker is merged in without disturbing
        whatever the caller supplied."""
        if original_size is not None:
            merged = dict(metadata or {})
            merged["compression"] = compression_metadata(original_size, level or DEFAULT_LEVEL)
            return {"metadata": merged}
        return {"metadata": metadata} if metadata else {}

    def delete_file(self, file_id: str) -> None:
        with self.connection.connect() as db:
            fs = gridfs.GridFS(db)
            fs.delete(ObjectId(file_id))
