"""Tests for the transparent zstd layer in the GridFS handler.

The layer exists to keep the Cooja output (~98% of the database) from being
stored verbatim. These tests pin the contract that makes it safe to turn on:
reads return exactly what writes were given, files stored before the layer
existed keep working, payloads that carry their own codec are left alone, and
the whole thing can be switched off without stranding data.
"""
import hashlib
import os

import pytest
import zstandard as zstd
from bson import ObjectId

from pylib.db import gridfs as gfs
from pylib.db.gridfs import MongoGridFSHandler

LOG_LINE = b'[t_us=2000] [Mote:76] {"node":"fd00::202:2:2:2","cpu_energy_mj":37}\n'
SAMPLE = LOG_LINE * 500  # ~34 KB of the repetitive text the layer targets


class _FakeGridOut:
    def __init__(self, data: bytes, metadata):
        self._data = data
        self.metadata = metadata
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk, self._pos = self._data[self._pos:], len(self._data)
            return chunk
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


class _FakeGridFS:
    """Just enough of :class:`gridfs.GridFS` for the handler's call pattern."""

    def __init__(self, db):
        self.store = db.setdefault("files", {})

    def put(self, data, filename=None, metadata=None, _id=None):
        if hasattr(data, "read"):
            data = data.read()
        oid = _id or ObjectId()
        self.store[str(oid)] = {"data": data, "filename": filename, "metadata": metadata}
        return oid

    def get(self, oid):
        entry = self.store[str(oid)]
        return _FakeGridOut(entry["data"], entry["metadata"])


class _FakeConnection:
    def __init__(self):
        self.db = {}

    def connect(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield self.db

        return _cm()


@pytest.fixture
def handler(monkeypatch):
    monkeypatch.setattr(gfs.gridfs, "GridFS", _FakeGridFS)
    monkeypatch.delenv("GRIDFS_COMPRESSION", raising=False)
    monkeypatch.delenv("GRIDFS_COMPRESSION_LEVEL", raising=False)
    return MongoGridFSHandler(_FakeConnection())


def _stored(handler, file_id):
    return handler.connection.db["files"][str(file_id)]


def test_upload_bytes_round_trips(handler):
    file_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    assert handler.read_file_content(str(file_id)) == SAMPLE


def test_payload_is_actually_compressed(handler):
    file_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    assert len(_stored(handler, file_id)["data"]) < len(SAMPLE) / 5


def test_filename_is_preserved_so_existing_queries_keep_working(handler):
    file_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    assert _stored(handler, file_id)["filename"] == "sim_result.log"


def test_compression_is_recorded_in_metadata(handler):
    file_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    block = _stored(handler, file_id)["metadata"]["compression"]
    assert block == {"codec": "zstd", "level": gfs.DEFAULT_LEVEL, "original_size": len(SAMPLE)}


def test_caller_metadata_survives_alongside_the_marker(handler):
    file_id = handler.upload_bytes(SAMPLE, "sim_result.log", metadata={"kind": "firmware"})
    stored = _stored(handler, file_id)["metadata"]
    assert stored["kind"] == "firmware"
    assert "compression" in stored


def test_already_compressed_extensions_are_left_alone(handler):
    payload = os.urandom(50_000)
    file_id = handler.upload_bytes(payload, "topology.png")
    stored = _stored(handler, file_id)
    assert stored["metadata"] is None
    assert stored["data"] == payload
    assert handler.read_file_content(str(file_id)) == payload


def test_small_payloads_are_left_alone(handler):
    file_id = handler.upload_bytes(b"x" * 100, "tiny.log")
    assert _stored(handler, file_id)["metadata"] is None


def test_files_written_before_this_layer_are_returned_verbatim(handler):
    fs = _FakeGridFS(handler.connection.db)
    oid = fs.put(SAMPLE, filename="sim_result.log", metadata=None)
    assert handler.read_file_content(str(oid)) == SAMPLE


def test_compression_can_be_switched_off_without_stranding_data(handler, monkeypatch):
    compressed_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    monkeypatch.setenv("GRIDFS_COMPRESSION", "off")

    plain_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    assert _stored(handler, plain_id)["metadata"] is None
    # Reads keep handling what was written while it was on.
    assert handler.read_file_content(str(compressed_id)) == SAMPLE


def test_upload_file_round_trips_through_both_read_paths(handler, tmp_path):
    src = tmp_path / "sim_result.log"
    src.write_bytes(SAMPLE)
    file_id = handler.upload_file(str(src), "sim_result.log")

    # Streamed frames do not declare their content size in the header, which
    # the one-shot decompressor rejects — both read paths must cope.
    assert handler.read_file_content(str(file_id)) == SAMPLE

    out = tmp_path / "out.log"
    handler.download_file(str(file_id), str(out))
    assert out.read_bytes() == SAMPLE


def test_download_file_passes_uncompressed_files_through(handler, tmp_path):
    payload = os.urandom(50_000)
    file_id = handler.upload_bytes(payload, "topology.png")
    out = tmp_path / "out.png"
    handler.download_file(str(file_id), str(out))
    assert out.read_bytes() == payload


def test_decode_rejects_an_unknown_codec():
    with pytest.raises(ValueError, match="Unsupported"):
        gfs.decode(b"whatever", {"compression": {"codec": "brotli"}})


def test_decode_accepts_a_streamed_frame():
    """The frame shape ``upload_file`` produces: no declared content size."""
    import io

    buf = io.BytesIO()
    with zstd.ZstdCompressor(level=1).stream_writer(buf, closefd=False) as writer:
        writer.write(SAMPLE)
    frame = buf.getvalue()

    assert gfs.decode(frame, {"compression": {"codec": "zstd"}}) == SAMPLE


def test_level_is_configurable(handler, monkeypatch):
    monkeypatch.setenv("GRIDFS_COMPRESSION_LEVEL", "3")
    file_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    assert _stored(handler, file_id)["metadata"]["compression"]["level"] == 3


def test_an_invalid_level_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("GRIDFS_COMPRESSION_LEVEL", "not-a-number")
    assert gfs.compression_level() == gfs.DEFAULT_LEVEL


def test_content_is_byte_identical_not_merely_equal_in_length(handler):
    payload = SAMPLE + os.urandom(10_000)
    file_id = handler.upload_bytes(payload, "sim_result.csv")
    got = handler.read_file_content(str(file_id))
    assert hashlib.sha256(got).hexdigest() == hashlib.sha256(payload).hexdigest()


class _BrokenGridFS(_FakeGridFS):
    """A store whose reads fail, standing in for a truncated chunk or a frame
    that does not decode."""

    def get(self, oid):
        raise RuntimeError("truncated chunk #0")


def test_download_file_raises_instead_of_leaving_a_silent_empty_file(handler, tmp_path, monkeypatch):
    """master-node ships whatever lands on disk straight to Cooja. A download
    that fails quietly runs a simulation against a broken input and records the
    result as valid, so the failure has to reach the caller."""
    monkeypatch.setattr(gfs.gridfs, "GridFS", _BrokenGridFS)
    dest = tmp_path / "simulation.csc"

    with pytest.raises(RuntimeError):
        handler.download_file(str(ObjectId()), str(dest))


def test_a_failed_download_does_not_touch_the_destination(handler, tmp_path, monkeypatch):
    dest = tmp_path / "simulation.csc"
    dest.write_bytes(b"the previous, valid input")
    monkeypatch.setattr(gfs.gridfs, "GridFS", _BrokenGridFS)

    with pytest.raises(RuntimeError):
        handler.download_file(str(ObjectId()), str(dest))

    assert dest.read_bytes() == b"the previous, valid input"
    assert list(tmp_path.iterdir()) == [dest], "a partial file was left behind"


def test_download_writes_through_a_temporary_and_leaves_none_behind(handler, tmp_path):
    file_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    dest = tmp_path / "out.log"
    handler.download_file(str(file_id), str(dest))

    assert dest.read_bytes() == SAMPLE
    assert list(tmp_path.iterdir()) == [dest]


@pytest.mark.parametrize("configured,expected", [
    ("99", gfs.MAX_LEVEL),
    ("0", gfs.MIN_LEVEL),
    ("-5", gfs.MIN_LEVEL),
    ("7", 7),
])
def test_an_out_of_range_level_is_clamped_rather_than_fatal(monkeypatch, configured, expected):
    """zstd rejects a level above 22. Left unclamped, one bad environment
    variable makes every upload in a running campaign raise."""
    monkeypatch.setenv("GRIDFS_COMPRESSION_LEVEL", configured)
    assert gfs.compression_level() == expected


def test_a_clamped_level_still_round_trips(handler, monkeypatch):
    monkeypatch.setenv("GRIDFS_COMPRESSION_LEVEL", "99")
    file_id = handler.upload_bytes(SAMPLE, "sim_result.log")
    assert handler.read_file_content(str(file_id)) == SAMPLE


def test_upload_file_survives_a_size_change_between_stat_and_read(handler, tmp_path, monkeypatch):
    """The frame deliberately does not declare its content size: declaring it
    turns any drift between stat and read into a failed upload."""
    src = tmp_path / "sim_result.log"
    src.write_bytes(SAMPLE)
    monkeypatch.setattr(gfs.os.path, "getsize", lambda _p: len(SAMPLE) + 5000)

    file_id = handler.upload_file(str(src), "sim_result.log")
    assert handler.read_file_content(str(file_id)) == SAMPLE


def test_incompressible_bytes_are_stored_raw(handler):
    """The extension list covers the formats known to carry their own codec.
    For anything else, the frame is kept only when it is actually smaller."""
    payload = os.urandom(40_000)  # no extension hint, and incompressible
    file_id = handler.upload_bytes(payload, "blob.bin")
    stored = _stored(handler, file_id)

    assert stored["data"] == payload
    assert stored["metadata"] is None
    assert handler.read_file_content(str(file_id)) == payload
