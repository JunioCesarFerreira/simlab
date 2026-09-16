"""Bounded, process-local cache of exact indicator results, keyed by input content."""
from collections import OrderedDict
from concurrent.futures import Future
from hashlib import blake2b
import json
from threading import Lock
from typing import Callable


class MetricsCache:
    def __init__(self, max_entries: int = 32, max_bytes: int = 8 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[bytes, tuple[dict, int]] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()
        # Entries exist only for active calculations and are removed on both
        # success and failure; unrelated keys never wait on the same calculation.
        self._pending: dict[bytes, Future] = {}

    def get_or_compute(self, inputs: object, compute: Callable[[], dict]) -> dict:
        key = blake2b(json.dumps(inputs, separators=(",", ":"), sort_keys=True).encode(),
                      digest_size=16).digest()
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key][0]
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = self._pending[key] = Future()
        if not owner:
            return future.result()
        try:
            result = compute()
            size = len(json.dumps(result).encode())
            if size <= self.max_bytes:
                with self._lock:
                    self._entries[key] = (result, size)
                    self._bytes += size
                    while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                        _, (_, old_size) = self._entries.popitem(last=False)
                        self._bytes -= old_size
            future.set_result(result)
            return result
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)


hv_gd_cache = MetricsCache()
