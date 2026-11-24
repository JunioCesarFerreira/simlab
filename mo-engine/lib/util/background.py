"""A few threads for work nobody waits on, such as topology pictures.

One thread per individual had no bound. Rendering holds a canvas of several
megabytes and the GIL lets about one draw at a time, so more threads made it
no faster: they only kept all those canvases in memory at once.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger(__name__)

#: One would do for speed; the second keeps a slow upload from stalling the rest.
WORKERS = 2

_tasks: "queue.SimpleQueue[Callable[[], None]]" = queue.SimpleQueue()
_start_lock = threading.Lock()
_threads: list[threading.Thread] = []


def submit(task: Callable[[], None]) -> None:
    """Run ``task`` later, on one of :data:`WORKERS` daemon threads."""
    _tasks.put(task)
    with _start_lock:
        while len(_threads) < WORKERS:
            thread = threading.Thread(
                target=_serve, daemon=True, name=f"background-{len(_threads)}")
            thread.start()
            _threads.append(thread)


def _serve() -> None:
    while True:
        task = _tasks.get()
        try:
            task()
        except Exception:
            logger.exception("A background task failed")
        # Not held while waiting for the next: it references a whole strategy.
        del task
