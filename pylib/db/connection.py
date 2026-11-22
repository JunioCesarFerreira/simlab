import logging
import time
import pymongo
from typing import Generator, Callable
from pymongo import MongoClient
from contextlib import contextmanager
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


#: How long an idle change stream that can be stopped waits on the server
#: before it asks again whether it was. An event still returns at once.
STOP_CHECK_MS = 1000

#: Every client this class creates reads dates as **aware UTC**.
#:
#: BSON stores an instant, not a wall clock, and without this PyMongo hands it
#: back with ``tzinfo=None``. Nothing downstream could then tell UTC from a
#: local reading of it: the API serialised ``2026-10-15T15:00:00`` with no
#: ``Z``, a browser in ``America/Sao_Paulo`` read that as 18:00 UTC, and a
#: measurement one minute old displayed as being in the future (R25).
#:
#: It is a read-side setting and changes no stored value. What it does change
#: is that a naive datetime can no longer be compared with one that came out
#: of the database, which is why the writers in this repository use
#: ``pylib.timeutil.utc_now`` rather than ``datetime.now()``.
TZ_AWARE = True


def _changes(stream, should_stop: Callable[[], bool] | None) -> Generator:
    """The stream's changes, until it ends or is told to stop."""
    if should_stop is None:
        yield from stream
        return
    while stream.alive and not should_stop():
        # Iterating blocks for as long as nothing happens; this returns None
        # after STOP_CHECK_MS instead.
        change = stream.try_next()
        if change is not None:
            yield change


class MongoDBConnection:
    def __init__(self, uri: str, db_name: str, *, reuse_client: bool = False):
        logger.info(f"[MongoDBConnection] uri:{uri} db_name:{db_name}")
        self.uri = uri
        self.db_name = db_name
        # The API owns one factory per worker and can share PyMongo's pool.
        # Other services retain their existing connection lifecycle.
        self._client = (
            MongoClient(uri, tz_aware=TZ_AWARE) if reuse_client else None)

    @contextmanager
    def connect(self) -> Generator:
        if self._client is not None:
            yield self._client[self.db_name]
            return
        client = MongoClient(self.uri, tz_aware=TZ_AWARE)
        try:
            yield client[self.db_name]
        finally:
            client.close()

    @contextmanager
    def transaction(self) -> Generator:
        """Run a block of writes atomically, yielding ``(db, session)``.

        Every operation inside has to be passed ``session=session`` or it
        runs outside the transaction and silently loses the guarantee.

        What this buys, and what it does not: MongoDB gives snapshot
        isolation, so two transactions that write the **same document**
        conflict and one aborts. It does *not* stop a document that did not
        exist in the snapshot from being inserted by someone else. So a
        composite operation that must exclude a concurrent insert has to make
        both sides write one shared document — the host, in the case this was
        added for — rather than merely read it.

        Requires a replica set, which this deployment already requires for
        change streams.
        """
        if self._client is not None:
            client = self._client
            with client.start_session() as session:
                with session.start_transaction():
                    yield client[self.db_name], session
            return
        client = MongoClient(self.uri, tz_aware=TZ_AWARE)
        try:
            with client.start_session() as session:
                with session.start_transaction():
                    yield client[self.db_name], session
        finally:
            client.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def waiting_ping(self) -> None:
        while True:
            try:
                with MongoClient(self.uri, tz_aware=TZ_AWARE) as client:
                    client.admin.command("ping")
                break
            except pymongo.errors.ConnectionFailure:
                logger.error("[WorkGenerator] Aguardando conexão com MongoDB...")
                time.sleep(3)

    def watch_collection(self,
        collection_name: str,
        pipeline: list[dict],
        on_change: Callable[[dict], None],
        full_document: str = "default",
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """
        Observes changes in a specific collection using a given pipeline.
        Automatically retries with exponential backoff on failure, resuming
        from the last processed event token to avoid losing events.

        Args:
            collection_name (str): MongoDB collection name.
            pipeline (list): Aggregation pipeline (e.g., [$match]).
            on_change (Callable): Callback function called on each event.
            full_document (str): Full document retrieval mode.
            on_open (Callable): Runs each time the stream opens, before its
                events are handled. The place to scan for what already exists:
                what happens after the scan is delivered by the stream, so
                nothing falls between the two. It has to be idempotent, since
                the same document can be seen by both.
            should_stop (Callable): Asked about once a second. When true the
                stream is closed, with its client, and this returns. Without
                it this never returns, which suits a watcher that lives as
                long as the process and leaks one that does not.
        """
        resume_token = None
        backoff = 1
        while not (should_stop and should_stop()):
            try:
                with self.connect() as db:
                    collection: Collection = db[collection_name]
                    kwargs = {"full_document": full_document}
                    if should_stop:
                        kwargs["max_await_time_ms"] = STOP_CHECK_MS
                    if resume_token:
                        kwargs["resume_after"] = resume_token
                    with collection.watch(pipeline, **kwargs) as stream:
                        backoff = 1
                        if on_open:
                            on_open()
                        for change in _changes(stream, should_stop):
                            resume_token = change.get("_id")
                            on_change(change)
            except PyMongoError as e:
                logger.error(
                    "[watch_collection] Error on '%s', retrying in %ss: %s",
                    collection_name, backoff, e
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
