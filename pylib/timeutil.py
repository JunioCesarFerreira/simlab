"""Timestamps that mean the same thing in every process.

Most of this repository writes ``datetime.now()`` — naive, and therefore in
whatever zone the process happens to run in. That worked while everything ran
in one container with its clock on UTC, and stopped being a safe assumption
once simulations started running on other machines: a reservation made by the
master-node and read back by an operator's tooling are the same instant only
if both agree on the zone, and a naive value does not say which it is.

So the collections this feature introduced store **aware UTC**, and the two
helpers below exist because the migration is partial on purpose. Documents
written before it — and the collections not yet converted — hold naive values,
and arithmetic that mixes the two raises ``TypeError`` rather than being
quietly wrong. ``elapsed_since`` takes either.
"""
from datetime import datetime, timezone
from typing import Any, Optional

__all__ = ["utc_now", "elapsed_since", "as_aware"]


def utc_now() -> datetime:
    """Now, as an aware UTC instant."""
    return datetime.now(timezone.utc)


def as_aware(value: datetime) -> datetime:
    """The same instant, with a zone attached — reading naive values as UTC.

    That reading is correct for the documents this system stores, because the
    processes that wrote them run with their clocks on UTC: the containers do,
    and so does the deployment. It is *not* correct for a naive value produced
    on a machine in another zone, and the difference is exactly the ambiguity
    the migration removes — a naive timestamp does not record which clock
    produced it, so the reader has to assume, and an assumption is all this
    function can offer for data already written.

    New writes go through :func:`utc_now` and carry their zone, so they need
    no assumption at all.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def elapsed_since(value: Any, *, now: Optional[datetime] = None) -> Optional[float]:
    """Seconds since ``value``, or ``None`` if it is not a timestamp.

    Accepts naive and aware alike, which is the point: a caller comparing a
    stored timestamp to the present should not have to know which kind the
    document happens to hold, and getting it wrong is a ``TypeError`` on a
    path that usually only runs when something has already gone wrong.
    """
    if not isinstance(value, datetime):
        return None
    reference = as_aware(now) if now is not None else utc_now()
    return (reference - as_aware(value)).total_seconds()
