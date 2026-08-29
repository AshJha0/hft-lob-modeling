"""Event stream schema, CSV i/o, and deterministic book replay.

An event stream is an ordered list of :class:`Event` rows.  Replaying a
stream through a fresh :class:`~lobmod.book.LimitOrderBook` is fully
deterministic, so recorded golden book snapshots must be reproduced exactly
by every language implementation.

After *each* event the replay records the top-of-book state; the arrays in
:class:`BookStates` are the single input to the feature pipeline
(:mod:`lobmod.features`).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .book import LimitOrderBook


@dataclass(frozen=True)
class Event:
    """One order-flow event.

    Attributes:
        seq: 1-based sequence number.
        etype: 'L' (limit add), 'C' (cancel by id), 'M' (market order).
        side: 'B' or 'S' ('L'/'M'; for 'C' it echoes the resting order's side).
        price: price in ticks ('L' only; 0 otherwise).
        size: order size ('L'/'M'; 0 for 'C').
        order_id: id of the order ('L': new id; 'C': id to cancel; 'M': 0).
    """

    seq: int
    etype: str
    side: str
    price: int
    size: int
    order_id: int


@dataclass
class BookStates:
    """Top-of-book state after each replayed event (arrays of length n_events).

    Sides that are empty at an event produce NaN in the float arrays and 0 in
    the size arrays for that row.
    """

    best_bid: np.ndarray  #: best bid price in ticks (float, NaN if none)
    best_ask: np.ndarray  #: best ask price in ticks (float, NaN if none)
    bid_size: np.ndarray  #: total size at best bid (float, 0 if none)
    ask_size: np.ndarray  #: total size at best ask (float, 0 if none)
    bid_depth_k: np.ndarray  #: summed size over top-k bid levels
    ask_depth_k: np.ndarray  #: summed size over top-k ask levels
    mid: np.ndarray  #: (bid+ask)/2 in ticks (NaN guarded)
    microprice: np.ndarray  #: size-weighted mid (NaN guarded)
    spread: np.ndarray  #: ask-bid in ticks (NaN guarded)

    def __len__(self) -> int:
        return int(self.mid.shape[0])


_FIELDS = ["seq", "type", "side", "price", "size", "order_id"]


def write_events_csv(path: str, events: Iterable[Event]) -> None:
    """Write events to CSV with the pinned header ``seq,type,side,price,size,order_id``."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_FIELDS)
        for e in events:
            writer.writerow([e.seq, e.etype, e.side, e.price, e.size, e.order_id])


def read_events_csv(path: str) -> List[Event]:
    """Read an event stream CSV (schema as written by :func:`write_events_csv`).

    Raises:
        ValueError: on a malformed header or row.
    """
    events: List[Event] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != _FIELDS:
            raise ValueError(f"unexpected event CSV header: {header}")
        for row in reader:
            if len(row) != 6:
                raise ValueError(f"malformed event row: {row}")
            events.append(
                Event(
                    seq=int(row[0]),
                    etype=row[1],
                    side=row[2],
                    price=int(row[3]),
                    size=int(row[4]),
                    order_id=int(row[5]),
                )
            )
    return events


def apply_event(book: LimitOrderBook, event: Event) -> None:
    """Apply one event to the book.

    Raises:
        ValueError: unknown event type, or any book-level validation error
            (e.g. cancel of an unknown id).
    """
    if event.etype == "L":
        book.limit(event.side, float(event.price), event.size, event.order_id)
    elif event.etype == "C":
        book.cancel(event.order_id)
    elif event.etype == "M":
        book.market(event.side, event.size)
    else:
        raise ValueError(f"unknown event type {event.etype!r}")


def replay(events: Sequence[Event], depth_k: int = 3, tick_size: float = 1.0) -> BookStates:
    """Replay ``events`` through a fresh book, recording state after each event.

    Args:
        events: the ordered event stream.
        depth_k: number of levels for the depth arrays.
        tick_size: book tick size.

    Returns:
        BookStates arrays of length ``len(events)``.

    Raises:
        ValueError: if the stream is empty or an event is invalid.
    """
    if len(events) == 0:
        raise ValueError("event stream is empty")
    n = len(events)
    st = BookStates(*(np.empty(n) for _ in range(9)))
    book = LimitOrderBook(tick_size=tick_size)
    for i, event in enumerate(events):
        apply_event(book, event)
        b, a = book.best_bid(), book.best_ask()
        st.best_bid[i] = math.nan if b is None else float(b)
        st.best_ask[i] = math.nan if a is None else float(a)
        st.bid_size[i] = float(book.best_bid_size())
        st.ask_size[i] = float(book.best_ask_size())
        st.bid_depth_k[i] = float(book.depth("B", depth_k))
        st.ask_depth_k[i] = float(book.depth("S", depth_k))
        st.mid[i] = book.mid()
        st.microprice[i] = book.microprice()
        st.spread[i] = book.spread()
    return st
