"""Event stream schema, CSV i/o, and deterministic book replay.

An event stream is an ordered list of :class:`Event` rows.  Replaying a
stream through a fresh :class:`~lobmod.book.LimitOrderBook` is fully
deterministic, so recorded golden book snapshots must be reproduced exactly
by every language implementation.

After *each* event the replay records the top-of-book state; the arrays in
:class:`BookStates` are the single input to the feature pipeline
(:mod:`lobmod.features`).

CSV rules (pinned, identical in every port)
--------------------------------------------
* header exactly ``seq,type,side,price,size,order_id``;
* a trailing ``\r`` (CRLF files) is stripped from every line;
* trailing blank lines are ignored; a blank line *before* the last data row
  is an error (truncated / corrupted file);
* every row has exactly six comma-separated fields with no surrounding
  whitespace; integer fields are an optional ``-`` followed by digits only
  and must fit a signed 64-bit integer; ``type`` is one of ``L``/``C``/``M``
  and ``side`` one of ``B``/``S`` — anything else is an error.

``Event.price`` is a **tick count**, never a currency price, which is why
:func:`replay` takes no tick size: the book it drives always has
``tick_size = 1`` and events are applied through
:meth:`~lobmod.book.LimitOrderBook.limit_ticks`.

Schema v2 (optional, separate type)
-----------------------------------
:class:`EventV2` adds a nanosecond timestamp and the L3 types ``R``, ``U`` and
``E``, under the header ``seq,ts_ns,type,side,price,size,order_id,new_order_id``.
:func:`read_events_v2_csv` detects the header, so it also reads v1 files, with
``ts_ns = 0``.  :func:`read_events_csv` and v1 files are unchanged.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, TypeVar

import numpy as np

from ._validate import as_positive_int
from .book import LimitOrderBook

_INT_RE = re.compile(r"-?[0-9]+")
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1


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


@dataclass(frozen=True)
class EventV2:
    """One order-flow event of schema v2.

    Attributes:
        seq: 1-based sequence number.
        ts_ns: non-negative timestamp in nanoseconds (0 when lifted from v1).
        etype: 'L', 'C', 'M' as in :class:`Event`, or 'R' (reduce), 'U'
            (replace), 'E' (execute a named resting order).
        side: 'B' or 'S' (for 'C', 'R', 'U', 'E' it echoes the resting side).
        price: price in ticks ('L' and 'U'; 0 otherwise).
        size: 'L'/'M' order size; 'R' size removed; 'U' new size; 'E' size
            executed; 0 for 'C'.
        order_id: 'L' new id; 'C'/'R'/'U'/'E' id of the resting order; 'M' 0.
        new_order_id: 'U' replacement id; 0 otherwise.
    """

    seq: int
    ts_ns: int
    etype: str
    side: str
    price: int
    size: int
    order_id: int
    new_order_id: int = 0


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
_HEADER = ",".join(_FIELDS)
_EVENT_TYPES = ("L", "C", "M")
_SIDES = ("B", "S")
_FIELDS_V2 = ["seq", "ts_ns", "type", "side", "price", "size", "order_id", "new_order_id"]
_HEADER_V2 = ",".join(_FIELDS_V2)
_EVENT_TYPES_V2 = ("L", "C", "M", "R", "U", "E")
_E = TypeVar("_E")


def parse_int64(field: str, context: str) -> int:
    """Strict signed 64-bit integer parse (pinned: ``-?[0-9]+`` only).

    Raises:
        ValueError: on any other text or an out-of-range value.
    """
    if not _INT_RE.fullmatch(field):
        raise ValueError(f"malformed integer field {field!r} in {context}")
    value = int(field)
    if value < _I64_MIN or value > _I64_MAX:
        raise ValueError(f"integer field {field!r} out of 64-bit range in {context}")
    return value


def read_csv_lines(path: str, expected_header: str) -> List[str]:
    """Read a pinned-schema CSV: validate the header, strip one ``\r`` per line, drop
    trailing blank lines, and return the data lines.

    Raises:
        ValueError: bad header, or a blank line before the last data row.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [ln[:-1] if ln.endswith("\r") else ln for ln in fh.read().split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0] != expected_header:
        got = lines[0] if lines else "<empty file>"
        raise ValueError(f"unexpected CSV header {got!r}; expected {expected_header!r}")
    data = lines[1:]
    for i, ln in enumerate(data):
        if ln == "":
            raise ValueError(f"blank line at row {i + 1} of {path} (only trailing blank lines are allowed)")
    return data


def write_events_csv(path: str, events: Iterable[Event]) -> None:
    """Write events to CSV with the pinned header ``seq,type,side,price,size,order_id``."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(_FIELDS)
        for e in events:
            writer.writerow([e.seq, e.etype, e.side, e.price, e.size, e.order_id])


def read_events_csv(path: str) -> List[Event]:
    """Read an event stream CSV (schema as written by :func:`write_events_csv`).

    Applies the pinned CSV rules in the module docstring.

    Raises:
        ValueError: on a malformed header or row.
    """
    events: List[Event] = []
    for i, line in enumerate(read_csv_lines(path, _HEADER)):
        row = line.split(",")
        context = f"event row {i + 1}"
        if len(row) != 6:
            raise ValueError(f"malformed {context}: expected 6 fields, got {len(row)}: {line!r}")
        if row[1] not in _EVENT_TYPES:
            raise ValueError(f"unknown event type {row[1]!r} in {context}")
        if row[2] not in _SIDES:
            raise ValueError(f"side must be 'B' or 'S', got {row[2]!r} in {context}")
        events.append(
            Event(
                seq=parse_int64(row[0], context),
                etype=row[1],
                side=row[2],
                price=parse_int64(row[3], context),
                size=parse_int64(row[4], context),
                order_id=parse_int64(row[5], context),
            )
        )
    return events


def write_events_v2_csv(path: str, events: Iterable[EventV2]) -> None:
    """Write schema v2 events with the header ``seq,ts_ns,type,side,price,size,order_id,new_order_id``."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(_FIELDS_V2)
        for e in events:
            writer.writerow(
                [e.seq, e.ts_ns, e.etype, e.side, e.price, e.size, e.order_id, e.new_order_id]
            )


def read_events_v2_csv(path: str) -> List[EventV2]:
    """Read a v2 event CSV, or a v1 file lifted to :class:`EventV2` with ``ts_ns = 0``.

    The header selects the schema.  The pinned CSV rules apply, with eight
    fields per row, ``type`` one of ``L C M R U E`` and ``ts_ns >= 0``.

    Raises:
        ValueError: on a header matching neither schema, or a malformed row.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        first = fh.readline().rstrip("\n")
    if (first[:-1] if first.endswith("\r") else first) != _HEADER_V2:
        return [
            EventV2(e.seq, 0, e.etype, e.side, e.price, e.size, e.order_id, 0)
            for e in read_events_csv(path)
        ]
    events: List[EventV2] = []
    for i, line in enumerate(read_csv_lines(path, _HEADER_V2)):
        row = line.split(",")
        context = f"event row {i + 1}"
        if len(row) != 8:
            raise ValueError(f"malformed {context}: expected 8 fields, got {len(row)}: {line!r}")
        if row[2] not in _EVENT_TYPES_V2:
            raise ValueError(f"unknown event type {row[2]!r} in {context}")
        if row[3] not in _SIDES:
            raise ValueError(f"side must be 'B' or 'S', got {row[3]!r} in {context}")
        ts_ns = parse_int64(row[1], context)
        if ts_ns < 0:
            raise ValueError(f"negative ts_ns {ts_ns} in {context}")
        events.append(
            EventV2(
                seq=parse_int64(row[0], context),
                ts_ns=ts_ns,
                etype=row[2],
                side=row[3],
                price=parse_int64(row[4], context),
                size=parse_int64(row[5], context),
                order_id=parse_int64(row[6], context),
                new_order_id=parse_int64(row[7], context),
            )
        )
    return events


def apply_event(book: LimitOrderBook, event: Event) -> None:
    """Apply one event to the book (``event.price`` is a tick count).

    Raises:
        ValueError: unknown event type, or any book-level validation error
            (e.g. cancel of an unknown id, price outside ``[1, MAX_TICKS]``).
    """
    if event.etype == "L":
        book.limit_ticks(event.side, event.price, event.size, event.order_id)
    elif event.etype == "C":
        book.cancel(event.order_id)
    elif event.etype == "M":
        book.market(event.side, event.size)
    else:
        raise ValueError(f"unknown event type {event.etype!r}")


def apply_event_v2(book: LimitOrderBook, event: EventV2) -> None:
    """Apply one schema v2 event to the book (``event.price`` is a tick count).

    Raises:
        ValueError: unknown event type, or any book-level validation error.
    """
    if event.etype == "L":
        book.limit_ticks(event.side, event.price, event.size, event.order_id)
    elif event.etype == "C":
        book.cancel(event.order_id)
    elif event.etype == "M":
        book.market(event.side, event.size)
    elif event.etype == "R":
        book.reduce(event.order_id, event.size)
    elif event.etype == "U":
        book.replace_ticks(event.order_id, event.new_order_id, event.price, event.size)
    elif event.etype == "E":
        book.execute(event.order_id, event.size)
    else:
        raise ValueError(f"unknown event type {event.etype!r}")


def replay(events: Sequence[Event], depth_k: int = 3) -> BookStates:
    """Replay ``events`` through a fresh ``tick_size = 1`` book, recording
    the top-of-book state after each event.

    ``Event.price`` is already a tick count, so no tick size is involved: all
    recorded prices are in ticks.

    Args:
        events: the ordered event stream.
        depth_k: number of levels for the depth arrays (positive integer).

    Returns:
        BookStates arrays of length ``len(events)``.

    Raises:
        ValueError: if the stream is empty, ``depth_k`` is invalid, or an
            event is invalid.
    """
    return _replay_with(events, depth_k, apply_event)


def replay_v2(events: Sequence[EventV2], depth_k: int = 3) -> BookStates:
    """As :func:`replay`, for schema v2 events."""
    return _replay_with(events, depth_k, apply_event_v2)


def _replay_with(
    events: Sequence[_E], depth_k: int, apply: Callable[[LimitOrderBook, _E], None]
) -> BookStates:
    depth_k = as_positive_int("depth_k", depth_k)
    if len(events) == 0:
        raise ValueError("event stream is empty")
    n = len(events)
    st = BookStates(*(np.empty(n) for _ in range(9)))
    book = LimitOrderBook(tick_size=1.0, record_trades=False)
    for i, event in enumerate(events):
        apply(book, event)
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
