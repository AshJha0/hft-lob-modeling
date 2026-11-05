"""Streaming parser for Nasdaq TotalView-ITCH 5.0 files.

**Verification status: tested on fixtures built by hand from the public
specification, and run once on one real sample file for ten symbols
with no anomalies.**  That is internal consistency only: one file, a few symbols,
and no comparison with an independent book or a reference parser.

Format
------
A file is a sequence of frames: a 2-byte big-endian length, then that many
message bytes.  Each message starts with type (1), stock locate (2), tracking
number (2) and a 6-byte big-endian timestamp in nanoseconds since midnight.
Prices are 4-byte integers with four implied decimals, so the raw value is a
tick count at a tick of 0.0001 and passes straight into ``EventV2.price``.

Mapping
-------
======  ==========================  =============================================
Type    Message                     Output
======  ==========================  =============================================
S       system event                :class:`StatusMessage`
R       stock directory             :class:`StatusMessage`; learns locate -> symbol
H       trading action              :class:`StatusMessage`
A, F    add order                   event ``L``
E       order executed              event ``E`` and a :class:`TradeReport`
C       executed with price         event ``E`` and a :class:`TradeReport` with the
                                    message's price, which may differ from the
                                    resting price the book uses
X       order cancel (partial)      event ``R``
D       order delete                event ``C``
U       order replace               event ``U``
P       hidden trade                :class:`TradeReport` only; the book is untouched
Q       cross trade                 :class:`TradeReport` only; the book is untouched
======  ==========================  =============================================

Other types the specification defines (for example ``L``, ``I``, ``B``) are
skipped and counted under ``ignored``.  A broken trade (``B``) is therefore not
reversed in any :class:`TradeReport` already emitted.

Strict and lenient
------------------
Every anomaly increments ``stats[name]``.  With ``strict=True`` it also raises
:class:`ItchError`; with ``strict=False`` the message is dropped, except that
an execution or cancel larger than the order is clamped to the remaining size.
Names: ``unknown_type``, ``bad_length``, ``truncated``, ``unknown_order``,
``duplicate_order``, ``oversize``, ``invalid_order``, ``ts_regression``.

An ITCH *file* has no sequence numbers, so a dropped message cannot be seen
directly.  It shows up only indirectly: as a reference to an unknown order, an
oversize, a timestamp going backwards within one stock locate, or a truncated
final frame.

Orders of other symbols are skipped, not anomalies.  A reference to an unknown
order counts as ``other_symbol`` when its locate is not known to belong to a
requested symbol, so with a filter a gap before the first directory or add
message for a stock cannot be told apart from another stock's flow.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterable, Iterator, Optional, Set, Tuple, Union

from ..book import LimitOrderBook
from ..replay import EventV2, apply_event_v2

_LENGTHS = {
    b"S": 12, b"R": 39, b"H": 25, b"A": 36, b"F": 40, b"E": 31, b"C": 36,
    b"X": 23, b"D": 19, b"U": 35, b"P": 44, b"Q": 40,
}  # fmt: skip
#: types the specification defines that have no effect here
_IGNORED = frozenset(b"LVWKJhNIYBO"[i : i + 1] for i in range(11))
_I64_MAX = (1 << 63) - 1
_HEADER = struct.Struct(">HH")  # stock locate, tracking number at offset 1


class ItchError(ValueError):
    """A malformed or inconsistent message under ``strict=True``."""


@dataclass(frozen=True)
class FeedEvent:
    """A book event for one symbol; ``event.seq`` counts per symbol from 1."""

    symbol: str
    event: EventV2


@dataclass(frozen=True)
class TradeReport:
    """An execution as printed by the feed.

    Attributes:
        ts_ns: nanoseconds since midnight.
        symbol: stock symbol.
        source: message type, 'E', 'C', 'P' or 'Q'.
        side: side of the resting order; '' for a cross.
        price: execution price in ticks of 0.0001.
        size: executed size.
        match_number: the feed's match number.
        order_id: resting order id; 0 for a cross, and as sent for 'P'.
        printable: False when 'C' says not to count it in volume.
    """

    ts_ns: int
    symbol: str
    source: str
    side: str
    price: int
    size: int
    match_number: int
    order_id: int
    printable: bool = True


@dataclass(frozen=True)
class StatusMessage:
    """A system event (S), stock directory (R) or trading action (H).

    ``code`` is the event code (S), market category (R) or trading state (H);
    ``symbol`` is '' for S.
    """

    ts_ns: int
    msg_type: str
    symbol: str
    code: str


FeedItem = Union[FeedEvent, TradeReport, StatusMessage]


def iter_frames(fh: BinaryIO) -> Iterator[Optional[bytes]]:
    """Yield message bytes from length-prefixed frames, reading incrementally.

    Yields ``None`` once, last, if the file ends inside a frame.  ``fh`` may be
    any binary stream with blocking ``read``, such as ``gzip.open(path, "rb")``.
    """
    while True:
        prefix = fh.read(2)
        if not prefix:
            return
        if len(prefix) < 2:
            yield None
            return
        (length,) = struct.unpack(">H", prefix)
        body = fh.read(length)
        if len(body) < length:
            yield None
            return
        yield body


def _ts(msg: bytes) -> int:
    return int.from_bytes(msg[5:11], "big")


def _u32(msg: bytes, offset: int) -> int:
    return int.from_bytes(msg[offset : offset + 4], "big")


def _u64(msg: bytes, offset: int) -> int:
    return int.from_bytes(msg[offset : offset + 8], "big")


def _stock(msg: bytes, offset: int) -> str:
    return msg[offset : offset + 8].decode("ascii").rstrip(" ")


class Itch50Parser:
    """Stateful ITCH 5.0 parser; see the module docstring.

    Args:
        symbols: symbols to keep, or None for every symbol.  Book events carry
            their symbol; keep one book per symbol.
        strict: raise on an anomaly instead of only counting it.

    Attributes:
        stats: :class:`collections.Counter` of ``messages``, each anomaly name,
            ``ignored`` and ``other_symbol``.
    """

    def __init__(self, symbols: Optional[Iterable[str]] = None, strict: bool = True) -> None:
        self.symbols: Optional[Set[str]] = None if symbols is None else set(symbols)
        self.strict = bool(strict)
        self.stats: Counter = Counter()
        self._locates: Dict[int, str] = {}
        # order ref -> [symbol, side, remaining size]
        self._orders: Dict[int, list] = {}
        self._seq: Counter = Counter()
        self._last_ts: Dict[int, int] = {}

    def _anomaly(self, name: str, detail: str) -> None:
        self.stats[name] += 1
        if self.strict:
            raise ItchError(f"{name}: {detail}")

    def _wanted(self, symbol: str) -> bool:
        return self.symbols is None or symbol in self.symbols

    def _event(self, symbol: str, ts_ns: int, etype: str, side: str, price: int,
               size: int, order_id: int, new_order_id: int = 0) -> FeedEvent:  # fmt: skip
        self._seq[symbol] += 1
        return FeedEvent(
            symbol,
            EventV2(self._seq[symbol], ts_ns, etype, side, price, size, order_id, new_order_id),
        )

    def parse(self, fh: BinaryIO) -> Iterator[FeedItem]:
        """Yield the items of a framed ITCH stream, in file order."""
        for frame in iter_frames(fh):
            if frame is None:
                self._anomaly("truncated", "file ends inside a frame")
                return
            yield from self.parse_message(frame)

    def parse_message(self, msg: bytes) -> Iterator[FeedItem]:
        """Yield the items of one unframed message."""
        self.stats["messages"] += 1
        mtype = msg[:1]
        if mtype in _IGNORED:
            self.stats["ignored"] += 1
            return
        expected = _LENGTHS.get(mtype)
        if expected is None:
            self._anomaly("unknown_type", f"message type {mtype!r}")
            return
        if len(msg) != expected:
            self._anomaly("bad_length", f"type {mtype!r} has {len(msg)} bytes, not {expected}")
            return
        locate = _HEADER.unpack_from(msg, 1)[0]
        ts_ns = _ts(msg)
        if ts_ns < self._last_ts.get(locate, 0):
            self._anomaly("ts_regression", f"locate {locate}: {ts_ns} after {self._last_ts[locate]}")
        self._last_ts[locate] = max(ts_ns, self._last_ts.get(locate, 0))
        yield from getattr(self, "_on_" + mtype.decode("ascii"))(msg, locate, ts_ns)

    # ------------------------------------------------------------ status

    def _on_S(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        yield StatusMessage(ts_ns, "S", "", msg[11:12].decode("ascii"))

    def _on_R(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        symbol = _stock(msg, 11)
        self._locates[locate] = symbol
        if self._wanted(symbol):
            yield StatusMessage(ts_ns, "R", symbol, msg[19:20].decode("ascii"))
        else:
            self.stats["other_symbol"] += 1

    def _on_H(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        symbol = _stock(msg, 11)
        if self._wanted(symbol):
            yield StatusMessage(ts_ns, "H", symbol, msg[19:20].decode("ascii"))
        else:
            self.stats["other_symbol"] += 1

    # ------------------------------------------------------------ orders

    def _valid_new(self, ref: int, size: int, price: int) -> bool:
        if ref > _I64_MAX or size < 1 or price < 1:
            self._anomaly("invalid_order", f"ref {ref}, size {size}, price {price}")
            return False
        if ref in self._orders:
            self._anomaly("duplicate_order", f"ref {ref} is already live")
            return False
        return True

    def _on_A(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        symbol = _stock(msg, 24)
        self._locates.setdefault(locate, symbol)
        if not self._wanted(symbol):
            self.stats["other_symbol"] += 1
            return
        ref, side = _u64(msg, 11), msg[19:20].decode("ascii")
        size, price = _u32(msg, 20), _u32(msg, 32)
        if side not in ("B", "S"):
            self._anomaly("invalid_order", f"side {side!r}")
            return
        if self._valid_new(ref, size, price):
            self._orders[ref] = [symbol, side, size]
            yield self._event(symbol, ts_ns, "L", side, price, size, ref)

    _on_F = _on_A  # the attribution field at offset 36 is not used

    def _live(self, ref: int, locate: int) -> Optional[list]:
        """The order behind ``ref``, or None after counting why it is missing."""
        order = self._orders.get(ref)
        if order is not None:
            return order
        if self.symbols is not None and self._locates.get(locate) not in self.symbols:
            self.stats["other_symbol"] += 1
        else:
            self._anomaly("unknown_order", f"ref {ref}")
        return None

    def _take(self, ref: int, order: list, size: int) -> Optional[int]:
        """Validate ``size`` against ``order`` and deduct it; None drops the message."""
        if size < 1:
            self._anomaly("invalid_order", f"ref {ref}: size {size}")
            return None
        if size > order[2]:
            self._anomaly("oversize", f"ref {ref}: {size} above remaining {order[2]}")
            size = order[2]
        order[2] -= size
        if order[2] == 0:
            del self._orders[ref]
        return size

    def _on_E(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        ref, order = _u64(msg, 11), self._live(_u64(msg, 11), locate)
        if order is None:
            return
        size = self._take(ref, order, _u32(msg, 19))
        if size is None:
            return
        yield self._event(order[0], ts_ns, "E", order[1], 0, size, ref)
        # an 'E' message has no price; 0 means "the resting order's price"
        yield TradeReport(ts_ns, order[0], "E", order[1], 0, size, _u64(msg, 23), ref)

    def _on_C(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        ref, order = _u64(msg, 11), self._live(_u64(msg, 11), locate)
        if order is None:
            return
        size = self._take(ref, order, _u32(msg, 19))
        if size is None:
            return
        yield self._event(order[0], ts_ns, "E", order[1], 0, size, ref)
        yield TradeReport(ts_ns, order[0], "C", order[1], _u32(msg, 32), size, _u64(msg, 23),
                          ref, msg[31:32] == b"Y")  # fmt: skip

    def _on_X(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        ref, order = _u64(msg, 11), self._live(_u64(msg, 11), locate)
        if order is None:
            return
        size = self._take(ref, order, _u32(msg, 19))
        if size is not None:
            yield self._event(order[0], ts_ns, "R", order[1], 0, size, ref)

    def _on_D(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        ref, order = _u64(msg, 11), self._live(_u64(msg, 11), locate)
        if order is not None:
            del self._orders[ref]
            yield self._event(order[0], ts_ns, "C", order[1], 0, 0, ref)

    def _on_U(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        ref, new_ref = _u64(msg, 11), _u64(msg, 19)
        order = self._live(ref, locate)
        if order is None:
            return
        size, price = _u32(msg, 27), _u32(msg, 31)
        if new_ref != ref and not self._valid_new(new_ref, size, price):
            return
        if new_ref == ref and (size < 1 or price < 1):
            self._anomaly("invalid_order", f"ref {ref}, size {size}, price {price}")
            return
        del self._orders[ref]
        self._orders[new_ref] = [order[0], order[1], size]
        yield self._event(order[0], ts_ns, "U", order[1], price, size, ref, new_ref)

    # ------------------------------------------------------------ off-book

    def _on_P(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        symbol = _stock(msg, 24)
        if self._wanted(symbol):
            yield TradeReport(ts_ns, symbol, "P", msg[19:20].decode("ascii"), _u32(msg, 32),
                              _u32(msg, 20), _u64(msg, 36), _u64(msg, 11))  # fmt: skip
        else:
            self.stats["other_symbol"] += 1

    def _on_Q(self, msg: bytes, locate: int, ts_ns: int) -> Iterator[FeedItem]:
        symbol = _stock(msg, 19)
        if self._wanted(symbol):
            yield TradeReport(ts_ns, symbol, "Q", "", _u32(msg, 27), _u64(msg, 11),
                              _u64(msg, 31), 0)  # fmt: skip
        else:
            self.stats["other_symbol"] += 1


def events_for_symbol(fh: BinaryIO, symbol: str, strict: bool = True) -> Iterator[EventV2]:
    """Yield only the book events of ``symbol`` from a framed ITCH stream."""
    for item in Itch50Parser([symbol], strict).parse(fh):
        if isinstance(item, FeedEvent):
            yield item.event


class BookIntegrityChecker:
    """Drive a book from one symbol's events and check it stays sound.

    :class:`~lobmod.book.LimitOrderBook` matches any add at or through the
    opposite best, while a feed reports executions separately, so a feed add
    that locks or crosses (conceivable while a stock is halted or before a
    cross) must not reach the book: it would invent trades.  Such an event
    counts under ``crossed``.  With ``strict=True`` it raises :class:`ItchError`;
    otherwise the order is held out of the book, and later events on it count
    under ``skipped``.  A book-level rejection counts under ``book_errors``.

    Attributes:
        book: the book, with ``tick_size = 1`` and no history.
        counts: :class:`collections.Counter` of ``events``, one entry per event
            type, and the names above.
    """

    def __init__(self, strict: bool = True) -> None:
        self.book = LimitOrderBook(tick_size=1.0, record_trades=False)
        self.strict = bool(strict)
        self.counts: Counter = Counter()
        self._held: Dict[int, int] = {}  # order id -> remaining size

    def _would_cross(self, side: str, price: int) -> bool:
        if side == "B":
            ask = self.book.best_ask()
            return ask is not None and price >= ask
        bid = self.book.best_bid()
        return bid is not None and price <= bid

    def _fail(self, name: str, detail: str) -> None:
        self.counts[name] += 1
        if self.strict:
            raise ItchError(f"{name}: {detail}")

    def apply(self, event: EventV2) -> None:
        """Apply one event, then check the top of book."""
        self.counts["events"] += 1
        self.counts[event.etype] += 1
        if event.order_id in self._held and event.etype != "L":
            self._apply_held(event)
            return
        try:
            if event.etype == "L" and self._would_cross(event.side, event.price):
                self._fail("crossed", f"add of order {event.order_id} at {event.price}")
                self._held[event.order_id] = event.size
            elif event.etype == "U" and self._would_cross(event.side, event.price):
                self._fail("crossed", f"replace to order {event.new_order_id} at {event.price}")
                self.book.cancel(event.order_id)
                self._held[event.new_order_id] = event.size
            else:
                apply_event_v2(self.book, event)
        except ItchError:
            raise
        except ValueError as exc:
            self._fail("book_errors", str(exc))
        bid, ask = self.book.best_bid(), self.book.best_ask()
        if bid is not None and ask is not None and bid >= ask:
            raise ItchError(f"book crossed at {bid}/{ask} after event {event.seq}")

    def _apply_held(self, event: EventV2) -> None:
        self.counts["skipped"] += 1
        if event.etype in ("E", "R"):
            self._held[event.order_id] -= event.size
            if self._held[event.order_id] <= 0:
                del self._held[event.order_id]
        if event.etype in ("C", "U"):
            self._held.pop(event.order_id, None)
        if event.etype == "U":
            if self._would_cross(event.side, event.price):
                self._held[event.new_order_id] = event.size
            else:
                self.book.limit_ticks(event.side, event.price, event.size, event.new_order_id)

    def verify(self) -> Tuple[int, int]:
        """Walk the whole book; returns the (bid, ask) order counts.

        Raises:
            ItchError: on a crossed book, a non-positive resting size, or
                totals or id index that disagree with the queues.
        """
        book = self.book
        counts = []
        for side, levels in ((book.BUY, book._bids), (book.SELL, book._asks)):
            sizes = [o[1] for queue in levels.values() for o in queue]
            if any(size <= 0 for size in sizes) or any(not q for q in levels.values()):
                raise ItchError(f"non-positive size or empty level on side {side}")
            if sum(sizes) != book.total_size(side) or len(sizes) != book.n_orders(side):
                raise ItchError(f"totals disagree with the queues on side {side}")
            counts.append(len(sizes))
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None and ask is not None and bid >= ask:
            raise ItchError(f"book crossed at {bid}/{ask}")
        return counts[0], counts[1]
