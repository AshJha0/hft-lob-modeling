"""Streaming parser for IEX HIST captures: IEX-TP framing, DEEP and DEEP+.

**Verification status: checked against byte fixtures built by hand from the
public specifications, and run once on one real HIST file, a DEEP+ capture of
2025-12-12 filtered to three symbols.**  That run reported no sequence gaps or
anomalies, and its book verified empty at the end of the day; see
research/results_iex.md.  The DEEP path has never met a real file, and one file
says nothing about other days, so claims below about how real files behave are
confirmed that far only.

Specifications
--------------
Fetched 2026-11-05 from the links at
https://www.iex.io/resources/trading/market-data :

* IEX Transport Specification v1.25 (May 12, 2017),
  https://www.iex.io/documents/iex-tp-v1
* IEX DEEP Specification v1.08, https://www.iex.io/documents/deep-v1-08
* IEX DEEP+ Specification v1.06 (October 8, 2026),
  https://www.iex.io/documents/iex-deep-plus-specification

The revision history of DEEP+ says the Add Order type byte was updated at an
early version (dated January 14, 2025 or before) without giving the old value,
so an older capture may report ``unknown_type`` for its adds.

Format
------
All binary fields are little endian.  A UDP payload is one segment: a 40-byte
header (version 1, reserved, protocol id at 2, channel at 4, session at 8,
payload length at 12, message count at 14, stream offset at 16, first message
sequence at 24, send time at 32), then message blocks, each a 2-byte length and
that many bytes.  A heartbeat has zero payload and count, and carries the *next*
sequence and offset, so it also reveals a gap.

Protocol ids: DEEP 0x8004, DEEP+ 0x8005.  Prices are signed 8-byte integers with
four implied decimals, so the raw value is a tick count at 0.0001, the same tick
as :mod:`lobmod.feeds.itch50`.  Timestamps are signed 8-byte nanoseconds since
the POSIX epoch, UTC; they are **not** nanoseconds since midnight as in ITCH.

DEEP+ is order by order for displayed orders.  Non-displayed orders and the
hidden part of reserve orders never appear.  DEEP is aggregated by price level.

Mapping
-------
======  ======  =======================  ======================================
Feed    Type    Message                  Output
======  ======  =======================  ======================================
both    S       system event             :class:`IexStatus`
both    D       security directory       :class:`SecurityDirectory`
both    H       trading status           :class:`IexStatus`, reason in ``detail``
both    O P E   halt, short sale, event  :class:`IexStatus`
both    B       trade break              :class:`TradeBreak`
both    T       trade                    :class:`IexTrade`, ``side`` ''
DEEP    8 5     price level update       :class:`LevelUpdate`
DEEP+   a       add order                event ``L``
DEEP+   M       order modify             event ``U``, ``R`` or none; see below
DEEP+   R       order delete             event ``C``
DEEP+   L       order executed           event ``E`` and an :class:`IexTrade`
DEEP+   C       clear book               event ``C`` for each live order
======  ======  =======================  ======================================

``I`` (retail indicator), and on DEEP ``X`` (official price) and ``A`` (auction
information), are skipped and counted under ``ignored``.  In DEEP ``T`` is every
fill; in DEEP+ it is only a match between non-displayed orders, displayed fills
arriving as ``L``.  A :class:`TradeBreak` does not reverse any earlier item.

An Order Modify keeps its Order ID, and its flag bit 0 says whether priority is
kept.  The book's replace always loses priority, so a modify that keeps priority
at the same price with a smaller size becomes ``R``, and one that changes nothing
emits no event (``modify_noop``).  Keeping priority while changing price or
raising size is not expressible and becomes ``U``, counted under
``modify_priority_lost``.

An Order Executed price may differ from the booked price (the specification
names price improvement and price sliding); :attr:`IexTrade.booked_price` keeps
both, and the book event is at the resting price as in ITCH.

Strict and lenient
------------------
Every anomaly increments ``stats[name]``.  With ``strict=True`` it also raises
:class:`IexError`; with ``strict=False`` the message or segment is dropped,
except that an oversize execution is clamped.  Segment names: ``bad_segment``,
``bad_version``, ``unknown_protocol``, ``bad_length``, ``bad_count``,
``truncated``, ``seq_gap``, ``offset_mismatch``.  Message names:
``unknown_type``, ``bad_length``, ``unknown_order``, ``duplicate_order``,
``oversize``, ``invalid_order``, ``ts_regression``.

The specification tells decoders to tolerate unknown types and messages that
grow at the end.  Growth is never an anomaly: extra bytes are ignored and counted
under ``grown_message``.  Unknown types stay anomalies, so that strict mode
exposes them; use ``strict=False`` to read on.

Sequence numbers
----------------
Each (protocol, channel, session) is one stream; DEEP+ may have several channels.
A forward jump in sequence is a ``seq_gap``, and ``gap_messages`` adds up what was
lost.  A segment wholly behind the stream is a line duplicate and is dropped
(``duplicate_segment``); the already-seen head of a partly overlapping segment is
dropped too (``duplicate_messages``), since the A and B lines may batch messages
differently.  A stream first seen beyond sequence 1 adds to ``late_start_messages``
without being an anomaly; that sequences start at 1 is an assumption, which the
specification does not state.  After a gap the order state is unreliable: expect
``unknown_order`` until the affected orders have gone.

Every message but S carries its symbol, so unlike ITCH a filter never confuses
another stock's flow with a gap, and no order needs tracking outside its symbol.
Orders are held per symbol, and ``ts_regression`` is checked per (type, symbol),
the only progression the specification promises.

Symbol filter
-------------
With ``symbols`` set, a segment is first walked at byte level: a well-formed
message of another symbol is counted and dropped without being copied.  A segment
with anything odd, and any message of an unknown type, of a length other than the
exact one, or of a wanted symbol, takes the ordinary path, so ``stats`` and the
items are the same as without the walk.  Sequence checks run on the header before
it, for every segment.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple, Union

from ..replay import EventV2
from .itch50 import BookIntegrityChecker, FeedEvent, ItchError, StatusMessage
from .pcap import CaptureReader, open_capture

__all__ = [
    "BookIntegrityChecker", "DEEP_PROTOCOL", "DEEP_PLUS_PROTOCOL", "FeedEvent", "IexError",
    "IexItem", "IexParser", "IexStatus", "IexTrade", "LevelBook", "LevelUpdate",
    "SecurityDirectory", "Segment", "TradeBreak", "iter_file", "parse_header",
]  # fmt: skip

DEEP_PROTOCOL = 0x8004
DEEP_PLUS_PROTOCOL = 0x8005
_HEADER = struct.Struct("<BBHIIHHqqq")
_SIDES = {0x38: "B", 0x35: "S"}
_COMMON = {b"S": 10, b"D": 31, b"H": 22, b"O": 18, b"P": 19, b"E": 18, b"T": 38, b"B": 38}
_LENGTHS = {
    DEEP_PROTOCOL: {**_COMMON, b"8": 30, b"5": 30},
    DEEP_PLUS_PROTOCOL: {**_COMMON, b"a": 38, b"M": 38, b"R": 26, b"L": 46, b"C": 18},
}
_IGNORED = {
    DEEP_PROTOCOL: frozenset((b"I", b"X", b"A")),
    DEEP_PLUS_PROTOCOL: frozenset((b"I",)),
}
#: protocol -> type byte -> exact length, -1 for ignored, 0 for unknown
_TABLES = {
    protocol: [
        -1 if bytes((t,)) in _IGNORED[protocol] else lengths.get(bytes((t,)), 0) for t in range(256)
    ]
    for protocol, lengths in _LENGTHS.items()
}
_SYSTEM_EVENT = 0x53
_I64 = struct.Struct("<q")
_U32 = struct.Struct("<I")


class IexError(ItchError):
    """A malformed or inconsistent segment or message under ``strict=True``."""


class Segment(NamedTuple):
    """A decoded IEX-TP header."""

    version: int
    protocol: int
    channel: int
    session: int
    payload_length: int
    message_count: int
    stream_offset: int
    first_seq: int
    send_time_ns: int


def parse_header(payload: bytes) -> Segment:
    """Decode the 40-byte header at the start of ``payload``."""
    v, _, proto, chan, sess, length, count, offset, seq, send = _HEADER.unpack_from(payload)
    return Segment(v, proto, chan, sess, length, count, offset, seq, send)


@dataclass(frozen=True)
class IexStatus(StatusMessage):
    """A status item; ``detail`` is the halt reason of H, or the detail byte of P."""

    detail: str = ""


@dataclass(frozen=True)
class SecurityDirectory:
    """A security directory message; ``flags``: 0x80 test, 0x40 when issued, 0x20 ETP."""

    ts_ns: int
    symbol: str
    flags: int
    round_lot: int
    adjusted_poc_price: int
    luld_tier: int


@dataclass(frozen=True)
class IexTrade:
    """A fill.

    Attributes:
        ts_ns: nanoseconds since the epoch; fills with one timestamp are one event.
        symbol: stock symbol.
        source: message type, 'L' or 'T'.
        side: side of the resting displayed order; '' for 'T'.
        price: execution price in ticks of 0.0001.
        size: executed size.
        trade_id: IEX trade id, the key a :class:`TradeBreak` refers to.
        order_id: resting order id; 0 for 'T'.
        sale_flags: 0x80 ISO, 0x40 extended hours, 0x20 odd lot, 0x10 trade-through
            exempt, 0x08 single-price cross.
        booked_price: the order's resting price; 0 for 'T'.
    """

    ts_ns: int
    symbol: str
    source: str
    side: str
    price: int
    size: int
    trade_id: int
    order_id: int
    sale_flags: int
    booked_price: int = 0

    @property
    def last_sale_eligible(self) -> bool:
        """Neither extended hours nor odd lot."""
        return not self.sale_flags & 0x60


@dataclass(frozen=True)
class TradeBreak:
    """A same-day break of the fill with ``trade_id``."""

    ts_ns: int
    symbol: str
    price: int
    size: int
    trade_id: int
    sale_flags: int


@dataclass(frozen=True)
class LevelUpdate:
    """A DEEP price level update; size 0 removes the level.

    ``complete`` is the Event Flags bit: the book is consistent only after an
    update with it set.
    """

    ts_ns: int
    symbol: str
    side: str
    price: int
    size: int
    complete: bool


IexItem = Union[FeedEvent, IexTrade, TradeBreak, StatusMessage, SecurityDirectory, LevelUpdate]


class LevelBook:
    """One symbol's price levels from DEEP, with the BBO held through transitions.

    The specification says the BBO keeps its prior value until the update with
    Event Flags on, so :meth:`bbo` changes only then.  A completed book with bid
    at or above ask counts under ``crossed`` and does not raise: a lost update
    leaves stale levels behind, which is a data problem to report.

    Attributes:
        bids, asks: price -> aggregate size.
        counts: :class:`collections.Counter` of ``updates``, ``events``,
            ``crossed``, ``remove_missing`` (size 0 for an absent level).
    """

    def __init__(self) -> None:
        self.bids: Dict[int, int] = {}
        self.asks: Dict[int, int] = {}
        self.counts: Counter = Counter()
        self._bbo: Tuple[Optional[int], int, Optional[int], int] = (None, 0, None, 0)

    def apply(self, update: LevelUpdate) -> bool:
        """Apply one update; True when it completed an event."""
        self.counts["updates"] += 1
        levels = self.bids if update.side == "B" else self.asks
        if update.size:
            levels[update.price] = update.size
        elif levels.pop(update.price, None) is None:
            self.counts["remove_missing"] += 1
        if not update.complete:
            return False
        self.counts["events"] += 1
        bid = max(self.bids) if self.bids else None
        ask = min(self.asks) if self.asks else None
        self._bbo = (bid, self.bids.get(bid, 0), ask, self.asks.get(ask, 0))  # type: ignore[arg-type]
        if bid is not None and ask is not None and bid >= ask:
            self.counts["crossed"] += 1
        return True

    def bbo(self) -> Tuple[Optional[int], int, Optional[int], int]:
        """(bid, bid size, ask, ask size) as of the last completed event."""
        return self._bbo


def _i64(msg: bytes, offset: int) -> int:
    return _I64.unpack_from(msg, offset)[0]


def _u32(msg: bytes, offset: int) -> int:
    return _U32.unpack_from(msg, offset)[0]


def _code(msg: bytes) -> str:
    return msg[1:2].decode("ascii", "replace")


class IexParser:
    """Stateful IEX-TP, DEEP and DEEP+ parser; see the module docstring.

    Args:
        symbols: symbols to keep, or None for every symbol.
        strict: raise on an anomaly instead of only counting it.

    Attributes:
        stats: :class:`collections.Counter` of ``segments``, ``heartbeats``,
            ``streams``, ``messages``, ``ignored``, ``other_symbol``,
            ``gap_messages``, the anomaly names, and ``protocol_<hex>``.
    """

    def __init__(self, symbols: Optional[Iterable[str]] = None, strict: bool = True) -> None:
        self.symbols: Optional[Set[str]] = None if symbols is None else set(symbols)
        self.strict = bool(strict)
        self.stats: Counter = Counter()
        # (protocol, channel, session) -> [next sequence, next stream offset]
        self._streams: Dict[Tuple[int, int, int], List[int]] = {}
        # symbol -> order id -> [side, remaining size, booked price]
        self._orders: Dict[str, Dict[int, list]] = {}
        self._seq: Counter = Counter()
        self._last_ts: Dict[Tuple[bytes, str], int] = {}
        self._types = [0] * 256
        self._raw_symbols = self._padded(self.symbols)

    @staticmethod
    def _padded(symbols: Optional[Set[str]]) -> Optional[Set[bytes]]:
        """Symbols as they sit in a message; None if some have no such form."""
        if symbols is None or not all(s.isascii() and len(s) <= 8 and s == s.rstrip(" ") for s in symbols):
            return None
        return {s.ljust(8).encode("ascii") for s in symbols}

    def type_counts(self) -> Dict[str, int]:
        """Messages seen of each type byte, whatever their symbol."""
        return {chr(t): n for t, n in enumerate(self._types) if n}

    def _anomaly(self, name: str, detail: str) -> None:
        self.stats[name] += 1
        if self.strict:
            raise IexError(f"{name}: {detail}")

    def _event(self, symbol: str, ts_ns: int, etype: str, side: str, price: int, size: int,
               order_id: int, new_order_id: int = 0) -> FeedEvent:  # fmt: skip
        self._seq[symbol] += 1
        return FeedEvent(
            symbol,
            EventV2(self._seq[symbol], ts_ns, etype, side, price, size, order_id, new_order_id),
        )

    # ------------------------------------------------------------ transport

    def _blocks(self, seg: Segment, payload: bytes) -> List[bytes]:
        """Split the message blocks, keeping the sound prefix of a bad segment."""
        blocks: List[bytes] = []
        i, end = 40, 40 + seg.payload_length
        while i < end:
            length = int.from_bytes(payload[i : i + 2], "little")
            if i + 2 > end or i + 2 + length > end:
                self._anomaly("truncated", f"message of {length} bytes crosses the payload end")
                return blocks
            blocks.append(payload[i + 2 : i + 2 + length])
            i += 2 + length
        if len(blocks) != seg.message_count:
            self._anomaly("bad_count", f"{len(blocks)} blocks, header says {seg.message_count}")
        return blocks

    def _prefilter(self, seg: Segment, payload: bytes) -> Optional[List[bytes]]:
        """The messages of ``seg`` needing a full parse; None if it needs the full path.

        Counts the others.  Nothing is counted when returning None.
        """
        table, wanted = _TABLES[seg.protocol], self._raw_symbols
        kept: List[bytes] = []
        dropped: List[int] = []
        ignored = 0
        i, end = 40, 40 + seg.payload_length
        while i < end:
            if i + 2 > end:
                return None
            start = i + 2
            length = payload[i] | (payload[i + 1] << 8)
            i = start + length
            if i > end:
                return None
            mtype = payload[start] if length else -1
            expected = table[mtype] if length else 0
            if expected == -1:
                ignored += 1
                dropped.append(mtype)
            elif (
                expected == length and mtype != _SYSTEM_EVENT
                and payload[start + 10 : start + 18] not in wanted  # type: ignore[operator]
            ):  # fmt: skip
                dropped.append(mtype)
            else:
                kept.append(payload[start:i])
        if len(kept) + len(dropped) != seg.message_count:
            return None
        stats, types = self.stats, self._types
        stats["messages"] += len(dropped)
        stats["ignored"] += ignored
        stats["other_symbol"] += len(dropped) - ignored
        for mtype in dropped:
            types[mtype] += 1
        return kept

    def _sequence(self, seg: Segment) -> int:
        """Advance the stream; returns how many leading messages to skip, -1 for all."""
        key = (seg.protocol, seg.channel, seg.session)
        following = [seg.first_seq + seg.message_count, seg.stream_offset + seg.payload_length]
        stream = self._streams.get(key)
        if stream is None:
            self._streams[key] = following
            self.stats["streams"] += 1
            self.stats["late_start_messages"] += max(seg.first_seq - 1, 0)
            return 0
        next_seq, next_offset = stream
        if seg.first_seq < next_seq and following[0] <= next_seq:
            self.stats["duplicate_segment"] += 1
            return -1
        self._streams[key] = following
        if seg.first_seq < next_seq:
            self.stats["duplicate_messages"] += next_seq - seg.first_seq
            return next_seq - seg.first_seq
        if seg.first_seq > next_seq:
            self.stats["gap_messages"] += seg.first_seq - next_seq
            self._anomaly("seq_gap", f"stream {key}: expected {next_seq}, got {seg.first_seq}")
        elif seg.stream_offset != next_offset:
            self._anomaly("offset_mismatch", f"stream {key}: {seg.stream_offset}, not {next_offset}")
        return 0

    def parse_segment(self, payload: bytes) -> Iterator[IexItem]:
        """Yield the items of one UDP payload."""
        self.stats["segments"] += 1
        if len(payload) < 40:
            self._anomaly("bad_segment", f"{len(payload)} bytes is shorter than the header")
            return
        seg = parse_header(payload)
        if seg.version != 1:
            self._anomaly("bad_version", f"version {seg.version}")
            return
        if seg.protocol not in _LENGTHS:
            self._anomaly("unknown_protocol", f"protocol {seg.protocol:#06x}")
            return
        self.stats[f"protocol_{seg.protocol:04x}"] += 1
        if len(payload) < 40 + seg.payload_length:
            self._anomaly("bad_length", f"payload {len(payload) - 40}, header {seg.payload_length}")
            return
        skip = self._sequence(seg)
        if skip < 0:
            return
        if seg.message_count == 0 and seg.payload_length == 0:
            self.stats["heartbeats"] += 1
            return
        kept = self._prefilter(seg, payload) if self._raw_symbols is not None and not skip else None
        if kept is None:
            kept = self._blocks(seg, payload)[skip:]
        for msg in kept:
            yield from self.parse_message(msg, seg.protocol)

    # ------------------------------------------------------------ messages

    def parse_message(self, msg: bytes, protocol: int = DEEP_PLUS_PROTOCOL) -> Iterator[IexItem]:
        """Yield the items of one message of ``protocol``."""
        self.stats["messages"] += 1
        mtype = msg[:1]
        if mtype:
            self._types[mtype[0]] += 1
        if mtype in _IGNORED[protocol]:
            self.stats["ignored"] += 1
            return
        expected = _LENGTHS[protocol].get(mtype)
        if expected is None:
            self._anomaly("unknown_type", f"message type {mtype!r} on {protocol:#06x}")
            return
        if len(msg) < expected:
            self._anomaly("bad_length", f"type {mtype!r} has {len(msg)} bytes, not {expected}")
            return
        if len(msg) > expected:
            self.stats["grown_message"] += 1
        ts_ns = _i64(msg, 2)
        if mtype == b"S":
            yield IexStatus(ts_ns, "S", "", _code(msg))
            return
        symbol = msg[10:18].decode("ascii", "replace").rstrip(" ")
        if self.symbols is not None and symbol not in self.symbols:
            self.stats["other_symbol"] += 1
            return
        key = (mtype, symbol)
        last = self._last_ts.get(key, ts_ns)
        self._last_ts[key] = max(ts_ns, last)
        if ts_ns < last:
            self._anomaly("ts_regression", f"{key}: {ts_ns} after {last}")
        yield from self._dispatch(mtype, msg, symbol, ts_ns)

    def _dispatch(self, mtype: bytes, msg: bytes, symbol: str, ts_ns: int) -> Iterator[IexItem]:
        if mtype == b"D":
            yield SecurityDirectory(ts_ns, symbol, msg[1], _u32(msg, 18), _i64(msg, 22), msg[30])
        elif mtype == b"H":
            yield IexStatus(ts_ns, "H", symbol, _code(msg), msg[18:22].decode("ascii", "replace").rstrip())
        elif mtype == b"P":
            yield IexStatus(ts_ns, "P", symbol, str(msg[1]), str(msg[18]))
        elif mtype in (b"O", b"E"):
            yield IexStatus(ts_ns, mtype.decode("ascii"), symbol, _code(msg))
        elif mtype == b"T":
            yield IexTrade(ts_ns, symbol, "T", "", _i64(msg, 22), _u32(msg, 18), _i64(msg, 30), 0, msg[1])
        elif mtype == b"B":
            yield TradeBreak(ts_ns, symbol, _i64(msg, 22), _u32(msg, 18), _i64(msg, 30), msg[1])
        elif mtype in (b"8", b"5"):
            yield from self._on_level(msg, symbol, ts_ns)
        elif mtype == b"a":
            yield from self._on_add(msg, symbol, ts_ns)
        elif mtype == b"M":
            yield from self._on_modify(msg, symbol, ts_ns)
        elif mtype == b"R":
            yield from self._on_delete(msg, symbol, ts_ns)
        elif mtype == b"L":
            yield from self._on_executed(msg, symbol, ts_ns)
        else:  # b"C"
            for ref, order in self._orders.pop(symbol, {}).items():
                yield self._event(symbol, ts_ns, "C", order[0], 0, 0, ref)

    def _on_level(self, msg: bytes, symbol: str, ts_ns: int) -> Iterator[IexItem]:
        size, price = _u32(msg, 18), _i64(msg, 22)
        if price < 1:
            self._anomaly("invalid_order", f"level price {price}")
            return
        yield LevelUpdate(ts_ns, symbol, _SIDES[msg[0]], price, size, bool(msg[1] & 1))

    def _on_add(self, msg: bytes, symbol: str, ts_ns: int) -> Iterator[IexItem]:
        side = _SIDES.get(msg[1])
        ref, size, price = _i64(msg, 18), _u32(msg, 26), _i64(msg, 30)
        if side is None or ref < 1 or size < 1 or price < 1:
            self._anomaly("invalid_order", f"side {msg[1]}, ref {ref}, size {size}, price {price}")
            return
        orders = self._orders.setdefault(symbol, {})
        if ref in orders:
            self._anomaly("duplicate_order", f"ref {ref} is already live")
            return
        orders[ref] = [side, size, price]
        yield self._event(symbol, ts_ns, "L", side, price, size, ref)

    def _live(self, symbol: str, ref: int) -> Optional[list]:
        order = self._orders.get(symbol, {}).get(ref)
        if order is None:
            self._anomaly("unknown_order", f"{symbol} ref {ref}")
        return order

    def _on_modify(self, msg: bytes, symbol: str, ts_ns: int) -> Iterator[IexItem]:
        ref, size, price = _i64(msg, 18), _u32(msg, 26), _i64(msg, 30)
        order = self._live(symbol, ref)
        if order is None:
            return
        if size < 1 or price < 1:
            self._anomaly("invalid_order", f"ref {ref}, size {size}, price {price}")
            return
        side, old_size, old_price = order
        order[1], order[2] = size, price
        if msg[1] & 1:  # keep priority
            if price == old_price and size == old_size:
                self.stats["modify_noop"] += 1
                return
            if price == old_price and size < old_size:
                yield self._event(symbol, ts_ns, "R", side, 0, old_size - size, ref)
                return
            self.stats["modify_priority_lost"] += 1
        yield self._event(symbol, ts_ns, "U", side, price, size, ref, ref)

    def _on_delete(self, msg: bytes, symbol: str, ts_ns: int) -> Iterator[IexItem]:
        ref = _i64(msg, 18)
        order = self._live(symbol, ref)
        if order is not None:
            del self._orders[symbol][ref]
            yield self._event(symbol, ts_ns, "C", order[0], 0, 0, ref)

    def _on_executed(self, msg: bytes, symbol: str, ts_ns: int) -> Iterator[IexItem]:
        ref, size, price = _i64(msg, 18), _u32(msg, 26), _i64(msg, 30)
        order = self._live(symbol, ref)
        if order is None:
            return
        if size < 1:
            self._anomaly("invalid_order", f"ref {ref}: size {size}")
            return
        if size > order[1]:
            self._anomaly("oversize", f"ref {ref}: {size} above remaining {order[1]}")
            size = order[1]
        order[1] -= size
        if order[1] == 0:
            del self._orders[symbol][ref]
        yield self._event(symbol, ts_ns, "E", order[0], 0, size, ref)
        yield IexTrade(ts_ns, symbol, "L", order[0], price, size, _i64(msg, 38), ref, msg[1], order[2])


def iter_file(path: str, symbols: Optional[Iterable[str]] = None, strict: bool = True,
              parser: Optional[IexParser] = None,
              progress: Optional[Callable[[Counter, Counter], None]] = None,
              progress_seconds: float = 60.0) -> Iterator[IexItem]:  # fmt: skip
    """Stream the items of a HIST capture, ``.pcap`` or ``.pcapng``, gzipped or not.

    Pass ``parser`` to read its ``stats`` afterwards; capture counters are added
    with the prefix ``pcap_``, and it then decides strictness and symbols.
    Datagrams that are not IEX-TP count as ``bad_segment`` or ``bad_version``.
    ``progress(capture stats, parser stats)`` is called about every
    ``progress_seconds``, whether or not any item has been yielded.
    """
    parser = parser if parser is not None else IexParser(symbols, strict)
    with open_capture(path) as fh:
        reader = CaptureReader(fh, parser.strict)
        due = monotonic() + progress_seconds
        count = 0
        for datagram in reader:
            yield from parser.parse_segment(datagram.payload)
            count += 1
            if progress is not None and not count & 0xFFFF and monotonic() >= due:
                due = monotonic() + progress_seconds
                progress(reader.stats, parser.stats)
        parser.stats.update({"pcap_" + k: v for k, v in reader.stats.items()})
