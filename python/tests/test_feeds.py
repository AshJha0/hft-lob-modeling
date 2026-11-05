"""ITCH 5.0 parser and book-integrity checker, on hand-built byte fixtures only."""

import io
import struct

import pytest

from lobmod.feeds import (
    BookIntegrityChecker,
    FeedEvent,
    Itch50Parser,
    ItchError,
    StatusMessage,
    TradeReport,
    events_for_symbol,
    iter_frames,
)
from lobmod.replay import EventV2, read_events_v2_csv, write_events_v2_csv


def head(mtype, locate=1, ts=1000):
    return mtype + struct.pack(">HH", locate, 0) + ts.to_bytes(6, "big")


def stock(name):
    return name.ljust(8).encode("ascii")


def sys_event(code=b"O", ts=1000):
    return head(b"S", 0, ts) + code


def directory(name="AAPL", locate=1):
    return head(b"R", locate) + stock(name) + b"Q" + b"N" + struct.pack(">I", 100) + b"N" + b"C" + b"Z " + b"P" + b"N" + b" " + b"1" + b"N" + struct.pack(">I", 0) + b"N"


def action(name="AAPL", state=b"T"):
    return head(b"H") + stock(name) + state + b" " + b"    "


def add(ref, side, size, price, name="AAPL", locate=1, ts=1000):
    return head(b"A", locate, ts) + struct.pack(">QcI", ref, side, size) + stock(name) + struct.pack(">I", price)


def add_mpid(ref, side, size, price):
    return head(b"F") + struct.pack(">QcI", ref, side, size) + stock("AAPL") + struct.pack(">I", price) + b"MPID"


def executed(ref, size, match=7, locate=1, ts=1000):
    return head(b"E", locate, ts) + struct.pack(">QIQ", ref, size, match)


def executed_price(ref, size, price, printable=b"Y", match=8):
    return head(b"C") + struct.pack(">QIQ", ref, size, match) + printable + struct.pack(">I", price)


def cancel(ref, size):
    return head(b"X") + struct.pack(">QI", ref, size)


def delete(ref):
    return head(b"D") + struct.pack(">Q", ref)


def replace(ref, new_ref, size, price):
    return head(b"U") + struct.pack(">QQII", ref, new_ref, size, price)


def hidden(size, price, name="AAPL", side=b"B", match=9):
    return head(b"P") + struct.pack(">QcI", 0, side, size) + stock(name) + struct.pack(">IQ", price, match)


def cross(size, price, name="AAPL", match=10):
    return head(b"Q") + struct.pack(">Q", size) + stock(name) + struct.pack(">IQ", price, match) + b"O"


def framed(*messages):
    return io.BytesIO(b"".join(struct.pack(">H", len(m)) + m for m in messages))


def parse(*messages, **kwargs):
    parser = Itch50Parser(**kwargs)
    return parser, list(parser.parse(framed(*messages)))


def events(items):
    return [i.event for i in items if isinstance(i, FeedEvent)]


def test_message_lengths_match_the_specification():
    built = {
        b"S": sys_event(), b"R": directory(), b"H": action(), b"A": add(1, b"B", 1, 1),
        b"F": add_mpid(1, b"B", 1, 1), b"E": executed(1, 1), b"C": executed_price(1, 1, 1),
        b"X": cancel(1, 1), b"D": delete(1), b"U": replace(1, 2, 1, 1), b"P": hidden(1, 1),
        b"Q": cross(1, 1),
    }  # fmt: skip
    lengths = {k: len(v) for k, v in built.items()}
    assert lengths == {
        b"S": 12, b"R": 39, b"H": 25, b"A": 36, b"F": 40, b"E": 31, b"C": 36,
        b"X": 23, b"D": 19, b"U": 35, b"P": 44, b"Q": 40,
    }  # fmt: skip


def test_framing_and_six_byte_timestamp():
    ts = 0x0123456789AB  # needs all six bytes
    _, items = parse(sys_event(b"Q", ts), add(5, b"S", 100, 1234500, ts=ts))
    assert items[0] == StatusMessage(ts, "S", "", "Q")
    assert events(items) == [EventV2(1, ts, "L", "S", 1234500, 100, 5, 0)]


def test_frames_stream_from_a_reader_without_loading_the_file():
    class OneByte(io.RawIOBase):
        def __init__(self, data):
            self.data = io.BytesIO(data)
            self.largest = 0

        def read(self, n=-1):
            self.largest = max(self.largest, n)
            return self.data.read(n)

    raw = OneByte(framed(*[sys_event() for _ in range(50)]).getvalue())
    assert len(list(iter_frames(raw))) == 50
    assert raw.largest == 12


def test_full_order_lifecycle_maps_to_v2_events():
    _, items = parse(
        directory(), action(),
        add(1, b"B", 100, 5000), add_mpid(2, b"S", 50, 5100),
        executed(1, 30), executed_price(1, 20, 4999, b"N"),
        cancel(1, 10), replace(1, 3, 40, 5001), delete(3), executed(2, 50),
    )  # fmt: skip
    assert items[0] == StatusMessage(1000, "R", "AAPL", "Q")
    assert items[1] == StatusMessage(1000, "H", "AAPL", "T")
    got = [(e.etype, e.side, e.price, e.size, e.order_id, e.new_order_id) for e in events(items)]
    assert got == [
        ("L", "B", 5000, 100, 1, 0),
        ("L", "S", 5100, 50, 2, 0),
        ("E", "B", 0, 30, 1, 0),
        ("E", "B", 0, 20, 1, 0),
        ("R", "B", 0, 10, 1, 0),
        ("U", "B", 5001, 40, 1, 3),
        ("C", "B", 0, 0, 3, 0),
        ("E", "S", 0, 50, 2, 0),
    ]
    assert [e.seq for e in events(items)] == list(range(1, 9))
    trades = [i for i in items if isinstance(i, TradeReport)]
    assert [(t.source, t.price, t.size, t.printable) for t in trades] == [
        ("E", 0, 30, True), ("C", 4999, 20, False), ("E", 0, 50, True),
    ]  # fmt: skip
    checker = BookIntegrityChecker()
    for e in events(items):
        checker.apply(e)
    assert checker.verify() == (0, 0)
    assert checker.counts["events"] == 8 and checker.counts["E"] == 3


def test_hidden_and_cross_trades_never_touch_the_book():
    _, items = parse(add(1, b"B", 10, 5000), hidden(70, 5050), cross(1 << 33, 5060))
    assert len(events(items)) == 1
    p, q = [i for i in items if isinstance(i, TradeReport)]
    assert (p.source, p.side, p.price, p.size, p.match_number) == ("P", "B", 5050, 70, 9)
    assert (q.source, q.side, q.price, q.size, q.match_number) == ("Q", "", 5060, 1 << 33, 10)


def test_symbol_filter_uses_order_state_and_locate():
    parser, items = parse(
        directory("AAPL", 1), directory("MSFT", 2),
        add(1, b"B", 10, 100, "AAPL", 1), add(2, b"B", 10, 200, "MSFT", 2),
        executed(2, 5, locate=2), executed(1, 5, locate=1),
        hidden(5, 200, "MSFT"), cross(5, 200, "MSFT"),
        symbols=["AAPL"],
    )  # fmt: skip
    assert {e.order_id for e in events(items)} == {1}
    assert len(events(items)) == 2
    assert parser.stats["other_symbol"] == 5
    assert parser.stats["unknown_order"] == 0


def test_filter_still_reports_a_gap_in_a_wanted_symbol():
    msgs = (directory("AAPL", 1), executed(99, 5, locate=1))
    with pytest.raises(ItchError, match="unknown_order"):
        parse(*msgs, symbols=["AAPL"])


def test_multiple_symbols_are_numbered_separately():
    _, items = parse(add(1, b"B", 1, 100, "AAPL", 1), add(2, b"B", 1, 100, "MSFT", 2))
    assert [(i.symbol, i.event.seq) for i in items] == [("AAPL", 1), ("MSFT", 1)]


def test_events_for_symbol_round_trips_through_v2_csv(tmp_path):
    stream = framed(add(1, b"B", 10, 100), add(2, b"B", 3, 200, "MSFT", 2), cancel(1, 4))
    got = list(events_for_symbol(stream, "AAPL"))
    path = str(tmp_path / "itch.csv")
    write_events_v2_csv(path, got)
    assert read_events_v2_csv(path) == got
    assert [e.etype for e in got] == ["L", "R"]


@pytest.mark.parametrize(
    "messages, name",
    [
        ((b"~" + bytes(10),), "unknown_type"),
        ((sys_event() + b"x",), "bad_length"),
        ((delete(5),), "unknown_order"),
        ((executed(5, 1),), "unknown_order"),
        ((cancel(5, 1),), "unknown_order"),
        ((replace(5, 6, 1, 1),), "unknown_order"),
        ((add(1, b"B", 1, 100), add(1, b"B", 1, 100)), "duplicate_order"),
        ((add(1, b"B", 1, 100), add(2, b"B", 1, 100), replace(1, 2, 1, 100)), "duplicate_order"),
        ((add(1, b"B", 0, 100),), "invalid_order"),
        ((add(1, b"B", 1, 0),), "invalid_order"),
        ((add(1, b"X", 1, 100),), "invalid_order"),
        ((add(1 << 63, b"B", 1, 100),), "invalid_order"),
        ((add(1, b"B", 5, 100), cancel(1, 0)), "invalid_order"),
        ((add(1, b"B", 5, 100, ts=50), add(2, b"B", 5, 100, ts=49)), "ts_regression"),
    ],
)
def test_strict_raises_and_lenient_counts(messages, name):
    with pytest.raises(ItchError, match=name):
        parse(*messages)
    parser, _ = parse(*messages, strict=False)
    assert parser.stats[name] == 1


def test_oversize_is_clamped_when_lenient():
    msgs = (add(1, b"B", 5, 100), executed(1, 9))
    with pytest.raises(ItchError, match="oversize"):
        parse(*msgs)
    parser, items = parse(*msgs, strict=False)
    assert parser.stats["oversize"] == 1
    assert events(items)[-1].size == 5
    checker = BookIntegrityChecker()
    for e in events(items):
        checker.apply(e)
    assert checker.verify() == (0, 0)


def test_truncated_final_frame():
    whole = framed(sys_event()).getvalue()
    for cut in (whole + b"\x00", whole + struct.pack(">H", 12) + b"S"):
        with pytest.raises(ItchError, match="truncated"):
            list(Itch50Parser().parse(io.BytesIO(cut)))
        parser = Itch50Parser(strict=False)
        assert len(list(parser.parse(io.BytesIO(cut)))) == 1
        assert parser.stats["truncated"] == 1


def test_defined_but_unused_types_are_skipped_even_when_strict():
    parser, items = parse(b"I" + bytes(49), b"B" + bytes(18))
    assert items == []
    assert parser.stats["ignored"] == 2


def e(seq, etype, side, price, size, oid, new=0):
    return EventV2(seq, seq, etype, side, price, size, oid, new)


def test_checker_refuses_a_crossing_add_when_strict():
    checker = BookIntegrityChecker()
    checker.apply(e(1, "L", "S", 100, 5, 1))
    with pytest.raises(ItchError, match="crossed"):
        checker.apply(e(2, "L", "B", 100, 5, 2))  # locked counts too
    assert checker.book.total_size("S") == 5


def test_checker_holds_crossed_orders_out_when_lenient():
    checker = BookIntegrityChecker(strict=False)
    stream = [
        e(1, "L", "S", 100, 5, 1),
        e(2, "L", "B", 101, 5, 2),  # crossed: held
        e(3, "E", "B", 0, 2, 2),  # skipped
        e(4, "U", "B", 99, 3, 2, 3),  # no longer crossing: joins the book
        e(5, "U", "S", 99, 5, 1, 4),  # crossing replace: old order leaves
        e(6, "C", "S", 0, 0, 4),  # skipped
        e(7, "R", "B", 0, 9, 3),  # book rejects: counted
    ]
    for event in stream:
        checker.apply(event)
    assert checker.counts["crossed"] == 2
    assert checker.counts["skipped"] == 3
    assert checker.counts["book_errors"] == 1
    assert checker.book.total_size("S") == 0
    assert checker.book.total_size("B") == 3
    assert checker.verify() == (1, 0)


def test_verify_detects_damage():
    checker = BookIntegrityChecker()
    checker.apply(e(1, "L", "B", 100, 5, 1))
    checker.book._bids[100][0][1] = -1
    with pytest.raises(ItchError):
        checker.verify()
    checker.book._bids[100][0][1] = 6
    with pytest.raises(ItchError, match="totals"):
        checker.verify()
