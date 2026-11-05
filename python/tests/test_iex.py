"""IEX HIST reader against hand-built bytes; no real capture has been used."""

import gzip
import importlib.util
import io
import json
import struct
from pathlib import Path

import pytest

from lobmod.feeds.iex import (
    DEEP_PLUS_PROTOCOL, DEEP_PROTOCOL, BookIntegrityChecker, FeedEvent, IexError, IexParser,
    IexStatus, IexTrade, LevelBook, LevelUpdate, SecurityDirectory, TradeBreak, iter_file,
    parse_header,
)  # fmt: skip
from lobmod.feeds.pcap import CaptureReader, PcapError

TS = 0x146D86ABA0A58FB2  # 2016-08-23 15:32:04.912754610 in the specification's examples


def sym(name="ZIEXT"):
    return name.ljust(8).encode("ascii")


def add(ref, side=b"8", size=100, price=990500, name="ZIEXT", ts=TS):
    return b"a" + side + struct.pack("<q", ts) + sym(name) + struct.pack("<qIq", ref, size, price)


def modify(ref, size, price, keep=0):
    return b"M" + bytes([keep]) + struct.pack("<q", TS) + sym() + struct.pack("<qIq", ref, size, price)


def delete(ref):
    return b"R\0" + struct.pack("<q", TS) + sym() + struct.pack("<q", ref)


def executed(ref, size, price=990500, ts=TS):
    return b"L\0" + struct.pack("<q", ts) + sym() + struct.pack("<qIqq", ref, size, price, 7)


def trade(mtype=b"T", flags=0):
    return mtype + bytes([flags]) + struct.pack("<q", TS) + sym() + struct.pack("<Iqq", 100, 990500, 9)


def level(side, size, price, complete=1):
    return side + bytes([complete]) + struct.pack("<q", TS) + sym() + struct.pack("<Iq", size, price)


def segment(msgs, seq=1, offset=0, protocol=DEEP_PLUS_PROTOCOL, channel=1, session=5, count=None):
    body = b"".join(struct.pack("<H", len(m)) + m for m in msgs)
    n = len(msgs) if count is None else count
    return struct.pack("<BBHIIHHqqq", 1, 0, protocol, channel, session, len(body), n, offset, seq, TS) + body


def udp_frame(payload, vlan=False):
    udp = struct.pack(">HHHH", 1000, 10378, 8 + len(payload), 0) + payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0x4000, 64, 17, 0, b"\1\1\1\1", b"\2\2\2\2")
    tag = b"\x81\0\0\1" if vlan else b""
    return b"\0" * 12 + tag + b"\x08\0" + ip + udp


def pcap(frames, magic=0xA1B2C3D4):
    out = struct.pack("<IHHiIII", magic, 2, 4, 0, 0, 65535, 1)
    for f in frames:
        out += struct.pack("<IIII", 100, 0, len(f), len(f)) + f
    return out


def block(btype, body):
    body += b"\0" * (-len(body) % 4)
    total = 12 + len(body)
    return struct.pack("<II", btype, total) + body + struct.pack("<I", total)


def pcapng(frames):
    shb = block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    opts = struct.pack("<HHB3x", 9, 1, 9) + struct.pack("<HH", 0, 0)
    out = shb + block(1, struct.pack("<HHI", 1, 0, 65535) + opts) + block(99, b"skip")
    for f in frames:
        out += block(6, struct.pack("<IIIII", 0, 1, 5, len(f), len(f)) + f)
    return out


def run(msgs, **kw):
    return list(IexParser(**kw).parse_segment(segment(msgs)))


def events(items):
    return [i.event for i in items if isinstance(i, FeedEvent)]


def test_header_layout_from_the_transport_specification():
    seg = parse_header(segment([b"xx"], seq=3, offset=8, channel=2, session=6))
    assert (seg.version, seg.protocol, seg.channel, seg.session) == (1, 0x8005, 2, 6)
    assert (seg.payload_length, seg.message_count, seg.stream_offset, seg.first_seq) == (4, 1, 8, 3)


def test_specification_example_add_order_bytes():
    raw = bytes.fromhex("6138b28fa5a0ab866d14" "5a49455854202020" "968f060000000000" "64000000" "241d0f0000000000")
    ev = events(run([raw]))[0]
    assert (ev.etype, ev.side, ev.order_id, ev.size, ev.price) == ("L", "B", 429974, 100, 990500)
    assert ev.ts_ns == TS


def test_order_lifecycle_maps_to_events():
    items = run([add(1), add(2, b"5", price=990600), executed(1, 40, price=990550), delete(2)])
    assert [e.etype for e in events(items)] == ["L", "L", "E", "C"]
    fill = next(i for i in items if isinstance(i, IexTrade))
    assert (fill.side, fill.price, fill.booked_price, fill.size) == ("B", 990550, 990500, 40)
    assert fill.last_sale_eligible
    check = BookIntegrityChecker()
    for ev in events(items):
        check.apply(ev)
    assert check.verify() == (1, 0)
    assert check.book.best_bid_size() == 60


def test_modify_respects_the_priority_flag():
    p = IexParser()
    items = list(p.parse_segment(segment([
        add(1), modify(1, 60, 990500, keep=1), modify(1, 60, 990500, keep=1),
        modify(1, 80, 990500, keep=1), modify(1, 80, 990400),
    ])))  # fmt: skip
    ev = events(items)
    assert [e.etype for e in ev] == ["L", "R", "U", "U"]
    assert ev[1].size == 40
    assert ev[3].new_order_id == 1 and ev[3].price == 990400
    assert p.stats["modify_noop"] == 1 and p.stats["modify_priority_lost"] == 1


def test_clear_book_cancels_every_order_and_allows_id_reuse():
    clear = b"C\0" + struct.pack("<q", TS) + sym()
    assert [e.etype for e in events(run([add(1), add(2), clear, add(1)]))] == ["L", "L", "C", "C", "L"]


def test_deep_level_book_holds_bbo_through_transition():
    # the worked example in the DEEP specification
    msgs = [level(b"5", 100, 253000), level(b"5", 100, 252000), level(b"5", 100, 251000),
            level(b"8", 100, 250000), level(b"8", 100, 249000)]  # fmt: skip
    book = LevelBook()
    for item in IexParser().parse_segment(segment(msgs, protocol=DEEP_PROTOCOL)):
        book.apply(item)
    assert book.bbo() == (250000, 100, 251000, 100)
    book.apply(LevelUpdate(TS, "ZIEXT", "S", 251000, 0, False))
    assert book.bbo()[2] == 251000
    book.apply(LevelUpdate(TS, "ZIEXT", "S", 252000, 0, True))
    assert book.bbo()[2] == 253000
    assert book.counts["crossed"] == 0


def test_deep_and_deep_plus_type_tables_are_separate():
    p = IexParser(strict=False)
    list(p.parse_segment(segment([level(b"8", 1, 1)])))
    assert p.stats["unknown_type"] == 1


def test_admin_messages():
    directory = b"D\x80" + struct.pack("<q", TS) + sym() + struct.pack("<IqB", 100, 990500, 1)
    status = b"HH" + struct.pack("<q", TS) + sym() + b"T1  "
    system = b"SE" + struct.pack("<q", TS)
    price_test = b"P\x01" + struct.pack("<q", TS) + sym() + b"A"
    d, h, s, _ = run([directory, status, system, price_test])
    assert isinstance(d, SecurityDirectory) and d.round_lot == 100 and d.luld_tier == 1 and d.flags == 0x80
    assert isinstance(h, IexStatus) and (h.code, h.detail) == ("H", "T1")
    assert (s.msg_type, s.code, s.symbol) == ("S", "E", "")


def test_trade_and_break():
    t, b = run([trade(flags=0x40), trade(b"B")])
    assert isinstance(t, IexTrade) and t.side == "" and not t.last_sale_eligible
    assert isinstance(b, TradeBreak) and b.trade_id == 9


def test_gap_is_detected_and_counted():
    first, later = segment([add(1)], seq=1), segment([add(2)], seq=4, offset=40)
    strict = IexParser()
    list(strict.parse_segment(first))
    with pytest.raises(IexError, match="seq_gap"):
        list(strict.parse_segment(later))
    lenient = IexParser(strict=False)
    list(lenient.parse_segment(first))
    assert len(list(lenient.parse_segment(later))) == 1
    assert lenient.stats["seq_gap"] == 1 and lenient.stats["gap_messages"] == 2


def test_heartbeat_reveals_a_gap():
    p = IexParser(strict=False)
    list(p.parse_segment(segment([add(1)])))
    list(p.parse_segment(segment([], seq=9, offset=999)))
    assert p.stats["seq_gap"] == 1 and p.stats["heartbeats"] == 1


def test_duplicates_are_dropped_and_overlap_is_trimmed():
    p = IexParser()
    assert len(list(p.parse_segment(segment([add(1), add(2)])))) == 2
    assert list(p.parse_segment(segment([add(1)]))) == []
    items = list(p.parse_segment(segment([add(2), add(3)], seq=2, offset=40)))
    assert [e.order_id for e in events(items)] == [3]
    assert p.stats["duplicate_segment"] == 1 and p.stats["duplicate_messages"] == 1


def test_channels_are_separate_streams():
    p = IexParser()
    list(p.parse_segment(segment([add(1)], channel=1)))
    list(p.parse_segment(segment([add(2)], channel=2, seq=50)))
    assert p.stats["streams"] == 2 and p.stats["late_start_messages"] == 49


def test_offset_mismatch():
    p = IexParser(strict=False)
    list(p.parse_segment(segment([add(1)])))
    list(p.parse_segment(segment([add(2)], seq=2, offset=7)))
    assert p.stats["offset_mismatch"] == 1


@pytest.mark.parametrize("payload, name", [
    (b"short", "bad_segment"),
    (b"\x02" + segment([add(1)])[1:], "bad_version"),
    (segment([add(1)], protocol=0x8003), "unknown_protocol"),
    (segment([add(1)])[:-1], "bad_length"),
    (segment([add(1)], count=2), "bad_count"),
    (segment([add(1)[:10]]), "bad_length"),
    (segment([b"z" + b"\0" * 20]), "unknown_type"),
])  # fmt: skip
def test_segment_anomalies_strict_and_lenient(payload, name):
    with pytest.raises(IexError, match=name):
        list(IexParser().parse_segment(payload))
    p = IexParser(strict=False)
    list(p.parse_segment(payload))
    assert p.stats[name] == 1


def test_block_crossing_the_payload_end():
    raw = bytearray(segment([add(1)]))
    raw[40:42] = struct.pack("<H", 500)
    p = IexParser(strict=False)
    assert list(p.parse_segment(bytes(raw))) == []
    assert p.stats["truncated"] == 1


@pytest.mark.parametrize("msgs, name", [
    ([delete(1)], "unknown_order"),
    ([add(1), add(1)], "duplicate_order"),
    ([add(1, size=0)], "invalid_order"),
    ([add(1, side=b"X")], "invalid_order"),
    ([add(1), executed(1, 101)], "oversize"),
    ([add(1, ts=TS), add(2, ts=TS - 1)], "ts_regression"),
])  # fmt: skip
def test_message_anomalies(msgs, name):
    with pytest.raises(IexError, match=name):
        run(msgs)
    p = IexParser(strict=False)
    list(p.parse_segment(segment(msgs)))
    assert p.stats[name] == 1


def test_oversize_is_clamped_when_lenient():
    assert events(run([add(1), executed(1, 500)], strict=False))[1].size == 100


def test_grown_message_is_tolerated():
    p = IexParser()
    assert len(list(p.parse_segment(segment([add(1) + b"more"])))) == 1
    assert p.stats["grown_message"] == 1


def test_symbol_filter_never_looks_like_a_gap():
    p = IexParser(["ZIEXT"])
    assert len(list(p.parse_segment(segment([add(1, name="OTHER"), add(1)])))) == 1
    assert p.stats["other_symbol"] == 1


def _mixed_stream():
    """Wanted and other symbols through every path of the byte-level walk."""
    def other(m):
        return m.replace(sym(), sym("OTHER"))

    retail = b"I\0" + struct.pack("<q", TS) + sym("OTHER") + b"\0" * 8
    system = b"SO" + struct.pack("<q", TS)
    return [
        segment([system, add(1), other(add(1)), retail], seq=1),
        segment([other(add(2)) + b"grown", add(2, b"5", price=990600), b"z" + b"\0" * 20], seq=5),
        segment([], seq=8),
        segment([other(executed(1, 10)), executed(1, 40), other(delete(1))], seq=8),
        segment([add(8), add(9)], seq=10),
        segment([add(9), other(add(3)), modify(2, 50, 990600, keep=1)], seq=11),  # overlap trimmed
        segment([other(add(5))[:12], delete(2)], seq=14),  # short message of another symbol
        segment([other(add(6))], seq=17),  # a gap
        segment([add(7)], seq=18, count=2),  # a wrong count
    ]


def _feed(symbols):
    p = IexParser(symbols, strict=False)
    return p, [i for seg in _mixed_stream() for i in p.parse_segment(seg)]


def test_filtered_and_unfiltered_runs_agree_for_wanted_symbols():
    filtered, kept = _feed(["ZIEXT"])
    plain, everything = _feed(None)
    assert kept == [i for i in everything if getattr(i, "symbol", "") in ("ZIEXT", "")] and events(kept)
    assert filtered.type_counts() == plain.type_counts()
    for name in ("segments", "messages", "ignored", "seq_gap", "gap_messages", "bad_count", "offset_mismatch",
                 "bad_length", "unknown_type", "heartbeats", "grown_message", "duplicate_messages"):  # fmt: skip
        assert filtered.stats[name] == plain.stats[name], name
    assert filtered.stats["seq_gap"] and filtered.stats["bad_count"] and filtered.stats["grown_message"]


def test_byte_level_walk_matches_the_plain_filter_exactly():
    fast, kept = _feed(["ZIEXT"])
    slow = IexParser(["ZIEXT"], strict=False)
    slow._raw_symbols = None
    assert [i for seg in _mixed_stream() for i in slow.parse_segment(seg)] == kept
    assert slow.stats == fast.stats and slow.type_counts() == fast.type_counts()


def test_symbol_prefix_is_not_a_match():
    p = IexParser(["ZIE"])
    assert list(p.parse_segment(segment([add(1)]))) == []
    assert p.stats["other_symbol"] == 1


def test_capture_crosses_the_read_buffer_and_gzip_members(tmp_path, monkeypatch):
    monkeypatch.setattr("lobmod.feeds.pcap._CHUNK", 64)
    monkeypatch.setattr("lobmod.feeds.pcap._RAW_CHUNK", 16)
    frames = [udp_frame(segment([add(i + 1)], seq=i + 1, offset=len(segment([add(1)])[40:]) * i)) for i in range(200)]
    path = tmp_path / "x.pcap.gz"
    raw = pcapng(frames)
    path.write_bytes(gzip.compress(raw[:999]) + gzip.compress(raw[999:]))
    parser = IexParser()
    assert len(events(iter_file(str(path), parser=parser))) == 200
    assert parser.stats["pcap_packets"] == 200


def test_progress_is_called_without_items(tmp_path):
    one = pcap([udp_frame(segment([], seq=1))])
    path = tmp_path / "x.pcap"
    path.write_bytes(one[:24] + one[24:] * 0x10000)
    seen = []
    report = lambda capture, feed: seen.append(capture["packets"])  # noqa: E731
    list(iter_file(str(path), strict=False, progress=report, progress_seconds=0))
    assert seen == [0x10000]


@pytest.mark.parametrize("builder", [pcap, pcapng])
@pytest.mark.parametrize("compress", [False, True])
def test_file_round_trip(tmp_path, builder, compress):
    raw = builder([udp_frame(segment([add(1)])), udp_frame(segment([delete(1)], seq=2, offset=40), vlan=True)])
    path = tmp_path / "20251212_IEXTP1_DPLS1.0.pcap.gz"
    path.write_bytes(gzip.compress(raw) if compress else raw)
    parser = IexParser()
    assert [e.etype for e in events(iter_file(str(path), parser=parser))] == ["L", "C"]
    assert parser.stats["pcap_datagrams"] == 2


def test_timestamps_and_padding():
    frame = udp_frame(b"abc") + b"\0" * 9  # Ethernet padding must not reach the payload
    (dg,) = CaptureReader(io.BytesIO(pcapng([frame])))
    assert dg.payload == b"abc" and dg.dst_port == 10378
    assert dg.ts_ns == (1 << 32) + 5
    (dg,) = CaptureReader(io.BytesIO(pcap([frame], magic=0xA1B23C4D)))
    assert dg.ts_ns == 100 * 10**9


def test_truncated_capture_and_gzip(tmp_path):
    raw = pcap([udp_frame(b"abc")])
    with pytest.raises(PcapError, match="truncated"):
        list(CaptureReader(io.BytesIO(raw[:-2])))
    reader = CaptureReader(io.BytesIO(raw[:-2]), strict=False)
    assert list(reader) == [] and reader.stats["truncated"] == 1
    path = tmp_path / "cut.pcap.gz"
    path.write_bytes(gzip.compress(raw)[:-12])
    parser = IexParser(strict=False)
    list(iter_file(str(path), parser=parser))
    assert parser.stats["pcap_truncated"] == 1


def test_non_udp_and_fragments_are_skipped():
    frame = udp_frame(b"abc")
    tcp = frame[:23] + b"\x06" + frame[24:]
    frag = frame[:20] + b"\x20\0" + frame[22:]
    reader = CaptureReader(io.BytesIO(pcap([tcp, frag])))
    assert list(reader) == []
    assert reader.stats["not_udp"] == 1 and reader.stats["ip_fragment"] == 1


def _replay_module():
    path = Path(__file__).resolve().parents[2] / "research" / "iex_replay.py"
    spec = importlib.util.spec_from_file_location("iex_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replay_report_cites_iex(tmp_path, capsys):
    system = b"SR" + struct.pack("<q", TS - 10)
    msgs = [system, add(1), add(2, b"5", price=990600, ts=TS + 1), executed(1, 40, ts=TS + 2)]
    src, out = tmp_path / "x.pcap", tmp_path / "o.json"
    src.write_bytes(pcap([udp_frame(segment(msgs))]))
    _replay_module().main(str(src), str(out), ["ZIEXT"])
    res = json.loads(out.read_text())
    assert res["attribution"].startswith("Data provided for free by IEX.")
    assert res["terms"] == "https://www.iex.io/products/market-data-connectivity/hist-terms"
    summary = res["symbols"]["ZIEXT"]
    assert summary["verify"] == "ok" and summary["touch"]["at_touch"] == 1
    capsys.readouterr()


def test_aggregate_merges_one_event_only():
    assert _replay_module().aggregate([(1, 1), (1, 1), (1, -1), (2, -1), (2, -1)]) == [1, -1, -1]
