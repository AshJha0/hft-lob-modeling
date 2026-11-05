"""Order book mechanics: priority, fills, edge cases, replay invariants."""

import math
import os

import numpy as np
import pytest

import lobmod as lm
from lobmod.book import LimitOrderBook
from lobmod.replay import apply_event

from conftest import DATA_DIR


def make_book():
    b = LimitOrderBook(tick_size=1.0)
    b.limit("B", 99, 5, 1)
    b.limit("B", 98, 7, 2)
    b.limit("S", 101, 4, 3)
    b.limit("S", 102, 6, 4)
    return b


def test_best_quotes_and_mid():
    b = make_book()
    assert b.best_bid() == 99
    assert b.best_ask() == 101
    assert b.mid() == 100.0
    assert b.spread() == 2.0
    assert b.best_bid_size() == 5
    assert b.best_ask_size() == 4


def test_microprice_leans_to_thin_side():
    b = make_book()
    # qb=5, qa=4 -> micro = (4*99 + 5*101)/9
    assert b.microprice() == pytest.approx((4 * 99 + 5 * 101) / 9.0, abs=1e-12)
    assert b.microprice() > b.mid()  # more depth on bid pushes micro up


def test_fifo_time_priority_within_level():
    b = LimitOrderBook()
    b.limit("B", 100, 3, 1)
    b.limit("B", 100, 4, 2)  # same price, later arrival
    fills, unfilled = b.market("S", 5)
    assert unfilled == 0
    assert [(f.maker_id, f.size) for f in fills] == [(1, 3), (2, 2)]  # id 1 first


def test_price_priority_across_levels():
    b = make_book()
    fills, _ = b.market("B", 6)
    assert [(f.price, f.size) for f in fills] == [(101, 4), (102, 2)]


def test_market_walks_multiple_levels_partial_front_order():
    b = LimitOrderBook()
    for i, oid in enumerate((1, 2, 3)):
        b.limit("S", 101 + i, 5, oid)
    fills, unfilled = b.market("B", 12)
    assert unfilled == 0
    assert [(f.price, f.size) for f in fills] == [(101, 5), (102, 5), (103, 2)]
    assert b.best_ask() == 103 and b.best_ask_size() == 3


def test_market_bigger_than_book_reports_unfilled():
    b = make_book()
    fills, unfilled = b.market("B", 25)  # total ask depth = 10
    assert sum(f.size for f in fills) == 10
    assert unfilled == 15
    assert b.best_ask() is None
    assert math.isnan(b.mid()) and math.isnan(b.microprice())  # empty side guarded


def test_crossed_limit_converts_to_execution():
    b = make_book()
    fills, rested = b.limit("B", 101, 6, 10)  # crosses best ask (4 @ 101)
    assert [(f.price, f.size) for f in fills] == [(101, 4)]
    assert rested == 2
    assert b.best_bid() == 101  # remainder rests at its limit
    assert b.best_ask() == 102


def test_crossed_limit_respects_its_own_price():
    b = make_book()
    fills, rested = b.limit("B", 101, 20, 10)  # must not lift 102s
    assert sum(f.size for f in fills) == 4
    assert all(f.price <= 101 for f in fills)
    assert rested == 16


def test_cancel_unknown_id_raises():
    b = make_book()
    with pytest.raises(ValueError, match="unknown order_id"):
        b.cancel(999)
    b.market("B", 4)  # fully fills order 3
    with pytest.raises(ValueError, match="unknown order_id"):
        b.cancel(3)  # filled orders are gone


def test_cancel_removes_size():
    b = make_book()
    assert b.cancel(1) == 5
    assert b.best_bid() == 98


def test_tick_size_validation():
    b = LimitOrderBook(tick_size=0.5)
    b.limit("B", 99.5, 1, 1)  # on grid
    with pytest.raises(ValueError, match="tick_size"):
        b.limit("B", 99.3, 1, 2)
    with pytest.raises(ValueError):
        LimitOrderBook(tick_size=0.0)
    with pytest.raises(ValueError):
        LimitOrderBook(tick_size=math.nan)


def test_input_validation():
    b = make_book()
    with pytest.raises(ValueError, match="side"):
        b.limit("X", 100, 1, 10)
    with pytest.raises(ValueError, match="size"):
        b.limit("B", 100, 0, 10)
    with pytest.raises(ValueError, match="size"):
        b.market("B", -3)
    with pytest.raises(ValueError, match="price"):
        b.limit("B", -5, 1, 10)
    with pytest.raises(ValueError, match="already live"):
        b.limit("B", 90, 1, 1)
    with pytest.raises(ValueError, match="k must be"):
        b.depth("B", 0)


def test_depth_top_k():
    b = make_book()
    assert b.depth("B", 1) == 5
    assert b.depth("B", 2) == 12
    assert b.depth("B", 10) == 12  # fewer levels than k
    assert b.depth("S", 2) == 10


def test_replay_invariants_and_size_conservation(events):
    """Full 5000-event replay: bid < ask whenever both exist, and size added
    minus size cancelled minus size executed equals the resting size."""
    book = LimitOrderBook()
    added = cancelled = maker_executed = limit_taker_executed = 0
    for e in events:
        pre_trades = len(book.trades)
        if e.etype == "L":
            book.limit(e.side, float(e.price), e.size, e.order_id)
            added += e.size
        elif e.etype == "C":
            cancelled += book.cancel(e.order_id)
        else:
            book.market(e.side, e.size)
        vol = sum(f.size for f in book.trades[pre_trades:])
        maker_executed += vol  # every fill removes resting (maker) size
        if e.etype == "L":
            limit_taker_executed += vol  # the crossed part of a limit never rests
        b, a = book.best_bid(), book.best_ask()
        if b is not None and a is not None:
            assert b < a
    resting = book.total_size("B") + book.total_size("S")
    # limit size either rests, was cancelled, executed as maker later, or
    # executed immediately as taker (marketable limits); market-order taker
    # size never enters the book at all
    assert added - cancelled - maker_executed - limit_taker_executed == resting


def test_replay_matches_recorded_states(events, states):
    assert len(states) == len(events)
    finite = np.isfinite(states.best_bid) & np.isfinite(states.best_ask)
    assert np.all(states.best_bid[finite] < states.best_ask[finite])
    # property-style: microprice always between best bid and ask where defined
    m = np.isfinite(states.microprice)
    assert np.all(states.microprice[m] >= states.best_bid[m])
    assert np.all(states.microprice[m] <= states.best_ask[m])


def test_event_stream_edge_content(events):
    """The bundled stream exercises sweeps and marketable limits."""
    assert sum(1 for e in events if e.etype == "M" and e.size >= 40) >= 3
    book = LimitOrderBook()
    marketable = 0
    for e in events:
        n = len(book.trades)
        apply_event(book, e)
        if e.etype == "L" and len(book.trades) > n:
            marketable += 1
    assert marketable > 50


def test_apply_event_unknown_type():
    with pytest.raises(ValueError, match="unknown event type"):
        apply_event(LimitOrderBook(), lm.Event(1, "Z", "B", 100, 1, 1))


def test_empty_replay_raises():
    with pytest.raises(ValueError, match="empty"):
        lm.replay([])


# ----------------------------------------------------------- review additions

def test_price_rounding_to_zero_ticks_rejected():
    """PT-1: a positive price below half a tick must not rest at tick 0."""
    b = LimitOrderBook(1.0)
    with pytest.raises(ValueError, match="0 ticks"):
        b.limit("B", 1e-12, 5, 1)
    b2 = LimitOrderBook(0.01)
    with pytest.raises(ValueError, match="0 ticks"):
        b2.limit("S", 4e-12, 1, 2)
    for book in (b, b2):
        assert book.best_bid() is None and book.best_ask() is None
        assert book.n_orders("B") == 0 and book.n_orders("S") == 0
        assert book.log == []
    with pytest.raises(ValueError, match=">= 1 tick"):
        b.limit_ticks("B", 0, 5, 1)


def test_large_tick_counts():
    """PT-2: fine-tick products need 64-bit tick counts, and overflow errors."""
    b = LimitOrderBook(1e-8)
    fills, rested = b.limit("S", 60000.0, 1, 1)
    assert (fills, rested) == ([], 1)
    assert b.best_ask() == 6_000_000_000_000
    fills, rested = b.limit("B", 60000.0, 1, 2)
    assert [(f.maker_id, f.price, f.size) for f in fills] == [(1, 6_000_000_000_000, 1)]
    assert rested == 0 and b.best_ask() is None and b.best_bid() is None
    with pytest.raises(ValueError, match="MAX_TICKS"):
        b.limit("B", 1e300, 1, 3)
    assert b.best_bid() is None  # never became a sweep or a resting order
    with pytest.raises(ValueError, match="MAX_TICKS"):
        b.limit_ticks("B", lm.MAX_TICKS + 1, 1, 3)
    b.limit_ticks("B", lm.MAX_TICKS, 1, 3)
    assert b.best_bid() == lm.MAX_TICKS


def test_tick_rounding_rule_is_half_up_within_tolerance():
    """Pinned conversion: round half up, residual within 1e-9 * max(1, ratio)."""
    b = LimitOrderBook(0.5)
    b.limit("B", 99.5, 1, 1)
    assert b.best_bid() == 199
    b.limit("B", 100.0000000001, 1, 2)  # 200.0000000002 ticks, residual 2e-10 <= 1e-9*200
    assert b.best_bid() == 200
    with pytest.raises(ValueError, match="not a multiple"):
        b.limit("B", 100.004, 1, 3)
    with pytest.raises(ValueError, match="not a multiple"):
        b.limit("B", 0.75, 1, 3)  # exactly 1.5 ticks: off-grid, never rounded
    with pytest.raises(ValueError, match="price"):
        b.limit("B", True, 1, 3)


def test_limit_ticks_matches_limit():
    a, c = LimitOrderBook(0.25), LimitOrderBook(0.25)
    a.limit("S", 25.5, 4, 1)
    c.limit_ticks("S", 102, 4, 1)
    assert a.best_ask() == c.best_ask() == 102
    fa, ra = a.limit("B", 26.0, 6, 2)
    fc, rc = c.limit_ticks("B", 104, 6, 2)
    assert fa == fc and ra == rc == 2
    with pytest.raises(ValueError, match="order_id"):
        c.limit_ticks("B", 100, 1, 2.5)


def test_replay_prices_are_tick_counts(events):
    """PT-4: Event.price is in ticks; replay never rescales it."""
    st = lm.replay(events[:12])
    assert st.best_bid[11] == 999.0 and st.best_ask[11] == 1001.0
    manual = LimitOrderBook(tick_size=0.5)
    for e in events[:12]:
        manual.limit("B" if e.side == "B" else "S", e.price * 0.5, e.size, e.order_id)
    assert manual.best_bid() == 999  # a 0.5-tick book fed price*tick lands on the same ticks
    with pytest.raises(TypeError):
        lm.replay(events[:12], tick_size=0.5)  # the parameter no longer exists
    with pytest.raises(ValueError, match="depth_k"):
        lm.replay(events[:12], depth_k=0)


def test_depth_monotone_in_k_over_replay(events):
    """PT-11: depth(side, k) is non-decreasing in k; depth(side, 1) is the best size."""
    book = LimitOrderBook()
    for e in events[:500]:
        apply_event(book, e)
        for side, best_size in (("B", book.best_bid_size()), ("S", book.best_ask_size())):
            assert book.depth(side, 1) == best_size
            for k in range(1, 6):
                assert book.depth(side, k) <= book.depth(side, k + 1)


def test_market_on_empty_book_and_reuse_of_cancelled_id():
    """PT-12."""
    b = LimitOrderBook()
    assert b.market("B", 5) == ([], 5)
    assert b.market("S", 3) == ([], 3)
    b.limit("B", 100, 2, 7)
    assert b.cancel(7) == 2
    fills, rested = b.limit("S", 101, 4, 7)  # same id, now free again
    assert fills == [] and rested == 4 and b.has_order(7)


def test_crossed_sell_limit_stops_at_its_price_and_price_improvement():
    """PT-13."""
    b = LimitOrderBook()
    b.limit("B", 100, 4, 1)
    b.limit("B", 99, 4, 2)
    fills, rested = b.limit("S", 100, 10, 3)
    assert [(f.maker_id, f.price, f.size) for f in fills] == [(1, 100, 4)]
    assert rested == 6 and b.best_ask() == 100 and b.best_bid() == 99
    assert b.best_bid_size() == 4  # the 99 bid was never touched
    fills, rested = b.limit("S", 95, 2, 4)  # aggressive: executes at the resting 99
    assert [(f.price, f.size) for f in fills] == [(99, 2)] and rested == 0


def test_record_trades_flag_and_clear_history():
    b = LimitOrderBook(record_trades=False)
    b.limit("S", 101, 5, 1)
    fills, _ = b.market("B", 2)
    assert len(fills) == 1 and b.trades == [] and b.log == []
    assert b.best_ask_size() == 3  # matching unaffected
    c = LimitOrderBook()
    c.limit("S", 101, 5, 1)
    c.market("B", 2)
    assert len(c.trades) == 1 and len(c.log) == 2
    c.clear_history()
    assert c.trades == [] and c.log == [] and c.best_ask_size() == 3


def test_numpy_integer_inputs_accepted():
    """PT-16: numpy integer scalars are integers too (bools never are)."""
    b = LimitOrderBook()
    b.limit("B", 100.0, np.int64(5), np.int64(1))
    b.limit("B", np.float64(99.0), np.int32(2), np.int32(2))
    assert b.depth("B", np.int64(3)) == 7
    assert b.cancel(np.int64(2)) == 2
    assert b.market("S", np.int64(1))[1] == 0
    assert lm.ks_pvalue(0.1, np.int64(50)) == lm.ks_pvalue(0.1, 50)
    tr, te = lm.chronological_split(np.int64(10), 0.7)
    assert tr.size == 7 and te.size == 3
    assert lm.Lcg64(np.int64(7)).next_u64() == lm.Lcg64(7).next_u64()
    assert lm.generate_events(np.int64(12), np.int64(3))[0].price == 999
    with pytest.raises(ValueError, match="size"):
        b.limit("B", 100.0, True, 9)
    with pytest.raises(ValueError, match="size"):
        b.limit("B", 100.0, 2.0, 9)


def test_seed_domain_is_pinned():
    lm.Lcg64(0)
    lm.Lcg64(lm.MAX_SEED)
    with pytest.raises(ValueError, match="seed"):
        lm.Lcg64(lm.MAX_SEED + 1)  # never silently masked
    with pytest.raises(ValueError, match="seed"):
        lm.Lcg64(-1)
    with pytest.raises(ValueError, match="seed"):
        lm.Lcg64(1.5)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_bytes(text.encode())
    return str(p)


def test_events_csv_robustness(tmp_path):
    """PT-15: the pinned CSV rules."""
    body = "seq,type,side,price,size,order_id\n1,L,B,999,10,1\n2,L,S,1001,10,2\n"
    ok = lm.read_events_csv(_write(tmp_path, "ok.csv", body))
    assert ok == [lm.Event(1, "L", "B", 999, 10, 1), lm.Event(2, "L", "S", 1001, 10, 2)]
    crlf = body.replace("\n", "\r\n")
    assert lm.read_events_csv(_write(tmp_path, "crlf.csv", crlf)) == ok
    assert lm.read_events_csv(_write(tmp_path, "trail.csv", body + "\n\n")) == ok
    assert lm.read_events_csv(_write(tmp_path, "noeol.csv", body.rstrip("\n"))) == ok
    bad = {
        "comma.csv": body + "3,M,B,0,2,0,\n",              # trailing comma: 7 fields
        "float.csv": body + "3,L,B,999.5,1,3\n",           # non-integer field
        "space.csv": body + "3, L,B,999,1,3\n",            # whitespace inside a field
        "plus.csv": body + "3,L,B,+999,1,3\n",             # sign other than '-'
        "type.csv": body + "3,X,B,999,1,3\n",              # unknown event type
        "side.csv": body + "3,L,Q,999,1,3\n",              # unknown side
        "blank.csv": "seq,type,side,price,size,order_id\n1,L,B,999,10,1\n\n2,L,S,1001,10,2\n",
        "header.csv": "seq,side,type,price,size,order_id\n1,B,L,999,10,1\n",
        "empty.csv": "",
        "big.csv": body + "3,L,B,9223372036854775808,1,3\n",  # 2^63: out of range
    }
    for name, text in bad.items():
        with pytest.raises(ValueError):
            lm.read_events_csv(_write(tmp_path, name, text))
    # unknown type / side are rejected at parse time even when structurally valid
    with pytest.raises(ValueError, match="event type"):
        lm.read_events_csv(_write(tmp_path, "t2.csv", body + "3,Z,B,999,1,3\n"))


def test_events_csv_round_trip(tmp_path, events):
    path = str(tmp_path / "stream.csv")
    lm.write_events_csv(path, events)
    assert lm.read_events_csv(path) == events
    with open(path, "rb") as fh, open(os.path.join(DATA_DIR, "lob_events.csv"), "rb") as ref:
        assert fh.read() == ref.read()  # byte-identical to the bundled file


# ---- 64-bit edge rules pinned in API_SPEC 2.1/2.3 --------------------------


def _quoted(bid, ask):
    b = LimitOrderBook()
    b.limit_ticks("B", bid, 1, 1)
    b.limit_ticks("S", ask, 1, 2)
    return b


def test_mid_is_exact_sum_rounded_once():
    assert _quoted(5 * 10**18, 5 * 10**18 + 1).mid() == 5e18
    assert _quoted(2**53 + 1, 2**53 + 2).mid() == float(2**53 + 2)
    assert _quoted(2**62, 2**62 + 1025).mid() == float(2**62 + 1024)


def test_side_capacity_rejected_before_mutation():
    b = LimitOrderBook()
    b.limit_ticks("B", 100, lm.MAX_TICKS, 1)
    with pytest.raises(ValueError, match="MAX_TICKS"):
        b.limit_ticks("B", 100, lm.MAX_TICKS, 2)
    with pytest.raises(ValueError, match="MAX_TICKS"):
        b.limit_ticks("B", 99, 1, 2)
    assert b.total_size("B") == lm.MAX_TICKS and b.n_orders("B") == 1
    assert b.best_bid_size() == lm.MAX_TICKS and b.depth("B", 5) == lm.MAX_TICKS
    b.market("S", 1)  # a fill frees capacity
    b.limit_ticks("B", 99, 1, 2)
    assert b.total_size("B") == lm.MAX_TICKS
    b.cancel(1)  # so does a cancel
    assert b.total_size("B") == 1
    b.limit_ticks("S", 200, lm.MAX_TICKS, 3)
    with pytest.raises(ValueError, match="MAX_TICKS"):
        b.limit_ticks("S", 200, 1, 4)
    assert b.total_size("S") == lm.MAX_TICKS


def test_grid_tolerance_is_capped_below_half_a_tick():
    b = LimitOrderBook(0.01)
    with pytest.raises(ValueError, match="not a multiple"):
        b.limit("B", 5000000.004, 1, 1)
    b.limit("B", 5000000.01, 1, 1)
    assert b.best_bid() == 500000001


def test_grid_validation_stops_at_2_pow_52():
    b = LimitOrderBook(1.0)
    with pytest.raises(ValueError, match="limit_ticks"):
        b.limit("B", float(2**52), 1, 1)
    b.limit("B", float(2**52 - 1), 1, 1)
    assert b.best_bid() == 2**52 - 1


def test_infinite_ratio_is_an_error():
    with pytest.raises(ValueError):
        LimitOrderBook(1e-300).limit("B", 1e308, 1, 1)


def test_geometric_out_of_range_is_an_error():
    with pytest.raises(ValueError, match="64-bit"):
        lm.Lcg64(1).geometric(1e-300)


def test_rosenbrock_4d_iteration_count_is_pinned():
    def rosen(x):
        total = 0.0
        for i in range(len(x) - 1):
            total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        return total

    assert lm.nelder_mead(rosen, [-1.2, 1.0, 0.7, 1.3]).n_iter == 308


def test_csv_strips_exactly_one_carriage_return(tmp_path):
    from lobmod.replay import read_csv_lines

    p = tmp_path / "x.csv"
    p.write_bytes(b"t\r\n1\r\n")
    assert read_csv_lines(str(p), "t") == ["1"]
    p.write_bytes(b"t\r\r\n1\r\n")
    with pytest.raises(ValueError, match="header"):
        read_csv_lines(str(p), "t")
    p.write_bytes(b"t\n1\r\r\n")
    assert read_csv_lines(str(p), "t") == ["1\r"]  # field parsers then reject it
