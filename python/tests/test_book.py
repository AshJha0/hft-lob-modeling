"""Order book mechanics: priority, fills, edge cases, replay invariants."""

import math

import numpy as np
import pytest

import lobmod as lm
from lobmod.book import LimitOrderBook
from lobmod.replay import apply_event


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
