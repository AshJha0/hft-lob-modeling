"""Zero-intelligence order-flow generator (pinned, seeded).

Flow loosely inspired by Smith et al. (2003), not their model: Poisson event-type rates around the best quotes with
geometric placement of passive limit orders.  Everything is pinned — RNG
(:class:`lobmod.rng.Lcg64`), draw order, placement and size formulas — so that
every language regenerates the identical 5000-event stream.

Pinned algorithm
----------------
Seed phase (events 1..12): for ``i = 0..5`` submit a buy limit at ``999 - i``
size 10 then a sell limit at ``1001 + i`` size 10 (ids = seq numbers).

Stochastic phase (events 13..n).  With ``nb, ns`` the live order counts per
side, the event-type rates are::

    limit buy 1.0 | limit sell 1.0 | cancel buy 0.04*nb | cancel sell 0.04*ns
    market buy 0.25 | market sell 0.25

Draw ``u1 = uniform()`` and pick the type by cumulative probability in the
order above.  Then:

* limit: ``u2 = uniform()``; if ``u2 < 0.12`` and the opposite best exists the
  order is *aggressive* — priced AT the opposite best (a crossed limit that
  executes on arrival).  Otherwise passive: ``k = geometric(0.35)`` and price
  ``opp_best -/+ (1 + k)`` (or ``1000 -/+ (1 + k)`` when the opposite side is
  empty), clamped to >= 1 tick.  Size ``1 + randint(5)``; id = seq.
* cancel: uniform pick from the side's live order ids sorted ascending:
  ``ids[randint(len(ids))]``.  (Rate is 0 when the side is empty.)
* market: ``u3 = uniform()``; size ``40 + randint(40)`` if ``u3 < 0.02``
  (an occasional sweep that walks several levels) else ``1 + randint(3)``.

Every event is applied to an internal book immediately so that quotes and
live-order sets evolve exactly as a replay of the emitted stream would.
"""

from __future__ import annotations

from typing import List

from ._validate import as_int
from .book import LimitOrderBook
from .replay import Event, apply_event
from .rng import Lcg64

#: pinned generator constants
_RATE_LIMIT = 1.0
_RATE_MARKET = 0.25
_RATE_CANCEL_PER_ORDER = 0.04
_P_AGGRESSIVE = 0.12
_P_GEOMETRIC = 0.35
_P_SWEEP = 0.02
_REF_PRICE = 1000


def generate_events(n_events: int, seed: int) -> List[Event]:
    """Generate a deterministic zero-intelligence event stream.

    Args:
        n_events: number of events to emit (>= 12; the first 12 seed the book).
        seed: LCG seed.

    Returns:
        List of :class:`~lobmod.replay.Event`, seq numbered 1..n_events.

    Raises:
        ValueError: if ``n_events < 12`` or the seed is invalid.
    """
    n_events = as_int("n_events", n_events)
    if n_events < 12:
        raise ValueError(f"n_events must be an integer >= 12, got {n_events}")
    rng = Lcg64(seed)
    book = LimitOrderBook(tick_size=1.0, record_trades=False)
    events: List[Event] = []

    def emit(etype: str, side: str, price: int, size: int, order_id: int) -> None:
        event = Event(len(events) + 1, etype, side, price, size, order_id)
        apply_event(book, event)
        events.append(event)

    # -- seed phase: 6 levels each side, size 10
    for i in range(6):
        emit("L", "B", _REF_PRICE - 1 - i, 10, len(events) + 1)
        emit("L", "S", _REF_PRICE + 1 + i, 10, len(events) + 1)

    # -- stochastic phase
    while len(events) < n_events:
        seq = len(events) + 1
        nb = book.n_orders("B")
        ns = book.n_orders("S")
        rates = [
            _RATE_LIMIT,                        # limit buy
            _RATE_LIMIT,                        # limit sell
            _RATE_CANCEL_PER_ORDER * nb,        # cancel buy
            _RATE_CANCEL_PER_ORDER * ns,        # cancel sell
            _RATE_MARKET,                       # market buy
            _RATE_MARKET,                       # market sell
        ]
        total = 0.0  # explicit left-to-right: builtin sum() is compensated on 3.12+
        for rate in rates:
            total += rate
        x = rng.uniform() * total
        kind = 0
        acc = rates[0]
        while x >= acc and kind < 5:
            kind += 1
            acc += rates[kind]

        if kind in (0, 1):  # limit order
            buy = kind == 0
            opp_best = book.best_ask() if buy else book.best_bid()
            u2 = rng.uniform()
            if u2 < _P_AGGRESSIVE and opp_best is not None:
                price = opp_best
            else:
                k = rng.geometric(_P_GEOMETRIC)
                anchor = opp_best if opp_best is not None else _REF_PRICE
                price = anchor - (1 + k) if buy else anchor + (1 + k)
            price = max(price, 1)
            size = 1 + rng.randint(5)
            emit("L", "B" if buy else "S", price, size, seq)
        elif kind in (2, 3):  # cancel
            side = "B" if kind == 2 else "S"
            ids = book.order_ids(side)
            oid = ids[rng.randint(len(ids))]
            emit("C", side, 0, 0, oid)
        else:  # market order
            side = "B" if kind == 4 else "S"
            u3 = rng.uniform()
            size = 40 + rng.randint(40) if u3 < _P_SWEEP else 1 + rng.randint(3)
            emit("M", side, 0, size, 0)
    return events
