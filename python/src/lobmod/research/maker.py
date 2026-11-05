r"""Passive maker simulator: one lot each side, injected into a replayed book.

Python-only analysis code, outside the four-language golden contract.

The plain idea: rest one lot at the best bid and one at the best ask, behind
everyone already there, and earn the spread when market flow reaches the
order.  After each event the simulator checks for fills, then requotes any
order no longer at the touch, waiting ``delay`` events after a cancel.

Measures, for a fill with sign ``s`` (+1 we bought) at price ``p``:

    quoted edge        s (mid_before - p)
    realised spread    s (mid_{i+h} - p)     at each horizon h
    run PnL            cash + inventory * final mid

The gap between quoted edge and realised spread is adverse selection: how far
the mid moved against the fill.  ``pnl_unwound`` closes the final inventory at
the touch instead of the mid, and charges one more fee per lot.

Documented limits; each one flatters the result:

* The flow does not react.  The stream was generated without us, so nobody
  fades our quotes, and no informed trader exists at all in ZI flow.
* No competitors.  No one else requotes or races us for queue position;
  a real touch would be contested.
* Requotes are instant apart from ``delay``, and joining the queue always
  succeeds.
* Our fills divert market flow from background orders, so this book drifts
  from the original.  A background cancel of an order that no longer exists
  here is skipped and counted in ``n_skipped``.
* Results are in ticks on synthetic zero-intelligence flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..book import LimitOrderBook
from ..replay import Event, apply_event

_OWN_ID_BASE = 10 ** 12


@dataclass(frozen=True)
class MakerConfig:
    """Maker settings.

    Attributes:
        delay: events to wait after a cancel before quoting again.
        inventory_cap: stop quoting a side once ``|inventory|`` reaches this.
        fee_per_side: fee in ticks per fill (negative would be a rebate).
        warmup: events before the first quote.
        horizons: realised-spread horizons in events.
    """

    delay: int = 0
    inventory_cap: int = 5
    fee_per_side: float = 0.0
    warmup: int = 100
    horizons: Tuple[int, ...] = (1, 10, 50, 200)

    def __post_init__(self) -> None:
        if self.delay < 0 or self.inventory_cap < 1 or self.warmup < 0:
            raise ValueError(f"invalid maker config: {self}")


@dataclass(frozen=True)
class MakerFill:
    """One fill.  ``sign`` is +1 for a buy; ``mid_before`` may be NaN."""

    event_index: int
    sign: int
    price: float
    mid_before: float
    inventory_after: int


@dataclass(frozen=True)
class MakerResult:
    """Run outcome.

    Attributes:
        fills: the fill log.
        pnl: cash plus inventory at the final mid, after fees.
        pnl_unwound: as ``pnl``, closing inventory at the touch with a fee.
        inventory: final inventory.
        quoted_edge: mean edge against the pre-fill mid.
        realised_spread: horizon -> mean realised spread, using only fills
            whose horizon lies inside the stream with a defined mid.
        n_skipped: background cancels skipped (module docstring).
    """

    fills: List[MakerFill]
    pnl: float
    pnl_unwound: float
    inventory: int
    quoted_edge: float
    realised_spread: Dict[int, float]
    n_skipped: int


def _mean(values: List[float]) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else math.nan


def run_maker(events: Sequence[Event], config: MakerConfig = MakerConfig()) -> MakerResult:
    """Replay ``events`` with the maker's quotes in the book.

    Raises:
        ValueError: a background event other than a cancel fails, or the book
            never has two sides.
    """
    book = LimitOrderBook(1.0, record_trades=False)
    n = len(events)
    mids = np.full(n, np.nan)
    live: Dict[str, Tuple[int, int]] = {}
    fills: List[MakerFill] = []
    next_id = _OWN_ID_BASE
    inventory = 0
    cash = 0.0
    skipped = 0
    next_ok = 0
    last_quotes = None

    def collect(i: int) -> None:
        nonlocal inventory, cash
        for side in list(live):
            oid, px = live[side]
            if book.has_order(oid):
                continue
            sign = 1 if side == "B" else -1
            inventory += sign
            cash += -sign * px - config.fee_per_side
            before = float(mids[i - 1]) if i > 0 else math.nan
            fills.append(MakerFill(i, sign, float(px), before, inventory))
            del live[side]

    for i, event in enumerate(events):
        try:
            apply_event(book, event)
        except ValueError:
            if event.etype != "C":
                raise
            skipped += 1
        collect(i)
        bid, ask = book.best_bid(), book.best_ask()
        if bid is None or ask is None:
            continue
        last_quotes = (bid, ask)
        mids[i] = 0.5 * (bid + ask)
        if i < config.warmup or i < next_ok:
            continue
        for side, px, allowed in (
            ("B", bid, inventory < config.inventory_cap),
            ("S", ask, inventory > -config.inventory_cap),
        ):
            current = live.get(side)
            if current is not None and (current[1] != px or not allowed):
                book.cancel(current[0])
                del live[side]
                current = None
                next_ok = i + config.delay
            if current is None and allowed and i >= next_ok:
                next_id += 1
                live[side] = (next_id, px)
                book.limit_ticks(side, px, 1, next_id)
                collect(i)
    if last_quotes is None:
        raise ValueError("book never had both sides")
    bid, ask = last_quotes
    pnl = cash + inventory * 0.5 * (bid + ask)
    close = bid if inventory > 0 else ask
    unwound = cash + inventory * close - abs(inventory) * config.fee_per_side
    return MakerResult(
        fills=fills, pnl=pnl, pnl_unwound=unwound, inventory=inventory,
        quoted_edge=_mean([f.sign * (f.mid_before - f.price) for f in fills]),
        realised_spread={
            h: _mean([f.sign * (mids[f.event_index + h] - f.price)
                      for f in fills if f.event_index + h < n])
            for h in config.horizons
        },
        n_skipped=skipped,
    )
