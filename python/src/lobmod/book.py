"""Limit order book with price-time (FIFO) priority on a discrete tick grid.

Model
-----
* Prices live on a grid of integer multiples of ``tick_size``; internally a
  price is stored as an integer number of ticks.
* Each price level holds a FIFO queue of resting orders ``(order_id, size)``;
  earlier arrivals at a level fill first (time priority), better prices fill
  first (price priority).
* Three event types:
    - **limit add**: rests at its price; if it crosses the opposite best it
      first executes as a marketable order against all crossing levels, and
      only the unfilled remainder (if any) rests.
    - **cancel by id**: removes an order entirely; unknown id is an error.
    - **market order**: walks the opposite side level by level, filling FIFO
      within each level; if the book runs out, the unfilled remainder is
      *reported*, never an error.
* Derived quantities: best bid/ask, mid, spread, microprice
  ``(q_a * P_b + q_b * P_a) / (q_a + q_b)`` with best-level sizes ``q_b, q_a``
  (size-weighted mid tilted toward the thin side), and summed depth over the
  top ``k`` price levels per side.

All quantities that need both sides of the book return NaN when a side is
empty (guarded, never a crash).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Fill:
    """One execution against a resting order.

    Attributes:
        maker_id: id of the resting (passive) order that was hit.
        price: execution price in *ticks* (the resting order's price).
        size: executed quantity (integer, > 0).
    """

    maker_id: int
    price: int
    size: int


class LimitOrderBook:
    """Price-time priority limit order book (see module docstring).

    Args:
        tick_size: positive tick; every incoming price must be an integer
            multiple of it (validated to 1e-9 relative precision).

    Raises:
        ValueError: on invalid tick size or invalid event parameters.
    """

    #: sides
    BUY = "B"
    SELL = "S"

    def __init__(self, tick_size: float = 1.0) -> None:
        if not (math.isfinite(tick_size) and tick_size > 0.0):
            raise ValueError(f"tick_size must be positive and finite, got {tick_size}")
        self.tick_size: float = float(tick_size)
        # price level (int ticks) -> FIFO queue of [order_id, remaining size]
        self._bids: Dict[int, Deque[List[int]]] = {}
        self._asks: Dict[int, Deque[List[int]]] = {}
        # order_id -> (side, price ticks)
        self._orders: Dict[int, Tuple[str, int]] = {}
        #: executed trades as Fill records, in execution order
        self.trades: List[Fill] = []
        #: event log: (event_type, side, price_ticks, size, order_id, executed, unfilled)
        self.log: List[Tuple[str, str, int, int, int, int, int]] = []

    # ------------------------------------------------------------------ utils

    def _to_ticks(self, price: float) -> int:
        """Validate that ``price`` sits on the tick grid and convert to ticks."""
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError(f"price must be positive and finite, got {price}")
        ratio = price / self.tick_size
        ticks = round(ratio)
        if abs(ratio - ticks) > 1e-9 * max(1.0, abs(ratio)):
            raise ValueError(
                f"price {price} is not a multiple of tick_size {self.tick_size}"
            )
        return int(ticks)

    @staticmethod
    def _check_side(side: str) -> str:
        if side not in (LimitOrderBook.BUY, LimitOrderBook.SELL):
            raise ValueError(f"side must be 'B' or 'S', got {side!r}")
        return side

    @staticmethod
    def _check_size(size: int) -> int:
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"size must be a positive integer, got {size}")
        return size

    # ------------------------------------------------------------- inspection

    def best_bid(self) -> Optional[int]:
        """Highest bid price in ticks, or None if the bid side is empty."""
        return max(self._bids) if self._bids else None

    def best_ask(self) -> Optional[int]:
        """Lowest ask price in ticks, or None if the ask side is empty."""
        return min(self._asks) if self._asks else None

    def best_bid_size(self) -> int:
        """Total resting size at the best bid level (0 if side empty)."""
        b = self.best_bid()
        return 0 if b is None else sum(o[1] for o in self._bids[b])

    def best_ask_size(self) -> int:
        """Total resting size at the best ask level (0 if side empty)."""
        a = self.best_ask()
        return 0 if a is None else sum(o[1] for o in self._asks[a])

    def mid(self) -> float:
        """Mid price in ticks, ``(bid + ask) / 2``; NaN if either side is empty."""
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return math.nan
        return 0.5 * (b + a)

    def spread(self) -> float:
        """Bid-ask spread in ticks; NaN if either side is empty."""
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return math.nan
        return float(a - b)

    def microprice(self) -> float:
        """Size-weighted mid: ``(q_a * P_b + q_b * P_a) / (q_a + q_b)``.

        ``P_b, P_a`` are best bid/ask prices (ticks) and ``q_b, q_a`` the total
        sizes at those levels.  The microprice leans toward the side with less
        depth — the direction the next mid move is more likely to take.
        NaN when either side is empty.
        """
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return math.nan
        qb = float(self.best_bid_size())
        qa = float(self.best_ask_size())
        return (qa * b + qb * a) / (qa + qb)

    def depth(self, side: str, k: int) -> int:
        """Total resting size over the top ``k`` price levels on ``side``.

        Fewer than ``k`` levels: sums what exists.  Raises ValueError on bad
        side or non-positive ``k``.
        """
        self._check_side(side)
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        levels = self._bids if side == self.BUY else self._asks
        prices = sorted(levels, reverse=(side == self.BUY))[:k]
        return sum(sum(o[1] for o in levels[p]) for p in prices)

    def order_ids(self, side: str) -> List[int]:
        """Sorted ids of live resting orders on ``side`` (pinned ordering
        used by the zero-intelligence generator's cancel selection)."""
        self._check_side(side)
        return sorted(i for i, (s, _) in self._orders.items() if s == side)

    def has_order(self, order_id: int) -> bool:
        """True if ``order_id`` is currently resting in the book."""
        return order_id in self._orders

    def total_size(self, side: str) -> int:
        """Total resting size on ``side`` across all levels."""
        self._check_side(side)
        levels = self._bids if side == self.BUY else self._asks
        return sum(sum(o[1] for o in q) for q in levels.values())

    def n_orders(self, side: str) -> int:
        """Number of live resting orders on ``side``."""
        self._check_side(side)
        return sum(1 for s, _ in self._orders.values() if s == side)

    # -------------------------------------------------------------- execution

    def _match(self, side: str, size: int, limit_ticks: Optional[int]) -> Tuple[List[Fill], int]:
        """Execute up to ``size`` against the side opposite to ``side``.

        ``limit_ticks`` bounds acceptable prices (None = market order, walks
        everything).  Consumes levels best-first and orders FIFO within a
        level.  Returns (fills, unfilled remainder).
        """
        fills: List[Fill] = []
        remaining = size
        opp = self._asks if side == self.BUY else self._bids
        while remaining > 0 and opp:
            best = min(opp) if side == self.BUY else max(opp)
            if limit_ticks is not None:
                if side == self.BUY and best > limit_ticks:
                    break
                if side == self.SELL and best < limit_ticks:
                    break
            queue = opp[best]
            while remaining > 0 and queue:
                order = queue[0]
                take = min(remaining, order[1])
                order[1] -= take
                remaining -= take
                fills.append(Fill(maker_id=order[0], price=best, size=take))
                if order[1] == 0:
                    queue.popleft()
                    del self._orders[order[0]]
            if not queue:
                del opp[best]
        self.trades.extend(fills)
        return fills, remaining

    def limit(self, side: str, price: float, size: int, order_id: int) -> Tuple[List[Fill], int]:
        """Submit a limit order.

        A crossed limit (buy priced at/above best ask, or sell at/below best
        bid) converts to a marketable execution against every crossing level;
        the unfilled remainder rests at ``price``.

        Args:
            side: 'B' or 'S'.
            price: limit price (multiple of tick_size).
            size: positive integer quantity.
            order_id: unique id for the resting part.

        Returns:
            (fills, rested_size) — ``rested_size`` is 0 if fully executed.

        Raises:
            ValueError: bad side/price/size, or duplicate live ``order_id``.
        """
        self._check_side(side)
        self._check_size(size)
        ticks = self._to_ticks(price)
        if not isinstance(order_id, int) or isinstance(order_id, bool):
            raise ValueError(f"order_id must be an integer, got {order_id!r}")
        if order_id in self._orders:
            raise ValueError(f"order_id {order_id} is already live in the book")
        fills, remaining = self._match(side, size, ticks)
        if remaining > 0:
            levels = self._bids if side == self.BUY else self._asks
            levels.setdefault(ticks, deque()).append([order_id, remaining])
            self._orders[order_id] = (side, ticks)
        executed = size - remaining
        self.log.append(("L", side, ticks, size, order_id, executed, 0))
        return fills, remaining

    def cancel(self, order_id: int) -> int:
        """Cancel a resting order by id; returns the size removed.

        Raises:
            ValueError: if the id is not live (never resting, already
                cancelled, or fully filled).
        """
        if order_id not in self._orders:
            raise ValueError(f"cancel of unknown order_id {order_id}")
        side, ticks = self._orders.pop(order_id)
        levels = self._bids if side == self.BUY else self._asks
        queue = levels[ticks]
        removed = 0
        for i, order in enumerate(queue):
            if order[0] == order_id:
                removed = order[1]
                del queue[i]
                break
        if not queue:
            del levels[ticks]
        self.log.append(("C", side, ticks, removed, order_id, 0, 0))
        return removed

    def market(self, side: str, size: int) -> Tuple[List[Fill], int]:
        """Submit a market order that walks the opposite side of the book.

        A market order larger than the total opposite depth fills what is
        available; the unfilled remainder is returned (reported), which is not
        an error.

        Returns:
            (fills, unfilled).
        """
        self._check_side(side)
        self._check_size(size)
        fills, unfilled = self._match(side, size, None)
        self.log.append(("M", side, 0, size, 0, size - unfilled, unfilled))
        return fills, unfilled
