r"""Event-driven taker backtest: what a signal earns after paying to trade.

Python-only analysis code, outside the four-language golden contract.

:func:`lobmod.nn.signal_pnl` stays as the pinned diagnostic: it sums overlapping
mid-to-mid returns, so it measures whether a signal knows anything, not what
could be banked.  This module walks through the signals in time order and does
what an account could do:

* one position open at a time, ``lots`` units, never above ``inventory_cap``;
* a signal seen after event ``i`` reaches the book at state ``e = i + delay``;
* entry crosses the spread: buy at the ask, sell at the bid;
* exit crosses back at state ``x = e + hold``, or at the first later state where
  both sides exist with at least ``lots`` at the touch we hit;
* signals seen while a position is open are ignored.

For a long round trip of ``q`` lots with fee ``c`` per side per lot,

    pnl = q (bid_x - ask_e) - 2 c q,

and a short is ``q (bid_e - ask_x) - 2 c q``.  Equivalently
``pnl = mid_pnl - q (s_e + s_x) / 2 - 2 c q`` with ``s`` the spread.

Limits: quotes come from the historic stream, so our orders have no impact and
the flow does not react; only the touch is used, so an entry without ``lots``
at the touch is skipped instead of walking the book.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest settings.

    Attributes:
        hold: events between entry and exit (>= 1).
        delay: decision delay in events (>= 0); 0 assumes zero latency.
        fee_per_side: fee in ticks per lot per side (>= 0).
        lots: position size (>= 1).
        inventory_cap: maximum absolute position (>= lots).
    """

    hold: int
    delay: int = 0
    fee_per_side: float = 0.0
    lots: int = 1
    inventory_cap: int = 1

    def __post_init__(self) -> None:
        if self.hold < 1 or self.delay < 0 or self.lots < 1 or self.fee_per_side < 0.0:
            raise ValueError(f"invalid backtest config: {self}")
        if self.lots > self.inventory_cap:
            raise ValueError(f"lots {self.lots} exceeds inventory cap {self.inventory_cap}")


@dataclass(frozen=True)
class RoundTrip:
    """One completed trade.  Prices in ticks, ``pnl`` net of fees."""

    side: int
    signal_index: int
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    mid_pnl: float
    fees: float
    pnl: float
    forced: bool


@dataclass(frozen=True)
class BacktestResult:
    """Round trips and summary.

    Attributes:
        trades: completed round trips, in time order.
        stats: ``n_trades, gross, fees, net, mid_pnl, spread_cost, mean_net,
            sd_net, hit_rate, max_drawdown, n_signals, n_ignored_busy,
            n_skipped_quote, n_skipped_end, n_forced``.  ``gross`` is after
            the spread and before fees; ``n_forced`` counts exits made at the
            last usable state because the book never recovered.
    """

    trades: List[RoundTrip]
    stats: Dict[str, float]


def _usable(states: Any, i: int, side: int, lots: int) -> bool:
    """True if a taker with sign ``side`` can trade ``lots`` at the touch at ``i``."""
    bid, ask = states.best_bid[i], states.best_ask[i]
    if not (math.isfinite(bid) and math.isfinite(ask)):
        return False
    return (states.ask_size[i] if side > 0 else states.bid_size[i]) >= lots


def run_backtest(
    states: Any, signal_index: Sequence[int], signal: Sequence[int], config: BacktestConfig
) -> BacktestResult:
    """Run the taker backtest (module docstring).

    Args:
        states: object with ``best_bid, best_ask, bid_size, ask_size`` arrays,
            normally :class:`lobmod.replay.BookStates`.
        signal_index: strictly increasing event indices of the decisions.
        signal: desired direction at each index, in ``{-1, 0, +1}``.
        config: see :class:`BacktestConfig`.

    Raises:
        ValueError: mismatched inputs, unsorted indices, or a bad signal value.
    """
    idx = np.asarray(signal_index, dtype=int)
    sig = np.asarray(signal, dtype=int)
    if idx.shape != sig.shape or idx.ndim != 1:
        raise ValueError("signal_index and signal must be equal-length 1-D arrays")
    if idx.size > 1 and np.any(np.diff(idx) <= 0):
        raise ValueError("signal_index must be strictly increasing")
    if np.any(np.abs(sig) > 1):
        raise ValueError("signal values must be in {-1, 0, +1}")
    n = len(states.best_bid)
    q = config.lots
    trades: List[RoundTrip] = []
    busy_until = -1
    ignored = skipped_quote = skipped_end = 0
    for i, s in zip(idx.tolist(), sig.tolist()):
        if s == 0:
            continue
        if i <= busy_until:
            ignored += 1
            continue
        e = i + config.delay
        if e + config.hold >= n:
            skipped_end += 1
            continue
        if not _usable(states, e, s, q):
            skipped_quote += 1
            continue
        x = e + config.hold
        while x < n and not _usable(states, x, -s, q):
            x += 1
        forced = x >= n
        if forced:
            x = n - 1
            while not (math.isfinite(states.best_bid[x]) and math.isfinite(states.best_ask[x])):
                x -= 1
        bid_e, ask_e = float(states.best_bid[e]), float(states.best_ask[e])
        bid_x, ask_x = float(states.best_bid[x]), float(states.best_ask[x])
        entry, leave = (ask_e, bid_x) if s > 0 else (bid_e, ask_x)
        mid_pnl = q * s * (0.5 * (bid_x + ask_x) - 0.5 * (bid_e + ask_e))
        fees = 2.0 * config.fee_per_side * q
        trades.append(RoundTrip(
            side=s, signal_index=i, entry_index=e, exit_index=x, entry_price=entry,
            exit_price=leave, mid_pnl=mid_pnl, fees=fees,
            pnl=q * s * (leave - entry) - fees, forced=forced,
        ))
        busy_until = x
    net = np.array([t.pnl for t in trades])
    fee_total = float(sum(t.fees for t in trades))
    mid_total = float(sum(t.mid_pnl for t in trades))
    total = float(net.sum())
    curve = np.r_[0.0, np.cumsum(net)]
    stats: Dict[str, float] = {
        "n_trades": len(trades),
        "gross": total + fee_total,
        "fees": fee_total,
        "net": total,
        "mid_pnl": mid_total,
        "spread_cost": mid_total - (total + fee_total),
        "mean_net": float(net.mean()) if net.size else math.nan,
        "sd_net": float(net.std(ddof=1)) if net.size > 1 else math.nan,
        "hit_rate": float(np.mean(net > 0.0)) if net.size else math.nan,
        "max_drawdown": float(np.max(np.maximum.accumulate(curve) - curve)),
        "n_signals": int(np.sum(sig != 0)),
        "n_ignored_busy": ignored,
        "n_skipped_quote": skipped_quote,
        "n_skipped_end": skipped_end,
        "n_forced": sum(t.forced for t in trades),
    }
    return BacktestResult(trades, stats)


def foresight_ceiling(states: Any, horizon: int) -> Dict[str, float]:
    r"""Taker with perfect foresight of the book ``H`` events ahead.

    Per event the better of the two crossing round trips,

        best_i = max(bid_{i+H} - ask_i,  bid_i - ask_{i+H}),

    is an upper bound for any taker holding exactly ``H`` events.  Returns
    ``share_profitable`` and ``mean_best``.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    bid = np.asarray(states.best_bid, dtype=float)
    ask = np.asarray(states.best_ask, dtype=float)
    best = np.fmax(bid[horizon:] - ask[:-horizon], bid[:-horizon] - ask[horizon:])
    best = best[np.isfinite(best)]
    return {"share_profitable": float(np.mean(best > 0.0)), "mean_best": float(best.mean())}
