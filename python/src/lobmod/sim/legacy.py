"""Replica of the review's passive maker, with attribution added.

Same logic as ``run`` in review/2026-11-05/exp_tradability.py, which cannot be
imported because it executes on import.  It exists to explain the old cap=20
result, not as a simulator: it replays a fixed zero-intelligence stream that
does not react to the injected orders.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..book import LimitOrderBook
from ..replay import apply_event, replay
from ..zi import generate_events


def legacy_maker(seed: int, delay: int, cap: int, fee: float = 0.0, n: int = 20000) -> Dict[str, float]:
    """Return pnl plus an exact split into spread capture and position P&L.

    ``spread`` sums ``sign * (mid before fill - price)``; ``position`` sums
    ``inventory * mid change`` from that same mid onward.  With fees they equal pnl.
    """
    ev = generate_events(n, seed)
    b = LimitOrderBook(1.0, record_trades=False)
    oid = 10 ** 12
    live: Dict[str, tuple] = {}
    inv, cash, spread, fills, next_ok = 0, 0.0, 0.0, 0, 0
    last_mid = None
    position = 0.0
    inv_path = []

    def sweep() -> None:
        nonlocal inv, cash, spread, fills
        for side in list(live):
            o, p = live[side]
            if not b.has_order(o):
                sgn = 1 if side == "B" else -1
                inv += sgn
                cash += -sgn * p - fee
                fills += 1
                spread += sgn * (last_mid - p)
                del live[side]

    for i, e in enumerate(ev):
        try:
            apply_event(b, e)
        except ValueError:
            pass
        sweep()
        bid, ask = b.best_bid(), b.best_ask()
        if bid is None or ask is None:
            continue
        mid = 0.5 * (bid + ask)
        if last_mid is not None:
            position += inv * (mid - last_mid)
        last_mid = mid
        inv_path.append(inv)
        if i < 100 or i < next_ok:
            continue
        for side, px, ok in (("B", bid, inv < cap), ("S", ask, inv > -cap)):
            cur = live.get(side)
            if cur and (cur[1] != px or not ok):
                b.cancel(cur[0])
                del live[side]
                cur = None
                next_ok = i + delay
            if not cur and ok and i >= next_ok:
                oid += 1
                b.limit_ticks(side, px, 1, oid)
                sweep()
                if b.has_order(oid):
                    live[side] = (oid, px)
    return {
        "pnl": cash + inv * last_mid, "spread": spread, "position": position, "fees": -fee * fills,
        "fills": fills, "inv_sd": float(np.std(inv_path)), "mean_abs_inv": float(np.mean(np.abs(inv_path))),
    }


def variance_ratio(seed: int, horizons: Sequence[int] = (10, 100, 1000, 5000), n: int = 20000) -> Dict[int, float]:
    """Var(H-event mid change) / (H * Var(1-event change)); 1 = random walk."""
    st = replay(generate_events(n, seed), 3)
    mid = 0.5 * (st.best_bid + st.best_ask)
    mid = mid[np.isfinite(mid)]
    v1 = np.var(np.diff(mid))
    return {h: float(np.var(mid[h:] - mid[:-h]) / (h * v1)) for h in horizons}
