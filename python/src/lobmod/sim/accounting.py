"""P&L attribution and execution statistics.

For fill ``i`` with signed quantity ``q``, price ``p``, pre-trade mid ``m0``,
mid ``mh`` one horizon later and final mid ``mT``::

    q * (mT - p) = q * (m0 - p)      spread capture
                 + q * (mh - m0)     adverse selection
                 + q * (mT - mh)     inventory (position held past the horizon)

Summed, this equals cash plus final position marked at ``mT``.  The cash figure
is booked separately in the engine, so the identity is a real check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .engine import SimResult


@dataclass(frozen=True)
class Attribution:
    spread: float
    adverse: float
    inventory: float
    fees: float
    total: float
    booked: float
    lots: int


def _arrays(res: SimResult, name: str):
    a = res.agents[name]
    q = np.asarray(a.sign, float) * np.asarray(a.qty, float)
    return a, np.asarray(a.t, float), q, np.asarray(a.price, float), np.asarray(a.mid_pre, float)


def attribute(res: SimResult, name: str, horizon: float = 1.0) -> Attribution:
    a, t, q, p, m0 = _arrays(res, name)
    m_end = float(res.mid[-1])
    mh = res.mid_at(t + horizon) if len(t) else t
    spread = float(np.sum(q * (m0 - p)))
    adverse = float(np.sum(q * (mh - m0)))
    inventory = float(np.sum(q * (m_end - mh)))
    return Attribution(spread, adverse, inventory, -a.fees, spread + adverse + inventory - a.fees,
                       a.cash + a.inventory * m_end - a.fees, int(np.sum(np.abs(q))))


def realised_spread(res: SimResult, name: str, horizons: Sequence[float]) -> Dict[float, float]:
    """Mean per lot of ``sign * (mid(t + h) - price)``; NaN without fills."""
    _, t, q, p, _ = _arrays(res, name)
    if not len(t):
        return {h: float("nan") for h in horizons}
    return {h: float(np.sum(q * (res.mid_at(t + h) - p)) / np.sum(np.abs(q))) for h in horizons}


def fill_prob_by_queue(res: SimResult, name: str, edges: Sequence[int] = (0, 3, 8, 15, 30)) -> Dict[str, Tuple[int, int]]:
    """Bucket by lots ahead at entry: (orders, orders with any fill).

    Orders the strategy pulled count as unfilled, so this is the realised rate
    for that strategy, not a property of the queue alone.
    """
    log = np.asarray(res.agents[name].queue_log, float).reshape(-1, 2)
    out: Dict[str, Tuple[int, int]] = {}
    bounds = list(edges) + [np.inf]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = (log[:, 0] >= lo) & (log[:, 0] < hi)
        out[f"{lo}+" if hi == np.inf else f"{lo}-{int(hi) - 1}"] = (int(m.sum()), int(log[m, 1].sum()))
    return out


def summarise(res: SimResult, name: str, horizon: float = 1.0) -> Dict[str, float]:
    a = res.agents[name]
    at = attribute(res, name, horizon)
    return {
        "total": at.total, "spread": at.spread, "adverse": at.adverse, "inventory": at.inventory,
        "fees": at.fees, "lots": at.lots, "late_cancels": a.late_cancels, "rejects": a.rejects,
        "max_inv": a.max_abs_inventory, "rested": a.orders_rested,
        "informed_share": float(np.mean(a.vs_informed)) if a.vs_informed else 0.0,
        "mean_spread": res.mean_spread,
    }


def assess_edge(mean: float, lo: float, hi: float, p_informed: float, maker_fee: float, taker_fee: float = 0.0) -> str:
    """Wording for a result.  No branch of this function can claim an edge.

    With no informed flow and no fees, the uninformed Hawkes flow pays the
    spread by construction, so a positive figure restates an assumption.
    """
    if p_informed == 0.0 and maker_fee == 0.0 and taker_fee == 0.0:
        return "not an edge: no informed flow and no fees, so the spread is paid by construction"
    if lo <= 0.0 <= hi:
        return "indistinguishable from zero in this model"
    if mean < 0.0:
        return "loses in this model"
    return "positive in this uncalibrated model only; not evidence of an edge"
