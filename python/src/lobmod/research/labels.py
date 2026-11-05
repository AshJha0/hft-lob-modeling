r"""Cost-aware labels: does crossing the spread both ways at horizon H pay?

Python-only analysis code, outside the four-language golden contract.

The pinned label asks whether the mid moved by more than ``epsilon``.  A taker
does not earn the mid.  With decision at state ``i``, entry at ``e = i + delay``,
exit at ``x = e + H`` and fee ``c`` per side per lot,

    long_i  = bid_x - ask_e - 2 c,
    short_i = bid_e - ask_x - 2 c.

The label is UP (2) if ``long_i > margin``, DOWN (0) if ``short_i > margin`` and
FLAT (1) otherwise.  Both cannot hold at once when spreads and costs are
non-negative: their sum is ``-(s_e + s_x) - 4 c``.  This is the per-event form of
:func:`lobmod.research.backtest.foresight_ceiling` with costs added, and matches
the fill rule of :func:`lobmod.research.backtest.run_backtest` when both touches
hold enough size.

Rows whose entry or exit side is empty are marked invalid, not labelled flat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from ..features import LABEL_DOWN, LABEL_FLAT, LABEL_UP


@dataclass(frozen=True)
class CostLabels:
    """Labels aligned with ``event_index``.

    Attributes:
        y: labels in {0, 1, 2}; FLAT on invalid rows.
        long_pnl: net ticks of the long round trip (NaN if invalid).
        short_pnl: net ticks of the short round trip (NaN if invalid).
        valid: both quotes exist at entry and exit, inside the stream.
    """

    y: np.ndarray
    long_pnl: np.ndarray
    short_pnl: np.ndarray
    valid: np.ndarray


def cost_aware_labels(
    states: Any, event_index: Sequence[int], horizon: int, fee_per_side: float = 0.0,
    delay: int = 0, margin: float = 0.0,
) -> CostLabels:
    """Label rows by the net result of crossing both ways (module docstring).

    Raises:
        ValueError: ``horizon < 1``, or negative ``delay``, ``fee_per_side``
            or ``margin``.
    """
    if horizon < 1 or delay < 0 or fee_per_side < 0.0 or margin < 0.0:
        raise ValueError("need horizon >= 1 and delay, fee_per_side, margin >= 0")
    bid = np.asarray(states.best_bid, dtype=float)
    ask = np.asarray(states.best_ask, dtype=float)
    idx = np.asarray(event_index, dtype=int)
    e, x = idx + delay, idx + delay + horizon
    inside = x < bid.size
    e_s, x_s = np.where(inside, e, 0), np.where(inside, x, 0)
    cost = 2.0 * fee_per_side
    long_pnl = np.where(inside, bid[x_s] - ask[e_s] - cost, np.nan)
    short_pnl = np.where(inside, bid[e_s] - ask[x_s] - cost, np.nan)
    valid = np.isfinite(long_pnl) & np.isfinite(short_pnl)
    y = np.full(idx.size, LABEL_FLAT, dtype=int)
    y[valid & (long_pnl > margin)] = LABEL_UP
    y[valid & (short_pnl > margin)] = LABEL_DOWN
    return CostLabels(y, long_pnl, short_pnl, valid)


def horizon_sweep(
    states: Any, horizons: Sequence[int], fee_per_side: float = 0.0, delay: int = 0,
) -> List[Dict[str, float]]:
    """Per horizon: how often a perfectly informed taker would be paid.

    ``share_up`` and ``share_down`` are label shares over valid rows;
    ``mean_best`` is the mean of ``max(long, short)``, a ceiling that no
    signal can reach, and ``mean_paid`` its mean over the paying rows only.
    """
    rows = []
    idx = np.arange(np.size(states.best_bid))
    for h in horizons:
        lab = cost_aware_labels(states, idx, h, fee_per_side, delay)
        y = lab.y[lab.valid]
        best = np.maximum(lab.long_pnl, lab.short_pnl)[lab.valid]
        paid = best[best > 0.0]
        rows.append({
            "horizon": float(h),
            "share_up": float(np.mean(y == LABEL_UP)),
            "share_down": float(np.mean(y == LABEL_DOWN)),
            "mean_best": float(best.mean()),
            "mean_paid": float(paid.mean()) if paid.size else 0.0,
        })
    return rows
