"""Fit :class:`FlowParams` to a dictionary of summary statistics.

The input is a plain stats dict, never a data file, so the fit can be tested on
synthetic cases with known answers.  The pass rules in :data:`TOLERANCES` are the
ones fixed in ``research/calibration_criteria.md``.

Stats dict
----------
``spread_dist``  time-weighted share of spread at 1, 2, 3, 4, 5+ ticks
``event_mix``    shares of adds, cancels, fills
``event_rate``   book events per second
``touch_median``, ``touch_p90``  time-weighted touch size, lots
``sign_acf``     aggressive-order sign autocorrelation at lags 1, 10, 100
``ofi_r2``       R2 of the 10-second mid change on 10-second order-flow imbalance
``vol10``        sd of the 10-second mid change, ticks
``order_rate``   aggressive orders per second

Stage 1 (:func:`moment_fit`) inverts rough steady-state relations.  Stage 2
(:func:`refine`) runs Nelder-Mead on five log multipliers against
:func:`distance`, with the same seeds at every step so that noise in the
comparison is shared.  The simulate function is an argument, so stage 2 can be
tested without running the market.  Only the listed parameters move; Hawkes
timescale, informed flow and sizes stay as given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..neldermead import nelder_mead
from .engine import FlowParams, Simulator

SIGN_LAGS = (1, 10, 100)
MEAN_LIMIT_LOTS = 2.5       # the engine draws limit sizes uniformly from 1..4
BUCKET_S = 10.0
GRID_S = 0.1
REFINED = ("limit_rate", "cancel_rate", "queue_scale", "p_geometric", "jump_rate")
_STEP = 6.0                 # log multiplier per unit: the optimiser's first step is 0.3

#: every row of research/calibration_criteria.md
TOLERANCES = {
    "C1": 0.15, "C2a": 0.10, "C2b": (0.75, 1.33), "C3_median": (0.7, 1.43), "C3_p90": (0.5, 2.0),
    "C4": (0.10, 0.05, 0.03), "C5": 0.15, "C6": (0.7, 1.43),
}


def acf(x: Sequence[float], lags: Sequence[int]) -> List[float]:
    """Sample autocorrelation; NaN where the series is too short or constant."""
    v = np.asarray(x, dtype=float)
    d = v - v.mean() if v.size else v
    var = float(np.mean(d * d)) if v.size else 0.0
    return [float(np.mean(d[:-k] * d[k:]) / var) if var > 0.0 and k < v.size else math.nan for k in lags]


def spread_distribution(spread_ticks: Sequence[float]) -> List[float]:
    s = np.clip(np.rint(np.asarray(spread_ticks, dtype=float)), 1, 5).astype(int)
    return (np.bincount(s, minlength=6)[1:6] / max(s.size, 1)).tolist()


def r_squared(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size < 3 or a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def build_stats(spread_ticks: Sequence[float], touch_lots: Sequence[float], n_add: int, n_cancel: int,
                n_fill: int, seconds: float, signs: Sequence[int], ofi: Sequence[float],
                dmid: Sequence[float]) -> Dict[str, object]:
    """One definition of each statistic, shared by real data and simulated paths."""
    n = n_add + n_cancel + n_fill
    if n <= 0 or seconds <= 0.0:
        raise ValueError("need positive event count and duration")
    q = np.asarray(touch_lots, dtype=float)
    return {
        "spread_dist": spread_distribution(spread_ticks),
        "event_mix": [n_add / n, n_cancel / n, n_fill / n],
        "event_rate": n / seconds,
        "touch_median": float(np.median(q)), "touch_p90": float(np.percentile(q, 90)),
        "sign_acf": acf(signs, SIGN_LAGS),
        "ofi_r2": r_squared(ofi, dmid),
        "vol10": float(np.std(np.asarray(dmid, dtype=float))),
        "order_rate": len(signs) / seconds,
    }


def _tv(a: Sequence[float], b: Sequence[float]) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def _ratio(sim: float, real: float) -> float:
    return sim / real if real > 0.0 else math.inf


def evaluate(sim: Mapping[str, object], real: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """Each criterion row: ``value`` and ``passed``.  NaN never passes."""
    def within(x: float, band: Tuple[float, float]) -> bool:
        return band[0] <= x <= band[1]

    tol = TOLERANCES
    rows: Dict[str, Dict[str, object]] = {}
    c1 = _tv(sim["spread_dist"], real["spread_dist"])
    rows["C1"] = {"value": c1, "passed": c1 <= tol["C1"]}
    c2 = _tv(sim["event_mix"], real["event_mix"])
    rows["C2a"] = {"value": c2, "passed": c2 <= tol["C2a"]}
    r = _ratio(sim["event_rate"], real["event_rate"])
    rows["C2b"] = {"value": r, "passed": within(r, tol["C2b"])}
    med, p90 = _ratio(sim["touch_median"], real["touch_median"]), _ratio(sim["touch_p90"], real["touch_p90"])
    rows["C3"] = {"value": [med, p90], "passed": within(med, tol["C3_median"]) and within(p90, tol["C3_p90"])}
    gaps = [abs(a - b) for a, b in zip(sim["sign_acf"], real["sign_acf"])]
    rows["C4"] = {"value": gaps, "passed": all(g <= t for g, t in zip(gaps, tol["C4"]))}
    c5 = abs(sim["ofi_r2"] - real["ofi_r2"])
    rows["C5"] = {"value": c5, "passed": c5 <= tol["C5"]}
    c6 = _ratio(sim["vol10"], real["vol10"])
    rows["C6"] = {"value": c6, "passed": within(c6, tol["C6"])}
    return rows


def distance(sim: Mapping[str, object], real: Mapping[str, object]) -> float:
    """Sum of squared gaps in units of each tolerance; 1 per row is the pass edge.

    Sign memory is left out: stage 2 moves no parameter that sets it.
    """
    def log_gap(x: float, band: Tuple[float, float]) -> float:
        if not x > 0.0 or math.isinf(x):
            return 10.0
        return math.log(x) / math.log(band[1])

    tol = TOLERANCES
    terms = [
        _tv(sim["spread_dist"], real["spread_dist"]) / tol["C1"],
        _tv(sim["event_mix"], real["event_mix"]) / tol["C2a"],
        log_gap(_ratio(sim["event_rate"], real["event_rate"]), tol["C2b"]),
        log_gap(_ratio(sim["touch_median"], real["touch_median"]), tol["C3_median"]),
        log_gap(_ratio(sim["touch_p90"], real["touch_p90"]), tol["C3_p90"]),
        abs(sim["ofi_r2"] - real["ofi_r2"]) / tol["C5"],
        log_gap(_ratio(sim["vol10"], real["vol10"]), tol["C6"]),
    ]
    return float(sum(t * t for t in terms))


def moment_fit(stats: Mapping[str, object], base: FlowParams = FlowParams(), branching: float = 0.5,
               beta: Optional[float] = None) -> FlowParams:
    """Stage 1: closed-form starting point from steady-state balance.

    Market orders: Hawkes mean rate ``2 mu / (1 - n)``, with the self-excited share
    of ``n`` set from lag-1 sign memory.  Prices: compound Poisson jumps give
    ``vol10^2 = 10 rate sd^2``, with the sd set to the mean spread.  Adds: the
    per-side rate, undoing the queue damping at the median touch.  Cancels: per-side
    cancels divided by the implied resting orders.
    """
    n = min(max(branching, 0.0), 0.95)
    b = base.beta if beta is None else float(beta)
    if b <= 0.0:
        raise ValueError(f"beta must be positive, got {b}")
    order_rate = float(stats["order_rate"])
    if order_rate <= 0.0:
        raise ValueError("order_rate must be positive")
    mu = 0.5 * order_rate * (1.0 - n)
    a1 = stats["sign_acf"][0]
    share = min(max(0.5 * (1.0 + (0.0 if math.isnan(a1) else a1)), 0.5), 0.99)
    alpha = ((n * share * b, n * (1.0 - share) * b), (n * (1.0 - share) * b, n * share * b))

    dist = np.asarray(stats["spread_dist"], dtype=float)
    mean_spread = float(dist @ np.arange(1, 6))
    jump_sd = max(mean_spread, 1.0)
    jump_rate = float(stats["vol10"]) ** 2 / (BUCKET_S * jump_sd ** 2)

    rate = float(stats["event_rate"])
    adds, cancels = stats["event_mix"][0] * rate / 2.0, stats["event_mix"][1] * rate / 2.0
    touch = max(float(stats["touch_median"]), 1.0)
    queue_scale = touch
    limit_rate = 2.0 * adds             # 1 + touch / queue_scale = 2 at the median
    resting = max(touch / MEAN_LIMIT_LOTS / base.p_geometric, 1.0)
    cancel_rate = cancels / resting
    return replace(base, mu=(mu, mu), alpha=alpha, beta=b, jump_rate=jump_rate, jump_sd=jump_sd,
                   limit_rate=limit_rate, queue_scale=queue_scale, cancel_rate=cancel_rate)


def _apply(flow: FlowParams, x: Sequence[float]) -> FlowParams:
    over = {}
    for name, v in zip(REFINED, x):
        value = getattr(flow, name) * math.exp(_STEP * (v - 1.0))
        over[name] = min(value, 0.95) if name == "p_geometric" else value
    return replace(flow, **over)


class _Spent(Exception):
    pass


@dataclass(frozen=True)
class FitResult:
    flow: FlowParams
    start: FlowParams
    value: float
    start_value: float
    n_evals: int


def refine(real: Mapping[str, object], start: FlowParams, simulate: Callable[[FlowParams], Mapping[str, object]],
           budget: int = 60) -> FitResult:
    """Stage 2: search within ``budget`` calls of ``simulate``; keeps the best seen."""
    if budget < 1:
        raise ValueError("budget must be >= 1")
    best: List = [math.inf, list(np.ones(len(REFINED)))]
    calls = [0]
    first: List[float] = []

    def objective(x: Sequence[float]) -> float:
        if calls[0] >= budget:
            raise _Spent
        calls[0] += 1
        value = distance(simulate(_apply(start, x)), real)
        if not first:
            first.append(value)
        if value < best[0]:
            best[0], best[1] = value, list(x)
        return value

    try:
        nelder_mead(objective, best[1], xatol=1e-3, fatol=1e-3, maxiter=budget)
    except _Spent:
        pass
    return FitResult(_apply(start, best[1]), start, best[0], first[0], calls[0])


class _Recording(Simulator):
    """The engine unchanged, plus a log of what statistics need."""

    def __init__(self, flow: FlowParams, T: float, seed: int) -> None:
        super().__init__(flow, [], T, seed)
        self.n_add = self.n_cancel = self.n_fill = 0
        self.order_t: List[float] = []
        self.order_sign: List[int] = []
        self.top: List[tuple] = []
        self.ofi: Dict[int, float] = {}

    def _bg_post(self, side, price, size) -> None:
        super()._bg_post(side, price, size)
        self.n_add += 1

    def _bg_cancel(self, oid) -> None:
        super()._bg_cancel(oid)
        self.n_cancel += 1

    def _settle(self, fills, t, aggr_side, taker, informed, mid_pre) -> None:
        super()._settle(fills, t, aggr_side, taker, informed, mid_pre)
        if fills:
            self.n_fill += len(fills)
            self.order_t.append(t)
            self.order_sign.append(1 if aggr_side == "B" else -1)

    def _touch(self, t) -> None:
        pb, pa, pqb, pqa = self.bid, self.ask, self.bid_size, self.ask_size
        super()._touch(t)
        b, a, qb, qa = self.bid, self.ask, self.bid_size, self.ask_size
        if (b, a, qb, qa) == (pb, pa, pqb, pqa) or b is None or a is None:
            return
        self.top.append((t, b, a, qb, qa))
        if pb is not None and pa is not None:
            e = (qb if b >= pb else 0) - (pqb if b <= pb else 0) - (qa if a <= pa else 0) + (pqa if a >= pa else 0)
            k = int(t // BUCKET_S)
            self.ofi[k] = self.ofi.get(k, 0.0) + e


def simulate_stats(flow: FlowParams, T: float, seed: int, warmup: float = 60.0) -> Dict[str, object]:
    """Run one path and return its stats dict; the first ``warmup`` seconds are dropped.

    Event counts include the warm-up in numerator and denominator alike, because
    the engine does not stamp them; a warm-up well below ``T`` keeps that small.
    """
    if not 0.0 <= warmup < T:
        raise ValueError("need 0 <= warmup < T")
    sim = _Recording(flow, T, seed)
    sim.run()
    top = np.asarray(sim.top, dtype=float)
    if top.shape[0] < 2:
        raise ValueError("path produced no two-sided book")
    grid = np.arange(warmup, T, GRID_S)
    row = top[np.maximum(np.searchsorted(top[:, 0], grid, side="right") - 1, 0)]
    spread = row[:, 2] - row[:, 1]
    touch = np.concatenate([row[:, 3], row[:, 4]])
    edges = np.arange(math.ceil(warmup / BUCKET_S), int(T // BUCKET_S) + 1)
    idx = np.maximum(np.searchsorted(top[:, 0], edges * BUCKET_S, side="right") - 1, 0)
    mid = 0.5 * (top[idx, 1] + top[idx, 2])
    ofi = [sim.ofi.get(int(k), 0.0) for k in edges[:-1]]
    t = np.asarray(sim.order_t)
    signs = np.asarray(sim.order_sign)[t >= warmup] if t.size else np.array([], dtype=int)
    return build_stats(spread, touch, sim.n_add, sim.n_cancel, sim.n_fill, T, signs, ofi, np.diff(mid))


def mean_stats(paths: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Average stats dicts key by key, over paths."""
    if not paths:
        raise ValueError("need at least one path")
    return {k: (np.mean([p[k] for p in paths], axis=0).tolist() if isinstance(paths[0][k], list)
                else float(np.mean([p[k] for p in paths]))) for k in paths[0]}
