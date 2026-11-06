"""Discrete-event market: one clock, one real book, delayed agents.

Exchange events (Hawkes market orders, fair-value jumps, background limit,
cancel and refresh events, informed takers, agent messages) run in clock order
against a :class:`LimitOrderBook`.  Queue priority is therefore the book's own
FIFO, not a model of it.  An agent sees book and signal changes ``md_latency``
late, and its messages arrive ``oe_latency`` after it decides, so a cancel can
reach the book after the order it targets has traded; that is counted as a
late cancel.

Background rates are constant between events and every event redraws the next
background time, which is exact for competing exponentials.  Fair-value jumps,
informed slots, market sizes and Hawkes arrivals come from streams no agent can
touch, so changing a latency keeps them identical across a paired comparison.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..book import LimitOrderBook
from ..hawkes import simulate_biv

_AGENT_ID0 = 10 ** 12
_MD, _SIG, _REPORT, _NEW, _CXL, _TAKE, _INFORMED, _SWITCH = range(8)


@dataclass(frozen=True)
class FlowParams:
    """Assumed flow parameters.  Units: ticks, seconds, lots.  Not calibrated."""

    mu: Sequence[float] = (0.8, 0.8)                             # Hawkes base, buy / sell
    alpha: Sequence[Sequence[float]] = ((1.0, 0.3), (0.3, 1.0))  # excitation
    beta: float = 3.0
    jump_rate: float = 0.3         # fair-value jumps per second
    jump_sd: float = 1.5           # jump size sd, ticks
    p_informed: float = 1.0        # chance each informed slot is live; 0 = no informed flow
    informed_per_jump: int = 2
    informed_delay: float = 0.02   # mean delay after a jump, seconds
    informed_size: int = 5
    informed_margin: float = 0.25  # only takes quotes this far through fair value
    limit_rate: float = 6.0        # per side, per second, before state scaling
    queue_scale: float = 20.0      # limit rate falls as touch size grows
    imbalance_limit: float = 0.5   # support for the heavier side
    cancel_rate: float = 0.5       # per resting order, per second
    imbalance_cancel: float = 0.5  # thin side cancels faster
    refresh_rate: float = 20.0     # background stale-quote checks per second
    anchor_tau: float = 0.2        # background's lag behind fair value, seconds
    p_geometric: float = 0.5       # placement depth
    start_price: int = 10000
    sign_persist: float = 0.0      # lag-1 memory of aggressive side; 0 keeps the Hawkes side
    impact: float = 0.0            # permanent fair-value move per aggressive lot, ticks
    active_mult: float = 1.0       # rate in the active state over the calm state; 1 = off
    active_share: float = 0.2      # long-run share of time in the active state
    active_dwell: float = 300.0    # mean active spell, seconds
    book_feedback: float = 0.0     # share of an aggressive order's mid move kept in fair value
    burst_prob: float = 0.0        # chance an aggressive order is followed by a same-side one
    burst_gap: float = 0.0002      # mean gap inside a burst, seconds


@dataclass(frozen=True)
class Fees:
    """Per-lot fee in ticks by liquidity side; a negative value is a rebate."""

    maker: float = 0.0
    taker: float = 0.0


@dataclass(frozen=True)
class AgentSpec:
    name: str
    strategy: object
    md_latency: float = 0.001
    oe_latency: float = 0.001
    fees: Fees = Fees()
    size: int = 1
    sees_signal: bool = True


@dataclass
class AgentResult:
    """Everything accounting needs.  One row per fill in the parallel lists."""

    spec: AgentSpec
    t: List[float] = field(default_factory=list)
    sign: List[int] = field(default_factory=list)
    price: List[int] = field(default_factory=list)
    qty: List[int] = field(default_factory=list)
    mid_pre: List[float] = field(default_factory=list)
    is_maker: List[bool] = field(default_factory=list)
    vs_informed: List[bool] = field(default_factory=list)
    cash: float = 0.0
    fees: float = 0.0
    inventory: int = 0
    max_abs_inventory: int = 0
    orders_rested: int = 0
    late_cancels: int = 0
    rejects: int = 0
    queue_log: List[List[int]] = field(default_factory=list)   # [lots ahead at entry, filled 0/1]


@dataclass
class SimResult:
    agents: Dict[str, AgentResult]
    mid_t: np.ndarray
    mid: np.ndarray
    final_fair: float
    T: float
    n_events: int
    mean_spread: float
    flow: FlowParams

    def mid_at(self, t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.mid_t, np.minimum(t, self.T), side="right") - 1
        return self.mid[np.maximum(idx, 0)]


class _Order:
    __slots__ = ("side", "price", "left", "agent", "log_row")

    def __init__(self, side: str, price: int, left: int, agent: "_Agent", log_row: List[int]) -> None:
        self.side, self.price, self.left, self.agent, self.log_row = side, price, left, agent, log_row


class _Agent:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        self.res = AgentResult(spec)
        self.view = SimpleNamespace(bid=None, ask=None, bid_size=0, ask_size=0, signal=None, inventory=0)
        self.target: Dict[str, Optional[int]] = {"B": None, "S": None}
        self.oid: Dict[str, Optional[int]] = {"B": None, "S": None}
        self.taking = False
        self.resting: Dict[str, Dict[int, int]] = {"B": {}, "S": {}}


class Simulator:
    """Run one seeded path.  Create a new instance for every run."""

    def __init__(self, flow: FlowParams, agents: Sequence[AgentSpec], T: float, seed: int) -> None:
        if T <= 0.0:
            raise ValueError(f"T must be positive, got {T}")
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
        if any(a.md_latency < 0.0 or a.oe_latency < 0.0 for a in agents):
            raise ValueError("latencies must be non-negative")
        self.flow, self.T, self.seed = flow, float(T), int(seed)
        self.book = LimitOrderBook(1.0, record_trades=False)
        self.agents = [_Agent(a) for a in agents]
        ss = np.random.SeedSequence(self.seed).spawn(6)
        self._rng_fair = np.random.default_rng(ss[0])
        self._rng_bg = np.random.default_rng(ss[1])
        self._hawkes_seed = int(ss[2].generate_state(1)[0]) | 1
        self._rng_side = np.random.default_rng(ss[3])
        self._rng_state = np.random.default_rng(ss[4])
        self._rng_burst = np.random.default_rng(ss[5])
        self.activity = 1.0
        self._ubuf: List[float] = []
        self.fair = self.anchor = float(flow.start_price)
        self.t_anchor = 0.0
        self.now = 0.0
        self.heap: List[tuple] = []
        self.seq = 0
        self.next_oid = 1                   # background and informed ids
        self.next_agent_oid = _AGENT_ID0
        self.bg_size: Dict[int, int] = {}
        self.bg_meta: Dict[int, tuple] = {}
        self.bg_ids: Dict[str, List[int]] = {"B": [], "S": []}
        self.bg_pos: Dict[int, int] = {}
        self.bg_level: Dict[str, Dict[int, Dict[int, None]]] = {"B": {}, "S": {}}
        self.levels: Dict[str, Dict[int, int]] = {"B": {}, "S": {}}   # shadow size per price
        self.orders: Dict[int, _Order] = {}
        self.bid = self.ask = None
        self.bid_size = self.ask_size = 0
        self.mid = float(flow.start_price)
        self.mid_t: List[float] = [0.0]
        self.mid_path: List[float] = [self.mid]
        self.spread_sum = 0.0
        self.spread_n = 0
        self.n_events = 0

    # ------------------------------------------------------------ small tools
    def _u(self) -> float:
        if not self._ubuf:
            self._ubuf = self._rng_bg.random(4096).tolist()
        return self._ubuf.pop()

    def _push(self, t: float, code: int, a=None, b=None) -> None:
        self.seq += 1
        heapq.heappush(self.heap, (t, self.seq, code, a, b))

    def _advance_anchor(self, t: float) -> None:
        w = 1.0 - math.exp(-(t - self.t_anchor) / self.flow.anchor_tau)
        self.anchor += (self.fair - self.anchor) * w
        self.t_anchor = t

    # ------------------------------------------------------------- book state
    def _touch(self, t: float) -> None:
        """Refresh quotes after any book change; publish if the top changed."""
        lv = self.levels
        bid = max(lv["B"]) if lv["B"] else None
        ask = min(lv["S"]) if lv["S"] else None
        bs = lv["B"][bid] if bid is not None else 0
        asz = lv["S"][ask] if ask is not None else 0
        if (bid, ask, bs, asz) == (self.bid, self.ask, self.bid_size, self.ask_size):
            return
        self.bid, self.ask, self.bid_size, self.ask_size = bid, ask, bs, asz
        if bid is not None and ask is not None:
            mid = 0.5 * (bid + ask)
            self.spread_sum += ask - bid
            self.spread_n += 1
            if mid != self.mid:
                self.mid = mid
                self.mid_t.append(t)
                self.mid_path.append(mid)
        for ag in self.agents:
            self._push(t + ag.spec.md_latency, _MD, ag, self._view_of(ag, bid, ask, bs, asz))

    def _view_of(self, ag: _Agent, bid, ask, bs, asz) -> tuple:
        """Top of book without the agent's own resting size.

        Real quoters net out their own orders.  Without this a lone order that
        sets the touch is chased by its own delayed echo, forever.
        """
        own = ag.resting
        if own["B"] and bid in own["B"]:
            lv = self.levels["B"]
            live = [p for p, n in lv.items() if n > own["B"].get(p, 0)]
            bid = max(live) if live else None
            bs = lv[bid] - own["B"].get(bid, 0) if bid is not None else 0
        if own["S"] and ask in own["S"]:
            lv = self.levels["S"]
            live = [p for p, n in lv.items() if n > own["S"].get(p, 0)]
            ask = min(live) if live else None
            asz = lv[ask] - own["S"].get(ask, 0) if ask is not None else 0
        return bid, ask, bs, asz

    @staticmethod
    def _own_delta(ag: _Agent, side: str, price: int, size: int) -> None:
        own = ag.resting[side]
        own[price] = own.get(price, 0) + size
        if own[price] == 0:
            del own[price]

    def _add_level(self, side: str, price: int, size: int) -> None:
        lv = self.levels[side]
        lv[price] = lv.get(price, 0) + size

    def _drop_level(self, side: str, price: int, size: int) -> None:
        lv = self.levels[side]
        lv[price] -= size
        if lv[price] == 0:
            del lv[price]

    def _bg_remove(self, oid: int) -> None:
        side, price = self.bg_meta.pop(oid)
        del self.bg_size[oid]
        ids, pos = self.bg_ids[side], self.bg_pos
        i = pos.pop(oid)
        last = ids.pop()
        if last != oid:
            ids[i] = last
            pos[last] = i
        lvl = self.bg_level[side][price]
        del lvl[oid]
        if not lvl:
            del self.bg_level[side][price]

    def _settle(self, fills, t: float, aggr_side: str, taker: Optional[_Agent], informed: bool,
                mid_pre: float) -> None:
        """Book the fills from one aggressive order, makers and taker alike."""
        rest_side = "S" if aggr_side == "B" else "B"
        for f in fills:
            self._drop_level(rest_side, f.price, f.size)
            oid = f.maker_id
            if oid in self.bg_size:
                self.bg_size[oid] -= f.size
                if self.bg_size[oid] == 0:
                    self._bg_remove(oid)
            else:
                o = self.orders[oid]
                o.left -= f.size
                self._own_delta(o.agent, o.side, o.price, -f.size)
                o.log_row[1] = 1
                done = o.left == 0
                if done:
                    del self.orders[oid]
                sign = 1 if o.side == "B" else -1
                self._record(o.agent, t, sign, f.price, f.size, mid_pre, True, informed)
                self._push(t + o.agent.spec.md_latency, _REPORT, o.agent,
                           (sign * f.size, o.side if done else None, oid))
            if taker is not None:
                sign = 1 if aggr_side == "B" else -1
                self._record(taker, t, sign, f.price, f.size, mid_pre, False, False)
                self._push(t + taker.spec.md_latency, _REPORT, taker, (sign * f.size, None, None))

    @staticmethod
    def _record(ag: _Agent, t, sign, price, qty, mid_pre, maker, informed) -> None:
        r = ag.res
        r.t.append(t)
        r.sign.append(sign)
        r.price.append(price)
        r.qty.append(qty)
        r.mid_pre.append(mid_pre)
        r.is_maker.append(maker)
        r.vs_informed.append(informed)
        r.cash -= sign * price * qty
        r.fees += (ag.spec.fees.maker if maker else ag.spec.fees.taker) * qty
        r.inventory += sign * qty
        r.max_abs_inventory = max(r.max_abs_inventory, abs(r.inventory))

    # ------------------------------------------------------- background events
    def _bg_post(self, side: str, price: int, size: int) -> None:
        oid = self.next_oid
        self.next_oid += 1
        self.book.limit_ticks(side, price, size, oid)
        self._add_level(side, price, size)
        self.bg_size[oid] = size
        self.bg_meta[oid] = (side, price)
        self.bg_pos[oid] = len(self.bg_ids[side])
        self.bg_ids[side].append(oid)
        self.bg_level[side].setdefault(price, {})[oid] = None

    def _bg_cancel(self, oid: int) -> None:
        side, price = self.bg_meta[oid]
        removed = self.book.cancel(oid)
        self._drop_level(side, price, removed)
        self._bg_remove(oid)

    def _bg_rates(self) -> List[float]:
        p = self.flow
        m = self.activity
        total = self.bid_size + self.ask_size
        imb = (self.bid_size - self.ask_size) / total if total else 0.0
        return [
            m * p.limit_rate * (1.0 + p.imbalance_limit * imb) / (1.0 + self.bid_size / p.queue_scale),
            m * p.limit_rate * (1.0 - p.imbalance_limit * imb) / (1.0 + self.ask_size / p.queue_scale),
            m * p.cancel_rate * len(self.bg_ids["B"]) * (1.0 - p.imbalance_cancel * imb),
            m * p.cancel_rate * len(self.bg_ids["S"]) * (1.0 + p.imbalance_cancel * imb),
            p.refresh_rate,
        ]

    def _bg_event(self, t: float, rates: List[float]) -> None:
        p = self.flow
        self._advance_anchor(t)
        x = self._u() * sum(rates)
        kind = 0
        while kind < 4 and x >= rates[kind]:
            x -= rates[kind]
            kind += 1
        if kind < 2:
            k = int(math.log(1.0 - self._u()) / math.log(1.0 - p.p_geometric))
            size = 1 + int(self._u() * 4)
            if kind == 0:
                price = math.ceil(self.anchor) - 1 - k
                if self.ask is not None:
                    price = min(price, self.ask - 1)
                self._bg_post("B", price, size)
            else:
                price = math.floor(self.anchor) + 1 + k
                if self.bid is not None:
                    price = max(price, self.bid + 1)
                self._bg_post("S", price, size)
        elif kind < 4:
            ids = self.bg_ids["B" if kind == 2 else "S"]
            if ids:
                self._bg_cancel(ids[int(self._u() * len(ids))])
        else:  # refresh: oldest background order on a touch through the anchor leaves
            if self.bid is not None and self.bid >= self.anchor and self.bid in self.bg_level["B"]:
                self._bg_cancel(next(iter(self.bg_level["B"][self.bid])))
            if self.ask is not None and self.ask <= self.anchor and self.ask in self.bg_level["S"]:
                self._bg_cancel(next(iter(self.bg_level["S"][self.ask])))
        self._touch(t)

    def _ioc(self, side: str, limit: int, size: int, t: float, taker: Optional[_Agent], informed: bool) -> None:
        oid = self.next_oid
        self.next_oid += 1
        mid_pre = self.mid
        fills, left = self.book.limit_ticks(side, limit, size, oid)
        if left:
            self.book.cancel(oid)
        self._settle(fills, t, side, taker, informed, mid_pre)
        self._touch(t)

    def _market(self, side: str, size: int, t: float) -> None:
        mid_pre = self.mid
        fills, _ = self.book.market(side, size)
        self._settle(fills, t, side, None, False, mid_pre)
        if self.flow.impact:
            self._advance_anchor(t)
            self.fair += (1.0 if side == "B" else -1.0) * self.flow.impact * size
        self._touch(t)
        if self.flow.book_feedback and self.mid != mid_pre:
            self._advance_anchor(t)
            move = self.flow.book_feedback * (self.mid - mid_pre)
            self.fair += move
            self.anchor += move

    def _informed(self, t: float) -> None:
        p = self.flow
        if self.ask is not None and self.ask <= self.fair - p.informed_margin:
            self._ioc("B", math.floor(self.fair - p.informed_margin), p.informed_size, t, None, True)
        elif self.bid is not None and self.bid >= self.fair + p.informed_margin:
            self._ioc("S", math.ceil(self.fair + p.informed_margin), p.informed_size, t, None, True)

    # ------------------------------------------------------------ agent logic
    def _decide(self, ag: _Agent, t: float) -> None:
        q = ag.spec.strategy.decide(ag.view)
        arrive = t + ag.spec.oe_latency
        for side, want in (("B", q.bid), ("S", q.ask)):
            if want == ag.target[side]:
                continue
            if ag.oid[side] is not None:
                self._push(arrive, _CXL, ag, ag.oid[side])
                ag.oid[side] = None
            ag.target[side] = want
            if want is not None:
                oid = self.next_agent_oid
                self.next_agent_oid += 1
                ag.oid[side] = oid
                self._push(arrive, _NEW, ag, (oid, side, want))
        if q.take_side is not None and not ag.taking:
            ag.taking = True
            self._push(arrive, _TAKE, ag, (q.take_side, q.take_limit))

    def _on_new(self, ag: _Agent, oid: int, side: str, price: int, t: float) -> None:
        crosses = (side == "B" and self.ask is not None and price >= self.ask) or (
            side == "S" and self.bid is not None and price <= self.bid)
        if crosses:  # post-only: the agent learns one md delay later
            ag.res.rejects += 1
            self._push(t + ag.spec.md_latency, _REPORT, ag, (0, side, oid))
            return
        row = [self.levels[side].get(price, 0), 0]
        ag.res.queue_log.append(row)
        ag.res.orders_rested += 1
        self.orders[oid] = _Order(side, price, ag.spec.size, ag, row)
        self.book.limit_ticks(side, price, ag.spec.size, oid)
        self._add_level(side, price, ag.spec.size)
        self._own_delta(ag, side, price, ag.spec.size)
        self._touch(t)

    def _on_cancel(self, ag: _Agent, oid: int, t: float) -> None:
        o = self.orders.pop(oid, None)
        if o is None:
            ag.res.late_cancels += 1   # traded or rejected before the cancel landed
            return
        removed = self.book.cancel(oid)
        self._drop_level(o.side, o.price, removed)
        self._own_delta(ag, o.side, o.price, -removed)
        self._touch(t)

    def _on_report(self, ag: _Agent, delta: int, done_side: Optional[str], oid, t: float) -> None:
        ag.view.inventory += delta
        # a report on a replaced order must not clear the newer working one
        if done_side is not None and ag.oid[done_side] == oid:
            ag.target[done_side] = None
            ag.oid[done_side] = None
        self._decide(ag, t)

    # -------------------------------------------------------------- main loop
    def _activity_path(self) -> tuple:
        """State switch times and levels; the time-average level is one."""
        p = self.flow
        calm = 1.0 / (1.0 - p.active_share + p.active_share * p.active_mult)
        calm_dwell = p.active_dwell * (1.0 - p.active_share) / p.active_share
        t, active = 0.0, False
        ts, levels = [0.0], [calm]
        while True:
            t += self._rng_state.exponential(p.active_dwell if active else calm_dwell)
            if t >= self.T:
                return np.asarray(ts), np.asarray(levels)
            active = not active
            ts.append(t)
            levels.append(calm * p.active_mult if active else calm)

    def _chain_sides(self, n: int) -> List[int]:
        """Two-state chain with lag-1 autocorrelation ``sign_persist``; 0 buy, 1 sell."""
        flips = self._rng_side.random(n) < 0.5 * (1.0 - self.flow.sign_persist)
        start = int(self._rng_side.integers(0, 2))
        return ((start + np.cumsum(flips)) % 2).tolist()

    def _add_bursts(self, times: List[float], types: List[int]) -> tuple:
        """Follow each order with a geometric number of same-side orders, in time order."""
        p = self.flow
        t = np.asarray(times, dtype=float)
        side = np.asarray(types, dtype=int)
        extra = self._rng_burst.geometric(1.0 - p.burst_prob, t.size) - 1
        parent = np.repeat(np.arange(t.size), extra)
        first = np.cumsum(extra) - extra
        step = self._rng_burst.exponential(p.burst_gap, parent.size)
        total = np.cumsum(step)
        base = np.where(extra > 0, total[np.minimum(first, max(parent.size - 1, 0))] - step[
            np.minimum(first, max(parent.size - 1, 0))], 0.0) if parent.size else np.zeros(t.size)
        child_t = t[parent] + total - base[parent] if parent.size else np.zeros(0)
        all_t = np.concatenate([t, child_t])
        all_s = np.concatenate([side, side[parent]])
        keep = all_t <= self.T
        order = np.argsort(all_t[keep], kind="stable")
        return all_t[keep][order].tolist(), all_s[keep][order].tolist()

    def run(self) -> SimResult:
        p = self.flow
        horizon = self.T
        if p.active_mult != 1.0:
            sw_t, sw_m = self._activity_path()
            clock = np.append(sw_t, self.T)
            busy = np.concatenate([[0.0], np.cumsum(sw_m * np.diff(clock))])
            horizon = float(busy[-1])
            self.activity = float(sw_m[0])
            for t_s, m_s in zip(sw_t[1:].tolist(), sw_m[1:].tolist()):
                self._push(t_s, _SWITCH, m_s)
        times, types = simulate_biv(p.mu, [list(r) for r in p.alpha], p.beta, horizon, self._hawkes_seed)
        if p.active_mult != 1.0:      # busy time back to clock time
            times = np.interp(times, busy, clock)
        times, types = times.tolist(), types.tolist()
        if p.sign_persist:
            types = self._chain_sides(len(times))
        if p.burst_prob:
            times, types = self._add_bursts(times, types)
        sizes = (1 + self._rng_fair.integers(0, 3, len(times))).tolist()
        n_j = int(self._rng_fair.poisson(p.jump_rate * self.T))
        jump_t = np.sort(self._rng_fair.uniform(0.0, self.T, n_j)).tolist()
        jump_x = (p.jump_sd * self._rng_fair.standard_normal(n_j)).tolist()
        slots = self._rng_fair.random((n_j, p.informed_per_jump)).tolist()
        delays = self._rng_fair.exponential(p.informed_delay, (n_j, p.informed_per_jump)).tolist()
        for i in range(1, 6):
            self._bg_post("B", p.start_price - i, 4)
            self._bg_post("S", p.start_price + i, 4)
        self._touch(0.0)
        ih = ij = 0
        inf = math.inf
        while True:
            rates = self._bg_rates()
            t_b = self.now - math.log(1.0 - self._u()) / sum(rates)
            t_heap = self.heap[0][0] if self.heap else inf
            t_h = times[ih] if ih < len(times) else inf
            t_j = jump_t[ij] if ij < n_j else inf
            t = min(t_b, t_heap, t_h, t_j)
            if t > self.T:
                break
            self.now = t
            self.n_events += 1
            if t == t_heap:
                _, _, code, a, b = heapq.heappop(self.heap)
                if code == _MD:
                    a.view.bid, a.view.ask, a.view.bid_size, a.view.ask_size = b
                    self._decide(a, t)
                elif code == _SIG:
                    a.view.signal = b
                    self._decide(a, t)
                elif code == _REPORT:
                    self._on_report(a, b[0], b[1], b[2], t)
                elif code == _NEW:
                    self._on_new(a, b[0], b[1], b[2], t)
                elif code == _CXL:
                    self._on_cancel(a, b, t)
                elif code == _TAKE:
                    a.taking = False
                    self._ioc(b[0], b[1], a.spec.size, t, a, False)
                elif code == _SWITCH:
                    self.activity = a
                else:
                    self._informed(t)
            elif t == t_j:
                self._advance_anchor(t)
                self.fair += jump_x[ij]
                for k in range(p.informed_per_jump):
                    if slots[ij][k] < p.p_informed:
                        self._push(t + delays[ij][k], _INFORMED)
                for ag in self.agents:
                    if ag.spec.sees_signal:
                        self._push(t + ag.spec.md_latency, _SIG, ag, self.fair)
                ij += 1
            elif t == t_h:
                self._market("B" if types[ih] == 0 else "S", sizes[ih], t)
                ih += 1
            else:
                self._bg_event(t, rates)
        return SimResult(
            agents={a.spec.name: a.res for a in self.agents},
            mid_t=np.asarray(self.mid_t), mid=np.asarray(self.mid_path), final_fair=self.fair,
            T=self.T, n_events=self.n_events,
            mean_spread=self.spread_sum / max(self.spread_n, 1), flow=p,
        )
