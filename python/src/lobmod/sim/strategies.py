"""Agent decision rules.  Each sees only delayed data through a ``view``.

``view`` carries ``bid, ask`` (last seen quotes, may be None), ``bid_size,
ask_size``, ``signal`` (last seen reference price, None if the agent has no
feed) and ``inventory`` (position known from fill reports received so far).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Quote:
    """Desired state: resting prices, plus an optional aggressive order."""

    bid: Optional[int] = None
    ask: Optional[int] = None
    take_side: Optional[str] = None
    take_limit: Optional[int] = None


def reference(view) -> Optional[float]:
    """Signal if the agent has one, else the seen microprice."""
    if view.signal is not None:
        return view.signal
    if view.bid is None or view.ask is None:
        return None
    total = view.bid_size + view.ask_size
    if total <= 0:
        return 0.5 * (view.bid + view.ask)
    return (view.bid * view.ask_size + view.ask * view.bid_size) / total


@dataclass(frozen=True)
class NaiveTouch:
    """Join both touches; pull a side the signal says is through fair value."""

    cap: int = 5

    def decide(self, view) -> Quote:
        if view.bid is None or view.ask is None:
            return Quote()
        bid: Optional[int] = view.bid
        ask: Optional[int] = view.ask
        if view.signal is not None:
            if bid >= view.signal:
                bid = None
            if ask <= view.signal:
                ask = None
        if view.inventory >= self.cap:
            bid = None
        if view.inventory <= -self.cap:
            ask = None
        return Quote(bid, ask)


@dataclass(frozen=True)
class AvellanedaStoikov:
    """Reservation price skewed by inventory, half-spread from the AS formula.

    ``r = ref - q * gamma * sigma2 * tau`` and
    ``delta = gamma * sigma2 * tau / 2 + ln(1 + gamma / kappa) / gamma``, with
    ``tau`` a fixed rolling horizon rather than time to a terminal date.
    """

    gamma: float = 0.1
    sigma2: float = 0.7
    tau: float = 5.0
    kappa: float = 0.5
    cap: int = 5

    def decide(self, view) -> Quote:
        ref = reference(view)
        if ref is None:
            return Quote()
        risk = self.gamma * self.sigma2 * self.tau
        res = ref - view.inventory * risk
        delta = 0.5 * risk + math.log1p(self.gamma / self.kappa) / self.gamma
        bid: Optional[int] = math.floor(res - delta)
        ask: Optional[int] = math.ceil(res + delta)
        if view.ask is not None:
            bid = min(bid, view.ask - 1)
        if view.bid is not None:
            ask = max(ask, view.bid + 1)
        if view.inventory >= self.cap:
            bid = None
        if view.inventory <= -self.cap:
            ask = None
        return Quote(bid, ask)


@dataclass(frozen=True)
class SignalTaker:
    """Cross when the seen quote is stale by more than ``edge`` plus the fee."""

    edge: float = 0.5
    fee: float = 0.0
    cap: int = 5

    def decide(self, view) -> Quote:
        ref = reference(view)
        if ref is None:
            return Quote()
        need = self.edge + self.fee
        if view.ask is not None and view.ask + need <= ref and view.inventory < self.cap:
            return Quote(take_side="B", take_limit=view.ask)
        if view.bid is not None and view.bid - need >= ref and view.inventory > -self.cap:
            return Quote(take_side="S", take_limit=view.bid)
        return Quote()
