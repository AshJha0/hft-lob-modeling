r"""Order-flow features and direction labels from replayed book states.

Order-flow imbalance (OFI)
--------------------------
Cont, Kukanov & Stoikov (2014) style.  For consecutive book states
``n-1 -> n`` with best bid price/size ``(Pb, qb)`` and best ask ``(Pa, qa)``,
the per-event order-flow contribution is (pinned)

    e_n =  1{Pb_n >= Pb_{n-1}} * qb_n  -  1{Pb_n <= Pb_{n-1}} * qb_{n-1}
         - 1{Pa_n <= Pa_{n-1}} * qa_n  +  1{Pa_n >= Pa_{n-1}} * qa_{n-1}

(bid improvements / additions push up, ask improvements / additions push
down; equal prices contribute the size *change* on that side).  The OFI over
a trailing window of ``w`` events ending at event ``i`` is

    OFI_i = sum_{n = i-w+1}^{i} e_n .

Feature vector (pinned order) at event ``i``:

    [ OFI_i,
      QI_i          = (qb_i - qa_i) / (qb_i + qa_i)      (queue imbalance),
      spread_i      = Pa_i - Pb_i                        (ticks),
      depth_ratio_i = bid_depth_k_i / ask_depth_k_i,
      ret_i         = mid_i - mid_{i - w_ret}            (short-horizon return) ]

Label: mid-price direction over the next ``H`` events with an epsilon dead
zone (pinned classes)::

    d = mid_{i+H} - mid_i
    y = 2 (up)   if d >  epsilon
        0 (down) if d < -epsilon
        1 (flat) otherwise

Rows are emitted only where every input is defined: ``i >= max(w_ofi,
w_ret)``, ``i + H < n_events``, both book sides non-empty at every state the
row touches, and all denominators positive.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .replay import BookStates

#: pinned feature column order
FEATURE_NAMES = ["ofi", "queue_imbalance", "spread", "depth_ratio", "ret"]

#: pinned label classes
LABEL_DOWN, LABEL_FLAT, LABEL_UP = 0, 1, 2


@dataclass(frozen=True)
class FeatureConfig:
    """Pinned feature / label / NN hyperparameters (data/features_config.json).

    Attributes:
        ofi_window: trailing OFI window length in events.
        ret_window: lookback for the short-horizon return feature.
        horizon: label horizon H in events.
        epsilon: dead-zone half-width for the flat class (ticks).
        depth_k: levels per side in the depth-ratio feature.
        hidden: MLP hidden layer sizes.
        learning_rate: full-batch gradient-descent step.
        epochs: training epochs (full-batch steps).
        train_frac: chronological train fraction.
    """

    ofi_window: int
    ret_window: int
    horizon: int
    epsilon: float
    depth_k: int
    hidden: Tuple[int, ...]
    learning_rate: float
    epochs: int
    train_frac: float

    def __post_init__(self) -> None:
        for name in ("ofi_window", "ret_window", "horizon", "depth_k", "epochs"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer, got {v}")
        if not (math.isfinite(self.epsilon) and self.epsilon >= 0.0):
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0.0):
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError(f"train_frac must be in (0, 1), got {self.train_frac}")
        if any((not isinstance(h, int)) or h <= 0 for h in self.hidden):
            raise ValueError(f"hidden sizes must be positive integers, got {self.hidden}")


def load_feature_config(path: str) -> FeatureConfig:
    """Load a :class:`FeatureConfig` from its pinned JSON file."""
    with open(path) as fh:
        raw = json.load(fh)
    return FeatureConfig(
        ofi_window=int(raw["ofi_window"]),
        ret_window=int(raw["ret_window"]),
        horizon=int(raw["horizon"]),
        epsilon=float(raw["epsilon"]),
        depth_k=int(raw["depth_k"]),
        hidden=tuple(int(h) for h in raw["hidden"]),
        learning_rate=float(raw["learning_rate"]),
        epochs=int(raw["epochs"]),
        train_frac=float(raw["train_frac"]),
    )


@dataclass
class FeatureSet:
    """Feature matrix with aligned labels and trading metadata.

    Attributes:
        X: (n, 5) feature matrix in :data:`FEATURE_NAMES` order.
        y: (n,) integer labels in {0, 1, 2}.
        event_index: (n,) 0-based index into the BookStates arrays.
        fwd_ret: (n,) realized ``mid_{i+H} - mid_i`` (for PnL evaluation).
        spread: (n,) spread at decision time (for cost modelling).
    """

    X: np.ndarray
    y: np.ndarray
    event_index: np.ndarray
    fwd_ret: np.ndarray
    spread: np.ndarray

    def __len__(self) -> int:
        return int(self.y.shape[0])


def ofi_series(states: BookStates) -> np.ndarray:
    """Per-event order-flow contributions ``e_n`` (module docstring formula).

    ``e_0 = NaN`` (no predecessor); any event whose own or previous state has
    an empty side yields NaN.
    """
    pb, qb = states.best_bid, states.bid_size
    pa, qa = states.best_ask, states.ask_size
    n = len(states)
    e = np.full(n, np.nan)
    # vectorized comparisons; NaN prices propagate NaN via the mask below
    with np.errstate(invalid="ignore"):
        bid_up = pb[1:] >= pb[:-1]
        bid_dn = pb[1:] <= pb[:-1]
        ask_dn = pa[1:] <= pa[:-1]
        ask_up = pa[1:] >= pa[:-1]
    contrib = (
        np.where(bid_up, qb[1:], 0.0)
        - np.where(bid_dn, qb[:-1], 0.0)
        - np.where(ask_dn, qa[1:], 0.0)
        + np.where(ask_up, qa[:-1], 0.0)
    )
    valid = (
        np.isfinite(pb[1:]) & np.isfinite(pb[:-1]) & np.isfinite(pa[1:]) & np.isfinite(pa[:-1])
    )
    e[1:] = np.where(valid, contrib, np.nan)
    return e


def ofi_at(states: BookStates, event_index: int, window: int) -> float:
    """OFI over the ``window`` events ending at ``event_index`` (0-based).

    Raises:
        ValueError: if the window does not fit (``event_index - window < 0``)
            or arguments are invalid.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}")
    n = len(states)
    if not (0 <= event_index < n):
        raise ValueError(f"event_index {event_index} out of range [0, {n})")
    if event_index - window < 0:
        raise ValueError(f"window {window} does not fit before event {event_index}")
    e = ofi_series(states)
    return float(np.sum(e[event_index - window + 1: event_index + 1]))


def compute_features(states: BookStates, cfg: FeatureConfig) -> FeatureSet:
    """Build the feature matrix and labels from replayed book states.

    See the module docstring for the pinned formulas; rows with any undefined
    input are skipped.
    """
    n = len(states)
    e = ofi_series(states)
    start = max(cfg.ofi_window, cfg.ret_window)
    rows = []
    for i in range(start, n - cfg.horizon):
        window = e[i - cfg.ofi_window + 1: i + 1]
        if not np.all(np.isfinite(window)):
            continue
        qb, qa = states.bid_size[i], states.ask_size[i]
        mid_now = states.mid[i]
        mid_past = states.mid[i - cfg.ret_window]
        mid_fwd = states.mid[i + cfg.horizon]
        spread = states.spread[i]
        bd, ad = states.bid_depth_k[i], states.ask_depth_k[i]
        if not all(map(math.isfinite, (mid_now, mid_past, mid_fwd, spread))):
            continue
        if qb + qa <= 0.0 or ad <= 0.0:
            continue
        ofi = float(np.sum(window))
        qi = (qb - qa) / (qb + qa)
        depth_ratio = bd / ad
        ret = mid_now - mid_past
        d = mid_fwd - mid_now
        if d > cfg.epsilon:
            label = LABEL_UP
        elif d < -cfg.epsilon:
            label = LABEL_DOWN
        else:
            label = LABEL_FLAT
        rows.append((i, ofi, qi, spread, depth_ratio, ret, label, d))
    if not rows:
        raise ValueError("no valid feature rows could be built from the states")
    arr = np.asarray([r[1:6] for r in rows], dtype=float)
    return FeatureSet(
        X=arr,
        y=np.asarray([r[6] for r in rows], dtype=int),
        event_index=np.asarray([r[0] for r in rows], dtype=int),
        fwd_ret=np.asarray([r[7] for r in rows], dtype=float),
        spread=np.asarray([r[3] for r in rows], dtype=float),
    )


def chronological_split(n: int, train_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """Pinned chronological split: first ``floor(train_frac * n)`` rows train.

    Raises:
        ValueError: if the split would leave either side empty.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n <= 1:
        raise ValueError(f"n must be an integer > 1, got {n}")
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    n_train = int(math.floor(train_frac * n))
    if n_train == 0 or n_train == n:
        raise ValueError(f"split leaves an empty side: n={n}, train_frac={train_frac}")
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:]
