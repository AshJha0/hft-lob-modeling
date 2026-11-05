r"""Honest evaluation: leak-free splits, dependence-aware intervals, metrics.

Python-only analysis code, outside the four-language golden contract.  numpy
``default_rng`` is allowed here (see :mod:`lobmod.research`).

Purged walk-forward
-------------------
A row at event ``t`` has a label that looks ahead to ``t + H`` and features that
look back ``W`` events.  A training row just before the test block therefore
shares information with it.  Each fold keeps only training rows with

    t + H + embargo < t_first_test,

so no training label reaches the test block (purge), and ``embargo`` leaves
a further gap; set it to the longest feature lookback to remove shared inputs
too.  Training rows always precede the test block.

Block bootstrap
---------------
Neighbouring rows overlap, so resampling single rows understates the noise.
Resample contiguous blocks instead, with block length well above ``H``, and take
percentiles of the statistic (moving-block bootstrap, Kuensch 1989).

Metrics
-------
With ``C`` the confusion matrix (rows true, columns predicted):

    log-loss   -mean log p_i(y_i)
    precision  C_kk / sum_i C_ik  (NaN if class k is never predicted)
    macro-F1   mean_k 2 P_k R_k / (P_k + R_k)  (0 for an undefined class)
    MCC        (c s - sum_k p_k t_k) / sqrt((s^2 - sum p_k^2)(s^2 - sum t_k^2))

where ``c = trace C``, ``s = sum C``, ``p_k, t_k`` are predicted and true class
counts (Gorodkin's multi-class form); MCC is 0 for a constant prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..features import FeatureConfig, compute_features
from ..nn import MLP, Standardizer, accuracy, majority_class, signal_pnl
from ..replay import Event, replay
from ..zi import generate_events
from .backtest import BacktestConfig, run_backtest


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate on the full sample with a percentile interval."""

    point: float
    low: float
    high: float
    n_boot: int
    block: int


def block_bootstrap_ci(
    statistic: Callable[[np.ndarray], float], n: int, block: int,
    n_boot: int = 2000, level: float = 0.95, seed: int = 0,
) -> BootstrapCI:
    """Moving-block bootstrap interval for ``statistic(row_indices)``.

    ``block=1`` is the ordinary iid bootstrap.

    Raises:
        ValueError: ``block`` outside ``[1, n]`` or ``level`` outside (0, 1).
    """
    if not 1 <= block <= n:
        raise ValueError(f"block must be in [1, {n}], got {block}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    offsets = np.arange(block)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        out[b] = statistic((starts[:, None] + offsets).ravel()[:n])
    tail = 50.0 * (1.0 - level)
    low, high = np.percentile(out, [tail, 100.0 - tail])
    return BootstrapCI(float(statistic(np.arange(n))), float(low), float(high), n_boot, block)


@dataclass(frozen=True)
class Split:
    """Row positions of one fold."""

    train: np.ndarray
    test: np.ndarray


def walk_forward_splits(
    event_index: Sequence[int], n_folds: int, horizon: int, embargo: int = 0,
    first_test_frac: float = 0.7,
) -> List[Split]:
    """Expanding-window purged splits (module docstring).

    The rows from ``first_test_frac`` onward are cut into ``n_folds``
    contiguous test blocks.  ``n_folds=1`` is the pinned 70/30 chronological
    split with the purge applied.

    Raises:
        ValueError: bad arguments, unsorted index, or an empty training set.
    """
    t = np.asarray(event_index, dtype=int)
    if n_folds < 1 or horizon < 0 or embargo < 0 or not 0.0 < first_test_frac < 1.0:
        raise ValueError("need n_folds >= 1, horizon, embargo >= 0, first_test_frac in (0, 1)")
    if t.size > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("event_index must be strictly increasing")
    first = int(t.size * first_test_frac)
    edges = np.linspace(first, t.size, n_folds + 1).astype(int)
    splits = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            raise ValueError("too many folds for the number of rows")
        train = np.flatnonzero(t + horizon + embargo < t[a])
        if train.size == 0:
            raise ValueError("purge left an empty training set")
        splits.append(Split(train, np.arange(a, b)))
    return splits


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("labels must be non-empty and of equal shape")
    c = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(c, (y_true, y_pred), 1)
    return c


def log_loss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log probability of the true class, clipped at ``eps``."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    return float(-np.mean(np.log(np.clip(p[np.arange(y.size), y], eps, 1.0))))


def per_class_precision(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Precision per class; NaN for a class that is never predicted."""
    c = _confusion(y_true, y_pred, n_classes)
    col = c.sum(axis=0).astype(float)
    return np.where(col > 0, np.diag(c) / np.where(col > 0, col, 1.0), np.nan)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Unweighted mean of per-class F1, with F1 = 0 where undefined."""
    c = _confusion(y_true, y_pred, n_classes).astype(float)
    tp = np.diag(c)
    denom = c.sum(axis=0) + c.sum(axis=1)
    return float(np.mean(np.where(denom > 0, 2.0 * tp / np.where(denom > 0, denom, 1.0), 0.0)))


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Multi-class Matthews correlation; 0 when either marginal is constant."""
    c = _confusion(y_true, y_pred, n_classes).astype(float)
    s = c.sum()
    t, p = c.sum(axis=1), c.sum(axis=0)
    denom = math.sqrt((s * s - p @ p) * (s * s - t @ t))
    return float((np.trace(c) * s - p @ t) / denom) if denom > 0.0 else 0.0


MODELS: Tuple[str, ...] = ("mlp", "logistic", "majority")


def _score(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    precision = per_class_precision(y, pred)
    return {
        "accuracy": accuracy(y, pred),
        "log_loss": log_loss(y, proba),
        "mcc": matthews_corrcoef(y, pred),
        "macro_f1": macro_f1(y, pred),
        "precision_down": float(precision[0]),
        "precision_flat": float(precision[1]),
        "precision_up": float(precision[2]),
    }


def evaluate_stream(
    events: Sequence[Event], cfg: FeatureConfig, embargo: int = 0,
    backtests: Sequence[BacktestConfig] = (),
) -> Dict[str, Dict[str, float]]:
    """Fit MLP, logistic and majority on one stream and score the test block.

    Uses one purged split.  The majority model predicts the training mode with
    training class frequencies as its probabilities.  Each model entry holds
    the metrics, ``gross``/``net``/``n_signals`` from the pinned ``signal_pnl``
    and ``bt{j}_net``, ``bt{j}_trades`` for the ``j``-th backtest config.

    Raises:
        ValueError: the stream gives no usable feature rows.
    """
    states = replay(events, cfg.depth_k)
    fs = compute_features(states, cfg)
    split = walk_forward_splits(fs.event_index, 1, cfg.horizon, embargo, cfg.train_frac)[0]
    tr, te = split.train, split.test
    scaler = Standardizer().fit(fs.X[tr])
    x_tr, x_te = scaler.transform(fs.X[tr]), scaler.transform(fs.X[te])
    y_tr, y_te = fs.y[tr], fs.y[te]
    prior = np.bincount(y_tr, minlength=3) / y_tr.size
    predictions = {
        "majority": (np.full_like(y_te, majority_class(y_tr)), np.tile(prior, (y_te.size, 1))),
    }
    for name, hidden in (("mlp", cfg.hidden), ("logistic", ())):
        model = MLP(x_tr.shape[1], hidden, 3)
        model.fit(x_tr, y_tr, cfg.learning_rate, cfg.epochs)
        predictions[name] = (model.predict(x_te), model.predict_proba(x_te))
    out: Dict[str, Dict[str, float]] = {}
    for name in MODELS:
        pred, proba = predictions[name]
        row = _score(y_te, pred, proba)
        row["gross"], row["net"], row["n_signals"] = signal_pnl(pred, fs.fwd_ret[te], fs.spread[te])
        for j, bt in enumerate(backtests):
            stats = run_backtest(states, fs.event_index[te], pred - 1, bt).stats
            row[f"bt{j}_net"] = stats["net"]
            row[f"bt{j}_trades"] = stats["n_trades"]
        out[name] = row
    return out


def multi_seed_compare(
    seeds: Sequence[int], n_events: int, cfg: FeatureConfig, embargo: int = 0,
    backtests: Sequence[BacktestConfig] = (),
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], List[int]]:
    """Run :func:`evaluate_stream` over ZI streams, one per seed.

    Returns ``(results by seed, seeds skipped)``.  A seed is skipped only if
    its stream cannot produce feature rows, and is reported, not hidden.
    """
    results: Dict[int, Dict[str, Dict[str, float]]] = {}
    skipped: List[int] = []
    for seed in seeds:
        try:
            results[seed] = evaluate_stream(generate_events(n_events, seed), cfg, embargo, backtests)
        except ValueError:
            skipped.append(seed)
    return results, skipped
