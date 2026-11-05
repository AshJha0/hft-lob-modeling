r"""A small gradient-boosted tree classifier, and a walk-forward comparison.

Python-only analysis code, outside the four-language golden contract; numpy only.

Boosting
--------
Multi-class softmax boosting with second-order leaves (Friedman 2001; Chen and
Guestrin 2016).  Each round fits one tree per class to ``g = p - y`` and
``h = p (1 - p)``; a leaf takes ``-G / (H + l2)`` and a split scores

    G_L^2 / (H_L + l2) + G_R^2 / (H_R + l2) - G^2 / (H + l2).

Features are cut at training quantiles into ``n_bins`` bins, so a node costs one
histogram per statistic.  Row subsampling is the only random part.

There is no early stopping or tuning: :class:`GbtConfig` defaults were set before
any result was seen, so the test blocks play no part in model choice.  This is
a baseline, not a replacement for a production library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..features import FeatureConfig, FeatureSet, compute_features
from ..nn import MLP, Standardizer, accuracy, softmax
from .backtest import BacktestConfig, run_backtest
from .evaluation import (
    BootstrapCI, block_bootstrap_ci, log_loss, macro_f1, matthews_corrcoef, walk_forward_splits,
)


@dataclass(frozen=True)
class GbtConfig:
    """Boosting settings; see the module docstring on how they were chosen."""

    n_rounds: int = 40
    depth: int = 3
    learning_rate: float = 0.1
    min_leaf: int = 30
    l2: float = 1.0
    n_bins: int = 16
    subsample: float = 0.8

    def __post_init__(self) -> None:
        if self.n_rounds < 1 or self.depth < 1 or self.min_leaf < 1 or self.n_bins < 2:
            raise ValueError(f"invalid config: {self}")
        if self.learning_rate <= 0.0 or self.l2 < 0.0 or not 0.0 < self.subsample <= 1.0:
            raise ValueError(f"invalid config: {self}")


class _Tree:
    """Flat arrays; ``feature[i] < 0`` marks a leaf."""

    def __init__(self) -> None:
        self.feature: List[int] = []
        self.cut: List[int] = []
        self.left: List[int] = []
        self.right: List[int] = []
        self.value: List[float] = []

    def _node(self) -> int:
        self.feature.append(-1)
        self.cut.append(0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(0.0)
        return len(self.feature) - 1

    def grow(self, B: np.ndarray, g: np.ndarray, h: np.ndarray, rows: np.ndarray,
             depth: int, c: GbtConfig) -> int:
        node = self._node()
        G, H = g[rows].sum(), h[rows].sum()
        self.value[node] = -G / (H + c.l2)
        if depth == 0 or rows.size < 2 * c.min_leaf:
            return node
        F = B.shape[1]
        flat = (B[rows] + np.arange(F) * c.n_bins).ravel()
        size, shape = F * c.n_bins, (F, c.n_bins)
        gl = np.bincount(flat, np.repeat(g[rows], F), size).reshape(shape).cumsum(axis=1)[:, :-1]
        hl = np.bincount(flat, np.repeat(h[rows], F), size).reshape(shape).cumsum(axis=1)[:, :-1]
        nl = np.bincount(flat, minlength=size).reshape(shape).cumsum(axis=1)[:, :-1]
        with np.errstate(divide="ignore", invalid="ignore"):  # empty sides are masked next
            gain = gl**2 / (hl + c.l2) + (G - gl) ** 2 / (H - hl + c.l2) - G**2 / (H + c.l2)
        gain[(nl < c.min_leaf) | (rows.size - nl < c.min_leaf)] = -np.inf
        f, b = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if gain[f, b] <= 1e-12:
            return node
        go_left = B[rows, f] <= b
        self.feature[node], self.cut[node] = int(f), int(b)
        self.left[node] = self.grow(B, g, h, rows[go_left], depth - 1, c)
        self.right[node] = self.grow(B, g, h, rows[~go_left], depth - 1, c)
        return node

    def freeze(self) -> None:
        for name in ("feature", "cut", "left", "right", "value"):
            setattr(self, name, np.asarray(getattr(self, name)))

    def predict(self, B: np.ndarray) -> np.ndarray:
        node = np.zeros(B.shape[0], dtype=int)
        rows = np.arange(B.shape[0])
        while True:
            f = self.feature[node]
            live = f >= 0
            if not live.any():
                return self.value[node]
            r, n = rows[live], node[live]
            node[live] = np.where(B[r, f[live]] <= self.cut[n], self.left[n], self.right[n])


class GradientBoostedTrees:
    """Softmax boosting classifier (module docstring)."""

    def __init__(self, n_classes: int = 3, config: GbtConfig = GbtConfig(), seed: int = 0) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes, self.config, self.seed = n_classes, config, seed
        self.trees: List[List[_Tree]] = []

    def _bin(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [np.searchsorted(e, X[:, j], side="left") for j, e in enumerate(self.edges)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedTrees":
        """Fit on ``(n, F)`` features and integer labels.

        Raises:
            ValueError: mismatched shapes or a label out of range.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
            raise ValueError("X must be (n, F) with n matching y, n > 0")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError("labels out of range")
        c, K = self.config, self.n_classes
        q = np.linspace(0.0, 1.0, c.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], q)) for j in range(X.shape[1])]
        B = self._bin(X)
        rng = np.random.default_rng(self.seed)
        self.init = np.log(np.maximum(np.bincount(y, minlength=K) / y.size, 1e-6))
        score = np.tile(self.init, (y.size, 1))
        onehot = np.eye(K)[y]
        self.trees = []
        m = max(1, int(c.subsample * y.size))
        for _ in range(c.n_rounds):
            p = softmax(score)
            g, h = p - onehot, np.maximum(p * (1.0 - p), 1e-12)
            rows = np.sort(rng.choice(y.size, m, replace=False)) if m < y.size else np.arange(y.size)
            stage = []
            for k in range(K):
                tree = _Tree()
                tree.grow(B, g[:, k], h[:, k], rows, c.depth, c)
                tree.freeze()
                stage.append(tree)
            for k in range(K):
                score[:, k] += c.learning_rate * stage[k].predict(B)
            self.trees.append(stage)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        B = self._bin(np.asarray(X, dtype=float))
        score = np.tile(self.init, (B.shape[0], 1))
        for stage in self.trees:
            for k, tree in enumerate(stage):
                score[:, k] += self.config.learning_rate * tree.predict(B)
        return softmax(score)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


COMPARED: Tuple[str, ...] = ("gbt", "mlp", "logistic", "majority")


@dataclass
class WalkForwardResult:
    """Pooled out-of-fold results of one stream.

    Attributes:
        metrics: per model ``accuracy, log_loss, mcc, macro_f1, trade_share``,
            plus ``net`` and ``n_trades`` when a backtest was requested.
        y: pooled test labels in time order.
        proba: per model pooled class probabilities.
        event_index: decision events of the pooled rows.
    """

    metrics: Dict[str, Dict[str, float]]
    y: np.ndarray
    proba: Dict[str, np.ndarray]
    event_index: np.ndarray


def walk_forward_compare(
    states, cfg: FeatureConfig, n_folds: int = 3, embargo: int = 0,
    relabel: Optional[Callable[[FeatureSet], Tuple[np.ndarray, np.ndarray]]] = None,
    backtest: Optional[BacktestConfig] = None, gbt: GbtConfig = GbtConfig(), seed: int = 0,
    extra_purge: int = 0,
) -> WalkForwardResult:
    """Refit every model on each purged fold; score pooled test rows.

    ``relabel(fs)`` may return ``(y, keep)`` to replace the pinned labels.
    ``extra_purge`` adds to the horizon in the purge, for labels that end after
    ``cfg.horizon`` (a decision delay).  MLP and logistic are the pinned
    deterministic fits, so ``seed`` only reaches the boosting.

    Raises:
        ValueError: no usable rows or a fold with an empty training set.
    """
    fs = compute_features(states, cfg)
    X, y, idx = fs.X, fs.y, fs.event_index
    if relabel is not None:
        y, keep = relabel(fs)
        X, y, idx = X[keep], y[keep], idx[keep]
    splits = walk_forward_splits(idx, n_folds, cfg.horizon + extra_purge, embargo, cfg.train_frac)
    parts: Dict[str, List[np.ndarray]] = {m: [] for m in COMPARED}
    for j, sp in enumerate(splits):
        scaler = Standardizer().fit(X[sp.train])
        a, b = scaler.transform(X[sp.train]), scaler.transform(X[sp.test])
        yt = y[sp.train]
        prior = np.maximum(np.bincount(yt, minlength=3) / yt.size, 1e-12)
        parts["majority"].append(np.tile(prior, (b.shape[0], 1)))
        parts["gbt"].append(GradientBoostedTrees(3, gbt, seed + j).fit(a, yt).predict_proba(b))
        for name, hidden in (("mlp", cfg.hidden), ("logistic", ())):
            net = MLP(a.shape[1], hidden, 3)
            net.fit(a, yt, cfg.learning_rate, cfg.epochs)
            parts[name].append(net.predict_proba(b))
    rows = np.concatenate([sp.test for sp in splits])
    y_te, idx_te = y[rows], idx[rows]
    proba = {m: np.vstack(v) for m, v in parts.items()}
    metrics: Dict[str, Dict[str, float]] = {}
    for m, p in proba.items():
        pred = np.argmax(p, axis=1)
        row = {
            "accuracy": accuracy(y_te, pred), "log_loss": log_loss(y_te, p),
            "mcc": matthews_corrcoef(y_te, pred), "macro_f1": macro_f1(y_te, pred),
            "trade_share": float(np.mean(pred != 1)),
        }
        if backtest is not None:
            stats = run_backtest(states, idx_te, pred - 1, backtest).stats
            row["net"], row["n_trades"] = stats["net"], stats["n_trades"]
        metrics[m] = row
    return WalkForwardResult(metrics, y_te, proba, idx_te)


def paired_log_loss_ci(
    y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, block: int,
    n_boot: int = 1000, seed: int = 0,
) -> BootstrapCI:
    """Block-bootstrap interval for ``log_loss(a) - log_loss(b)`` on shared rows."""
    rows = np.arange(y.size)
    diff = (np.log(np.clip(proba_b[rows, y], 1e-15, 1.0))
            - np.log(np.clip(proba_a[rows, y], 1e-15, 1.0)))
    return block_bootstrap_ci(lambda r: float(diff[r].mean()), y.size, block, n_boot, seed=seed)


def seed_mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> BootstrapCI:
    """Mean over independent seeds with an iid percentile bootstrap interval."""
    v = np.asarray(values, dtype=float)
    return block_bootstrap_ci(lambda r: float(v[r].mean()), v.size, 1, n_boot, seed=seed)
