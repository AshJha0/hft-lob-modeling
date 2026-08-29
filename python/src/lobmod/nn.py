r"""Feed-forward neural network from scratch — fully deterministic, no RNG.

Architecture (pinned for the prediction task): input -> 16 ReLU -> 8 ReLU ->
3-way softmax.  The same class with ``hidden=()`` *is* the multinomial
logistic-regression baseline (input -> softmax).

Determinism
-----------
Cross-language golden reproducibility requires that training involve no
randomness at all:

* **Pinned init** (no RNG): for layer ``l`` (0-based) with shape
  ``(n_in, n_out)``,

      W_l[i, j] = 0.5 * sqrt(2 / n_in) * sin(l * 10000 + i * n_out + j + 1),
      b_l = 0.

  The ``sin`` of distinct integers gives well-spread values in [-1, 1]; the
  ``sqrt(2/n_in)`` factor is He-style scaling for ReLU units.
* **Full-batch gradient descent**: one gradient step per epoch on the entire
  training set (no shuffling, no minibatches), pinned learning rate and epoch
  count.
* **Pinned standardization**: per-feature ``(x - mean) / std`` with training
  -set population moments (``std`` divisor ``n``); ``std < 1e-12`` is
  replaced by 1.

Forward / backward equations
----------------------------
Forward, for input matrix ``X`` (n x d), labels ``y``::

    A_0 = X
    Z_l = A_{l-1} W_l + b_l ;  A_l = relu(Z_l)      for hidden layers
    Z_L = A_{L-1} W_L + b_L ;  P = softmax(Z_L)     (row-wise, max-shifted)
    L(theta) = -(1/n) * sum_i log P[i, y_i]         (mean cross-entropy)

Backward (derivation: for a softmax + cross-entropy head,
``dL/dZ_L = (P - Y) / n`` with ``Y`` one-hot; then standard chain rule)::

    G_L   = (P - Y) / n
    dW_l  = A_{l-1}^T G_l ;   db_l = sum_rows(G_l)
    G_{l-1} = (G_l W_l^T) .* 1{Z_{l-1} > 0}         (ReLU gate)

Update: ``W_l <- W_l - lr * dW_l``, ``b_l <- b_l - lr * db_l``.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

_STD_FLOOR = 1e-12
_LOG_FLOOR = 1e-300


def pinned_init(n_in: int, n_out: int, layer_index: int) -> np.ndarray:
    """Deterministic sin-based weight matrix (module docstring formula)."""
    if min(n_in, n_out) <= 0 or layer_index < 0:
        raise ValueError("n_in, n_out must be positive and layer_index >= 0")
    i = np.arange(n_in)[:, None]
    j = np.arange(n_out)[None, :]
    scale = 0.5 * math.sqrt(2.0 / n_in)
    return scale * np.sin(layer_index * 10000 + i * n_out + j + 1)


class Standardizer:
    """Pinned per-feature standardization fitted on the training split only."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        """Compute population mean/std per column; std below 1e-12 becomes 1.

        Raises:
            ValueError: empty or non-finite input.
        """
        X = _check_matrix(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)  # population (divisor n) — pinned
        self.std = np.where(self.std < _STD_FLOOR, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply ``(x - mean) / std``; requires :meth:`fit` first."""
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer.transform called before fit")
        X = _check_matrix(X)
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected {self.mean.shape[0]} columns, got {X.shape[1]}")
        return (X - self.mean) / self.std


def _check_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty 2-D array")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    return X


def _check_labels(y: np.ndarray, n_rows: int, n_classes: int) -> np.ndarray:
    y = np.asarray(y)
    if y.shape != (n_rows,):
        raise ValueError(f"y must have shape ({n_rows},), got {y.shape}")
    y = y.astype(int)
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError(f"labels must be in [0, {n_classes})")
    return y


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the pinned max-shift for overflow safety."""
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class MLP:
    """Deterministic MLP classifier (see module docstring).

    Args:
        n_inputs: input dimension.
        hidden: hidden layer sizes; empty tuple = logistic regression.
        n_classes: number of output classes (>= 2).
    """

    def __init__(self, n_inputs: int, hidden: Sequence[int], n_classes: int) -> None:
        if n_inputs <= 0 or n_classes < 2:
            raise ValueError("n_inputs must be positive and n_classes >= 2")
        if any((not isinstance(h, int)) or isinstance(h, bool) or h <= 0 for h in hidden):
            raise ValueError(f"hidden sizes must be positive integers, got {hidden}")
        sizes = [int(n_inputs), *[int(h) for h in hidden], int(n_classes)]
        self.sizes = sizes
        self.weights: List[np.ndarray] = [
            pinned_init(sizes[l], sizes[l + 1], l) for l in range(len(sizes) - 1)
        ]
        self.biases: List[np.ndarray] = [
            np.zeros(sizes[l + 1]) for l in range(len(sizes) - 1)
        ]

    # ---------------------------------------------------------------- forward

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Return (pre-activations Z_l, activations A_l, probabilities P)."""
        zs: List[np.ndarray] = []
        activations: List[np.ndarray] = [X]
        a = X
        n_layers = len(self.weights)
        for l in range(n_layers):
            z = a @ self.weights[l] + self.biases[l]
            zs.append(z)
            if l < n_layers - 1:
                a = np.maximum(z, 0.0)  # ReLU
                activations.append(a)
        probs = softmax(zs[-1])
        return zs, activations, probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, shape (n, n_classes); rows sum to 1."""
        X = _check_matrix(X)
        if X.shape[1] != self.sizes[0]:
            raise ValueError(f"expected {self.sizes[0]} features, got {X.shape[1]}")
        return self._forward(X)[2]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax class per row (ties resolve to the lowest class index)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy ``-(1/n) sum log P[i, y_i]`` (probability floored
        at 1e-300 before the log so a confident wrong answer cannot produce
        -inf)."""
        X = _check_matrix(X)
        y = _check_labels(y, X.shape[0], self.sizes[-1])
        probs = self._forward(X)[2]
        picked = np.maximum(probs[np.arange(X.shape[0]), y], _LOG_FLOOR)
        return float(-np.mean(np.log(picked)))

    # --------------------------------------------------------------- backward

    def loss_and_grads(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        """Loss plus gradients d(loss)/dW_l, d(loss)/db_l via backprop.

        Implements exactly the equations in the module docstring.
        """
        X = _check_matrix(X)
        y = _check_labels(y, X.shape[0], self.sizes[-1])
        n = X.shape[0]
        zs, activations, probs = self._forward(X)
        picked = np.maximum(probs[np.arange(n), y], _LOG_FLOOR)
        loss = float(-np.mean(np.log(picked)))

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), y] = 1.0
        g = (probs - one_hot) / n  # dL/dZ_L
        grads_w: List[np.ndarray] = [np.empty(0)] * len(self.weights)
        grads_b: List[np.ndarray] = [np.empty(0)] * len(self.weights)
        for l in range(len(self.weights) - 1, -1, -1):
            grads_w[l] = activations[l].T @ g
            grads_b[l] = np.sum(g, axis=0)
            if l > 0:
                g = (g @ self.weights[l].T) * (zs[l - 1] > 0.0)  # ReLU gate
        return loss, grads_w, grads_b

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int) -> List[float]:
        """Full-batch gradient descent (pinned schedule).

        One gradient step per epoch on the whole training set.  Returns the
        loss history *after* each epoch's update (``history[0]`` is the
        golden "loss after epoch 1").

        Raises:
            ValueError: invalid inputs or hyperparameters.
        """
        if not (math.isfinite(learning_rate) and learning_rate > 0.0):
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs}")
        X = _check_matrix(X)
        y = _check_labels(y, X.shape[0], self.sizes[-1])
        history: List[float] = []
        for _ in range(epochs):
            _, grads_w, grads_b = self.loss_and_grads(X, y)
            for l in range(len(self.weights)):
                self.weights[l] = self.weights[l] - learning_rate * grads_w[l]
                self.biases[l] = self.biases[l] - learning_rate * grads_b[l]
            history.append(self.loss(X, y))
        return history


# ------------------------------------------------------------------ metrics

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of matching labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true and y_pred must be equal-length non-empty 1-D arrays")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(n_classes x n_classes) counts, rows = true class, cols = predicted."""
    y_true = _check_labels(np.asarray(y_true), np.asarray(y_true).shape[0], n_classes)
    y_pred = _check_labels(np.asarray(y_pred), np.asarray(y_true).shape[0], n_classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def majority_class(y: np.ndarray) -> int:
    """Most frequent label; ties resolve to the smallest label (pinned)."""
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y must be a non-empty 1-D array")
    values, counts = np.unique(y.astype(int), return_counts=True)
    return int(values[np.argmax(counts)])


def signal_pnl(
    y_pred: np.ndarray, fwd_ret: np.ndarray, spread: np.ndarray
) -> Tuple[float, float, int]:
    """PnL of trading the predicted direction with half-spread costs.

    Position per row: +1 for class 2 (up), -1 for class 0 (down), 0 for flat.
    Gross PnL adds ``pos * fwd_ret`` (mid-to-mid, ticks); net PnL subtracts a
    half-spread cost ``spread / 2`` per round-trip signal — the honest floor
    for crossing the spread on entry.  Returns ``(gross, net, n_trades)``.

    Raises:
        ValueError: mismatched or non-finite inputs.
    """
    y_pred = np.asarray(y_pred).astype(int)
    fwd_ret = np.asarray(fwd_ret, dtype=float)
    spread = np.asarray(spread, dtype=float)
    if not (y_pred.shape == fwd_ret.shape == spread.shape) or y_pred.ndim != 1:
        raise ValueError("y_pred, fwd_ret, spread must be equal-length 1-D arrays")
    if y_pred.size == 0:
        raise ValueError("empty prediction array")
    if not (np.all(np.isfinite(fwd_ret)) and np.all(np.isfinite(spread))):
        raise ValueError("fwd_ret and spread must be finite")
    pos = y_pred - 1  # {0,1,2} -> {-1,0,+1}
    gross = float(np.sum(pos * fwd_ret))
    net = float(gross - np.sum(np.abs(pos) * spread * 0.5))
    return gross, net, int(np.sum(pos != 0))
