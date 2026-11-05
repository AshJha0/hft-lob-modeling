r"""Time-varying Hawkes baseline, fitted jointly with the kernel.

Python-only analysis code, outside the four-language golden contract; numpy only.

A flat ``mu`` on a path whose arrival rate has a daily U shape forces the
kernel to explain the slow swing: nearby events look like they trigger each
other because they share a busy period.  The branching ratio then comes out too
high.  Here ``mu(t)`` enters the same likelihood as the kernel:

    mu(t) = sum_j m_j B_j(t),   m_j >= 0.

Two bases are given, both linear in ``m`` so that the concave inner solver of
:mod:`lobmod.research.kernels` applies unchanged:

* :class:`PiecewiseConstantBaseline`: ``J`` equal bins;
* :class:`LinearSplineBaseline`: hat functions on ``J`` equal intervals
  (``J + 1`` coefficients), a continuous degree-1 spline.  ``m >= 0`` is
  exactly ``mu >= 0`` for this basis.

:func:`select_baseline` picks ``J`` by BIC.  A slow kernel component and a fine
baseline describe nearly the same thing, so a fine baseline can take some
branching away from the kernel; the flat-baseline control in
``research/run_models.py`` measures that cost.  A fit on one path says nothing
about whether a shape repeats from day to day.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .kernels import FlatBaseline, KernelFit, fit_sum_exp


class PiecewiseConstantBaseline:
    """``J`` equal-width bins on ``[0, T]``."""

    def __init__(self, T: float, J: int) -> None:
        if J < 1 or T <= 0.0:
            raise ValueError("need J >= 1 and T > 0")
        self.T, self.J, self.n_coef = float(T), int(J), int(J)
        self.h = self.T / self.J

    def design(self, t: np.ndarray) -> np.ndarray:
        x = np.asarray(t, dtype=float).ravel()
        j = np.minimum((x / self.h).astype(int), self.J - 1)
        out = np.zeros((x.size, self.J))
        out[np.arange(x.size), j] = 1.0
        return out

    def cumulative(self, t: np.ndarray) -> np.ndarray:
        x = np.asarray(t, dtype=float).ravel()
        return np.clip(x[:, None] - np.arange(self.J) * self.h, 0.0, self.h)


class LinearSplineBaseline:
    """Hat functions on knots ``k T / J``, ``J + 1`` coefficients."""

    def __init__(self, T: float, J: int) -> None:
        if J < 1 or T <= 0.0:
            raise ValueError("need J >= 1 and T > 0")
        self.T, self.J, self.n_coef = float(T), int(J), int(J) + 1
        self.h = self.T / self.J

    def design(self, t: np.ndarray) -> np.ndarray:
        x = np.asarray(t, dtype=float).ravel()
        knots = np.arange(self.n_coef) * self.h
        return np.maximum(1.0 - np.abs(x[:, None] - knots) / self.h, 0.0)

    def cumulative(self, t: np.ndarray) -> np.ndarray:
        x = np.asarray(t, dtype=float).ravel()
        u = (x[:, None] - np.arange(self.n_coef) * self.h) / self.h  # signed distance from knot
        left = 0.5 * np.clip(u + 1.0, 0.0, 1.0) ** 2  # rising half, reaches 1/2 at the knot
        v = np.clip(u, 0.0, 1.0)
        out = self.h * (left + v - 0.5 * v * v)
        out[:, 0] -= 0.5 * self.h  # the first hat has no rising half inside [0, T]
        return out


def baseline_curve(baseline: Any, coef: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``mu(t)`` at the given times."""
    return baseline.design(t) @ np.asarray(coef, dtype=float)


def fit_seasonal(times: Sequence[float], T: float, baseline: Any, K: int = 1) -> KernelFit:
    """Joint MLE of the baseline and a ``K``-exponential kernel."""
    return fit_sum_exp(times, T, K, baseline)


def select_baseline(
    times: Sequence[float], T: float, kind: str = "spline",
    grid: Sequence[int] = (1, 2, 4, 8, 16), K: int = 1,
) -> Tuple[KernelFit, Any, List[Dict[str, float]]]:
    """Fit a flat baseline and each ``J`` in ``grid``; return the lowest BIC.

    Row ``J = 0`` is the flat baseline.  Returns ``(best fit, its baseline,
    one row per candidate)``.

    Raises:
        ValueError: unknown ``kind``.
    """
    if kind not in ("spline", "piecewise"):
        raise ValueError(f"kind must be 'spline' or 'piecewise', got {kind}")
    cls = LinearSplineBaseline if kind == "spline" else PiecewiseConstantBaseline
    candidates = [(0, FlatBaseline(T))]
    candidates += [(J, cls(T, J)) for J in grid if not (J == 1 and kind == "piecewise")]
    best, best_base, rows = None, None, []
    for J, base in candidates:
        f = fit_sum_exp(times, T, K, base, n_starts=1)
        rows.append({"J": float(J), "bic": f.bic, "aic": f.aic, "branching": f.branching})
        if best is None or f.bic < best.bic:
            best, best_base = f, base
    return best, best_base, rows
