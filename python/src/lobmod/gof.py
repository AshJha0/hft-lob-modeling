r"""Goodness of fit: Kolmogorov-Smirnov test of Exp(1) residuals.

By the time-rescaling theorem (see :mod:`lobmod.hawkes`), a correctly
specified Hawkes model turns its inter-arrival compensator increments into
iid Exp(1) draws.  This module tests that natively:

* KS statistic against the Exp(1) CDF ``F(x) = 1 - exp(-x)``:

      D_n = max_i max( i/n - F(x_(i)),  F(x_(i)) - (i-1)/n )

  over the sorted sample ``x_(1) <= ... <= x_(n)``.

* Asymptotic p-value with the Stephens small-sample correction (Numerical
  Recipes pinned form): with

      lambda = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * D_n,

  the p-value is the Kolmogorov tail series

      Q(lambda) = 2 * sum_{j=1}^{100} (-1)^{j-1} exp(-2 j^2 lambda^2),

  clipped to ``[0, 1]``.  The series is pinned at exactly 100 terms (it
  converges to machine precision long before that for any lambda of
  interest).  With very small n the asymptotic p-value is only indicative —
  callers should treat it as such (documented, not an error).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np


def ks_statistic_exp(sample: Sequence[float]) -> float:
    """KS statistic of ``sample`` against the Exp(1) CDF.

    Raises:
        ValueError: empty sample, non-finite or negative values.
    """
    x = np.asarray(sample, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("sample must be a non-empty 1-D array")
    if not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("sample values must be finite and non-negative")
    xs = np.sort(x)
    n = xs.size
    cdf = -np.expm1(-xs)  # 1 - exp(-x), accurately
    i = np.arange(1, n + 1, dtype=float)
    d_plus = np.max(i / n - cdf)
    d_minus = np.max(cdf - (i - 1.0) / n)
    return float(max(d_plus, d_minus))


def ks_pvalue(d: float, n: int) -> float:
    """Asymptotic Kolmogorov p-value for statistic ``d`` on ``n`` points.

    Pinned formula: ``lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * d`` and the
    100-term alternating tail series (module docstring), clipped to [0, 1].

    Raises:
        ValueError: if ``d`` is not in [0, 1] or ``n`` is not positive.
    """
    if not (math.isfinite(d) and 0.0 <= d <= 1.0):
        raise ValueError(f"d must be in [0, 1], got {d}")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    if d == 0.0:
        return 1.0
    sqrt_n = math.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d
    total = 0.0
    for j in range(1, 101):
        term = math.exp(-2.0 * j * j * lam * lam)
        total += term if j % 2 == 1 else -term
    p = 2.0 * total
    return min(1.0, max(0.0, p))


def ks_test_exp(sample: Sequence[float]) -> Tuple[float, float]:
    """KS test of ``sample`` against Exp(1): returns ``(statistic, p_value)``."""
    d = ks_statistic_exp(sample)
    x = np.asarray(sample, dtype=float)
    return d, ks_pvalue(d, int(x.size))
