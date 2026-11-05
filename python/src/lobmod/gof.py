r"""Goodness of fit: Kolmogorov-Smirnov test of Exp(1) residuals.

By the time-rescaling theorem (see :mod:`lobmod.hawkes`), a correctly
specified Hawkes model turns its inter-arrival compensator increments into
iid Exp(1) draws.  The p-value below assumes the parameters were fixed in
advance; with parameters fitted to the same data it is far too large and the
test has little power (see LEARN section 3.7 and the bootstrap test in
:mod:`lobmod.research`).  This module tests that natively:

* KS statistic against the Exp(1) CDF ``F(x) = 1 - exp(-x)``:

      D_n = max_i max( i/n - F(x_(i)),  F(x_(i)) - (i-1)/n )

  over the sorted sample ``x_(1) <= ... <= x_(n)``.

* Asymptotic p-value with the Stephens small-sample correction (Numerical
  Recipes pinned form): with

      lambda = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * D_n,

  the p-value is the Kolmogorov tail series

      Q(lambda) = 2 * sum_{j=1}^{100} (-1)^{j-1} exp(-2 j^2 lambda^2)     (lambda >= 0.1)

  clipped to ``[0, 1]``.  That alternating series converges slowly as
  lambda -> 0 (100 terms are off by 0.13 at lambda = 0.01), so for
  ``lambda < 0.1`` the pinned evaluation switches to the dual (Jacobi theta)
  form, Numerical Recipes eq. 14.3.20,

      Q(lambda) = 1 - (sqrt(2 pi) / lambda) * sum_{j=1}^{100} exp(-(2j-1)^2 pi^2 / (8 lambda^2)),

  whose terms vanish for small lambda.  Both branches use exactly 100 terms
  (pinned).  With very small n the asymptotic p-value is only indicative —
  callers should treat it as such (documented, not an error).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from ._validate import as_positive_int

#: pinned switch point between the two Kolmogorov series
_LAMBDA_SWITCH = 0.1
_N_TERMS = 100


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

    Pinned formula: ``lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * d``, then the
    100-term alternating tail series for ``lambda >= 0.1`` or the 100-term
    dual form for ``lambda < 0.1`` (module docstring), clipped to [0, 1].

    Raises:
        ValueError: if ``d`` is not in [0, 1] or ``n`` is not positive.
    """
    if isinstance(d, bool) or not (math.isfinite(d) and 0.0 <= d <= 1.0):
        raise ValueError(f"d must be in [0, 1], got {d}")
    n = as_positive_int("n", n)
    if d == 0.0:
        return 1.0
    sqrt_n = math.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d
    return _kolmogorov_q(lam)


def _kolmogorov_q(lam: float) -> float:
    """Kolmogorov tail probability ``Q(lambda)`` (pinned two-branch evaluation)."""
    if lam < _LAMBDA_SWITCH:
        total = 0.0
        for j in range(1, _N_TERMS + 1):
            m = 2 * j - 1
            total += math.exp(-(m * m) * math.pi * math.pi / (8.0 * lam * lam))
        p = 1.0 - (math.sqrt(2.0 * math.pi) / lam) * total
    else:
        total = 0.0
        for j in range(1, _N_TERMS + 1):
            term = math.exp(-2.0 * j * j * lam * lam)
            total += term if j % 2 == 1 else -term
        p = 2.0 * total
    return min(1.0, max(0.0, p))


def ks_test_exp(sample: Sequence[float]) -> Tuple[float, float]:
    """KS test of ``sample`` against Exp(1): returns ``(statistic, p_value)``."""
    d = ks_statistic_exp(sample)
    x = np.asarray(sample, dtype=float)
    return d, ks_pvalue(d, int(x.size))
