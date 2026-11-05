r"""Inference for the Hawkes fits: standard errors, LR tests, valid GoF.

Python-only analysis code, outside the four-language golden contract.  numpy
``default_rng`` is allowed here (see :mod:`lobmod.research`).

Standard errors
---------------
The log-likelihood is sharply curved around a well-determined parameter and
flat around a poorly determined one.  The inverse of that curvature is the
asymptotic covariance:

    Cov(theta_hat) ~= H^{-1},   H = -d^2 ll / d theta d theta'  at theta_hat.

``H`` comes from central differences with steps proportional to each
parameter.  Wald intervals are ``theta_hat +/- 1.96 se``; they mean little for
a parameter close to zero, where the normal approximation fails.

Likelihood-ratio tests
----------------------
``LR = 2 (ll_full - ll_restricted)``.  Both nulls here sit on the boundary of
the parameter space (``alpha >= 0``), so the textbook chi-square is wrong:

* Hawkes vs Poisson: ``alpha = 0`` is on the boundary **and** ``beta`` is not
  identified under the null (Davies' problem).  The mixture
  ``0.5 chi2_0 + 0.5 chi2_1`` is reported as indicative only and can be far too
  small: on a true Poisson path a free ``beta`` latches on to one close pair of
  events.  The parametric bootstrap p-value (``n_boot > 0``) is the one to quote.
* No cross-excitation: two parameters on the boundary give the mixture
  ``0.25 chi2_0 + 0.5 chi2_1 + 0.25 chi2_2`` (Self & Liang 1987).  Plain
  ``chi2_2`` is conservative and is reported next to it.

Goodness of fit with estimated parameters
-----------------------------------------
Residuals computed with *fitted* parameters sit closer to Exp(1) than the true
ones would, so the Kolmogorov table in :mod:`lobmod.gof` gives p-values that
are far too large (Lilliefors' effect).  The parametric bootstrap repeats the
whole procedure under the fitted model:

    p = (1 + #{b : D*_b >= D_obs}) / (B + 1),

where ``D*_b`` is the KS statistic after simulating from the fit, refitting
and recomputing residuals.  Its validity is asymptotic, because the reference
law is taken at ``theta_hat`` rather than the unknown true parameter.

Serial dependence
-----------------
KS looks at the marginal law only.  A wrong kernel often leaves the marginal
near Exp(1) but correlates successive residuals, so also use the Ljung-Box
statistic

    Q = n (n + 2) sum_{k=1}^{h} r_k^2 / (n - k)   ~  chi2_h.

No degrees of freedom are removed for the fitted parameters; treat its p-value
as approximate.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ..gof import ks_statistic_exp, ks_test_exp
from ..hawkes import (
    HawkesBivFit,
    HawkesUniFit,
    biv_loglik,
    fit_biv,
    fit_uni,
    simulate_uni,
    uni_loglik,
    uni_residuals,
)
from ..neldermead import nelder_mead

_Z95 = 1.959963984540054
_PENALTY = 1e10


def _gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma ``Q(a, x)`` (series / continued fraction)."""
    if x <= 0.0:
        return 1.0
    log_pre = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(log_pre))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(1.0, h * math.exp(log_pre))


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function ``P(chi2_df > x) = Q(df/2, x/2)``."""
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    return _gamma_q(0.5 * df, 0.5 * x)


def numeric_hessian(
    f: Callable[[np.ndarray], float], x: Sequence[float], rel_step: float = 1e-3
) -> np.ndarray:
    r"""Central-difference Hessian of ``f`` at ``x``.

    With ``h_i = rel_step * max(|x_i|, 1e-8)``,

        H_ij ~= [f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)] / (4 h_i h_j).

    Steps scale with ``x`` so positive parameters stay positive.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    h = rel_step * np.maximum(np.abs(x), 1e-8)
    steps = np.diag(h)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b = steps[i], steps[j]
            val = (f(x + a + b) - f(x + a - b) - f(x - a + b) + f(x - a - b)) / (4.0 * h[i] * h[j])
            out[i, j] = out[j, i] = val
    return out


@dataclass(frozen=True)
class HawkesInference:
    """Point estimates with Wald standard errors.

    Attributes:
        names: parameter names, in the order of every array below.
        estimate: fitted values.
        se: standard errors, NaN when ``information_ok`` is False.
        ci_low, ci_high: 95% Wald limits.
        cov: covariance matrix (NaN-filled when not available).
        information_ok: the observed information was positive definite.
        branching: ``alpha/beta`` (univariate) or the spectral radius.
        branching_se: delta-method se for the univariate ratio, else NaN.
    """

    names: Tuple[str, ...]
    estimate: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    cov: np.ndarray
    information_ok: bool
    branching: float
    branching_se: float

    def t_stat(self, name: str) -> float:
        """``estimate / se`` for one parameter (not normal near a boundary)."""
        i = self.names.index(name)
        return float(self.estimate[i] / self.se[i])


def _wald(names, p, neg_ll, branching, grad=None) -> HawkesInference:
    def guarded(v: np.ndarray) -> float:
        # an estimate on the boundary pushes the stencil out of the domain;
        # no curvature exists there, so the se is reported as NaN
        try:
            return neg_ll(v)
        except ValueError:
            return math.nan

    info = numeric_hessian(guarded, p)
    n = p.size
    ok = bool(np.all(np.isfinite(info)) and np.all(np.linalg.eigvalsh(info) > 0.0))
    cov = np.linalg.inv(info) if ok else np.full((n, n), np.nan)
    se = np.sqrt(np.diag(cov))
    bse = float(math.sqrt(grad @ cov @ grad)) if ok and grad is not None else math.nan
    return HawkesInference(
        names=tuple(names), estimate=p, se=se, ci_low=p - _Z95 * se, ci_high=p + _Z95 * se,
        cov=cov, information_ok=ok, branching=float(branching), branching_se=bse,
    )


def inference_uni(
    times: Sequence[float], T: float, fit: Optional[HawkesUniFit] = None
) -> HawkesInference:
    """Wald inference for ``(mu, alpha, beta)``; fits first if ``fit`` is None."""
    t = np.asarray(times, dtype=float)
    f = fit if fit is not None else fit_uni(t, T)
    p = np.array([f.mu, f.alpha, f.beta])
    grad = np.array([0.0, 1.0 / f.beta, -f.alpha / f.beta ** 2])
    return _wald(("mu", "alpha", "beta"), p, lambda v: -uni_loglik(t, T, *v), f.branching, grad)


def inference_biv(
    times: Sequence[float], types: Sequence[int], T: float, fit: Optional[HawkesBivFit] = None
) -> HawkesInference:
    """Wald inference for ``(mu0, mu1, a00, a01, a10, a11, beta)``."""
    t = np.asarray(times, dtype=float)
    k = np.asarray(types)
    f = fit if fit is not None else fit_biv(t, k, T)
    p = np.r_[f.mu, f.alpha.ravel(), f.beta]

    def neg_ll(v: np.ndarray) -> float:
        return -biv_loglik(t, k, T, v[:2], [[v[2], v[3]], [v[4], v[5]]], v[6])

    names = ("mu0", "mu1", "a00", "a01", "a10", "a11", "beta")
    return _wald(names, p, neg_ll, f.spectral_radius)


@dataclass(frozen=True)
class LrTest:
    """Likelihood-ratio test result.

    Attributes:
        statistic: ``2 (ll_full - ll_restricted)``, floored at 0.
        ll_full, ll_restricted: the two maximised log-likelihoods.
        p_mixture: boundary-corrected chi-square mixture p-value.
        p_chi2: plain chi-square p-value at 2 df (conservative for the
            no-cross test; only a reference for the Poisson test).
        p_bootstrap: parametric bootstrap p-value, NaN if not run.
    """

    statistic: float
    ll_full: float
    ll_restricted: float
    p_mixture: float
    p_chi2: float
    p_bootstrap: float = math.nan


def _lr_poisson_stat(t: np.ndarray, T: float) -> Tuple[float, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ll1 = fit_uni(t, T).loglik
    ll0 = t.size * math.log(t.size / T) - t.size
    return max(0.0, 2.0 * (ll1 - ll0)), ll1, ll0


def lr_test_poisson(
    times: Sequence[float], T: float, n_boot: int = 0, seed: int = 0
) -> LrTest:
    r"""Test Hawkes against a homogeneous Poisson process.

    Restricted log-likelihood in closed form: ``N log(N/T) - N``.  With
    ``n_boot > 0``, Poisson paths at rate ``N/T`` are simulated and refitted,
    giving ``p = (1 + #{LR*_b >= LR}) / (B + 1)``, which handles the
    unidentified ``beta`` that the mixture cannot.
    """
    t = np.asarray(times, dtype=float)
    stat, ll1, ll0 = _lr_poisson_stat(t, T)
    p_mix = 1.0 if stat == 0.0 else 0.5 * chi2_sf(stat, 1)
    p_boot = math.nan
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        hits = 0
        done = 0
        for _ in range(n_boot):
            m = int(rng.poisson(t.size))
            if m < 2:
                continue
            x = np.sort(rng.uniform(0.0, T, m))
            hits += _lr_poisson_stat(x, T)[0] >= stat
            done += 1
        p_boot = (1.0 + hits) / (done + 1.0)
    return LrTest(stat, ll1, ll0, p_mix, chi2_sf(stat, 2), p_boot)


def lr_test_no_cross(
    times: Sequence[float], types: Sequence[int], T: float,
    fit: Optional[HawkesBivFit] = None,
) -> LrTest:
    r"""Test ``a01 = a10 = 0`` in the bivariate model (shared ``beta`` kept).

    The restricted model is fitted with the pinned Nelder-Mead on log
    parameters, restarted once, starting from the full fit.  p-values are
    the boundary mixture and the conservative ``chi2_2`` (module docstring).
    """
    t = np.asarray(times, dtype=float)
    k = np.asarray(types)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = fit if fit is not None else fit_biv(t, k, T)

    def objective(x: Sequence[float]) -> float:
        try:
            v = [math.exp(c) for c in x]
        except OverflowError:
            return _PENALTY
        if v[0] <= 0.0 or v[1] <= 0.0 or v[4] <= 0.0 or not all(map(math.isfinite, v)):
            return _PENALTY
        ll = biv_loglik(t, k, T, v[:2], [[v[2], 0.0], [0.0, v[3]]], v[4])
        return -ll if math.isfinite(ll) else _PENALTY

    x0 = np.log([f.mu[0], f.mu[1], f.alpha[0, 0], f.alpha[1, 1], f.beta])
    res = nelder_mead(objective, nelder_mead(objective, x0).x)
    ll0 = -res.fval
    stat = max(0.0, 2.0 * (f.loglik - ll0))
    p2 = chi2_sf(stat, 2)
    p_mix = 1.0 if stat == 0.0 else 0.5 * chi2_sf(stat, 1) + 0.25 * p2
    return LrTest(stat, f.loglik, ll0, p_mix, p2)


@dataclass(frozen=True)
class BootstrapGof:
    """Parametric bootstrap KS result.

    Attributes:
        statistic: observed KS distance.
        p_value: bootstrap p-value (the valid one).
        p_naive: table p-value ignoring estimation (too large; for contrast).
        n_boot: replicates used.
        n_failed: replicates dropped (simulated path too short to refit).
        null_stats: the bootstrap statistics.
    """

    statistic: float
    p_value: float
    p_naive: float
    n_boot: int
    n_failed: int
    null_stats: np.ndarray


def bootstrap_ks_uni(
    times: Sequence[float], T: float, n_boot: int = 99, seed: int = 1,
    fit: Optional[HawkesUniFit] = None,
) -> BootstrapGof:
    """Parametric bootstrap KS test of a univariate fit (module docstring).

    Replicate ``b`` uses pinned simulator seed ``seed + b``.

    Raises:
        ValueError: if the fit is non-stationary, so cannot be simulated.
    """
    t = np.asarray(times, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = fit if fit is not None else fit_uni(t, T)
        if f.branching >= 1.0:
            raise ValueError(f"fit is non-stationary (branching {f.branching}); cannot bootstrap")
        d_obs, p_naive = ks_test_exp(uni_residuals(t, T, f.mu, f.alpha, f.beta))
        stats = []
        failed = 0
        for b in range(n_boot):
            x = simulate_uni(f.mu, f.alpha, f.beta, T, seed + b)
            if x.size < 5:
                failed += 1
                continue
            g = fit_uni(x, T)
            stats.append(ks_statistic_exp(uni_residuals(x, T, g.mu, g.alpha, g.beta)))
    arr = np.asarray(stats)
    p = (1.0 + float(np.sum(arr >= d_obs))) / (arr.size + 1.0)
    return BootstrapGof(d_obs, p, p_naive, int(arr.size), failed, arr)


def residual_acf(x: Sequence[float], lags: int) -> np.ndarray:
    r"""Sample autocorrelations ``r_1..r_lags``.

        r_k = sum_{i>k} (x_i - m)(x_{i-k} - m) / sum_i (x_i - m)^2.

    Raises:
        ValueError: ``lags`` not in ``[1, n-1]`` or a constant series.
    """
    a = np.asarray(x, dtype=float)
    if not 1 <= lags < a.size:
        raise ValueError(f"lags must be in [1, {a.size - 1}], got {lags}")
    a = a - a.mean()
    denom = float(a @ a)
    if denom <= 0.0:
        raise ValueError("series is constant")
    return np.array([float(a[k:] @ a[:-k]) / denom for k in range(1, lags + 1)])


@dataclass(frozen=True)
class LjungBox:
    """Ljung-Box result: statistic ``Q``, p-value, and the ``r_k`` used."""

    statistic: float
    p_value: float
    acf: np.ndarray


def ljung_box(x: Sequence[float], lags: int = 10) -> LjungBox:
    """Ljung-Box portmanteau test on ``x`` (module docstring)."""
    r = residual_acf(x, lags)
    n = len(x)
    k = np.arange(1, lags + 1)
    q = float(n * (n + 2.0) * np.sum(r ** 2 / (n - k)))
    return LjungBox(q, chi2_sf(q, lags), r)
