r"""Hawkes processes with exponential kernels: likelihood, MLE, simulation.

Univariate model
----------------
Conditional intensity

    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i)),

with background rate ``mu > 0``, jump size ``alpha >= 0`` and decay
``beta > 0``.  The branching ratio ``n = alpha / beta`` is the expected number
of children per event; the process is stationary iff ``n < 1``.

Recursive log-likelihood (the pinned O(N) form)
-----------------------------------------------
The point-process log-likelihood on ``[0, T]`` is

    ll = sum_i log lambda(t_i) - int_0^T lambda(s) ds.

Define ``R_i = sum_{j < i} exp(-beta (t_i - t_j))``.  Splitting the sum at
``t_{i-1}`` gives the recursion (Ozaki 1979)

    R_1 = 0,   R_i = exp(-beta (t_i - t_{i-1})) * (1 + R_{i-1}),

because every term for ``j < i-1`` decays by the common factor
``exp(-beta (t_i - t_{i-1}))`` and the ``j = i-1`` term contributes 1 before
that decay.  The compensator integrates in closed form,

    int_0^T lambda = mu T + (alpha/beta) * sum_i (1 - exp(-beta (T - t_i))),

so

    ll = sum_i log(mu + alpha R_i) - mu T
         + (alpha/beta) * sum_i expm1(-beta (T - t_i)),

computed in O(N) (``expm1`` pinned for accuracy in the tail terms).

Bivariate model
---------------
Two event types with full 2x2 excitation and a *scalar shared* ``beta``
(pinned parameterization):

    lambda_m(t) = mu_m + sum_n alpha_{mn} sum_{t_i^n < t} exp(-beta (t - t_i^n)).

The same recursion runs on the merged stream: keep decayed sums
``S_n(t) = sum_{t_i^n < t} exp(-beta (t - t_i^n))``, decay both by
``exp(-beta dt)`` between events and add 1 to ``S_{type}`` after each event.

Time-rescaling theorem
----------------------
If events follow a point process with compensator ``Lambda(t) = int_0^t
lambda``, the rescaled times ``Lambda(t_i)`` form a unit-rate Poisson
process, so the increments ``tau_i = Lambda(t_i) - Lambda(t_{i-1})`` are iid
Exp(1).  Goodness of fit therefore reduces to a KS test of the residuals
``tau_i`` against Exp(1) (see :mod:`lobmod.gof`).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .neldermead import NMResult, nelder_mead
from .rng import Lcg64

_PENALTY = 1e10


# --------------------------------------------------------------- validation

def _check_times(times: np.ndarray, T: float) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a non-empty 1-D array")
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")
    if t[0] <= 0.0:
        raise ValueError("times must be strictly positive")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError(f"T must be positive and finite, got {T}")
    if t[-1] > T:
        raise ValueError(f"last event time {t[-1]} exceeds horizon T={T}")
    return t


def _check_uni_params(mu: float, alpha: float, beta: float) -> None:
    if not (math.isfinite(mu) and mu > 0.0):
        raise ValueError(f"mu must be positive and finite, got {mu}")
    if not (math.isfinite(alpha) and alpha >= 0.0):
        raise ValueError(f"alpha must be non-negative and finite, got {alpha}")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError(f"beta must be positive and finite, got {beta}")


def _check_biv_params(mu: Sequence[float], alpha: Sequence[Sequence[float]], beta: float) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mu, dtype=float)
    a = np.asarray(alpha, dtype=float)
    if m.shape != (2,) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("mu must be two positive finite values")
    if a.shape != (2, 2) or not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("alpha must be a 2x2 non-negative finite matrix")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError(f"beta must be positive and finite, got {beta}")
    return m, a


def _check_types(types: np.ndarray, n: int) -> np.ndarray:
    k = np.asarray(types)
    if k.shape != (n,):
        raise ValueError("types must have the same length as times")
    k = k.astype(int)
    if np.any((k != 0) & (k != 1)):
        raise ValueError("types must be 0 or 1")
    return k


# ------------------------------------------------------------ log-likelihood

def uni_loglik(times: Sequence[float], T: float, mu: float, alpha: float, beta: float) -> float:
    """Exact univariate log-likelihood via the pinned O(N) recursion."""
    t = _check_times(np.asarray(times, dtype=float), T)
    _check_uni_params(mu, alpha, beta)
    decay = np.exp(-beta * np.diff(t))
    ll = math.log(mu)  # R_1 = 0
    r = 0.0
    for w in decay:
        r = w * (1.0 + r)
        ll += math.log(mu + alpha * r)
    ll -= mu * T
    ll += (alpha / beta) * float(np.sum(np.expm1(-beta * (T - t))))
    return ll


def uni_loglik_brute(times: Sequence[float], T: float, mu: float, alpha: float, beta: float) -> float:
    """Reference O(N^2) log-likelihood (direct double sum).

    Used by tests and by data generation to validate the recursion; not for
    production use.
    """
    t = _check_times(np.asarray(times, dtype=float), T)
    _check_uni_params(mu, alpha, beta)
    ll = 0.0
    for i in range(t.size):
        excitation = float(np.sum(np.exp(-beta * (t[i] - t[:i]))))
        ll += math.log(mu + alpha * excitation)
    ll -= mu * T
    ll += (alpha / beta) * float(np.sum(np.expm1(-beta * (T - t))))
    return ll


def biv_loglik(
    times: Sequence[float],
    types: Sequence[int],
    T: float,
    mu: Sequence[float],
    alpha: Sequence[Sequence[float]],
    beta: float,
) -> float:
    """Bivariate log-likelihood (full 2x2 alpha, scalar shared beta), O(N)."""
    t = _check_times(np.asarray(times, dtype=float), T)
    k = _check_types(np.asarray(types), t.size)
    m, a = _check_biv_params(mu, alpha, beta)
    s0 = s1 = 0.0
    prev = 0.0
    ll = 0.0
    for i in range(t.size):
        w = math.exp(-beta * (t[i] - prev))
        s0 *= w
        s1 *= w
        lam = m[k[i]] + a[k[i], 0] * s0 + a[k[i], 1] * s1
        ll += math.log(lam)
        if k[i] == 0:
            s0 += 1.0
        else:
            s1 += 1.0
        prev = t[i]
    ll -= (m[0] + m[1]) * T
    tail = np.expm1(-beta * (T - t))
    for n in (0, 1):
        s = float(np.sum(tail[k == n]))
        ll += ((a[0, n] + a[1, n]) / beta) * s
    return ll


def biv_loglik_brute(
    times: Sequence[float],
    types: Sequence[int],
    T: float,
    mu: Sequence[float],
    alpha: Sequence[Sequence[float]],
    beta: float,
) -> float:
    """Reference O(N^2) bivariate log-likelihood (tests / validation only)."""
    t = _check_times(np.asarray(times, dtype=float), T)
    k = _check_types(np.asarray(types), t.size)
    m, a = _check_biv_params(mu, alpha, beta)
    ll = 0.0
    for i in range(t.size):
        lam = m[k[i]]
        for n in (0, 1):
            mask = k[:i] == n
            lam += a[k[i], n] * float(np.sum(np.exp(-beta * (t[i] - t[:i][mask]))))
        ll += math.log(lam)
    ll -= (m[0] + m[1]) * T
    tail = np.expm1(-beta * (T - t))
    for n in (0, 1):
        ll += ((a[0, n] + a[1, n]) / beta) * float(np.sum(tail[k == n]))
    return ll


# ------------------------------------------------------------------ residuals

def uni_residuals(times: Sequence[float], T: float, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Time-rescaling residuals ``tau_i = Lambda(t_i) - Lambda(t_{i-1})``.

    With ``S_i = sum_{j<i} exp(-beta (t_i - t_j))`` (same recursion as the
    likelihood), the compensator at an event time is

        Lambda(t_i) = mu t_i + (alpha/beta) * ((i-1) - S_i),

    and under the true model the increments are iid Exp(1) (time-rescaling
    theorem, see module docstring).
    """
    t = _check_times(np.asarray(times, dtype=float), T)
    _check_uni_params(mu, alpha, beta)
    n = t.size
    lam_int = np.empty(n)
    s = 0.0
    for i in range(n):
        if i > 0:
            s = math.exp(-beta * (t[i] - t[i - 1])) * (s + 1.0)
        lam_int[i] = mu * t[i] + (alpha / beta) * (i - s)
    tau = np.diff(lam_int, prepend=0.0)
    return tau


def biv_residuals(
    times: Sequence[float],
    types: Sequence[int],
    T: float,
    mu: Sequence[float],
    alpha: Sequence[Sequence[float]],
    beta: float,
) -> List[np.ndarray]:
    """Per-type time-rescaling residuals for the bivariate model.

    For each type ``m`` the compensator ``Lambda_m`` is evaluated at that
    type's event times; increments between consecutive same-type events are
    iid Exp(1) under the model.  Returns ``[tau_type0, tau_type1]``.
    """
    t = _check_times(np.asarray(times, dtype=float), T)
    k = _check_types(np.asarray(types), t.size)
    m, a = _check_biv_params(mu, alpha, beta)
    s = [0.0, 0.0]  # decayed exponential sums per source type
    c = [0.0, 0.0]  # event counts per source type (strictly before current t)
    prev_t = 0.0
    lam_int: List[List[float]] = [[], []]
    for i in range(t.size):
        w = math.exp(-beta * (t[i] - prev_t))
        s[0] *= w
        s[1] *= w
        mm = k[i]
        val = m[mm] * t[i]
        for n in (0, 1):
            val += (a[mm, n] / beta) * (c[n] - s[n])
        lam_int[mm].append(val)
        s[mm] += 1.0
        c[mm] += 1.0
        prev_t = t[i]
    return [np.diff(np.asarray(v), prepend=0.0) for v in lam_int]


# ----------------------------------------------------------------- simulation

def simulate_uni(mu: float, alpha: float, beta: float, T: float, seed: int) -> np.ndarray:
    """Simulate a univariate Hawkes process on ``[0, T]`` by Ogata thinning.

    Pinned algorithm: after each step the decayed excitation sum ``S`` gives
    the current intensity ``lambda_bar = mu + alpha S``, which upper-bounds
    the intensity until the next event (exponential decay).  Draw
    ``w ~ Exp(lambda_bar)`` (one uniform), advance and decay, then accept the
    candidate iff ``u * lambda_bar <= lambda(t)`` (second uniform).

    Raises:
        ValueError: bad params, ``alpha/beta >= 1`` (non-stationary), or bad T.
    """
    _check_uni_params(mu, alpha, beta)
    if alpha / beta >= 1.0:
        raise ValueError(f"cannot simulate non-stationary process: alpha/beta = {alpha / beta}")
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError(f"T must be positive and finite, got {T}")
    rng = Lcg64(seed)
    t = 0.0
    s = 0.0
    out: List[float] = []
    while True:
        lam_bar = mu + alpha * s
        w = rng.exponential(lam_bar)
        t += w
        if t > T:
            break
        s *= math.exp(-beta * w)
        lam_t = mu + alpha * s
        if rng.uniform() * lam_bar <= lam_t:
            out.append(t)
            s += 1.0
    return np.asarray(out)


def simulate_biv(
    mu: Sequence[float],
    alpha: Sequence[Sequence[float]],
    beta: float,
    T: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate a bivariate Hawkes process by Ogata thinning (pinned).

    Same scheme as :func:`simulate_uni` on the total intensity; an accepted
    candidate with ``d = u * lambda_bar`` is type 0 if ``d <= lambda_0(t)``,
    else type 1 (given ``d <= lambda_0 + lambda_1``).

    Returns:
        (times, types) arrays.
    """
    m, a = _check_biv_params(mu, alpha, beta)
    disc = (a[0, 0] - a[1, 1]) ** 2 + 4.0 * a[0, 1] * a[1, 0]
    rad = 0.5 * (a[0, 0] + a[1, 1] + math.sqrt(disc)) / beta
    if rad >= 1.0:
        raise ValueError(f"cannot simulate non-stationary process: spectral radius {rad}")
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError(f"T must be positive and finite, got {T}")
    rng = Lcg64(seed)
    t = 0.0
    s = [0.0, 0.0]
    times: List[float] = []
    types: List[int] = []
    while True:
        lam0 = m[0] + a[0, 0] * s[0] + a[0, 1] * s[1]
        lam1 = m[1] + a[1, 0] * s[0] + a[1, 1] * s[1]
        lam_bar = lam0 + lam1
        w = rng.exponential(lam_bar)
        t += w
        if t > T:
            break
        decay = math.exp(-beta * w)
        s[0] *= decay
        s[1] *= decay
        lam0 = m[0] + a[0, 0] * s[0] + a[0, 1] * s[1]
        lam1 = m[1] + a[1, 0] * s[0] + a[1, 1] * s[1]
        d = rng.uniform() * lam_bar
        if d <= lam0 + lam1:
            typ = 0 if d <= lam0 else 1
            times.append(t)
            types.append(typ)
            s[typ] += 1.0
    return np.asarray(times), np.asarray(types, dtype=int)


# ------------------------------------------------------------------------ MLE

@dataclass(frozen=True)
class HawkesUniFit:
    """Univariate MLE result.

    Attributes:
        mu, alpha, beta: fitted parameters.
        loglik: maximized log-likelihood.
        branching: ``alpha / beta``.
        converged: Nelder-Mead tolerance convergence flag.
        n_iter: Nelder-Mead iterations.
    """

    mu: float
    alpha: float
    beta: float
    loglik: float
    branching: float
    converged: bool
    n_iter: int


@dataclass(frozen=True)
class HawkesBivFit:
    """Bivariate MLE result.

    Attributes:
        mu: length-2 background rates.
        alpha: 2x2 excitation matrix.
        beta: shared decay.
        loglik: maximized log-likelihood.
        branching: 2x2 branching matrix ``alpha / beta``.
        spectral_radius: stationarity measure (stationary iff < 1).
        converged: Nelder-Mead tolerance convergence flag.
        n_iter: total Nelder-Mead iterations (both pinned stages).
    """

    mu: np.ndarray
    alpha: np.ndarray
    beta: float
    loglik: float
    branching: np.ndarray
    spectral_radius: float
    converged: bool
    n_iter: int


def _warn_branching(value: float) -> None:
    if value >= 1.0:
        warnings.warn(
            f"fitted Hawkes process is non-stationary (branching {value:.4f} >= 1)",
            RuntimeWarning,
            stacklevel=3,
        )
    elif value >= 0.9:
        warnings.warn(
            f"fitted Hawkes process is near-critical (branching {value:.4f} >= 0.9)",
            RuntimeWarning,
            stacklevel=3,
        )


def fit_uni(times: Sequence[float], T: float) -> HawkesUniFit:
    """Univariate MLE via Nelder-Mead on log-transformed parameters.

    Pinned setup: minimize ``-ll(exp(x))`` over ``x = (log mu, log alpha,
    log beta)`` from ``x0 = (log(0.5 N / T), log 0.5, 0)`` with
    ``xatol=1e-8, fatol=1e-10, maxiter=5000``.  The log transform enforces
    positivity without constraints.  Emits a RuntimeWarning when the fitted
    branching ratio is >= 0.9 (near-critical) or >= 1 (non-stationary) —
    the fit itself still returns.
    """
    t = _check_times(np.asarray(times, dtype=float), T)

    def objective(x: Sequence[float]) -> float:
        try:
            mu, alpha, beta = (math.exp(v) for v in x)
        except OverflowError:
            return _PENALTY
        if not all(map(math.isfinite, (mu, alpha, beta))) or mu <= 0.0 or beta <= 0.0:
            return _PENALTY
        val = uni_loglik(t, T, mu, alpha, beta)
        return -val if math.isfinite(val) else _PENALTY

    x0 = [math.log(0.5 * t.size / T), math.log(0.5), 0.0]
    res = nelder_mead(objective, x0)
    mu, alpha, beta = (math.exp(v) for v in res.x)
    branching = alpha / beta
    _warn_branching(branching)
    return HawkesUniFit(
        mu=mu, alpha=alpha, beta=beta, loglik=-res.fval,
        branching=branching, converged=res.converged, n_iter=res.n_iter,
    )


def spectral_radius_2x2(matrix: Sequence[Sequence[float]]) -> float:
    """Spectral radius of a 2x2 matrix via the pinned closed form.

    With trace ``tr`` and determinant ``det``, the eigenvalues are
    ``(tr +/- sqrt(tr^2 - 4 det)) / 2``; when the discriminant is negative the
    complex pair has modulus ``sqrt(det)``.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (2, 2) or not np.all(np.isfinite(m)):
        raise ValueError("matrix must be a finite 2x2 array")
    tr = m[0, 0] + m[1, 1]
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        r = math.sqrt(disc)
        return max(abs(0.5 * (tr + r)), abs(0.5 * (tr - r)))
    return math.sqrt(det)


def fit_biv(times: Sequence[float], types: Sequence[int], T: float) -> HawkesBivFit:
    """Bivariate MLE (full 2x2 alpha, scalar beta) via pinned Nelder-Mead.

    Pinned setup: parameters ``x = (log mu0, log mu1, log a00, log a01,
    log a10, log a11, log beta)``; start ``mu_m = 0.5 N_m / T``, all
    ``a = 0.25``, ``beta = 1``; two Nelder-Mead stages (the second restarts
    from the first stage's optimum) with ``xatol=1e-8, fatol=1e-10,
    maxiter=5000`` each — a pinned deterministic polish that guards against
    premature simplex collapse in 7 dimensions.  Warns when the branching
    spectral radius is >= 0.9 (near-critical) or >= 1 (non-stationary).
    """
    t = _check_times(np.asarray(times, dtype=float), T)
    k = _check_types(np.asarray(types), t.size)
    n0 = int(np.sum(k == 0))
    n1 = t.size - n0
    if n0 == 0 or n1 == 0:
        raise ValueError("both event types must be present to fit the bivariate model")

    def objective(x: Sequence[float]) -> float:
        try:
            p = [math.exp(v) for v in x]
        except OverflowError:
            return _PENALTY
        if not all(map(math.isfinite, p)):
            return _PENALTY
        mu = p[0:2]
        alpha = [[p[2], p[3]], [p[4], p[5]]]
        val = biv_loglik(t, k, T, mu, alpha, p[6])
        return -val if math.isfinite(val) else _PENALTY

    x0 = [
        math.log(0.5 * n0 / T), math.log(0.5 * n1 / T),
        math.log(0.25), math.log(0.25), math.log(0.25), math.log(0.25),
        0.0,
    ]
    res1 = nelder_mead(objective, x0)
    res2 = nelder_mead(objective, res1.x)
    res: NMResult = res2
    p = [math.exp(v) for v in res.x]
    mu = np.asarray(p[0:2])
    alpha = np.asarray([[p[2], p[3]], [p[4], p[5]]])
    beta = p[6]
    branching = alpha / beta
    rad = spectral_radius_2x2(branching)
    _warn_branching(rad)
    return HawkesBivFit(
        mu=mu, alpha=alpha, beta=beta, loglik=-res.fval,
        branching=branching, spectral_radius=rad,
        converged=res.converged, n_iter=res1.n_iter + res2.n_iter,
    )
