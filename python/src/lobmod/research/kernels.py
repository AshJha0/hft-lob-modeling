r"""Hawkes processes with a sum-of-exponentials kernel, in O(N K).

Python-only analysis code, outside the four-language golden contract; numpy
only.  The pinned model in :mod:`lobmod.hawkes` has one exponential.  When the
real kernel has two timescales or a power-law tail, that one exponential fits a
compromise and understates the branching ratio.

The model
---------
Each event raises the intensity by a mixture of exponential densities,

    lambda(t) = mu(t) + sum_k a_k sum_{t_j < t} b_k exp(-b_k (t - t_j)),

so ``a_k`` is the expected number of children through component ``k`` and the
branching ratio is ``n = sum_k a_k``.  The pinned parameters map as
``a = alpha / beta``, ``b = beta``.

With ``R_k(i) = sum_{j<i} exp(-b_k (t_i - t_j))``, the recursion
``R_k(i) = exp(-b_k (t_i - t_{i-1})) (1 + R_k(i-1))`` gives the likelihood in
O(N K) (Ozaki 1979, one recursion per component):

    log L = sum_i log lambda(t_i) - int_0^T mu - sum_k a_k sum_i (1 - exp(-b_k (T - t_i))).

Fitting
-------
For fixed rates ``b`` the intensity is linear in the unknowns ``theta`` (baseline
coefficients and ``a``), so ``log L = sum_i log (X theta)_i - C . theta`` is
concave on ``theta >= 0`` and has one maximum value.  :func:`solve_poisson_linear`
finds it with projected Newton steps.  The rates are then profiled: a coarse grid
picks starts, and Nelder-Mead moves ``log b``.  The outer problem is not concave,
so that search can in principle stop at a local optimum.

Power-law kernels
-----------------
The Lomax density ``(a/c)(1 + t/c)^-(a+1)`` is exactly a Gamma(a, rate c) mixture
of exponential densities.  Discretising on fixed geometric rates ``s_m`` gives
weights ``w_m ~ s_m^a exp(-c s_m)`` (:func:`power_law_weights`), so a tail
with three parameters ``(n, a, c)`` stays O(N M) and the recursions are computed
once per path.  Approximating a power law by exponentials on a geometric grid is
an established device (Bochud and Challet 2007; Hardiman, Bercot and Bouchaud
2013); the weight rule here is the Gamma mixture above.

Limits: stationarity (``n < 1``) is not imposed, and ``n`` is reported as fitted.
Rates are bounded to ``[5/T, 1000]``, so a component slower than ``T/5`` cannot
be told apart from the baseline.  Only simulated paths have been used.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..hawkes import _check_times
from ..neldermead import nelder_mead

MAX_RATE = 1000.0
#: fixed rates of the power-law mixture; covers timescales 1 ms to 1000 s
POWER_LAW_RATES = tuple(np.geomspace(1e-3, 1e3, 16))


class FlatBaseline:
    """Constant ``mu``: the one-coefficient case of the baseline interface.

    A baseline gives ``design(t)`` (``mu(t) = design(t) @ coef``), ``cumulative(t)``
    (its integral from 0) and ``n_coef``; see :mod:`lobmod.research.seasonality`.
    """

    n_coef = 1

    def __init__(self, T: float) -> None:
        self.T = float(T)

    def design(self, t: np.ndarray) -> np.ndarray:
        return np.ones((np.size(t), 1))

    def cumulative(self, t: np.ndarray) -> np.ndarray:
        return np.asarray(t, dtype=float).reshape(-1, 1)


@dataclass(frozen=True)
class KernelFit:
    """Maximum-likelihood fit.

    Attributes:
        name: model label.
        coef: baseline coefficients (one value, ``mu``, for a flat baseline).
        weights: branching mass ``a_k`` per component.
        betas: rates ``b_k``.
        loglik: maximised log-likelihood.
        n_params: free parameters used for AIC and BIC.
        n_events: events in the path.
        converged: inner solver and outer search both met their tolerances.
        shape: ``(a, c)`` for the power-law family, otherwise empty.
    """

    name: str
    coef: np.ndarray
    weights: np.ndarray
    betas: np.ndarray
    loglik: float
    n_params: int
    n_events: int
    converged: bool
    shape: Tuple[float, ...] = ()

    @property
    def branching(self) -> float:
        return float(np.sum(self.weights))

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * math.log(self.n_events) - 2.0 * self.loglik


def excitation(times: np.ndarray, betas: Sequence[float]) -> np.ndarray:
    """``R[i, k] = sum_{j<i} exp(-b_k (t_i - t_j))`` by the O(N K) recursion."""
    t = np.asarray(times, dtype=float)
    b = np.asarray(betas, dtype=float)
    out = np.zeros((t.size, b.size))
    decay = np.exp(-np.outer(np.diff(t), b))
    for k in range(b.size):
        r = 0.0
        col = [0.0]
        for w in decay[:, k].tolist():
            r = w * (1.0 + r)
            col.append(r)
        out[:, k] = col
    return out


def _tail_mass(t: np.ndarray, T: float, betas: np.ndarray) -> np.ndarray:
    return -np.expm1(-np.outer(T - t, betas)).sum(axis=0)


def _check_mix(weights: Sequence[float], betas: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(weights, dtype=float)
    b = np.asarray(betas, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or np.any(a < 0.0) or np.any(b <= 0.0):
        raise ValueError("need equal-length weights >= 0 and betas > 0")
    return a, b


def sumexp_loglik(
    times: Sequence[float], T: float, coef: Any, weights: Sequence[float],
    betas: Sequence[float], baseline: Any = None,
) -> float:
    """Exact log-likelihood in O(N K); ``coef`` is ``mu`` for a flat baseline."""
    t = _check_times(np.asarray(times, dtype=float), T)
    a, b = _check_mix(weights, betas)
    base = baseline or FlatBaseline(T)
    c = np.atleast_1d(np.asarray(coef, dtype=float))
    lam = base.design(t) @ c + excitation(t, b) @ (a * b)
    if np.any(lam <= 0.0):
        return -math.inf
    total = base.cumulative(np.array([T]))[0] @ c
    return float(np.sum(np.log(lam)) - total - a @ _tail_mass(t, T, b))


def sumexp_loglik_brute(
    times: Sequence[float], T: float, mu: float, weights: Sequence[float], betas: Sequence[float],
) -> float:
    """O(N^2) flat-baseline reference, for tests only."""
    t = np.asarray(times, dtype=float)
    a, b = _check_mix(weights, betas)
    ll = -mu * T
    for i in range(t.size):
        lag = t[i] - t[:i]
        ll += math.log(mu + float(np.sum((a * b) * np.exp(-np.outer(lag, b)))))
        ll -= float(np.sum(a * (1.0 - np.exp(-b * (T - t[i])))))
    return ll


def sumexp_residuals(
    times: Sequence[float], T: float, coef: Any, weights: Sequence[float],
    betas: Sequence[float], baseline: Any = None,
) -> np.ndarray:
    """Time-rescaling residuals, iid Exp(1) under the true model.

    ``Lambda(t_i) = M(t_i) + sum_k a_k (i - R_k(i))`` with ``i`` counted from 0
    and ``M`` the integrated baseline, the same convention as
    :func:`lobmod.hawkes.uni_residuals`.
    """
    t = _check_times(np.asarray(times, dtype=float), T)
    a, b = _check_mix(weights, betas)
    base = baseline or FlatBaseline(T)
    c = np.atleast_1d(np.asarray(coef, dtype=float))
    comp = base.cumulative(t) @ c + (np.arange(t.size)[:, None] - excitation(t, b)) @ a
    return np.diff(comp, prepend=0.0)


def fit_residuals(times: Sequence[float], T: float, fit: KernelFit, baseline: Any = None) -> np.ndarray:
    """:func:`sumexp_residuals` at a fitted model."""
    return sumexp_residuals(times, T, fit.coef, fit.weights, fit.betas, baseline)


def solve_poisson_linear(
    X: np.ndarray, C: np.ndarray, theta0: Optional[np.ndarray] = None,
    tol: float = 1e-9, max_iter: int = 100,
) -> Tuple[np.ndarray, float, bool]:
    r"""Maximise ``sum_i log (X theta)_i - C . theta`` over ``theta >= 0``.

    A few EM steps (``theta_c <- theta_c (X^T 1/lambda)_c / C_c``, monotone,
    Shepp and Vardi 1982) give a start; projected Newton on the free set then
    finishes.  Stops when the Newton decrement is below ``tol`` with no bound
    variable wanting to move.  Returns ``(theta, value, converged)``.
    """
    n = X.shape[0]
    C = np.maximum(C, 1e-300)
    theta = np.full(X.shape[1], n / (2.0 * C.sum())) if theta0 is None else np.maximum(theta0, 1e-6)

    def value(th: np.ndarray) -> float:
        lam = X @ th
        return -math.inf if np.any(lam <= 0.0) else float(np.sum(np.log(lam)) - C @ th)

    for _ in range(8):
        theta = theta * (X.T @ (1.0 / (X @ theta))) / C
    current = value(theta)
    for _ in range(max_iter):
        lam = X @ theta
        grad = X.T @ (1.0 / lam) - C
        free = (theta > 1e-12) | (grad > 0.0)
        theta = np.where(free, theta, 0.0)
        A = X[:, free] / lam[:, None]
        H = A.T @ A
        H[np.diag_indices_from(H)] *= 1.0 + 1e-10
        H[np.diag_indices_from(H)] += 1e-12
        d = np.linalg.solve(H, grad[free])
        if 0.5 * float(grad[free] @ d) < tol:
            return theta, value(theta), True
        step = 1.0
        while step > 1e-10:
            cand = theta.copy()
            cand[free] = np.maximum(theta[free] + step * d, 0.0)
            v = value(cand)
            if v > current:
                break
            step *= 0.5
        else:
            return theta, value(theta), True  # no ascent left at double precision
        theta, current = cand, v
    return theta, current, False


def _design(t: np.ndarray, T: float, betas: np.ndarray, base: Any) -> Tuple[np.ndarray, np.ndarray]:
    X = np.hstack([base.design(t), excitation(t, betas) * betas])
    C = np.concatenate([base.cumulative(np.array([T]))[0], _tail_mass(t, T, betas)])
    return X, C


def fit_fixed_mixture(
    times: Sequence[float], T: float, betas: Sequence[float], baseline: Any = None,
    name: str = "fixed rates",
) -> KernelFit:
    """Free weights on fixed rates: one concave problem, global optimum.

    ``n_params`` counts the baseline and the weights left above zero, so with a
    long grid it is a data-chosen count and AIC is only indicative.
    """
    t = _check_times(np.asarray(times, dtype=float), T)
    base = baseline or FlatBaseline(T)
    b = np.asarray(betas, dtype=float)
    X, C = _design(t, T, b, base)
    theta, ll, ok = solve_poisson_linear(X, C)
    a = theta[base.n_coef:]
    return KernelFit(name, theta[:base.n_coef], a, b, ll,
                     base.n_coef + int(np.sum(a > 1e-9)), t.size, ok)


def fit_sum_exp(
    times: Sequence[float], T: float, K: int = 2, baseline: Any = None, n_starts: int = 2,
) -> KernelFit:
    """MLE of ``K`` exponentials with free rates (module docstring).

    Rates come back in decreasing order.  ``n_params`` is baseline + 2K.

    Raises:
        ValueError: ``K < 1``.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    t = _check_times(np.asarray(times, dtype=float), T)
    base = baseline or FlatBaseline(T)
    lo, hi = math.log(5.0 / T), math.log(MAX_RATE)

    def solve(log_b: np.ndarray, start: Optional[np.ndarray]) -> Tuple[np.ndarray, float, bool]:
        X, C = _design(t, T, np.exp(log_b), base)
        return solve_poisson_linear(X, C, start)

    grid = np.linspace(max(lo, math.log(0.02)), math.log(200.0), 8)
    scored = []
    for combo in itertools.combinations(grid, K):
        lb = np.array(combo)
        scored.append((solve(lb, None)[1], tuple(lb)))
    scored.sort(reverse=True)

    best_ll, best_lb, best_ok = -math.inf, None, False
    for _, lb0 in scored[:n_starts]:
        warm: List[Optional[np.ndarray]] = [None]

        def objective(v: Sequence[float]) -> float:
            lb = np.asarray(v, dtype=float)
            over = float(np.sum(np.maximum(lb - hi, 0.0) + np.maximum(lo - lb, 0.0)))
            th, ll, _ = solve(np.clip(lb, lo, hi), warm[0])
            warm[0] = th
            return -ll + 1e3 * over

        res = nelder_mead(objective, list(lb0), xatol=1e-3, fatol=1e-5, maxiter=400)
        if -res.fval > best_ll:
            best_ll, best_lb, best_ok = -res.fval, np.clip(res.x, lo, hi), res.converged
    theta, ll, ok = solve(best_lb, None)
    order = np.argsort(-best_lb)
    return KernelFit(f"exp K={K}", theta[:base.n_coef], theta[base.n_coef:][order],
                     np.exp(best_lb[order]), ll, base.n_coef + 2 * K, t.size, ok and best_ok)


def power_law_weights(a: float, c: float, rates: Sequence[float] = POWER_LAW_RATES) -> np.ndarray:
    """Weights on fixed rates approximating the Lomax density; they sum to 1.

    Rates must be geometric, so ``d log s`` is constant (module docstring).

    Raises:
        ValueError: ``a`` or ``c`` not positive.
    """
    if a <= 0.0 or c <= 0.0:
        raise ValueError("a and c must be positive")
    s = np.asarray(rates, dtype=float)
    logw = a * np.log(s) - c * s
    w = np.exp(logw - logw.max())
    return w / w.sum()


def fit_power_law(
    times: Sequence[float], T: float, baseline: Any = None,
    rates: Sequence[float] = POWER_LAW_RATES,
) -> KernelFit:
    """Fit ``(baseline, n, a, c)`` with the fixed-rate power-law mixture.

    Recursions are computed once; each profile step solves a small concave
    problem.  ``a`` is kept in [0.05, 5] and ``c`` in [0.002, 20] so that the
    fixed rates still cover the kernel.
    """
    t = _check_times(np.asarray(times, dtype=float), T)
    base = baseline or FlatBaseline(T)
    s = np.asarray(rates, dtype=float)
    Bx = base.design(t)
    Bc = base.cumulative(np.array([T]))[0]
    Rs = excitation(t, s) * s
    tail = _tail_mass(t, T, s)
    lo = np.log([0.05, 0.002])
    hi = np.log([5.0, 20.0])

    def solve(v: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        a, c = np.exp(v)
        w = power_law_weights(a, c, s)
        X = np.column_stack([Bx, Rs @ w])
        return solve_poisson_linear(X, np.append(Bc, tail @ w))

    starts = [np.log([a, c]) for a in (0.4, 1.0, 2.5) for c in (0.02, 0.2, 2.0)]
    start = max(starts, key=lambda v: solve(v)[1])

    def objective(v: Sequence[float]) -> float:
        x = np.asarray(v, dtype=float)
        over = float(np.sum(np.maximum(x - hi, 0.0) + np.maximum(lo - x, 0.0)))
        return -solve(np.clip(x, lo, hi))[1] + 1e3 * over

    res = nelder_mead(objective, list(start), xatol=1e-3, fatol=1e-5, maxiter=300)
    v = np.clip(res.x, lo, hi)
    theta, ll, ok = solve(v)
    a, c = np.exp(v)
    return KernelFit("power-law mix", theta[:base.n_coef],
                     theta[base.n_coef] * power_law_weights(a, c, s), s, ll,
                     base.n_coef + 3, t.size, ok and res.converged, (float(a), float(c)))


def simulate_cluster(
    mu_fn: Callable[[np.ndarray], np.ndarray], mu_max: float, n: float,
    offsets: Callable[[int], np.ndarray], T: float, rng: np.random.Generator,
) -> np.ndarray:
    """Branching simulation: thinned immigrants, Poisson(n) children per event.

    ``offsets(k)`` draws ``k`` parent-to-child lags, which is how any kernel
    shape is expressed.  Starts empty at time 0, matching the likelihood.
    Exact ties, possible only through rounding, are dropped.
    """
    m = rng.poisson(mu_max * T)
    gen = rng.uniform(0.0, T, m)
    gen = gen[rng.uniform(0.0, mu_max, m) < mu_fn(gen)]
    out = [gen]
    while gen.size:
        parents = np.repeat(gen, rng.poisson(n, gen.size))
        gen = parents + offsets(parents.size)
        gen = gen[gen < T]
        out.append(gen)
    x = np.sort(np.concatenate(out))
    return x[np.r_[True, np.diff(x) > 0]]
