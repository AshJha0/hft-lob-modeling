"""Native Nelder-Mead simplex minimizer (P02-pinned configuration).

Deterministic, derivative-free minimization of ``f: R^n -> float``.  The
configuration is pinned so every language reproduces the same iterates:

* Coefficients: reflection ``alpha = 1``, expansion ``gamma = 2``,
  contraction ``rho = 0.5``, shrink ``sigma = 0.5``.
* Initial simplex (scipy's rule): vertex 0 is ``x0``; vertex ``i`` perturbs
  coordinate ``i-1`` — multiply by 1.05 if nonzero, else set it to 0.00025.
* Iteration: order vertices by ``f`` (stable sort), reflect the worst through
  the centroid of the rest, then expand / outside-contract / inside-contract /
  shrink following the Lagarias et al. (1998) decision tree.
* Termination: max abs coordinate spread of the simplex ``< xatol`` AND max
  abs ``f`` spread ``< fatol``, or ``maxiter`` iterations.

Objectives must be pure and never raise: out-of-domain points should return a
large penalty value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence


@dataclass(frozen=True)
class NMResult:
    """Nelder-Mead outcome.

    Attributes:
        x: best vertex found.
        fval: objective value at ``x``.
        n_iter: iterations performed.
        converged: True if the tolerance test (not the iteration cap) stopped it.
    """

    x: List[float]
    fval: float
    n_iter: int
    converged: bool


def nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    xatol: float = 1e-8,
    fatol: float = 1e-10,
    maxiter: int = 5000,
) -> NMResult:
    """Minimize ``f`` from ``x0`` with the pinned Nelder-Mead scheme above.

    Raises:
        ValueError: on empty ``x0``, non-finite start, or bad tolerances.
    """
    x0 = [float(v) for v in x0]
    n = len(x0)
    if n == 0:
        raise ValueError("x0 must be non-empty")
    if any(not math.isfinite(v) for v in x0):
        raise ValueError("x0 must be finite")
    if not (xatol > 0.0 and fatol > 0.0 and maxiter >= 1):
        raise ValueError("tolerances must be positive and maxiter >= 1")

    # initial simplex (scipy rule)
    simplex: List[List[float]] = [list(x0)]
    for i in range(n):
        v = list(x0)
        v[i] = v[i] * 1.05 if v[i] != 0.0 else 0.00025
        simplex.append(v)
    fvals = [float(f(v)) for v in simplex]

    n_iter = 0
    converged = False
    while n_iter < maxiter:
        # stable ascending order by f
        order = sorted(range(n + 1), key=lambda i: (fvals[i], i))
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        xspread = max(
            abs(simplex[i][j] - simplex[0][j]) for i in range(1, n + 1) for j in range(n)
        )
        fspread = max(abs(fvals[i] - fvals[0]) for i in range(1, n + 1))
        if xspread < xatol and fspread < fatol:
            converged = True
            break
        n_iter += 1

        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[n]
        xr = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        fr = float(f(xr))

        if fr < fvals[0]:
            # try expansion
            xe = [centroid[j] + 2.0 * (centroid[j] - worst[j]) for j in range(n)]
            fe = float(f(xe))
            if fe < fr:
                simplex[n], fvals[n] = xe, fe
            else:
                simplex[n], fvals[n] = xr, fr
        elif fr < fvals[n - 1]:
            simplex[n], fvals[n] = xr, fr
        else:
            if fr < fvals[n]:
                # outside contraction
                xc = [centroid[j] + 0.5 * (xr[j] - centroid[j]) for j in range(n)]
                fc = float(f(xc))
                accept = fc <= fr
            else:
                # inside contraction
                xc = [centroid[j] - 0.5 * (centroid[j] - worst[j]) for j in range(n)]
                fc = float(f(xc))
                accept = fc < fvals[n]
            if accept:
                simplex[n], fvals[n] = xc, fc
            else:
                # shrink toward the best vertex
                for i in range(1, n + 1):
                    simplex[i] = [
                        simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j])
                        for j in range(n)
                    ]
                    fvals[i] = float(f(simplex[i]))

    order = sorted(range(n + 1), key=lambda i: (fvals[i], i))
    best = order[0]
    return NMResult(x=list(simplex[best]), fval=fvals[best], n_iter=n_iter, converged=converged)
