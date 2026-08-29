//! Native Nelder-Mead simplex minimizer (P02-pinned configuration).
//!
//! Deterministic, derivative-free minimization of `f: R^n -> f64`.  The
//! configuration is pinned so every language reproduces the same iterates:
//!
//! * Coefficients: reflection `1`, expansion `2`, contraction `0.5`,
//!   shrink `0.5`.
//! * Initial simplex (scipy's rule): vertex 0 is `x0`; vertex `i` perturbs
//!   coordinate `i-1` — multiply by 1.05 if nonzero, else set it to 0.00025.
//! * Iteration: order vertices by `f` (stable — ties keep prior order, key
//!   `(f_i, index)`), reflect the worst through the centroid of the rest,
//!   then expand / outside-contract / inside-contract / shrink following the
//!   Lagarias et al. (1998) decision tree.
//! * Termination *before* an iteration: max abs coordinate spread of the
//!   simplex to the best vertex `< xatol` AND max abs `f` spread `< fatol`,
//!   or `maxiter` iterations.
//!
//! Objectives must be pure and never fail: out-of-domain points should return
//! a large penalty value.

use std::cmp::Ordering;

use crate::error::{LobError, Result};

/// Pinned default `xatol` (used by all Hawkes fits).
pub const DEFAULT_XATOL: f64 = 1e-8;
/// Pinned default `fatol`.
pub const DEFAULT_FATOL: f64 = 1e-10;
/// Pinned default iteration cap.
pub const DEFAULT_MAXITER: usize = 5000;

/// Nelder-Mead outcome.
#[derive(Debug, Clone)]
pub struct NmResult {
    /// Best vertex found.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub fval: f64,
    /// Iterations performed.
    pub n_iter: usize,
    /// True if the tolerance test (not the iteration cap) stopped it.
    pub converged: bool,
}

/// NaN-tolerant float ordering (NaN sorts last; a pure objective returning a
/// finite penalty never produces NaN, this is belt-and-braces).
fn cmp_f(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        if a.is_nan() && b.is_nan() {
            Ordering::Equal
        } else if a.is_nan() {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    })
}

/// Stable ascending order of vertex indices by `(f, index)`.
fn sorted_order(fvals: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..fvals.len()).collect();
    order.sort_by(|&a, &b| cmp_f(fvals[a], fvals[b]).then(a.cmp(&b)));
    order
}

/// Minimize `f` from `x0` with the pinned Nelder-Mead scheme above.
///
/// # Errors
/// [`LobError::InvalidInput`] on empty `x0`, non-finite start, or bad
/// tolerances (`xatol <= 0`, `fatol <= 0`, `maxiter == 0`).
pub fn nelder_mead<F: FnMut(&[f64]) -> f64>(
    mut f: F,
    x0: &[f64],
    xatol: f64,
    fatol: f64,
    maxiter: usize,
) -> Result<NmResult> {
    let n = x0.len();
    if n == 0 {
        return Err(LobError::InvalidInput("x0 must be non-empty".to_string()));
    }
    if x0.iter().any(|v| !v.is_finite()) {
        return Err(LobError::InvalidInput("x0 must be finite".to_string()));
    }
    if !(xatol > 0.0 && fatol > 0.0 && maxiter >= 1) {
        return Err(LobError::InvalidInput(
            "tolerances must be positive and maxiter >= 1".to_string(),
        ));
    }

    // initial simplex (scipy rule)
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] = if v[i] != 0.0 { v[i] * 1.05 } else { 0.00025 };
        simplex.push(v);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let mut n_iter = 0usize;
    let mut converged = false;
    while n_iter < maxiter {
        // stable ascending order by f
        let order = sorted_order(&fvals);
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        fvals = order.iter().map(|&i| fvals[i]).collect();

        let mut xspread: f64 = 0.0;
        for v in simplex.iter().skip(1) {
            for j in 0..n {
                xspread = xspread.max((v[j] - simplex[0][j]).abs());
            }
        }
        let mut fspread: f64 = 0.0;
        for &fv in fvals.iter().skip(1) {
            fspread = fspread.max((fv - fvals[0]).abs());
        }
        if xspread < xatol && fspread < fatol {
            converged = true;
            break;
        }
        n_iter += 1;

        // centroid of all but the worst vertex
        let centroid: Vec<f64> = (0..n)
            .map(|j| simplex.iter().take(n).map(|v| v[j]).sum::<f64>() / n as f64)
            .collect();
        let worst = simplex[n].clone();
        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + (centroid[j] - worst[j]))
            .collect();
        let fr = f(&xr);

        if cmp_f(fr, fvals[0]) == Ordering::Less {
            // try expansion
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 2.0 * (centroid[j] - worst[j]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            let (xc, fc, accept) = if fr < fvals[n] {
                // outside contraction
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + 0.5 * (xr[j] - centroid[j]))
                    .collect();
                let fc = f(&xc);
                let accept = fc <= fr;
                (xc, fc, accept)
            } else {
                // inside contraction
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] - 0.5 * (centroid[j] - worst[j]))
                    .collect();
                let fc = f(&xc);
                let accept = fc < fvals[n];
                (xc, fc, accept)
            };
            if accept {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                // shrink toward the best vertex
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                    }
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    let order = sorted_order(&fvals);
    let best = order[0];
    Ok(NmResult {
        x: simplex[best].clone(),
        fval: fvals[best],
        n_iter,
        converged,
    })
}

/// [`nelder_mead`] with the pinned default tolerances
/// (`xatol = 1e-8`, `fatol = 1e-10`, `maxiter = 5000`).
pub fn nelder_mead_default<F: FnMut(&[f64]) -> f64>(f: F, x0: &[f64]) -> Result<NmResult> {
    nelder_mead(f, x0, DEFAULT_XATOL, DEFAULT_FATOL, DEFAULT_MAXITER)
}
