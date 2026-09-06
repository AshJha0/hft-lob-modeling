//! Goodness of fit: Kolmogorov-Smirnov test of Exp(1) residuals.
//!
//! By the time-rescaling theorem (see [`crate::hawkes`]), a correctly
//! specified Hawkes model turns its inter-arrival compensator increments into
//! iid Exp(1) draws.  This module tests that natively:
//!
//! * KS statistic against the Exp(1) CDF `F(x) = 1 - exp(-x)`:
//!
//! ```text
//! D_n = max_i max( i/n - F(x_(i)),  F(x_(i)) - (i-1)/n )
//! ```
//!
//! over the sorted sample `x_(1) <= ... <= x_(n)`.
//!
//! * Asymptotic p-value with the Stephens small-sample correction (Numerical
//!   Recipes pinned form): with
//!   `lambda = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * D_n`, the p-value is the
//!   Kolmogorov tail series
//!   `Q(lambda) = 2 * sum_{j=1}^{100} (-1)^{j-1} exp(-2 j^2 lambda^2)` for
//!   `lambda >= 0.1`, or — because that alternating series converges too
//!   slowly for small lambda (100 terms are off by 0.13 at lambda = 0.01) —
//!   the dual (Jacobi theta) form, Numerical Recipes eq. 14.3.20,
//!   `Q(lambda) = 1 - (sqrt(2 pi)/lambda) * sum_{j=1}^{100}
//!   exp(-(2j-1)^2 pi^2 / (8 lambda^2))` for `lambda < 0.1`; clipped to
//!   `[0, 1]`.  Both branches are pinned at exactly 100 terms.  With very
//!   small n the asymptotic p-value is only indicative — callers should treat
//!   it as such (documented, not an error).

use crate::error::{LobError, Result};

const LAMBDA_SWITCH: f64 = 0.1;
const N_TERMS: u32 = 100;

/// KS statistic of `sample` against the Exp(1) CDF.
///
/// # Errors
/// [`LobError::InvalidInput`] on an empty sample, non-finite or negative
/// values.
pub fn ks_statistic_exp(sample: &[f64]) -> Result<f64> {
    if sample.is_empty() {
        return Err(LobError::InvalidInput(
            "sample must be a non-empty 1-D array".to_string(),
        ));
    }
    if sample.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(LobError::InvalidInput(
            "sample values must be finite and non-negative".to_string(),
        ));
    }
    let mut xs = sample.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));
    let n = xs.len() as f64;
    let mut d = 0.0_f64;
    for (idx, &x) in xs.iter().enumerate() {
        let i = (idx + 1) as f64;
        let cdf = -(-x).exp_m1(); // 1 - exp(-x), accurately
        d = d.max(i / n - cdf).max(cdf - (i - 1.0) / n);
    }
    Ok(d)
}

/// Asymptotic Kolmogorov p-value for statistic `d` on `n` points.
///
/// Pinned formula: `lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * d` and the
/// 100-term alternating tail series (module docs), clipped to `[0, 1]`.
/// `p = 1` exactly when `d = 0`.
///
/// # Errors
/// [`LobError::InvalidInput`] if `d` is not in `[0, 1]` or `n == 0`.
pub fn ks_pvalue(d: f64, n: usize) -> Result<f64> {
    if !(d.is_finite() && (0.0..=1.0).contains(&d)) {
        return Err(LobError::InvalidInput(format!(
            "d must be in [0, 1], got {d}"
        )));
    }
    if n == 0 {
        return Err(LobError::InvalidInput(
            "n must be a positive integer, got 0".to_string(),
        ));
    }
    if d == 0.0 {
        return Ok(1.0);
    }
    let sqrt_n = (n as f64).sqrt();
    let lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d;
    Ok(kolmogorov_q(lam))
}

/// Kolmogorov tail probability `Q(lambda)` by the pinned two-branch
/// evaluation (`lambda >= 0`), clipped to `[0, 1]`.
pub fn kolmogorov_q(lam: f64) -> f64 {
    let p = if lam < LAMBDA_SWITCH {
        let pi = std::f64::consts::PI;
        let mut total = 0.0;
        for j in 1..=N_TERMS {
            let m = 2.0 * j as f64 - 1.0;
            total += (-(m * m) * pi * pi / (8.0 * lam * lam)).exp();
        }
        1.0 - ((2.0 * pi).sqrt() / lam) * total
    } else {
        let mut total = 0.0;
        for j in 1..=N_TERMS {
            let jj = j as f64;
            let term = (-2.0 * jj * jj * lam * lam).exp();
            if j % 2 == 1 {
                total += term;
            } else {
                total -= term;
            }
        }
        2.0 * total
    };
    p.clamp(0.0, 1.0)
}

/// KS test of `sample` against Exp(1): returns `(statistic, p_value)`.
///
/// # Errors
/// Propagates [`ks_statistic_exp`] / [`ks_pvalue`] validation.
pub fn ks_test_exp(sample: &[f64]) -> Result<(f64, f64)> {
    let d = ks_statistic_exp(sample)?;
    let p = ks_pvalue(d, sample.len())?;
    Ok((d, p))
}
