//! Hawkes processes with exponential kernels: likelihood, MLE, simulation.
//!
//! Univariate model
//! ----------------
//! Conditional intensity
//!
//! ```text
//! lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i)),
//! ```
//!
//! with background rate `mu > 0`, jump size `alpha >= 0` and decay
//! `beta > 0`.  The branching ratio `n = alpha / beta` is the expected number
//! of children per event; the process is stationary iff `n < 1`.
//!
//! Recursive log-likelihood (the pinned O(N) form)
//! -----------------------------------------------
//! With `R_i = sum_{j < i} exp(-beta (t_i - t_j))` the Ozaki (1979) recursion
//! `R_1 = 0, R_i = exp(-beta (t_i - t_{i-1})) (1 + R_{i-1})` gives
//!
//! ```text
//! ll = sum_i log(mu + alpha R_i) - mu T
//!      + (alpha/beta) * sum_i expm1(-beta (T - t_i))
//! ```
//!
//! (`expm1` pinned for accuracy in the tail terms).
//!
//! Bivariate model
//! ---------------
//! Two event types with full 2x2 excitation and a *scalar shared* `beta`
//! (pinned parameterization): the same recursion runs on the merged stream,
//! keeping decayed sums `S_n(t)` per source type.
//!
//! Time-rescaling theorem
//! ----------------------
//! If events follow a point process with compensator `Lambda(t)`, the
//! rescaled increments `tau_i = Lambda(t_i) - Lambda(t_{i-1})` are iid
//! Exp(1), so goodness of fit reduces to a KS test against Exp(1)
//! (see [`crate::gof`]).
//!
//! Units: times and `T` share one arbitrary time unit; `mu` and `beta` are
//! per unit time, `alpha` per event.  Simulators stop with an error after
//! [`MAX_CANDIDATES`] thinning draws (the loop terminates almost surely but
//! has no finite worst case).

use std::fs;
use std::io::Write;

use crate::csv::{parse_double_strict, parse_int64_strict, read_csv_lines};
use crate::error::{LobError, Result};
use crate::neldermead::{nelder_mead, NmResult, DEFAULT_FATOL, DEFAULT_MAXITER, DEFAULT_XATOL};
use crate::rng::Lcg64;

const PENALTY: f64 = 1e10;
/// Pinned upper bound on thinning candidates per simulation (error if hit).
pub const MAX_CANDIDATES: u64 = 50_000_000;

// ---------------------------------------------------------------- validation

fn check_times(t: &[f64], t_end: f64) -> Result<()> {
    if t.is_empty() {
        return Err(LobError::InvalidInput(
            "times must be a non-empty 1-D array".to_string(),
        ));
    }
    if t.iter().any(|v| !v.is_finite()) {
        return Err(LobError::InvalidInput("times must be finite".to_string()));
    }
    if t[0] <= 0.0 {
        return Err(LobError::InvalidInput(
            "times must be strictly positive".to_string(),
        ));
    }
    if t.windows(2).any(|w| w[1] - w[0] <= 0.0) {
        return Err(LobError::InvalidInput(
            "times must be strictly increasing".to_string(),
        ));
    }
    if !(t_end.is_finite() && t_end > 0.0) {
        return Err(LobError::InvalidInput(format!(
            "T must be positive and finite, got {t_end}"
        )));
    }
    if t[t.len() - 1] > t_end {
        return Err(LobError::InvalidInput(format!(
            "last event time {} exceeds horizon T={t_end}",
            t[t.len() - 1]
        )));
    }
    Ok(())
}

fn check_uni_params(mu: f64, alpha: f64, beta: f64) -> Result<()> {
    if !(mu.is_finite() && mu > 0.0) {
        return Err(LobError::InvalidInput(format!(
            "mu must be positive and finite, got {mu}"
        )));
    }
    if !(alpha.is_finite() && alpha >= 0.0) {
        return Err(LobError::InvalidInput(format!(
            "alpha must be non-negative and finite, got {alpha}"
        )));
    }
    if !(beta.is_finite() && beta > 0.0) {
        return Err(LobError::InvalidInput(format!(
            "beta must be positive and finite, got {beta}"
        )));
    }
    Ok(())
}

fn check_biv_params(mu: &[f64; 2], alpha: &[[f64; 2]; 2], beta: f64) -> Result<()> {
    if mu.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(LobError::InvalidInput(
            "mu must be two positive finite values".to_string(),
        ));
    }
    if alpha
        .iter()
        .flatten()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(LobError::InvalidInput(
            "alpha must be a 2x2 non-negative finite matrix".to_string(),
        ));
    }
    if !(beta.is_finite() && beta > 0.0) {
        return Err(LobError::InvalidInput(format!(
            "beta must be positive and finite, got {beta}"
        )));
    }
    Ok(())
}

fn check_types(types: &[usize], n: usize) -> Result<()> {
    if types.len() != n {
        return Err(LobError::InvalidInput(
            "types must have the same length as times".to_string(),
        ));
    }
    if types.iter().any(|&k| k > 1) {
        return Err(LobError::InvalidInput(
            "types must be 0 or 1".to_string(),
        ));
    }
    Ok(())
}

// ------------------------------------------------------------ log-likelihood

/// Unvalidated core of the univariate recursion (used by the fitting
/// objective, where the parameters are positive by construction).
fn uni_loglik_core(t: &[f64], t_end: f64, mu: f64, alpha: f64, beta: f64) -> f64 {
    let mut ll = mu.ln(); // R_1 = 0
    let mut r = 0.0;
    for i in 1..t.len() {
        let w = (-beta * (t[i] - t[i - 1])).exp();
        r = w * (1.0 + r);
        ll += (mu + alpha * r).ln();
    }
    ll -= mu * t_end;
    let mut tail = 0.0;
    for &ti in t {
        tail += (-beta * (t_end - ti)).exp_m1();
    }
    ll + (alpha / beta) * tail
}

/// Exact univariate log-likelihood on `[0, T]` via the pinned O(N) recursion.
///
/// # Errors
/// [`LobError::InvalidInput`] on invalid event times or parameters
/// (times non-empty, finite, strictly increasing, `t_1 > 0`, `t_N <= T`;
/// `mu > 0`, `alpha >= 0`, `beta > 0`).
pub fn uni_loglik(times: &[f64], t_end: f64, mu: f64, alpha: f64, beta: f64) -> Result<f64> {
    check_times(times, t_end)?;
    check_uni_params(mu, alpha, beta)?;
    Ok(uni_loglik_core(times, t_end, mu, alpha, beta))
}

/// Reference O(N^2) log-likelihood (direct double sum).
///
/// Used by tests to validate the recursion; not for production use.
///
/// # Errors
/// Same validation as [`uni_loglik`].
pub fn uni_loglik_brute(times: &[f64], t_end: f64, mu: f64, alpha: f64, beta: f64) -> Result<f64> {
    check_times(times, t_end)?;
    check_uni_params(mu, alpha, beta)?;
    let t = times;
    let mut ll = 0.0;
    for i in 0..t.len() {
        let mut excitation = 0.0;
        for j in 0..i {
            excitation += (-beta * (t[i] - t[j])).exp();
        }
        ll += (mu + alpha * excitation).ln();
    }
    ll -= mu * t_end;
    let mut tail = 0.0;
    for &ti in t {
        tail += (-beta * (t_end - ti)).exp_m1();
    }
    Ok(ll + (alpha / beta) * tail)
}

/// Unvalidated core of the bivariate merged-stream recursion.
fn biv_loglik_core(
    t: &[f64],
    k: &[usize],
    t_end: f64,
    mu: &[f64; 2],
    alpha: &[[f64; 2]; 2],
    beta: f64,
) -> f64 {
    let mut s = [0.0_f64, 0.0_f64];
    let mut prev = 0.0;
    let mut ll = 0.0;
    for i in 0..t.len() {
        let w = (-beta * (t[i] - prev)).exp();
        s[0] *= w;
        s[1] *= w;
        let m = k[i];
        let lam = mu[m] + alpha[m][0] * s[0] + alpha[m][1] * s[1];
        ll += lam.ln();
        s[m] += 1.0;
        prev = t[i];
    }
    ll -= (mu[0] + mu[1]) * t_end;
    for n in 0..2 {
        let mut tail = 0.0;
        for i in 0..t.len() {
            if k[i] == n {
                tail += (-beta * (t_end - t[i])).exp_m1();
            }
        }
        ll += ((alpha[0][n] + alpha[1][n]) / beta) * tail;
    }
    ll
}

/// Bivariate log-likelihood (full 2x2 alpha, scalar shared beta), O(N).
///
/// # Errors
/// [`LobError::InvalidInput`] on invalid times, types (must be 0/1, aligned
/// with times) or parameters.
pub fn biv_loglik(
    times: &[f64],
    types: &[usize],
    t_end: f64,
    mu: &[f64; 2],
    alpha: &[[f64; 2]; 2],
    beta: f64,
) -> Result<f64> {
    check_times(times, t_end)?;
    check_types(types, times.len())?;
    check_biv_params(mu, alpha, beta)?;
    Ok(biv_loglik_core(times, types, t_end, mu, alpha, beta))
}

/// Reference O(N^2) bivariate log-likelihood (tests / validation only).
///
/// # Errors
/// Same validation as [`biv_loglik`].
pub fn biv_loglik_brute(
    times: &[f64],
    types: &[usize],
    t_end: f64,
    mu: &[f64; 2],
    alpha: &[[f64; 2]; 2],
    beta: f64,
) -> Result<f64> {
    check_times(times, t_end)?;
    check_types(types, times.len())?;
    check_biv_params(mu, alpha, beta)?;
    let (t, k) = (times, types);
    let mut ll = 0.0;
    for i in 0..t.len() {
        let mut lam = mu[k[i]];
        for n in 0..2 {
            let mut sum = 0.0;
            for j in 0..i {
                if k[j] == n {
                    sum += (-beta * (t[i] - t[j])).exp();
                }
            }
            lam += alpha[k[i]][n] * sum;
        }
        ll += lam.ln();
    }
    ll -= (mu[0] + mu[1]) * t_end;
    for n in 0..2 {
        let mut tail = 0.0;
        for i in 0..t.len() {
            if k[i] == n {
                tail += (-beta * (t_end - t[i])).exp_m1();
            }
        }
        ll += ((alpha[0][n] + alpha[1][n]) / beta) * tail;
    }
    Ok(ll)
}

// ------------------------------------------------------------------ residuals

/// Time-rescaling residuals `tau_i = Lambda(t_i) - Lambda(t_{i-1})`.
///
/// With `S_i = sum_{j<i} exp(-beta (t_i - t_j))` (same recursion as the
/// likelihood), the compensator at an event time is
/// `Lambda(t_i) = mu t_i + (alpha/beta) ((i-1) - S_i)`; under the true model
/// the increments are iid Exp(1) and `Lambda` is nondecreasing.
///
/// # Errors
/// Same validation as [`uni_loglik`].
pub fn uni_residuals(times: &[f64], t_end: f64, mu: f64, alpha: f64, beta: f64) -> Result<Vec<f64>> {
    check_times(times, t_end)?;
    check_uni_params(mu, alpha, beta)?;
    let t = times;
    let n = t.len();
    let mut tau = Vec::with_capacity(n);
    let mut s = 0.0;
    let mut prev_lam = 0.0; // Lambda(t_0) = 0
    for i in 0..n {
        if i > 0 {
            s = (-beta * (t[i] - t[i - 1])).exp() * (s + 1.0);
        }
        let lam_int = mu * t[i] + (alpha / beta) * (i as f64 - s);
        tau.push(lam_int - prev_lam);
        prev_lam = lam_int;
    }
    Ok(tau)
}

/// Per-type time-rescaling residuals for the bivariate model.
///
/// Walks the merged stream keeping decayed sums `S_n` and prior event counts
/// `C_n`; at each type-`m` event (before adding it)
/// `Lambda_m(t_k) = mu_m t_k + sum_n (alpha_mn/beta)(C_n - S_n(t_k))`.
/// Residuals are per-type first differences; returns `[tau_type0, tau_type1]`.
///
/// # Errors
/// Same validation as [`biv_loglik`].
pub fn biv_residuals(
    times: &[f64],
    types: &[usize],
    t_end: f64,
    mu: &[f64; 2],
    alpha: &[[f64; 2]; 2],
    beta: f64,
) -> Result<[Vec<f64>; 2]> {
    check_times(times, t_end)?;
    check_types(types, times.len())?;
    check_biv_params(mu, alpha, beta)?;
    let (t, k) = (times, types);
    let mut s = [0.0_f64, 0.0_f64];
    let mut c = [0.0_f64, 0.0_f64];
    let mut prev_t = 0.0;
    let mut lam_int: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for i in 0..t.len() {
        let w = (-beta * (t[i] - prev_t)).exp();
        s[0] *= w;
        s[1] *= w;
        let m = k[i];
        let mut val = mu[m] * t[i];
        for n in 0..2 {
            val += (alpha[m][n] / beta) * (c[n] - s[n]);
        }
        lam_int[m].push(val);
        s[m] += 1.0;
        c[m] += 1.0;
        prev_t = t[i];
    }
    Ok(lam_int.map(|v| {
        let mut prev = 0.0;
        v.into_iter()
            .map(|x| {
                let d = x - prev;
                prev = x;
                d
            })
            .collect()
    }))
}

// ----------------------------------------------------------------- simulation

/// Simulate a univariate Hawkes process on `[0, T]` by Ogata thinning.
///
/// Pinned algorithm: after each step the decayed excitation sum `S` gives the
/// current intensity `lambda_bar = mu + alpha S`, which upper-bounds the
/// intensity until the next event (exponential decay).  Draw
/// `w ~ Exp(lambda_bar)` (one uniform), advance and decay, then accept the
/// candidate iff `u * lambda_bar <= lambda(t)` (second uniform).  Exactly two
/// uniforms per loop iteration.
///
/// # Errors
/// [`LobError::InvalidInput`] on bad params, `alpha/beta >= 1`
/// (non-stationary), or bad `T`.
pub fn simulate_uni(mu: f64, alpha: f64, beta: f64, t_end: f64, seed: u64) -> Result<Vec<f64>> {
    check_uni_params(mu, alpha, beta)?;
    if alpha / beta >= 1.0 {
        return Err(LobError::InvalidInput(format!(
            "cannot simulate non-stationary process: alpha/beta = {}",
            alpha / beta
        )));
    }
    if !(t_end.is_finite() && t_end > 0.0) {
        return Err(LobError::InvalidInput(format!(
            "T must be positive and finite, got {t_end}"
        )));
    }
    let mut rng = Lcg64::new(seed)?;
    let mut t = 0.0;
    let mut s = 0.0;
    let mut out = Vec::new();
    for _ in 0..MAX_CANDIDATES {
        let lam_bar = mu + alpha * s; // intensity just after last step
        let w = rng.exponential(lam_bar)?; // 1st uniform
        t += w;
        if t > t_end {
            return Ok(out);
        }
        s *= (-beta * w).exp();
        let lam_t = mu + alpha * s;
        if rng.uniform() * lam_bar <= lam_t {
            // 2nd uniform
            out.push(t);
            s += 1.0;
        }
    }
    Err(LobError::InvalidInput(format!(
        "simulate_uni exceeded MAX_CANDIDATES = {MAX_CANDIDATES} thinning draws"
    )))
}

/// Simulate a bivariate Hawkes process by Ogata thinning (pinned).
///
/// Same scheme as [`simulate_uni`] on the total intensity; an accepted
/// candidate with `d = u * lambda_bar` is type 0 iff `d <= lambda_0(t)`, else
/// type 1 (given `d <= lambda_0 + lambda_1`).  Exactly two uniforms per loop
/// iteration.
///
/// Returns `(times, types)`.
///
/// # Errors
/// [`LobError::InvalidInput`] on bad params, branching spectral radius >= 1,
/// or bad `T`.
pub fn simulate_biv(
    mu: &[f64; 2],
    alpha: &[[f64; 2]; 2],
    beta: f64,
    t_end: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<usize>)> {
    check_biv_params(mu, alpha, beta)?;
    let disc = (alpha[0][0] - alpha[1][1]).powi(2) + 4.0 * alpha[0][1] * alpha[1][0];
    let rad = 0.5 * (alpha[0][0] + alpha[1][1] + disc.sqrt()) / beta;
    if rad >= 1.0 {
        return Err(LobError::InvalidInput(format!(
            "cannot simulate non-stationary process: spectral radius {rad}"
        )));
    }
    if !(t_end.is_finite() && t_end > 0.0) {
        return Err(LobError::InvalidInput(format!(
            "T must be positive and finite, got {t_end}"
        )));
    }
    let mut rng = Lcg64::new(seed)?;
    let mut t = 0.0;
    let mut s = [0.0_f64, 0.0_f64];
    let mut times = Vec::new();
    let mut types = Vec::new();
    for _ in 0..MAX_CANDIDATES {
        let lam0 = mu[0] + alpha[0][0] * s[0] + alpha[0][1] * s[1];
        let lam1 = mu[1] + alpha[1][0] * s[0] + alpha[1][1] * s[1];
        let lam_bar = lam0 + lam1;
        let w = rng.exponential(lam_bar)?;
        t += w;
        if t > t_end {
            return Ok((times, types));
        }
        let decay = (-beta * w).exp();
        s[0] *= decay;
        s[1] *= decay;
        let lam0 = mu[0] + alpha[0][0] * s[0] + alpha[0][1] * s[1];
        let lam1 = mu[1] + alpha[1][0] * s[0] + alpha[1][1] * s[1];
        let d = rng.uniform() * lam_bar;
        if d <= lam0 + lam1 {
            let typ = if d <= lam0 { 0 } else { 1 };
            times.push(t);
            types.push(typ);
            s[typ] += 1.0;
        }
    }
    Err(LobError::InvalidInput(format!(
        "simulate_biv exceeded MAX_CANDIDATES = {MAX_CANDIDATES} thinning draws"
    )))
}

// ------------------------------------------------------------------------ MLE

/// Univariate MLE result.
#[derive(Debug, Clone)]
pub struct HawkesUniFit {
    /// Fitted background rate.
    pub mu: f64,
    /// Fitted excitation jump.
    pub alpha: f64,
    /// Fitted decay.
    pub beta: f64,
    /// Maximized log-likelihood.
    pub loglik: f64,
    /// Branching ratio `alpha / beta`.
    pub branching: f64,
    /// Nelder-Mead tolerance convergence flag.
    pub converged: bool,
    /// Nelder-Mead iterations.
    pub n_iter: usize,
    /// Stationarity diagnostic (near-critical / non-stationary), reported —
    /// never an error.
    pub warning: Option<String>,
}

/// Bivariate MLE result.
#[derive(Debug, Clone)]
pub struct HawkesBivFit {
    /// Fitted background rates (buy, sell).
    pub mu: [f64; 2],
    /// Fitted 2x2 excitation matrix.
    pub alpha: [[f64; 2]; 2],
    /// Fitted shared decay.
    pub beta: f64,
    /// Maximized log-likelihood.
    pub loglik: f64,
    /// Branching matrix `alpha_mn / beta`.
    pub branching: [[f64; 2]; 2],
    /// Spectral radius of the branching matrix (stationary iff < 1).
    pub spectral_radius: f64,
    /// Nelder-Mead tolerance convergence flag (second stage).
    pub converged: bool,
    /// Total Nelder-Mead iterations (both pinned stages).
    pub n_iter: usize,
    /// Stationarity diagnostic (near-critical / non-stationary), reported —
    /// never an error.
    pub warning: Option<String>,
}

fn branching_warning(value: f64) -> Option<String> {
    if value >= 1.0 {
        Some(format!(
            "fitted Hawkes process is non-stationary (branching {value:.4} >= 1)"
        ))
    } else if value >= 0.9 {
        Some(format!(
            "fitted Hawkes process is near-critical (branching {value:.4} >= 0.9)"
        ))
    } else {
        None
    }
}

/// The pure objective minimized by [`fit_uni`]: `x = (log mu, log alpha,
/// log beta)` maps to `-uni_loglik`; returns the penalty `1e10` (never
/// fails) on overflow, non-finite values, or `exp` underflow of `mu`/`beta`
/// to exactly 0.
///
/// # Errors
/// [`LobError::InvalidInput`] on invalid event times (at construction).
pub fn uni_fit_objective<'a>(times: &'a [f64], t_end: f64) -> Result<impl Fn(&[f64]) -> f64 + 'a> {
    check_times(times, t_end)?;
    Ok(move |x: &[f64]| -> f64 {
        let mu = x[0].exp();
        let alpha = x[1].exp();
        let beta = x[2].exp();
        if !(mu.is_finite() && alpha.is_finite() && beta.is_finite()) || mu <= 0.0 || beta <= 0.0 {
            return PENALTY;
        }
        let val = uni_loglik_core(times, t_end, mu, alpha, beta);
        if val.is_finite() {
            -val
        } else {
            PENALTY
        }
    })
}

/// Univariate MLE via Nelder-Mead on log-transformed parameters.
///
/// Pinned setup: minimize `-ll(exp(x))` over `x = (log mu, log alpha,
/// log beta)` from `x0 = (log(0.5 N / T), log 0.5, 0)` with
/// `xatol=1e-8, fatol=1e-10, maxiter=5000`.  The log transform enforces
/// positivity without constraints.  A near-critical (>= 0.9) or
/// non-stationary (>= 1) fitted branching ratio is reported in
/// [`HawkesUniFit::warning`] — the fit still returns.  `converged == false`
/// means the iteration cap stopped the search and the parameters are only
/// the best vertex found.
///
/// # Errors
/// [`LobError::InvalidInput`] on invalid event times.
pub fn fit_uni(times: &[f64], t_end: f64) -> Result<HawkesUniFit> {
    fit_uni_with(times, t_end, DEFAULT_MAXITER)
}

/// [`fit_uni`] with an explicit Nelder-Mead iteration cap.
///
/// # Errors
/// As [`fit_uni`]; also if `maxiter == 0`.
pub fn fit_uni_with(times: &[f64], t_end: f64, maxiter: usize) -> Result<HawkesUniFit> {
    let objective = uni_fit_objective(times, t_end)?;
    let x0 = [
        (0.5 * times.len() as f64 / t_end).ln(),
        0.5_f64.ln(),
        0.0,
    ];
    let res = nelder_mead(objective, &x0, DEFAULT_XATOL, DEFAULT_FATOL, maxiter)?;
    let (mu, alpha, beta) = (res.x[0].exp(), res.x[1].exp(), res.x[2].exp());
    let branching = alpha / beta;
    Ok(HawkesUniFit {
        mu,
        alpha,
        beta,
        loglik: -res.fval,
        branching,
        converged: res.converged,
        n_iter: res.n_iter,
        warning: branching_warning(branching),
    })
}

/// Spectral radius of a 2x2 matrix via the pinned closed form.
///
/// With trace `tr` and determinant `det`, the eigenvalues are
/// `(tr +/- sqrt(tr^2 - 4 det)) / 2`; when the discriminant is negative the
/// complex pair has modulus `sqrt(det)`.
///
/// # Errors
/// [`LobError::InvalidInput`] on non-finite entries.
pub fn spectral_radius_2x2(m: &[[f64; 2]; 2]) -> Result<f64> {
    if m.iter().flatten().any(|v| !v.is_finite()) {
        return Err(LobError::InvalidInput(
            "matrix must be a finite 2x2 array".to_string(),
        ));
    }
    let tr = m[0][0] + m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let r = disc.sqrt();
        Ok((0.5 * (tr + r)).abs().max((0.5 * (tr - r)).abs()))
    } else {
        Ok(det.sqrt())
    }
}

/// The pure objective minimized by [`fit_biv`]: `x = (log mu0, log mu1,
/// log a00, log a01, log a10, log a11, log beta)` maps to `-biv_loglik`;
/// returns the penalty `1e10` (never fails) on overflow, non-finite values,
/// or `exp` underflow of `mu0`/`mu1`/`beta` to exactly 0.
///
/// # Errors
/// [`LobError::InvalidInput`] on invalid times/types (at construction).
pub fn biv_fit_objective<'a>(
    times: &'a [f64],
    types: &'a [usize],
    t_end: f64,
) -> Result<impl Fn(&[f64]) -> f64 + 'a> {
    check_times(times, t_end)?;
    check_types(types, times.len())?;
    Ok(move |x: &[f64]| -> f64 {
        let p: Vec<f64> = x.iter().map(|v| v.exp()).collect();
        if p.iter().any(|v| !v.is_finite()) {
            return PENALTY;
        }
        // exp() can underflow to exactly 0, which the likelihood's parameter
        // validation would reject: treat it as out of domain (pure objective)
        if p[0] <= 0.0 || p[1] <= 0.0 || p[6] <= 0.0 {
            return PENALTY;
        }
        let mu = [p[0], p[1]];
        let alpha = [[p[2], p[3]], [p[4], p[5]]];
        let val = biv_loglik_core(times, types, t_end, &mu, &alpha, p[6]);
        if val.is_finite() {
            -val
        } else {
            PENALTY
        }
    })
}

/// Bivariate MLE (full 2x2 alpha, scalar beta) via pinned Nelder-Mead.
///
/// Pinned setup: parameters `x = (log mu0, log mu1, log a00, log a01,
/// log a10, log a11, log beta)`; start `mu_m = 0.5 N_m / T`, all
/// `a = 0.25`, `beta = 1`; two Nelder-Mead stages (the second restarts from
/// the first stage's optimum) with the default tolerances each — a pinned
/// deterministic polish that guards against premature simplex collapse in 7
/// dimensions.  Reports the second run's result with summed iteration counts.
/// A near-critical / non-stationary branching spectral radius is reported in
/// [`HawkesBivFit::warning`].
///
/// # Errors
/// [`LobError::InvalidInput`] on invalid input, or when either event type is
/// absent.
pub fn fit_biv(times: &[f64], types: &[usize], t_end: f64) -> Result<HawkesBivFit> {
    fit_biv_with(times, types, t_end, DEFAULT_MAXITER)
}

/// [`fit_biv`] with an explicit per-stage Nelder-Mead iteration cap.
///
/// # Errors
/// As [`fit_biv`]; also if `maxiter == 0`.
pub fn fit_biv_with(
    times: &[f64],
    types: &[usize],
    t_end: f64,
    maxiter: usize,
) -> Result<HawkesBivFit> {
    check_times(times, t_end)?;
    check_types(types, times.len())?;
    let n0 = types.iter().filter(|&&k| k == 0).count();
    let n1 = times.len() - n0;
    if n0 == 0 || n1 == 0 {
        return Err(LobError::InvalidInput(
            "both event types must be present to fit the bivariate model".to_string(),
        ));
    }
    let objective = biv_fit_objective(times, types, t_end)?;
    let x0 = [
        (0.5 * n0 as f64 / t_end).ln(),
        (0.5 * n1 as f64 / t_end).ln(),
        0.25_f64.ln(),
        0.25_f64.ln(),
        0.25_f64.ln(),
        0.25_f64.ln(),
        0.0,
    ];
    let res1 = nelder_mead(&objective, &x0, DEFAULT_XATOL, DEFAULT_FATOL, maxiter)?;
    let res2: NmResult = nelder_mead(&objective, &res1.x, DEFAULT_XATOL, DEFAULT_FATOL, maxiter)?;
    let p: Vec<f64> = res2.x.iter().map(|v| v.exp()).collect();
    let mu = [p[0], p[1]];
    let alpha = [[p[2], p[3]], [p[4], p[5]]];
    let beta = p[6];
    let branching = [
        [alpha[0][0] / beta, alpha[0][1] / beta],
        [alpha[1][0] / beta, alpha[1][1] / beta],
    ];
    let rad = spectral_radius_2x2(&branching)?;
    Ok(HawkesBivFit {
        mu,
        alpha,
        beta,
        loglik: -res2.fval,
        branching,
        spectral_radius: rad,
        converged: res2.converged,
        n_iter: res1.n_iter + res2.n_iter,
        warning: branching_warning(rad),
    })
}

// ------------------------------------------------------------------ CSV i/o

/// Read a one-column event-time CSV with header `t` (pinned rules of
/// [`crate::csv`]; values must be finite — ordering is validated by the
/// likelihood functions).
///
/// # Errors
/// [`LobError::Io`] / [`LobError::Parse`] on any violation.
pub fn read_times_csv(path: &str) -> Result<Vec<f64>> {
    let lines = read_csv_lines(path, "t")?;
    let mut out = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let context = format!("time row {}", i + 1);
        if line.contains(',') {
            return Err(LobError::Parse(format!(
                "malformed {context}: expected 1 field: {line:?}"
            )));
        }
        out.push(parse_double_strict(line, &context)?);
    }
    Ok(out)
}

/// Read a two-column `t,type` CSV; types are validated to be 0 or 1.
///
/// # Errors
/// [`LobError::Io`] / [`LobError::Parse`] on any violation.
pub fn read_marked_times_csv(path: &str) -> Result<(Vec<f64>, Vec<usize>)> {
    let lines = read_csv_lines(path, "t,type")?;
    let mut times = Vec::with_capacity(lines.len());
    let mut types = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let context = format!("marked-time row {}", i + 1);
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 2 {
            return Err(LobError::Parse(format!(
                "malformed {context}: expected 2 fields: {line:?}"
            )));
        }
        let k = parse_int64_strict(fields[1], &context)?;
        if k != 0 && k != 1 {
            return Err(LobError::Parse(format!(
                "types must be 0 or 1, got {k} in {context}"
            )));
        }
        times.push(parse_double_strict(fields[0], &context)?);
        types.push(k as usize);
    }
    Ok((times, types))
}

fn write_text(path: &str, text: &str) -> Result<()> {
    let mut file = fs::File::create(path).map_err(|e| LobError::Io(format!("{path}: {e}")))?;
    file.write_all(text.as_bytes())
        .map_err(|e| LobError::Io(format!("{path}: {e}")))
}

/// Write event times with header `t`; every value round-trips exactly
/// (Rust's shortest round-trip `Display` for `f64`).
///
/// # Errors
/// [`LobError::InvalidInput`] on non-finite values; [`LobError::Io`] on
/// file problems.
pub fn write_times_csv(path: &str, times: &[f64]) -> Result<()> {
    if times.iter().any(|v| !v.is_finite()) {
        return Err(LobError::InvalidInput("times must be finite".to_string()));
    }
    let mut out = String::from("t\n");
    for v in times {
        out.push_str(&format!("{v:?}\n"));
    }
    write_text(path, &out)
}

/// Write `t,type` rows (exact round-trip times, integer types 0/1).
///
/// # Errors
/// [`LobError::InvalidInput`] on non-finite times or invalid types;
/// [`LobError::Io`] on file problems.
pub fn write_marked_times_csv(path: &str, times: &[f64], types: &[usize]) -> Result<()> {
    if times.iter().any(|v| !v.is_finite()) {
        return Err(LobError::InvalidInput("times must be finite".to_string()));
    }
    check_types(types, times.len())?;
    let mut out = String::from("t,type\n");
    for (v, k) in times.iter().zip(types) {
        out.push_str(&format!("{v:?},{k}\n"));
    }
    write_text(path, &out)
}
