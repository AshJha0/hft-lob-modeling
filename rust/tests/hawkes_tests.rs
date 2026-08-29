//! Hawkes process tests: recursion vs brute force, pinned simulation
//! (including bit-reproduction of the bundled samples), MLE refits,
//! near-critical warning, time-rescaling residuals and KS goodness of fit.

mod common;

use common::read_hawkes_csv;
use lobmod::{
    biv_loglik, biv_loglik_brute, biv_residuals, fit_biv, fit_uni, ks_test_exp, simulate_biv,
    simulate_uni, spectral_radius_2x2, uni_loglik, uni_loglik_brute, uni_residuals,
};

const T_UNI: f64 = 1000.0;
const T_BIV: f64 = 600.0;

// -------------------------------------------------------- likelihood recursion

#[test]
fn uni_recursion_matches_brute_force() {
    let (t, _) = read_hawkes_csv("hawkes_uni.csv");
    for (mu, alpha, beta) in [(0.4, 0.6, 1.2), (0.5, 1.0, 1.6), (1.0, 0.2, 3.0)] {
        let fast = uni_loglik(&t, T_UNI, mu, alpha, beta).unwrap();
        let brute = uni_loglik_brute(&t, T_UNI, mu, alpha, beta).unwrap();
        assert!(
            (fast - brute).abs() <= 1e-10,
            "({mu},{alpha},{beta}): {fast} vs {brute}"
        );
    }
}

#[test]
fn biv_recursion_matches_brute_force() {
    let (t, k) = read_hawkes_csv("hawkes_biv.csv");
    let mu = [0.25, 0.25];
    let alpha = [[0.5, 0.2], [0.2, 0.5]];
    let fast = biv_loglik(&t, &k, T_BIV, &mu, &alpha, 1.8).unwrap();
    let brute = biv_loglik_brute(&t, &k, T_BIV, &mu, &alpha, 1.8).unwrap();
    assert!((fast - brute).abs() <= 1e-10, "{fast} vs {brute}");
}

#[test]
fn loglik_validation_errors() {
    assert!(uni_loglik(&[], 10.0, 0.5, 0.5, 1.0).is_err()); // empty
    assert!(uni_loglik(&[1.0, 1.0], 10.0, 0.5, 0.5, 1.0).is_err()); // not increasing
    assert!(uni_loglik(&[-1.0, 2.0], 10.0, 0.5, 0.5, 1.0).is_err()); // t1 <= 0
    assert!(uni_loglik(&[1.0, 20.0], 10.0, 0.5, 0.5, 1.0).is_err()); // t_N > T
    assert!(uni_loglik(&[1.0, 2.0], -1.0, 0.5, 0.5, 1.0).is_err()); // bad T
    assert!(uni_loglik(&[1.0, 2.0], 10.0, 0.0, 0.5, 1.0).is_err()); // mu <= 0
    assert!(uni_loglik(&[1.0, 2.0], 10.0, 0.5, -0.1, 1.0).is_err()); // alpha < 0
    assert!(uni_loglik(&[1.0, 2.0], 10.0, 0.5, 0.5, 0.0).is_err()); // beta <= 0
    let mu = [0.3, 0.3];
    let a = [[0.1, 0.1], [0.1, 0.1]];
    assert!(biv_loglik(&[1.0, 2.0], &[0], 10.0, &mu, &a, 1.0).is_err()); // len mismatch
    assert!(biv_loglik(&[1.0, 2.0], &[0, 2], 10.0, &mu, &a, 1.0).is_err()); // bad type
    assert!(biv_loglik(&[1.0, 2.0], &[0, 1], 10.0, &[0.0, 0.3], &a, 1.0).is_err());
}

// ------------------------------------------------------------------ simulation

#[test]
fn simulation_is_deterministic_per_seed() {
    let a = simulate_uni(0.5, 1.0, 1.6, 100.0, 42).unwrap();
    let b = simulate_uni(0.5, 1.0, 1.6, 100.0, 42).unwrap();
    assert_eq!(a, b);
    let c = simulate_uni(0.5, 1.0, 1.6, 100.0, 43).unwrap();
    assert_ne!(a, c);
}

#[test]
fn simulate_uni_reproduces_bundled_sample() {
    // data/hawkes_uni.csv was written by the pinned generator with %.17g
    // (lossless), so the Rust simulation must reproduce it bit-for-bit.
    let (bundled, _) = read_hawkes_csv("hawkes_uni.csv");
    let ours = simulate_uni(0.5, 1.0, 1.6, T_UNI, 101).unwrap();
    assert_eq!(ours.len(), 1211);
    assert_eq!(ours, bundled);
}

#[test]
fn simulate_biv_reproduces_bundled_sample() {
    let (bundled_t, bundled_k) = read_hawkes_csv("hawkes_biv.csv");
    let (t, k) = simulate_biv(
        &[0.3, 0.3],
        &[[0.6, 0.3], [0.25, 0.5]],
        2.0,
        T_BIV,
        33,
    )
    .unwrap();
    assert_eq!(t.len(), 611);
    assert_eq!(t, bundled_t);
    assert_eq!(k, bundled_k);
}

#[test]
fn simulate_rejects_non_stationary_params() {
    assert!(simulate_uni(0.5, 2.0, 1.6, 100.0, 1).is_err()); // alpha/beta >= 1
    assert!(simulate_uni(0.5, 0.5, 1.0, 0.0, 1).is_err()); // bad T
    assert!(simulate_biv(&[0.3, 0.3], &[[3.0, 0.0], [0.0, 3.0]], 2.0, 100.0, 1).is_err());
}

// ------------------------------------------------------------------------- MLE

#[test]
fn uni_refit_recovers_dgp_loosely() {
    let (t, _) = read_hawkes_csv("hawkes_uni.csv");
    let fit = fit_uni(&t, T_UNI).unwrap();
    assert!(fit.converged);
    assert!((fit.mu - 0.5).abs() < 0.1, "mu {}", fit.mu);
    assert!((fit.alpha - 1.0).abs() < 0.15, "alpha {}", fit.alpha);
    assert!((fit.beta - 1.6).abs() < 0.25, "beta {}", fit.beta);
    assert!(fit.branching < 1.0);
    assert!(fit.warning.is_none(), "comfortably subcritical fit");
    // the reported loglik is the likelihood at the fitted params
    let ll = uni_loglik(&t, T_UNI, fit.mu, fit.alpha, fit.beta).unwrap();
    assert!((ll - fit.loglik).abs() < 1e-8);
}

#[test]
fn biv_refit_recovers_dgp_loosely() {
    let (t, k) = read_hawkes_csv("hawkes_biv.csv");
    let fit = fit_biv(&t, &k, T_BIV).unwrap();
    let truth = [[0.6, 0.3], [0.25, 0.5]];
    for m in 0..2 {
        for n in 0..2 {
            assert!(
                (fit.alpha[m][n] - truth[m][n]).abs() < 0.2,
                "alpha[{m}][{n}] = {}",
                fit.alpha[m][n]
            );
        }
    }
    assert!((fit.mu[0] - 0.3).abs() < 0.15 && (fit.mu[1] - 0.3).abs() < 0.15);
    assert!((fit.beta - 2.0).abs() < 0.6);
    assert!(fit.spectral_radius < 1.0);
    assert!(fit.warning.is_none());
}

#[test]
fn near_critical_fit_succeeds_with_warning() {
    // pinned near-critical DGP: branching 1.9/2.0 = 0.95
    let t = simulate_uni(0.2, 1.9, 2.0, 150.0, 999).unwrap();
    let fit = fit_uni(&t, 150.0).unwrap();
    assert!(fit.branching >= 0.9, "branching {}", fit.branching);
    let warning = fit.warning.expect("near-critical fit must warn");
    assert!(warning.contains("near-critical") || warning.contains("non-stationary"));
}

#[test]
fn fit_biv_requires_both_types() {
    let t = vec![1.0, 2.0, 3.0];
    assert!(fit_biv(&t, &[0, 0, 0], 10.0).is_err());
    assert!(fit_biv(&t, &[1, 1, 1], 10.0).is_err());
}

#[test]
fn spectral_radius_closed_form() {
    // symmetric: eigenvalues 0.7 and 0.3
    assert!((spectral_radius_2x2(&[[0.5, 0.2], [0.2, 0.5]]).unwrap() - 0.7).abs() < 1e-15);
    // rotation-like: complex pair with modulus sqrt(det) = 1
    assert!((spectral_radius_2x2(&[[0.0, -1.0], [1.0, 0.0]]).unwrap() - 1.0).abs() < 1e-15);
    assert!(spectral_radius_2x2(&[[f64::NAN, 0.0], [0.0, 0.0]]).is_err());
}

// ------------------------------------------------------- residuals + KS on fit

#[test]
fn residuals_at_true_params_behave() {
    let (t, _) = read_hawkes_csv("hawkes_uni.csv");
    let tau = uni_residuals(&t, T_UNI, 0.5, 1.0, 1.6).unwrap();
    assert_eq!(tau.len(), t.len());
    // compensator is monotone -> residuals strictly positive
    assert!(tau.iter().all(|&v| v > 0.0));
    // Exp(1) residuals have mean 1
    let mean = tau.iter().sum::<f64>() / tau.len() as f64;
    assert!((mean - 1.0).abs() < 0.1, "mean residual {mean}");
    // KS fails to reject at the true parameters...
    let (_, p) = ks_test_exp(&tau).unwrap();
    assert!(p > 0.05, "p at true params {p}");
    // ...and rejects at badly wrong parameters
    let tau_wrong = uni_residuals(&t, T_UNI, 1.5, 0.1, 3.0).unwrap();
    let (_, p_wrong) = ks_test_exp(&tau_wrong).unwrap();
    assert!(p_wrong < 0.01, "p at wrong params {p_wrong}");
}

#[test]
fn biv_residuals_per_type_behave() {
    let (t, k) = read_hawkes_csv("hawkes_biv.csv");
    let mu = [0.3, 0.3];
    let alpha = [[0.6, 0.3], [0.25, 0.5]];
    let taus = biv_residuals(&t, &k, T_BIV, &mu, &alpha, 2.0).unwrap();
    let n0 = k.iter().filter(|&&v| v == 0).count();
    assert_eq!(taus[0].len(), n0);
    assert_eq!(taus[1].len(), t.len() - n0);
    for tau in &taus {
        assert!(tau.iter().all(|&v| v > 0.0), "compensator monotone per type");
        let (_, p) = ks_test_exp(tau).unwrap();
        assert!(p > 0.05, "true-param residuals should pass, p = {p}");
    }
}
