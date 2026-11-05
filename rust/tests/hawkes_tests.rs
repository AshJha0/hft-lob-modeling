//! Hawkes process tests: recursion vs brute force, pinned simulation
//! (including bit-reproduction of the bundled samples), MLE refits,
//! near-critical warning, time-rescaling residuals and KS goodness of fit.

mod common;

use common::{biv_data, uni_times};
use lobmod::{
    biv_fit_objective, biv_loglik, biv_loglik_brute, biv_residuals, fit_biv, fit_biv_with, fit_uni,
    fit_uni_with, ks_test_exp, nelder_mead, read_marked_times_csv, read_times_csv, simulate_biv,
    simulate_uni, spectral_radius_2x2, uni_fit_objective, uni_loglik, uni_loglik_brute,
    uni_residuals, write_marked_times_csv, write_times_csv,
};

const T_UNI: f64 = 1000.0;
const T_BIV: f64 = 600.0;

// -------------------------------------------------------- likelihood recursion

#[test]
fn uni_recursion_matches_brute_force() {
    let t = uni_times();
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
    let (t, k) = biv_data();
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
    let bundled = uni_times();
    let ours = simulate_uni(0.5, 1.0, 1.6, T_UNI, 101).unwrap();
    assert_eq!(ours.len(), 1211);
    assert_eq!(ours, bundled);
}

#[test]
fn simulate_biv_reproduces_bundled_sample() {
    let (bundled_t, bundled_k) = biv_data();
    let (t, k) = simulate_biv(&[0.3, 0.3], &[[0.6, 0.3], [0.25, 0.5]], 2.0, T_BIV, 33).unwrap();
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
    let t = uni_times();
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
    let (t, k) = biv_data();
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
    let t = uni_times();
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
    let (t, k) = biv_data();
    let mu = [0.3, 0.3];
    let alpha = [[0.6, 0.3], [0.25, 0.5]];
    let taus = biv_residuals(&t, &k, T_BIV, &mu, &alpha, 2.0).unwrap();
    let n0 = k.iter().filter(|&&v| v == 0).count();
    assert_eq!(taus[0].len(), n0);
    assert_eq!(taus[1].len(), t.len() - n0);
    for tau in &taus {
        assert!(
            tau.iter().all(|&v| v > 0.0),
            "compensator monotone per type"
        );
        let (_, p) = ks_test_exp(tau).unwrap();
        assert!(p > 0.05, "true-param residuals should pass, p = {p}");
    }
}

// ---------------------------------------------------------------- review additions

fn scratch(name: &str, text: &str) -> String {
    let path = std::env::temp_dir().join(format!("lobmod_rs_h_{name}"));
    std::fs::write(&path, text).expect("scratch file");
    path.to_string_lossy().into_owned()
}

#[test]
fn fit_objectives_are_pure_under_underflow() {
    // PT-6
    let (t, k) = biv_data();
    let obj = biv_fit_objective(&t, &k, T_BIV).unwrap();
    assert_eq!(obj(&[-800.0, -800.0, 0.0, 0.0, 0.0, 0.0, -800.0]), 1e10);
    assert_eq!(obj(&[-800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1e10); // mu0 underflow
    assert_eq!(obj(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -800.0]), 1e10); // beta underflow
    assert_eq!(obj(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 800.0]), 1e10); // overflow
    let (mu0, mu1) = (0.5 * 323.0 / T_BIV, 0.5 * 288.0 / T_BIV);
    let x0 = [
        mu0.ln(),
        mu1.ln(),
        0.25_f64.ln(),
        0.25_f64.ln(),
        0.25_f64.ln(),
        0.25_f64.ln(),
        0.0,
    ];
    let want = -biv_loglik(
        &t,
        &k,
        T_BIV,
        &[mu0, mu1],
        &[[0.25, 0.25], [0.25, 0.25]],
        1.0,
    )
    .unwrap();
    assert!((obj(&x0) - want).abs() < 1e-12);
    let res = nelder_mead(
        &obj,
        &[-800.0, -800.0, 0.0, 0.0, 0.0, 0.0, -800.0],
        1e-8,
        1e-10,
        5,
    )
    .unwrap();
    assert!(res.fval.is_finite());
    let tu = uni_times();
    let uobj = uni_fit_objective(&tu, T_UNI).unwrap();
    assert_eq!(uobj(&[-800.0, 0.0, 0.0]), 1e10);
    assert_eq!(uobj(&[0.0, 0.0, -800.0]), 1e10);
    let poisson = -uni_loglik(&tu, T_UNI, 1.0, 0.0, 1.0).unwrap();
    assert!((uobj(&[0.0, -800.0, 0.0]) - poisson).abs() < 1e-12);
    let capped = fit_biv_with(&t, &k, T_BIV, 1).unwrap();
    assert!(!capped.converged);
    assert_eq!(capped.n_iter, 2); // two stages x 1 iteration
    let capped_u = fit_uni_with(&tu, T_UNI, 1).unwrap();
    assert!(!capped_u.converged);
    assert_eq!(capped_u.n_iter, 1);
    assert!(fit_uni_with(&tu, T_UNI, 0).is_err());
}

#[test]
fn poisson_limit_of_fit() {
    // PT-8: alpha = 0 is a homogeneous Poisson process; the MLE must find it
    let t_end = 2000.0;
    let t = simulate_uni(1.0, 0.0, 1.0, t_end, 5).unwrap();
    assert!(t.len() > 1500);
    let fit = fit_uni(&t, t_end).unwrap();
    assert!(fit.converged);
    assert!(fit.branching < 0.05);
    let n = t.len() as f64;
    let poisson_ll = uni_loglik(&t, t_end, n / t_end, 0.0, 1.0).unwrap();
    assert!(fit.loglik >= poisson_ll - 1e-6);
    assert!((poisson_ll - (n * (n / t_end).ln() - n)).abs() < 1e-9);
    assert_eq!(
        uni_loglik(&t, t_end, 0.7, 0.0, 1.0).unwrap(),
        uni_loglik(&t, t_end, 0.7, 0.0, 3.3).unwrap()
    );
}

#[test]
fn single_event_paths() {
    // PT-9
    let (mu, alpha, beta): (f64, f64, f64) = (0.5, 0.8, 1.5);
    let t = [0.7];
    let expected = mu.ln() - mu * 1.0 + (alpha / beta) * (-beta * 0.3_f64).exp_m1();
    assert!((uni_loglik(&t, 1.0, mu, alpha, beta).unwrap() - expected).abs() < 1e-15);
    assert!((uni_loglik_brute(&t, 1.0, mu, alpha, beta).unwrap() - expected).abs() < 1e-15);
    let tau = uni_residuals(&t, 1.0, mu, alpha, beta).unwrap();
    assert_eq!(tau.len(), 1);
    assert!((tau[0] - mu * 0.7).abs() < 1e-15);
    let (d, p) = ks_test_exp(&tau).unwrap();
    assert!(d > 0.0 && d < 1.0 && p > 0.0 && p <= 1.0);
    let fit = fit_uni(&t, 1.0).unwrap();
    assert!(fit.loglik.is_finite() && fit.mu > 0.0 && fit.n_iter >= 1);
    let t_end = [1.0]; // event exactly at the horizon: expm1(0) = 0
    assert!((uni_loglik(&t_end, 1.0, mu, alpha, beta).unwrap() - (mu.ln() - mu)).abs() < 1e-15);
    assert!((uni_residuals(&t_end, 1.0, mu, alpha, beta).unwrap()[0] - mu).abs() < 1e-15);
    assert!(fit_biv(&t_end, &[0], 1.0).is_err());
    let ll = biv_loglik(
        &t_end,
        &[0],
        1.0,
        &[0.2, 0.3],
        &[[0.1, 0.1], [0.1, 0.1]],
        2.0,
    )
    .unwrap();
    assert!((ll - (0.2_f64.ln() - 0.5)).abs() < 1e-15);
}

#[test]
fn times_csv_round_trip_and_robustness() {
    let tu = uni_times();
    let p = scratch("u.csv", "");
    write_times_csv(&p, &tu).unwrap();
    assert_eq!(read_times_csv(&p).unwrap(), tu);
    let (t, k) = biv_data();
    let q = scratch("b.csv", "");
    write_marked_times_csv(&q, &t, &k).unwrap();
    assert_eq!(read_marked_times_csv(&q).unwrap(), (t.clone(), k.clone()));

    let body = "t\n0.5\n1.25e0\n";
    let want = vec![0.5, 1.25];
    assert_eq!(
        read_times_csv(&scratch("crlf.csv", &body.replace('\n', "\r\n"))).unwrap(),
        want
    );
    assert_eq!(
        read_times_csv(&scratch("trail.csv", &format!("{body}\n"))).unwrap(),
        want
    );
    for (name, text) in [
        ("nan.csv", format!("{body}nan\n")),
        ("inf.csv", format!("{body}inf\n")),
        ("two.csv", format!("{body}1.5,0\n")),
        ("word.csv", format!("{body}abc\n")),
        ("hdr.csv", "time\n0.5\n".to_string()),
        ("blank.csv", "t\n0.5\n\n1.0\n".to_string()),
        ("space.csv", "t\n 0.5\n".to_string()),
    ] {
        assert!(read_times_csv(&scratch(name, &text)).is_err(), "{name}");
    }
    let mbody = "t,type\n0.5,0\n1.0,1\n";
    assert_eq!(
        read_marked_times_csv(&scratch("m.csv", mbody)).unwrap(),
        (vec![0.5, 1.0], vec![0, 1])
    );
    for (name, text) in [
        ("type2.csv", format!("{mbody}1.5,2\n")),
        ("typef.csv", format!("{mbody}1.5,1.0\n")),
        ("one.csv", format!("{mbody}1.5\n")),
        ("three.csv", format!("{mbody}1.5,1,\n")),
        ("mhdr.csv", "type,t\n0,0.5\n".to_string()),
    ] {
        assert!(
            read_marked_times_csv(&scratch(name, &text)).is_err(),
            "{name}"
        );
    }
    assert!(write_times_csv(&scratch("x.csv", ""), &[1.0, f64::NAN]).is_err());
    assert!(write_marked_times_csv(&scratch("y.csv", ""), &[1.0, 2.0], &[0, 2]).is_err());
}
