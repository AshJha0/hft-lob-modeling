//! KS goodness-of-fit tests: statistic values, pinned p-value formula, tiny
//! samples, monotonicity in D, validation errors.

use lobmod::{kolmogorov_q, ks_pvalue, ks_statistic_exp, ks_test_exp};

#[test]
fn ks_statistic_single_point() {
    // n = 1, x = ln 2: F(x) = 0.5, D = max(1 - 0.5, 0.5 - 0) = 0.5
    let d = ks_statistic_exp(&[std::f64::consts::LN_2]).unwrap();
    assert!((d - 0.5).abs() < 1e-15);
}

#[test]
fn ks_tiny_samples_work() {
    // tiny N is allowed: the asymptotic p is indicative only, not an error
    let (d1, p1) = ks_test_exp(&[0.5]).unwrap();
    assert!((0.0..=1.0).contains(&d1) && (0.0..=1.0).contains(&p1));
    let (d3, p3) = ks_test_exp(&[0.1, 0.9, 2.5]).unwrap();
    assert!((0.0..=1.0).contains(&d3) && (0.0..=1.0).contains(&p3));
}

#[test]
fn ks_statistic_is_sorted_and_order_free() {
    let a = ks_statistic_exp(&[2.0, 0.1, 1.0]).unwrap();
    let b = ks_statistic_exp(&[0.1, 1.0, 2.0]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn ks_pvalue_pinned_formula() {
    // d = 0 is exactly p = 1
    assert_eq!(ks_pvalue(0.0, 10).unwrap(), 1.0);
    // hand-evaluated 100-term series for d = 0.2, n = 25:
    // lambda = (5 + 0.12 + 0.11/5) * 0.2 = 1.0284
    let lam: f64 = (5.0 + 0.12 + 0.11 / 5.0) * 0.2;
    let mut total = 0.0;
    for j in 1..=100u32 {
        let jj = j as f64;
        let term = (-2.0 * jj * jj * lam * lam).exp();
        total += if j % 2 == 1 { term } else { -term };
    }
    let expected = (2.0 * total).clamp(0.0, 1.0);
    assert!((ks_pvalue(0.2, 25).unwrap() - expected).abs() < 1e-15);
}

#[test]
fn ks_pvalue_monotone_in_d() {
    let n = 50;
    let mut prev = ks_pvalue(0.01, n).unwrap();
    for d in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8] {
        let p = ks_pvalue(d, n).unwrap();
        assert!((0.0..=1.0).contains(&p));
        assert!(p <= prev, "p must not increase with D: p({d}) = {p} > {prev}");
        prev = p;
    }
    assert!(prev < 1e-9, "huge D is a certain rejection");
}

#[test]
fn ks_validation_errors() {
    assert!(ks_statistic_exp(&[]).is_err());
    assert!(ks_statistic_exp(&[0.5, -0.1]).is_err());
    assert!(ks_statistic_exp(&[0.5, f64::NAN]).is_err());
    assert!(ks_pvalue(-0.1, 10).is_err());
    assert!(ks_pvalue(1.1, 10).is_err());
    assert!(ks_pvalue(f64::NAN, 10).is_err());
    assert!(ks_pvalue(0.5, 0).is_err());
}

#[test]
fn ks_detects_wrong_distribution() {
    // a clearly non-Exp(1) sample (all mass near 3) is strongly rejected
    let sample: Vec<f64> = (0..200).map(|i| 3.0 + 0.001 * i as f64).collect();
    let (d, p) = ks_test_exp(&sample).unwrap();
    assert!(d > 0.5);
    assert!(p < 1e-6);
}

// ---------------------------------------------------------------- review additions

#[test]
fn ks_pvalue_small_lambda_accuracy() {
    // PT-10: the dual series makes tiny statistics give p = 1, not 0.87
    assert!((ks_pvalue(1e-4, 100).unwrap() - 1.0).abs() < 1e-9);
    let n = 100_000usize;
    assert!(ks_pvalue(0.5 / n as f64, n).unwrap() >= 0.999);
    let mut prev = 1.0;
    for m in [0.5, 2.0, 8.0, 32.0, 128.0, 512.0] {
        let p = ks_pvalue(m / n as f64, n).unwrap();
        assert!(p <= prev + 1e-15); // monotone across the branch switch
        prev = p;
    }
    assert_eq!(kolmogorov_q(0.01), 1.0);
    assert_eq!(kolmogorov_q(0.005), 1.0);
    for lam in [0.0999, 0.1, 0.1001] {
        assert!((kolmogorov_q(lam) - 1.0).abs() < 1e-14);
    }
    // reference values (scipy.special.kolmogorov)
    assert!((kolmogorov_q(0.5) - 0.963945243664875).abs() < 1e-14);
    assert!((kolmogorov_q(1.0) - 0.269999671677355).abs() < 1e-14);
    assert!((kolmogorov_q(2.0) - 0.000670925255780).abs() < 1e-14);
}
