//! Golden-value suite: recompute every case of `data/golden/golden.json` and
//! compare within tolerance.  The same cases are asserted by the Python, C++
//! and Java test suites, which is what pins cross-language agreement.

mod common;

use std::sync::OnceLock;

use common::{check_case, load_golden, read_hawkes_csv};
use lobmod::{
    accuracy, biv_loglik, chronological_split, compute_features, fit_biv, fit_uni, ks_test_exp,
    load_feature_config, majority_class, ofi_at, read_events_csv, replay, signal_pnl,
    uni_loglik, uni_loglik_brute, uni_residuals, BookStates, FeatureSet, HawkesBivFit,
    HawkesUniFit, Matrix, Mlp, Standardizer,
};

const T_UNI: f64 = 1000.0;
const T_BIV: f64 = 600.0;

fn states() -> &'static BookStates {
    static STATES: OnceLock<BookStates> = OnceLock::new();
    STATES.get_or_init(|| {
        let events = read_events_csv(&common::data_path("lob_events.csv")).expect("events");
        replay(&events, 3, 1.0).expect("replay")
    })
}

fn uni_times() -> &'static Vec<f64> {
    static TIMES: OnceLock<Vec<f64>> = OnceLock::new();
    TIMES.get_or_init(|| read_hawkes_csv("hawkes_uni.csv").0)
}

fn uni_fit() -> &'static HawkesUniFit {
    static FIT: OnceLock<HawkesUniFit> = OnceLock::new();
    FIT.get_or_init(|| fit_uni(uni_times(), T_UNI).expect("uni fit"))
}

fn biv_data() -> &'static (Vec<f64>, Vec<usize>) {
    static DATA: OnceLock<(Vec<f64>, Vec<usize>)> = OnceLock::new();
    DATA.get_or_init(|| read_hawkes_csv("hawkes_biv.csv"))
}

fn biv_fit() -> &'static HawkesBivFit {
    static FIT: OnceLock<HawkesBivFit> = OnceLock::new();
    FIT.get_or_init(|| {
        let (t, k) = biv_data();
        fit_biv(t, k, T_BIV).expect("biv fit")
    })
}

/// The full learning pipeline (features -> split -> standardize -> train),
/// computed once and shared by the NN golden cases.
struct Pipeline {
    fs: FeatureSet,
    test_start: usize,
    x_test: Matrix,
    y_test: Vec<usize>,
    hist: Vec<f64>,
    pred_nn: Vec<usize>,
    pred_lg: Vec<usize>,
    majority: usize,
}

fn pipeline() -> &'static Pipeline {
    static PIPE: OnceLock<Pipeline> = OnceLock::new();
    PIPE.get_or_init(|| {
        let cfg = load_feature_config(&common::data_path("features_config.json")).expect("config");
        let fs = compute_features(states(), &cfg).expect("features");
        let (tr, te) = chronological_split(fs.len(), cfg.train_frac).expect("split");
        let x_train_raw = fs.x.slice_rows(tr.start, tr.end);
        let x_test_raw = fs.x.slice_rows(te.start, te.end);
        let mut std = Standardizer::new();
        std.fit(&x_train_raw).expect("fit standardizer");
        let x_train = std.transform(&x_train_raw).expect("transform train");
        let x_test = std.transform(&x_test_raw).expect("transform test");
        let y_train = fs.y[tr.clone()].to_vec();
        let y_test = fs.y[te.clone()].to_vec();
        let mut net = Mlp::new(5, &cfg.hidden, 3).expect("mlp");
        let hist = net
            .fit(&x_train, &y_train, cfg.learning_rate, cfg.epochs)
            .expect("train mlp");
        let mut logistic = Mlp::new(5, &[], 3).expect("logistic");
        logistic
            .fit(&x_train, &y_train, cfg.learning_rate, cfg.epochs)
            .expect("train logistic");
        Pipeline {
            test_start: te.start,
            x_test: x_test.clone(),
            y_test,
            hist,
            pred_nn: net.predict(&x_test).expect("predict nn"),
            pred_lg: logistic.predict(&x_test).expect("predict logistic"),
            majority: majority_class(&y_train).expect("majority"),
            fs,
        }
    })
}

#[test]
fn golden_book_snapshots() {
    let golden = load_golden();
    let st = states();
    for seq in [1000usize, 3000, 5000] {
        let case = &golden[&format!("book_snapshot_event_{seq}")];
        let i = seq - 1;
        check_case(
            case,
            &[
                ("best_bid", st.best_bid[i]),
                ("best_ask", st.best_ask[i]),
                ("bid_size", st.bid_size[i]),
                ("ask_size", st.ask_size[i]),
                ("bid_depth_k", st.bid_depth_k[i]),
                ("ask_depth_k", st.ask_depth_k[i]),
                ("mid", st.mid[i]),
            ],
        );
    }
}

#[test]
fn golden_microprice() {
    let golden = load_golden();
    let case = &golden["microprice_event_2500"];
    let event = case["inputs"]["event"].as_f64().unwrap() as usize;
    check_case(case, &[("microprice", states().microprice[event - 1])]);
}

#[test]
fn golden_ofi() {
    let golden = load_golden();
    let case = &golden["ofi_event_2200_w100"];
    let event = case["inputs"]["event"].as_f64().unwrap() as usize;
    let window = case["inputs"]["window"].as_f64().unwrap() as usize;
    let ofi = ofi_at(states(), event - 1, window).expect("ofi");
    check_case(case, &[("ofi", ofi)]);
}

#[test]
fn golden_hawkes_uni_loglik_pinned() {
    let golden = load_golden();
    let case = &golden["hawkes_uni_loglik_pinned"];
    let p = &case["inputs"];
    let (mu, alpha, beta, t_end) = (
        p["mu"].as_f64().unwrap(),
        p["alpha"].as_f64().unwrap(),
        p["beta"].as_f64().unwrap(),
        p["T"].as_f64().unwrap(),
    );
    let ll = uni_loglik(uni_times(), t_end, mu, alpha, beta).expect("loglik");
    check_case(case, &[("loglik", ll)]);
    // and the recursion agrees with brute force on the full bundled sample
    let brute = uni_loglik_brute(uni_times(), t_end, mu, alpha, beta).expect("brute");
    assert!((ll - brute).abs() <= 1e-10, "recursion {ll} vs brute {brute}");
}

#[test]
fn golden_hawkes_uni_fit_and_branching() {
    let golden = load_golden();
    let fit = uni_fit();
    check_case(
        &golden["hawkes_uni_fit"],
        &[("mu", fit.mu), ("alpha", fit.alpha), ("beta", fit.beta)],
    );
    check_case(&golden["hawkes_uni_branching"], &[("branching", fit.branching)]);
    assert!(fit.converged, "univariate NM should converge");
}

#[test]
fn golden_hawkes_uni_ks_fitted() {
    let golden = load_golden();
    let fit = uni_fit();
    let tau = uni_residuals(uni_times(), T_UNI, fit.mu, fit.alpha, fit.beta).expect("residuals");
    let (d, p) = ks_test_exp(&tau).expect("ks");
    check_case(
        &golden["hawkes_uni_ks_fitted"],
        &[("ks_stat", d), ("p_value", p)],
    );
}

#[test]
fn golden_hawkes_biv_loglik_pinned() {
    let golden = load_golden();
    let case = &golden["hawkes_biv_loglik_pinned"];
    let p = &case["inputs"];
    let mu = [p["mu0"].as_f64().unwrap(), p["mu1"].as_f64().unwrap()];
    let alpha = [
        [p["a00"].as_f64().unwrap(), p["a01"].as_f64().unwrap()],
        [p["a10"].as_f64().unwrap(), p["a11"].as_f64().unwrap()],
    ];
    let (t, k) = biv_data();
    let ll = biv_loglik(
        t,
        k,
        p["T"].as_f64().unwrap(),
        &mu,
        &alpha,
        p["beta"].as_f64().unwrap(),
    )
    .expect("biv loglik");
    check_case(case, &[("loglik", ll)]);
}

#[test]
fn golden_hawkes_biv_fit() {
    let golden = load_golden();
    let fit = biv_fit();
    check_case(
        &golden["hawkes_biv_fit_alpha"],
        &[
            ("a00", fit.alpha[0][0]),
            ("a01", fit.alpha[0][1]),
            ("a10", fit.alpha[1][0]),
            ("a11", fit.alpha[1][1]),
        ],
    );
    check_case(
        &golden["hawkes_biv_spectral_radius"],
        &[("spectral_radius", fit.spectral_radius)],
    );
}

#[test]
fn golden_nn_losses() {
    let golden = load_golden();
    let pipe = pipeline();
    check_case(&golden["nn_loss_epoch1"], &[("loss", pipe.hist[0])]);
    check_case(
        &golden["nn_loss_final"],
        &[("loss", pipe.hist[pipe.hist.len() - 1])],
    );
}

#[test]
fn golden_holdout_accuracies() {
    let golden = load_golden();
    let pipe = pipeline();
    let acc_nn = accuracy(&pipe.y_test, &pipe.pred_nn).expect("acc nn");
    check_case(&golden["nn_holdout_accuracy"], &[("accuracy", acc_nn)]);
    let acc_lg = accuracy(&pipe.y_test, &pipe.pred_lg).expect("acc lg");
    check_case(&golden["logistic_accuracy"], &[("accuracy", acc_lg)]);
    let pred_mj = vec![pipe.majority; pipe.y_test.len()];
    let acc_mj = accuracy(&pipe.y_test, &pred_mj).expect("acc mj");
    check_case(&golden["majority_accuracy"], &[("accuracy", acc_mj)]);
    assert_eq!(
        pipe.majority,
        golden["majority_accuracy"]["inputs"]["majority_class"]
            .as_f64()
            .unwrap() as usize
    );
}

#[test]
fn golden_signal_pnl() {
    let golden = load_golden();
    let pipe = pipeline();
    let te = pipe.test_start..pipe.fs.len();
    let (gross, net, n_trades) = signal_pnl(
        &pipe.pred_nn,
        &pipe.fs.fwd_ret[te.clone()],
        &pipe.fs.spread[te],
    )
    .expect("pnl");
    check_case(
        &golden["signal_pnl_after_costs"],
        &[("gross", gross), ("net", net), ("n_trades", n_trades as f64)],
    );
    assert!(net < 0.0, "half-spread costs kill the naive signal");
    assert_eq!(pipe.x_test.rows(), pipe.y_test.len());
}

#[test]
fn golden_all_cases_covered() {
    assert_eq!(load_golden().len(), 18);
}
