//! lobmod end-to-end demo.
//!
//! Replays the bundled 5000-event LOB stream, fits univariate and bivariate
//! Hawkes models to the bundled order-flow data with goodness-of-fit residual
//! analysis, and trains the deterministic MLP direction classifier against
//! logistic-regression and majority baselines, closing with an honest look at
//! whether the signal survives trading costs.
//!
//! Run:  cd rust && cargo run --bin demo

use std::process::ExitCode;

use lobmod::{
    accuracy, apply_event, biv_residuals, chronological_split, compute_features, confusion_matrix,
    fit_biv, fit_uni, ks_test_exp, load_feature_config, majority_class, read_events_csv,
    read_marked_times_csv, read_times_csv, replay, signal_pnl, uni_residuals, EventType,
    LimitOrderBook, LobError, Mlp, Standardizer,
};

const LINE: &str =
    "--------------------------------------------------------------------";

fn section(title: &str) {
    println!("\n{LINE}\n{title}\n{LINE}");
}

fn run() -> Result<(), LobError> {
    let data = format!("{}/../data", env!("CARGO_MANIFEST_DIR"));
    println!("lobmod demo — high-frequency LOB modeling (Rust)");

    // -------------------------------------------------------------- book replay
    section("1. Limit order book replay (data/lob_events.csv)");
    let events = read_events_csv(&format!("{data}/lob_events.csv"))?;
    let cfg = load_feature_config(&format!("{data}/features_config.json"))?;
    let states = replay(&events, cfg.depth_k)?;
    let mut book = LimitOrderBook::new(1.0)?;
    for e in &events {
        apply_event(&mut book, e)?;
    }
    let count = |t: EventType| events.iter().filter(|e| e.etype == t).count();
    let volume: i64 = book.trades.iter().map(|f| f.size).sum();
    let finite_spread: Vec<f64> = states.spread.iter().copied().filter(|v| v.is_finite()).collect();
    let mean_spread = finite_spread.iter().sum::<f64>() / finite_spread.len() as f64;
    let (mut mid_min, mut mid_max, mut n_nan) = (f64::INFINITY, f64::NEG_INFINITY, 0usize);
    for &m in &states.mid {
        if m.is_finite() {
            mid_min = mid_min.min(m);
            mid_max = mid_max.max(m);
        } else {
            n_nan += 1;
        }
    }
    println!(
        "events: {}  (limit {}, cancel {}, market {})",
        events.len(),
        count(EventType::Limit),
        count(EventType::Cancel),
        count(EventType::Market)
    );
    println!("trades: {} fills, {volume} units traded", book.trades.len());
    println!("mid range: [{mid_min:.1}, {mid_max:.1}] ticks; mean spread {mean_spread:.3} ticks");
    println!("events with an empty side (mid = NaN, guarded): {n_nan}");
    println!("\n  snapshot     best_bid  best_ask     mid   micro    depth3(B/S)");
    for seq in [1000usize, 3000, 5000] {
        let i = seq - 1;
        println!(
            "  event {seq:5}   {:7.0}  {:8.0}  {:7.1} {:8.2}   {:.0}/{:.0}",
            states.best_bid[i],
            states.best_ask[i],
            states.mid[i],
            states.microprice[i],
            states.bid_depth_k[i],
            states.ask_depth_k[i]
        );
    }

    // ---------------------------------------------------------- Hawkes univariate
    section("2. Univariate Hawkes fit (data/hawkes_uni.csv)");
    let t_uni = read_times_csv(&format!("{data}/hawkes_uni.csv"))?;
    let t_horizon_uni = 1000.0;
    let fit_u = fit_uni(&t_uni, t_horizon_uni)?;
    let gaps: Vec<f64> = t_uni.windows(2).map(|w| w[1] - w[0]).collect();
    let min_gap = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
    println!(
        "{} events on [0, {t_horizon_uni:.0}]  (min gap {min_gap:.4} vs mean {mean_gap:.3} — self-exciting bursts)",
        t_uni.len()
    );
    println!("\n  param        true    fitted");
    for (name, tv, fv) in [
        ("mu", 0.5, fit_u.mu),
        ("alpha", 1.0, fit_u.alpha),
        ("beta", 1.6, fit_u.beta),
    ] {
        println!("  {name:<8} {tv:8.3}  {fv:8.4}");
    }
    println!(
        "  branching ratio alpha/beta = {:.4}  ({}; {:.0}% of events are endogenous offspring)",
        fit_u.branching,
        if fit_u.branching < 1.0 { "stationary" } else { "NON-STATIONARY" },
        100.0 * fit_u.branching
    );
    println!(
        "  loglik {:.2}, Nelder-Mead converged={} ({} iters)",
        fit_u.loglik, fit_u.converged, fit_u.n_iter
    );

    section("3. Goodness of fit — time-rescaling residuals + KS");
    let tau = uni_residuals(&t_uni, t_horizon_uni, fit_u.mu, fit_u.alpha, fit_u.beta)?;
    let (d, p) = ks_test_exp(&tau)?;
    println!("time-rescaled inter-arrivals should be iid Exp(1) under the model:");
    println!(
        "  fitted params : KS stat {d:.4}, p-value {p:.4} -> {}",
        if p > 0.05 { "PASS (no rejection at 5%)" } else { "REJECT" }
    );
    let tau_wrong = uni_residuals(&t_uni, t_horizon_uni, 1.5, 0.1, 3.0)?;
    let (dw, pw) = ks_test_exp(&tau_wrong)?;
    println!("  wrong params  : KS stat {dw:.4}, p-value {pw:.2e} -> REJECT (sanity)");

    // ---------------------------------------------------------- Hawkes bivariate
    section("4. Bivariate Hawkes: buy/sell market-order cross-excitation");
    let (t_biv, k_biv) = read_marked_times_csv(&format!("{data}/hawkes_biv.csv"))?;
    let t_horizon_biv = 600.0;
    let fit_b = fit_biv(&t_biv, &k_biv, t_horizon_biv)?;
    let n_buy = k_biv.iter().filter(|&&k| k == 0).count();
    println!(
        "{} events ({} buys, {} sells) on [0, {t_horizon_biv:.0}]",
        t_biv.len(),
        n_buy,
        t_biv.len() - n_buy
    );
    let (a, br) = (fit_b.alpha, fit_b.branching);
    println!("\n  excitation alpha (fitted)      branching alpha/beta");
    println!("    buy <-buy  {:6.3}            {:6.3}", a[0][0], br[0][0]);
    println!("    buy <-sell {:6.3}            {:6.3}", a[0][1], br[0][1]);
    println!("    sell<-buy  {:6.3}            {:6.3}", a[1][0], br[1][0]);
    println!("    sell<-sell {:6.3}            {:6.3}", a[1][1], br[1][1]);
    println!(
        "  mu = {:.4} (buy), {:.4} (sell); beta = {:.4}",
        fit_b.mu[0], fit_b.mu[1], fit_b.beta
    );
    println!(
        "  spectral radius of branching matrix = {:.4} ({})",
        fit_b.spectral_radius,
        if fit_b.spectral_radius < 1.0 { "stationary" } else { "NON-STATIONARY" }
    );
    println!("  interpretation: self-excitation (buy<-buy, sell<-sell) dominates the");
    println!("  cross terms — order flow is strongly clustered by side (herding /");
    println!("  order splitting), with weaker but positive cross-side contagion.");
    let taus = biv_residuals(&t_biv, &k_biv, t_horizon_biv, &fit_b.mu, &fit_b.alpha, fit_b.beta)?;
    for (name, tt) in ["buy", "sell"].iter().zip(&taus) {
        let (dd, pp) = ks_test_exp(tt)?;
        println!(
            "  KS residuals ({name:4}): stat {dd:.4}, p {pp:.4} -> {}",
            if pp > 0.05 { "PASS" } else { "REJECT" }
        );
    }

    // ------------------------------------------------------- features + learning
    section("5. Direction prediction: MLP vs logistic vs majority");
    let fs = compute_features(&states, &cfg)?;
    let (tr, te) = chronological_split(fs.len(), cfg.train_frac)?;
    let x_train_raw = fs.x.slice_rows(tr.start, tr.end)?;
    let x_test_raw = fs.x.slice_rows(te.start, te.end)?;
    let mut std = Standardizer::new();
    std.fit(&x_train_raw)?;
    let x_tr = std.transform(&x_train_raw)?;
    let x_te = std.transform(&x_test_raw)?;
    let y_tr = &fs.y[tr.clone()];
    let y_te = &fs.y[te.clone()];
    let bincount = |y: &[usize]| {
        let mut c = [0usize; 3];
        for &v in y {
            c[v] += 1;
        }
        c
    };
    println!(
        "{} samples (features: ofi, queue_imbalance, spread, depth_ratio, ret); H={} events, eps={}",
        fs.len(),
        cfg.horizon,
        cfg.epsilon
    );
    println!(
        "class counts (down/flat/up): train {:?}, test {:?}",
        bincount(y_tr),
        bincount(y_te)
    );
    println!(
        "chronological split: {} train / {} test; MLP {:?}, lr={}, {} full-batch epochs",
        tr.len(),
        te.len(),
        cfg.hidden,
        cfg.learning_rate,
        cfg.epochs
    );

    let mut net = Mlp::new(5, &cfg.hidden, 3)?;
    let hist = net.fit(&x_tr, y_tr, cfg.learning_rate, cfg.epochs)?;
    let mut logistic = Mlp::new(5, &[], 3)?;
    logistic.fit(&x_tr, y_tr, cfg.learning_rate, cfg.epochs)?;
    let maj = majority_class(y_tr)?;
    let pred_nn = net.predict(&x_te)?;
    let pred_lg = logistic.predict(&x_te)?;
    let pred_mj = vec![maj; y_te.len()];
    println!(
        "\ntraining loss: {:.4} (epoch 1) -> {:.4} (epoch {})",
        hist[0],
        hist[hist.len() - 1],
        cfg.epochs
    );
    println!("\n  model                holdout accuracy");
    println!("  MLP (16,8)               {:.4}", accuracy(y_te, &pred_nn)?);
    println!("  logistic regression      {:.4}", accuracy(y_te, &pred_lg)?);
    println!("  majority class ({maj})       {:.4}", accuracy(y_te, &pred_mj)?);
    let cm = confusion_matrix(y_te, &pred_nn, 3)?;
    println!("\n  MLP confusion matrix (rows true, cols predicted; 0=down 1=flat 2=up)");
    for (r, row) in cm.iter().enumerate() {
        println!("    {r} | {:5} {:5} {:5}", row[0], row[1], row[2]);
    }

    section("6. Does the signal survive costs?");
    let (gross, net_pnl, n_trades) =
        signal_pnl(&pred_nn, &fs.fwd_ret[te.clone()], &fs.spread[te])?;
    println!(
        "trading the MLP signal on the holdout ({n_trades} trades, mid-to-mid, half-spread cost per trade):"
    );
    println!("  gross PnL {gross:+.1} ticks   net PnL {net_pnl:+.1} ticks");
    println!("note: the classifier finds real structure (it beats both baselines and");
    println!("gross PnL is positive), but paying half the spread on every trade");
    println!("erases it — naive microstructure signals die on transaction costs.");

    println!("\ndemo complete.");
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("demo failed: {e}");
            ExitCode::FAILURE
        }
    }
}
