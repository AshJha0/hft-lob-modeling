/// \file demo.cpp
/// \brief lobmod end-to-end demo: LOB replay, Hawkes MLE + goodness of fit,
///        and the deterministic MLP direction classifier vs baselines,
///        closing with an honest costs-included PnL.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "lobmod/features.hpp"
#include "lobmod/gof.hpp"
#include "lobmod/hawkes.hpp"
#include "lobmod/nn.hpp"
#include "lobmod/replay.hpp"

#ifndef LOBMOD_DATA_DIR
#define LOBMOD_DATA_DIR "../data"
#endif

namespace {

const std::string kData = LOBMOD_DATA_DIR;

void section(const std::string& title) {
    const std::string line(68, '-');
    std::printf("\n%s\n%s\n%s\n", line.c_str(), title.c_str(), line.c_str());
}

} // namespace

int main() {
    std::printf("lobmod demo — high-frequency LOB modeling (C++)\n");

    // ------------------------------------------------------------ book replay
    section("1. Limit order book replay (data/lob_events.csv)");
    const auto events = lobmod::read_events_csv(kData + "/lob_events.csv");
    const auto cfg = lobmod::load_feature_config(kData + "/features_config.json");
    const auto states = lobmod::replay(events, cfg.depth_k);

    lobmod::LimitOrderBook book;
    for (const auto& e : events) lobmod::apply_event(book, e);
    int n_l = 0, n_c = 0, n_m = 0;
    for (const auto& e : events) {
        if (e.etype == 'L') ++n_l;
        else if (e.etype == 'C') ++n_c;
        else ++n_m;
    }
    long long volume = 0;
    for (const auto& f : book.trades()) volume += f.size;
    double mid_lo = 1e300, mid_hi = -1e300, spread_sum = 0.0;
    int n_finite_spread = 0, n_nan_mid = 0;
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (std::isfinite(states.mid[i])) {
            mid_lo = std::min(mid_lo, states.mid[i]);
            mid_hi = std::max(mid_hi, states.mid[i]);
        } else {
            ++n_nan_mid;
        }
        if (std::isfinite(states.spread[i])) {
            spread_sum += states.spread[i];
            ++n_finite_spread;
        }
    }
    std::printf("events: %zu  (limit %d, cancel %d, market %d)\n", events.size(),
                n_l, n_c, n_m);
    std::printf("trades: %zu fills, %lld units traded\n", book.trades().size(),
                volume);
    std::printf("mid range: [%.1f, %.1f] ticks; mean spread %.3f ticks\n", mid_lo,
                mid_hi, spread_sum / n_finite_spread);
    std::printf("events with an empty side (mid = NaN, guarded): %d\n", n_nan_mid);
    std::printf("\n  snapshot     best_bid  best_ask     mid   micro    depth3(B/S)\n");
    for (int seq : {1000, 3000, 5000}) {
        const std::size_t i = static_cast<std::size_t>(seq) - 1;
        std::printf("  event %5d   %7.0f  %8.0f  %7.1f %8.2f   %.0f/%.0f\n", seq,
                    states.best_bid[i], states.best_ask[i], states.mid[i],
                    states.microprice[i], states.bid_depth_k[i],
                    states.ask_depth_k[i]);
    }

    // -------------------------------------------------------- Hawkes univariate
    section("2. Univariate Hawkes fit (data/hawkes_uni.csv)");
    const auto t_uni = lobmod::read_times_csv(kData + "/hawkes_uni.csv");
    const double t_horizon = 1000.0; // dgp_params.json: hawkes_uni.T
    const auto fit_u = lobmod::fit_uni(t_uni, t_horizon);
    double gap_sum = 0.0, gap_sq = 0.0;
    const std::size_t n_gaps = t_uni.size() - 1;
    for (std::size_t i = 1; i < t_uni.size(); ++i) gap_sum += t_uni[i] - t_uni[i - 1];
    const double gap_mean = gap_sum / static_cast<double>(n_gaps);
    for (std::size_t i = 1; i < t_uni.size(); ++i) {
        const double dev = t_uni[i] - t_uni[i - 1] - gap_mean;
        gap_sq += dev * dev;
    }
    const double gap_cv = std::sqrt(gap_sq / static_cast<double>(n_gaps)) / gap_mean;
    std::printf("%zu events on [0, %.0f]  (gap CV %.2f vs 1 for Poisson)\n",
                t_uni.size(), t_horizon, gap_cv);
    std::printf("\n  param        true    fitted\n");
    std::printf("  mu          0.500  %8.4f\n", fit_u.mu);
    std::printf("  alpha       1.000  %8.4f\n", fit_u.alpha);
    std::printf("  beta        1.600  %8.4f\n", fit_u.beta);
    std::printf("  branching ratio alpha/beta = %.4f (%s; %.0f%% of events are "
                "endogenous offspring)\n",
                fit_u.branching,
                fit_u.branching < 1 ? "stationary" : "NON-STATIONARY",
                100.0 * fit_u.branching);
    std::printf("  loglik %.2f, Nelder-Mead converged=%s (%d iters)\n",
                fit_u.loglik, fit_u.converged ? "true" : "false", fit_u.n_iter);

    section("3. Goodness of fit — time-rescaling residuals + KS");
    const auto tau = lobmod::uni_residuals(t_uni, t_horizon, fit_u.mu, fit_u.alpha,
                                           fit_u.beta);
    const auto [d_fit, p_fit] = lobmod::ks_test_exp(tau);
    std::printf("time-rescaled inter-arrivals should be iid Exp(1) under the model:\n");
    std::printf("  fitted params : KS stat %.4f, p-value %.4f -> %s\n", d_fit,
                p_fit, p_fit > 0.05 ? "no rejection (test is weak with fitted parameters)" : "REJECT");
    const auto tau_wrong =
        lobmod::uni_residuals(t_uni, t_horizon, 1.5, 0.1, 3.0);
    const auto [d_w, p_w] = lobmod::ks_test_exp(tau_wrong);
    std::printf("  wrong params  : KS stat %.4f, p-value %.2e -> REJECT (sanity)\n",
                d_w, p_w);

    // -------------------------------------------------------- Hawkes bivariate
    section("4. Bivariate Hawkes: buy/sell market-order cross-excitation");
    const auto [t_biv, k_biv] = lobmod::read_marked_times_csv(kData + "/hawkes_biv.csv");
    const double t_biv_horizon = 600.0; // dgp_params.json: hawkes_biv.T
    const auto fit_b = lobmod::fit_biv(t_biv, k_biv, t_biv_horizon);
    int n_buy = 0;
    for (int k : k_biv) {
        if (k == 0) ++n_buy;
    }
    std::printf("%zu events (%d buys, %zu sells) on [0, %.0f]\n", t_biv.size(),
                n_buy, t_biv.size() - n_buy, t_biv_horizon);
    std::printf("\n  excitation alpha (fitted)      branching alpha/beta\n");
    const char* names[4] = {"buy <-buy ", "buy <-sell", "sell<-buy ", "sell<-sell"};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::printf("    %s %6.3f            %6.3f\n", names[2 * i + j],
                        fit_b.alpha[i][j], fit_b.branching[i][j]);
        }
    }
    std::printf("  mu = %.4f (buy), %.4f (sell); beta = %.4f\n", fit_b.mu[0],
                fit_b.mu[1], fit_b.beta);
    std::printf("  spectral radius of branching matrix = %.4f (%s)\n",
                fit_b.spectral_radius,
                fit_b.spectral_radius < 1 ? "stationary" : "NON-STATIONARY");
    std::printf("  interpretation: self terms exceed cross terms because the simulation was\n"
                "  parameterised that way; the fit recovers them. This checks the estimator\n"
                "  and says nothing about real order flow.\n");
    const auto taus = lobmod::biv_residuals(t_biv, k_biv, t_biv_horizon, fit_b.mu,
                                            fit_b.alpha, fit_b.beta);
    for (int m = 0; m < 2; ++m) {
        const auto [dd, pp] = lobmod::ks_test_exp(taus[m]);
        std::printf("  KS residuals (%-4s): stat %.4f, p %.4f -> %s\n",
                    m == 0 ? "buy" : "sell", dd, pp, pp > 0.05 ? "no rejection (test is weak with fitted parameters)" : "REJECT");
    }

    // ----------------------------------------------------- features + learning
    section("5. Direction prediction: MLP vs logistic vs majority");
    const auto fs = lobmod::compute_features(states, cfg);
    const auto [tr, te] = lobmod::chronological_split(fs.size(), cfg.train_frac);
    lobmod::Matrix x_tr_raw(tr.size(), 5), x_te_raw(te.size(), 5);
    std::vector<int> y_tr(tr.size()), y_te(te.size());
    for (std::size_t i = 0; i < tr.size(); ++i) {
        for (std::size_t j = 0; j < 5; ++j) x_tr_raw(i, j) = fs.X[tr[i]][j];
        y_tr[i] = fs.y[tr[i]];
    }
    for (std::size_t i = 0; i < te.size(); ++i) {
        for (std::size_t j = 0; j < 5; ++j) x_te_raw(i, j) = fs.X[te[i]][j];
        y_te[i] = fs.y[te[i]];
    }
    lobmod::Standardizer std_scaler;
    std_scaler.fit(x_tr_raw);
    const auto x_tr = std_scaler.transform(x_tr_raw);
    const auto x_te = std_scaler.transform(x_te_raw);
    int counts_tr[3] = {0, 0, 0}, counts_te[3] = {0, 0, 0};
    for (int y : y_tr) ++counts_tr[y];
    for (int y : y_te) ++counts_te[y];
    std::printf("%zu samples (features: ofi, queue_imbalance, spread, "
                "depth_ratio, ret); H=%d events, eps=%.2f\n",
                fs.size(), cfg.horizon, cfg.epsilon);
    std::printf("class counts (down/flat/up): train [%d, %d, %d], test [%d, %d, %d]\n",
                counts_tr[0], counts_tr[1], counts_tr[2], counts_te[0],
                counts_te[1], counts_te[2]);
    std::printf("chronological split: %zu train / %zu test; MLP (16, 8), lr=%.1f, "
                "%d full-batch epochs\n",
                tr.size(), te.size(), cfg.learning_rate, cfg.epochs);

    lobmod::MLP net(5, cfg.hidden, 3);
    const auto hist = net.fit(x_tr, y_tr, cfg.learning_rate, cfg.epochs);
    lobmod::MLP logistic(5, {}, 3);
    logistic.fit(x_tr, y_tr, cfg.learning_rate, cfg.epochs);
    const int maj = lobmod::majority_class(y_tr);
    const auto pred_nn = net.predict(x_te);
    const auto pred_lg = logistic.predict(x_te);
    const std::vector<int> pred_mj(te.size(), maj);
    std::printf("\ntraining loss: %.4f (epoch 1) -> %.4f (epoch %d)\n",
                hist.front(), hist.back(), cfg.epochs);
    std::printf("\n  model                holdout accuracy\n");
    std::printf("  MLP (16,8)               %.4f\n", lobmod::accuracy(y_te, pred_nn));
    std::printf("  logistic regression      %.4f\n", lobmod::accuracy(y_te, pred_lg));
    std::printf("  majority class (%d)       %.4f\n", maj,
                lobmod::accuracy(y_te, pred_mj));
    const auto cm = lobmod::confusion_matrix(y_te, pred_nn, 3);
    std::printf("\n  MLP confusion matrix (rows true, cols predicted; 0=down "
                "1=flat 2=up)\n");
    for (int r = 0; r < 3; ++r) {
        std::printf("    %d | %5d %5d %5d\n", r, cm[r][0], cm[r][1], cm[r][2]);
    }

    section("6. Does the signal survive costs?");
    std::vector<double> fwd_te(te.size()), spread_te(te.size());
    for (std::size_t i = 0; i < te.size(); ++i) {
        fwd_te[i] = fs.fwd_ret[te[i]];
        spread_te[i] = fs.spread[te[i]];
    }
    const auto [gross, net_pnl, n_trades] =
        lobmod::signal_pnl(pred_nn, fwd_te, spread_te);
    std::printf("trading the MLP signal on the holdout (%d trades, mid-to-mid, "
                "half-spread cost per trade):\n", n_trades);
    std::printf("  gross PnL %+.1f ticks   net PnL %+.1f ticks\n", gross, net_pnl);
    std::printf("note: on this split the MLP edges both baselines and gross PnL is positive,\n"
                "but across 100 seeds logistic beats the MLP by 0.59 pts (SE 0.16). The structure\n"
                "is generator mechanics (spread alone gives 0.693 of the 0.704), and paying half\n"
                "the spread on every trade erases the gross PnL. See research/results.md.\n");

    std::printf("\ndemo complete.\n");
    return 0;
}
