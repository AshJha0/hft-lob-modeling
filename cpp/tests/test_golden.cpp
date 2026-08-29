/// Golden-value suite: recompute every case in data/golden/golden.json and
/// compare within its tolerance.  The same cases are asserted by the Python,
/// Rust and Java suites, which is what pins cross-language agreement.

#include <cmath>
#include <gtest/gtest.h>

#include "lobmod/features.hpp"
#include "lobmod/gof.hpp"
#include "lobmod/hawkes.hpp"
#include "lobmod/nn.hpp"
#include "test_data.hpp"

namespace {

using testdata::golden;

void check(const testdata::GoldenCase& c, const std::string& key, double value) {
    ASSERT_TRUE(c.expect.count(key) > 0) << "missing expect key " << key;
    EXPECT_NEAR(value, c.expect.at(key), c.tol) << "key: " << key;
}

TEST(Golden, BookSnapshots) {
    const auto& st = testdata::states();
    for (int seq : {1000, 3000, 5000}) {
        const auto& c = golden().at("book_snapshot_event_" + std::to_string(seq));
        const std::size_t i = static_cast<std::size_t>(seq) - 1;
        check(c, "best_bid", st.best_bid[i]);
        check(c, "best_ask", st.best_ask[i]);
        check(c, "bid_size", st.bid_size[i]);
        check(c, "ask_size", st.ask_size[i]);
        check(c, "bid_depth_k", st.bid_depth_k[i]);
        check(c, "ask_depth_k", st.ask_depth_k[i]);
        check(c, "mid", st.mid[i]);
    }
}

TEST(Golden, Microprice) {
    const auto& c = golden().at("microprice_event_2500");
    const std::size_t i = static_cast<std::size_t>(c.inputs.at("event")) - 1;
    check(c, "microprice", testdata::states().microprice[i]);
}

TEST(Golden, Ofi) {
    const auto& c = golden().at("ofi_event_2200_w100");
    const std::size_t i = static_cast<std::size_t>(c.inputs.at("event")) - 1;
    const int w = static_cast<int>(c.inputs.at("window"));
    check(c, "ofi", lobmod::ofi_at(testdata::states(), i, w));
}

TEST(Golden, HawkesUniLoglikPinned) {
    const auto& c = golden().at("hawkes_uni_loglik_pinned");
    const double ll =
        lobmod::uni_loglik(testdata::uni_times(), c.inputs.at("T"),
                           c.inputs.at("mu"), c.inputs.at("alpha"),
                           c.inputs.at("beta"));
    check(c, "loglik", ll);
    // and the recursion agrees with brute force on the full bundled sample
    const double brute =
        lobmod::uni_loglik_brute(testdata::uni_times(), c.inputs.at("T"),
                                 c.inputs.at("mu"), c.inputs.at("alpha"),
                                 c.inputs.at("beta"));
    EXPECT_NEAR(ll, brute, 1e-10);
}

TEST(Golden, HawkesUniFit) {
    const auto& fit = testdata::uni_fit();
    const auto& c = golden().at("hawkes_uni_fit");
    check(c, "mu", fit.mu);
    check(c, "alpha", fit.alpha);
    check(c, "beta", fit.beta);
    check(golden().at("hawkes_uni_branching"), "branching", fit.branching);
    EXPECT_TRUE(fit.converged);
}

TEST(Golden, HawkesUniKsFitted) {
    const auto& fit = testdata::uni_fit();
    const auto tau = lobmod::uni_residuals(testdata::uni_times(), testdata::kUniT,
                                           fit.mu, fit.alpha, fit.beta);
    const auto [d, p] = lobmod::ks_test_exp(tau);
    const auto& c = golden().at("hawkes_uni_ks_fitted");
    check(c, "ks_stat", d);
    check(c, "p_value", p);
}

TEST(Golden, HawkesBivLoglikPinned) {
    const auto& c = golden().at("hawkes_biv_loglik_pinned");
    const std::array<double, 2> mu = {c.inputs.at("mu0"), c.inputs.at("mu1")};
    const lobmod::Mat2 alpha = {{{c.inputs.at("a00"), c.inputs.at("a01")},
                                 {c.inputs.at("a10"), c.inputs.at("a11")}}};
    const double ll =
        lobmod::biv_loglik(testdata::biv_data().first, testdata::biv_data().second,
                           c.inputs.at("T"), mu, alpha, c.inputs.at("beta"));
    check(c, "loglik", ll);
}

TEST(Golden, HawkesBivFit) {
    const auto& fit = testdata::biv_fit();
    const auto& c = golden().at("hawkes_biv_fit_alpha");
    check(c, "a00", fit.alpha[0][0]);
    check(c, "a01", fit.alpha[0][1]);
    check(c, "a10", fit.alpha[1][0]);
    check(c, "a11", fit.alpha[1][1]);
    check(golden().at("hawkes_biv_spectral_radius"), "spectral_radius",
          fit.spectral_radius);
    EXPECT_TRUE(fit.converged);
}

TEST(Golden, NnLosses) {
    const auto& pl = testdata::pipeline();
    check(golden().at("nn_loss_epoch1"), "loss", pl.history.front());
    check(golden().at("nn_loss_final"), "loss", pl.history.back());
}

TEST(Golden, HoldoutAccuracies) {
    const auto& pl = testdata::pipeline();
    const auto pred_nn = pl.net->predict(pl.x_test);
    check(golden().at("nn_holdout_accuracy"), "accuracy",
          lobmod::accuracy(pl.y_test, pred_nn));
    const auto pred_lg = pl.logistic->predict(pl.x_test);
    check(golden().at("logistic_accuracy"), "accuracy",
          lobmod::accuracy(pl.y_test, pred_lg));
    const int maj = lobmod::majority_class(pl.y_train);
    const std::vector<int> pred_mj(pl.y_test.size(), maj);
    check(golden().at("majority_accuracy"), "accuracy",
          lobmod::accuracy(pl.y_test, pred_mj));
    EXPECT_EQ(maj, static_cast<int>(
                       golden().at("majority_accuracy").inputs.at("majority_class")));
}

TEST(Golden, SignalPnl) {
    const auto& pl = testdata::pipeline();
    const auto pred = pl.net->predict(pl.x_test);
    std::vector<double> fwd(pl.test_idx.size()), spr(pl.test_idx.size());
    for (std::size_t i = 0; i < pl.test_idx.size(); ++i) {
        fwd[i] = pl.fs.fwd_ret[pl.test_idx[i]];
        spr[i] = pl.fs.spread[pl.test_idx[i]];
    }
    const auto [gross, net, n_trades] = lobmod::signal_pnl(pred, fwd, spr);
    const auto& c = golden().at("signal_pnl_after_costs");
    check(c, "gross", gross);
    check(c, "net", net);
    check(c, "n_trades", static_cast<double>(n_trades));
    EXPECT_LT(net, 0.0); // half-spread costs kill the naive signal
}

TEST(Golden, AllCasesCovered) { EXPECT_EQ(golden().size(), 18u); }

} // namespace
