/// Hawkes processes: recursion vs brute force, simulation determinism,
/// refits close to the DGP, near-critical warning, residual properties, and
/// the pinned Nelder-Mead minimizer.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <gtest/gtest.h>

#include "lobmod/gof.hpp"
#include "lobmod/hawkes.hpp"
#include "lobmod/neldermead.hpp"
#include "lobmod/zi.hpp"
#include "test_data.hpp"

namespace {

using lobmod::Mat2;

std::string scratch(const std::string& name, const std::string& text) {
    const std::string path = testing::TempDir() + "lobmod_h_" + name;
    std::ofstream out(path, std::ios::binary);
    out << text;
    return path;
}

std::string slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

TEST(NelderMead, MinimizesQuadratic) {
    const auto f = [](const std::vector<double>& x) {
        return (x[0] - 3.0) * (x[0] - 3.0) + 2.0 * (x[1] + 1.0) * (x[1] + 1.0);
    };
    const auto res = lobmod::nelder_mead(f, {0.0, 0.0});
    EXPECT_TRUE(res.converged);
    EXPECT_NEAR(res.x[0], 3.0, 1e-6);
    EXPECT_NEAR(res.x[1], -1.0, 1e-6);
    EXPECT_NEAR(res.fval, 0.0, 1e-10);
}

TEST(NelderMead, Validation) {
    const auto f = [](const std::vector<double>& x) { return x[0] * x[0]; };
    EXPECT_THROW(lobmod::nelder_mead(f, {}), std::invalid_argument);
    EXPECT_THROW(lobmod::nelder_mead(f, {std::nan("")}), std::invalid_argument);
    EXPECT_THROW(lobmod::nelder_mead(f, {1.0}, -1.0), std::invalid_argument);
    EXPECT_THROW(lobmod::nelder_mead(f, {1.0}, 1e-8, 1e-10, 0),
                 std::invalid_argument);
}

TEST(HawkesUni, RecursionMatchesBruteForceSmall) {
    const std::vector<double> t = {0.5, 1.1, 1.15, 2.4, 3.0, 5.5, 5.6, 7.9};
    const double ll = lobmod::uni_loglik(t, 10.0, 0.7, 0.9, 1.7);
    const double brute = lobmod::uni_loglik_brute(t, 10.0, 0.7, 0.9, 1.7);
    EXPECT_NEAR(ll, brute, 1e-10);
}

TEST(HawkesBiv, RecursionMatchesBruteForce) {
    const std::vector<double> t = {0.4, 0.9, 1.3, 2.0, 2.05, 3.7, 4.4};
    const std::vector<int> k = {0, 1, 0, 0, 1, 1, 0};
    const std::array<double, 2> mu = {0.3, 0.4};
    const Mat2 a = {{{0.5, 0.2}, {0.3, 0.6}}};
    EXPECT_NEAR(lobmod::biv_loglik(t, k, 6.0, mu, a, 1.9),
                lobmod::biv_loglik_brute(t, k, 6.0, mu, a, 1.9), 1e-10);
    // and on a slice of the bundled sample
    const auto& [tb, kb] = testdata::biv_data();
    const std::vector<double> ts(tb.begin(), tb.begin() + 200);
    const std::vector<int> ks(kb.begin(), kb.begin() + 200);
    const double T = ts.back() + 1.0;
    EXPECT_NEAR(lobmod::biv_loglik(ts, ks, T, mu, a, 1.9),
                lobmod::biv_loglik_brute(ts, ks, T, mu, a, 1.9), 1e-10);
}

TEST(HawkesUni, InputValidation) {
    const std::vector<double> t = {1.0, 2.0, 3.0};
    EXPECT_THROW(lobmod::uni_loglik({}, 10.0, 0.5, 0.5, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(lobmod::uni_loglik({0.0, 1.0}, 10.0, 0.5, 0.5, 1.0),
                 std::invalid_argument); // t_1 <= 0
    EXPECT_THROW(lobmod::uni_loglik({2.0, 1.0}, 10.0, 0.5, 0.5, 1.0),
                 std::invalid_argument); // not increasing
    EXPECT_THROW(lobmod::uni_loglik({1.0, 1.0}, 10.0, 0.5, 0.5, 1.0),
                 std::invalid_argument); // ties
    EXPECT_THROW(lobmod::uni_loglik(t, 2.5, 0.5, 0.5, 1.0),
                 std::invalid_argument); // t_N > T
    EXPECT_THROW(lobmod::uni_loglik(t, -1.0, 0.5, 0.5, 1.0),
                 std::invalid_argument);
    EXPECT_THROW(lobmod::uni_loglik(t, 10.0, 0.0, 0.5, 1.0),
                 std::invalid_argument); // mu <= 0
    EXPECT_THROW(lobmod::uni_loglik(t, 10.0, 0.5, -0.1, 1.0),
                 std::invalid_argument); // alpha < 0
    EXPECT_THROW(lobmod::uni_loglik(t, 10.0, 0.5, 0.5, 0.0),
                 std::invalid_argument); // beta <= 0
    EXPECT_THROW(lobmod::biv_loglik(t, {0, 1}, 10.0, {0.3, 0.3},
                                    {{{0.1, 0.1}, {0.1, 0.1}}}, 1.0),
                 std::invalid_argument); // types length mismatch
    EXPECT_THROW(lobmod::biv_loglik(t, {0, 1, 2}, 10.0, {0.3, 0.3},
                                    {{{0.1, 0.1}, {0.1, 0.1}}}, 1.0),
                 std::invalid_argument); // bad type value
}

TEST(HawkesUni, SimulationDeterministicPerSeed) {
    const auto a = lobmod::simulate_uni(0.5, 0.8, 1.6, 200.0, 12345);
    const auto b = lobmod::simulate_uni(0.5, 0.8, 1.6, 200.0, 12345);
    const auto c = lobmod::simulate_uni(0.5, 0.8, 1.6, 200.0, 54321);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) ASSERT_EQ(a[i], b[i]);
    EXPECT_NE(a.size(), c.size()); // overwhelmingly likely for real streams
    ASSERT_FALSE(a.empty());
    for (std::size_t i = 1; i < a.size(); ++i) ASSERT_GT(a[i], a[i - 1]);
    EXPECT_GT(a.front(), 0.0);
    EXPECT_LE(a.back(), 200.0);
}

TEST(HawkesUni, SimulationRejectsNonStationary) {
    EXPECT_THROW(lobmod::simulate_uni(0.5, 2.0, 1.0, 100.0, 1),
                 std::domain_error);
    EXPECT_THROW(lobmod::simulate_biv({0.3, 0.3}, {{{1.5, 0.9}, {0.9, 1.5}}},
                                      2.0, 100.0, 1),
                 std::domain_error);
}

TEST(HawkesUni, BundledSampleMatchesGeneration) {
    // hawkes_uni.csv is simulate_uni(0.5, 1.0, 1.6, 1000, seed 101)
    const auto sim = lobmod::simulate_uni(0.5, 1.0, 1.6, 1000.0, 101);
    const auto& bundled = testdata::uni_times();
    ASSERT_EQ(sim.size(), bundled.size());
    for (std::size_t i = 0; i < sim.size(); ++i) {
        ASSERT_NEAR(sim[i], bundled[i], 1e-12) << "event " << i;
    }
}

TEST(HawkesBiv, BundledSampleMatchesGeneration) {
    // hawkes_biv.csv is simulate_biv((0.3,0.3), [[0.6,0.3],[0.25,0.5]], 2, 600, 33)
    const auto [times, types] = lobmod::simulate_biv(
        {0.3, 0.3}, {{{0.6, 0.3}, {0.25, 0.5}}}, 2.0, 600.0, 33);
    const auto& [tb, kb] = testdata::biv_data();
    ASSERT_EQ(times.size(), tb.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        ASSERT_NEAR(times[i], tb[i], 1e-12);
        ASSERT_EQ(types[i], kb[i]);
    }
}

TEST(HawkesUni, RefitRecoversDgpLoosely) {
    const auto& fit = testdata::uni_fit();
    // true params mu=0.5, alpha=1.0, beta=1.6 (data/dgp_params.json)
    EXPECT_NEAR(fit.mu, 0.5, 0.15);
    EXPECT_NEAR(fit.alpha, 1.0, 0.25);
    EXPECT_NEAR(fit.beta, 1.6, 0.35);
    EXPECT_LT(fit.branching, 1.0);
    EXPECT_TRUE(fit.warning.empty()); // comfortably subcritical
}

TEST(HawkesBiv, RefitRecoversDgpLoosely) {
    const auto& fit = testdata::biv_fit();
    EXPECT_NEAR(fit.mu[0], 0.3, 0.15);
    EXPECT_NEAR(fit.mu[1], 0.3, 0.15);
    EXPECT_NEAR(fit.beta, 2.0, 0.5);
    EXPECT_NEAR(fit.alpha[0][0], 0.6, 0.3);
    EXPECT_NEAR(fit.alpha[1][1], 0.5, 0.3);
    EXPECT_LT(fit.spectral_radius, 1.0);
    EXPECT_TRUE(fit.warning.empty());
}

TEST(HawkesUni, NearCriticalFitWarnsButSucceeds) {
    // pinned near-critical DGP: branching 0.95
    const auto t = lobmod::simulate_uni(0.2, 1.9, 2.0, 150.0, 999);
    ASSERT_GT(t.size(), 50u);
    const auto fit = lobmod::fit_uni(t, 150.0);
    EXPECT_TRUE(fit.converged);
    EXPECT_GE(fit.branching, 0.9); // near-critical region recovered
    EXPECT_FALSE(fit.warning.empty()); // reported, not thrown
    EXPECT_NE(fit.warning.find("critical"), std::string::npos);
}

TEST(HawkesBiv, FitRequiresBothTypes) {
    const std::vector<double> t = {1.0, 2.0, 3.0};
    const std::vector<int> k = {0, 0, 0};
    EXPECT_THROW(lobmod::fit_biv(t, k, 10.0), std::invalid_argument);
}

TEST(HawkesUni, ResidualsExpOneAtTrueParams) {
    const auto& t = testdata::uni_times();
    const auto tau = lobmod::uni_residuals(t, testdata::kUniT, 0.5, 1.0, 1.6);
    ASSERT_EQ(tau.size(), t.size());
    double mean = 0.0;
    for (double v : tau) {
        ASSERT_GT(v, 0.0); // compensator strictly increasing at events
        mean += v;
    }
    mean /= static_cast<double>(tau.size());
    EXPECT_NEAR(mean, 1.0, 0.1); // Exp(1) residuals
    const auto [d, p] = lobmod::ks_test_exp(tau);
    EXPECT_GT(p, 0.05); // true params: no rejection
    (void)d;
    // wrong params must be rejected
    const auto tau_wrong = lobmod::uni_residuals(t, testdata::kUniT, 1.5, 0.1, 3.0);
    const auto [dw, pw] = lobmod::ks_test_exp(tau_wrong);
    EXPECT_LT(pw, 0.01);
    EXPECT_GT(dw, 0.05);
}

TEST(HawkesBiv, ResidualsExpOneAtTrueParams) {
    const auto& [t, k] = testdata::biv_data();
    const auto taus = lobmod::biv_residuals(t, k, testdata::kBivT, {0.3, 0.3},
                                            {{{0.6, 0.3}, {0.25, 0.5}}}, 2.0);
    std::size_t total = 0;
    for (int m = 0; m < 2; ++m) {
        ASSERT_FALSE(taus[m].empty());
        total += taus[m].size();
        double mean = 0.0;
        for (double v : taus[m]) {
            ASSERT_GT(v, 0.0); // per-type compensator monotone
            mean += v;
        }
        mean /= static_cast<double>(taus[m].size());
        EXPECT_NEAR(mean, 1.0, 0.15);
        const auto [d, p] = lobmod::ks_test_exp(taus[m]);
        EXPECT_GT(p, 0.05);
        (void)d;
    }
    EXPECT_EQ(total, t.size());
}

TEST(Hawkes, SpectralRadius2x2) {
    EXPECT_NEAR(lobmod::spectral_radius_2x2({{{2.0, 0.0}, {0.0, 3.0}}}), 3.0,
                1e-12);
    // rotation-like matrix: complex eigenvalues, modulus sqrt(det)
    EXPECT_NEAR(lobmod::spectral_radius_2x2({{{0.0, -1.0}, {1.0, 0.0}}}), 1.0,
                1e-12);
    EXPECT_THROW(
        lobmod::spectral_radius_2x2({{{std::nan(""), 0.0}, {0.0, 1.0}}}),
        std::invalid_argument);
}

// ------------------------------------------------------------ review additions

TEST(HawkesFit, ObjectivesArePureUnderUnderflow) {
    // PT-6
    const auto& [t, k] = testdata::biv_data();
    const auto obj = lobmod::biv_fit_objective(t, k, testdata::kBivT);
    EXPECT_EQ(obj({-800.0, -800.0, 0.0, 0.0, 0.0, 0.0, -800.0}), 1e10);
    EXPECT_EQ(obj({-800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}), 1e10);  // mu0 underflow
    EXPECT_EQ(obj({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -800.0}), 1e10);  // beta underflow
    EXPECT_EQ(obj({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 800.0}), 1e10);   // overflow
    const double mu0 = 0.5 * 323 / testdata::kBivT, mu1 = 0.5 * 288 / testdata::kBivT;
    const std::vector<double> x0 = {std::log(mu0), std::log(mu1), std::log(0.25),
                                    std::log(0.25), std::log(0.25), std::log(0.25), 0.0};
    EXPECT_NEAR(obj(x0), -lobmod::biv_loglik(t, k, testdata::kBivT, {mu0, mu1},
                                             {{{0.25, 0.25}, {0.25, 0.25}}}, 1.0), 1e-12);
    const auto res = lobmod::nelder_mead(obj, {-800.0, -800.0, 0.0, 0.0, 0.0, 0.0, -800.0},
                                         1e-8, 1e-10, 5);
    EXPECT_TRUE(std::isfinite(res.fval));
    const auto uobj = lobmod::uni_fit_objective(testdata::uni_times(), testdata::kUniT);
    EXPECT_EQ(uobj({-800.0, 0.0, 0.0}), 1e10);
    EXPECT_EQ(uobj({0.0, 0.0, -800.0}), 1e10);
    EXPECT_NEAR(uobj({0.0, -800.0, 0.0}),
                -lobmod::uni_loglik(testdata::uni_times(), testdata::kUniT, 1.0, 0.0, 1.0), 1e-12);
    const auto capped = lobmod::fit_biv(t, k, testdata::kBivT, 1);
    EXPECT_FALSE(capped.converged);
    EXPECT_EQ(capped.n_iter, 2); // two stages x 1 iteration
    const auto capped_u = lobmod::fit_uni(testdata::uni_times(), testdata::kUniT, 1);
    EXPECT_FALSE(capped_u.converged);
    EXPECT_EQ(capped_u.n_iter, 1);
    EXPECT_THROW(lobmod::fit_uni(testdata::uni_times(), testdata::kUniT, 0), std::invalid_argument);
}

TEST(HawkesUni, PoissonLimitOfFit) {
    // PT-8: alpha = 0 is a homogeneous Poisson process; the MLE must find it
    const double T = 2000.0;
    const auto t = lobmod::simulate_uni(1.0, 0.0, 1.0, T, 5);
    ASSERT_GT(t.size(), 1500u);
    const auto fit = lobmod::fit_uni(t, T);
    EXPECT_TRUE(fit.converged);
    EXPECT_LT(fit.branching, 0.05);
    const double n = static_cast<double>(t.size());
    const double poisson_ll = lobmod::uni_loglik(t, T, n / T, 0.0, 1.0);
    EXPECT_GE(fit.loglik, poisson_ll - 1e-6);
    EXPECT_NEAR(poisson_ll, n * std::log(n / T) - n, 1e-9);
    EXPECT_EQ(lobmod::uni_loglik(t, T, 0.7, 0.0, 1.0), lobmod::uni_loglik(t, T, 0.7, 0.0, 3.3));
}

TEST(HawkesUni, SingleEventPaths) {
    // PT-9
    const double mu = 0.5, alpha = 0.8, beta = 1.5;
    const std::vector<double> t = {0.7};
    const double expected = std::log(mu) - mu * 1.0 + (alpha / beta) * std::expm1(-beta * 0.3);
    EXPECT_NEAR(lobmod::uni_loglik(t, 1.0, mu, alpha, beta), expected, 1e-15);
    EXPECT_NEAR(lobmod::uni_loglik_brute(t, 1.0, mu, alpha, beta), expected, 1e-15);
    const auto tau = lobmod::uni_residuals(t, 1.0, mu, alpha, beta);
    ASSERT_EQ(tau.size(), 1u);
    EXPECT_NEAR(tau[0], mu * 0.7, 1e-15);
    const auto [d, p] = lobmod::ks_test_exp(tau);
    EXPECT_GT(d, 0.0);
    EXPECT_LT(d, 1.0);
    EXPECT_GT(p, 0.0);
    EXPECT_LE(p, 1.0);
    const auto fit = lobmod::fit_uni(t, 1.0);
    EXPECT_TRUE(std::isfinite(fit.loglik));
    EXPECT_GT(fit.mu, 0.0);
    EXPECT_GE(fit.n_iter, 1);
    const std::vector<double> tT = {1.0}; // event exactly at the horizon: expm1(0) = 0
    EXPECT_NEAR(lobmod::uni_loglik(tT, 1.0, mu, alpha, beta), std::log(mu) - mu, 1e-15);
    EXPECT_NEAR(lobmod::uni_residuals(tT, 1.0, mu, alpha, beta)[0], mu, 1e-15);
    EXPECT_THROW(lobmod::fit_biv(tT, {0}, 1.0), std::invalid_argument);
    EXPECT_NEAR(lobmod::biv_loglik(tT, {0}, 1.0, {0.2, 0.3}, {{{0.1, 0.1}, {0.1, 0.1}}}, 2.0),
                std::log(0.2) - 0.5, 1e-15);
}

TEST(Hawkes, TimesCsvRoundTripAndRobustness) {
    const std::string path = testing::TempDir() + "lobmod_u.csv";
    lobmod::write_times_csv(path, testdata::uni_times());
    EXPECT_EQ(lobmod::read_times_csv(path), testdata::uni_times());
    EXPECT_EQ(slurp(path), slurp(testdata::data_dir() + "/hawkes_uni.csv")); // %.17g: byte-identical
    const auto& [t, k] = testdata::biv_data();
    const std::string q = testing::TempDir() + "lobmod_b.csv";
    lobmod::write_marked_times_csv(q, t, k);
    const auto back = lobmod::read_marked_times_csv(q);
    EXPECT_EQ(back.first, t);
    EXPECT_EQ(back.second, k);
    EXPECT_EQ(slurp(q), slurp(testdata::data_dir() + "/hawkes_biv.csv"));
    std::remove(path.c_str());
    std::remove(q.c_str());

    const std::string body = "t\n0.5\n1.25e0\n";
    const std::vector<double> want = {0.5, 1.25};
    EXPECT_EQ(lobmod::read_times_csv(scratch("crlf.csv", "t\r\n0.5\r\n1.25e0\r\n")), want);
    EXPECT_EQ(lobmod::read_times_csv(scratch("trail.csv", body + "\n")), want);
    const std::pair<const char*, std::string> bad[] = {
        {"nan.csv", body + "nan\n"},   {"inf.csv", body + "inf\n"},
        {"two.csv", body + "1.5,0\n"}, {"word.csv", body + "abc\n"},
        {"hdr.csv", "time\n0.5\n"},   {"blank.csv", "t\n0.5\n\n1.0\n"},
        {"space.csv", "t\n 0.5\n"},
    };
    for (const auto& [name, text] : bad) {
        EXPECT_THROW(lobmod::read_times_csv(scratch(name, text)), std::invalid_argument) << name;
    }
    const std::string mbody = "t,type\n0.5,0\n1.0,1\n";
    const auto m = lobmod::read_marked_times_csv(scratch("m.csv", mbody));
    EXPECT_EQ(m.first, (std::vector<double>{0.5, 1.0}));
    EXPECT_EQ(m.second, (std::vector<int>{0, 1}));
    const std::pair<const char*, std::string> mbad[] = {
        {"type2.csv", mbody + "1.5,2\n"}, {"typef.csv", mbody + "1.5,1.0\n"},
        {"one.csv", mbody + "1.5\n"},     {"three.csv", mbody + "1.5,1,\n"},
        {"mhdr.csv", "type,t\n0,0.5\n"},
    };
    for (const auto& [name, text] : mbad) {
        EXPECT_THROW(lobmod::read_marked_times_csv(scratch(name, text)), std::invalid_argument) << name;
    }
    EXPECT_THROW(lobmod::write_times_csv(scratch("x.csv", ""), {1.0, std::nan("")}),
                 std::invalid_argument);
    EXPECT_THROW(lobmod::write_marked_times_csv(scratch("y.csv", ""), {1.0, 2.0}, {0, 2}),
                 std::invalid_argument);
}

TEST(HawkesUni, ExactPoissonLoglikWithZeroAlphaIndependentOfBeta) {
    const std::vector<double> t = {0.5, 1.2, 1.35, 2.8, 3.1, 3.15, 4.9, 7.2, 7.5, 9.0};
    EXPECT_NEAR(lobmod::uni_loglik(t, 10.0, 0.7, 0.0, 1.0),
                10.0 * std::log(0.7) - 7.0, 1e-12);
}

} // namespace
