/// KS test against Exp(1): exact small-sample statistics, pinned p-value
/// series, tiny-N behavior, monotonicity, and validation errors.

#include <cmath>
#include <stdexcept>
#include <gtest/gtest.h>

#include "lobmod/gof.hpp"

namespace {

TEST(Ks, StatisticSingleObservation) {
    // n = 1: D = max(1 - F(x), F(x)) with F(x) = 1 - exp(-x)
    const double x = 0.3;
    const double f = -std::expm1(-x);
    const double d = lobmod::ks_statistic_exp({x});
    EXPECT_NEAR(d, std::max(1.0 - f, f), 1e-15);
    const auto [ds, p] = lobmod::ks_test_exp({x});
    EXPECT_EQ(ds, d);
    EXPECT_GT(p, 0.0);
    EXPECT_LE(p, 1.0);
}

TEST(Ks, TinySamplesWork) {
    // n = 3 (documented: asymptotic p only indicative — but never an error)
    const auto [d, p] = lobmod::ks_test_exp({0.2, 0.9, 2.5});
    EXPECT_GT(d, 0.0);
    EXPECT_LE(d, 1.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
}

TEST(Ks, PerfectExpQuantilesGiveSmallD) {
    // quantiles of Exp(1) at (i - 0.5)/n: D should be about 1/(2n)
    const int n = 100;
    std::vector<double> x;
    for (int i = 1; i <= n; ++i) {
        x.push_back(-std::log1p(-(i - 0.5) / n));
    }
    const auto [d, p] = lobmod::ks_test_exp(x);
    EXPECT_NEAR(d, 0.5 / n, 1e-12);
    EXPECT_GT(p, 0.99);
}

TEST(Ks, PvaluePinnedFormulaProperties) {
    EXPECT_DOUBLE_EQ(lobmod::ks_pvalue(0.0, 10), 1.0); // exactly 1 at D = 0
    EXPECT_DOUBLE_EQ(lobmod::ks_pvalue(1.0, 1000), 0.0); // clipped at 0
    // monotone decreasing in D on a grid (property-style check)
    double prev = 1.0;
    for (int i = 1; i <= 20; ++i) {
        const double p = lobmod::ks_pvalue(0.05 * i, 50);
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
        EXPECT_LE(p, prev + 1e-15) << "p not monotone at D = " << 0.05 * i;
        prev = p;
    }
}

TEST(Ks, PvalueMatchesKnownValue) {
    // lambda = (sqrt(100) + 0.12 + 0.11/10) * 0.1 = 1.0131; series value
    const double lam = (10.0 + 0.12 + 0.011) * 0.1;
    double total = 0.0;
    for (int j = 1; j <= 100; ++j) {
        const double term = std::exp(-2.0 * j * j * lam * lam);
        total += j % 2 == 1 ? term : -term;
    }
    EXPECT_NEAR(lobmod::ks_pvalue(0.1, 100), 2.0 * total, 1e-15);
}

TEST(Ks, ValidationErrors) {
    EXPECT_THROW(lobmod::ks_statistic_exp({}), std::invalid_argument);
    EXPECT_THROW(lobmod::ks_statistic_exp({0.1, -0.2}), std::invalid_argument);
    EXPECT_THROW(lobmod::ks_statistic_exp({0.1, std::nan("")}),
                 std::invalid_argument);
    EXPECT_THROW(lobmod::ks_pvalue(-0.1, 10), std::domain_error);
    EXPECT_THROW(lobmod::ks_pvalue(1.1, 10), std::domain_error);
    EXPECT_THROW(lobmod::ks_pvalue(std::nan(""), 10), std::domain_error);
    EXPECT_THROW(lobmod::ks_pvalue(0.5, 0), std::invalid_argument);
    EXPECT_THROW(lobmod::ks_pvalue(0.5, -3), std::invalid_argument);
}

// ------------------------------------------------------------ review additions

TEST(Ks, PvalueSmallLambdaAccuracy) {
    // PT-10: the dual series makes tiny statistics give p = 1, not 0.87
    EXPECT_NEAR(lobmod::ks_pvalue(1e-4, 100), 1.0, 1e-9);
    const long long n = 100000;
    EXPECT_GE(lobmod::ks_pvalue(0.5 / static_cast<double>(n), n), 0.999);
    double prev = 1.0;
    for (double d : {0.5 / n, 2.0 / n, 8.0 / n, 32.0 / n, 128.0 / n, 512.0 / n}) {
        const double p = lobmod::ks_pvalue(d, n);
        EXPECT_LE(p, prev + 1e-15); // monotone across the branch switch
        prev = p;
    }
    EXPECT_DOUBLE_EQ(lobmod::kolmogorov_q(0.01), 1.0);
    EXPECT_DOUBLE_EQ(lobmod::kolmogorov_q(0.005), 1.0);
    // continuity at the pinned switch point (both branches are exact here)
    EXPECT_NEAR(lobmod::kolmogorov_q(0.0999), 1.0, 1e-14);
    EXPECT_NEAR(lobmod::kolmogorov_q(0.1), 1.0, 1e-14);
    EXPECT_NEAR(lobmod::kolmogorov_q(0.1001), 1.0, 1e-14);
    // reference values (scipy.special.kolmogorov)
    EXPECT_NEAR(lobmod::kolmogorov_q(0.5), 0.963945243664875, 1e-14);
    EXPECT_NEAR(lobmod::kolmogorov_q(1.0), 0.269999671677355, 1e-14);
    EXPECT_NEAR(lobmod::kolmogorov_q(2.0), 0.000670925255780, 1e-14);
}

} // namespace
