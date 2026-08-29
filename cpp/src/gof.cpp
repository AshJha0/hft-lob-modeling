#include "lobmod/gof.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lobmod {

double ks_statistic_exp(const std::vector<double>& sample) {
    if (sample.empty()) {
        throw std::invalid_argument("sample must be a non-empty array");
    }
    for (double v : sample) {
        if (!std::isfinite(v) || v < 0.0) {
            throw std::invalid_argument(
                "sample values must be finite and non-negative");
        }
    }
    std::vector<double> xs = sample;
    std::sort(xs.begin(), xs.end());
    const double n = static_cast<double>(xs.size());
    double d = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double cdf = -std::expm1(-xs[i]); // 1 - exp(-x), accurately
        const double hi = static_cast<double>(i + 1) / n - cdf;
        const double lo = cdf - static_cast<double>(i) / n;
        d = std::max(d, std::max(hi, lo));
    }
    return d;
}

double ks_pvalue(double d, long long n) {
    if (!(std::isfinite(d) && d >= 0.0 && d <= 1.0)) {
        throw std::domain_error("d must be in [0, 1], got " + std::to_string(d));
    }
    if (n <= 0) {
        throw std::invalid_argument("n must be a positive integer, got " +
                                    std::to_string(n));
    }
    if (d == 0.0) return 1.0;
    const double sqrt_n = std::sqrt(static_cast<double>(n));
    const double lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d;
    double total = 0.0;
    for (int j = 1; j <= 100; ++j) { // pinned: exactly 100 terms
        const double term =
            std::exp(-2.0 * static_cast<double>(j) * static_cast<double>(j) *
                     lam * lam);
        total += j % 2 == 1 ? term : -term;
    }
    const double p = 2.0 * total;
    return std::min(1.0, std::max(0.0, p));
}

std::pair<double, double> ks_test_exp(const std::vector<double>& sample) {
    const double d = ks_statistic_exp(sample);
    return {d, ks_pvalue(d, static_cast<long long>(sample.size()))};
}

} // namespace lobmod
