#ifndef LOBMOD_GOF_HPP
#define LOBMOD_GOF_HPP

/// \file gof.hpp
/// \brief Goodness of fit: Kolmogorov-Smirnov test of Exp(1) residuals.
///
/// By the time-rescaling theorem a correctly specified Hawkes model turns
/// its compensator increments into iid Exp(1) draws.  The KS statistic
/// against the Exp(1) CDF F(x) = 1 - exp(-x) is
///
///     D_n = max_i max( i/n - F(x_(i)), F(x_(i)) - (i-1)/n )
///
/// over the sorted sample, and the asymptotic p-value uses the Stephens
/// small-sample correction lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D_n
/// and the Kolmogorov tail series 2 sum_{j=1..100} (-1)^{j-1}
/// exp(-2 j^2 lambda^2), clipped to [0, 1] — pinned at exactly 100 terms.
/// With very small n the asymptotic p-value is indicative only (documented,
/// not an error).

#include <cstddef>
#include <utility>
#include <vector>

namespace lobmod {

/// KS statistic of a sample against the Exp(1) CDF.
/// \throws std::invalid_argument on an empty sample or non-finite/negative
///         values.
double ks_statistic_exp(const std::vector<double>& sample);

/// Asymptotic Kolmogorov p-value for statistic d on n points (pinned
/// 100-term series with the Stephens correction); exactly 1 when d = 0.
/// \throws std::domain_error if d is outside [0, 1].
/// \throws std::invalid_argument if n <= 0.
double ks_pvalue(double d, long long n);

/// KS test of a sample against Exp(1): returns (statistic, p_value).
std::pair<double, double> ks_test_exp(const std::vector<double>& sample);

} // namespace lobmod

#endif // LOBMOD_GOF_HPP
