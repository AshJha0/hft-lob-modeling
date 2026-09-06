#ifndef LOBMOD_HAWKES_HPP
#define LOBMOD_HAWKES_HPP

/// \file hawkes.hpp
/// \brief Hawkes processes with exponential kernels: likelihood, MLE,
///        simulation, and time-rescaling residuals.
///
/// Univariate model: lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta (t-t_i))
/// with mu > 0, alpha >= 0, beta > 0.  The branching ratio alpha/beta is the
/// expected number of children per event; stationary iff < 1.
///
/// The log-likelihood on [0, T] uses the pinned O(N) Ozaki recursion
///
///     R_1 = 0,  R_i = exp(-beta (t_i - t_{i-1})) (1 + R_{i-1}),
///     ll = sum_i log(mu + alpha R_i) - mu T
///          + (alpha/beta) sum_i expm1(-beta (T - t_i)).
///
/// Bivariate model (pinned parameterization): two event types, full 2x2
/// excitation matrix alpha and a *scalar shared* beta.  The same recursion
/// runs on the merged stream with per-type decayed sums.
///
/// By the time-rescaling theorem, compensator increments between events are
/// iid Exp(1) under the true model — the residuals fed to the KS test in
/// gof.hpp.
///
/// Units: times and T share one arbitrary time unit; mu and beta are per
/// unit time, alpha per event.  Simulators stop with an error after
/// kMaxCandidates thinning draws (the loop terminates almost surely but has
/// no finite worst case).

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace lobmod {

/// 2x2 matrix as nested std::array (row major: m[row][col]).
using Mat2 = std::array<std::array<double, 2>, 2>;

/// Pinned upper bound on thinning candidates per simulation.
constexpr std::int64_t kMaxCandidates = 50000000LL;

/// Read a one-column event-time CSV with header `t` (pinned CSV rules of
/// csv.hpp; values must be finite — ordering is validated by the
/// likelihood functions).
/// \throws std::invalid_argument on any violation.
std::vector<double> read_times_csv(const std::string& path);

/// Read a two-column `t,type` CSV; types are validated to be 0 or 1.
/// \returns (times, types).
std::pair<std::vector<double>, std::vector<int>>
read_marked_times_csv(const std::string& path);

/// Write event times with header `t` using `%.17g` (exact round trip).
/// \throws std::invalid_argument on non-finite values or an unwritable file.
void write_times_csv(const std::string& path, const std::vector<double>& times);

/// Write `t,type` rows (`%.17g` times, integer types 0/1).
void write_marked_times_csv(const std::string& path,
                            const std::vector<double>& times,
                            const std::vector<int>& types);

/// Exact univariate log-likelihood via the pinned O(N) recursion.
/// \throws std::invalid_argument on bad times (empty, non-finite,
///         non-increasing, t_1 <= 0, t_N > T) or bad parameters.
double uni_loglik(const std::vector<double>& times, double T, double mu,
                  double alpha, double beta);

/// Reference O(N^2) log-likelihood (direct double sum) for tests only.
double uni_loglik_brute(const std::vector<double>& times, double T, double mu,
                        double alpha, double beta);

/// Bivariate log-likelihood (full 2x2 alpha, scalar shared beta), O(N).
/// \param types 0 (buy) / 1 (sell) aligned with times.
double biv_loglik(const std::vector<double>& times,
                  const std::vector<int>& types, double T,
                  const std::array<double, 2>& mu, const Mat2& alpha,
                  double beta);

/// Reference O(N^2) bivariate log-likelihood (tests / validation only).
double biv_loglik_brute(const std::vector<double>& times,
                        const std::vector<int>& types, double T,
                        const std::array<double, 2>& mu, const Mat2& alpha,
                        double beta);

/// Time-rescaling residuals tau_i = Lambda(t_i) - Lambda(t_{i-1}) for the
/// univariate model; iid Exp(1) under the true parameters.
std::vector<double> uni_residuals(const std::vector<double>& times, double T,
                                  double mu, double alpha, double beta);

/// Per-type time-rescaling residuals for the bivariate model:
/// returns {tau_type0, tau_type1}.
std::array<std::vector<double>, 2>
biv_residuals(const std::vector<double>& times, const std::vector<int>& types,
              double T, const std::array<double, 2>& mu, const Mat2& alpha,
              double beta);

/// Simulate a univariate Hawkes process on [0, T] by pinned Ogata thinning
/// with Lcg64(seed): exactly two uniforms per loop iteration.
/// \throws std::invalid_argument on bad parameters or T.
/// \throws std::domain_error if alpha/beta >= 1 (non-stationary).
std::vector<double> simulate_uni(double mu, double alpha, double beta, double T,
                                 std::int64_t seed);

/// Simulate a bivariate Hawkes process by pinned Ogata thinning.
/// \returns (times, types).
/// \throws std::domain_error if the branching spectral radius is >= 1.
std::pair<std::vector<double>, std::vector<int>>
simulate_biv(const std::array<double, 2>& mu, const Mat2& alpha, double beta,
             double T, std::int64_t seed);

/// Spectral radius of a 2x2 matrix via the pinned closed form: eigenvalues
/// (tr +/- sqrt(tr^2 - 4 det)) / 2, complex pair modulus sqrt(det).
double spectral_radius_2x2(const Mat2& m);

/// Univariate MLE result.  Stationarity diagnostics are reported in
/// `warning` (empty when branching < 0.9), never thrown.
struct HawkesUniFit {
    double mu;           ///< fitted background rate
    double alpha;        ///< fitted excitation jump
    double beta;         ///< fitted decay
    double loglik;       ///< maximized log-likelihood
    double branching;    ///< alpha / beta
    bool converged;      ///< Nelder-Mead tolerance convergence flag
    int n_iter;          ///< Nelder-Mead iterations
    std::string warning; ///< near-critical / non-stationary diagnostic
};

/// Bivariate MLE result (full 2x2 alpha, scalar beta).
struct HawkesBivFit {
    std::array<double, 2> mu; ///< fitted background rates
    Mat2 alpha;               ///< fitted excitation matrix
    double beta;              ///< fitted shared decay
    double loglik;            ///< maximized log-likelihood
    Mat2 branching;           ///< alpha / beta
    double spectral_radius;   ///< stationarity measure (stationary iff < 1)
    bool converged;           ///< Nelder-Mead tolerance convergence flag
    int n_iter;               ///< total iterations over both pinned stages
    std::string warning;      ///< near-critical / non-stationary diagnostic
};

/// Objective type minimized by the fits (log-parameter vector -> -loglik).
using FitObjective = std::function<double(const std::vector<double>&)>;

/// The pure objective minimized by fit_uni(): x = (log mu, log alpha,
/// log beta) -> -uni_loglik; returns the penalty 1e10 (never throws) on
/// overflow, non-finite values, or exp underflow of mu/beta to exactly 0.
/// \throws std::invalid_argument on invalid event times (at construction).
FitObjective uni_fit_objective(const std::vector<double>& times, double T);

/// The pure objective minimized by fit_biv(): x = (log mu0, log mu1,
/// log a00, log a01, log a10, log a11, log beta) -> -biv_loglik with the
/// same penalty rules (mu0, mu1, beta underflow -> 1e10).
FitObjective biv_fit_objective(const std::vector<double>& times,
                               const std::vector<int>& types, double T);

/// Univariate MLE via Nelder-Mead on log-transformed parameters.  Pinned
/// setup: x = (log mu, log alpha, log beta), start (log(0.5 N / T),
/// log 0.5, 0), xatol 1e-8 / fatol 1e-10 and `maxiter` (pinned default
/// 5000).  `converged == false` means the iteration cap stopped the
/// search and the parameters are only the best vertex found.
HawkesUniFit fit_uni(const std::vector<double>& times, double T,
                     int maxiter = 5000);

/// Bivariate MLE (pinned): x = (log mu0, log mu1, log a00, log a01,
/// log a10, log a11, log beta), start mu_m = 0.5 N_m / T, a = 0.25,
/// beta = 1; two chained Nelder-Mead runs (each capped at `maxiter`),
/// iterations summed, converged flag from the second run.
/// \throws std::invalid_argument if either event type is absent.
HawkesBivFit fit_biv(const std::vector<double>& times,
                     const std::vector<int>& types, double T,
                     int maxiter = 5000);

} // namespace lobmod

#endif // LOBMOD_HAWKES_HPP
