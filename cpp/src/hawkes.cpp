#include "lobmod/hawkes.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

#include "lobmod/csv.hpp"
#include "lobmod/neldermead.hpp"
#include "lobmod/rng.hpp"

namespace lobmod {

namespace {

constexpr double kPenalty = 1e10;

void check_times(const std::vector<double>& t, double T) {
    if (t.empty()) {
        throw std::invalid_argument("times must be a non-empty array");
    }
    for (double v : t) {
        if (!std::isfinite(v)) throw std::invalid_argument("times must be finite");
    }
    if (t.front() <= 0.0) {
        throw std::invalid_argument("times must be strictly positive");
    }
    for (std::size_t i = 1; i < t.size(); ++i) {
        if (t[i] - t[i - 1] <= 0.0) {
            throw std::invalid_argument("times must be strictly increasing");
        }
    }
    if (!(std::isfinite(T) && T > 0.0)) {
        throw std::invalid_argument("T must be positive and finite, got " +
                                    std::to_string(T));
    }
    if (t.back() > T) {
        throw std::invalid_argument("last event time " + std::to_string(t.back()) +
                                    " exceeds horizon T=" + std::to_string(T));
    }
}

void check_uni_params(double mu, double alpha, double beta) {
    if (!(std::isfinite(mu) && mu > 0.0)) {
        throw std::invalid_argument("mu must be positive and finite, got " +
                                    std::to_string(mu));
    }
    if (!(std::isfinite(alpha) && alpha >= 0.0)) {
        throw std::invalid_argument("alpha must be non-negative and finite, got " +
                                    std::to_string(alpha));
    }
    if (!(std::isfinite(beta) && beta > 0.0)) {
        throw std::invalid_argument("beta must be positive and finite, got " +
                                    std::to_string(beta));
    }
}

void check_biv_params(const std::array<double, 2>& mu, const Mat2& alpha,
                      double beta) {
    for (double m : mu) {
        if (!(std::isfinite(m) && m > 0.0)) {
            throw std::invalid_argument("mu must be two positive finite values");
        }
    }
    for (const auto& row : alpha) {
        for (double a : row) {
            if (!(std::isfinite(a) && a >= 0.0)) {
                throw std::invalid_argument(
                    "alpha must be a 2x2 non-negative finite matrix");
            }
        }
    }
    if (!(std::isfinite(beta) && beta > 0.0)) {
        throw std::invalid_argument("beta must be positive and finite, got " +
                                    std::to_string(beta));
    }
}

void check_types(const std::vector<int>& types, std::size_t n) {
    if (types.size() != n) {
        throw std::invalid_argument("types must have the same length as times");
    }
    for (int k : types) {
        if (k != 0 && k != 1) {
            throw std::invalid_argument("types must be 0 or 1");
        }
    }
}

std::string branching_warning(double value) {
    if (value >= 1.0) {
        return "fitted Hawkes process is non-stationary (branching " +
               std::to_string(value) + " >= 1)";
    }
    if (value >= 0.9) {
        return "fitted Hawkes process is near-critical (branching " +
               std::to_string(value) + " >= 0.9)";
    }
    return std::string();
}

} // namespace

double uni_loglik(const std::vector<double>& times, double T, double mu,
                  double alpha, double beta) {
    check_times(times, T);
    check_uni_params(mu, alpha, beta);
    double ll = std::log(mu); // R_1 = 0
    double r = 0.0;
    for (std::size_t i = 1; i < times.size(); ++i) {
        const double w = std::exp(-beta * (times[i] - times[i - 1]));
        r = w * (1.0 + r);
        ll += std::log(mu + alpha * r);
    }
    ll -= mu * T;
    double tail = 0.0;
    for (double t : times) tail += std::expm1(-beta * (T - t));
    ll += (alpha / beta) * tail;
    return ll;
}

double uni_loglik_brute(const std::vector<double>& times, double T, double mu,
                        double alpha, double beta) {
    check_times(times, T);
    check_uni_params(mu, alpha, beta);
    double ll = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        double excitation = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            excitation += std::exp(-beta * (times[i] - times[j]));
        }
        ll += std::log(mu + alpha * excitation);
    }
    ll -= mu * T;
    double tail = 0.0;
    for (double t : times) tail += std::expm1(-beta * (T - t));
    ll += (alpha / beta) * tail;
    return ll;
}

double biv_loglik(const std::vector<double>& times, const std::vector<int>& types,
                  double T, const std::array<double, 2>& mu, const Mat2& alpha,
                  double beta) {
    check_times(times, T);
    check_types(types, times.size());
    check_biv_params(mu, alpha, beta);
    double s0 = 0.0, s1 = 0.0, prev = 0.0, ll = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double w = std::exp(-beta * (times[i] - prev));
        s0 *= w;
        s1 *= w;
        const int m = types[i];
        const double lam = mu[m] + alpha[m][0] * s0 + alpha[m][1] * s1;
        ll += std::log(lam);
        if (m == 0) {
            s0 += 1.0;
        } else {
            s1 += 1.0;
        }
        prev = times[i];
    }
    ll -= (mu[0] + mu[1]) * T;
    for (int n = 0; n < 2; ++n) {
        double s = 0.0;
        for (std::size_t i = 0; i < times.size(); ++i) {
            if (types[i] == n) s += std::expm1(-beta * (T - times[i]));
        }
        ll += ((alpha[0][n] + alpha[1][n]) / beta) * s;
    }
    return ll;
}

double biv_loglik_brute(const std::vector<double>& times,
                        const std::vector<int>& types, double T,
                        const std::array<double, 2>& mu, const Mat2& alpha,
                        double beta) {
    check_times(times, T);
    check_types(types, times.size());
    check_biv_params(mu, alpha, beta);
    double ll = 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        double lam = mu[types[i]];
        for (int n = 0; n < 2; ++n) {
            double s = 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                if (types[j] == n) s += std::exp(-beta * (times[i] - times[j]));
            }
            lam += alpha[types[i]][n] * s;
        }
        ll += std::log(lam);
    }
    ll -= (mu[0] + mu[1]) * T;
    for (int n = 0; n < 2; ++n) {
        double s = 0.0;
        for (std::size_t i = 0; i < times.size(); ++i) {
            if (types[i] == n) s += std::expm1(-beta * (T - times[i]));
        }
        ll += ((alpha[0][n] + alpha[1][n]) / beta) * s;
    }
    return ll;
}

std::vector<double> uni_residuals(const std::vector<double>& times, double T,
                                  double mu, double alpha, double beta) {
    check_times(times, T);
    check_uni_params(mu, alpha, beta);
    const std::size_t n = times.size();
    std::vector<double> tau(n);
    double s = 0.0;
    double prev_lambda = 0.0; // Lambda(t_0) = 0
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) {
            s = std::exp(-beta * (times[i] - times[i - 1])) * (s + 1.0);
        }
        const double lam_int =
            mu * times[i] + (alpha / beta) * (static_cast<double>(i) - s);
        tau[i] = lam_int - prev_lambda;
        prev_lambda = lam_int;
    }
    return tau;
}

std::array<std::vector<double>, 2>
biv_residuals(const std::vector<double>& times, const std::vector<int>& types,
              double T, const std::array<double, 2>& mu, const Mat2& alpha,
              double beta) {
    check_times(times, T);
    check_types(types, times.size());
    check_biv_params(mu, alpha, beta);
    double s[2] = {0.0, 0.0}; // decayed exponential sums per source type
    double c[2] = {0.0, 0.0}; // event counts strictly before the current time
    double prev_t = 0.0;
    std::array<std::vector<double>, 2> lam_int;
    for (std::size_t i = 0; i < times.size(); ++i) {
        const double w = std::exp(-beta * (times[i] - prev_t));
        s[0] *= w;
        s[1] *= w;
        const int m = types[i];
        double val = mu[m] * times[i];
        for (int n = 0; n < 2; ++n) {
            val += (alpha[m][n] / beta) * (c[n] - s[n]);
        }
        lam_int[m].push_back(val);
        s[m] += 1.0;
        c[m] += 1.0;
        prev_t = times[i];
    }
    std::array<std::vector<double>, 2> tau;
    for (int m = 0; m < 2; ++m) {
        tau[m].resize(lam_int[m].size());
        double prev = 0.0;
        for (std::size_t i = 0; i < lam_int[m].size(); ++i) {
            tau[m][i] = lam_int[m][i] - prev;
            prev = lam_int[m][i];
        }
    }
    return tau;
}

std::vector<double> simulate_uni(double mu, double alpha, double beta, double T,
                                 std::int64_t seed) {
    check_uni_params(mu, alpha, beta);
    if (alpha / beta >= 1.0) {
        throw std::domain_error("cannot simulate non-stationary process: "
                                "alpha/beta = " + std::to_string(alpha / beta));
    }
    if (!(std::isfinite(T) && T > 0.0)) {
        throw std::invalid_argument("T must be positive and finite, got " +
                                    std::to_string(T));
    }
    Lcg64 rng(seed);
    double t = 0.0;
    double s = 0.0;
    std::vector<double> out;
    for (std::int64_t n = 0; n < kMaxCandidates; ++n) {
        const double lam_bar = mu + alpha * s; // intensity just after last step
        const double w = rng.exponential(lam_bar);
        t += w;
        if (t > T) return out;
        s *= std::exp(-beta * w);
        const double lam_t = mu + alpha * s;
        if (rng.uniform() * lam_bar <= lam_t) {
            out.push_back(t);
            s += 1.0;
        }
    }
    throw std::invalid_argument("simulate_uni exceeded kMaxCandidates thinning draws");
}

std::pair<std::vector<double>, std::vector<int>>
simulate_biv(const std::array<double, 2>& mu, const Mat2& alpha, double beta,
             double T, std::int64_t seed) {
    check_biv_params(mu, alpha, beta);
    const double disc = (alpha[0][0] - alpha[1][1]) * (alpha[0][0] - alpha[1][1]) +
                        4.0 * alpha[0][1] * alpha[1][0];
    const double rad = 0.5 * (alpha[0][0] + alpha[1][1] + std::sqrt(disc)) / beta;
    if (rad >= 1.0) {
        throw std::domain_error("cannot simulate non-stationary process: "
                                "spectral radius " + std::to_string(rad));
    }
    if (!(std::isfinite(T) && T > 0.0)) {
        throw std::invalid_argument("T must be positive and finite, got " +
                                    std::to_string(T));
    }
    Lcg64 rng(seed);
    double t = 0.0;
    double s[2] = {0.0, 0.0};
    std::vector<double> times;
    std::vector<int> types;
    for (std::int64_t n = 0; n < kMaxCandidates; ++n) {
        double lam0 = mu[0] + alpha[0][0] * s[0] + alpha[0][1] * s[1];
        double lam1 = mu[1] + alpha[1][0] * s[0] + alpha[1][1] * s[1];
        const double lam_bar = lam0 + lam1;
        const double w = rng.exponential(lam_bar);
        t += w;
        if (t > T) return {std::move(times), std::move(types)};
        const double decay = std::exp(-beta * w);
        s[0] *= decay;
        s[1] *= decay;
        lam0 = mu[0] + alpha[0][0] * s[0] + alpha[0][1] * s[1];
        lam1 = mu[1] + alpha[1][0] * s[0] + alpha[1][1] * s[1];
        const double d = rng.uniform() * lam_bar;
        if (d <= lam0 + lam1) {
            const int typ = d <= lam0 ? 0 : 1;
            times.push_back(t);
            types.push_back(typ);
            s[typ] += 1.0;
        }
    }
    throw std::invalid_argument("simulate_biv exceeded kMaxCandidates thinning draws");
}

double spectral_radius_2x2(const Mat2& m) {
    for (const auto& row : m) {
        for (double v : row) {
            if (!std::isfinite(v)) {
                throw std::invalid_argument("matrix must be a finite 2x2 array");
            }
        }
    }
    const double tr = m[0][0] + m[1][1];
    const double det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    const double disc = tr * tr - 4.0 * det;
    if (disc >= 0.0) {
        const double r = std::sqrt(disc);
        return std::max(std::abs(0.5 * (tr + r)), std::abs(0.5 * (tr - r)));
    }
    return std::sqrt(det);
}

FitObjective uni_fit_objective(const std::vector<double>& times, double T) {
    check_times(times, T);
    return [times, T](const std::vector<double>& x) -> double {
        const double mu = std::exp(x[0]);
        const double alpha = std::exp(x[1]);
        const double beta = std::exp(x[2]);
        if (!(std::isfinite(mu) && std::isfinite(alpha) && std::isfinite(beta)) ||
            mu <= 0.0 || beta <= 0.0) {
            return kPenalty;
        }
        const double val = uni_loglik(times, T, mu, alpha, beta);
        return std::isfinite(val) ? -val : kPenalty;
    };
}

FitObjective biv_fit_objective(const std::vector<double>& times,
                               const std::vector<int>& types, double T) {
    check_times(times, T);
    check_types(types, times.size());
    return [times, types, T](const std::vector<double>& x) -> double {
        double p[7];
        for (int i = 0; i < 7; ++i) {
            p[i] = std::exp(x[i]);
            if (!std::isfinite(p[i])) return kPenalty;
        }
        // exp() can underflow to exactly 0, which the likelihood's parameter
        // validation would reject: treat it as out of domain (pure objective).
        if (p[0] <= 0.0 || p[1] <= 0.0 || p[6] <= 0.0) return kPenalty;
        const std::array<double, 2> mu = {p[0], p[1]};
        const Mat2 alpha = {{{p[2], p[3]}, {p[4], p[5]}}};
        const double val = biv_loglik(times, types, T, mu, alpha, p[6]);
        return std::isfinite(val) ? -val : kPenalty;
    };
}

HawkesUniFit fit_uni(const std::vector<double>& times, double T, int maxiter) {
    const FitObjective objective = uni_fit_objective(times, T);
    const std::vector<double> x0 = {
        std::log(0.5 * static_cast<double>(times.size()) / T), std::log(0.5), 0.0};
    const NMResult res = nelder_mead(objective, x0, 1e-8, 1e-10, maxiter);
    const double mu = std::exp(res.x[0]);
    const double alpha = std::exp(res.x[1]);
    const double beta = std::exp(res.x[2]);
    const double branching = alpha / beta;
    return HawkesUniFit{mu,        alpha,        beta,
                        -res.fval, branching,    res.converged,
                        res.n_iter, branching_warning(branching)};
}

HawkesBivFit fit_biv(const std::vector<double>& times,
                     const std::vector<int>& types, double T, int maxiter) {
    check_times(times, T);
    check_types(types, times.size());
    std::size_t n0 = 0;
    for (int k : types) {
        if (k == 0) ++n0;
    }
    const std::size_t n1 = times.size() - n0;
    if (n0 == 0 || n1 == 0) {
        throw std::invalid_argument(
            "both event types must be present to fit the bivariate model");
    }
    const FitObjective objective = biv_fit_objective(times, types, T);
    const std::vector<double> x0 = {
        std::log(0.5 * static_cast<double>(n0) / T),
        std::log(0.5 * static_cast<double>(n1) / T),
        std::log(0.25), std::log(0.25), std::log(0.25), std::log(0.25),
        0.0};
    const NMResult res1 = nelder_mead(objective, x0, 1e-8, 1e-10, maxiter);
    // pinned polish stage
    const NMResult res2 = nelder_mead(objective, res1.x, 1e-8, 1e-10, maxiter);
    double p[7];
    for (int i = 0; i < 7; ++i) p[i] = std::exp(res2.x[i]);
    const std::array<double, 2> mu = {p[0], p[1]};
    const Mat2 alpha = {{{p[2], p[3]}, {p[4], p[5]}}};
    const double beta = p[6];
    Mat2 branching;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) branching[i][j] = alpha[i][j] / beta;
    }
    const double rad = spectral_radius_2x2(branching);
    return HawkesBivFit{mu,
                        alpha,
                        beta,
                        -res2.fval,
                        branching,
                        rad,
                        res2.converged,
                        res1.n_iter + res2.n_iter,
                        branching_warning(rad)};
}

// -------------------------------------------------------------------- CSV i/o

namespace {

std::string format_g17(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

} // namespace

std::vector<double> read_times_csv(const std::string& path) {
    const auto lines = read_csv_lines(path, "t");
    std::vector<double> out;
    out.reserve(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const std::string context = "time row " + std::to_string(i + 1);
        if (lines[i].find(',') != std::string::npos) {
            throw std::invalid_argument("malformed " + context +
                                        ": expected 1 field: '" + lines[i] + "'");
        }
        out.push_back(parse_double_strict(lines[i], context));
    }
    return out;
}

std::pair<std::vector<double>, std::vector<int>>
read_marked_times_csv(const std::string& path) {
    const auto lines = read_csv_lines(path, "t,type");
    std::pair<std::vector<double>, std::vector<int>> out;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const std::string context = "marked-time row " + std::to_string(i + 1);
        const auto f = split_csv_fields(lines[i]);
        if (f.size() != 2) {
            throw std::invalid_argument("malformed " + context +
                                        ": expected 2 fields: '" + lines[i] + "'");
        }
        const std::int64_t k = parse_int64_strict(f[1], context);
        if (k != 0 && k != 1) {
            throw std::invalid_argument("types must be 0 or 1, got " +
                                        std::to_string(k) + " in " + context);
        }
        out.first.push_back(parse_double_strict(f[0], context));
        out.second.push_back(static_cast<int>(k));
    }
    return out;
}

void write_times_csv(const std::string& path, const std::vector<double>& times) {
    for (double v : times) {
        if (!std::isfinite(v)) throw std::invalid_argument("times must be finite");
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::invalid_argument("cannot write CSV: " + path);
    out << "t\n";
    for (double v : times) out << format_g17(v) << '\n';
    if (!out) throw std::invalid_argument("write failed for CSV: " + path);
}

void write_marked_times_csv(const std::string& path,
                            const std::vector<double>& times,
                            const std::vector<int>& types) {
    for (double v : times) {
        if (!std::isfinite(v)) throw std::invalid_argument("times must be finite");
    }
    check_types(types, times.size());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::invalid_argument("cannot write CSV: " + path);
    out << "t,type\n";
    for (std::size_t i = 0; i < times.size(); ++i) {
        out << format_g17(times[i]) << ',' << types[i] << '\n';
    }
    if (!out) throw std::invalid_argument("write failed for CSV: " + path);
}

} // namespace lobmod
