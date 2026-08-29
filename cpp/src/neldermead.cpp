#include "lobmod/neldermead.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace lobmod {

namespace {

/// Stable ascending sort of the simplex by objective value (key (f_i, i)).
void sort_simplex(std::vector<std::vector<double>>& simplex,
                  std::vector<double>& fvals) {
    const std::size_t m = fvals.size();
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return fvals[a] < fvals[b]; });
    std::vector<std::vector<double>> s2(m);
    std::vector<double> f2(m);
    for (std::size_t i = 0; i < m; ++i) {
        s2[i] = std::move(simplex[order[i]]);
        f2[i] = fvals[order[i]];
    }
    simplex = std::move(s2);
    fvals = std::move(f2);
}

} // namespace

NMResult nelder_mead(const std::function<double(const std::vector<double>&)>& f,
                     std::vector<double> x0, double xatol, double fatol,
                     int maxiter) {
    const std::size_t n = x0.size();
    if (n == 0) {
        throw std::invalid_argument("x0 must be non-empty");
    }
    for (double v : x0) {
        if (!std::isfinite(v)) throw std::invalid_argument("x0 must be finite");
    }
    if (!(xatol > 0.0 && fatol > 0.0 && maxiter >= 1)) {
        throw std::invalid_argument("tolerances must be positive and maxiter >= 1");
    }

    // initial simplex (scipy rule)
    std::vector<std::vector<double>> simplex;
    simplex.push_back(x0);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> v = x0;
        v[i] = v[i] != 0.0 ? v[i] * 1.05 : 0.00025;
        simplex.push_back(std::move(v));
    }
    std::vector<double> fvals(n + 1);
    for (std::size_t i = 0; i <= n; ++i) fvals[i] = f(simplex[i]);

    int n_iter = 0;
    bool converged = false;
    while (n_iter < maxiter) {
        sort_simplex(simplex, fvals);

        double xspread = 0.0;
        for (std::size_t i = 1; i <= n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                xspread = std::max(xspread, std::abs(simplex[i][j] - simplex[0][j]));
            }
        }
        double fspread = 0.0;
        for (std::size_t i = 1; i <= n; ++i) {
            fspread = std::max(fspread, std::abs(fvals[i] - fvals[0]));
        }
        if (xspread < xatol && fspread < fatol) {
            converged = true;
            break;
        }
        ++n_iter;

        // centroid of all vertices but the worst
        std::vector<double> centroid(n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) s += simplex[i][j];
            centroid[j] = s / static_cast<double>(n);
        }
        const std::vector<double>& worst = simplex[n];
        std::vector<double> xr(n);
        for (std::size_t j = 0; j < n; ++j) {
            xr[j] = centroid[j] + (centroid[j] - worst[j]);
        }
        const double fr = f(xr);

        if (fr < fvals[0]) {
            // try expansion
            std::vector<double> xe(n);
            for (std::size_t j = 0; j < n; ++j) {
                xe[j] = centroid[j] + 2.0 * (centroid[j] - worst[j]);
            }
            const double fe = f(xe);
            if (fe < fr) {
                simplex[n] = std::move(xe);
                fvals[n] = fe;
            } else {
                simplex[n] = std::move(xr);
                fvals[n] = fr;
            }
        } else if (fr < fvals[n - 1]) {
            simplex[n] = std::move(xr);
            fvals[n] = fr;
        } else {
            std::vector<double> xc(n);
            double fc;
            bool accept;
            if (fr < fvals[n]) {
                // outside contraction
                for (std::size_t j = 0; j < n; ++j) {
                    xc[j] = centroid[j] + 0.5 * (xr[j] - centroid[j]);
                }
                fc = f(xc);
                accept = fc <= fr;
            } else {
                // inside contraction
                for (std::size_t j = 0; j < n; ++j) {
                    xc[j] = centroid[j] - 0.5 * (centroid[j] - worst[j]);
                }
                fc = f(xc);
                accept = fc < fvals[n];
            }
            if (accept) {
                simplex[n] = std::move(xc);
                fvals[n] = fc;
            } else {
                // shrink toward the best vertex
                for (std::size_t i = 1; i <= n; ++i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        simplex[i][j] =
                            simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                    }
                    fvals[i] = f(simplex[i]);
                }
            }
        }
    }

    sort_simplex(simplex, fvals);
    return NMResult{simplex[0], fvals[0], n_iter, converged};
}

} // namespace lobmod
