#ifndef LOBMOD_NELDERMEAD_HPP
#define LOBMOD_NELDERMEAD_HPP

/// \file neldermead.hpp
/// \brief Native Nelder-Mead simplex minimizer (pinned scipy-compatible configuration).
///
/// Deterministic derivative-free minimization of f: R^n -> double.  The
/// configuration is pinned so that every language reproduces the same
/// iterates:
///
/// * coefficients: reflection 1, expansion 2, contraction 0.5, shrink 0.5;
/// * initial simplex (scipy's rule): vertex 0 is x0; vertex i multiplies
///   coordinate i-1 by 1.05 if nonzero, else sets it to 0.00025;
/// * iteration (Lagarias et al. 1998): stable-sort vertices ascending by f
///   (ties keep prior order), reflect the worst through the centroid of the
///   rest, then expand / outside-contract / inside-contract / shrink;
/// * termination *before* an iteration: max abs coordinate spread to the
///   best vertex < xatol AND max abs f spread < fatol, or maxiter.
///
/// Objectives must be pure and never throw: out-of-domain points should
/// return a large penalty value.

#include <functional>
#include <vector>

namespace lobmod {

/// Nelder-Mead outcome.
struct NMResult {
    std::vector<double> x; ///< best vertex found
    double fval;           ///< objective value at x
    int n_iter;            ///< iterations performed
    bool converged;        ///< true if the tolerance test (not the cap) stopped it
};

/// Minimize f from x0 with the pinned Nelder-Mead scheme above.
/// \throws std::invalid_argument on empty or non-finite x0, non-positive
///         tolerances, or maxiter < 1.
NMResult nelder_mead(const std::function<double(const std::vector<double>&)>& f,
                     std::vector<double> x0, double xatol = 1e-8,
                     double fatol = 1e-10, int maxiter = 5000);

} // namespace lobmod

#endif // LOBMOD_NELDERMEAD_HPP
