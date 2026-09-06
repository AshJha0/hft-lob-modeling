#ifndef LOBMOD_RNG_HPP
#define LOBMOD_RNG_HPP

/// \file rng.hpp
/// \brief Pinned deterministic random number generator (64-bit LCG).
///
/// Every stochastic component of lobmod (zero-intelligence order flow, Ogata
/// thinning) draws from this generator so that all language implementations
/// reproduce bit-identical streams.  The generator is the Knuth/MMIX linear
/// congruential generator
///
///     state_{n+1} = (A * state_n + C) mod 2^64,
///     A = 6364136223846793005, C = 1442695040888963407.
///
/// Uniforms take the top 53 bits: u = (state >> 11) * 2^-53 in [0, 1).
/// No other RNG may be used anywhere in the library.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace lobmod {

/// Pinned 64-bit LCG; the single source of randomness in lobmod.
class Lcg64 {
public:
    /// \param seed seed in [0, 2^63 - 1] (the pinned domain; negative is an
    ///        error, nothing is masked); the initial state is the seed itself.
    explicit Lcg64(std::int64_t seed) {
        if (seed < 0) {
            throw std::invalid_argument("seed must be non-negative, got " +
                                        std::to_string(seed));
        }
        state_ = static_cast<std::uint64_t>(seed);
    }

    /// Advance the state once and return it as an unsigned 64-bit integer.
    std::uint64_t next_u64() {
        state_ = 6364136223846793005ULL * state_ + 1442695040888963407ULL;
        return state_;
    }

    /// Uniform double in [0, 1) from the top 53 bits of the next state.
    double uniform() {
        return static_cast<double>(next_u64() >> 11) *
               (1.0 / 9007199254740992.0); // 2^-53
    }

    /// Exponential variate with the given rate: -log1p(-u) / rate.
    ///
    /// log1p keeps the draw finite and >= 0 for every u in [0, 1).
    /// \throws std::invalid_argument if rate is not positive and finite.
    double exponential(double rate) {
        if (!(std::isfinite(rate) && rate > 0.0)) {
            throw std::invalid_argument("rate must be positive and finite, got " +
                                        std::to_string(rate));
        }
        return -std::log1p(-uniform()) / rate;
    }

    /// Geometric variate on {0, 1, 2, ...}: failures before the first success,
    /// by the pinned inversion formula floor(log1p(-u) / log1p(-p)).
    /// \throws std::invalid_argument if p is not in (0, 1).
    std::int64_t geometric(double p) {
        if (!(p > 0.0 && p < 1.0)) {
            throw std::invalid_argument("p must be in (0, 1), got " +
                                        std::to_string(p));
        }
        const double u = uniform();
        return static_cast<std::int64_t>(
            std::floor(std::log1p(-u) / std::log1p(-p)));
    }

    /// Uniform integer in {0, ..., n-1} as min(floor(uniform() * n), n-1).
    /// \throws std::invalid_argument if n <= 0.
    std::int64_t randint(std::int64_t n) {
        if (n <= 0) {
            throw std::invalid_argument("n must be a positive integer, got " +
                                        std::to_string(n));
        }
        const auto k = static_cast<std::int64_t>(uniform() * static_cast<double>(n));
        return k >= n ? n - 1 : k;
    }

private:
    std::uint64_t state_;
};

} // namespace lobmod

#endif // LOBMOD_RNG_HPP
