package com.quant.lobmod;

/**
 * Pinned deterministic random number generator (64-bit LCG).
 *
 * <p>Every stochastic component of {@code lobmod} (zero-intelligence order
 * flow, Ogata thinning) draws from this generator so that all four language
 * implementations reproduce bit-identical streams. The generator is the
 * Knuth/MMIX linear congruential generator</p>
 *
 * <pre>
 *   state_{n+1} = (A * state_n + C) mod 2^64,
 *   A = 6364136223846793005,  C = 1442695040888963407.
 * </pre>
 *
 * <p>Uniforms take the top 53 bits: {@code u = (state >>> 11) * 2^-53} in
 * [0, 1). No other RNG may be used anywhere in the package.</p>
 */
public final class Lcg64 {

    private static final long A = 6364136223846793005L;
    private static final long C = 1442695040888963407L;
    private static final double INV_2_53 = 0x1.0p-53;

    private long state;

    /**
     * Creates the generator with the given non-negative seed.
     *
     * @param seed non-negative seed; the initial state is {@code seed} taken
     *     as an unsigned 64-bit value
     * @throws IllegalArgumentException if {@code seed} is negative
     */
    public Lcg64(long seed) {
        if (seed < 0) {
            throw new IllegalArgumentException("seed must be non-negative, got " + seed);
        }
        this.state = seed;
    }

    /** Advances the state once and returns it as an unsigned 64-bit integer. */
    public long nextU64() {
        state = A * state + C; // Java long arithmetic wraps mod 2^64
        return state;
    }

    /** Uniform double in [0, 1) from the top 53 bits of the next state. */
    public double uniform() {
        return (nextU64() >>> 11) * INV_2_53;
    }

    /**
     * Exponential variate with the given rate: {@code -log1p(-u) / rate}.
     *
     * <p>Uses {@code log1p(-u)} (u in [0,1)) so the argument of the log never
     * rounds to zero and the draw is always finite and non-negative.</p>
     *
     * @throws IllegalArgumentException if {@code rate} is not positive finite
     */
    public double exponential(double rate) {
        if (!(Double.isFinite(rate) && rate > 0.0)) {
            throw new IllegalArgumentException("rate must be positive and finite, got " + rate);
        }
        return -Math.log1p(-uniform()) / rate;
    }

    /**
     * Geometric variate on {0, 1, 2, ...}: number of failures before success,
     * via the pinned inversion {@code floor(log1p(-u) / log1p(-p))}.
     *
     * @throws IllegalArgumentException if {@code p} is not in (0, 1)
     */
    public int geometric(double p) {
        if (!(p > 0.0 && p < 1.0)) {
            throw new IllegalArgumentException("p must be in (0, 1), got " + p);
        }
        double u = uniform();
        return (int) Math.floor(Math.log1p(-u) / Math.log1p(-p));
    }

    /**
     * Uniform integer in {0, ..., n-1} as {@code floor(uniform() * n)},
     * clamped to {@code n - 1} (the pinned, trivially portable construction).
     *
     * @throws IllegalArgumentException if {@code n} is not positive
     */
    public int randint(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be a positive integer, got " + n);
        }
        int k = (int) (uniform() * n);
        return k >= n ? n - 1 : k;
    }
}
