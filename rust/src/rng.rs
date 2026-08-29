//! Pinned deterministic random number generator (64-bit LCG).
//!
//! Every stochastic component of `lobmod` (zero-intelligence order flow, Ogata
//! thinning) draws from this generator so that all four language
//! implementations reproduce bit-identical streams.  The generator is the
//! Knuth/MMIX linear congruential generator
//!
//! ```text
//! state_{n+1} = (A * state_n + C) mod 2^64,
//! A = 6364136223846793005,  C = 1442695040888963407.
//! ```
//!
//! Uniforms take the top 53 bits: `u = (state >> 11) * 2^-53 in [0, 1)`.
//! No other RNG (the `rand` crate included) may be used anywhere in the crate.

use crate::error::{LobError, Result};

const A: u64 = 6364136223846793005;
const C: u64 = 1442695040888963407;
const INV_2_53: f64 = 1.0 / (1u64 << 53) as f64;

/// Pinned 64-bit LCG; the single source of randomness in `lobmod`.
///
/// The contract requires a non-negative seed; the `u64` type makes a negative
/// seed unrepresentable, so construction is infallible.
#[derive(Debug, Clone)]
pub struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    /// Create a generator with initial state `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state once and return it as an unsigned 64-bit integer.
    pub fn next_u64(&mut self) -> u64 {
        self.state = A.wrapping_mul(self.state).wrapping_add(C);
        self.state
    }

    /// Uniform double in `[0, 1)` from the top 53 bits of the next state.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * INV_2_53
    }

    /// Exponential variate with the given rate: `-log1p(-u) / rate`.
    ///
    /// Uses `log1p(-u)` (u in [0,1)) so the argument of the log never rounds
    /// to zero and the draw is always finite and >= 0.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] if `rate` is not a positive finite number.
    pub fn exponential(&mut self, rate: f64) -> Result<f64> {
        if !(rate.is_finite() && rate > 0.0) {
            return Err(LobError::InvalidInput(format!(
                "rate must be positive and finite, got {rate}"
            )));
        }
        Ok(-(-self.uniform()).ln_1p() / rate)
    }

    /// Geometric variate on {0, 1, 2, ...}: number of failures before success.
    ///
    /// Pinned inversion formula `k = floor(log1p(-u) / log1p(-p))`.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] if `p` is not in `(0, 1)`.
    pub fn geometric(&mut self, p: f64) -> Result<i64> {
        if !(p > 0.0 && p < 1.0) {
            return Err(LobError::InvalidInput(format!(
                "p must be in (0, 1), got {p}"
            )));
        }
        let u = self.uniform();
        Ok(((-u).ln_1p() / (-p).ln_1p()).floor() as i64)
    }

    /// Uniform integer in `{0, ..., n-1}` as `min(floor(uniform() * n), n-1)`.
    ///
    /// The tiny modulo bias of this construction is irrelevant here and the
    /// formula is trivially portable, which is what matters for golden
    /// reproducibility.
    ///
    /// # Errors
    /// [`LobError::InvalidInput`] if `n == 0`.
    pub fn randint(&mut self, n: usize) -> Result<usize> {
        if n == 0 {
            return Err(LobError::InvalidInput(
                "n must be a positive integer, got 0".to_string(),
            ));
        }
        let k = (self.uniform() * n as f64) as usize;
        Ok(if k >= n { n - 1 } else { k })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_sequence_is_pinned() {
        // First states from seed 1 of the Knuth/MMIX LCG.
        let mut rng = Lcg64::new(1);
        let s1 = rng.next_u64();
        assert_eq!(s1, 6364136223846793005u64.wrapping_add(1442695040888963407));
        let mut rng2 = Lcg64::new(1);
        assert_eq!(rng2.next_u64(), s1); // deterministic per seed
    }

    #[test]
    fn uniform_in_unit_interval() {
        let mut rng = Lcg64::new(42);
        for _ in 0..1000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn variate_validation() {
        let mut rng = Lcg64::new(7);
        assert!(rng.exponential(0.0).is_err());
        assert!(rng.exponential(f64::NAN).is_err());
        assert!(rng.geometric(0.0).is_err());
        assert!(rng.geometric(1.0).is_err());
        assert!(rng.randint(0).is_err());
        let k = rng.randint(5).unwrap();
        assert!(k < 5);
        assert!(rng.exponential(2.0).unwrap() >= 0.0);
        assert!(rng.geometric(0.35).unwrap() >= 0);
    }
}
