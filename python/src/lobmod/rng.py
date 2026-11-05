"""Pinned deterministic random number generator (64-bit LCG).

Every stochastic component of ``lobmod`` (zero-intelligence order flow,
Ogata thinning) draws from this generator so that all four language
implementations reproduce bit-identical streams.  The generator is the
Knuth/MMIX linear congruential generator

    state_{n+1} = (A * state_n + C) mod 2^64,
    A = 6364136223846793005,  C = 1442695040888963407.

Uniforms take the top 53 bits: ``u = (state >> 11) * 2^-53 in [0, 1)``.
No other RNG (``random``, ``numpy.random``) may be used anywhere in the
package.
"""

from __future__ import annotations

import math

from ._validate import as_int, as_positive_int

_A: int = 6364136223846793005
_C: int = 1442695040888963407
_MASK: int = (1 << 64) - 1
_INV_2_53: float = 1.0 / (1 << 53)
#: pinned seed domain: [0, MAX_SEED] (representable as a signed 64-bit
#: integer in every port; never masked or wrapped)
MAX_SEED: int = (1 << 63) - 1
_TWO_63: float = 9223372036854775808.0


class Lcg64:
    """Pinned 64-bit LCG; the single source of randomness in ``lobmod``.

    Args:
        seed: integer seed in ``[0, MAX_SEED]`` (``MAX_SEED = 2^63 - 1``);
            the initial state is the seed itself.

    Raises:
        ValueError: if ``seed`` is not an integer, negative, or above
            ``MAX_SEED`` (never silently masked).
    """

    def __init__(self, seed: int) -> None:
        seed = as_int("seed", seed)
        if seed < 0 or seed > MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
        self._state: int = seed

    def next_u64(self) -> int:
        """Advance the state once and return it as an unsigned 64-bit integer."""
        self._state = (_A * self._state + _C) & _MASK
        return self._state

    def uniform(self) -> float:
        """Uniform double in ``[0, 1)`` from the top 53 bits of the next state."""
        return (self.next_u64() >> 11) * _INV_2_53

    def exponential(self, rate: float) -> float:
        """Exponential variate with the given rate: ``-log1p(-u) / rate``.

        Uses ``log1p(-u)`` (u in [0,1)) so the argument of the log never
        rounds to zero and the draw is always finite and >= 0.

        Raises:
            ValueError: if ``rate`` is not a positive finite number.
        """
        if not (math.isfinite(rate) and rate > 0.0):
            raise ValueError(f"rate must be positive and finite, got {rate}")
        return -math.log1p(-self.uniform()) / rate

    def geometric(self, p: float) -> int:
        """Geometric variate on {0, 1, 2, ...}: number of failures before success.

        Pinned inversion formula ``k = floor(log(1-u) / log(1-p))`` with
        ``u = uniform()``.

        Raises:
            ValueError: if ``p`` is not in ``(0, 1)``, or the result does not
                fit a signed 64-bit integer (quotient >= ``2^63``).
        """
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")
        u = self.uniform()
        x = math.log1p(-u) / math.log1p(-p)
        if x >= _TWO_63:  # also catches x == inf
            raise ValueError(f"geometric variate {x} for p={p} exceeds the 64-bit range")
        return int(math.floor(x))

    def randint(self, n: int) -> int:
        """Uniform integer in ``{0, ..., n-1}`` as ``floor(uniform() * n)``.

        The tiny modulo bias of this construction is irrelevant here and the
        formula is trivially portable, which is what matters for golden
        reproducibility.

        Raises:
            ValueError: if ``n`` is not a positive integer.
        """
        n = as_positive_int("n", n)
        k = int(self.uniform() * n)
        return n - 1 if k >= n else k
