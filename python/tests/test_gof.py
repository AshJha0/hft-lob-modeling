"""KS goodness-of-fit machinery: statistic, pinned p-value, tiny samples."""

import numpy as np
import pytest

import lobmod as lm


def test_ks_statistic_hand_computed():
    # single point x: D = max(1 - F(x), F(x)) with F = 1 - exp(-x)
    x = 0.3
    f = 1.0 - np.exp(-x)
    assert lm.ks_statistic_exp([x]) == pytest.approx(max(1.0 - f, f), abs=1e-15)


def test_ks_statistic_on_exact_exp_quantiles():
    """Property-style grid: plugging in exact Exp(1) quantiles at plotting
    positions (i-0.5)/n gives D = 0.5/n for every n."""
    for n in (4, 10, 25, 100):
        q = -np.log1p(-(np.arange(1, n + 1) - 0.5) / n)
        assert lm.ks_statistic_exp(q) == pytest.approx(0.5 / n, abs=1e-12)


def test_ks_pvalue_matches_scipy_asymptotic():
    from scipy.special import kolmogorov

    for n in (50, 200, 1000):
        for d in (0.02, 0.05, 0.1):
            lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
            assert lm.ks_pvalue(d, n) == pytest.approx(float(kolmogorov(lam)), abs=1e-10)


def test_ks_pvalue_bounds_and_monotonicity():
    ps = [lm.ks_pvalue(d, 100) for d in (0.01, 0.05, 0.1, 0.2, 0.5)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert all(a >= b for a, b in zip(ps, ps[1:]))  # larger D -> smaller p
    assert lm.ks_pvalue(0.0, 10) == 1.0


def test_ks_tiny_samples():
    # n = 1 and n = 3 must work (asymptotic p is indicative, not an error)
    d, p = lm.ks_test_exp([0.7])
    assert 0.0 <= d <= 1.0 and 0.0 <= p <= 1.0
    d, p = lm.ks_test_exp([0.1, 0.9, 2.5])
    assert 0.0 <= d <= 1.0 and 0.0 <= p <= 1.0


def test_ks_rejects_obviously_wrong_sample():
    _, p = lm.ks_test_exp(np.full(200, 3.0))  # constant sample is not Exp(1)
    assert p < 1e-6


def test_ks_validation():
    with pytest.raises(ValueError, match="non-empty"):
        lm.ks_statistic_exp([])
    with pytest.raises(ValueError, match="non-negative"):
        lm.ks_statistic_exp([0.5, -0.1])
    with pytest.raises(ValueError, match="finite"):
        lm.ks_statistic_exp([np.nan])
    with pytest.raises(ValueError, match="d must be"):
        lm.ks_pvalue(1.5, 10)
    with pytest.raises(ValueError, match="n must be"):
        lm.ks_pvalue(0.1, 0)


# ----------------------------------------------------------- review additions

def test_ks_pvalue_small_lambda_accuracy():
    """PT-10: the dual series makes tiny statistics give p = 1, not 0.87."""
    assert abs(lm.ks_pvalue(1e-4, 100) - 1.0) < 1e-9
    n = 100_000
    assert lm.ks_pvalue(0.5 / n, n) >= 0.999
    ds = [0.5 / n, 2.0 / n, 8.0 / n, 32.0 / n, 128.0 / n, 512.0 / n]
    ps = [lm.ks_pvalue(d, n) for d in ds]
    assert all(a >= b for a, b in zip(ps, ps[1:]))  # monotone across the branch switch
    from lobmod.gof import _kolmogorov_q
    # lambda = 0.01 was 0.867 with the 100-term alternating series alone
    assert _kolmogorov_q(0.01) == 1.0
    assert _kolmogorov_q(0.005) == 1.0
    # continuity at the pinned switch point, checked against scipy
    from scipy.special import kolmogorov
    for lam in (0.02, 0.05, 0.0999, 0.1, 0.1001, 0.3):
        assert _kolmogorov_q(lam) == pytest.approx(float(kolmogorov(lam)), abs=1e-14)
    with pytest.raises(ValueError, match="n must be"):
        lm.ks_pvalue(0.1, 2.5)
