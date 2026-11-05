"""Tests for the research model extensions: known answers, kept fast."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import lobmod as lm
from lobmod.gof import ks_test_exp
from lobmod.research.backtest import foresight_ceiling
from lobmod.research.baselines import (
    GbtConfig, GradientBoostedTrees, paired_log_loss_ci, seed_mean_ci, walk_forward_compare,
)
from lobmod.research.evaluation import log_loss
from lobmod.research.kernels import (
    POWER_LAW_RATES, excitation, fit_fixed_mixture, fit_power_law, fit_residuals, fit_sum_exp,
    power_law_weights, simulate_cluster, solve_poisson_linear, sumexp_loglik,
    sumexp_loglik_brute, sumexp_residuals,
)
from lobmod.research.labels import cost_aware_labels, horizon_sweep
from lobmod.research.seasonality import (
    LinearSplineBaseline, PiecewiseConstantBaseline, baseline_curve, fit_seasonal, select_baseline,
)


def flat(mu):
    return lambda x: 0.0 * x + mu


def two_scale(rng):
    return lambda k: np.where(rng.uniform(size=k) < 0.5, rng.exponential(0.02, k), rng.exponential(5.0, k))


@pytest.fixture(scope="module")
def path():
    return lm.simulate_uni(0.5, 1.0, 1.6, 300.0, 11)


# ---------------------------------------------------------------- likelihood

def test_excitation_matches_definition():
    t = np.array([0.1, 0.4, 0.45, 2.0])
    r = excitation(t, [1.0, 3.0])
    for i in range(4):
        for k, b in enumerate((1.0, 3.0)):
            assert r[i, k] == pytest.approx(np.exp(-b * (t[i] - t[:i])).sum())


def test_loglik_reduces_to_pinned_single_exponential(path):
    expected = lm.uni_loglik(path, 300.0, 0.5, 1.0, 1.6)
    assert sumexp_loglik(path, 300.0, 0.5, [1.0 / 1.6], [1.6]) == pytest.approx(expected, rel=1e-12)


def test_split_component_is_the_same_model(path):
    one = sumexp_loglik(path, 300.0, 0.5, [0.6], [1.6])
    assert sumexp_loglik(path, 300.0, 0.5, [0.2, 0.4], [1.6, 1.6]) == pytest.approx(one, rel=1e-12)


def test_recursion_matches_brute_force(path):
    args = (path, 300.0, 0.4, [0.3, 0.2], [20.0, 0.5])
    assert sumexp_loglik(*args) == pytest.approx(sumexp_loglik_brute(*args), rel=1e-11)


def test_residuals_match_pinned(path):
    pinned = lm.uni_residuals(path, 300.0, 0.5, 1.0, 1.6)
    np.testing.assert_allclose(sumexp_residuals(path, 300.0, 0.5, [1.0 / 1.6], [1.6]), pinned, rtol=1e-10)


def test_bad_mixture_rejected(path):
    with pytest.raises(ValueError):
        sumexp_loglik(path, 300.0, 0.5, [0.1, 0.2], [1.0])
    with pytest.raises(ValueError):
        fit_sum_exp(path, 300.0, K=0)


# -------------------------------------------------------------------- solver

def test_solver_poisson_closed_form():
    # one constant column: theta = n / C
    theta, _, ok = solve_poisson_linear(np.ones((50, 1)), np.array([200.0]))
    assert ok and theta[0] == pytest.approx(0.25, rel=1e-8)


def test_solver_kkt_and_bound():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(400), rng.uniform(size=400), np.zeros(400)])
    C = np.array([300.0, 150.0, 40.0])
    theta, _, ok = solve_poisson_linear(X, C)
    grad = X.T @ (1.0 / (X @ theta)) - C
    assert ok and theta[2] == 0.0  # a useless column goes to the bound
    assert np.all(grad < 1e-6)
    assert np.all(np.abs(grad[theta > 0]) < 1e-5)


# ------------------------------------------------------------------- fitting

def test_k1_matches_pinned_fit(path):
    g = lm.fit_uni(path, 300.0)
    f = fit_sum_exp(path, 300.0, K=1)
    assert f.loglik == pytest.approx(g.loglik, abs=1e-3)
    assert f.branching == pytest.approx(g.branching, abs=2e-3)
    assert f.coef[0] == pytest.approx(g.mu, abs=2e-3)


def test_two_timescale_recovered_and_preferred():
    rng = np.random.default_rng(5)
    x = simulate_cluster(flat(0.5), 0.5, 0.6, two_scale(rng), 6000.0, rng)
    one = fit_sum_exp(x, 6000.0, K=1)
    two = fit_sum_exp(x, 6000.0, K=2)
    assert one.branching < 0.4
    assert two.branching == pytest.approx(0.6, abs=0.06)
    assert two.betas[0] == pytest.approx(50.0, rel=0.25)
    assert two.betas[1] == pytest.approx(0.2, rel=0.35)
    assert two.aic < one.aic - 10.0 and two.bic < one.bic - 10.0
    assert ks_test_exp(fit_residuals(x, 6000.0, two))[1] > 0.01


def test_fixed_rates_at_truth_recover_weights():
    rng = np.random.default_rng(6)
    x = simulate_cluster(flat(0.5), 0.5, 0.6, two_scale(rng), 6000.0, rng)
    f = fit_fixed_mixture(x, 6000.0, [50.0, 0.2])
    assert f.converged
    np.testing.assert_allclose(f.weights, [0.3, 0.3], atol=0.05)


def test_power_law_weights_approximate_lomax():
    a, c = 0.6, 0.05
    w = power_law_weights(a, c)
    assert w.sum() == pytest.approx(1.0)
    lag = np.geomspace(0.01, 100.0, 30)
    approx = np.exp(-np.outer(lag, POWER_LAW_RATES)) @ w
    np.testing.assert_allclose(approx, (1.0 + lag / c) ** -a, rtol=0.0, atol=3e-3)


def test_power_law_family_recovers_branching():
    rng = np.random.default_rng(7)
    off = lambda k: 0.05 * ((1.0 - rng.uniform(size=k)) ** (-1.0 / 0.6) - 1.0)  # noqa: E731
    x = simulate_cluster(flat(0.5), 0.5, 0.6, off, 6000.0, rng)
    f = fit_power_law(x, 6000.0)
    assert f.branching == pytest.approx(0.6, abs=0.06)
    assert f.shape[0] == pytest.approx(0.6, abs=0.2)
    assert f.aic < fit_sum_exp(x, 6000.0, K=1).aic


# --------------------------------------------------------------- seasonality

@pytest.mark.parametrize("cls,J", [(PiecewiseConstantBaseline, 5), (LinearSplineBaseline, 5)])
def test_baseline_cumulative_is_integral_of_design(cls, J):
    b = cls(10.0, J)
    t = np.linspace(0.0, 10.0, 200001)
    d = b.design(t)
    numeric = np.vstack([np.zeros(d.shape[1]), np.cumsum((d[1:] + d[:-1]) / 2 * np.diff(t)[:, None], axis=0)])
    picks = [0, 33333, 100000, 171717, 200000]
    np.testing.assert_allclose(b.cumulative(t[picks]), numeric[picks], atol=2e-4)


def test_spline_partition_of_unity_and_exact_line():
    b = LinearSplineBaseline(10.0, 4)
    t = np.linspace(0.0, 10.0, 41)
    np.testing.assert_allclose(b.design(t).sum(axis=1), 1.0)
    coef = 1.0 + 0.5 * np.arange(5) * 2.5
    np.testing.assert_allclose(baseline_curve(b, coef, t), 1.0 + 0.5 * t)


def test_seasonal_loglik_reduces_to_flat(path):
    b = PiecewiseConstantBaseline(300.0, 6)
    flat_ll = sumexp_loglik(path, 300.0, 0.5, [0.6], [1.6])
    assert sumexp_loglik(path, 300.0, np.full(6, 0.5), [0.6], [1.6], b) == pytest.approx(flat_ll, rel=1e-12)


def test_u_shape_corrected_by_joint_fit():
    rng = np.random.default_rng(8)
    T = 1000.0
    u = lambda x: 1.0 + 0.9 * np.cos(2.0 * np.pi * x / T)  # noqa: E731
    x = simulate_cluster(u, 1.9, 0.5, lambda k: rng.exponential(0.6, k), T, rng)
    assert fit_sum_exp(x, T, K=1).branching > 0.7
    base = LinearSplineBaseline(T, 8)
    f = fit_seasonal(x, T, base)
    assert f.branching == pytest.approx(0.5, abs=0.1)
    curve = baseline_curve(base, f.coef, np.array([0.0, 500.0]))
    assert curve[0] > 3.0 * curve[1]
    best, _, rows = select_baseline(x, T)
    assert best.bic < rows[0]["bic"] and best.branching == pytest.approx(0.5, abs=0.1)


def test_bic_keeps_flat_on_flat_data(path):
    best, chosen, _ = select_baseline(path, 300.0, kind="piecewise", grid=(2, 4, 8))
    assert chosen.n_coef == 1


# -------------------------------------------------------------------- labels

def book(bid, ask):
    return SimpleNamespace(best_bid=np.array(bid, float), best_ask=np.array(ask, float))


def test_labels_hand_built():
    s = book([100, 100, 103, 96, 100], [101, 101, 104, 97, 101])
    lab = cost_aware_labels(s, [0, 1, 2, 3], 1)
    # 0: 100-101<0, 100-101<0 flat; 1: 103-101=2 up; 2: 103-97=6... short; 3: 100-97=3 up
    np.testing.assert_array_equal(lab.y, [1, 2, 0, 2])
    assert lab.long_pnl[1] == 2.0 and lab.short_pnl[2] == 6.0


def test_fee_and_margin_remove_a_marginal_trade():
    s = book([100, 102], [101, 103])
    assert cost_aware_labels(s, [0], 1).y[0] == 2  # pays 1 tick
    assert cost_aware_labels(s, [0], 1, fee_per_side=0.5).y[0] == 1  # pays exactly 0
    assert cost_aware_labels(s, [0], 1, margin=1.0).y[0] == 1


def test_delay_moves_entry():
    s = book([100, 100, 105], [101, 104, 106])
    lab = cost_aware_labels(s, [0], 1, delay=1)
    assert lab.long_pnl[0] == 1.0


def test_invalid_rows_marked():
    s = book([100, np.nan, 100], [101, 101, 101])
    lab = cost_aware_labels(s, [0, 1, 2], 1)
    np.testing.assert_array_equal(lab.valid, [False, False, False])
    assert np.all(lab.y == 1)


def test_sweep_matches_foresight_ceiling():
    rng = np.random.default_rng(1)
    mid = np.cumsum(rng.integers(-1, 2, 500)) + 1000.0
    s = book(mid - 1, mid + 1)
    row = horizon_sweep(s, [7])[0]
    ceiling = foresight_ceiling(s, 7)
    assert row["share_up"] + row["share_down"] == pytest.approx(ceiling["share_profitable"])
    assert row["mean_best"] == pytest.approx(ceiling["mean_best"])


# ---------------------------------------------------------------------- GBT

def test_gbt_learns_xor_where_logistic_cannot():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    g = GradientBoostedTrees(2, GbtConfig(n_rounds=30)).fit(X[:1000], y[:1000])
    logistic = lm.MLP(2, (), 2)
    logistic.fit(X[:1000], y[:1000], 0.1, 300)
    assert np.mean(g.predict(X[1000:]) == y[1000:]) > 0.95
    assert np.mean(logistic.predict(X[1000:]) == y[1000:]) < 0.75  # a line gets at most 3 of 4 quadrants


def test_gbt_pure_noise_stays_near_prior():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 3))
    y = rng.integers(0, 3, 3000)
    g = GradientBoostedTrees(3, GbtConfig(n_rounds=10)).fit(X[:2000], y[:2000])
    assert log_loss(y[2000:], g.predict_proba(X[2000:])) == pytest.approx(math.log(3.0), abs=0.03)


def test_gbt_one_round_stump_known_leaf():
    X = np.array([[0.0]] * 50 + [[1.0]] * 50)
    y = np.array([0] * 50 + [1] * 50)
    c = GbtConfig(n_rounds=1, depth=1, learning_rate=1.0, min_leaf=1, l2=0.0, subsample=1.0)
    g = GradientBoostedTrees(2, c).fit(X, y)
    # p = 1/2, h = 1/4: leaf is -(sum g)/(sum h) = +-2, softmax over +-2 margins
    expected = 1.0 / (1.0 + math.exp(-4.0))
    np.testing.assert_allclose(g.predict_proba(X)[[0, 99], [0, 1]], expected)


def test_gbt_is_seeded_and_validates():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(300, 2)), rng.integers(0, 3, 300)
    a = GradientBoostedTrees(3, GbtConfig(n_rounds=3), seed=4).fit(X, y).predict_proba(X)
    b = GradientBoostedTrees(3, GbtConfig(n_rounds=3), seed=4).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        GradientBoostedTrees(2).fit(X, y)


def test_walk_forward_compare_on_bundled_stream(states, cfg):
    from dataclasses import replace
    small = replace(cfg, epochs=20)
    r = walk_forward_compare(states, small, n_folds=2, embargo=50, gbt=GbtConfig(n_rounds=3))
    assert set(r.metrics) == {"gbt", "mlp", "logistic", "majority"}
    assert np.all(np.diff(r.event_index) > 0)
    assert r.metrics["majority"]["mcc"] == 0.0
    ci = paired_log_loss_ci(r.y, r.proba["gbt"], r.proba["gbt"], 50, n_boot=20)
    assert ci.point == 0.0 and ci.low == 0.0 and ci.high == 0.0


def test_seed_mean_ci():
    ci = seed_mean_ci([1.0, 2.0, 3.0, 4.0], n_boot=200)
    assert ci.point == 2.5 and 1.0 <= ci.low <= 2.5 <= ci.high <= 4.0
