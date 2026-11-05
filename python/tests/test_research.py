"""Tests for lobmod.research: known answers, kept fast."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import lobmod as lm
from lobmod.replay import Event
from lobmod.research import (
    BacktestConfig,
    MakerConfig,
    block_bootstrap_ci,
    bootstrap_ks_uni,
    chi2_sf,
    evaluate_stream,
    foresight_ceiling,
    inference_biv,
    inference_uni,
    ljung_box,
    log_loss,
    lr_test_no_cross,
    lr_test_poisson,
    macro_f1,
    matthews_corrcoef,
    multi_seed_compare,
    numeric_hessian,
    per_class_precision,
    residual_acf,
    run_backtest,
    run_maker,
    walk_forward_splits,
)


# ---------------------------------------------------------------- inference

def test_chi2_sf_closed_forms():
    assert chi2_sf(3.0, 2) == pytest.approx(math.exp(-1.5), rel=1e-12)
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, rel=1e-9)
    assert chi2_sf(0.0, 3) == 1.0


def test_chi2_sf_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 7, 10, 30):
        for x in (0.1, 1.0, 5.0, 20.0, 60.0):
            assert chi2_sf(x, df) == pytest.approx(stats.chi2.sf(x, df), rel=1e-9, abs=1e-14)


def test_numeric_hessian_quadratic():
    a = np.array([[2.0, 0.5], [0.5, 3.0]])
    h = numeric_hessian(lambda v: 0.5 * v @ a @ v, [1.0, 2.0])
    assert np.allclose(h, a, atol=1e-6)


def test_poisson_rate_se_via_hessian():
    # ll(l) = N log l - l T has information N / l^2, so se = l / sqrt(N)
    n, T = 400, 100.0
    lam = n / T
    h = numeric_hessian(lambda v: -(n * math.log(v[0]) - v[0] * T), [lam])
    assert 1.0 / math.sqrt(h[0, 0]) == pytest.approx(lam / math.sqrt(n), rel=1e-5)


def test_inference_uni_covers_truth():
    t = lm.simulate_uni(0.5, 1.0, 1.6, 400.0, 11)
    r = inference_uni(t, 400.0)
    assert r.information_ok
    assert np.all(r.ci_low < [0.5, 1.0, 1.6]) and np.all(r.ci_high > [0.5, 1.0, 1.6])
    assert 0.0 < r.branching_se < 0.2


def test_inference_biv_and_no_cross_lr():
    mu, alpha, beta, T = [0.4, 0.4], [[0.5, 0.5], [0.5, 0.5]], 2.5, 100.0
    t, k = lm.simulate_biv(mu, alpha, beta, T, 5)
    fit = lm.fit_biv(t, k, T)
    r = inference_biv(t, k, T, fit)
    assert r.names[3] == "a01"
    # this path puts an estimate on the zero boundary: se must be NaN, not a crash
    assert r.information_ok == bool(np.all(np.isfinite(r.se)))
    lr = lr_test_no_cross(t, k, T, fit)
    assert lr.ll_full >= lr.ll_restricted
    assert lr.p_mixture <= lr.p_chi2


def test_lr_poisson_detects_clustering_not_poisson():
    T = 300.0
    assert lr_test_poisson(lm.simulate_uni(0.5, 1.0, 1.6, T, 3), T).p_mixture < 1e-6


def test_lr_poisson_bootstrap_corrects_davies_problem():
    # a true Poisson path: free beta latches on to one close pair, so the
    # mixture p is far too small, and the bootstrap p is the honest one
    T = 150.0
    x = np.sort(np.random.default_rng(4).uniform(0.0, T, 150))
    r = lr_test_poisson(x, T, n_boot=9, seed=2)
    assert r.p_bootstrap > 0.05
    assert r.p_bootstrap > r.p_mixture


def test_lr_poisson_bootstrap_p_on_grid():
    T = 80.0
    r = lr_test_poisson(lm.simulate_uni(0.5, 1.0, 1.6, T, 3), T, n_boot=5, seed=1)
    assert 1.0 / 6.0 - 1e-12 <= r.p_bootstrap <= 1.0


def test_bootstrap_gof_roughly_uniform_under_true_model():
    # the table p-value under estimation piles up near 1; the bootstrap one must not
    T, reps, B = 60.0, 8, 5
    p, naive = [], []
    for s in range(reps):
        g = bootstrap_ks_uni(lm.simulate_uni(0.5, 1.0, 1.6, T, 900 + s), T, B, seed=7000 + 50 * s)
        assert g.n_boot + g.n_failed == B
        p.append(g.p_value)
        naive.append(g.p_naive)
    assert 0.25 < np.mean(p) < 0.75
    assert np.mean(naive) > np.mean(p)


def test_bootstrap_gof_rejects_non_stationary_fit():
    fit = SimpleNamespace(mu=1.0, alpha=2.0, beta=1.0, branching=2.0)
    with pytest.raises(ValueError):
        bootstrap_ks_uni([1.0, 2.0], 3.0, 1, fit=fit)


def test_acf_and_ljung_box_known_values():
    x = [1.0, 2.0, 3.0, 4.0]
    # mean 2.5, deviations -1.5 -.5 .5 1.5: sum sq 5, lag-1 products .75 -.25 .75
    assert residual_acf(x, 1)[0] == pytest.approx(0.25)
    lb = ljung_box(x, 1)
    assert lb.statistic == pytest.approx(4 * 6 * 0.0625 / 3)
    rng = np.random.default_rng(0)
    assert ljung_box(rng.exponential(size=2000), 10).p_value > 0.01
    ar = np.zeros(2000)
    e = rng.normal(size=2000)
    for i in range(1, 2000):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    assert ljung_box(ar, 10).p_value < 1e-10


# --------------------------------------------------------------- evaluation

def test_block_bootstrap_iid_mean_se():
    x = np.random.default_rng(1).normal(size=500)
    ci = block_bootstrap_ci(lambda i: x[i].mean(), 500, 1, 800, seed=2)
    assert ci.point == pytest.approx(x.mean())
    assert (ci.high - ci.low) == pytest.approx(2 * 1.96 / math.sqrt(500), rel=0.15)


def test_block_bootstrap_widens_for_dependent_data():
    x = np.repeat(np.random.default_rng(1).normal(size=50), 20)  # runs of 20
    iid = block_bootstrap_ci(lambda i: x[i].mean(), 1000, 1, 500, seed=2)
    blk = block_bootstrap_ci(lambda i: x[i].mean(), 1000, 40, 500, seed=2)
    assert (blk.high - blk.low) > 2.0 * (iid.high - iid.low)


def test_walk_forward_no_overlap_and_purge():
    t = np.cumsum(np.random.default_rng(3).integers(1, 4, 1000))
    H, E = 10, 50
    splits = walk_forward_splits(t, 4, H, E, 0.5)
    assert len(splits) == 4
    seen = []
    for s in splits:
        assert not set(s.train) & set(s.test)
        assert t[s.train].max() + H + E < t[s.test].min()
        # tight: the next row would break the purge
        assert t[s.train.max() + 1] + H + E >= t[s.test].min()
        seen.extend(s.test.tolist())
    assert seen == list(range(500, 1000))
    assert len(splits[3].train) > len(splits[0].train)


def test_walk_forward_rejects_bad_input():
    with pytest.raises(ValueError):
        walk_forward_splits([3, 2, 1], 1, 0)
    with pytest.raises(ValueError):
        walk_forward_splits(np.arange(10), 1, 100)


def test_metrics_hand_values():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 0])
    # confusion [[1,1,0],[0,2,0],[1,0,1]]
    assert np.allclose(per_class_precision(y, p), [0.5, 2 / 3, 1.0])
    assert macro_f1(y, p) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    # c=4, s=6, p=(2,3,1), t=(2,2,2): (24-12)/sqrt((36-14)(36-12))
    assert matthews_corrcoef(y, p) == pytest.approx(12 / math.sqrt(22 * 24))
    assert matthews_corrcoef(y, y) == pytest.approx(1.0)
    assert matthews_corrcoef(y, np.ones(6, int)) == 0.0
    proba = np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    assert log_loss([0, 1], proba) == pytest.approx(-(math.log(0.5) + math.log(0.8)) / 2)


def test_metrics_match_binary_mcc_and_undefined_precision():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0, 0, 1, 1, 0])
    # binary phi: tp=1 tn=2 fp=1 fn=1
    phi = (1 * 2 - 1 * 1) / math.sqrt(2 * 2 * 3 * 3)
    assert matthews_corrcoef(y, p, 2) == pytest.approx(phi)
    assert math.isnan(per_class_precision(y, p, 3)[2])


def test_multi_seed_runner_small():
    cfg = lm.FeatureConfig(50, 20, 10, 0.25, 3, (4,), 0.1, 5, 0.7)
    res, skipped = multi_seed_compare([1, 2], 800, cfg, 50, [BacktestConfig(hold=10)])
    assert skipped == [] and set(res) == {1, 2}
    row = res[1]
    assert set(row) == {"mlp", "logistic", "majority"}
    assert row["majority"]["mcc"] == 0.0
    assert row["mlp"]["bt0_trades"] >= 0
    again = evaluate_stream(lm.generate_events(800, 1), cfg, 50, [BacktestConfig(hold=10)])
    assert again["mlp"]["accuracy"] == row["mlp"]["accuracy"]


# ----------------------------------------------------------------- backtest

def book(bids, asks, bsz=None, asz=None):
    n = len(bids)
    return SimpleNamespace(
        best_bid=np.array(bids, float), best_ask=np.array(asks, float),
        bid_size=np.array(bsz if bsz else [5] * n, float),
        ask_size=np.array(asz if asz else [5] * n, float),
    )


TINY = book([10, 10, 11, 12, 12, 11, 10, 10], [12, 12, 12, 13, 14, 12, 12, 12])


def test_backtest_long_and_short_hand_computed():
    r = run_backtest(TINY, [0, 4], [1, -1], BacktestConfig(hold=3, fee_per_side=0.5))
    a, b = r.trades
    # long: buy ask[0]=12, sell bid[3]=12; mid 11 -> 12.5
    assert (a.entry_price, a.exit_price, a.mid_pnl, a.pnl) == (12, 12, 1.5, -1.0)
    # short: sell bid[4]=12, buy ask[7]=12; mid 13 -> 11
    assert (b.entry_price, b.exit_price, b.mid_pnl, b.pnl) == (12, 12, 2.0, -1.0)
    assert r.stats["net"] == -2.0 and r.stats["fees"] == 2.0 and r.stats["gross"] == 0.0
    assert r.stats["spread_cost"] == 3.5
    assert r.stats["max_drawdown"] == 2.0


def test_backtest_delay_moves_fill():
    t = run_backtest(TINY, [0], [1], BacktestConfig(hold=2, delay=1)).trades[0]
    assert (t.entry_index, t.exit_index, t.entry_price, t.exit_price) == (1, 3, 12, 12)
    t = run_backtest(TINY, [1], [1], BacktestConfig(hold=2, delay=2)).trades[0]
    # enters at state 3 (ask 13), exits state 5 (bid 11)
    assert t.pnl == -2.0


def test_backtest_one_position_at_a_time():
    r = run_backtest(TINY, [0, 1, 2, 3, 4], [1, 1, -1, 1, 1], BacktestConfig(hold=3))
    assert [t.signal_index for t in r.trades] == [0, 4]
    assert r.stats["n_ignored_busy"] == 3


def test_backtest_entry_needs_touch_size_and_exit_waits():
    nan = float("nan")
    s = book([10, 10, nan, 11, 11], [12, 12, 13, 13, 13], asz=[1, 5, 5, 5, 5])
    cfg = BacktestConfig(hold=1, lots=2, inventory_cap=2)
    r = run_backtest(s, [0, 1], [1, 1], cfg)
    assert r.stats["n_skipped_quote"] == 1
    t = r.trades[0]
    # entry at state 1 for 2 lots; bid empty at state 2, so exit at state 3
    assert (t.entry_index, t.exit_index, t.exit_price, t.pnl) == (1, 3, 11, -2.0)


def test_backtest_skips_when_hold_runs_off_the_end():
    r = run_backtest(TINY, [6], [1], BacktestConfig(hold=3))
    assert r.trades == [] and r.stats["n_skipped_end"] == 1 and r.stats["net"] == 0.0


def test_backtest_config_and_input_validation():
    with pytest.raises(ValueError):
        BacktestConfig(hold=3, lots=2, inventory_cap=1)
    with pytest.raises(ValueError):
        run_backtest(TINY, [2, 1], [1, 1], BacktestConfig(hold=1))
    with pytest.raises(ValueError):
        run_backtest(TINY, [1], [2], BacktestConfig(hold=1))


def test_foresight_ceiling_hand():
    r = foresight_ceiling(book([10, 13], [11, 14]), 1)
    assert r == {"share_profitable": 1.0, "mean_best": 2.0}


def test_signal_pnl_still_pinned_diagnostic():
    g, n, k = lm.signal_pnl(np.array([2, 0, 1]), np.array([1.0, -1.0, 5.0]), np.ones(3))
    assert (g, n, k) == (2.0, 1.0, 2)


# -------------------------------------------------------------------- maker

def ev(i, etype, side, price, size, oid):
    return Event(i, etype, side, price, size, oid)


def test_maker_hand_round_trip():
    events = [
        ev(1, "L", "B", 99, 1, 1),
        ev(2, "L", "S", 101, 1, 2),    # quotes go in after this event
        ev(3, "M", "S", 0, 2, 0),      # eats background bid, then our 99 bid
        ev(4, "L", "B", 98, 1, 3),     # bid 98: maker joins there
        ev(5, "M", "B", 0, 2, 0),      # eats background ask, then our 101 ask
        ev(6, "L", "S", 103, 1, 4),
    ]
    r = run_maker(events, MakerConfig(warmup=0, fee_per_side=0.25, horizons=(1,)))
    assert [(f.event_index, f.sign, f.price) for f in r.fills] == [(2, 1, 99.0), (4, -1, 101.0)]
    assert r.fills[0].mid_before == 100.0
    # edges 100 - 99 = 1 and 101 - 99.5 = 1.5
    assert r.quoted_edge == pytest.approx(1.25)
    assert r.inventory == 0
    assert r.pnl == pytest.approx(2.0 - 0.5)
    assert r.pnl_unwound == r.pnl


def test_maker_inventory_cap_and_unwind():
    events = [ev(1, "L", "B", 99, 5, 1), ev(2, "L", "S", 101, 5, 2),
              ev(3, "M", "S", 0, 6, 0), ev(4, "L", "B", 98, 5, 3), ev(5, "L", "B", 97, 5, 4)]
    r = run_maker(events, MakerConfig(warmup=0, inventory_cap=1, horizons=()))
    assert r.inventory == 1 and len(r.fills) == 1
    # bought 99; final book 98 x 101 -> mid 99.5, touch 98
    assert r.pnl == pytest.approx(0.5) and r.pnl_unwound == pytest.approx(-1.0)


def test_maker_runs_on_zi():
    r = run_maker(lm.generate_events(3000, 7919), MakerConfig(inventory_cap=3))
    assert len(r.fills) > 20
    assert all(abs(f.inventory_after) <= 3 for f in r.fills)
    assert r.quoted_edge > 0.0
    assert set(r.realised_spread) == {1, 10, 50, 200}
