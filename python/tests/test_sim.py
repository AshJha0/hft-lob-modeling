"""Execution simulator: mechanics, accounting identities, acceptance checks."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lobmod.sim import (
    AgentSpec, AvellanedaStoikov, Fees, FlowParams, NaiveTouch, SignalTaker, Simulator, assess_edge,
    attribute, fill_prob_by_queue, legacy_maker, realised_spread, summarise, variance_ratio,
)

FLOW = FlowParams()
SEEDS = range(1, 7)


def run(strategy=None, lat=0.01, T=120.0, seed=1, flow=FLOW, **kw):
    spec = AgentSpec("a", strategy or NaiveTouch(), lat, lat, **kw)
    return Simulator(flow, [spec], T, seed).run()


def paired_ci(d):
    d = np.asarray(d, float)
    half = 2.57 * d.std(ddof=1) / np.sqrt(len(d))  # t, 5 df
    return d.mean() - half, d.mean() + half


def test_seeded_runs_repeat():
    a, b = run(seed=3), run(seed=3)
    assert summarise(a, "a") == summarise(b, "a")
    assert np.array_equal(a.mid, b.mid)
    assert summarise(run(seed=4), "a") != summarise(a, "a")


def test_exogenous_flow_is_shared_across_latencies():
    assert run(lat=0.001).final_fair == run(lat=0.2).final_fair


def test_shadow_levels_match_book():
    s = Simulator(FLOW, [AgentSpec("a", NaiveTouch())], 60.0, 2)
    s.run()
    for side in "BS":
        assert sum(s.levels[side].values()) == s.book.total_size(side)
    assert s.bid == s.book.best_bid() and s.ask == s.book.best_ask()


@pytest.mark.parametrize("strategy", [NaiveTouch(), AvellanedaStoikov(), SignalTaker()])
def test_attribution_sums_to_booked_cash(strategy):
    res = run(strategy, flow=FLOW, fees=Fees(0.1, 0.2))
    at = attribute(res, "a")
    assert at.lots > 0
    assert at.total == pytest.approx(at.booked, abs=1e-6)


def test_fees_and_rebates_by_liquidity_side():
    base = attribute(run(), "a")
    assert attribute(run(fees=Fees(maker=0.2)), "a").total == pytest.approx(base.total - 0.2 * base.lots)
    assert attribute(run(fees=Fees(maker=-0.2)), "a").total == pytest.approx(base.total + 0.2 * base.lots)
    taker = attribute(run(SignalTaker(), fees=Fees(maker=9.0, taker=0.3)), "a")
    assert taker.fees == pytest.approx(-0.3 * taker.lots)


def test_queue_priority_is_the_books_fifo():
    s = Simulator(FLOW, [AgentSpec("a", NaiveTouch())], 1.0, 1)
    ag = s.agents[0]
    s._bg_post("B", 100, 3)
    s._bg_post("S", 105, 50)
    s._touch(0.0)
    s._on_new(ag, 10 ** 12, "B", 100, 0.0)
    assert ag.res.queue_log == [[3, 0]]
    s._market("S", 3, 0.0)
    assert ag.res.inventory == 0 and s.book.has_order(10 ** 12)
    s._market("S", 1, 0.0)
    assert ag.res.inventory == 1 and not s.book.has_order(10 ** 12)
    assert ag.res.queue_log == [[3, 1]]


def test_cancel_that_arrives_after_the_fill_is_late():
    s = Simulator(FLOW, [AgentSpec("a", NaiveTouch())], 1.0, 1)
    ag = s.agents[0]
    s._bg_post("S", 105, 50)
    s._touch(0.0)
    s._on_new(ag, 10 ** 12, "B", 100, 0.0)
    s._market("S", 1, 0.0)
    s._on_cancel(ag, 10 ** 12, 0.001)
    assert ag.res.late_cancels == 1 and ag.res.inventory == 1


def test_late_cancels_grow_with_latency():
    assert run(lat=0.1).agents["a"].late_cancels > run(lat=0.001).agents["a"].late_cancels


def test_own_order_does_not_chase_its_echo():
    res = run(T=60.0)
    assert res.agents["a"].orders_rested / 60.0 < 10.0


def test_post_only_rejects_a_crossing_order():
    s = Simulator(FLOW, [AgentSpec("a", NaiveTouch())], 1.0, 1)
    ag = s.agents[0]
    s._bg_post("S", 105, 5)
    s._touch(0.0)
    s._on_new(ag, 10 ** 12, "B", 105, 0.0)
    assert ag.res.rejects == 1 and s.book.total_size("B") == 0


def test_as_quotes_skew_against_inventory():
    v = dict(bid=None, ask=None, bid_size=0, ask_size=0, signal=1000.0)
    st = AvellanedaStoikov(cap=50)
    flat = st.decide(SimpleNamespace(inventory=0, **v))
    long_ = st.decide(SimpleNamespace(inventory=10, **v))
    assert long_.bid < flat.bid and long_.ask < flat.ask
    assert st.decide(SimpleNamespace(inventory=50, **v)).bid is None


def test_reports_have_expected_shape():
    res = run()
    rs = realised_spread(res, "a", (0.1, 5.0))
    assert all(np.isfinite(x) for x in rs.values())
    q = fill_prob_by_queue(res, "a")
    assert sum(n for n, _ in q.values()) == res.agents["a"].orders_rested
    assert all(f <= n for n, f in q.values())


def test_fast_rival_takes_fills():
    def lots(fast):
        specs = [AgentSpec("a", NaiveTouch(), 0.05, 0.05)]
        if fast:
            specs.append(AgentSpec("f", NaiveTouch(), 0.0002, 0.0002, size=3))
        return attribute(Simulator(FLOW, specs, 200.0, 5).run(), "a").total
    assert lots(True) < lots(False)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Simulator(FLOW, [], 0.0, 1)
    with pytest.raises(ValueError):
        Simulator(FLOW, [AgentSpec("a", NaiveTouch(), -1.0)], 1.0, 1)
    with pytest.raises(ValueError):
        Simulator(FLOW, [AgentSpec("a", NaiveTouch()), AgentSpec("a", NaiveTouch())], 1.0, 1)


# ---- acceptance -----------------------------------------------------------
def test_acceptance_latency_hurts_maker_with_informed_flow():
    d = [summarise(run(lat=0.0005, T=200.0, seed=s), "a")["total"]
         - summarise(run(lat=0.3, T=200.0, seed=s), "a")["total"] for s in SEEDS]
    assert paired_ci(d)[0] > 0.0


def test_acceptance_loose_cap_is_risk_not_free_money():
    tight = [summarise(run(NaiveTouch(cap=2), T=200.0, seed=s), "a") for s in SEEDS]
    loose = [summarise(run(NaiveTouch(cap=80), T=200.0, seed=s), "a") for s in SEEDS]
    assert np.std([r["total"] for r in loose], ddof=1) > np.std([r["total"] for r in tight], ddof=1)
    lo, hi = paired_ci([r["inventory"] for r in loose])
    assert lo <= 0.0 <= hi


def test_acceptance_zero_informed_zero_fees_is_not_an_edge():
    flow = replace(FLOW, p_informed=0.0)
    total = summarise(run(flow=flow, T=300.0), "a")["total"]
    assert assess_edge(total, total - 1.0, total + 1.0, 0.0, 0.0).startswith("not an edge")
    assert "not evidence" in assess_edge(5.0, 1.0, 9.0, 1.0, 0.1)


def test_legacy_split_is_exact_and_stream_mean_reverts():
    r = legacy_maker(7919, 0, 5, fee=0.1, n=2000)
    assert r["spread"] + r["position"] + r["fees"] == pytest.approx(r["pnl"], abs=1e-6)
    assert variance_ratio(7919, (1000,), n=8000)[1000] < 1.0


def test_v2_flow_defaults_are_off_and_features_work():
    import dataclasses

    import numpy as np

    from lobmod.sim.calibrate import simulate_stats
    from lobmod.sim.engine import FlowParams, Simulator

    base = FlowParams()
    a = Simulator(base, [], 120.0, 5).run()
    b = Simulator(dataclasses.replace(base, active_share=0.4, active_dwell=9.0), [], 120.0, 5).run()
    assert a.n_events == b.n_events and np.array_equal(a.mid, b.mid)   # inert while multiplier is 1

    plain = simulate_stats(base, 900.0, 3)
    sticky = simulate_stats(dataclasses.replace(base, sign_persist=0.7), 900.0, 3)
    assert sticky["sign_acf"][0] > plain["sign_acf"][0] + 0.2

    burst = dataclasses.replace(base, active_mult=4.0, active_dwell=30.0)
    rates = [simulate_stats(burst, 900.0, s)["order_rate"] for s in (1, 2, 3)]
    assert 0.6 < np.mean(rates) / plain["order_rate"] < 1.6            # mean level kept

    moved = Simulator(dataclasses.replace(base, impact=0.5, jump_rate=0.0), [], 300.0, 3).run()
    still = Simulator(dataclasses.replace(base, jump_rate=0.0), [], 300.0, 3).run()
    assert moved.final_fair != 10000.0 and still.final_fair == 10000.0
