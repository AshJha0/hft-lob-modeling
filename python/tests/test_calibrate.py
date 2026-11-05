"""Calibration on synthetic known-answer cases; no market data."""
import math
from dataclasses import replace

import pytest

from lobmod.sim import FlowParams
from lobmod.sim.calibrate import (
    REFINED, _apply, acf, build_stats, distance, evaluate, mean_stats, moment_fit, refine, simulate_stats,
    spread_distribution,
)

BASE = {
    "spread_dist": [0.6, 0.3, 0.1, 0.0, 0.0], "event_mix": [0.5, 0.4, 0.1], "event_rate": 40.0,
    "touch_median": 10.0, "touch_p90": 30.0, "sign_acf": [0.3, 0.1, 0.02], "ofi_r2": 0.3, "vol10": 2.0,
    "order_rate": 2.0,
}


def test_statistics_on_hand_cases():
    assert spread_distribution([1, 1, 2, 7]) == [0.5, 0.25, 0.0, 0.0, 0.25]
    assert acf([1, -1] * 50, (1,))[0] == pytest.approx(-1.0)
    assert math.isnan(acf([1, 1, 1], (1,))[0])
    s = build_stats([1, 2], [3, 5, 7], 5, 3, 2, 10.0, [1, -1], [1, 2, 3, 5], [2, 4, 6, 10])
    assert s["event_mix"] == [0.5, 0.3, 0.2] and s["event_rate"] == 1.0
    assert s["touch_median"] == 5.0 and s["ofi_r2"] == pytest.approx(1.0)


def test_identical_stats_pass_every_row_and_score_zero():
    rows = evaluate(BASE, BASE)
    assert all(r["passed"] for r in rows.values())
    assert distance(BASE, BASE) == 0.0


@pytest.mark.parametrize("key,value,row", [
    ("spread_dist", [0.3, 0.3, 0.4, 0.0, 0.0], "C1"),
    ("event_mix", [0.3, 0.6, 0.1], "C2a"),
    ("event_rate", 20.0, "C2b"),
    ("touch_median", 5.0, "C3"),
    ("touch_p90", 100.0, "C3"),
    ("sign_acf", [0.3, 0.1, 0.1], "C4"),
    ("ofi_r2", 0.6, "C5"),
    ("vol10", 4.0, "C6"),
])
def test_each_tolerance_fails_alone(key, value, row):
    rows = evaluate({**BASE, key: value}, BASE)
    assert [k for k, r in rows.items() if not r["passed"]] == [row]


def test_nan_sign_memory_never_passes():
    assert not evaluate({**BASE, "sign_acf": [math.nan, 0.1, 0.02]}, BASE)["C4"]["passed"]


def test_moment_fit_inverts_known_relations():
    f = moment_fit(BASE, branching=0.5, beta=4.0)
    assert sum(f.mu) / (1 - 0.5) == pytest.approx(2.0)
    assert (f.alpha[0][0] + f.alpha[0][1]) / f.beta == pytest.approx(0.5)
    assert f.alpha[0][0] / (f.alpha[0][0] + f.alpha[0][1]) == pytest.approx(0.65)
    sd2 = f.jump_sd ** 2
    assert 10 * f.jump_rate * sd2 == pytest.approx(BASE["vol10"] ** 2)
    assert f.limit_rate / 2 == pytest.approx(0.5 * 40 / 2)
    with pytest.raises(ValueError):
        moment_fit({**BASE, "order_rate": 0.0})


def _toy(target):
    """A cheap stand-in market whose statistics are smooth in the refined parameters."""
    def simulate(flow):
        return {**BASE,
                "event_rate": BASE["event_rate"] * flow.limit_rate / target.limit_rate,
                "touch_median": BASE["touch_median"] * flow.queue_scale / target.queue_scale,
                "touch_p90": BASE["touch_p90"] * flow.cancel_rate / target.cancel_rate,
                "vol10": BASE["vol10"] * math.sqrt(flow.jump_rate / target.jump_rate),
                "ofi_r2": BASE["ofi_r2"] * flow.p_geometric / target.p_geometric}
    return simulate


def test_refine_recovers_a_known_answer_on_a_stand_in_market():
    target = FlowParams()
    start = replace(target, limit_rate=target.limit_rate * 1.5, queue_scale=target.queue_scale / 1.4,
                    cancel_rate=target.cancel_rate * 1.3, jump_rate=target.jump_rate * 1.6, p_geometric=0.4)
    fit = refine(BASE, start, _toy(target), budget=600)
    assert fit.value < 1e-2 < fit.start_value
    for name in REFINED:
        assert getattr(fit.flow, name) == pytest.approx(getattr(target, name), rel=0.1)


def test_refine_respects_its_budget_and_keeps_the_best():
    calls = []
    def simulate(flow):
        calls.append(flow)
        return _toy(FlowParams())(flow)
    fit = refine(BASE, replace(FlowParams(), limit_rate=9.0), simulate, budget=9)
    assert len(calls) == fit.n_evals == 9
    assert fit.value <= fit.start_value


def test_apply_keeps_geometric_probability_valid():
    assert _apply(FlowParams(), [1, 1, 1, 2, 1]).p_geometric <= 0.95


def test_simulated_stats_match_what_was_put_in():
    flow = FlowParams()
    paths = [simulate_stats(flow, 200.0, s, warmup=20.0) for s in (1, 2)]
    s = mean_stats(paths)
    assert sum(s["event_mix"]) == pytest.approx(1.0)
    assert sum(s["spread_dist"]) == pytest.approx(1.0)
    mean_rate = 2 * 0.8 / (1 - 1.3 / 3.0)        # Hawkes mean, market orders only
    assert 0.7 * mean_rate < s["order_rate"] < 1.8 * mean_rate
    assert s["touch_median"] >= 1.0 and s["vol10"] > 0.0 and 0.0 <= s["ofi_r2"] <= 1.0
    assert s == mean_stats(paths)


def test_simulated_stats_are_seed_reproducible():
    a = simulate_stats(FlowParams(), 60.0, 7, warmup=10.0)
    b = simulate_stats(FlowParams(), 60.0, 7, warmup=10.0)
    assert a["event_rate"] == b["event_rate"] and a["spread_dist"] == b["spread_dist"]
    with pytest.raises(ValueError):
        simulate_stats(FlowParams(), 10.0, 7, warmup=10.0)
