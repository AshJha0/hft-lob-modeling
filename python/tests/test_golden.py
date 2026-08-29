"""Golden-value suite: recompute every case and compare within tolerance.

The same cases (data/golden/golden.json) are asserted by the C++, Rust and
Java test suites, which is what pins cross-language agreement.
"""

import numpy as np
import pytest

import lobmod as lm


def check(case, computed):
    for key, expected in case["expect"].items():
        assert computed[key] == pytest.approx(expected, abs=case["tol"]), (
            case["name"], key, computed[key], expected)


@pytest.mark.parametrize("seq", [1000, 3000, 5000])
def test_book_snapshots(golden, states, seq):
    case = golden[f"book_snapshot_event_{seq}"]
    i = seq - 1
    check(case, {
        "best_bid": states.best_bid[i],
        "best_ask": states.best_ask[i],
        "bid_size": states.bid_size[i],
        "ask_size": states.ask_size[i],
        "bid_depth_k": states.bid_depth_k[i],
        "ask_depth_k": states.ask_depth_k[i],
        "mid": states.mid[i],
    })


def test_microprice(golden, states):
    case = golden["microprice_event_2500"]
    check(case, {"microprice": states.microprice[int(case["inputs"]["event"]) - 1]})


def test_ofi(golden, states):
    case = golden["ofi_event_2200_w100"]
    ofi = lm.ofi_at(states, int(case["inputs"]["event"]) - 1, int(case["inputs"]["window"]))
    check(case, {"ofi": ofi})


def test_hawkes_uni_loglik_pinned(golden, uni_times):
    case = golden["hawkes_uni_loglik_pinned"]
    p = case["inputs"]
    ll = lm.uni_loglik(uni_times, p["T"], p["mu"], p["alpha"], p["beta"])
    check(case, {"loglik": ll})
    # and the recursion agrees with brute force on the full bundled sample
    assert ll == pytest.approx(
        lm.uni_loglik_brute(uni_times, p["T"], p["mu"], p["alpha"], p["beta"]), abs=1e-10)


def test_hawkes_uni_fit(golden, uni_fit):
    check(golden["hawkes_uni_fit"],
          {"mu": uni_fit.mu, "alpha": uni_fit.alpha, "beta": uni_fit.beta})
    check(golden["hawkes_uni_branching"], {"branching": uni_fit.branching})


def test_hawkes_uni_ks(golden, uni_times, uni_fit, dgp):
    case = golden["hawkes_uni_ks_fitted"]
    tau = lm.uni_residuals(uni_times, dgp["hawkes_uni"]["T"],
                           uni_fit.mu, uni_fit.alpha, uni_fit.beta)
    d, p = lm.ks_test_exp(tau)
    check(case, {"ks_stat": d, "p_value": p})


def test_hawkes_biv_loglik_pinned(golden, biv_data):
    case = golden["hawkes_biv_loglik_pinned"]
    p = case["inputs"]
    t, k = biv_data
    mu = [p["mu0"], p["mu1"]]
    alpha = [[p["a00"], p["a01"]], [p["a10"], p["a11"]]]
    check(case, {"loglik": lm.biv_loglik(t, k, p["T"], mu, alpha, p["beta"])})


def test_hawkes_biv_fit(golden, biv_fit):
    check(golden["hawkes_biv_fit_alpha"], {
        "a00": biv_fit.alpha[0, 0], "a01": biv_fit.alpha[0, 1],
        "a10": biv_fit.alpha[1, 0], "a11": biv_fit.alpha[1, 1],
    })
    check(golden["hawkes_biv_spectral_radius"],
          {"spectral_radius": biv_fit.spectral_radius})


def test_nn_losses(golden, pipeline):
    check(golden["nn_loss_epoch1"], {"loss": pipeline["history"][0]})
    check(golden["nn_loss_final"], {"loss": pipeline["history"][-1]})


def test_holdout_accuracies(golden, pipeline):
    y_test = pipeline["y_test"]
    pred_nn = pipeline["net"].predict(pipeline["x_test"])
    check(golden["nn_holdout_accuracy"], {"accuracy": lm.accuracy(y_test, pred_nn)})
    pred_lg = pipeline["logistic"].predict(pipeline["x_test"])
    check(golden["logistic_accuracy"], {"accuracy": lm.accuracy(y_test, pred_lg)})
    maj = lm.majority_class(pipeline["y_train"])
    check(golden["majority_accuracy"],
          {"accuracy": lm.accuracy(y_test, np.full(y_test.size, maj))})


def test_signal_pnl(golden, pipeline, feature_set):
    te = pipeline["test_idx"]
    pred = pipeline["net"].predict(pipeline["x_test"])
    gross, net, n = lm.signal_pnl(pred, feature_set.fwd_ret[te], feature_set.spread[te])
    check(golden["signal_pnl_after_costs"], {"gross": gross, "net": net, "n_trades": n})
    assert net < 0.0  # half-spread costs kill the naive signal


def test_all_golden_cases_covered(golden):
    assert len(golden) == 18
