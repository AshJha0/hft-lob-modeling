#!/usr/bin/env python3
"""Regenerate all bundled datasets and golden values (fixed seeds).

Outputs (all deterministic):
    lob_events.csv        5000-event zero-intelligence stream (seed 20240613)
    hawkes_uni.csv        univariate Hawkes event times (seed 101)
    hawkes_biv.csv        bivariate Hawkes times + types (seed 33)
    dgp_params.json       true DGP parameters and seeds
    features_config.json  pinned feature/label/NN hyperparameters
    golden/golden.json    18 cross-language golden cases

Before writing golden values the script validates
    * the recursive Hawkes log-likelihoods against brute-force O(N^2) sums
      (uni and biv, agreement < 1e-10), and
    * the MLP backprop gradient against central finite differences
      (max relative error < 1e-6),
aborting with an AssertionError if either check fails.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python", "src"))

import numpy as np  # noqa: E402

import lobmod as lm  # noqa: E402

# ------------------------------------------------------------ pinned settings
LOB_SEED = 20240613
LOB_N_EVENTS = 5000

UNI = {"mu": 0.5, "alpha": 1.0, "beta": 1.6, "T": 1000.0, "seed": 101}
BIV = {
    "mu": [0.3, 0.3],
    "alpha": [[0.6, 0.3], [0.25, 0.5]],
    "beta": 2.0,
    "T": 600.0,
    "seed": 33,
}
# parameter points at which golden log-likelihoods are evaluated
UNI_EVAL = {"mu": 0.4, "alpha": 0.6, "beta": 1.2}
BIV_EVAL = {
    "mu0": 0.25, "mu1": 0.25,
    "a00": 0.5, "a01": 0.2, "a10": 0.2, "a11": 0.5,
    "beta": 1.8,
}

CFG = lm.FeatureConfig(
    ofi_window=50, ret_window=20, horizon=10, epsilon=0.25, depth_k=3,
    hidden=(16, 8), learning_rate=0.1, epochs=800, train_frac=0.7,
)

SNAPSHOT_EVENTS = [1000, 3000, 5000]
MICROPRICE_EVENT = 2500
OFI_EVENT, OFI_WINDOW = 2200, 100

cases: List[Dict] = []


def add_case(name: str, inputs: Dict[str, float], expect: Dict[str, float], tol: float) -> None:
    for k, v in expect.items():
        assert isinstance(v, (int, float)) and math.isfinite(float(v)), (name, k, v)
    cases.append({"name": name, "inputs": inputs, "expect": expect, "tol": tol})


def main() -> None:
    # ------------------------------------------------------------- LOB stream
    events = lm.generate_events(LOB_N_EVENTS, LOB_SEED)
    lm.write_events_csv(os.path.join(HERE, "lob_events.csv"), events)
    events = lm.read_events_csv(os.path.join(HERE, "lob_events.csv"))
    states = lm.replay(events, depth_k=CFG.depth_k)

    for seq in SNAPSHOT_EVENTS:
        i = seq - 1
        add_case(
            f"book_snapshot_event_{seq}",
            {"event": seq, "depth_k": CFG.depth_k},
            {
                "best_bid": states.best_bid[i],
                "best_ask": states.best_ask[i],
                "bid_size": states.bid_size[i],
                "ask_size": states.ask_size[i],
                "bid_depth_k": states.bid_depth_k[i],
                "ask_depth_k": states.ask_depth_k[i],
                "mid": states.mid[i],
            },
            1e-12,
        )
    add_case(
        f"microprice_event_{MICROPRICE_EVENT}",
        {"event": MICROPRICE_EVENT},
        {"microprice": states.microprice[MICROPRICE_EVENT - 1]},
        1e-12,
    )
    add_case(
        f"ofi_event_{OFI_EVENT}_w{OFI_WINDOW}",
        {"event": OFI_EVENT, "window": OFI_WINDOW},
        {"ofi": lm.ofi_at(states, OFI_EVENT - 1, OFI_WINDOW)},
        1e-10,
    )

    # ------------------------------------------------------- univariate Hawkes
    t_uni = lm.simulate_uni(UNI["mu"], UNI["alpha"], UNI["beta"], UNI["T"], UNI["seed"])
    lm.write_times_csv(os.path.join(HERE, "hawkes_uni.csv"), t_uni)
    t_uni = lm.read_times_csv(os.path.join(HERE, "hawkes_uni.csv"))
    print(f"hawkes_uni: {t_uni.size} events on [0, {UNI['T']}]")

    ll_rec = lm.uni_loglik(t_uni, UNI["T"], **UNI_EVAL)
    ll_brute = lm.uni_loglik_brute(t_uni, UNI["T"], **UNI_EVAL)
    assert abs(ll_rec - ll_brute) < 1e-10, (ll_rec, ll_brute)
    add_case(
        "hawkes_uni_loglik_pinned",
        {**UNI_EVAL, "T": UNI["T"]},
        {"loglik": ll_rec},
        1e-8,
    )

    fit_u = lm.fit_uni(t_uni, UNI["T"])
    assert fit_u.converged
    add_case(
        "hawkes_uni_fit",
        {"T": UNI["T"]},
        {"mu": fit_u.mu, "alpha": fit_u.alpha, "beta": fit_u.beta},
        1e-4,
    )
    add_case("hawkes_uni_branching", {"T": UNI["T"]}, {"branching": fit_u.branching}, 1e-6)

    resid = lm.uni_residuals(t_uni, UNI["T"], fit_u.mu, fit_u.alpha, fit_u.beta)
    d, p = lm.ks_test_exp(resid)
    add_case("hawkes_uni_ks_fitted", {"n": int(resid.size)}, {"ks_stat": d, "p_value": p}, 1e-6)
    print(f"uni fit: mu={fit_u.mu:.4f} alpha={fit_u.alpha:.4f} beta={fit_u.beta:.4f} "
          f"branching={fit_u.branching:.4f} KS p={p:.4f}")

    # -------------------------------------------------------- bivariate Hawkes
    t_biv, k_biv = lm.simulate_biv(BIV["mu"], BIV["alpha"], BIV["beta"], BIV["T"], BIV["seed"])
    lm.write_marked_times_csv(os.path.join(HERE, "hawkes_biv.csv"), t_biv, k_biv)
    t_biv, k_biv = lm.read_marked_times_csv(os.path.join(HERE, "hawkes_biv.csv"))
    print(f"hawkes_biv: {t_biv.size} events ({int((k_biv == 0).sum())} type-0)")

    mu_e = [BIV_EVAL["mu0"], BIV_EVAL["mu1"]]
    a_e = [[BIV_EVAL["a00"], BIV_EVAL["a01"]], [BIV_EVAL["a10"], BIV_EVAL["a11"]]]
    ll_rec = lm.biv_loglik(t_biv, k_biv, BIV["T"], mu_e, a_e, BIV_EVAL["beta"])
    ll_brute = lm.biv_loglik_brute(t_biv, k_biv, BIV["T"], mu_e, a_e, BIV_EVAL["beta"])
    assert abs(ll_rec - ll_brute) < 1e-10, (ll_rec, ll_brute)
    add_case("hawkes_biv_loglik_pinned", {**BIV_EVAL, "T": BIV["T"]}, {"loglik": ll_rec}, 1e-8)

    fit_b = lm.fit_biv(t_biv, k_biv, BIV["T"])
    assert fit_b.converged
    add_case(
        "hawkes_biv_fit_alpha",
        {"T": BIV["T"]},
        {
            "a00": fit_b.alpha[0, 0], "a01": fit_b.alpha[0, 1],
            "a10": fit_b.alpha[1, 0], "a11": fit_b.alpha[1, 1],
        },
        1e-3,
    )
    add_case(
        "hawkes_biv_spectral_radius",
        {"T": BIV["T"]},
        {"spectral_radius": fit_b.spectral_radius},
        1e-6,
    )
    print(f"biv fit: mu={np.round(fit_b.mu, 4)} alpha={np.round(fit_b.alpha, 4).tolist()} "
          f"beta={fit_b.beta:.4f} radius={fit_b.spectral_radius:.4f}")

    # -------------------------------------------------- features / NN pipeline
    fs = lm.compute_features(states, CFG)
    train_idx, test_idx = lm.chronological_split(len(fs), CFG.train_frac)
    std = lm.Standardizer().fit(fs.X[train_idx])
    x_train, x_test = std.transform(fs.X[train_idx]), std.transform(fs.X[test_idx])
    y_train, y_test = fs.y[train_idx], fs.y[test_idx]
    print(f"features: {len(fs)} rows, classes {np.bincount(fs.y, minlength=3).tolist()}")

    # gradient check on a tiny net before trusting training
    tiny = lm.MLP(5, (4, 3), 3)
    xg, yg = x_train[:16], y_train[:16]
    _, gw, gb = tiny.loss_and_grads(xg, yg)
    h = 1e-6
    worst = 0.0
    for l in range(len(tiny.weights)):
        for idx in [(0, 0), (tiny.weights[l].shape[0] - 1, tiny.weights[l].shape[1] - 1)]:
            w0 = tiny.weights[l][idx]
            tiny.weights[l][idx] = w0 + h
            lp = tiny.loss(xg, yg)
            tiny.weights[l][idx] = w0 - h
            lms = tiny.loss(xg, yg)
            tiny.weights[l][idx] = w0
            fd = (lp - lms) / (2 * h)
            an = gw[l][idx]
            worst = max(worst, abs(fd - an) / max(1e-8, abs(fd)))
    assert worst < 1e-6, f"gradient check failed: {worst}"
    print(f"gradient check max rel err: {worst:.2e}")

    net = lm.MLP(5, CFG.hidden, 3)
    history = net.fit(x_train, y_train, CFG.learning_rate, CFG.epochs)
    add_case("nn_loss_epoch1", {"epochs": 1}, {"loss": history[0]}, 1e-8)
    add_case("nn_loss_final", {"epochs": CFG.epochs}, {"loss": history[-1]}, 1e-8)

    pred_nn = net.predict(x_test)
    acc_nn = lm.accuracy(y_test, pred_nn)
    add_case("nn_holdout_accuracy", {"n_test": int(test_idx.size)}, {"accuracy": acc_nn}, 1e-10)

    logistic = lm.MLP(5, (), 3)
    logistic.fit(x_train, y_train, CFG.learning_rate, CFG.epochs)
    acc_lg = lm.accuracy(y_test, logistic.predict(x_test))
    add_case("logistic_accuracy", {"n_test": int(test_idx.size)}, {"accuracy": acc_lg}, 1e-10)

    maj = lm.majority_class(y_train)
    acc_mj = lm.accuracy(y_test, np.full(test_idx.size, maj))
    add_case(
        "majority_accuracy",
        {"majority_class": maj, "n_test": int(test_idx.size)},
        {"accuracy": acc_mj},
        1e-10,
    )

    gross, net_pnl, n_trades = lm.signal_pnl(pred_nn, fs.fwd_ret[test_idx], fs.spread[test_idx])
    add_case(
        "signal_pnl_after_costs",
        {"n_test": int(test_idx.size)},
        {"gross": gross, "net": net_pnl, "n_trades": n_trades},
        1e-8,
    )
    print(f"acc nn={acc_nn:.4f} logistic={acc_lg:.4f} majority={acc_mj:.4f}; "
          f"pnl gross={gross:.2f} net={net_pnl:.2f} trades={n_trades}")

    # -------------------------------------------------------------- json files
    with open(os.path.join(HERE, "dgp_params.json"), "w") as fh:
        json.dump(
            {
                "lob": {"seed": LOB_SEED, "n_events": LOB_N_EVENTS},
                "hawkes_uni": UNI,
                "hawkes_biv": BIV,
                "pinned_eval_uni": UNI_EVAL,
                "pinned_eval_biv": BIV_EVAL,
            },
            fh, indent=2,
        )
        fh.write("\n")
    with open(os.path.join(HERE, "features_config.json"), "w") as fh:
        json.dump(
            {
                "ofi_window": CFG.ofi_window, "ret_window": CFG.ret_window,
                "horizon": CFG.horizon, "epsilon": CFG.epsilon,
                "depth_k": CFG.depth_k, "hidden": list(CFG.hidden),
                "learning_rate": CFG.learning_rate, "epochs": CFG.epochs,
                "train_frac": CFG.train_frac,
            },
            fh, indent=2,
        )
        fh.write("\n")
    os.makedirs(os.path.join(HERE, "golden"), exist_ok=True)
    with open(os.path.join(HERE, "golden", "golden.json"), "w") as fh:
        json.dump({"cases": cases}, fh, indent=2)
        fh.write("\n")
    print(f"wrote {len(cases)} golden cases")


if __name__ == "__main__":
    main()
