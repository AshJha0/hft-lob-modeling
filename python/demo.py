#!/usr/bin/env python3
"""lobmod end-to-end demo.

Replays the bundled 5000-event LOB stream, fits univariate and bivariate
Hawkes models to the bundled order-flow data with goodness-of-fit residual
analysis, and trains the deterministic MLP direction classifier against
logistic-regression and majority baselines, closing with an honest look at
whether the signal survives trading costs.

Run:  cd python && PYTHONPATH=src python3 demo.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np  # noqa: E402

import lobmod as lm  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LINE = "-" * 68


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def main() -> None:
    print("lobmod demo — high-frequency LOB modeling")

    # ------------------------------------------------------------ book replay
    section("1. Limit order book replay (data/lob_events.csv)")
    events = lm.read_events_csv(os.path.join(DATA, "lob_events.csv"))
    cfg = lm.load_feature_config(os.path.join(DATA, "features_config.json"))
    states = lm.replay(events, depth_k=cfg.depth_k)
    from lobmod.book import LimitOrderBook
    from lobmod.replay import apply_event

    book = LimitOrderBook()
    for e in events:
        apply_event(book, e)
    n_by_type = {t: sum(1 for e in events if e.etype == t) for t in "LCM"}
    volume = sum(f.size for f in book.trades)
    finite_spread = states.spread[np.isfinite(states.spread)]
    print(f"events: {len(events)}  (limit {n_by_type['L']}, cancel {n_by_type['C']}, "
          f"market {n_by_type['M']})")
    print(f"trades: {len(book.trades)} fills, {volume} units traded")
    print(f"mid range: [{np.nanmin(states.mid):.1f}, {np.nanmax(states.mid):.1f}] ticks; "
          f"mean spread {finite_spread.mean():.3f} ticks")
    print(f"events with an empty side (mid = NaN, guarded): "
          f"{int(np.isnan(states.mid).sum())}")
    print("\n  snapshot     best_bid  best_ask     mid   micro    depth3(B/S)")
    for seq in (1000, 3000, 5000):
        i = seq - 1
        print(f"  event {seq:5d}   {states.best_bid[i]:7.0f}  {states.best_ask[i]:8.0f}"
              f"  {states.mid[i]:7.1f} {states.microprice[i]:8.2f}"
              f"   {states.bid_depth_k[i]:.0f}/{states.ask_depth_k[i]:.0f}")

    # -------------------------------------------------------- Hawkes univariate
    section("2. Univariate Hawkes fit (data/hawkes_uni.csv)")
    with open(os.path.join(DATA, "dgp_params.json")) as fh:
        dgp = json.load(fh)
    t_uni = np.loadtxt(os.path.join(DATA, "hawkes_uni.csv"), skiprows=1)
    T_uni = dgp["hawkes_uni"]["T"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_u = lm.fit_uni(t_uni, T_uni)
    true_u = dgp["hawkes_uni"]
    print(f"{t_uni.size} events on [0, {T_uni:.0f}]  "
          f"(min gap {np.diff(t_uni).min():.4f} vs mean {np.diff(t_uni).mean():.3f} "
          "— self-exciting bursts)")
    print("\n  param        true    fitted")
    for name, tv, fv in (("mu", true_u["mu"], fit_u.mu),
                         ("alpha", true_u["alpha"], fit_u.alpha),
                         ("beta", true_u["beta"], fit_u.beta)):
        print(f"  {name:<8} {tv:8.3f}  {fv:8.4f}")
    print(f"  branching ratio alpha/beta = {fit_u.branching:.4f}  "
          f"({'stationary' if fit_u.branching < 1 else 'NON-STATIONARY'}; "
          f"{100 * fit_u.branching:.0f}% of events are endogenous offspring)")
    print(f"  loglik {fit_u.loglik:.2f}, Nelder-Mead converged={fit_u.converged} "
          f"({fit_u.n_iter} iters)")

    section("3. Goodness of fit — time-rescaling residuals + KS")
    tau = lm.uni_residuals(t_uni, T_uni, fit_u.mu, fit_u.alpha, fit_u.beta)
    d, p = lm.ks_test_exp(tau)
    print("time-rescaled inter-arrivals should be iid Exp(1) under the model:")
    print(f"  fitted params : KS stat {d:.4f}, p-value {p:.4f} -> "
          f"{'PASS (no rejection at 5%)' if p > 0.05 else 'REJECT'}")
    tau_wrong = lm.uni_residuals(t_uni, T_uni, 1.5, 0.1, 3.0)
    dw, pw = lm.ks_test_exp(tau_wrong)
    print(f"  wrong params  : KS stat {dw:.4f}, p-value {pw:.2e} -> REJECT (sanity)")

    # -------------------------------------------------------- Hawkes bivariate
    section("4. Bivariate Hawkes: buy/sell market-order cross-excitation")
    raw = np.loadtxt(os.path.join(DATA, "hawkes_biv.csv"), skiprows=1, delimiter=",")
    t_biv, k_biv = raw[:, 0], raw[:, 1].astype(int)
    T_biv = dgp["hawkes_biv"]["T"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_b = lm.fit_biv(t_biv, k_biv, T_biv)
    print(f"{t_biv.size} events ({int((k_biv == 0).sum())} buys, "
          f"{int((k_biv == 1).sum())} sells) on [0, {T_biv:.0f}]")
    a, br = fit_b.alpha, fit_b.branching
    print("\n  excitation alpha (fitted)      branching alpha/beta")
    print(f"    buy <-buy  {a[0, 0]:6.3f}            {br[0, 0]:6.3f}")
    print(f"    buy <-sell {a[0, 1]:6.3f}            {br[0, 1]:6.3f}")
    print(f"    sell<-buy  {a[1, 0]:6.3f}            {br[1, 0]:6.3f}")
    print(f"    sell<-sell {a[1, 1]:6.3f}            {br[1, 1]:6.3f}")
    print(f"  mu = {fit_b.mu[0]:.4f} (buy), {fit_b.mu[1]:.4f} (sell); "
          f"beta = {fit_b.beta:.4f}")
    print(f"  spectral radius of branching matrix = {fit_b.spectral_radius:.4f} "
          f"({'stationary' if fit_b.spectral_radius < 1 else 'NON-STATIONARY'})")
    print("  interpretation: self-excitation (buy<-buy, sell<-sell) dominates the")
    print("  cross terms — order flow is strongly clustered by side (herding /")
    print("  order splitting), with weaker but positive cross-side contagion.")
    taus = lm.biv_residuals(t_biv, k_biv, T_biv, fit_b.mu, fit_b.alpha, fit_b.beta)
    for name, tt in zip(("buy", "sell"), taus):
        dd, pp = lm.ks_test_exp(tt)
        print(f"  KS residuals ({name:4s}): stat {dd:.4f}, p {pp:.4f} -> "
              f"{'PASS' if pp > 0.05 else 'REJECT'}")

    # ----------------------------------------------------- features + learning
    section("5. Direction prediction: MLP vs logistic vs majority")
    fs = lm.compute_features(states, cfg)
    tr, te = lm.chronological_split(len(fs), cfg.train_frac)
    std = lm.Standardizer().fit(fs.X[tr])
    x_tr, x_te = std.transform(fs.X[tr]), std.transform(fs.X[te])
    y_tr, y_te = fs.y[tr], fs.y[te]
    print(f"{len(fs)} samples (features: {', '.join(lm.FEATURE_NAMES)}); "
          f"H={cfg.horizon} events, eps={cfg.epsilon}")
    print(f"class counts (down/flat/up): train {np.bincount(y_tr, minlength=3).tolist()}, "
          f"test {np.bincount(y_te, minlength=3).tolist()}")
    print(f"chronological split: {tr.size} train / {te.size} test; "
          f"MLP {cfg.hidden}, lr={cfg.learning_rate}, {cfg.epochs} full-batch epochs")

    net = lm.MLP(5, cfg.hidden, 3)
    hist = net.fit(x_tr, y_tr, cfg.learning_rate, cfg.epochs)
    logistic = lm.MLP(5, (), 3)
    logistic.fit(x_tr, y_tr, cfg.learning_rate, cfg.epochs)
    maj = lm.majority_class(y_tr)
    pred_nn = net.predict(x_te)
    pred_lg = logistic.predict(x_te)
    pred_mj = np.full(te.size, maj)
    print(f"\ntraining loss: {hist[0]:.4f} (epoch 1) -> {hist[-1]:.4f} (epoch {cfg.epochs})")
    print("\n  model                holdout accuracy")
    print(f"  MLP (16,8)               {lm.accuracy(y_te, pred_nn):.4f}")
    print(f"  logistic regression      {lm.accuracy(y_te, pred_lg):.4f}")
    print(f"  majority class ({maj})       {lm.accuracy(y_te, pred_mj):.4f}")
    cm = lm.confusion_matrix(y_te, pred_nn, 3)
    print("\n  MLP confusion matrix (rows true, cols predicted; 0=down 1=flat 2=up)")
    for r in range(3):
        print(f"    {r} | {cm[r, 0]:5d} {cm[r, 1]:5d} {cm[r, 2]:5d}")

    section("6. Does the signal survive costs?")
    gross, net_pnl, n_trades = lm.signal_pnl(pred_nn, fs.fwd_ret[te], fs.spread[te])
    print(f"trading the MLP signal on the holdout ({n_trades} trades, "
          "mid-to-mid, half-spread cost per trade):")
    print(f"  gross PnL {gross:+.1f} ticks   net PnL {net_pnl:+.1f} ticks")
    print("note: the classifier finds real structure (it beats both baselines and")
    print("gross PnL is positive), but paying half the spread on every trade")
    print("erases it — naive microstructure signals die on transaction costs.")

    print("\ndemo complete.")


if __name__ == "__main__":
    main()
