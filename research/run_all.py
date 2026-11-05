"""Reproduce the key review numbers and write research/results.md.

Run from anywhere:  python research/run_all.py [--workers 4] [--quick]

Everything is seeded, so results do not depend on the worker count.  Default
size takes a few minutes; --quick is a smoke run whose numbers mean little.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from lobmod.features import compute_features, load_feature_config  # noqa: E402
from lobmod.gof import ks_test_exp  # noqa: E402
from lobmod.hawkes import (  # noqa: E402
    fit_biv, fit_uni, read_marked_times_csv, read_times_csv, simulate_uni, uni_residuals,
)
from lobmod.nn import MLP, Standardizer, accuracy, majority_class, signal_pnl  # noqa: E402
from lobmod.replay import read_events_csv, replay  # noqa: E402
from lobmod.research import (  # noqa: E402
    BacktestConfig, MakerConfig, block_bootstrap_ci, bootstrap_ks_uni, evaluate_stream,
    foresight_ceiling, inference_biv, inference_uni, ljung_box, lr_test_no_cross,
    lr_test_poisson, run_backtest, run_maker, walk_forward_splits,
)
from lobmod.zi import generate_events  # noqa: E402

DATA = ROOT / "data"
CFG = load_feature_config(str(DATA / "features_config.json"))
EMBARGO = max(CFG.ofi_window, CFG.ret_window)
T_UNI, T_BIV = 1000.0, 600.0
TRUE_UNI = (0.5, 1.0, 1.6)
T_SIZE = 200.0
T_POWER = 300.0
BOOT_SEED = 20261105
BT = (BacktestConfig(hold=CFG.horizon), BacktestConfig(hold=CFG.horizon, delay=1),
      BacktestConfig(hold=CFG.horizon, delay=1, fee_per_side=0.1))
MAKER_CASES = ((0, 5, 0.0), (5, 5, 0.0), (20, 5, 0.0), (0, 5, 0.1), (0, 5, 0.3), (0, 20, 0.0))


def size_job(args):
    s, b = args
    x = simulate_uni(*TRUE_UNI, T_SIZE, 5000 + s)
    g = bootstrap_ks_uni(x, T_SIZE, b, seed=100000 + 1000 * s)
    return g.p_value, g.p_naive


def power_job(args):
    """Exponential-kernel Hawkes with a U-shaped baseline: a misspecified fit."""
    s, b = args
    rng = np.random.default_rng(7000 + s)
    m = rng.poisson(1.9 * T_POWER)
    im = rng.uniform(0.0, T_POWER, m)
    im = im[rng.uniform(0.0, 1.9, m) < 1.0 + 0.9 * np.cos(2.0 * np.pi * im / T_POWER)]
    out, gen = [im], im
    while gen.size:
        par = np.repeat(gen, rng.poisson(0.5, gen.size))
        gen = par + rng.exponential(0.6, par.size)
        gen = gen[gen < T_POWER]
        out.append(gen)
    x = np.sort(np.concatenate(out))
    x = x[np.r_[True, np.diff(x) > 0]]
    f = fit_uni(x, T_POWER)
    try:
        g = bootstrap_ks_uni(x, T_POWER, b, seed=200000 + 1000 * s, fit=f)
    except ValueError:      # fit went non-stationary: the model is plainly wrong
        return 0.0, 0.0, 0.0
    tau = uni_residuals(x, T_POWER, f.mu, f.alpha, f.beta)
    return g.p_value, g.p_naive, ljung_box(tau, 10).p_value


def stream_job(seed):
    try:
        return seed, evaluate_stream(generate_events(5000, seed), CFG, EMBARGO, BT)
    except ValueError:
        return seed, None


def maker_job(args):
    seed, case = args
    delay, cap, fee = case
    r = run_maker(generate_events(20000, seed), MakerConfig(delay, cap, fee))
    return (case, r.pnl, r.pnl_unwound, len(r.fills), r.quoted_edge, r.realised_spread[1],
            r.realised_spread[50], r.realised_spread[200], r.n_skipped)


def ceiling_job(args):
    seed, horizons = args
    st = replay(generate_events(20000, seed), CFG.depth_k)
    return [foresight_ceiling(st, h) for h in horizons]


def mean_ci(values, seed=BOOT_SEED):
    """Mean over independent seeds with an iid percentile bootstrap interval."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return block_bootstrap_ci(lambda i: v[i].mean(), v.size, 1, 4000, seed=seed)


def fmt(ci, d=3):
    return f"{ci.point:.{d}f} [{ci.low:.{d}f}, {ci.high:.{d}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def pval(p):
    return "nan" if math.isnan(p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")


def hawkes_section(pool, a):
    t = read_times_csv(str(DATA / "hawkes_uni.csv"))
    f = fit_uni(t, T_UNI)
    inf = inference_uni(t, T_UNI, f)
    lr = lr_test_poisson(t, T_UNI, n_boot=a.lr_boot, seed=BOOT_SEED)
    gof = bootstrap_ks_uni(t, T_UNI, a.gof_boot, seed=BOOT_SEED)
    lb = ljung_box(uni_residuals(t, T_UNI, f.mu, f.alpha, f.beta), 10)
    tb, kb = read_marked_times_csv(str(DATA / "hawkes_biv.csv"))
    fb = fit_biv(tb, kb, T_BIV)
    ib = inference_biv(tb, kb, T_BIV, fb)
    lc = lr_test_no_cross(tb, kb, T_BIV, fb)
    size = np.array(list(pool.map(size_job, [(s, a.size_boot) for s in range(a.size_reps)])))
    power = np.array(list(pool.map(power_job, [(s, a.power_boot) for s in range(a.power_reps)])))

    L = ["## 1. Hawkes inference", "",
         f"Bundled `hawkes_uni.csv`, {t.size} events, T = {T_UNI:.0f}. "
         f"Truth: mu, alpha, beta = {TRUE_UNI}.", ""]
    L += table(["parameter", "estimate", "se", "95% Wald"], [
        (n, f"{e:.3f}", f"{s:.3f}", f"[{lo:.3f}, {hi:.3f}]")
        for n, e, s, lo, hi in zip(inf.names, inf.estimate, inf.se, inf.ci_low, inf.ci_high)
    ] + [("branching", f"{inf.branching:.3f}", f"{inf.branching_se:.3f}", "delta method")])
    L += table(["test", "statistic", "p"], [
        ("LR vs Poisson, chi-bar mixture (indicative)", f"{lr.statistic:.1f}", pval(lr.p_mixture)),
        (f"LR vs Poisson, bootstrap B={a.lr_boot}", "", pval(lr.p_bootstrap)),
        ("KS, table p (ignores estimation)", f"{gof.statistic:.4f}", pval(gof.p_naive)),
        (f"KS, parametric bootstrap B={gof.n_boot}", f"{gof.statistic:.4f}", pval(gof.p_value)),
        ("Ljung-Box on residuals, 10 lags", f"{lb.statistic:.2f}", pval(lb.p_value)),
    ])
    L += [f"Bundled `hawkes_biv.csv`, {tb.size} events, T = {T_BIV:.0f}.", ""]
    L += table(["parameter", "estimate", "se"], [
        (n, f"{e:.3f}", f"{s:.3f}") for n, e, s in zip(ib.names, ib.estimate, ib.se)])
    L += [f"Cross t-stats: a01 {ib.t_stat('a01'):.2f}, a10 {ib.t_stat('a10'):.2f}. "
          f"LR no-cross {lc.statistic:.2f}: p = {pval(lc.p_mixture)} (boundary mixture), "
          f"{pval(lc.p_chi2)} (chi2_2, conservative).", ""]
    ps, pn = size[:, 0], size[:, 1]
    L += [f"KS test size under the true model: {a.size_reps} paths, T = {T_SIZE:.0f}, "
          f"B = {a.size_boot}.", ""]
    L += table(["p-value", "mean (0.5 if valid)", "reject at 5%", "reject at 10%"], [
        ("table", f"{pn.mean():.3f}", f"{np.mean(pn < .05):.3f}", f"{np.mean(pn < .10):.3f}"),
        ("bootstrap", f"{ps.mean():.3f}", f"{np.mean(ps <= .05):.3f}", f"{np.mean(ps <= .10):.3f}"),
    ])
    L += [f"Power against a U-shaped baseline fitted with constant mu: {a.power_reps} paths, "
          f"T = {T_POWER:.0f}, B = {a.power_boot}. Rejection at 5%: table KS "
          f"{np.mean(power[:, 1] < .05):.2f}, bootstrap KS {np.mean(power[:, 0] <= .05):.2f}, "
          f"Ljung-Box {np.mean(power[:, 2] < .05):.2f}. With so few paths these are rough "
          "(se up to about 0.14).", ""]
    return L


def bundled_section():
    st = replay(read_events_csv(str(DATA / "lob_events.csv")), CFG.depth_k)
    fs = compute_features(st, CFG)
    sp = walk_forward_splits(fs.event_index, 1, CFG.horizon, EMBARGO, CFG.train_frac)[0]
    sd = Standardizer().fit(fs.X[sp.train])
    xa, xb = sd.transform(fs.X[sp.train]), sd.transform(fs.X[sp.test])
    ya, yb = fs.y[sp.train], fs.y[sp.test]
    pred = {}
    for name, hidden in (("mlp", CFG.hidden), ("logistic", ())):
        m = MLP(xa.shape[1], hidden, 3)
        m.fit(xa, ya, CFG.learning_rate, CFG.epochs)
        pred[name] = m.predict(xb)
    maj = majority_class(ya)
    pm, pl = pred["mlp"], pred["logistic"]
    pos = pm - 1
    fwd = fs.fwd_ret[sp.test]
    n = yb.size
    block = 50
    d_maj = block_bootstrap_ci(lambda i: np.mean(pm[i] == yb[i]) - np.mean(yb[i] == maj),
                               n, block, 4000, seed=BOOT_SEED)
    d_log = block_bootstrap_ci(lambda i: np.mean(pm[i] == yb[i]) - np.mean(pl[i] == yb[i]),
                               n, block, 4000, seed=BOOT_SEED)
    gross = block_bootstrap_ci(lambda i: float(np.sum(pos[i] * fwd[i])), n, block, 4000,
                               seed=BOOT_SEED)
    g, net, k = signal_pnl(pm, fwd, fs.spread[sp.test])
    L = ["## 2. Bundled stream, one purged split", "",
         f"`lob_events.csv`; train rows {sp.train.size} after purge (H = {CFG.horizon}, "
         f"embargo = {EMBARGO}), test rows {n}, block length {block}, 4000 resamples.", ""]
    L += table(["quantity", "point [95% CI]"], [
        ("MLP accuracy", f"{accuracy(yb, pm):.4f}"),
        ("MLP minus majority, accuracy", fmt(d_maj, 4)),
        ("MLP minus logistic, accuracy", fmt(d_log, 4)),
        ("signal_pnl gross (overlapping diagnostic, ticks)", fmt(gross, 1)),
        ("signal_pnl net, half spread on entry", f"{net:.1f} ({k} signals)"),
    ])
    rows = []
    for delay in (0, 1, 5):
        cfg = BacktestConfig(hold=CFG.horizon, delay=delay)
        s = run_backtest(st, fs.event_index[sp.test], pos, cfg).stats
        rows.append((delay, s["n_trades"], f"{s['mid_pnl']:.1f}", f"{s['spread_cost']:.1f}",
                     f"{s['net']:.1f}", f"{s['mean_net']:.2f}", f"{s['hit_rate']:.2f}"))
    L += ["Executable taker backtest of the MLP signal, one lot, no fee:", ""]
    L += table(["delay", "round trips", "mid PnL", "spread paid", "net", "net per trip",
                "hit rate"], rows)
    return L


def seeds_section(pool, a):
    seeds = [s * 7919 for s in range(1, a.seeds + 1)]
    done = dict(pool.map(stream_job, seeds))
    res = {s: r for s, r in done.items() if r is not None}
    skipped = [s for s, r in done.items() if r is None]

    def col(model, key):
        return [res[s][model][key] for s in res]

    L = [f"## 3. Multi-seed comparison, {len(res)} ZI streams of 5000 events", "",
         f"Purged split, embargo {EMBARGO}. Mean over seeds with an iid bootstrap 95% CI. "
         f"Skipped seeds: {skipped or 'none'}.", ""]
    keys = ("accuracy", "log_loss", "mcc", "macro_f1", "precision_up", "precision_down")
    L += table(["metric", "mlp", "logistic", "majority"], [
        (k, *(fmt(mean_ci(col(m, k))) if np.isfinite(col(m, k)).any() else "undefined"
              for m in ("mlp", "logistic", "majority"))) for k in keys])
    diff = np.array(col("mlp", "accuracy")) - np.array(col("logistic", "accuracy"))
    dmaj = np.array(col("mlp", "accuracy")) - np.array(col("majority", "accuracy"))
    L += [f"Paired accuracy difference: MLP minus logistic {fmt(mean_ci(diff), 4)}, "
          f"MLP above logistic on {np.mean(diff > 0):.0%} of seeds; MLP minus majority "
          f"{fmt(mean_ci(dmaj), 4)}, above on {np.mean(dmaj > 0):.0%}.", ""]
    rows = [("signal_pnl gross", *(fmt(mean_ci(col(m, "gross")), 1) for m in ("mlp", "logistic"))),
            ("signal_pnl net", *(fmt(mean_ci(col(m, "net")), 1) for m in ("mlp", "logistic")))]
    for j, label in enumerate(("backtest net, delay 0", "backtest net, delay 1",
                               "backtest net, delay 1, fee 0.1")):
        rows.append((label, *(fmt(mean_ci(col(m, f"bt{j}_net")), 1) for m in ("mlp", "logistic"))))
    rows.append(("round trips, delay 0", *(f"{np.mean(col(m, 'bt0_trades')):.1f}"
                                           for m in ("mlp", "logistic"))))
    L += ["PnL per stream in ticks:", ""] + table(["measure", "mlp", "logistic"], rows)
    share = np.mean(np.array(col("mlp", "bt0_net")) > 0.0)
    L += [f"Share of seeds with positive MLP backtest net at delay 0: {share:.0%}.", ""]
    return L, seeds


def maker_section(pool, a):
    seeds = [s * 7919 for s in range(1, a.maker_seeds + 1)]
    horizons = (10, 50, 200, 1000)
    ceil = list(pool.map(ceiling_job, [(s, horizons) for s in seeds[:4]]))
    L = ["## 4. Tradability", "",
         f"Taker with perfect foresight, crossing both ways, {min(4, len(seeds))} streams of "
         "20000 events:", ""]
    L += table(["hold H", "share profitable", "mean best-side PnL"], [
        (h, f"{np.mean([c[i]['share_profitable'] for c in ceil]):.3f}",
         f"{np.mean([c[i]['mean_best'] for c in ceil]):.2f}") for i, h in enumerate(horizons)])
    jobs = [(s, c) for c in MAKER_CASES for s in seeds]
    out = list(pool.map(maker_job, jobs))
    rows = []
    for c in MAKER_CASES:
        r = [o[1:] for o in out if o[0] == c]
        v = np.array(r, dtype=float)
        rows.append((c[0], c[1], c[2], fmt(mean_ci(v[:, 0]), 1), f"{v[:, 1].mean():.1f}",
                     f"{np.mean(v[:, 0] > 0):.2f}", f"{v[:, 2].mean():.0f}",
                     f"{v[:, 3].mean():.3f}", f"{v[:, 4].mean():.3f}", f"{v[:, 5].mean():.3f}",
                     f"{v[:, 6].mean():.3f}", f"{v[:, 7].mean():.0f}"))
    L += [f"Passive maker, {len(seeds)} streams of 20000 events, PnL in ticks per stream, "
          "marked at mid; `unwound` closes inventory at the touch:", ""]
    L += table(["delay", "cap", "fee", "PnL [95% CI]", "unwound", "share > 0", "fills",
                "quoted edge", "realised @1", "realised @50", "realised @200", "skipped"], rows)
    L += ["The maker PnL flatters: the flow does not react to our quotes, there are no "
          "competitors, and no informed traders exist in ZI flow.", ""]
    return L, seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "research" / "results.md"))
    a = ap.parse_args()
    scale = dict(seeds=12, maker_seeds=6, lr_boot=49, gof_boot=49, size_reps=30, size_boot=29,
                 power_reps=10, power_boot=19)
    if a.quick:
        scale = dict(seeds=2, maker_seeds=2, lr_boot=5, gof_boot=9, size_reps=4, size_boot=5,
                     power_reps=2, power_boot=5)
    for k, v in scale.items():
        setattr(a, k, v)
    warnings.simplefilter("ignore", RuntimeWarning)
    start = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        L = ["# Research results", "",
             "Generated by `python research/run_all.py`; do not edit by hand. "
             "Numbers are ticks unless stated.", "",
             "**Limits.** Synthetic zero-intelligence flow and simulated Hawkes paths only. "
             "This says nothing about real markets. ZI flow has no informed traders and does "
             "not react to orders, so neither the classifier results nor the PnL figures "
             "transfer to any venue.", ""]
        L += hawkes_section(pool, a)
        L += bundled_section()
        part, stream_seeds = seeds_section(pool, a)
        L += part
        part, maker_seeds = maker_section(pool, a)
        L += part
    L += ["## Seeds", "",
          f"- Bootstrap seed {BOOT_SEED}; Hawkes size paths 5000+s, replicates 100000+1000s+b; "
          "power paths 7000+s, replicates 200000+1000s+b.",
          f"- ZI streams: {stream_seeds[0]} to {stream_seeds[-1]} in steps of 7919.",
          f"- Maker streams: {maker_seeds[0]} to {maker_seeds[-1]} in steps of 7919.", ""]
    Path(a.out).write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"wrote {a.out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
