"""Run the model extensions and write research/results_models.md.

Run from anywhere:  python research/run_models.py [--workers 8] [--quick]

Everything is seeded, so results do not depend on the worker count.  --quick is
a smoke run whose numbers mean little.  Synthetic data only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # small matrices; workers supply the parallelism
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "src"))

from lobmod.features import load_feature_config  # noqa: E402
from lobmod.gof import ks_test_exp  # noqa: E402
from lobmod.replay import replay  # noqa: E402
from lobmod.research import BacktestConfig, ljung_box  # noqa: E402
from lobmod.research.baselines import (  # noqa: E402
    COMPARED, paired_log_loss_ci, seed_mean_ci, walk_forward_compare,
)
from lobmod.research.kernels import (  # noqa: E402
    POWER_LAW_RATES, fit_fixed_mixture, fit_power_law, fit_residuals, fit_sum_exp, simulate_cluster,
)
from lobmod.research.labels import cost_aware_labels, horizon_sweep  # noqa: E402
from lobmod.research.seasonality import (  # noqa: E402
    LinearSplineBaseline, PiecewiseConstantBaseline, fit_seasonal, select_baseline,
)
from lobmod.zi import generate_events  # noqa: E402

CFG = load_feature_config(str(ROOT / "data" / "features_config.json"))
EMBARGO = max(CFG.ofi_window, CFG.ret_window)
FEE = 0.1
DELAY = 1
SWEEP_H = (5, 10, 25, 50, 100, 200, 400)
MODEL_H = (10, 50, 200)
GRID_RATES = tuple(np.geomspace(0.01, 300.0, 10))
KERNEL_CASES = {"two-timescale": 0.6, "power-law": 0.6, "single exp (control)": 0.5}
SEASON_CASES = {"U-shaped mu": 0.5, "flat mu (control)": 0.5, "U-shaped mu, two-timescale kernel": 0.6}


def offsets(case, rng):
    if "two-timescale" in case:
        return lambda k: np.where(rng.uniform(size=k) < 0.5, rng.exponential(0.02, k), rng.exponential(5.0, k))
    if case == "power-law":
        return lambda k: 0.05 * ((1.0 - rng.uniform(size=k)) ** (-1.0 / 0.6) - 1.0)
    return lambda k: rng.exponential(0.6, k)


def gof(x, T, fit, base=None):
    tau = fit_residuals(x, T, fit, base)
    return ks_test_exp(tau)[1] < 0.05, ljung_box(tau, 10).p_value < 0.05


def record(x, T, fit, base=None):
    ks, lb = gof(x, T, fit, base)
    return {"n": fit.branching, "aic": fit.aic, "bic": fit.bic, "ks": ks, "lb": lb, "ok": fit.converged}


def kernel_job(args):
    case, T, s = args
    rng = np.random.default_rng([11, list(KERNEL_CASES).index(case), int(T), s])
    x = simulate_cluster(lambda t: 0.0 * t + 0.5, 0.5, KERNEL_CASES[case], offsets(case, rng), T, rng)
    fits = [fit_sum_exp(x, T, 1), fit_sum_exp(x, T, 2), fit_sum_exp(x, T, 3),
            fit_fixed_mixture(x, T, GRID_RATES, name="10 fixed rates"), fit_power_law(x, T)]
    return case, T, {f.name: record(x, T, f) for f in fits}


def season_job(args):
    case, T, s = args
    rng = np.random.default_rng([12, list(SEASON_CASES).index(case), s])
    shaped = "U-shaped" in case
    mu = (lambda t: 1.0 + 0.9 * np.cos(2.0 * np.pi * t / T)) if shaped else (lambda t: 0.0 * t + 1.0)
    x = simulate_cluster(mu, 1.9 if shaped else 1.0, SEASON_CASES[case], offsets(case, rng), T, rng)
    K = 2 if "two-timescale" in case else 1
    out = {}
    out[f"flat mu, exp K={K}"] = record(x, T, fit_sum_exp(x, T, K))
    pc, sp = PiecewiseConstantBaseline(T, 8), LinearSplineBaseline(T, 8)
    out["piecewise J=8"] = record(x, T, fit_seasonal(x, T, pc, K), pc)
    out["spline J=8"] = record(x, T, fit_seasonal(x, T, sp, K), sp)
    best, base, _ = select_baseline(x, T, "spline", K=K)
    out["spline, J by BIC"] = record(x, T, best, base)
    out["spline, J by BIC"]["flat_chosen"] = base.n_coef == 1
    if K == 2:
        out["flat mu, exp K=1"] = record(x, T, fit_sum_exp(x, T, 1))
    return case, out


def sweep_job(args):
    seed, n_events = args
    states = replay(generate_events(n_events, seed), CFG.depth_k)
    return {fee: horizon_sweep(states, SWEEP_H, fee, DELAY if fee else 0) for fee in (0.0, FEE)}


def compare_job(args):
    """Pinned labels at the pinned horizon: the plain classifier comparison."""
    seed, n_events, folds = args
    states = replay(generate_events(n_events, seed), CFG.depth_k)
    r = walk_forward_compare(states, CFG, folds, EMBARGO, seed=seed,
                             backtest=BacktestConfig(hold=CFG.horizon, delay=DELAY, fee_per_side=FEE))
    block = 5 * CFG.horizon
    d = {m: paired_log_loss_ci(r.y, r.proba["gbt"], r.proba[m], block, 300, seed) for m in ("logistic", "mlp")}
    return seed, r.metrics, {m: (c.point, c.low, c.high) for m, c in d.items()}


def cost_job(args):
    seed, n_events, folds, h = args
    states = replay(generate_events(n_events, seed), CFG.depth_k)
    cfg = replace(CFG, horizon=h)

    def relabel(fs):
        lab = cost_aware_labels(states, fs.event_index, h, FEE, DELAY)
        return lab.y, lab.valid

    r = walk_forward_compare(states, cfg, folds, EMBARGO, relabel, seed=seed, extra_purge=DELAY,
                             backtest=BacktestConfig(hold=h, delay=DELAY, fee_per_side=FEE))
    return h, r.metrics, float(np.mean(r.y != 1))


def ci_text(values):
    c = seed_mean_ci(values)
    return f"{c.point:.3f} [{c.low:.3f}, {c.high:.3f}]"


def hawkes_table(rows, truth, ref):
    names = list(rows[0])
    L = ["| Model | mean n (95% CI of mean) | sd | bias | mean AIC - ref | BIC beats ref | KS reject | LB reject | converged |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in names:
        n = np.array([r[m]["n"] for r in rows])
        da = np.mean([r[m]["aic"] - r[ref]["aic"] for r in rows])
        wins = np.mean([r[m]["bic"] < r[ref]["bic"] for r in rows])
        L.append(f"| {m} | {ci_text(n)} | {n.std(ddof=1):.3f} | {n.mean() - truth:+.3f} | {da:.1f} | {wins:.0%} "
                 f"| {np.mean([r[m]['ks'] for r in rows]):.0%} | {np.mean([r[m]['lb'] for r in rows]):.0%} "
                 f"| {np.mean([r[m]['ok'] for r in rows]):.0%} |")
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=str(ROOT / "research" / "results_models.md"))
    a = p.parse_args()
    reps, long_reps, seeds, cost_seeds, n_events, folds = (60, 20, 10, 5, 6000, 3)
    if a.quick:
        reps, long_reps, seeds, cost_seeds, n_events, folds = (4, 2, 2, 2, 4000, 2)
    warnings.simplefilter("ignore", RuntimeWarning)
    start = time.time()
    zi = [s * 7919 for s in range(1, seeds + 1)]
    with ProcessPoolExecutor(a.workers) as pool:
        kjobs = [(c, 1000.0, s) for c in KERNEL_CASES for s in range(reps)]
        kjobs += [(c, 5000.0, s) for c in list(KERNEL_CASES)[:2] for s in range(long_reps)]
        kres = list(pool.map(kernel_job, kjobs))
        sres = list(pool.map(season_job, [(c, 1000.0, s) for c in SEASON_CASES for s in range(reps)]))
        sweeps = list(pool.map(sweep_job, [(s, n_events) for s in zi]))
        comp = list(pool.map(compare_job, [(s, n_events, folds) for s in zi]))
        cost = list(pool.map(cost_job, [(s, n_events, folds, h) for h in MODEL_H for s in zi[:cost_seeds]]))

    L = ["# Model extension results", "",
         "Generated by `python research/run_models.py`; do not edit by hand. Prices are ticks.", "",
         "**Limits.** Synthetic only: simulated Hawkes paths with known kernels, and the "
         "zero-intelligence stream, which has no informed traders and does not react to orders. "
         "Nothing here says anything about real markets, and no number below is evidence of a "
         "tradable edge. Every Hawkes fit knows the right family is among its candidates; "
         "real kernels need not be. Rates and branching are not constrained to stationarity.", "",
         "## 1. Kernels", "",
         "Reference for AIC and BIC columns: `exp K=1`. n is the branching ratio. KS and LB "
         "(Ljung-Box, 10 lags) are 5% rejection rates on time-rescaling residuals; the KS "
         "p-value ignores parameter estimation, so it is conservative. The 95% interval is "
         "for the mean over paths, not for one path. `10 fixed rates` counts only non-zero "
         "weights as parameters, so its AIC is indicative.", ""]
    for T in (1000.0, 5000.0):
        for case, truth in KERNEL_CASES.items():
            rows = [r for c, t, r in kres if c == case and t == T]
            if rows:
                L += [f"### {case}, true n = {truth}, T = {T:.0f}, {len(rows)} paths", ""]
                L += hawkes_table(rows, truth, "exp K=1") + [""]
    L += [f"Power-law mixture: {len(POWER_LAW_RATES)} fixed rates from 1e-3 to 1e3, three "
          "kernel parameters (n, a, c).", "", "## 2. Seasonality", "",
          "Baseline and kernel fitted jointly. The reference is the flat-baseline fit. "
          "T = 1000; the flat control has mu = 1.", ""]
    for case, truth in SEASON_CASES.items():
        rows = [r for c, r in sres if c == case]
        ref = "flat mu, exp K=2" if "two-timescale" in case else "flat mu, exp K=1"
        L += [f"### {case}, true n = {truth}, {len(rows)} paths", ""]
        L += hawkes_table(rows, truth, ref) + [""]
        share = np.mean([r["spline, J by BIC"]["flat_chosen"] for r in rows])
        L += [f"BIC kept the flat baseline on {share:.0%} of paths.", ""]

    L += ["## 3. Cost-aware labels", "",
          f"{seeds} ZI streams of {n_events} events. `up`, `down`: share of events where crossing "
          "both ways pays. `best`: mean of the better round trip, a ceiling needing perfect "
          "foresight; negative means even that loses on the average event.", ""]
    for fee, title in ((0.0, "no fee, no delay"), (FEE, f"fee {FEE} per side, delay {DELAY}")):
        L += [f"### {title}", "", "| H | up | down | best | mean when paid |", "|---|---|---|---|---|"]
        for j, h in enumerate(SWEEP_H):
            v = {k: np.mean([s[fee][j][k] for s in sweeps]) for k in ("share_up", "share_down", "mean_best", "mean_paid")}
            L.append(f"| {h} | {v['share_up']:.3f} | {v['share_down']:.3f} | {v['mean_best']:.3f} | {v['mean_paid']:.3f} |")
        L.append("")
    L += [f"Models trained on the cost-aware label (fee {FEE}, delay {DELAY}), {cost_seeds} seeds, "
          f"{folds} purged folds, embargo {EMBARGO}. `net` is the event-driven backtest in ticks on "
          "pooled test rows, one position at a time. Mean over seeds, iid bootstrap 95% CI. "
          "`majority` predicts flat when flat is the training mode, so 0 trades is the bar to beat.", "",
          "| H | payable share | model | MCC | trade share | net | trades |", "|---|---|---|---|---|---|---|"]
    for h in MODEL_H:
        rows = [(m, s) for hh, m, s in cost if hh == h]
        pay = np.mean([s for _, s in rows])
        for name in COMPARED:
            v = [m[name] for m, _ in rows]
            L.append(f"| {h} | {pay:.3f} | {name} | {ci_text([x['mcc'] for x in v])} "
                     f"| {np.mean([x['trade_share'] for x in v]):.3f} | {ci_text([x['net'] for x in v])} "
                     f"| {np.mean([x['n_trades'] for x in v]):.0f} |")
    L += ["", "## 4. Boosted trees against logistic and MLP", "",
          f"Pinned labels, H = {CFG.horizon}, {seeds} ZI streams of {n_events} events, {folds} purged "
          f"walk-forward folds, embargo {EMBARGO}. Boosting used fixed defaults with no tuning. "
          "Mean over seeds with an iid bootstrap 95% CI.", "",
          f"`net`: backtest with hold {CFG.horizon}, delay {DELAY}, fee {FEE}.", "",
          "| Model | accuracy | log-loss | MCC | macro-F1 | net | trades |", "|---|---|---|---|---|---|---|"]
    for name in COMPARED:
        v = [m[name] for _, m, _ in comp]
        L.append(f"| {name} | " + " | ".join(ci_text([x[k] for x in v]) for k in ("accuracy", "log_loss", "mcc", "macro_f1", "net"))
                 + f" | {np.mean([x['n_trades'] for x in v]):.0f} |")
    L += ["", "Paired log-loss difference, boosted trees minus the other model (negative favours "
          f"boosting). Seed row: mean over seeds. Streams: how many of {seeds} per-stream block-bootstrap "
          f"intervals (block {5 * CFG.horizon}) lie wholly below or above zero.", "",
          "| Against | mean difference (95% CI) | below 0 | above 0 |", "|---|---|---|---|"]
    for m in ("logistic", "mlp"):
        d = [x[2][m] for x in comp]
        L.append(f"| {m} | {ci_text([v[0] for v in d])} | {sum(v[2] < 0 for v in d)} | {sum(v[1] > 0 for v in d)} |")
    L += ["", "## Seeds", "",
          "- Kernel paths `default_rng([11, case, T, s])`; seasonal paths `default_rng([12, case, s])`.",
          f"- ZI streams {zi[0]} to {zi[-1]} in steps of 7919.", ""]
    Path(a.out).write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"wrote {a.out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
